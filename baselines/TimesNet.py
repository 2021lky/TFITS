import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.fft
from utils import masked_mae_cal


class Inception_Block_V1(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super(Inception_Block_V1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels):
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i))
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res


class PositionalEncoding(nn.Module):
    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()
        # 不是模型参数，而是注册为缓冲区
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        """ Sinusoid position encoding table """
        # 生成位置编码表的函数
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        # 奇数列使用正弦函数编码
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        # 偶数列使用余弦函数编码
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        # 将位置编码表加到输入张量上
        return x + self.pos_table[:, :x.size(1)].clone().detach()


def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class TimesBlock(nn.Module):
    def __init__(self, configs):
        super(TimesBlock, self).__init__()
        self.seq_len = configs.seq_len
        self.k = configs.top_k
        # parameter-efficient design
        self.conv = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_inner,
                               num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_inner, configs.d_model,
                               num_kernels=configs.num_kernels)
        )

    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            # padding
            if self.seq_len % period != 0:
                length = ((self.seq_len // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - self.seq_len), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = self.seq_len
                out = x
            # reshape
            out = out.reshape(B, length // period, period,
                              N).permute(0, 3, 1, 2).contiguous()
            # 2D conv: from 1d Variation to 2d Variation
            out = self.conv(out)
            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :self.seq_len, :])
        res = torch.stack(res, dim=-1)
        # adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(
            1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        # residual connection
        res = res + x
        return res


class Model(nn.Module):
    """
    Paper link: https://openreview.net/pdf?id=ju_Uqw384Oq
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.seq_len = configs.seq_len

        self.model = nn.ModuleList([TimesBlock(configs)
                                    for _ in range(configs.n_layer)])

        self.enc_embedding = nn.Linear(configs.feature_num, configs.d_model)
        self.position_enc = PositionalEncoding(configs.d_model, n_position=configs.seq_len)
        self.layer_norm = nn.LayerNorm(configs.d_model)

        self.dropout = nn.Dropout(p=configs.dropout)
        self.projection = nn.Linear(configs.d_model, configs.feature_num, bias=True)



    def impute(self, inputs):
        X, masks = inputs['X'], inputs['missing_mask']
        L = X.shape[1]
        means = X.mean(1, keepdim=True).detach()
        x_enc = X - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # embedding
        enc_out = self.enc_embedding(x_enc)  # [B,T,C]
        enc_out = self.dropout(self.position_enc(enc_out))  # 位置编码
        # TimesNet
        for i in range(len(self.model)):
            enc_out = self.layer_norm(self.model[i](enc_out))
        # porject back
        dec_out = self.projection(enc_out)
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, L, 1))

        X_c = masks * X + (1 - masks) * dec_out
        return X_c, [dec_out]



    def forward(self, inputs, stage):
        X, masks = inputs['X'], inputs['missing_mask']

        imputed_data, [X_tilde_1] = self.impute(inputs)

        # 计算重构损失
        final_reconstruction_MAE = masked_mae_cal(X_tilde_1, X, masks)
        # 综合考虑多个重构损失
        reconstruction_loss = final_reconstruction_MAE

        # 计算插补损失
        if stage != 'test':
            # 在验证阶段需要计算imputation loss
            imputation_MAE = masked_mae_cal(X_tilde_1, inputs['X_holdout'], inputs['indicating_mask'])
        else:
            # 在测试阶段不需要计算imputation loss
            imputation_MAE = torch.tensor(0.0)

        # 返回结果字典
        return {'imputed_data': imputed_data,  #
                'reconstruction_loss': reconstruction_loss,  # 重构累积损失
                'imputation_loss': imputation_MAE,  # 插补损失
                'reconstruction_MAE': final_reconstruction_MAE,
                'imputation_MAE': imputation_MAE}
