import numpy as np
from modeling.layers import *
from modeling.UNetFuse import UNetFuse
from utils import masked_mae_cal

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


# 前向融合模块
class FrontFeature(nn.Module):
    def __init__(self, d_time, d_feature, triangular='lower'):
        super(FrontFeature, self).__init__()
        assert triangular == 'lower' or triangular == 'upper', 'triangular must be lower or upper'
        self.build(d_time, d_feature, triangular)

    def build(self, d_time, d_feature, triangular='lower'):
        self.W = nn.Parameter(torch.Tensor(d_time, d_time))
        self.b = nn.Parameter(torch.Tensor(d_feature))
        # 生成矩阵掩码，只计算“看见”的时间点
        if triangular == 'lower':
            m = torch.tril(torch.ones(d_time, d_time), 0)
        else:
            m = torch.triu(torch.ones(d_time, d_time), 0)
        self.register_buffer('m', m)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-stdv, stdv)
        if self.b is not None:
            self.b.data.uniform_(-stdv, stdv)

    def forward(self, x):
        return torch.einsum('dj,ijk->idk', self.W * self.m, x) + self.b


# 用于学习特征维度的关系,通道独立机制
class RowFeature(nn.Module):
    def __init__(self, configs):
        super(RowFeature, self).__init__()
        self.features = configs.feature_num
        self.d_feature_model = configs.d_feature_model
        self.model = nn.ModuleList()
        self.position_enc = PositionalEncoding(configs.d_feature_model, n_position=self.features)
        self.dropout = nn.Dropout(configs.dropout)
        for i in range(self.features):
            self.model.append(nn.Linear(configs.seq_len, configs.d_feature_model))
        self.encode = nn.ModuleList([EncoderLayer(configs.d_feature_model, configs.d_feature_inner, configs.n_feature_head, configs.d_feature_k,
                                                  configs.d_feature_v, configs.dropout)
                      for _ in range(configs.n_feature_layer)])

    def forward(self, x):
        B = x.shape[0]
        input = x.permute(0, 2, 1)
        y = torch.zeros(B, self.features, self.d_feature_model).to(x)
        for i in range(self.features):
            out = self.model[i](input[:, i, :])
            out = self.dropout(out)
            y[:, i, :] = out
        out = self.position_enc(y)  # shape[B, N, d_state]
        for block in self.encode:
            out = block(out)
        return out


# 用于学习时间维度的关系，在频域维度上进行编码
class ColFeature(nn.Module):
    def __init__(self, configs):
        super(ColFeature, self).__init__()
        self.seq_len = configs.seq_len
        self.w1 = nn.Linear(configs.feature_num, configs.d_model//2)
        self.w2 = nn.Linear(configs.feature_num, configs.d_model//2)
        self.w3 = nn.Linear(configs.feature_num, configs.d_model//2)
        self.w4 = nn.Linear(configs.feature_num, configs.d_model//2)
        self.embedding = nn.Linear(configs.feature_num, configs.d_model//2)
        self.position_enc = PositionalEncoding(configs.d_model, n_position=configs.seq_len)
        self.dropout = nn.Dropout(configs.dropout)
        self.model = nn.ModuleList([
            EncoderLayer(configs.d_model, configs.d_inner, configs.n_head, configs.d_k,
                         configs.d_v, configs.dropout)
            for _ in range(configs.n_layer)])

    def forward(self, x):
        # 从频域中学习隐藏状态
        xf_co = torch.fft.rfft(x, dim=1)  # 从时域转换为频域  [B, d_time//2+1, N]
        xf_co_re = self.w1(xf_co.real) - self.w2(xf_co.imag)
        xf_co_im = self.w3(xf_co.real) + self.w4(xf_co.imag)
        x_ = torch.stack([xf_co_re, xf_co_im], dim=-1).float()
        x_ = torch.view_as_complex(x_)
        out = torch.cat([self.embedding(x), torch.fft.irfft(x_, dim=1)[:, -self.seq_len:, :]], dim=-1)

        out = self.dropout(out)
        out = self.position_enc(out)  # shape[B, seq_len, d_state]

        for block in self.model:
            out = block(out)
        return out


# 基于时间与特征双视角注意力融合策略
class TFITS(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.n_layer = configs.n_layer  # 注意力组数
        self.n_head = configs.n_head  # 注意力头数量
        self.device = configs.device
        self.seq_len = configs.seq_len
        self.feature_num = configs.feature_num
        self.norm = configs.norm

        # 时间与特征双分支Transformer架构
        self.time_angle = ColFeature(configs)
        self.feature_angle = RowFeature(configs)
        # 映射
        self.time_out = nn.Linear(configs.d_model, configs.feature_num)
        self.feature_out = nn.Linear(configs.d_feature_model, configs.seq_len)
        # self.w = nn.Parameter(torch.zeros(configs.seq_len, configs.feature_num, 4))
        self.w = UNetFuse(5, 1, configs.feature_num)


    def impute(self, inputs):
        X, masks = inputs['X'], inputs['missing_mask']

        if self.norm:
            means = X.mean(1, keepdim=True).detach()
            X = X - means
            stdev = torch.sqrt(
                torch.var(X, dim=1, keepdim=True, unbiased=False) + 1e-5)
            X /= stdev

        ###### 双视角特征提取
        enc_time = self.time_angle(X)
        enc_feature = self.feature_angle(X)
        time_out = self.time_out(enc_time)
        feature_out = self.feature_out(enc_feature).permute(0, 2, 1)
        # 自适应的拼接权重
        reconstruction_MAE_1 = torch.abs(time_out - X) * masks
        reconstruction_MAE_2 = torch.abs(feature_out - X) * masks
        fusion_weight = torch.stack([reconstruction_MAE_1, reconstruction_MAE_2, time_out, feature_out, masks], dim=1)  # (B,3,seq_len,N)

        out = self.w(fusion_weight).squeeze(1)
        if self.norm:
            X = X * stdev + means
            out = out * stdev + means

        imputed_data = masks * X + (1 - masks) * out  # 根据掩码融合数据

        return imputed_data, out

    def forward(self, inputs, stage):
        X, masks = inputs['X'], inputs['missing_mask']

        imputed_data, X_tilde_1 = self.impute(inputs)

        # 计算重构损失
        final_reconstruction_MAE = masked_mae_cal(X_tilde_1, X, masks)
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