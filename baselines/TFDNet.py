import torch
from torch import nn
import torch.nn.functional as F
from utils import masked_mae_cal



class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        if self.subtract_last:
            self.last = x[:,-1,:].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x

# Encoder

def complex_tanh(input):
    return F.tanh(input.real).type(torch.complex64) + 1j * F.tanh(input.imag).type(torch.complex64)


class seasonal_encoder(nn.Module):
    def __init__(self, enc_in, kernel_num, individual, mode, seq_len, n_fft = [48], dropout = 0.05):
        super(seasonal_encoder, self).__init__()
        self.enc = enc_in
        self.n_fft = n_fft
        self.window = int((seq_len / (n_fft * 0.5)) + 1)  # window number
        self.window_len = int(n_fft / 2) + 1  # window length

        self.mode = mode
        if self.mode == 'MK':
            self.wg1 = nn.Parameter(torch.rand(kernel_num, self.window_len, self.window, dtype = torch.cfloat))
            self.wc = nn.Parameter(
                torch.rand(kernel_num, self.window_len, self.window, self.window, dtype = torch.cfloat))
            nn.init.xavier_normal_(self.wg1)
            nn.init.xavier_normal_(self.wc)
        elif self.mode == 'IK':
            self.wc1 = nn.Parameter(torch.rand(enc_in, individual, dtype = torch.cfloat))
            self.wc2 = nn.Parameter(
                torch.rand(individual, int(n_fft / 2) + 1, self.window, self.window, dtype = torch.cfloat))
            nn.init.xavier_normal_(self.wc1)
            nn.init.xavier_normal_(self.wc2)

        self.wf1 = nn.Parameter(torch.rand(self.window_len, self.window_len, dtype = torch.cfloat))
        self.bf1 = nn.Parameter(torch.rand(int(n_fft / 2) + 1, 1, dtype = torch.cfloat))
        self.wf2 = nn.Parameter(torch.rand(self.window_len, self.window_len, dtype = torch.cfloat))
        self.bf2 = nn.Parameter(torch.rand(self.window_len, 1, dtype = torch.cfloat))
        self.norm = nn.LayerNorm(seq_len)
        self.dropout = nn.Dropout(p = dropout)

        nn.init.xavier_normal_(self.wf1)
        nn.init.xavier_normal_(self.bf1)
        nn.init.xavier_normal_(self.wf2)
        nn.init.xavier_normal_(self.bf2)

    def forward(self, q):
        # STFT
        xq_stft = torch.stft(q, n_fft = self.n_fft, return_complex = True,
                             hop_length = int(self.n_fft * 0.5))  # [B*N,M,N]
        # seasonal-TFB
        if self.mode == 'MK':
            g = self.dropout(F.sigmoid(torch.abs(torch.einsum("bhw,nhw->bn", xq_stft, self.wg1)))).cfloat()  # [B,k]
            h = torch.einsum("bhi,nhio->bnho", xq_stft, self.wc)  # [B*C,k,M,N]
            out = torch.einsum("bnhw,bn->bhw", h, g)
        elif self.mode == 'IK':
            xq_stft = xq_stft.reshape(int(q.shape[0] / self.enc), self.enc, xq_stft.shape[1],
                                      xq_stft.shape[2])  # [B,C,M,N]
            wc = torch.einsum("fi,ihlw->fhlw", self.wc1, self.wc2)  # [B,C,M,N]
            out = torch.einsum("bfhi,fhio->bfho", xq_stft, wc)
            out = out.reshape(out.shape[0] * out.shape[1], out.shape[2], out.shape[3])
        # Frequency-FFN
        out_ = out
        out = torch.einsum("biw,io->bow", out, self.wf1) + self.bf1.repeat(1, out.shape[2])
        out = complex_tanh(out)
        out = out_ + out

        # Inverse STFB
        out = torch.istft(out, n_fft = self.n_fft, hop_length = int(self.n_fft * 0.5))
        out = self.dropout(out)
        return out


class trend_encoder(nn.Module):
    def __init__(self, seq_len, n_fft = [48], dropout = 0.05):
        super(trend_encoder, self).__init__()
        self.n_fft = n_fft
        self.window = int((seq_len / (n_fft * 0.5)) + 1)  # window number N
        self.window_len = int(1 * (int(n_fft / 2) + 1))  # window length M
        self.wc = nn.Parameter(torch.rand(self.window_len, self.window, self.window, dtype = torch.cfloat))

        self.wf1 = nn.Parameter(torch.rand((int(n_fft / 2) + 1), int(n_fft / 2) + 1, dtype = torch.cfloat))
        self.bf1 = nn.Parameter(torch.rand((int(n_fft / 2) + 1), 1, dtype = torch.cfloat))

        self.dropout = nn.Dropout(p = dropout)
        nn.init.xavier_normal_(self.wc)
        nn.init.xavier_normal_(self.wf1)
        nn.init.xavier_normal_(self.bf1)

    def forward(self, q):
        # STFB
        xq_stft = torch.stft(q, n_fft = self.n_fft, return_complex = True,
                             hop_length = int(self.n_fft * 0.5))  # [B*C,M.N]
        # Trend-TFB
        h = torch.einsum("bhi,hio->bho", xq_stft, self.wc)  # [B,n_channel,M,N]

        h_ = h
        h = torch.einsum("biw,io->bow", h, self.wf1) + self.bf1.repeat(1, h.shape[2])
        h = complex_tanh(h)
        out = h_ + h
        # Inverse STFB
        out = torch.istft(out, n_fft = self.n_fft, hop_length = int(self.n_fft * 0.5))
        out = self.dropout(out)
        return out


class Encoder(nn.Module):
    def __init__(self, enc_in, seq_len = 512, kernel_num = 16, individual = 7, mode = 'MK', n_fft = [16],
                 dropout = 0.05):
        super(Encoder, self).__init__()
        self.block_seasonal = seasonal_encoder(enc_in = enc_in, kernel_num = kernel_num, individual = individual,
                                               mode = mode, seq_len = seq_len,
                                               n_fft = n_fft, dropout = dropout)
        self.block_trend = trend_encoder(seq_len = seq_len, n_fft = n_fft, dropout = dropout)

    def forward(self, q1, q2):
        seasonal = self.block_seasonal(q1)
        trend = self.block_trend(q2)
        return seasonal, trend


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size = kernel_size, stride = stride, padding = 0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim = 1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride = 1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.channels = configs.feature_num
        self.encoder = nn.ModuleList()
        self.n_fft = configs.n_fft  # 不同频域尺度的列表
        self.encoder_list = nn.ModuleList()
        for i in self.n_fft:
            self.encoder_list.append(Encoder(enc_in = configs.feature_num, seq_len = configs.seq_len, n_fft = i,
                                             dropout = configs.dropout, kernel_num = configs.kernel_num,
                                             individual = configs.individual_factor,
                                             mode = configs.mode))
        self.mlp1 = nn.Linear(len(self.n_fft), 1, bias = False)
        self.mlp2 = nn.Linear(len(self.n_fft), 1, bias = False)
        self.dropout = nn.Dropout(p = configs.dropout)
        self.linear1 = nn.Linear(configs.seq_len, configs.seq_len)

        self.decompsition = series_decomp(configs.kernel_size)

        self.revin_layer = RevIN(configs.feature_num, affine = False, subtract_last = False)

    def impute(self, inputs):
        x_enc, masks = inputs['X'], inputs['missing_mask']
        X = x_enc
        B, L, variation = x_enc.shape
        x_enc = self.revin_layer(x_enc, 'norm')

        seasonal_init, trend_init = self.decompsition(x_enc)

        trend_init = trend_init.permute(0, 2, 1)
        trend_init = trend_init.reshape(B * variation, L)

        seasonal_init = seasonal_init.permute(0, 2, 1)
        seasonal_init = seasonal_init.reshape(B * variation, L)

        out_seasonal = torch.zeros((B * variation, L, len(self.n_fft))).to(seasonal_init.device)
        out_trend = torch.zeros((B * variation, L, len(self.n_fft))).to(trend_init.device)
        for index, encoder in enumerate(self.encoder_list):
            out_seasonal[:, :, index], out_trend[:, :, index] = encoder(seasonal_init, trend_init)
        if len(self.n_fft) > 1:
            out_seasonal = self.mlp1(out_seasonal).squeeze(dim = -1)
            out_trend = self.mlp2(out_trend).squeeze(dim = -1)
        else:
            out_seasonal = out_seasonal.squeeze(dim = -1)
            out_trend = out_trend.squeeze(dim = -1)

        out = out_seasonal + out_trend
        out = self.linear1(out)
        out = self.dropout(out)
        out = out.reshape(B, variation, self.seq_len)

        out = out.permute(0, 2, 1)
        out = self.revin_layer(out, 'denorm')
        X_c = masks * X + (1 - masks) * out
        return X_c, [out]

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
