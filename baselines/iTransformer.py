import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from utils import masked_mae_cal
from math import sqrt


class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask

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


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, attention_dropout=0.1):
        super(FullAttention, self).__init__()
        self.mask_flag = mask_flag
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)
        return V.contiguous()


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        out = out.view(B, L, -1)

        return self.out_projection(out)


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        new_x = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)
        # mlp
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y)


class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        for attn_layer in self.attn_layers:
            x = attn_layer(x, attn_mask=attn_mask)

        if self.norm is not None:
            x = self.norm(x)

        return x


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.enc_embedding = nn.Linear(configs.seq_len, configs.d_model)
        self.position_enc = PositionalEncoding(configs.d_model, n_position=configs.feature_num)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, attention_dropout=configs.dropout), configs.d_model, configs.n_head),
                    configs.d_model,
                    configs.d_inner,
                    dropout=configs.dropout,
                    activation='gelu'
                ) for _ in range(configs.n_layer)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        self.dropout = nn.Dropout(p=configs.dropout)
        self.projection = nn.Linear(configs.d_model, configs.seq_len, bias=True)

    def impute(self, inputs):
        X, masks = inputs['X'], inputs['missing_mask']
        L = X.shape[1]
        means = X.mean(1, keepdim=True).detach()
        x_enc = X - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        input_X = self.enc_embedding(x_enc.permute(0, 2, 1))  # 特征映射
        enc_out = self.dropout(self.position_enc(input_X))  # 位置编码
        enc_out = self.encoder(enc_out, attn_mask=None)

        dec_out = self.projection(enc_out).permute(0, 2, 1)
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