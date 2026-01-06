import math
import torch
import torch.nn as nn
import torch.nn.functional as F


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


class ScaledDotProductAttention(nn.Module):
    """缩放点积注意力"""

    def __init__(self, temperature, dropout=0.1):
        super().__init__()
        # 温度参数
        self.temperature = temperature
        # 注意力机制中的dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, attn_mask=None):
        # 基于q、k计算注意力权重
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        # 如果存在注意力掩码，则将掩码应用于注意力权重
        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask == 1, -1e9)

        # 在softmax之前应用dropout
        attn = self.dropout(F.softmax(attn, dim=-1))
        # 计算加权后的v
        output = torch.matmul(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    """原始Transformer多头注意力"""

    def __init__(self, n_head, d_model, d_k, d_v, dropout):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        # 线性变换，用于Q、K、V的投影
        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)

        # 注意力机制
        self.attention = ScaledDotProductAttention(d_k ** 0.5, dropout)
        # LayerNorm
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        # 线性变换，用于多头注意力输出的合并
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

    def forward(self, q, k, v, attn_mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        self.d_time_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
        residual = q
        # 通过线性变换进行投影
        q = self.w_qs(q).view(self.d_time_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(self.d_time_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(self.d_time_b, len_v, n_head, d_v)

        # 转置以便进行注意力计算
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        # 应用注意力掩码
        if attn_mask is not None:
            # 这个掩码是插补掩码，不是每个批次生成的，因此需要在批次维度上进行广播
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(1)  # 对批次和头部维度进行广播

        v, attn_weights = self.attention(q, k, v, attn_mask)

        # 将头部维度转置回来并合并
        v = v.transpose(1, 2).contiguous().view(self.d_time_b, len_q, -1)

        # 线性变换，将多头注意力输出合并
        v = self.fc(v)

        return v, attn_weights


class CrossMultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        # 交叉注意力部分
        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.attention = ScaledDotProductAttention(d_k ** 0.5, dropout)
        # MLP层
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

    def forward(self, q, k, v, attn_mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        self.d_time_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        # 通过线性变换进行投影
        q = self.w_qs(q).view(self.d_time_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(self.d_time_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(self.d_time_b, len_v, n_head, d_v)

        # 转置以便进行注意力计算
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        # 应用注意力掩码
        if attn_mask is not None:
            # 这个掩码是插补掩码，不是每个批次生成的，因此需要在批次维度上进行广播
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(1)  # 对批次和头部维度进行广播

        v, attn_weights = self.attention(q, k, v, attn_mask)

        # 将头部维度转置回来并合并
        v = v.transpose(1, 2).contiguous().view(self.d_time_b, len_q, -1)

        # 线性变换，将多头注意力输出合并
        v = self.fc(v)

        return v, attn_weights


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        # 第一个线性层
        self.w_1 = nn.Linear(d_in, d_hid)
        # 第二个线性层
        self.w_2 = nn.Linear(d_hid, d_in)
        # LayerNorm
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        # dropout层
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        # 应用LayerNorm
        x = self.layer_norm(x)
        # 第一个线性层，ReLU激活函数
        x = self.w_2(F.relu(self.w_1(x)))
        # 应用dropout
        x = self.dropout(x)
        # 加上残差连接
        x += residual
        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()

        self.layer_norm = nn.LayerNorm(d_model)
        # 自注意力层
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout)
        # dropout层
        self.dropout = nn.Dropout(dropout)
        # 前馈网络
        self.pos_ffn = PositionWiseFeedForward(d_model, d_inner, dropout)

    def forward(self, enc_input):
        residual = enc_input
        # 应用LayerNorm
        enc_input = self.layer_norm(enc_input)
        # 自注意力层
        enc_output, _ = self.slf_attn(enc_input, enc_input, enc_input, attn_mask=None)
        # 应用dropout
        enc_output = self.dropout(enc_output)
        # 加上残差连接
        enc_output += residual
        # 经过前馈网络
        enc_output = self.pos_ffn(enc_output)
        return enc_output


class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(DecoderLayer, self).__init__()

        self.layer_norm = nn.LayerNorm(d_model)
        # 自注意力层
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout)
        self.cross_atten = CrossMultiHeadAttention(n_head, d_model, d_k, d_v, dropout)
        # dropout层
        self.dropout = nn.Dropout(dropout)
        # 前馈网络
        self.pos_ffn = PositionWiseFeedForward(d_model, d_inner, dropout)

    def forward(self, time_input, feature_input):
        residual = time_input
        time_input = self.layer_norm(time_input)
        # 自注意力层
        enc_output, _ = self.slf_attn(time_input, time_input, time_input, attn_mask=None)
        # 应用dropout
        enc_output = self.dropout(enc_output)
        # 加上残差连接
        enc_output += residual
        # 交叉注意力机制
        enc_output = self.layer_norm(enc_output)
        residual = enc_output
        enc_output, _ = self.cross_atten(enc_output, feature_input, feature_input)
        enc_output = self.dropout(enc_output)
        enc_output += residual
        # 经过前馈网络
        enc_output = self.pos_ffn(enc_output)
        return enc_output

# class DecoderLayer(nn.Module):
#     def __init__(self, seq_len, feature_num, d_model, d_inner, top_k, n_head, d_k, d_v, dropout=0.1):
#         super(DecoderLayer, self).__init__()
#         self.k = top_k
#         self.seq_len = seq_len
#         self.feature_num = feature_num
#         self.layer_norm = nn.LayerNorm(d_model)
#         # 融合注意力
#         self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout)
#         # A
#         self.A = nn.Linear(d_model, feature_num)
#         self.enhance_layer = FrontFeature(seq_len, d_model)
#         self.max_layer = nn.Linear(d_model, d_model)
#         self.min_layer = nn.Linear(d_model, d_model)
#         self.convert_layer = nn.Linear(d_model, d_model)
#         self.value_layer = nn.Linear(d_model, d_model)
#         # dropout层
#         self.dropout = nn.Dropout(dropout)
#         # 前馈网络
#         self.pos_ffn = PositionWiseFeedForward(d_model, d_inner, dropout)
#
#     def get_top_k(self, A, k):
#         B, S, N = A.shape
#
#         a, _ = A.topk(k=k, dim=2)
#         a_min = torch.min(a, dim=-1).values
#
#         a_min = a_min.unsqueeze(-1).repeat(1, 1, N)
#         ge = torch.ge(A, a_min)
#
#         zero = torch.zeros_like(A)
#         result = torch.where(ge, A, zero)
#
#         return result
#
#     def forward(self, enc_input, dec_input):
#         # 融合注意力
#         dec_output = torch.cat([dec_input, enc_input], dim=1)
#
#         residual = dec_output
#
#         dec_output, _ = self.slf_attn(dec_output, dec_output, dec_output, attn_mask=None)
#
#         # 注意力融合
#         dec_input = dec_output[:, 0:self.seq_len, :]
#         enc_input = dec_output[:, -self.feature_num:, :]
#         A = self.A(dec_input)  # [B, seq_len, N)
#         A = F.softmax(self.get_top_k(A, self.k))
#         value = self.value_layer(enc_input)
#         convert_result = self.convert_layer(torch.einsum("bnd,bsn->bsd", value, A))
#         out = torch.cat([convert_result, enc_input], dim=1)
#         dec_output = residual + out
#         # 经过前馈网络
#         dec_output = self.pos_ffn(dec_output)
#         return dec_output[:, -self.feature_num:, :], dec_output[:, 0:self.seq_len, :]
