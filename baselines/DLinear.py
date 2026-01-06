import torch
import torch.nn as nn
from utils import masked_mae_cal


class moving_avg(nn.Module):

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2205.13504.pdf
    """

    def __init__(self, configs):
        """
        individual: Bool, 不同变量之间是否共享模型
        """
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.seq_len

        assert configs.moving_avg % 2 == 1, "configs.moving_avg must be odd number"
        self.decompsition = series_decomp(configs.moving_avg)
        self.individual = configs.channel_independence
        self.channels = configs.feature_num

        if self.individual:  # 表示通道独立，不同特征变量之间使用不同的权重模型
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()

            for i in range(self.channels):
                self.Linear_Seasonal.append(
                    nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Trend.append(
                    nn.Linear(self.seq_len, self.pred_len))

                self.Linear_Seasonal[i].weight = nn.Parameter(
                    (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
                self.Linear_Trend[i].weight = nn.Parameter(
                    (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

            self.Linear_Seasonal.weight = nn.Parameter(
                (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
            self.Linear_Trend.weight = nn.Parameter(
                (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))

    def impute(self, inputs):
        x, masks = inputs['X'], inputs['missing_mask']

        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(
            0, 2, 1), trend_init.permute(0, 2, 1)
        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0), seasonal_init.size(1), self.pred_len],
                                          dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0), trend_init.size(1), self.pred_len],
                                       dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](
                    seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](
                    trend_init[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)
        out = (seasonal_output + trend_output).permute(0, 2, 1)
        X_c = masks * x + (1 - masks) * out
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
