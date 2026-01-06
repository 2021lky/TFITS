import os

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class LoadDataset(Dataset):
    def __init__(self, file_path, seq_len, feature_num):
        super(LoadDataset, self).__init__()
        self.file_path = file_path  # 处理后数据集路径
        self.seq_len = seq_len  # 时间窗口长度
        self.feature_num = feature_num  # 特征维度数量


class LoadValTestDataset(LoadDataset):
    """加载验证集或测试集的过程"""

    def __init__(self, file_path, set_name, seq_len, feature_num):
        super(LoadValTestDataset, self).__init__(file_path, seq_len, feature_num)
        with h5py.File(self.file_path, 'r') as hf:  # 从 h5 文件读取数据
            self.X = hf[set_name]['X'][:]
            self.X_hat = hf[set_name]['X_hat'][:]
            self.missing_mask = hf[set_name]['missing_mask'][:]
            self.indicating_mask = hf[set_name]['indicating_mask'][:]

        # 用 0 填充缺失值
        self.X = np.nan_to_num(self.X)
        self.X_hat = np.nan_to_num(self.X_hat)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        sample = (
            torch.tensor(idx),
            torch.from_numpy(self.X_hat[idx].astype('float32')),
            torch.from_numpy(self.missing_mask[idx].astype('float32')),
            torch.from_numpy(self.X[idx].astype('float32')),
            torch.from_numpy(self.indicating_mask[idx].astype('float32')),
        )
        return sample


class LoadTrainDataset(LoadDataset):
    """训练集加载过程"""
    def __init__(self, file_path, seq_len, feature_num, artificial_missing_rate, mask_type='random'):
        super(LoadTrainDataset, self).__init__(file_path, seq_len, feature_num)
        self.mask_type = mask_type
        self.artificial_missing_rate = artificial_missing_rate
        assert 0 < self.artificial_missing_rate < 1, 'artificial_missing_rate 应大于 0 且小于 1'

        with h5py.File(self.file_path, 'r') as hf:  # 从 h5 文件读取数据
            self.X = hf['train']['X'][:]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X = self.X[idx]  # 原始数据

        T, N = X.shape

        X_hat = np.copy(X)
        if self.mask_type == 'continuous':
            gap_size = int(self.artificial_missing_rate * T // 1)
            start = np.random.randint(0, T-gap_size)
            end = start + gap_size if start + gap_size < T else T
            X_hat[start:end] = np.nan

        elif self.mask_type == 'random':
            X = X.reshape(-1)
            X_hat = X_hat.reshape(-1)
            indices = np.where(~np.isnan(X))[0].tolist()
            indices = np.random.choice(indices, round(len(indices) * self.artificial_missing_rate))
            X_hat[indices] = np.nan  # 使用 indices 选定的值掩盖值
            # 重塑成时间序列
            X = X.reshape(T, N)
            X_hat = X_hat.reshape(T, N)

        else:
            raise ValueError("the type {} is not exist!".format(self.mask_type))

        missing_mask = (~np.isnan(X_hat)).astype(np.float32)
        indicating_mask = ((~np.isnan(X)) ^ (~np.isnan(X_hat))).astype(np.float32)  # 标识人工设置缺失值

        X = np.nan_to_num(X)
        X_hat = np.nan_to_num(X_hat)

        sample = (
            torch.tensor(idx),
            torch.from_numpy(X_hat.astype('float32')),
            torch.from_numpy(missing_mask.astype('float32')),
            torch.from_numpy(X.astype('float32')),
            torch.from_numpy(indicating_mask.astype('float32')),
        )

        return sample

class UnifiedDataLoader:
    def __init__(self, dataset_path, seq_len, feature_num, batch_size=1024, num_workers=1,
                 artificial_missing_rate=0.2, mask_type='random'):
        """
        dataset_path: 存储 h5 数据集的目录路径;
        seq_len: 序列长度，即时间步长;
        feature_num: 特征数量，即特征维度;
        batch_size: 小批量大小;
        num_workers: 数据加载的子进程数量;
        model_type: 模型类型，确定返回的值;
        artificial_missing_rate: 人工设置训练集的缺失率;
        """
        self.dataset_path = os.path.join(dataset_path, 'datasets.h5')
        self.seq_len = seq_len
        self.feature_num = feature_num
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.artificial_missing_rate = artificial_missing_rate
        self.mask_type = mask_type

        self.train_dataset, self.train_loader, self.train_set_size = None, None, None
        self.val_dataset, self.val_loader, self.val_set_size = None, None, None
        self.test_dataset, self.test_loader, self.test_set_size = None, None, None

    def get_train_val_dataloader(self):
        self.train_dataset = LoadTrainDataset(self.dataset_path, self.seq_len, self.feature_num,
                                              self.artificial_missing_rate, self.mask_type)
        self.val_dataset = LoadValTestDataset(self.dataset_path, 'val', self.seq_len, self.feature_num)
        self.train_set_size = self.train_dataset.__len__()
        self.val_set_size = self.val_dataset.__len__()
        self.train_loader = DataLoader(self.train_dataset, self.batch_size, shuffle=True, num_workers=self.num_workers)
        self.val_loader = DataLoader(self.val_dataset, self.batch_size, shuffle=True, num_workers=self.num_workers)
        return self.train_loader, self.val_loader

    def get_test_dataloader(self):
        self.test_dataset = LoadValTestDataset(self.dataset_path, 'test', self.seq_len, self.feature_num)
        self.test_set_size = self.test_dataset.__len__()
        self.test_loader = DataLoader(self.test_dataset, self.batch_size, shuffle=False, num_workers=self.num_workers)
        return self.test_loader

    def prepare_dataloader_for_imputation(self, set_name):
        data_for_imputation = LoadValTestDataset(self.dataset_path, set_name, self.seq_len, self.feature_num)
        dataloader_for_imputation = DataLoader(data_for_imputation, self.batch_size, shuffle=False)
        return dataloader_for_imputation

    def prepare_all_data_for_imputation(self):
        test_set_for_imputation = self.prepare_dataloader_for_imputation('test')
        return test_set_for_imputation