import argparse
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler

from utils import (
    window_truncate,
    add_artificial_mask,
    saving_into_h5,
)

# features:352
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Generate PEMS-Bay dataset")
    parser.add_argument("--file_path", help="path of dataset file", type=str,
                        default='./origin/PEMS-BAY/pems-bay.h5')
    parser.add_argument("--artificial_missing_rate", help="人工设置缺失率，在测试集和验证集中设置", type=float,
                        default=0.1)
    parser.add_argument("--seq_len", help="sequence length", type=int, default=288)
    parser.add_argument("--dataset_name", help="生成数据集的名称", type=str, default="PEMS-BAY")
    parser.add_argument(
        "--saving_path", type=str, help="生成数据集存放目录", default="../generated_datasets"
    )
    args = parser.parse_args()

    args.dataset_name = "{}_seq_len{}_rate{}".format(args.dataset_name, args.seq_len, args.artificial_missing_rate)
    dataset_saving_dir = os.path.join(args.saving_path, args.dataset_name)
    if not os.path.exists(dataset_saving_dir):
        os.makedirs(dataset_saving_dir)

    df = pd.read_hdf(args.file_path)
    print(df.shape)
    length = len(df) // args.seq_len

    test_num = int(length * 0.15)
    val_num = int(length * 0.15)
    train_num = length - test_num - val_num

    test_set = df[(train_num + val_num) * args.seq_len:]
    val_set = df[train_num * args.seq_len:(train_num + val_num) * args.seq_len]
    train_set = df[0:train_num * args.seq_len]

    scaler = StandardScaler()
    train_set_X = scaler.fit_transform(train_set)
    val_set_X = scaler.transform(val_set)
    test_set_X = scaler.transform(test_set)

    train_set_X = window_truncate(train_set_X, args.seq_len)
    val_set_X = window_truncate(val_set_X, args.seq_len)
    test_set_X = window_truncate(test_set_X, args.seq_len)

    train_set_dict = add_artificial_mask(
        train_set_X, args.artificial_missing_rate, "train"
    )
    val_set_dict = add_artificial_mask(
        val_set_X, args.artificial_missing_rate, "val"
    )
    test_set_dict = add_artificial_mask(
        test_set_X, args.artificial_missing_rate, "test"
    )
    print(
        f'In val set, num of artificially-masked values: {val_set_dict["indicating_mask"].sum()}'
    )
    print(
        f'In test set, num of artificially-masked values: {test_set_dict["indicating_mask"].sum()}'
    )

    processed_data = {
        "train": train_set_dict,
        "val": val_set_dict,
        "test": test_set_dict,
        "mean": scaler.mean_,
        'var': scaler.scale_
    }

    saving_into_h5(dataset_saving_dir, processed_data, classification_dataset=False)
    print(f"All done. Saved to {dataset_saving_dir}.")
