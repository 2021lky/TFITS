import argparse
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

from utils import (
    window_truncate,
    add_artificial_mask,
    saving_into_h5,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Traffic dataset")
    parser.add_argument("--file_path", help="path of dataset file", type=str, default='./origin/Traffic/traffic.csv')
    parser.add_argument("--artificial_missing_rate", help="人工设置缺失率，在测试集和验证集中设置", type=float, default=0.1)
    parser.add_argument("--seq_len", help="sequence length", type=int, default=24)
    parser.add_argument("--dataset_name", help="生成数据集的名称", type=str, default="Traffic")
    parser.add_argument(
        "--saving_path", type=str, help="生成数据集存放目录", default="../generated_datasets"
    )
    args = parser.parse_args()

    args.dataset_name = "{}_seq_len{}_rate{}".format(args.dataset_name, args.seq_len, args.artificial_missing_rate)
    # 生成数据集路径
    dataset_saving_dir = os.path.join(args.saving_path, args.dataset_name)
    if not os.path.exists(dataset_saving_dir):
        os.makedirs(dataset_saving_dir)

    df = pd.read_csv(args.file_path)
    df["date"] = pd.to_datetime(df["date"])
    feature_names = list(filter(lambda x: x != "date", df.columns.tolist()))

    unique_days = df["date"].dt.to_period("D").unique()
    # 对于每天采集不够seq_len大小的数据集，删除

    date_counts = df.groupby(df["date"].dt.to_period("D")).size()

    # 长度不满足直接删除
    satisfic_days = date_counts[date_counts == args.seq_len].index

    length = len(satisfic_days)

    test_num = int(length * 0.15)
    val_num = int(length * 0.15)
    train_num = length - test_num - val_num

    selected_as_train = satisfic_days[0:train_num]
    print(f"days selected as train set are {selected_as_train}")
    selected_as_val = satisfic_days[train_num:train_num+val_num]
    print(f"days selected as val set are {selected_as_val}")
    selected_as_test = satisfic_days[train_num+val_num:]
    print(f"months selected as test set are {selected_as_test}")

    test_set = df[df["date"].dt.to_period("D").isin(selected_as_test)].drop(columns=["date"])
    val_set = df[df["date"].dt.to_period("D").isin(selected_as_val)].drop(columns=["date"])
    train_set = df[df["date"].dt.to_period("D").isin(selected_as_train)].drop(columns=["date"])

    # 标准化
    scaler = StandardScaler()
    train_set_X = scaler.fit_transform(train_set.loc[:, feature_names])
    val_set_X = scaler.transform(val_set.loc[:, feature_names])
    test_set_X = scaler.transform(test_set.loc[:, feature_names])

    train_set_X = window_truncate(train_set_X, args.seq_len)
    val_set_X = window_truncate(val_set_X, args.seq_len)
    test_set_X = window_truncate(test_set_X, args.seq_len)

    # 人工设置掩码值
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
