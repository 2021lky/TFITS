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
    parser = argparse.ArgumentParser(description="Generate UCI AirQuality dataset")
    parser.add_argument("--file_path", help="path of dataset file", type=str, default="./origin/AirQuality/PRSA_Data_20130301-20170228")
    parser.add_argument(
        "--artificial_missing_rate",
        help="artificially mask out additional values",
        type=float,
        default=0.1,
    )
    parser.add_argument("--seq_len", help="sequence length", type=int, default=24)
    parser.add_argument(
        "--dataset_name",
        help="name of generated dataset, will be the name of saving dir",
        type=str,
        default="AirQuality",
    )
    parser.add_argument(
        "--saving_path", type=str, help="parent dir of generated dataset", default="../generated_datasets"
    )
    args = parser.parse_args()

    args.dataset_name = "{}_seq_len{}_rate{}".format(args.dataset_name, args.seq_len, args.artificial_missing_rate)
    dataset_saving_dir = os.path.join(args.saving_path, args.dataset_name)
    if not os.path.exists(dataset_saving_dir):
        os.makedirs(dataset_saving_dir)


    df_collector = []
    station_name_collector = []  # 收集站点信息
    file_list = os.listdir(args.file_path)
    for filename in file_list:  # 每个数据文件表示为该站点的传感器信息
        file_path = os.path.join(args.file_path, filename)
        current_df = pd.read_csv(file_path)
        current_df["date"] = pd.to_datetime(
            current_df[["year", "month", "day", "hour"]]
        )
        station_name_collector.append(current_df.loc[0, "station"])

        current_df = current_df.drop(
            ["year", "month", "day", "hour", "wd", "No", "station"], axis=1
        )
        df_collector.append(current_df)
        print(f'reading {file_path}, data shape {current_df.shape}')

    print(
        f"There are total {len(station_name_collector)} stations, they are {station_name_collector}"
    )
    date = df_collector[0]["date"]
    df_collector = [i.drop("date", axis=1) for i in df_collector]
    df = pd.concat(df_collector, axis=1)
    args.feature_names = [
        station + "_" + feature
        for station in station_name_collector
        for feature in df_collector[0].columns
    ]  # 收集不同站点的不同特征信息，合计为总的特征数目
    args.feature_num = len(args.feature_names)
    df.columns = args.feature_names
    print(
        f"Original df missing rate: "
        f"{(df[args.feature_names].isna().sum().sum() / (df.shape[0] * args.feature_num)):.3f}"
    )

    df["date"] = date

    unique_days = df["date"].dt.to_period("D").unique()
    # 对于每天采集不够seq_len大小的数据集，删除

    date_counts = df.groupby(df["date"].dt.to_period("D")).size()

    # 长度不满足直接删除
    satisfic_days = date_counts[date_counts == args.seq_len].index

    length = len(satisfic_days)

    test_num = int(length * 0.15)
    val_num = int(length * 0.15)
    train_num = length - test_num - val_num

    selected_as_train = satisfic_days[:train_num]
    print(f"months selected as train set are {selected_as_train}")
    selected_as_val = satisfic_days[train_num:train_num+val_num]
    print(f"months selected as val set are {selected_as_val}")
    selected_as_test = satisfic_days[train_num+val_num:]
    print(f"months selected as test set are {selected_as_test}")

    test_set = df[df["date"].dt.to_period("D").isin(selected_as_test)]
    val_set = df[df["date"].dt.to_period("D").isin(selected_as_val)]
    train_set = df[df["date"].dt.to_period("D").isin(selected_as_train)]

    scaler = StandardScaler()
    train_set_X = scaler.fit_transform(train_set.loc[:, args.feature_names])
    val_set_X = scaler.transform(val_set.loc[:, args.feature_names])
    test_set_X = scaler.transform(test_set.loc[:, args.feature_names])

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
