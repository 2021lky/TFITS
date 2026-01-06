import argparse
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from utils import (
    add_artificial_mask,
    saving_into_h5,
)

np.random.seed(26)


def process_each_set(set_df, all_labels):
    # gene labels, y
    sample_ids = set_df["RecordID"].to_numpy().reshape(-1, 48)[:, 0]
    y = all_labels.loc[sample_ids].to_numpy().reshape(-1, 1)
    # gene feature vectors, X
    set_df = set_df.drop("RecordID", axis=1)
    feature_names = set_df.columns.tolist()
    X = set_df.to_numpy()
    X = X.reshape(len(sample_ids), 48, len(feature_names))
    return X, y


def keep_only_features_to_normalize(all_feats, to_remove):
    for i in to_remove:
        all_feats.remove(i)
    return all_feats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate PhysioNet2012 dataset")
    parser.add_argument(
        "--file_path", help="path of physio 2012 raw dataset", type=str, default="./origin/PhysioNet2012"
    )
    parser.add_argument("--seq_len", help="sequence length", type=int, default=48)
    parser.add_argument(
        "--artificial_missing_rate",
        help="artificially mask out additional values",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--dataset_name",
        help="name of generated dataset, will be the name of saving dir",
        type=str,
        default="PhysioNet2012",
    )
    parser.add_argument(
        "--saving_path", type=str, help="parent dir of generated dataset", default="../generated_datasets"
    )
    args = parser.parse_args()

    args.dataset_name = "{}_seq_len{}_rate{}".format(args.dataset_name, args.seq_len, args.artificial_missing_rate)
    dataset_saving_dir = os.path.join(args.saving_path, args.dataset_name)
    if not os.path.exists(dataset_saving_dir):
        os.makedirs(dataset_saving_dir)

    # 处理分类目标值
    outcome_files = ["Outcomes-a.txt", "Outcomes-b.txt", "Outcomes-c.txt"]
    outcome_collector = []
    for o_ in outcome_files:
        outcome_file_path = os.path.join(args.file_path, o_)
        with open(outcome_file_path, "r") as f:
            outcome = pd.read_csv(f)[["In-hospital_death", "RecordID"]]
        outcome = outcome.set_index("RecordID")
        outcome_collector.append(outcome)
    all_outcomes = pd.concat(outcome_collector)

    # 处理特征值文件
    all_recordID = []
    df_collector = []
    outcome_dirs = ["set-a", "set-b", "set-c"]
    for dir in outcome_dirs:
        outcome_dir_path = os.path.join(args.file_path, dir)
        for filename in os.listdir(outcome_dir_path):
            recordID = int(filename.split(".txt")[0])
            with open(os.path.join(outcome_dir_path, filename), "r") as f:
                df_temp = pd.read_csv(f)
            df_temp["Time"] = df_temp["Time"].apply(lambda x: int(x.split(":")[0]))
            df_temp = df_temp.pivot_table("Value", "Time", "Parameter")
            df_temp = df_temp.reset_index()  # take Time from index as a col
            if len(df_temp) == 1:
                print(
                    f"Pass {recordID}, because its len==1, having no time series data"
                )
                continue
            all_recordID.append(recordID)  # only count valid recordID
            if df_temp.shape[0] != 48:
                missing = list(set(range(0, 48)).difference(set(df_temp["Time"])))
                missing_part = pd.DataFrame({"Time": missing})
                df_temp = pd.concat([df_temp, missing_part], ignore_index=False, sort=False)
                df_temp = df_temp.set_index("Time").sort_index().reset_index()
            df_temp = df_temp.iloc[:48]  # only take 48 hours, some samples may have more records, like 49 hours
            df_temp["RecordID"] = recordID
            df_temp["Age"] = df_temp.loc[0, "Age"]
            df_temp["Height"] = df_temp.loc[0, "Height"]
            df_collector.append(df_temp)

    df = pd.concat(df_collector, sort=True)
    df = df.drop(["Age", "Gender", "ICUType", "Height"], axis=1)
    df = df.reset_index(drop=True)
    df = df.drop("Time", axis=1)  # dont need Time col

    length = len(all_recordID)
    test_num = int(length * 0.15)
    val_num = int(length * 0.15)
    train_num = length - test_num - val_num

    train_set_ids = all_recordID[:train_num]
    val_set_ids = all_recordID[test_num:train_num+val_num]
    test_set_ids = all_recordID[train_num+val_num:]

    print(f"There are total {len(train_set_ids)} patients in train set.")
    print(f"There are total {len(val_set_ids)} patients in val set.")
    print(f"There are total {len(test_set_ids)} patients in test set.")

    all_features = df.columns.tolist()
    feat_no_need_to_norm = ["RecordID"]
    feats_to_normalize = keep_only_features_to_normalize(
        all_features, feat_no_need_to_norm
    )

    train_set = df[df["RecordID"].isin(train_set_ids)]
    val_set = df[df["RecordID"].isin(val_set_ids)]
    test_set = df[df["RecordID"].isin(test_set_ids)]

    # standardization
    scaler = StandardScaler()
    train_set.loc[:, feats_to_normalize] = scaler.fit_transform(
        train_set.loc[:, feats_to_normalize]
    )
    val_set.loc[:, feats_to_normalize] = scaler.transform(
        val_set.loc[:, feats_to_normalize]
    )
    test_set.loc[:, feats_to_normalize] = scaler.transform(
        test_set.loc[:, feats_to_normalize]
    )

    train_set_X, train_set_y = process_each_set(train_set, all_outcomes)
    val_set_X, val_set_y = process_each_set(val_set, all_outcomes)
    test_set_X, test_set_y = process_each_set(test_set, all_outcomes)

    train_set_dict = add_artificial_mask(
        train_set_X, args.artificial_missing_rate, "train"
    )
    val_set_dict = add_artificial_mask(val_set_X, args.artificial_missing_rate, "val")
    test_set_dict = add_artificial_mask(
        test_set_X, args.artificial_missing_rate, "test"
    )
    print(
        f'In val set, num of artificially-masked values: {val_set_dict["indicating_mask"].sum()}'
    )
    print(
        f'In test set, num of artificially-masked values: {test_set_dict["indicating_mask"].sum()}'
    )

    train_set_dict["labels"] = train_set_y
    val_set_dict["labels"] = val_set_y
    test_set_dict["labels"] = test_set_y

    processed_data = {
        "train": train_set_dict,
        "val": val_set_dict,
        "test": test_set_dict,
        "mean": scaler.mean_,
        'var': scaler.scale_
    }

    saving_into_h5(dataset_saving_dir, processed_data, classification_dataset=True)
    print(f"All done. Saved to {dataset_saving_dir}.")
