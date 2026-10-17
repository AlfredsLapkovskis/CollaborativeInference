import os
import pathlib
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict

from utils.resource_consumption_model import *


def run(args):
    data_path_pattern = args.data_path_pattern
    output_dir = args.output_dir
    output_name_prefix = args.output_name_prefix

    paths = [str(p) for p in pathlib.Path().glob(data_path_pattern)]

    last_trial = 0
    df_all = pd.DataFrame()
    for path in paths:
        curr_df = pd.read_csv(
            path, parse_dates=["timestamp"]).set_index("timestamp")
        curr_df.loc[:, "conf_trial"] += last_trial
        # +1 to distinguish between different csv files
        last_trial = curr_df["conf_trial"].max() + 1
        df_all = pd.concat([df_all, curr_df])

    config_columns = ["conf_bandwidth_in",
                      "conf_bandwidth_out", "conf_trial", "conf_model_name", "conf_task_count"]

    last_trial = 0
    dfs = []
    for conf, df in df_all.groupby(config_columns):
        columns = ["res_cpu", "res_memory", "res_net_read", "res_net_write"]
        df = (
            df[columns]
            .resample("1s").mean()
            .interpolate(method="time", limit_direction="both")
        )

        conf_bandwidth_in = conf[0] * 8  # To bit/s
        conf_bandwidth_out = conf[1] * 8  # To bit/s
        # To bit/s and fix possible measurement errors
        df["res_net_read"] = np.clip(
            df["res_net_read"] * 8, a_min=0., a_max=conf_bandwidth_in)
        df["res_net_write"] = np.clip(
            df["res_net_write"] * 8, a_min=0., a_max=conf_bandwidth_out)
        df["conf_bandwidth_in"] = conf_bandwidth_in
        df["conf_bandwidth_out"] = conf_bandwidth_out
        df["conf_trial"] = conf[2] + last_trial
        df["conf_model_name"] = conf[3]
        df["conf_task_count"] = conf[4]

        last_trial = df["conf_trial"].max() + 1

        dfs.append(df)

    conf_res_net_reads = defaultdict(list)
    conf_res_net_writes = defaultdict(list)

    for df in dfs:
        conf = (df["conf_bandwidth_in"].iloc[0],
                df["conf_bandwidth_out"].iloc[0], df["conf_task_count"].iloc[0])
        res_net_reads = df[df["res_net_read"] > 0]["res_net_read"].tolist()
        res_net_writes = df[df["res_net_write"] > 0]["res_net_write"].tolist()

        conf_res_net_reads[conf].extend(res_net_reads)
        conf_res_net_writes[conf].extend(res_net_writes)

    for conf in conf_res_net_reads.keys():
        assert len(conf_res_net_reads[conf]) > 0
        assert len(conf_res_net_writes[conf]) > 0

    dfs_downloading = []
    dfs_uploading = []
    dfs_executing = []

    for i, df in enumerate(dfs):
        print(
            f"\rProcessing df {i + 1}/{len(dfs)} with conf {df.iloc[0][config_columns].to_dict()}", end="")

        conf_net = (df["conf_bandwidth_in"].iloc[0],
                    df["conf_bandwidth_out"].iloc[0], df["conf_task_count"].iloc[0])

        if df["res_net_write"].max() == 0.:
            df.iloc[-1, df.columns.get_loc("res_net_write")] = np.random.choice(
                conf_res_net_writes[conf_net])

        uploading_index = df["res_net_write"] > 0
        first_upload_idx = df.index[uploading_index][0]
        last_upload_idx = df.index[uploading_index][-1]

        if not np.all((df["res_net_read"] > 0) == ((df["res_net_read"] > 0) & (df["res_net_write"] > 0))):
            df.loc[uploading_index, "res_net_read"] = 0

        df = df.loc[:last_upload_idx]
        uploading_index = uploading_index.loc[:last_upload_idx]

        df_uploading = df.loc[first_upload_idx: last_upload_idx].copy()
        df_uploading.loc[:, "res_net_read"] = 0

        # Fix uploading zeros

        res_net_write_zeros = df_uploading["res_net_write"] == 0.
        n_res_net_write_zeros = res_net_write_zeros.sum()
        if n_res_net_write_zeros > 0:
            df_uploading.loc[res_net_write_zeros, "res_net_write"] = np.random.choice(
                conf_res_net_writes[conf_net], size=n_res_net_write_zeros)

        if df["res_net_read"].max() == 0.:
            df.iloc[0, df.columns.get_loc("res_net_read")] = np.random.choice(
                conf_res_net_reads[conf_net])

        downloading_index = df["res_net_read"] > 0
        last_download_idx = df.index[downloading_index][-1]

        df_downloading = df.loc[:last_download_idx].copy()
        df_downloading.loc[:, "res_net_write"] = 0

        # Fix downloading zeros

        res_net_read_zeros = df_downloading["res_net_read"] == 0.
        n_res_net_read_zeros = res_net_read_zeros.sum()
        if n_res_net_read_zeros > 0:
            df_downloading.loc[res_net_read_zeros, "res_net_read"] = np.random.choice(
                conf_res_net_reads[conf_net], size=n_res_net_read_zeros)

        first_upload_idx = df.index[uploading_index][0]

        df_executing = df.loc[last_download_idx: first_upload_idx].copy()
        if len(df_executing) == 0:
            # Execution happened quickly between download and upload
            df_executing = df.loc[first_upload_idx: last_download_idx].copy()

        df_executing.loc[:, "res_net_read"] = 0
        df_executing.loc[:, "res_net_write"] = 0

        assert len(df_downloading) > 0
        assert len(df_executing) > 0
        assert len(df_uploading) > 0

        dfs_downloading.append(df_downloading)
        dfs_uploading.append(df_uploading)
        dfs_executing.append(df_executing)
    print()

    df_downloading_result = pd.concat(dfs_downloading)
    df_uploading_result = pd.concat(dfs_uploading)
    df_executing_result = pd.concat(dfs_executing)

    df_downloading_result.to_csv(os.path.join(
        output_dir, f"{output_name_prefix}_downloading.csv"))
    df_uploading_result.to_csv(os.path.join(
        output_dir, f"{output_name_prefix}_uploading.csv"))
    df_executing_result.to_csv(os.path.join(
        output_dir, f"{output_name_prefix}_executing.csv"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path_pattern", type=str, required=True,
                        help="Path pattern to resource consumption data files (CSV), e.g., 'data/resource_consumption_*.csv'")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Path to save the processed data csv")
    parser.add_argument("--output_name_prefix", type=str, required=True,
                        help="Filename prefix for the processed data csv")

    args = parser.parse_args()
    run(args)
