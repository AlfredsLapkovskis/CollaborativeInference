import pathlib
import argparse
import numpy as np
import pandas as pd
from utils.resource_consumption_model import *


def run(args):
    data_path_pattern = args.data_path_pattern
    output_path = args.output_path

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

    config_columns = ["conf_bandwidth_in", "conf_trial", "conf_model_name"]

    last_trial = 0
    dfs = []
    for conf, df in df_all.groupby(config_columns):
        columns = ["res_cpu", "res_memory", "res_net_read", "res_storage"]
        df = (
            df[columns]
            .resample("1s").mean()
            .interpolate(method="time", limit_direction="both")
        )

        conf_bandwidth_in = conf[0] * 8  # To bit/s
        # To bit/s and fix possible measurement errors
        df["res_net_read"] = np.clip(
            df["res_net_read"] * 8, a_min=0., a_max=conf_bandwidth_in)
        df["res_storage"] = np.clip(df["res_storage"].diff().fillna(
            0), a_min=0., a_max=None)  # Keep deltas instead of accumulated storage
        df["conf_bandwidth_in"] = conf_bandwidth_in
        df["conf_trial"] = conf[1] + last_trial

        last_trial = df["conf_trial"].max() + 1

        dfs.append(df)

    df_result = pd.concat(dfs)
    df_result.to_csv(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path_pattern", type=str, required=True,
                        help="Path pattern to resource consumption data files (CSV), e.g., 'data/resource_consumption_*.csv'")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to save the processed data csv")

    args = parser.parse_args()
    run(args)
