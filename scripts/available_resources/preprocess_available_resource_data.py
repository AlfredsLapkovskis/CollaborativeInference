import argparse
import pandas as pd

from utils.available_resources_model import *


def run(args):
    bandwidth_path = args.bandwidth_path
    resource_path = args.resource_path
    output_path = args.output_path

    df_bandwidth = (
        pd.read_csv(bandwidth_path, parse_dates=["timestamp"])
        .set_index("timestamp")
        .rename(columns={"download": "bandwidth_in", "upload": "bandwidth_out"})[["bandwidth_in", "bandwidth_out"]]
    )

    df_resources = pd.read_csv(resource_path, parse_dates=["timestamp"])
    df_resources["cpu_percent"] = df_resources["cpu_total"] - \
        df_resources["cpu_percent"]

    df_resources = (
        df_resources
        .set_index("timestamp")
        .rename(columns={
                "storage_free": "storage",
                "memory_available": "memory",
                "cpu_percent": "cpu",
                })[["storage", "memory", "cpu"]]
    )

    df_all = df_bandwidth.join(df_resources, how="outer")

    start = max(df_bandwidth.index.min(), df_resources.index.min())
    end = min(df_bandwidth.index.max(), df_resources.index.max())

    df_all = df_all[(df_all.index >= start) & (df_all.index <= end)]

    df_all: pd.DataFrame = (
        df_all
        .resample("1S").mean()
        .interpolate(method="time", limit_direction="both")
    )

    df_all.to_csv(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bandwidth_path", type=str, required=True,
                        help="Path to bandwidth data")
    parser.add_argument("--resource_path", type=str, required=True,
                        help="Path to resource data")
    parser.add_argument("--output_path", type=str, required=False,
                        help="Path to save the preprocessed data")

    args = parser.parse_args()
    run(args)
