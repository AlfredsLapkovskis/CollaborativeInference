import os
import json
import argparse
from itertools import product


def generate(args):
    df_path_x = args.df_path_x
    df_path_y_download = args.df_path_y_download
    df_path_y_execution = args.df_path_y_execution
    df_path_y_upload = args.df_path_y_upload
    output_dir = args.output_dir
    exp_output_dir = args.exp_output_dir

    def get_x_series_names():
        return [
            "res_net_read",
            "res_cpu",
            "res_memory",
            "res_storage",
        ]

    def get_y_series_names(task_state):
        match task_state:
            case "download":
                return ["res_net_read", "res_cpu", "res_memory"]
            case "execution":
                return ["res_cpu", "res_memory"]
            case "upload":
                return ["res_net_write", "res_cpu", "res_memory"]
            case _:
                raise ValueError(f"Invalid task state: {task_state}")

    def generate_x_configs(series_name):
        p = [1, 2, 5, 10]
        d = [0]
        q = [0, 1, 2, 5, 10]

        for p_val, d_val, q_val in product(p, d, q):
            yield {
                "df_path": df_path_x,
                "series_name": series_name,
                "task_type": "x",
                "task_state": "download",
                "arima_order": [p_val, d_val, q_val],
                "output_dir": output_dir,
            }

    def generate_y_configs(series_name, task_state, df_path):
        p = [1, 2, 5, 10]
        d = [0]
        q = [0, 1, 2, 5, 10]

        for p_val, d_val, q_val in product(p, d, q):
            yield {
                "df_path": df_path,
                "series_name": series_name,
                "task_type": "y",
                "task_state": task_state,
                "arima_order": [p_val, d_val, q_val],
                "output_dir": output_dir,
            }

    def save_configs(configs, prefix):
        for i, config in enumerate(configs):
            path = os.path.join(exp_output_dir, f"{prefix}_{i}.json")

            with open(path, "w") as f:
                json.dump(config, f, indent=2)

    for series_name in get_x_series_names():
        save_configs(generate_x_configs(series_name),
                     f"exp_fit_cons_x_download_{series_name}")
    for task_state, df_path in zip(["download", "execution", "upload"], [df_path_y_download, df_path_y_execution, df_path_y_upload]):
        for series_name in get_y_series_names(task_state):
            save_configs(generate_y_configs(series_name, task_state, df_path),
                         f"exp_fit_cons_y_{task_state}_{series_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--df_path_x", type=str, required=True,
                        help="Path to the input dataframe CSV file")
    parser.add_argument("--df_path_y_download", type=str, required=True,
                        help="Path to the input dataframe CSV file")
    parser.add_argument("--df_path_y_execution", type=str, required=True,
                        help="Path to the input dataframe CSV file")
    parser.add_argument("--df_path_y_upload", type=str, required=True,
                        help="Path to the input dataframe CSV file")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save the fitted model configuration")
    parser.add_argument("--exp_output_dir", type=str, required=True,
                        help="Directory to save the generated experiment configurations")

    args = parser.parse_args()
    generate(args)
