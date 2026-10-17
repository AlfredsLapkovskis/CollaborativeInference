import os
import json
import argparse
from itertools import product


from simulation.types import Resource


def generate(args):
    df_path = args.df_path
    output_dir = args.output_dir
    exp_output_dir = args.exp_output_dir

    series_names = [
        "bandwidth_in",
        "bandwidth_out",
        "cpu",
        "memory",
        "storage",
    ]

    p = [1, 2, 5, 10]
    d = [0]
    q = [0, 1, 2, 5, 10]

    for series_name in series_names:
        for i, (p_val, d_val, q_val) in enumerate(product(p, d, q)):
            experiment_config = {
                "df_path": df_path,
                "series_name": series_name,
                "arima_order": [p_val, d_val, q_val],
                "output_dir": output_dir,
            }

            path = os.path.join(
                exp_output_dir, f"exp_fit_avail_{series_name}_{i}.json")

            with open(path, "w") as f:
                json.dump(experiment_config, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--df_path", type=str, required=True,
                        help="Path to the input dataframe CSV file")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save the fitted model configuration")
    parser.add_argument("--exp_output_dir", type=str, required=True,
                        help="Directory to save the generated experiment configurations")

    args = parser.parse_args()
    generate(args)
