import os
import json
import argparse
from itertools import product


def generate(args):
    df_path_download = args.df_path_y_download
    df_path_execution = args.df_path_y_execution
    df_path_upload = args.df_path_y_upload
    output_dir = args.output_dir
    exp_output_dir = args.exp_output_dir

    standardize = [True]
    polynomial_degree = [1, 2, 3]

    for task_type, df_path in zip(["download", "execution", "upload"], [df_path_download, df_path_execution, df_path_upload]):
        for i, (standardize_val, polynomial_degree_val) in enumerate(product(standardize, polynomial_degree)):
            experiment_config = {
                "df_path": df_path,
                "task_type": task_type,
                "output_dir": output_dir,
                "standardize": standardize_val,
                "polynomial_degree": polynomial_degree_val,
            }

            path = os.path.join(
                exp_output_dir, f"exp_fit_task_completion_{task_type}_{i}.json")

            with open(path, "w") as f:
                json.dump(experiment_config, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
