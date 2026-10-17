import json
import argparse

from utils.available_resources_model import *


def generate(args):
    n_trials = args.n_trials
    bandwidth_in_model_path = args.bandwidth_in_model_path
    bandwidth_out_model_path = args.bandwidth_out_model_path
    output_path = args.output_path
    max_bandwidths_in = args.max_bandwidths_in
    max_bandwidths_out = args.max_bandwidths_out

    available_resources_model_in = AvailableResourcesModel.load(
        bandwidth_in_model_path)
    available_resources_model_out = AvailableResourcesModel.load(
        bandwidth_out_model_path)

    scale_factors_in = [
        bandwidth / available_resources_model_in.original_max for bandwidth in max_bandwidths_in]
    scale_factors_out = [
        bandwidth / available_resources_model_out.original_max for bandwidth in max_bandwidths_out]

    means_in = [available_resources_model_in.original_mean *
                scale for scale in scale_factors_in]
    stds_in = [available_resources_model_in.original_std *
               scale for scale in scale_factors_in]

    means_out = [available_resources_model_out.original_mean *
                 scale for scale in scale_factors_out]
    stds_out = [available_resources_model_out.original_std *
                scale for scale in scale_factors_out]

    net_in_limits = [
        [x for x in (available_resources_model_in.z_discrete_levels *
                     std + mean).astype(int).tolist() if x > 0]
        for mean, std in zip(means_in, stds_in)
    ]
    net_out_limits = [
        [x for x in (available_resources_model_out.z_discrete_levels *
                     std + mean).astype(int).tolist() if x > 0]
        for mean, std in zip(means_out, stds_out)
    ]
    net_in_limits = [x for sublist in net_in_limits for x in sublist]
    net_out_limits = [x for sublist in net_out_limits for x in sublist]

    experiment_config = {
        "trials": n_trials,
        "task_counts": [1, 2, 5, 10, 20, 50],
        "net_in_limits": net_in_limits,
        "net_out_limits": net_out_limits,
    }

    with open(output_path, "w") as f:
        json.dump(experiment_config, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_trials", type=int, required=True,
                        help="Number of trials in experiment")
    parser.add_argument("--bandwidth_in_model_path", type=str, required=True,
                        help="Path to available bandwidth_in model")
    parser.add_argument("--bandwidth_out_model_path", type=str, required=True,
                        help="Path to available bandwidth_out model")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to save the generated download experiment config")
    parser.add_argument("--max_bandwidths_in", type=int, nargs="+", required=True,
                        help="Maximum download bandwidths in bits per second")
    parser.add_argument("--max_bandwidths_out", type=int, nargs="+", required=True,
                        help="Maximum upload bandwidths in bits per second")

    args = parser.parse_args()
    generate(args)
