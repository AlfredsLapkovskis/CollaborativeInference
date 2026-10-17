import json
import argparse

from utils.available_resources_model import *


def generate(args):
    n_trials = args.n_trials
    bandwidth_in_model_path = args.bandwidth_in_model_path
    output_path = args.output_path
    max_bandwidths = args.max_bandwidths
    assert max_bandwidths

    available_resources_model = AvailableResourcesModel.load(
        bandwidth_in_model_path)

    scale_factors = [
        bandwidth / available_resources_model.original_max for bandwidth in max_bandwidths]
    means = [available_resources_model.original_mean *
             scale for scale in scale_factors]
    stds = [available_resources_model.original_std *
            scale for scale in scale_factors]

    net_in_limits = [
        [x for x in (available_resources_model.z_discrete_levels *
                     std + mean).astype(int).tolist() if x > 0]
        for mean, std in zip(means, stds)
    ]
    net_in_limits = [x for sublist in net_in_limits for x in sublist]

    experiment_config = {
        "trials": n_trials,
        "net_in_limits": net_in_limits,
    }

    with open(output_path, "w") as f:
        json.dump(experiment_config, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_trials", type=int, required=True,
                        help="Number of trials in experiment")
    parser.add_argument("--bandwidth_in_model_path", type=str, required=True,
                        help="Path to available bandwidth_in model")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to save the generated download experiment config")
    parser.add_argument("--max_bandwidths", type=int, nargs="+", required=False,
                        help="Maximum bandwidths")

    args = parser.parse_args()
    generate(args)
