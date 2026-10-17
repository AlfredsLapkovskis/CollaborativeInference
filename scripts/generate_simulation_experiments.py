import os
import json
import argparse
from itertools import product


def generate(args):
    model_dir = args.model_dir
    output_dir = args.output_dir

    n_clients = [
        100,
        1000,
        5_000,
        10_000,
    ]
    n_task_exec_slots = 100
    n_timesteps = 100_000
    # Easy-Many, Hard-Many, Easy-Few, Hard-Few
    partition_models = [
        ["vit-b-32", "vit-h-14", "vit-b-32", "vit-h-14"],
        ["googlenet", "vit-l-16", "googlenet", "vit-l-16"],
        ["vit-l-32", "resnet152", "densenet169", "resnet152"],
        ["densenet169", "vgg19", "vit-b-32", "vit-l-16"],
        ["vgg13", "densenet201", "googlenet", "resnet152"],
        ["vgg16", "vit-h-14"],
        ["vit-b-16", "vit-l-16"],
        ["vit-b-32", "resnet152"],
        ["googlenet", "vgg19"],
        ["vit-l-32", "densenet201"],
    ]
    model_to_size = {
        'resnet152': 241745227,
        'densenet169': 57663892,
        'vit-b-32': 352978285,
        'vit-h-14': 2534087677,
        'vgg13': 532207637,
        'googlenet': 26708107,
        'vit-b-16': 347519341,
        'vit-l-32': 1226297693,
        'vgg16': 553449335,
        'vgg19': 574691033,
        'vit-l-16': 1220854109,
        'densenet201': 81485780
    }
    partition_sizes = [
        [model_to_size[model] for model in partition]
        for partition in partition_models
    ]
    task_query_probabilities = [0.3, 0.1, 0.4, 0.2]
    available_resource_model_paths = {
        "bandwidth_in": os.path.join(model_dir, "available_bandwidth_in_model.json"),
        "bandwidth_out": os.path.join(model_dir, "available_bandwidth_out_model.json"),
        "cpu": os.path.join(model_dir, "available_cpu_model.json"),
        "memory": os.path.join(model_dir, "available_memory_model.json"),
        "storage": os.path.join(model_dir, "available_storage_model.json"),
    }
    x_resource_consumption_model_paths = {
        "bandwidth_in": os.path.join(model_dir, "consumption_x_download_res_net_read_model.json"),
        "cpu": os.path.join(model_dir, "consumption_x_download_res_cpu_model.json"),
        "memory": os.path.join(model_dir, "consumption_x_download_res_memory_model.json"),
        "storage": os.path.join(model_dir, "consumption_x_download_res_storage_model.json"),
    }
    y_download_resource_consumption_model_paths = {
        "bandwidth_in": os.path.join(model_dir, "consumption_y_download_res_net_read_model.json"),
        "cpu": os.path.join(model_dir, "consumption_y_download_res_cpu_model.json"),
        "memory": os.path.join(model_dir, "consumption_y_download_res_memory_model.json"),
    }
    y_execution_resource_consumption_model_paths = {
        "cpu": os.path.join(model_dir, "consumption_y_execution_res_cpu_model.json"),
        "memory": os.path.join(model_dir, "consumption_y_execution_res_memory_model.json"),
    }
    y_upload_resource_consumption_model_paths = {
        "bandwidth_out": os.path.join(model_dir, "consumption_y_upload_res_net_write_model.json"),
        "cpu": os.path.join(model_dir, "consumption_y_upload_res_cpu_model.json"),
        "memory": os.path.join(model_dir, "consumption_y_upload_res_memory_model.json"),
    }
    y_task_completion_model_paths = {
        "download": os.path.join(model_dir, "task_completion_download_model.json"),
        "execution": os.path.join(model_dir, "task_completion_execution_model.json"),
        "upload": os.path.join(model_dir, "task_completion_upload_model.json"),
    }

    def get_client_resources(multiplier):
        return {
            "bandwidth_in": {
                "mean": 12301969 * multiplier,
                "std": 3429124,
                "max": 12500000 * multiplier
            },
            "bandwidth_out": {
                "mean": 12175458 * multiplier,
                "std": 3712870,
                "max": 12500000 * multiplier
            },
            # In these resources, we preserve original variance
            "cpu": {
                "mean": 953.4450033333333 * multiplier,
                "std": 78.8521822691509,
                "max": 200.0 * multiplier
            },
            "memory": {
                "mean": 3618423248 * multiplier,
                "std": 468580277,
                "max": 2684354560 * multiplier
            },
            "storage": {
                "mean": 275039403427 * multiplier,
                "std": 2574260485,
                "max": 3221225472 * multiplier
            }
        }

    def get_server_resources(multiplier):
        # Because these resources are dedicated, we will assume that mean is equal to max
        return {
            "bandwidth_in": {
                "mean": 12500000 * multiplier,
                "std": 3429124,
                "max": 12500000 * multiplier
            },
            "bandwidth_out": {
                "mean": 12500000 * multiplier,
                "std": 3712870,
                "max": 12500000 * multiplier
            },
            "cpu": {
                "mean": 200.0 * multiplier,
                "std": 78.8521822691509,
                "max": 200.0 * multiplier
            },
            "memory": {
                "mean": 2684354560 * multiplier,
                "std": 468580277,
                "max": 2684354560 * multiplier
            },
            "storage": {
                "mean": 2750394034270,
                "std":  0,
                "max": 2750394034270
            }
        }

    client_resource_multiplier = [
        1,
        2,
    ]
    server_resource_multiplier = [
        10,
        30,
        50,
        80,
        100,
    ]

    policies = ["centralized", "default", "smart", "smartx"]

    trajectory = "metrics"
    device = ["cuda", "cpu"]
    double_precision = False

    configs = []

    for n_clients_val, client_multiplier_val, server_multiplier_val, policy, device_val in product(n_clients, client_resource_multiplier, server_resource_multiplier, policies, device):
        smartx = policy == "smartx"
        if smartx:
            policy = "smart"

        experiment_config = {
            "n_clients": n_clients_val,
            "n_task_exec_slots": n_task_exec_slots,
            "n_timesteps": n_timesteps,
            "partition_models": partition_models,
            "partition_sizes": partition_sizes,
            "task_query_probabilities": task_query_probabilities,
            "available_resource_model_paths": available_resource_model_paths,
            "x_resource_consumption_model_paths": x_resource_consumption_model_paths,
            "y_download_resource_consumption_model_paths": y_download_resource_consumption_model_paths,
            "y_execution_resource_consumption_model_paths": y_execution_resource_consumption_model_paths,
            "y_upload_resource_consumption_model_paths": y_upload_resource_consumption_model_paths,
            "y_task_completion_model_paths": y_task_completion_model_paths,
            "available_resource_params": {
                "server": get_server_resources(server_multiplier_val),
                "client": get_client_resources(client_multiplier_val),
            },
            "server_policy": policy,
            "trajectory": trajectory,
            "device": device_val,
            "double_precision": double_precision,
        }

        if smartx:
            experiment_config["server_policy_smart_priority_scale"] = server_multiplier_val

        configs.append(experiment_config)

    cuda_configs = [config for config in configs if config["device"] == "cuda"]
    cpu_configs = [config for config in configs if config["device"] == "cpu"]

    for device_type, configs in zip(["cuda", "cpu"], [cuda_configs, cpu_configs]):
        for idx, config in enumerate(configs):
            with open(os.path.join(output_dir, f"exp_simulation_{device_type}_{idx}.json"), "w") as f:
                json.dump(config, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Path to directory containing the resource consumption models")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Path to save generated experiments")
    args = parser.parse_args()

    generate(args)
