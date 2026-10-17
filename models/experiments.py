import os
import json
import torch
from enum import Enum

from config import Config
from simulation.model_context import ModelContext
from simulation.impl.client_policy_default import ClientPolicyDefault
from simulation.impl.server_policy_default import ServerPolicyDefault
from simulation.impl.server_policy_centralized import ServerPolicyCentralized
from simulation.impl.server_policy_smart import ServerPolicySmart
from simulation.impl.trajectory_default import TrajectoryDefault
from simulation.impl.trajectory_with_metrics import TrajectoryWithMetrics
from utils.available_resources_model import AvailableResourcesModel
from utils.resource_consumption_model import ResourceConsumptionModel
from utils.task_completion_model import TaskCompletionModel
from utils.torch_device import get_device


class Experiment:
    def __init__(self, handle, config: Config = None):
        config = config or Config()
        self.experiment_handle = handle
        self.experiment_path = os.path.join(
            config.experiment_dir, f"exp_{handle}.json")

        if not os.path.exists(self.experiment_path):
            raise ValueError(
                f"Experiment not found at path {self.experiment_path}.")

        with open(self.experiment_path, "rb") as f:
            self.json: dict = json.loads(f.read())

    def _get(self, key, default=None):
        return self.json.get(key, default)


class MeasurementExperiment(Experiment):
    def __init__(self, handle, config=None):
        super().__init__(handle, config)

        self.trials = self._get("trials", 1)

        self.memory_limits = self._get("memory_limits", [])
        self.cpu_limits = self._get("cpu_limits", [])
        self.storage_limits = self._get("storage_limits", [])
        self.net_in_limits = self._get("net_in_limits", [])  # bps
        self.net_out_limits = self._get("net_out_limits", [])  # bps
        self.disk_in_limits = self._get("disk_in_limits", [])
        self.disk_out_limits = self._get("disk_out_limits", [])
        self.task_counts = self._get("task_counts", [])


class AvailableResourceModelExperiment(Experiment):
    def __init__(self, handle, config=None):
        super().__init__(handle, config)

        self.df_path = self._get("df_path", "")
        self.series_name = self._get("series_name", "")
        self.arima_order = tuple(self._get("arima_order", (1, 0, 1)))
        self.output_dir = self._get("output_dir", "")


class ResourceConsumptionModelExperiment(Experiment):
    class TaskType(Enum):
        X = "x"
        Y = "y"

    class TaskState(Enum):
        DOWNLOAD = "download"
        EXECUTION = "execution"
        UPLOAD = "upload"

    def __init__(self, handle, config=None):
        super().__init__(handle, config)

        self.df_path = self._get("df_path", "")
        self.series_name = self._get("series_name", "")
        self.task_type = self.TaskType(self._get("task_type", ""))
        self.task_state = self.TaskState(self._get("task_state", ""))
        self.arima_order = tuple(self._get("arima_order", (1, 0, 1)))
        self.arima_trend = self._get("arima_trend", None)
        self.output_dir = self._get("output_dir", "")


class TaskCompletionModelExperiment(Experiment):
    class TaskType(Enum):
        DOWNLOAD = "download"
        EXECUTION = "execution"
        UPLOAD = "upload"

    def __init__(self, handle, config=None):
        super().__init__(handle, config)

        self.df_path = self._get("df_path", "")
        self.task_type = self.TaskType(self._get("task_type", ""))
        self.output_dir = self._get("output_dir", "")
        self.standardize = self._get("standardize", True)
        self.polynomial_degree = self._get("polynomial_degree", 1)
        # self.penalizer = self._get("penalizer", 0.0)
        # self.l1_ratio = self._get("l1_ratio", 0.0)


class SimulationExperiment(Experiment):
    def __init__(self, handle, config=None):
        super().__init__(handle, config)

        self.n_clients = self._get("n_clients", 1)
        self.n_task_exec_slots = self._get("n_task_exec_slots", 10)
        self.n_timesteps = self._get("n_timesteps", 100)

        self.partition_models = self._get("partition_models", [])
        self.partition_sizes = self._get("partition_sizes", [])
        assert [True
                for m, s in zip(self.partition_models, self.partition_sizes)
                if m and s and len(m) == len(s)
                ], "partition_models and partition_sizes lists and their sublists must have identical lengths. The lists must have at least one sublist and all sublists must not be empty."

        self.task_query_probabilities = self._get(
            "task_query_probabilities", [])
        assert sum(self.task_query_probabilities) == 1.

        self.device = get_device(self._get("device", "cpu"))

        n_tasks = len(self.partition_sizes)
        n_partitions = max(len(p) for p in self.partition_sizes)

        partition_sizes = torch.zeros(
            (n_tasks, n_partitions), device=self.device)
        for p, sizes in enumerate(self.partition_sizes):
            for t, size in enumerate(sizes):
                partition_sizes[p, t] = size

        assert len(self.task_query_probabilities) == partition_sizes.shape[1]

        self.available_resource_models: dict = self._get(
            "available_resource_model_paths", {})
        self.available_resource_models: list = [
            AvailableResourcesModel.load(
                self.available_resource_models[resource])
            for resource in ["bandwidth_in", "bandwidth_out", "cpu", "memory", "storage"]
        ]

        available_resource_names = ["bandwidth_in",
                                    "bandwidth_out", "cpu", "memory", "storage"]
        available_resource_params: dict = self._get(
            "available_resource_params", {})

        self.server_available_resource_means = [
            available_resource_params["server"][resource]["mean"]
            for resource in available_resource_names
        ]
        self.server_available_resource_stds = [
            available_resource_params["server"][resource]["std"]
            for resource in available_resource_names
        ]
        self.server_available_resource_maxs = [
            available_resource_params["server"][resource]["max"]
            for resource in available_resource_names
        ]

        self.client_available_resource_means = [
            available_resource_params["client"][resource]["mean"]
            for resource in available_resource_names
        ]
        self.client_available_resource_stds = [
            available_resource_params["client"][resource]["std"]
            for resource in available_resource_names
        ]
        self.client_available_resource_maxs = [
            available_resource_params["client"][resource]["max"]
            for resource in available_resource_names
        ]

        self.x_resource_consumption_models: dict = self._get(
            "x_resource_consumption_model_paths", {})
        self.x_resource_consumption_models: list = [
            ResourceConsumptionModel.load(
                self.x_resource_consumption_models[resource])
            for resource in ["bandwidth_in", "cpu", "memory", "storage"]
        ]

        self.y_download_resource_consumption_models: dict = self._get(
            "y_download_resource_consumption_model_paths", {})
        self.y_download_resource_consumption_models: list = [
            ResourceConsumptionModel.load(
                self.y_download_resource_consumption_models[resource])
            for resource in ["bandwidth_in", "cpu", "memory"]
        ]
        self.y_execution_resource_consumption_models: dict = self._get(
            "y_execution_resource_consumption_model_paths", {})
        self.y_execution_resource_consumption_models: list = [
            ResourceConsumptionModel.load(
                self.y_execution_resource_consumption_models[resource])
            for resource in ["cpu", "memory"]
        ]
        self.y_upload_resource_consumption_models: dict = self._get(
            "y_upload_resource_consumption_model_paths", {})
        self.y_upload_resource_consumption_models: list = [
            ResourceConsumptionModel.load(
                self.y_upload_resource_consumption_models[resource])
            for resource in ["bandwidth_out", "cpu", "memory"]
        ]

        self.y_task_completion_models: dict = self._get(
            "y_task_completion_model_paths", {})
        self.y_task_completion_models: list = [
            TaskCompletionModel.load(
                self.y_task_completion_models[type], device=self.device)
            for type in ["download", "execution", "upload"]
        ]

        self.model_context = ModelContext(
            n_clients=self.n_clients,
            n_task_exec_slots=self.n_task_exec_slots,
            n_timesteps=self.n_timesteps,
            partition_sizes=partition_sizes,
            partition_models=self.partition_models,
            task_query_probabilities=torch.as_tensor(
                self.task_query_probabilities, device=self.device),
            x_resource_models=self.x_resource_consumption_models,
            y_downloading_resource_models=self.y_download_resource_consumption_models,
            y_executing_resource_models=self.y_execution_resource_consumption_models,
            y_uploading_resource_models=self.y_upload_resource_consumption_models,
            device=self.device,
            double_precision=self._get("double_precision", False),
        )

        self.server_policy = self._get("server_policy", "default")
        self.server_policy_smart_priority_scale = self._get(
            "server_policy_smart_priority_scale", 1.)

        match self.server_policy:
            case "default":
                self.server_policy = ServerPolicyDefault(self.model_context)
            case "centralized":
                self.server_policy = ServerPolicyCentralized(
                    self.model_context)
            case "smart":
                self.server_policy = ServerPolicySmart(
                    self.model_context,
                    server_priority_scale=self.server_policy_smart_priority_scale,
                    randomize=True
                )
            case _:
                raise ValueError(
                    f"Unknown server policy {self.server_policy}.")

        self.client_policy = self._get("client_policy", "default")
        match self.client_policy:
            case "default":
                self.client_policy = ClientPolicyDefault()
            case _:
                raise ValueError(
                    f"Unknown client policy {self.client_policy}.")

        self.trajectory = self._get("trajectory", "default")
        match self.trajectory:
            case "default":
                self.trajectory = TrajectoryDefault()
            case "metrics":
                self.trajectory = TrajectoryWithMetrics(self.model_context)
            case _:
                raise ValueError(
                    f"Unknown trajectory {self.trajectory}.")
