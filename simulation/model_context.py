import torch

from utils.resource_consumption_model import ResourceConsumptionModel


class ModelContext:

    @property
    def n_clients(self):
        return self._n_clients

    @property
    def n_nodes(self):
        return self._n_nodes

    @property
    def n_timesteps(self):
        return self._n_timesteps

    @property
    def n_tasks(self):
        return self._n_tasks

    @property
    def n_partitions(self):
        return self._n_partitions

    @property
    def n_task_exec_slots(self):
        return self._n_task_exec_slots

    @property
    def partition_sizes(self):
        return self._partition_sizes

    @property
    def partition_mask(self):
        return self._partition_mask

    @property
    def partition_models(self):
        return self._partition_models

    @property
    def task_query_probabilities(self):
        return self._task_query_probabilities

    @property
    def float_type(self):
        return self._float_type

    @property
    def device(self):
        return self._device

    @property
    def model_list(self):
        return self._model_list

    @property
    def partition_model_indices(self):
        return self._partition_model_indices

    @property
    def x_resource_models(self):
        return self._x_resource_models

    @property
    def y_downloading_resource_models(self):
        return self._y_downloading_resource_models

    @property
    def y_executing_resource_models(self):
        return self._y_executing_resource_models

    @property
    def y_uploading_resource_models(self):
        return self._y_uploading_resource_models

    def __init__(
        self,
        n_clients: int,
        n_task_exec_slots: int,
        n_timesteps: int,
        partition_sizes: torch.Tensor,
        partition_models: list[list[str]],
        task_query_probabilities: torch.Tensor,
        x_resource_models: list[ResourceConsumptionModel] | None = None,
        y_downloading_resource_models: list[ResourceConsumptionModel] | None = None,
        y_executing_resource_models: list[ResourceConsumptionModel] | None = None,
        y_uploading_resource_models: list[ResourceConsumptionModel] | None = None,
        device: torch.device | None = None,
        double_precision: bool = False,
    ):
        assert len(partition_sizes.shape) == 2  # shape [partition, task_type]

        device = device or torch.device("cpu")
        x_resource_models = x_resource_models or []
        y_downloading_resource_models = y_downloading_resource_models or []
        y_executing_resource_models = y_executing_resource_models or []
        y_uploading_resource_models = y_uploading_resource_models or []

        self._n_clients = n_clients
        self._n_nodes = n_clients + 1  # including server node
        self._n_task_exec_slots = n_task_exec_slots
        self._n_timesteps = n_timesteps
        self._n_tasks = partition_sizes.shape[1]
        self._n_partitions = partition_sizes.shape[0]
        self._device = device
        self._float_type = torch.float64 if double_precision else torch.float32
        self._partition_sizes = torch.as_tensor(
            partition_sizes, dtype=self._float_type, device=device)
        # Tasks may have different number of partitions, hence mask is useful
        self._partition_mask = partition_sizes > 0

        self._partition_models = partition_models
        self._task_query_probabilities = task_query_probabilities
        self._x_resource_models = x_resource_models
        self._y_downloading_resource_models = y_downloading_resource_models
        self._y_executing_resource_models = y_executing_resource_models
        self._y_uploading_resource_models = y_uploading_resource_models

        self._model_list = []
        partition_model_indices = []

        for partition_idx in range(self._n_partitions):
            for task_idx in range(self._n_tasks):
                if self._partition_mask[partition_idx, task_idx]:
                    model = partition_models[partition_idx][task_idx]
                    if model not in self._model_list:
                        self._model_list.append(model)
                    model_idx = self._model_list.index(model)
                    partition_model_indices.append(model_idx)
                else:
                    partition_model_indices.append(-1)

        self._partition_model_indices = torch.tensor(
            partition_model_indices, dtype=torch.int32, device=device).reshape(self._n_partitions, self._n_tasks)
