import torch
import pyro.distributions as dist

from simulation.model_context import ModelContext
from simulation.transition_functions import ATransitionFunction
from utils.vector_arima_simulator import *
from utils.available_resources_model import AvailableResourcesModel


class ATransitionFunctionDefault(ATransitionFunction):
    def __init__(
        self,
        model_context: ModelContext,
        resource_models: list[AvailableResourcesModel],
        server_means: list[float | None] = None,
        server_stds: list[float | None] = None,
        server_maxs: list[float | None] = None,
        client_means: list[float | None] = None,
        client_stds: list[float | None] = None,
        client_maxs: list[float | None] = None,
    ):
        n_resources = len(resource_models)
        client_means = client_means or [None] * n_resources
        client_stds = client_stds or [None] * n_resources
        client_maxs = client_maxs or [float('inf')] * n_resources
        server_means = server_means or [None] * n_resources
        server_stds = server_stds or [None] * n_resources
        server_maxs = server_maxs or [float('inf')] * n_resources

        assert n_resources > 0
        assert len(client_means) == n_resources
        assert len(client_stds) == n_resources
        assert len(server_means) == n_resources
        assert len(server_stds) == n_resources
        assert len(client_maxs) == n_resources
        assert len(server_maxs) == n_resources

        client_means = [mean or model.original_mean for mean,
                        model in zip(client_means, resource_models)]
        client_stds = [std or model.original_std for std,
                       model in zip(client_stds, resource_models)]
        client_maxs = [max_value or float('inf') for max_value in client_maxs]
        server_means = [mean or model.original_mean for mean,
                        model in zip(server_means, resource_models)]
        server_stds = [std or model.original_std for std,
                       model in zip(server_stds, resource_models)]
        server_maxs = [max_value or float('inf') for max_value in server_maxs]

        self._model_context = model_context
        self._models = resource_models
        self._simulators = [
            VectorArimaSimulator(
                params=model.model_param_dict, n_series=model_context.n_nodes)
            for model in resource_models
        ]

        resource_means = [torch.repeat_interleave(torch.as_tensor([client_means[i]]), model_context.n_nodes)
                          for i in range(n_resources)]
        resource_stds = [torch.repeat_interleave(torch.as_tensor([client_stds[i]]), model_context.n_nodes)
                         for i in range(n_resources)]
        resource_maxs = [torch.repeat_interleave(torch.as_tensor([client_maxs[i]]), model_context.n_nodes)
                         for i in range(n_resources)]

        for i in range(n_resources):
            resource_means[i][0] = server_means[i]
            resource_stds[i][0] = server_stds[i]
            resource_maxs[i][0] = server_maxs[i]

        self._configs = [
            SimulatorConfig(
                n_series=model_context.n_nodes,
                min=torch.repeat_interleave(
                    torch.tensor([0.]), model_context.n_nodes),
                mean=resource_means[i],
                std=resource_stds[i],
                max=resource_maxs[i],
                power_transform_lambda=model.power_transform_lambda,
            )
            for i, model in enumerate(resource_models)
        ]

    def __call__(self, A_prev, O_curr):
        # Server is always online
        online_mask = torch.full((self._model_context.n_nodes,), True)
        online_mask[1:] &= O_curr == 1

        A_curr = torch.stack([
            simulator.simulate(
                n_simulations=1, mask=online_mask, config=config).flatten()
            for config, simulator in zip(self._configs, self._simulators)
        ])

        return dist.Delta(A_curr)
