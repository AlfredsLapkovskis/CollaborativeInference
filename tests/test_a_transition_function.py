import pytest
import torch

from simulation.model_context import ModelContext
from simulation.types import *
from simulation.impl.a_transition_function_default import ATransitionFunctionDefault
from utils.vector_plr_simulator import VectorPLRSimulator
from utils.available_resources_model import AvailableResourcesModel
from utils.unique import *


class TestATransitionFunctionDefault:
    @pytest.fixture(
        params=[
            ModelContext(
                n_clients=1000,
                n_task_exec_slots=8,
                n_timesteps=1000,
                partition_sizes=torch.tensor([
                    [2, 1, 3, 2, 1],
                    [3, 2, 1, 2, 0],
                    [1, 2, 1, 0, 0],
                    [4, 1, 0, 0, 0]
                ]),
                partition_models=[
                    ["a", "b", "c", "d", "e"],
                    ["e", "f", "g", "h"],
                    ["i", "j", "k"],
                    ["l", "m"],
                ],
                task_query_probabilities=torch.tensor(
                    [0.1, 0.5, 0.05, 0.15, 0.2]),
            ),
        ],
    )
    def model_context(self, request):
        return request.param

    @pytest.fixture(
        params=[
            [
                AvailableResourcesModel.load(
                    "resources/models/available_bandwidth_in_model.json"),
                AvailableResourcesModel.load(
                    "resources/models/available_bandwidth_out_model.json"),
                AvailableResourcesModel.load(
                    "resources/models/available_cpu_model.json"),
                AvailableResourcesModel.load(
                    "resources/models/available_memory_model.json"),
                AvailableResourcesModel.load(
                    "resources/models/available_storage_model.json"),
            ],
        ],
    )
    def resource_models(self, request):
        return request.param

    @pytest.fixture(
        params=[
            [924430038, 906977843, 9530.4450033333333, 3618423248, 1073741824000],
        ],
    )
    def server_means(self, request):
        return request.param

    @pytest.fixture(
        params=[
            [27432989, 29702961, 78.8521822691509, 468580277, 0.0],
        ],
    )
    def server_stds(self, request):
        return request.param

    @pytest.fixture(
        params=[
            [1000000000, 1000000000, 2000.0, 32212254720, 1073741824000],
        ],
    )
    def server_maxs(self, request):
        return request.param

    @pytest.fixture(
        params=[
            [98415753, 98415753, 953.4450033333333, 3618423248, 275039403427],
        ],
    )
    def client_means(self, request):
        return request.param

    @pytest.fixture(
        params=[
            [2920543, 3189909, 78.8521822691509, 468580277, 2574260485],
        ],
    )
    def client_stds(self, request):
        return request.param

    @pytest.fixture(
        params=[
            [100000000, 100000000, 200.0, 3221225472, 3221225472],
        ],
    )
    def client_maxs(self, request):
        return request.param

    @pytest.fixture
    def a_transition_function(self, model_context, resource_models, client_means, client_stds, client_maxs, server_means, server_stds, server_maxs):
        return ATransitionFunctionDefault(
            model_context,
            resource_models,
            server_means,
            server_stds,
            server_maxs,
            client_means,
            client_stds,
            client_maxs,
        )

    def test_online_offline(self, a_transition_function, model_context: ModelContext, resource_models: list[AvailableResourcesModel]):
        O = torch.where(
            torch.arange(model_context.n_clients) % 2 == 0,
            torch.tensor(0, dtype=torch.int8),
            torch.tensor(1, dtype=torch.int8),
        )
        n_resources = len(resource_models)
        A_prev = torch.zeros(
            (n_resources, model_context.n_nodes), dtype=torch.float32)

        for _ in range(1000):
            result = a_transition_function(A_prev, O)
            sample = result.sample()

            assert torch.all(sample[:, 0] >= 0.)
            assert torch.all(sample[:, 1:][:, O == 0] == 0.)
            assert torch.all(sample[:, 1:][:, O == 1] >= 0.)

    def test_output(
        self,
        a_transition_function,
        model_context: ModelContext,
        resource_models: list[AvailableResourcesModel],
        server_maxs,
        client_maxs,
    ):
        O = torch.ones(model_context.n_clients, dtype=torch.int8)
        n_resources = len(resource_models)
        A_prev = torch.zeros(
            (n_resources, model_context.n_nodes), dtype=torch.float32)

        server_maxs = torch.as_tensor(server_maxs)
        client_maxs = torch.as_tensor(client_maxs).unsqueeze(1)

        for _ in range(1000):
            result = a_transition_function(A_prev, O)
            sample = result.sample()

            assert torch.all(sample >= 0.)
            assert torch.all(sample[:, 0] <= server_maxs)
            assert torch.all(sample[:, 1:] <= client_maxs)
