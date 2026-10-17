import pytest
import torch

from simulation.model_context import ModelContext
from simulation.impl.tx_transition_function_default import TxTransitionFunctionDefault
from simulation.types import *


class TestTxTransitionFunctionDefault:
    @pytest.fixture(
        params=[
            ModelContext(
                n_clients=100,
                n_task_exec_slots=8,
                n_timesteps=1000,
                partition_sizes=torch.tensor([
                    [200, 100, 300, 200, 100],
                    [300, 200, 100, 200, 0],
                    [100, 200, 100, 0, 0],
                    [400, 100, 0, 0, 0]
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

    @pytest.fixture
    def transition_function(self, model_context):
        return TxTransitionFunctionDefault(model_context)

    def test_output(self, model_context: ModelContext, transition_function: TxTransitionFunctionDefault):
        n_partitions = model_context.n_partitions
        n_tasks = model_context.n_tasks
        n_nodes = model_context.n_nodes

        x_shape = (n_partitions, n_tasks, n_nodes)
        cx_shape = (len(Resource.X_resources()), *x_shape)

        for _ in range(10):
            random_values = torch.randint(
                low=0, high=3, size=x_shape, dtype=torch.int8)
            Xhat = torch.zeros(x_shape, dtype=torch.float32)
            Xhat[random_values == 0] = XState.EMPTY
            Xhat[random_values == 1] = XState.DOWNLOADING
            Xhat[random_values == 2] = XState.READY

            Xhat[~model_context.partition_mask] = XState.EMPTY

            Tx_prev = torch.randint(
                low=0, high=1000, size=x_shape, dtype=torch.int32)
            Tx_prev[(Xhat == XState.EMPTY) | (Xhat == XState.READY)] = 0

            Cx_prev = torch.zeros(cx_shape, dtype=torch.float32)

            storage_idx = Resource.X_resource_idx(Resource.STORAGE)
            Cx_prev[storage_idx] = torch.randint(
                low=0, high=100, size=x_shape, dtype=torch.float32)

            expected_storage = model_context.partition_sizes.unsqueeze(
                2).expand_as(Xhat)

            non_downloading_tested = False
            downloading_finished_tested = False
            downloading_in_progress_tested = False

            for _ in range(1000):
                result = transition_function(Tx_prev, Xhat, Cx_prev)
                sample = result.sample()

                non_downloading_sample = sample[Xhat != XState.DOWNLOADING]

                downloading_finished_sample = sample[(Xhat == XState.DOWNLOADING) & (
                    Cx_prev[storage_idx] >= expected_storage)]

                download_in_progress_mask = (Xhat == XState.DOWNLOADING) & (
                    Cx_prev[storage_idx] < expected_storage)
                downloading_in_progress_sample = sample[download_in_progress_mask]

                if non_downloading_sample.numel() > 0:
                    non_downloading_tested = True
                if downloading_finished_sample.numel() > 0:
                    downloading_finished_tested = True
                if downloading_in_progress_sample.numel() > 0:
                    downloading_in_progress_tested = True

                assert torch.all(non_downloading_sample == 0)
                assert torch.all(downloading_finished_sample == 0)
                assert torch.all(downloading_in_progress_sample ==
                                 Tx_prev[download_in_progress_mask] + 1)

                Tx_prev = sample
                Cx_prev[storage_idx] += torch.clamp_min(
                    torch.normal(mean=10., std=10., size=x_shape), min=0.)

            assert non_downloading_tested
            assert downloading_finished_tested
            assert downloading_in_progress_tested
