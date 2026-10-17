import pytest
import torch

from simulation.impl.ys_transition_function_default import YsTransitionFunctionDefault
from simulation.types import *


class TestYsTransitionFunctionDefault:
    @pytest.fixture
    def transition_function(self):
        return YsTransitionFunctionDefault()

    def test_output(self, transition_function):
        n_partitions = 4
        n_tasks = 5
        n_nodes = 1000
        n_client_slots = 10
        n_clients = n_nodes - 1

        for _ in range(10_000):
            shape = (n_partitions, n_tasks, n_nodes, n_client_slots)

            Yihat = torch.randint(low=1, high=n_clients + 1,
                                  size=shape, dtype=torch.int32)
            Yihat[torch.randint(low=0, high=2, size=shape,
                                dtype=torch.bool)] = 0

            random_values = torch.randint(
                0, 5, (n_partitions, n_tasks, n_nodes, n_client_slots), dtype=torch.int8)

            Yshat = torch.full((n_partitions, n_tasks, n_nodes, n_client_slots),
                               YState.IDLE, dtype=torch.int8)
            Yshat[random_values == 0] = YState.IDLE
            Yshat[random_values == 1] = YState.DOWNLOADING
            Yshat[random_values == 2] = YState.EXECUTING
            Yshat[random_values == 3] = YState.UPLOADING
            Yshat[random_values == 4] = YState.DONE

            Yshat[Yihat == 0] = YState.IDLE
            downloading_uploading_mask = (
                Yshat == YState.DOWNLOADING) | (Yshat == YState.UPLOADING)
            Yshat[:, :, 0, :][downloading_uploading_mask[:, :, 0, :]
                              ] = YState.EXECUTING

            Ty = torch.randint(low=1, high=1_000,
                               size=shape, dtype=torch.int32)
            Ty[(Yshat == YState.IDLE) | (Yshat == YState.DONE)] = 0
            Ty[torch.randint(low=0, high=2, size=shape, dtype=torch.bool)] = 0

            result = transition_function(Yshat, Yihat, Ty)
            sample = result.sample()

            assert torch.all(sample[Ty > 0] == Yshat[Ty > 0])

            idle_or_done = (Yshat == YState.IDLE) | (Yshat == YState.DONE)
            assert torch.all(sample[idle_or_done] == Yshat[idle_or_done])

            assert torch.all(sample[:, :, 0, :] != YState.DOWNLOADING)
            assert torch.all(sample[:, :, 0, :] != YState.UPLOADING)

            assert torch.all(
                sample[(Yshat == YState.DOWNLOADING) & (Ty == 0)] == YState.EXECUTING)
            assert torch.all(
                sample[:, :, 0, :][(Yshat[:, :, 0, :] == YState.EXECUTING) & (Ty[:, :, 0, :] == 0)] == YState.DONE)
            assert torch.all(
                sample[:, :, 1:, :][(Yshat[:, :, 1:, :] == YState.EXECUTING) & (Ty[:, :, 1:, :] == 0)] == YState.UPLOADING)
            assert torch.all(
                sample[(Yshat == YState.UPLOADING) & (Ty == 0)] == YState.DONE)
