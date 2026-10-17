import pytest
import torch

from simulation.impl.yi_transition_function_default import YiTransitionFunctionDefault
from simulation.types import *


class TestYiTransitionFunctionDefault:
    @pytest.fixture
    def transition_function(self):
        return YiTransitionFunctionDefault()

    def test_output(self, transition_function):
        n_partitions = 4
        n_tasks = 5
        n_nodes = 1000
        n_client_slots = 10
        n_clients = n_nodes - 1

        for _ in range(10_00):
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

            Ty = torch.randint(low=1, high=1_000,
                               size=shape, dtype=torch.int32)
            Ty[(Yshat == YState.IDLE) | (Yshat == YState.DONE)] = 0
            Ty[torch.randint(low=0, high=2, size=shape, dtype=torch.bool)] = 0

            result = transition_function(Yihat, Yshat, Ty)
            sample = result.sample()

            assert torch.all(sample == Yihat)
