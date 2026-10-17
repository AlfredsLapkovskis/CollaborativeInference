import pytest
import torch

from simulation.impl.x_transition_function_default import XTransitionFunctionDefault
from simulation.types import *


class TestXTransitionFunctionDefault:
    @pytest.fixture
    def transition_function(self):
        return XTransitionFunctionDefault()

    def test_output(self, transition_function):
        n_partitions = 4
        n_tasks = 5
        n_nodes = 1000

        for _ in range(10_000):
            shape = (n_partitions, n_tasks, n_nodes)

            random_values = torch.randint(
                low=0, high=3, size=shape, dtype=torch.int8)

            Xhat = torch.full(shape, XState.EMPTY, dtype=torch.int8)
            Xhat[random_values == 0] = XState.EMPTY
            Xhat[random_values == 1] = XState.DOWNLOADING
            Xhat[random_values == 2] = XState.READY

            Tx = torch.randint(low=1, high=1_000,
                               size=shape, dtype=torch.int32)
            Tx[(Xhat == XState.EMPTY) | (Xhat == XState.READY)] = 0
            Tx[torch.randint(low=0, high=2, size=shape, dtype=torch.bool)] = 0

            result = transition_function(Xhat, Tx)
            sample = result.sample()

            assert torch.all(sample[Xhat == XState.EMPTY] == XState.EMPTY)
            assert torch.all(sample[Xhat == XState.READY] == XState.READY)
            assert torch.all(
                sample[(Xhat == XState.DOWNLOADING) & (Tx == 0)] == XState.READY)
            assert torch.all(
                sample[(Xhat == XState.DOWNLOADING) & (Tx > 0)] == XState.DOWNLOADING)
