import pytest
import torch

from simulation.impl.xhat_transition_function_default import XhatTransitionFunctionDefault
from simulation.types import *


class TestXhatTransitionFunctionDefault:
    @pytest.fixture
    def transition_function(self):
        return XhatTransitionFunctionDefault()

    def test_output(self, transition_function):
        n_partitions = 4
        n_tasks = 5
        n_nodes = 1000
        n_clients = n_nodes - 1

        for _ in range(10_000):
            O = torch.randint(0, 2, (n_clients,), dtype=torch.int8)

            random_values = torch.randint(
                0, 3, (n_partitions, n_tasks, n_nodes), dtype=torch.int8)

            X = torch.full((n_partitions, n_tasks, n_nodes),
                           XState.EMPTY, dtype=torch.int8)
            X[random_values == 0] = XState.EMPTY
            X[random_values == 1] = XState.DOWNLOADING
            X[random_values == 2] = XState.READY

            server_actions = torch.full_like(
                X, fill_value=ServerXAction.NONE, dtype=torch.int8)
            client_actions = torch.full_like(
                X, fill_value=ClientXAction.NONE, dtype=torch.int8)

            mask = X == XState.EMPTY
            shape = (mask.sum(),)
            server_actions[mask] = torch.where(
                torch.randint(low=0, high=2, size=shape) == 0,
                torch.full(shape, ServerXAction.NONE, dtype=torch.int8),
                torch.full(shape, ServerXAction.DOWNLOAD, dtype=torch.int8),
            )

            mask = (X == XState.DOWNLOADING) | (X == XState.READY)
            shape = (mask.sum(),)
            server_actions[mask] = torch.where(
                torch.randint(low=0, high=2, size=shape) == 0,
                torch.full(shape, ServerXAction.NONE, dtype=torch.int8),
                torch.full(shape, ServerXAction.CANCEL, dtype=torch.int8),
            )
            client_actions[mask] = torch.where(
                torch.randint(low=0, high=2, size=shape) == 0,
                torch.full(shape, ClientXAction.NONE, dtype=torch.int8),
                torch.full(shape, ClientXAction.ABORT, dtype=torch.int8),
            )

            result = transition_function(X, O, server_actions, client_actions)
            sample = result.sample()

            online_mask = torch.ones_like(X, dtype=torch.bool)
            online_mask[:, :, 1:] = (O == 1).view(1, 1, -1)

            assert torch.all(sample[~online_mask] == X[~online_mask])

            assert torch.all(sample[online_mask & (
                client_actions == ClientXAction.ABORT)] == XState.EMPTY)

            assert torch.all(sample[online_mask & (
                server_actions == ServerXAction.CANCEL)] == XState.EMPTY)

            assert torch.all(sample[online_mask & (server_actions == ServerXAction.DOWNLOAD) & (
                client_actions != ClientXAction.ABORT)] == XState.DOWNLOADING)

            mask = online_mask & (server_actions == ServerXAction.NONE) & (
                client_actions == ClientXAction.NONE)
            assert torch.all(sample[mask] == X[mask])
