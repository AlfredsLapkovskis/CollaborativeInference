import pytest
import torch

from simulation.impl.yihat_transition_function_default import YihatTransitionFunctionDefault
from simulation.types import *


class TestYihatTransitionFunctionDefault:
    @pytest.fixture
    def transition_function(self):
        return YihatTransitionFunctionDefault()

    def test_output(self, transition_function):
        n_partitions = 4
        n_tasks = 5
        n_nodes = 1000
        n_client_slots = 10
        n_clients = n_nodes - 1

        for _ in range(10_000):
            O_prev = torch.randint(0, 2, (n_clients,), dtype=torch.int8)

            random_values = torch.randint(
                0, 5, (n_partitions, n_tasks, n_nodes, n_client_slots), dtype=torch.int8)

            Ys = torch.full((n_partitions, n_tasks, n_nodes, n_client_slots),
                            YState.IDLE, dtype=torch.int8)
            Ys[random_values == 0] = YState.IDLE
            Ys[random_values == 1] = YState.DOWNLOADING
            Ys[random_values == 2] = YState.EXECUTING
            Ys[random_values == 3] = YState.UPLOADING
            Ys[random_values == 4] = YState.DONE

            y_prev_online_mask = torch.ones_like(Ys, dtype=torch.bool)
            y_prev_online_mask[:, :, 1:] = (O_prev == 1).view(1, 1, -1, 1)

            # Fix states for offline clients
            Ys[(Ys != YState.IDLE) & (Ys != YState.DONE)
                & (~y_prev_online_mask)] = YState.IDLE

            Yi = torch.randint(low=1, high=n_clients + 1, size=(n_partitions, n_tasks, n_nodes, n_client_slots),
                               dtype=torch.int32)
            Yi[Ys == YState.IDLE] = 0

            server_actions = torch.where(
                (Yi == 0) & y_prev_online_mask & torch.randint(
                    low=0, high=2, size=Yi.shape, dtype=torch.bool),
                torch.randint(low=1, high=n_clients + 1,
                              size=Yi.shape, dtype=torch.int32),
                Yi,
            )
            server_actions = torch.where(
                (Yi != 0) & (y_prev_online_mask | (Ys == YState.DONE)) & torch.randint(
                    low=0, high=2, size=Yi.shape, dtype=torch.bool),
                torch.zeros_like(Yi, dtype=torch.int32),
                server_actions,
            )

            client_actions = torch.where(
                (Yi != 0) & y_prev_online_mask & torch.randint(
                    low=0, high=2, size=Yi.shape, dtype=torch.bool),
                torch.zeros_like(Yi, dtype=torch.int32),
                Yi,
            )

            O_curr = torch.randint(low=0, high=2, size=(
                n_clients,), dtype=torch.int8)
            y_curr_online_mask = torch.ones_like(Ys, dtype=torch.bool)
            y_curr_online_mask[:, :, 1:] = (O_curr == 1).view(1, 1, -1, 1)

            result = transition_function(
                Yi, Ys, O_curr, server_actions, client_actions)
            sample = result.sample()

            assert torch.all(sample[Yi == 0] >= 0)

            assert torch.all((sample[Yi != 0] == 0) | (
                sample[Yi != 0] == Yi[Yi != 0]))

            zero_by_offline_mask = ~y_curr_online_mask & (Ys != YState.DONE)
            assert torch.all(sample[zero_by_offline_mask] == 0)

            client_abort_mask = (client_actions == 0) & (
                Yi != 0) & y_curr_online_mask
            assert torch.all(sample[client_abort_mask] == 0)

            server_abort_mask = (server_actions == 0) & y_curr_online_mask
            assert torch.all(sample[server_abort_mask] == 0)

            server_execute_mask = (
                server_actions != 0) & y_curr_online_mask & ~client_abort_mask
            assert torch.all(sample[server_execute_mask]
                             == server_actions[server_execute_mask])

            done_remained_mask = ~y_curr_online_mask & (
                Ys == YState.DONE) & (server_actions != 0)
            assert torch.all(sample[done_remained_mask]
                             == Yi[done_remained_mask])
