import pytest
import torch

from simulation.impl.yshat_transition_function_default import YshatTransitionFunctionDefault
from simulation.types import *


class TestYshatTransitionFunctionDefault:
    @pytest.fixture
    def transition_function(self):
        return YshatTransitionFunctionDefault()

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

            downloading_uploading_mask = (
                Ys == YState.DOWNLOADING) | (Ys == YState.UPLOADING)
            Ys[:, :, 0, :][downloading_uploading_mask[:, :, 0, :]
                           ] = YState.EXECUTING

            y_prev_online_mask = torch.ones_like(Ys, dtype=torch.bool)
            y_prev_online_mask[:, :, 1:] = (O_prev == 1).view(1, 1, -1, 1)

            # Fix states for offline clients
            Ys[(Ys != YState.IDLE) & (Ys != YState.DONE)
                & (~y_prev_online_mask)] = YState.IDLE

            Yi = torch.randint(low=1, high=n_clients + 1, size=(n_partitions, n_tasks, n_nodes, n_client_slots),
                               dtype=torch.int32)
            Yi[Ys == YState.IDLE] = 0

            server_actions = torch.where(
                (Ys == YState.IDLE) & y_prev_online_mask & torch.randint(
                    low=0, high=2, size=Ys.shape, dtype=torch.bool),
                torch.full_like(
                    Ys, fill_value=ServerYAction.EXECUTE, dtype=torch.int8),
                torch.full_like(
                    Ys, fill_value=ServerYAction.NONE, dtype=torch.int8),
            )

            finish_or_cancel = torch.randint(
                low=0, high=2, size=Ys.shape, dtype=torch.bool)
            finish_or_cancel = torch.where(
                finish_or_cancel,
                torch.full_like(
                    finish_or_cancel, fill_value=ServerYAction.FINISH, dtype=torch.int8),
                torch.full_like(
                    finish_or_cancel, fill_value=ServerYAction.CANCEL, dtype=torch.int8),
            )

            server_actions = torch.where(
                (Ys == YState.DONE) & torch.randint(
                    low=0, high=2, size=Ys.shape, dtype=torch.bool),
                finish_or_cancel,
                server_actions,
            )

            server_actions = torch.where(
                (Ys != YState.IDLE) & (Ys != YState.DONE) & y_prev_online_mask & torch.randint(
                    low=0, high=2, size=Ys.shape, dtype=torch.bool),
                torch.full_like(
                    Ys, fill_value=ServerYAction.CANCEL, dtype=torch.int8),
                server_actions,
            )

            client_actions = torch.where(
                (Ys != YState.IDLE) & (Ys != YState.DONE) & y_prev_online_mask & torch.randint(
                    low=0, high=2, size=Ys.shape, dtype=torch.bool),
                torch.full_like(
                    Ys, fill_value=ClientYAction.ABORT, dtype=torch.int8),
                torch.full_like(
                    Ys, fill_value=ClientYAction.NONE, dtype=torch.int8),
            )

            O_curr = torch.randint(low=0, high=2, size=(
                n_clients,), dtype=torch.int8)
            y_curr_online_mask = torch.ones_like(Ys, dtype=torch.bool)
            y_curr_online_mask[:, :, 1:] = (O_curr == 1).view(1, 1, -1, 1)

            result = transition_function(
                Ys, Yi, O_curr, server_actions, client_actions)
            sample = result.sample()

            assert torch.all(sample[:, :, 0, :] != YState.DOWNLOADING)
            assert torch.all(sample[:, :, 0, :] != YState.UPLOADING)

            assert torch.all((sample[:, :, 0, :][Ys[:, :, 0, :] == YState.IDLE] == YState.IDLE) |
                             (sample[:, :, 0, :][Ys[:, :, 0, :] == YState.IDLE] == YState.EXECUTING))
            assert torch.all((sample[:, :, 1:, :][Ys[:, :, 1:, :] == YState.IDLE] == YState.IDLE) |
                             (sample[:, :, 1:, :][Ys[:, :, 1:, :] == YState.IDLE] == YState.DOWNLOADING))

            downloading_sample = sample[Ys == YState.DOWNLOADING]
            assert torch.all((downloading_sample == YState.DOWNLOADING) | (
                downloading_sample == YState.IDLE) | (downloading_sample == YState.EXECUTING))

            # executing_sample = sample[Ys == YState.EXECUTING]
            assert torch.all((sample[:, :, 0, :][Ys[:, :, 0, :] == YState.EXECUTING] == YState.EXECUTING) | (
                sample[:, :, 0, :][Ys[:, :, 0, :] == YState.EXECUTING] == YState.IDLE) | (sample[:, :, 0, :][Ys[:, :, 0, :] == YState.EXECUTING] == YState.DONE))

            assert torch.all((sample[:, :, 1:, :][Ys[:, :, 1:, :] == YState.EXECUTING] == YState.EXECUTING) | (
                sample[:, :, 1:, :][Ys[:, :, 1:, :] == YState.EXECUTING] == YState.IDLE) | (sample[:, :, 1:, :][Ys[:, :, 1:, :] == YState.EXECUTING] == YState.UPLOADING))

            uploading_sample = sample[Ys == YState.UPLOADING]
            assert torch.all((uploading_sample == YState.UPLOADING) | (
                uploading_sample == YState.IDLE) | (uploading_sample == YState.DONE))

            idle_by_offline_mask = ~y_curr_online_mask & (Ys != YState.DONE)
            assert torch.all(sample[idle_by_offline_mask] == YState.IDLE)

            client_abort_mask = y_curr_online_mask & (
                client_actions == ClientYAction.ABORT)
            assert torch.all(sample[client_abort_mask] == YState.IDLE)

            server_cancel_mask = y_curr_online_mask & (
                server_actions == ServerYAction.CANCEL)
            assert torch.all(sample[server_cancel_mask] == YState.IDLE)

            server_done_mask = (Ys == YState.DONE) & ((
                server_actions == ServerYAction.FINISH) | (server_actions == ServerYAction.CANCEL))
            assert torch.all(sample[server_done_mask] == YState.IDLE)

            server_execute_mask = y_curr_online_mask & (
                server_actions == ServerYAction.EXECUTE) & ~client_abort_mask
            assert torch.all(
                sample[:, :, 0, :][server_execute_mask[:, :, 0, :]] == YState.EXECUTING)
            assert torch.all(
                sample[:, :, 1:, :][server_execute_mask[:, :, 1:, :]] == YState.DOWNLOADING)

            done_remained_mask = ~y_curr_online_mask & (
                Ys == YState.DONE) & (server_actions != ServerYAction.FINISH) & (server_actions != ServerYAction.CANCEL)

            assert torch.all(sample[done_remained_mask] == YState.DONE)
