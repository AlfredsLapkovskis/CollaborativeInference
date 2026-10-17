import pytest
import torch
import pyro.distributions as dist
from itertools import product

from simulation.model_context import ModelContext
from simulation.impl.server_policy_centralized import ServerPolicyCentralized
from simulation.types import *
from utils.unique import *


class TestServerPolicyCentralized:
    @pytest.fixture(
        params=[
            ModelContext(
                n_clients=1000,
                n_task_exec_slots=100,
                n_timesteps=1000,
                partition_sizes=torch.tensor([
                    [241745227, 57663892, 352978285, 347519341],
                    [574691033, 553449335, 81485780, 241745227],
                    [1226297693, 1220854109, 0, 0],
                ]),
                partition_models=[
                    ["resnet152", "densenet169", "vit-b-32", "vit-b-16"],
                    ["vgg19", "vgg16", "densenet201", "resnet152"],
                    ["vit-l-32", "vit-l-16"],
                ],
                task_query_probabilities=torch.tensor([0.5, 0.15, 0.25, 0.1]),
            ),
        ],
    )
    def model_context(self, request):
        return request.param

    @pytest.fixture
    def server_policy(self, model_context):
        return ServerPolicyCentralized(model_context)

    def test_action_y(self, model_context: ModelContext, server_policy: ServerPolicyCentralized):
        resource_dtype = torch.float64

        for i in range(100):
            O = torch.randint(low=0, high=2, size=(
                model_context.n_clients,), dtype=torch.int8)

            online_mask = torch.ones(
                (model_context.n_nodes,), dtype=torch.bool)
            online_mask[1:] = O == 1

            To = torch.where(
                O == 0,
                torch.zeros(model_context.n_clients, dtype=torch.int32),
                torch.randint(low=1, high=1000, size=(
                    model_context.n_clients,), dtype=torch.int32),
            )
            Q = torch.zeros((model_context.n_clients,), dtype=torch.int32)
            # Q = torch.randint(low=0, high=model_context.n_tasks + 1,
            #   size=(model_context.n_clients,), dtype=torch.int8)
            Tq = torch.where(
                Q == 0,
                torch.zeros(model_context.n_clients, dtype=torch.int32),
                torch.randint(low=1, high=1000, size=(
                    model_context.n_clients,), dtype=torch.int32),
            )

            x_shape = (model_context.n_partitions,
                       model_context.n_tasks, model_context.n_nodes)
            cx_shape = (len(Resource.X_resources()), *x_shape)
            random_values = torch.randint(
                low=0, high=3, size=x_shape, dtype=torch.int8)
            downloading_allowed_mask = torch.randint(low=0, high=2, size=(
                model_context.n_nodes,), dtype=torch.bool)

            X = torch.zeros(x_shape, dtype=torch.int8)
            X[random_values == 0] = XState.EMPTY
            # Make sure some nodes don't download anything, so that policy can actually select something to download.
            X[(random_values == 1) & downloading_allowed_mask[None, None, :]
              ] = XState.DOWNLOADING
            X[random_values == 2] = XState.READY
            # server is always ready
            X[:, :, 0] = XState.READY
            X[~model_context.partition_mask] = XState.EMPTY

            Tx = torch.where(
                (X == XState.EMPTY) | (X == XState.READY),
                torch.zeros(x_shape, dtype=torch.int32),
                torch.randint(low=1, high=1000, size=x_shape,
                              dtype=torch.int32),
            )

            Cx = torch.where(
                (X == XState.EMPTY) | (X == XState.READY),
                torch.zeros(cx_shape, dtype=resource_dtype),
                torch.rand(cx_shape, dtype=resource_dtype) * 100. + 10.,
            )

            storage_idx = Resource.X_resource_idx(Resource.STORAGE)
            storage = model_context.partition_sizes.unsqueeze(
                2).expand_as(X).to(resource_dtype)

            Cx[storage_idx, X == XState.READY] = storage[X == XState.READY]
            Cx[storage_idx, X == XState.DOWNLOADING] = (
                storage[X == XState.DOWNLOADING] * torch.rand_like(storage[X == XState.DOWNLOADING]))

            y_shape = (model_context.n_partitions, model_context.n_tasks,
                       model_context.n_nodes, model_context.n_task_exec_slots)
            cy_shape = (len(Resource.Y_resources()), *y_shape)
            random_values = torch.randint(
                low=0, high=5, size=y_shape, dtype=torch.int8)

            batches = self._generate_y_batches(model_context)
            Ty = torch.zeros(y_shape, dtype=torch.int32)
            Yi = torch.zeros(y_shape, dtype=torch.int32)
            Ys = torch.full(
                y_shape, fill_value=YState.IDLE, dtype=torch.int8)
            Cy = torch.zeros(cy_shape, dtype=resource_dtype)

            random_values = torch.randint_like(Ys, low=0, high=4)
            Ys[random_values == 3] = YState.DONE
            # Yi[Ys == YState.DONE] = torch.randint_like(
            #     Yi[Ys == YState.DONE], low=1, high=model_context.n_clients + 1)

            for idx, batch in enumerate(batches):
                t_value = (idx % model_context.n_task_exec_slots)  # + 1
                status = {
                    0: YState.EXECUTING,
                    1: YState.DOWNLOADING,
                    2: YState.UPLOADING,
                    3: YState.DONE,
                }[torch.randint(low=0, high=4, size=(1,)).item()]
                c_value = torch.rand(
                    len(Resource.Y_resources()), dtype=resource_dtype) * 100. + 10.
                for indices in batch:
                    client = torch.randint(
                        low=1, high=model_context.n_clients + 1, size=(1,)).item()
                    if Q[client - 1] == 0 and torch.rand(1).item() > 0.95:
                        task = indices[1] + 1

                        Ty[*indices] = t_value
                        Cy[:, *indices] = c_value
                        Yi[*indices] = client
                        Ys[*indices] = status

                        Q[client -
                            1] = task

            Q[Q > 0] = torch.randint(0, 2, size=(
                (Q > 0).sum(),), dtype=torch.int8) * Q[Q > 0]

            # Setting some queries finished

            finished_idx = torch.take(
                Q.nonzero(), torch.randperm(Q[Q > 0].shape[0])[:5])

            Ys[torch.isin(Yi, finished_idx)] = YState.IDLE
            Yi[torch.isin(Yi, finished_idx)] = 0
            completed_mask = torch.zeros_like(Ys, dtype=torch.bool)

            for idx in finished_idx:
                task = Q[idx] - 1
                # Find Ys which are IDLE and node is online, set partitions for task 'task' to DONE and Yi to idx
                # idle_mask = (Ys[:, task, :, :] ==
                #              YState.IDLE) & online_mask[None, :, None]
                # Ys[:, task, :, :][idle_mask] = YState.DONE
                # For each partition, set only one (random) instance (across all nodes and slots) to DONE and Yi=idx
                for partition in range(model_context.n_partitions):
                    if model_context.partition_mask[partition, task]:
                        # Find all idle slots for this partition and task where node is online
                        idle_indices = torch.nonzero(
                            Ys[partition, task, 0, :] == YState.IDLE, as_tuple=False
                        )
                        assert idle_indices.numel() > 0
                        # Pick one random (node, slot) pair
                        chosen = idle_indices[torch.randint(
                            0, idle_indices.shape[0], (1,)).item()]
                        s = chosen.tolist()
                        Ys[partition, task, 0, s] = YState.DONE
                        Yi[partition, task, 0, s] = idx.to(torch.int32) + 1
                        completed_mask[partition, task, 0, s] = True

            # End of setting some queries finished

            W = torch.zeros(model_context.n_clients, dtype=torch.int32)

            A = torch.zeros((len(Resource.available_resources()),
                            model_context.n_nodes), dtype=resource_dtype)
            A[Resource.X_resources()] += Cx.sum(dim=(1, 2))
            A[Resource.Y_resources()] += Cy.sum(dim=(1, 2, 4))

            _, yi_actions_dist, ys_actions_dist = server_policy(
                O, To, Q, Tq, A, X, Tx, Cx, Yi, Ys, Ty, Cy, W)
            yi_actions = yi_actions_dist.sample()
            ys_actions = ys_actions_dist.sample()

            ys_actions_0 = ys_actions[yi_actions == 0]
            assert torch.all((ys_actions_0 == ServerYAction.NONE) | (
                ys_actions_0 == ServerYAction.FINISH) | (ys_actions_0 == ServerYAction.CANCEL))
            # assert torch.all()

            ys_actions_1 = ys_actions[yi_actions != 0]
            assert torch.all((ys_actions_1 == ServerYAction.NONE) | (
                ys_actions_1 == ServerYAction.EXECUTE))

            assert torch.all(
                ys_actions[completed_mask] == ServerYAction.FINISH)
            assert torch.all(yi_actions[completed_mask] == 0)

            assert torch.all(yi_actions[:, :, 1:, :] == 0)
            assert torch.all(ys_actions[:, :, 1:, :] == ServerYAction.NONE)

            for task in range(model_context.n_tasks):
                query_ids = (Q == (task + 1)).nonzero() + 1

                yi = yi_actions.clone()
                yi_compl = yi_actions.clone()
                yi_compl[:, task, :, :] = 0

                slot_count = model_context.n_task_exec_slots

                for partition in range(model_context.n_partitions):
                    if model_context.partition_mask[partition, task]:
                        unique, counts = yi[partition, task,
                                            0, :].unique(return_counts=True)
                        counts = counts[unique > 0]
                        unique = unique[unique > 0]

                        # Finished tasks have Yi == 0
                        assert torch.sum(torch.isin(query_ids, unique)) == min(
                            query_ids.numel(), slot_count) - torch.sum(torch.isin(query_ids, finished_idx + 1))
                        assert torch.all(counts == 1)

                assert torch.all(~torch.isin(query_ids, yi_compl))

            no_query_ids = (Q == 0).nonzero() + 1

            # This assertion is wrong here, because we may have different tasks Ys from Q and in this case they must also be removed.
            assert torch.all(~torch.isin(no_query_ids, yi_actions))

    def _generate_y_batches(self, model_context: ModelContext):
        n_partitions = model_context.n_partitions
        n_tasks = model_context.n_tasks
        n_exec_slots = model_context.n_task_exec_slots

        batches = []

        for (p, t) in product(
            range(n_partitions), range(n_tasks),
        ):
            if model_context.partition_mask[p, t] and torch.rand(1).item() > 0.5:
                temp_batches = [[] for _ in range(n_exec_slots)]

                for s in range(n_exec_slots):
                    if torch.rand(1).item() > 0.5:
                        idx = torch.randint(
                            low=0, high=n_exec_slots, size=(1,)).item()
                        temp_batches[idx].append((p, t, 0, s))

                for batch in temp_batches:
                    if batch:
                        batches.append(batch)

        return batches
