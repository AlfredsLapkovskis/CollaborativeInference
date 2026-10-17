import pytest
import torch
import pyro.distributions as dist
from itertools import product

from simulation.model_context import ModelContext
from simulation.impl.server_policy_smart import ServerPolicySmart
from simulation.types import *
from utils.unique import *
from utils.resource_consumption_model import ResourceConsumptionModel


class TestServerPolicySmart:
    # bandwidth in, cpu, memory, storage
    @pytest.fixture(
        params=[
            [
                ResourceConsumptionModel.load(
                    "resources/models/consumption_x_download_res_net_read_model.json"),
                ResourceConsumptionModel.load(
                    "resources/models/consumption_x_download_res_cpu_model.json"),
                ResourceConsumptionModel.load(
                    "resources/models/consumption_x_download_res_memory_model.json"),
                ResourceConsumptionModel.load(
                    "resources/models/consumption_x_download_res_storage_model.json"),
            ],
        ],
    )
    def x_resource_models(self, request):
        return request.param

    # bandwidth in, cpu, memory
    @pytest.fixture(
        params=[
            [
                ResourceConsumptionModel.load(
                    "resources/models/consumption_y_download_res_net_read_model.json"),
                ResourceConsumptionModel.load(
                    "resources/models/consumption_y_download_res_cpu_model.json"),
                ResourceConsumptionModel.load(
                    "resources/models/consumption_y_download_res_memory_model.json"),
            ],
        ],
    )
    def y_downloading_resource_models(self, request):
        return request.param

    # cpu, memory
    @pytest.fixture(
        params=[
            [
                ResourceConsumptionModel.load(
                    "resources/models/consumption_y_execution_res_cpu_model.json"),
                ResourceConsumptionModel.load(
                    "resources/models/consumption_y_execution_res_memory_model.json"),
            ],
        ],
    )
    def y_executing_resource_models(self, request):
        return request.param

    # bandwidth out, cpu, memory
    @pytest.fixture(
        params=[
            [
                ResourceConsumptionModel.load(
                    "resources/models/consumption_y_upload_res_net_write_model.json"),
                ResourceConsumptionModel.load(
                    "resources/models/consumption_y_upload_res_cpu_model.json"),
                ResourceConsumptionModel.load(
                    "resources/models/consumption_y_upload_res_memory_model.json"),
            ],
        ],
    )
    def y_uploading_resource_models(self, request):
        return request.param

    @pytest.fixture(
        params=[
            ModelContext(
                n_clients=1000,
                n_task_exec_slots=8,
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
    def model_context(self, request, x_resource_models, y_downloading_resource_models, y_executing_resource_models, y_uploading_resource_models):
        return ModelContext(
            n_clients=request.param.n_clients,
            n_task_exec_slots=request.param.n_task_exec_slots,
            n_timesteps=request.param.n_timesteps,
            partition_sizes=request.param.partition_sizes,
            partition_models=request.param.partition_models,
            task_query_probabilities=request.param.task_query_probabilities,
            x_resource_models=x_resource_models,
            y_downloading_resource_models=y_downloading_resource_models,
            y_executing_resource_models=y_executing_resource_models,
            y_uploading_resource_models=y_uploading_resource_models,
        )

    @pytest.fixture(
        params=[1., 10.],
    )
    def server_policy(self, request, model_context):
        return ServerPolicySmart(
            model_context,
            server_priority_scale=request.param,
            # Hard to test with randomization
            randomize=False,
        )

    def test_action_y(self, model_context: ModelContext, server_policy: ServerPolicySmart):
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
                            (Ys[partition, task, :, :] == YState.IDLE) & online_mask[0], as_tuple=False
                        )
                        assert idle_indices.numel() > 0
                        # Pick one random (node, slot) pair
                        chosen = idle_indices[torch.randint(
                            0, idle_indices.shape[0], (1,)).item()]
                        n, s = chosen.tolist()
                        Ys[partition, task, n, s] = YState.DONE
                        Yi[partition, task, n, s] = idx + 1
                        completed_mask[partition, task, n, s] = True

            # End of setting some queries finished

            W = torch.zeros(model_context.n_clients, dtype=torch.int32)

            A = torch.zeros((len(Resource.available_resources()),
                            model_context.n_nodes), dtype=resource_dtype)
            A[Resource.X_resources()] += Cx.sum(dim=(1, 2))
            A[Resource.Y_resources()] += Cy.sum(dim=(1, 2, 4))

            # 1) Test that only READY partitions can be executed and that offline clients cannot execute anything.

            _, _, ys_actions_dist = server_policy(
                O, To, Q, Tq, A, X, Tx, Cx, Yi, Ys, Ty, Cy, W)
            ys_actions = ys_actions_dist.sample()

            assert torch.all(
                X[(ys_actions == ServerYAction.EXECUTE).any(dim=3)] == XState.READY)
            assert torch.all(
                online_mask[(ys_actions == ServerYAction.EXECUTE).any(dim=(0, 1, 3))])

            mem_free_initial = A[Resource.MEMORY] - Cx[Resource.X_resource_idx(Resource.MEMORY)].sum(
                dim=(0, 1)) - Cy[Resource.Y_resource_idx(Resource.MEMORY)].sum(dim=(0, 1, 3))
            self._check_y_memory(
                model_context, ys_actions, mem_free_initial)

            # 2) Test the rest (hard to test with X with different values, because there is no guarantee that all pending requests will be scheduled)

            _, yi_actions_dist, ys_actions_dist = server_policy(
                O, To, Q, Tq, A, torch.full_like(X, fill_value=XState.READY), torch.zeros_like(Tx), Cx, Yi, Ys, Ty, Cy, W)
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

            self._check_y_memory(
                model_context, ys_actions, mem_free_initial)

            required_by_batch = self._calc_y_exec_required_memory_by_model_batch(
                model_context)

            for partition in range(model_context.n_partitions):
                for task in range(model_context.n_tasks):
                    if not model_context.partition_mask[partition, task]:
                        continue

                    model_idx = int(
                        model_context.partition_model_indices[partition, task].item())
                    req = torch.cummax(required_by_batch[model_idx].to(
                        mem_free_initial.dtype), dim=0).values

                    node_priority = ((Ys != YState.IDLE) & (Ys != YState.DONE) & (
                        ys_actions != ServerYAction.EXECUTE)).sum(dim=(0, 1, 3)).to(torch.float32)
                    node_priority[0] *= 1 / \
                        max(server_policy.server_priority_scale, 1e-12)
                    nodes = torch.argsort(node_priority, stable=True)

                    idle_slots = (Ys[partition, task, :, :]
                                  == YState.IDLE).sum(dim=1)
                    online_node_mask = torch.cat(
                        [torch.ones(1, dtype=torch.bool), online_mask[1:]])
                    max_by_mem = torch.searchsorted(
                        req, mem_free_initial.clamp(min=0.0), right=True) - 1
                    max_by_mem = torch.clamp(max_by_mem, min=0)
                    node_capacity = torch.minimum(idle_slots, max_by_mem)
                    node_capacity[~online_node_mask] = 0

                    assigned = (ys_actions[partition, task, :, :]
                                == ServerYAction.EXECUTE).sum(dim=1)

                    # 1) No node is assigned more than its capacity.
                    assert torch.all(assigned <= node_capacity)

                    # 2) Priority order: if a node got fewer assignments than its capacity,
                    #    all higher-priority nodes must be at capacity.
                    for rank, node in enumerate(nodes):
                        node = int(node.item())
                        if int(assigned[node].item()) < int(node_capacity[node].item()):
                            higher_priority_nodes = nodes[:rank]
                            assert torch.all(
                                assigned[higher_priority_nodes] == node_capacity[higher_priority_nodes]
                            ), f"Node {node} under-assigned but higher-priority node has spare capacity"
                            break  # once we find the first under-assigned node, all subsequent are irrelevant

            for task in range(model_context.n_tasks):
                query_ids = (Q == (task + 1)).nonzero() + 1

                yi = yi_actions.clone()
                yi_compl = yi_actions.clone()
                yi_compl[:, task, :, :] = 0

                for partition in range(model_context.n_partitions):
                    if model_context.partition_mask[partition, task]:
                        unique, counts = yi[partition, task,
                                            :, :].unique(return_counts=True)
                        counts = counts[unique > 0]
                        unique = unique[unique > 0]

                        required_by_batch = self._calc_y_exec_required_memory_by_model_batch(
                            model_context)
                        model_idx = int(
                            model_context.partition_model_indices[partition, task].item())
                        if required_by_batch is not None:
                            delta_req = required_by_batch[model_idx,
                                                          1] - required_by_batch[model_idx, 0]
                        else:
                            delta_req = self._calc_required_y_memory(
                                model_context)
                        has_idle_slot = (
                            Ys[partition, task, :, :] == YState.IDLE).any(dim=1)
                        node_online_mask = torch.cat(
                            [torch.ones(1, dtype=torch.bool), online_mask[1:]])
                        has_eligible_node = (has_idle_slot & node_online_mask & (
                            mem_free_initial.clamp(min=0.0) >= delta_req)).any()

                        assert torch.all(torch.isin(query_ids, unique) | torch.isin(
                            query_ids, finished_idx + 1) | ~has_eligible_node)
                        assert torch.all(counts == 1)

                assert torch.all(~torch.isin(query_ids, yi_compl))

            no_query_ids = (Q == 0).nonzero() + 1

            # This assertion is wrong here, because we may have different tasks Ys from Q and in this case they must also be removed.
            assert torch.all(~torch.isin(no_query_ids, yi_actions))

    def _check_y_memory(self, model_context: ModelContext, ys_actions: torch.Tensor, mem_free_initial: torch.Tensor):
        required_by_batch = self._calc_y_exec_required_memory_by_model_batch(
            model_context)
        assert required_by_batch is not None

        n_nodes = mem_free_initial.shape[0]
        total_deducted = torch.zeros(n_nodes, dtype=mem_free_initial.dtype)

        for p in range(model_context.n_partitions):
            for t in range(model_context.n_tasks):
                if not model_context.partition_mask[p, t]:
                    continue
                model_idx = int(
                    model_context.partition_model_indices[p, t].item())
                req = torch.cummax(required_by_batch[model_idx].to(
                    mem_free_initial.dtype), dim=0).values
                assigned_counts = (
                    ys_actions[p, t] == ServerYAction.EXECUTE).sum(dim=1)
                for n in range(n_nodes):
                    count = int(assigned_counts[n].item())
                    if count > 0:
                        total_deducted[n] += req[min(count,
                                                     req.shape[0] - 1)].item()

        assert bool(torch.all(total_deducted <=
                    mem_free_initial.clamp(min=0.0)))

    def _calc_required_y_memory(self, model_context: ModelContext) -> float:
        k = 2.0
        try:
            y_mem_down = model_context.y_downloading_resource_models[
                Resource.Y_resource_idx(
                    Resource.MEMORY, state=YState.DOWNLOADING)
            ]
            y_mem_exec = model_context.y_executing_resource_models[
                Resource.Y_resource_idx(
                    Resource.MEMORY, state=YState.EXECUTING)
            ]
            y_mem_up = model_context.y_uploading_resource_models[
                Resource.Y_resource_idx(
                    Resource.MEMORY, state=YState.UPLOADING)
            ]
        except Exception:
            return 0.0
        reqs = []
        for m in (y_mem_down, y_mem_exec, y_mem_up):
            means = list(getattr(m, "original_means", {}).values())
            stds = list(getattr(m, "original_stds", {}).values())
            mu = float(sum(means) / max(len(means), 1))
            sigma = float(sum(stds) / max(len(stds), 1))
            reqs.append(mu + k * sigma)
        return max(0.0, float(max(reqs)))

    def _calc_y_exec_required_memory_by_model_batch(self, model_context: ModelContext):
        k = 2.0
        try:
            y_mem_exec = model_context.y_executing_resource_models[
                Resource.Y_resource_idx(
                    Resource.MEMORY, state=YState.EXECUTING)
            ]
        except Exception:
            return None

        keys = list(getattr(y_mem_exec, "keys", []) or [])
        if not keys:
            return None

        model_names = list(model_context.model_list)
        if not model_names:
            return None

        task_counts = sorted({int(tc) for (tc, _) in keys})
        if not task_counts:
            return None

        dtype = model_context.float_type
        counts_t = torch.as_tensor(task_counts, dtype=torch.int64)
        n_models = len(model_names)
        n_counts = len(task_counts)

        mu_table = torch.zeros((n_models, n_counts), dtype=dtype)
        sigma_table = torch.zeros_like(mu_table)

        model_idx_map = {name: i for i, name in enumerate(model_names)}
        count_idx_map = {c: i for i, c in enumerate(task_counts)}

        for (tc, model_name), mu in getattr(y_mem_exec, "original_means", {}).items():
            mi = model_idx_map.get(model_name)
            ci = count_idx_map.get(int(tc))
            if mi is not None and ci is not None:
                mu_table[mi, ci] = float(mu)

        for (tc, model_name), sigma in getattr(y_mem_exec, "original_stds", {}).items():
            mi = model_idx_map.get(model_name)
            ci = count_idx_map.get(int(tc))
            if mi is not None and ci is not None:
                sigma_table[mi, ci] = float(sigma)

        max_batch = max(int(model_context.n_task_exec_slots),
                        int(max(task_counts)))
        required = torch.zeros((n_models, max_batch + 1), dtype=dtype)

        for b in range(1, max_batch + 1):
            dist = (counts_t - int(b)).abs()
            exact = dist == 0
            if bool(exact.any()):
                weights = exact.to(dtype)
            else:
                weights = 1.0 / dist.to(dtype)
            wsum = weights.sum()
            if float(wsum) <= 0.0:
                closest_idx = int(dist.argmin().item())
                mu = mu_table[:, closest_idx]
                sigma = sigma_table[:, closest_idx]
            else:
                mu = (mu_table * weights[None, :]).sum(dim=1) / wsum
                sigma = (sigma_table * weights[None, :]).sum(dim=1) / wsum
            required[:, b] = torch.clamp(mu + k * sigma, min=0.0)

        return required

    def _generate_y_batches(self, model_context: ModelContext):
        n_partitions = model_context.n_partitions
        n_tasks = model_context.n_tasks
        n_nodes = model_context.n_nodes
        n_exec_slots = model_context.n_task_exec_slots

        batches = []

        for (p, t, n) in product(
            range(n_partitions), range(n_tasks), range(n_nodes),
        ):
            if model_context.partition_mask[p, t] and torch.rand(1).item() > 0.5:
                temp_batches = [[] for _ in range(n_exec_slots)]

                for s in range(n_exec_slots):
                    if torch.rand(1).item() > 0.5:
                        idx = torch.randint(
                            low=0, high=n_exec_slots, size=(1,)).item()
                        temp_batches[idx].append((p, t, n, s))

                for batch in temp_batches:
                    if batch:
                        batches.append(batch)

        return batches
