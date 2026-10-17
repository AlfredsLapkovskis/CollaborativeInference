import pytest
import torch
import pyro.distributions as dist
from itertools import product

from simulation.model_context import ModelContext
from simulation.impl.client_policy_default import ClientPolicyDefault
from simulation.types import *
from utils.unique import *


class TestClientPolicyDefault:
    @pytest.fixture(
        params=[
            ModelContext(
                n_clients=1000,
                n_task_exec_slots=8,
                n_timesteps=1000,
                # partition_sizes=torch.tensor([
                #     [241745227, 57663892, 352978285, 347519341],
                #     [574691033, 553449335, 81485780, 241745227],
                #     [1226297693, 1220854109, 0, 0],
                # ]),
                partition_sizes=torch.tensor([
                    [24, 57, 35, 34],
                    [57, 55, 81, 24],
                    [12, 12, 0, 0],
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
    def client_policy(self):
        return ClientPolicyDefault()

    def test_outputs(self, model_context: ModelContext, client_policy: ClientPolicyDefault):
        # Easier to test with float64 due to precision issues with float32
        resource_dtype = torch.float64

        for i in range(100):
            print(f"Iteration {i}")
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
            Q = torch.randint(low=0, high=model_context.n_tasks + 1,
                              size=(model_context.n_clients,), dtype=torch.int8)
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
                low=0, high=4, size=x_shape, dtype=torch.int8)

            X = torch.zeros(x_shape, dtype=torch.int8)
            X[random_values == 0] = XState.EMPTY
            X[random_values == 1] = XState.DOWNLOADING
            X[random_values == 2] = XState.PAUSED
            X[random_values == 3] = XState.READY
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
                (X == XState.EMPTY) | (X == XState.PAUSED) | (X == XState.READY),
                torch.zeros(cx_shape, dtype=resource_dtype),
                torch.rand(cx_shape, dtype=resource_dtype) * 100. + 10.,
            )

            storage_idx = Resource.X_resource_idx(Resource.STORAGE)
            storage = model_context.partition_sizes.unsqueeze(
                2).expand_as(X).to(resource_dtype)

            Cx[storage_idx, X == XState.READY] = storage[X == XState.READY]
            Cx[storage_idx, X == XState.DOWNLOADING] = (
                storage[X == XState.DOWNLOADING] * torch.rand_like(storage[X == XState.DOWNLOADING]))
            Cx[storage_idx, X == XState.PAUSED] = (
                storage[X == XState.PAUSED] * torch.rand_like(storage[X == XState.PAUSED]))

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
            Yi[Ys == YState.DONE] = torch.randint_like(
                Yi[Ys == YState.DONE], low=1, high=model_context.n_clients + 1)

            for idx, batch in enumerate(batches):
                t_value = (idx % model_context.n_task_exec_slots)  # + 1
                status = {
                    0: YState.EXECUTING,
                    1: YState.DOWNLOADING,
                    2: YState.UPLOADING,
                }[torch.randint(low=0, high=3, size=(1,)).item()]
                c_value = torch.rand(
                    len(Resource.Y_resources()), dtype=resource_dtype) * 100. + 10.
                for indices in batch:
                    X[*indices[:-1]] = XState.READY
                    Cx[:, *indices[:-1]] = 0.
                    Cx[storage_idx, *indices[:-1]] = storage[*indices[:-1]]
                    Ty[*indices] = t_value
                    Cy[:, *indices] = c_value
                    Yi[*indices] = torch.randint(
                        low=1, high=model_context.n_clients + 1, size=(1,)).item()
                    Ys[*indices] = status

            y_active_node_mask = ((Ys == YState.DOWNLOADING) | (
                Ys == YState.EXECUTING) | (Ys == YState.UPLOADING)).any(dim=(0, 1, 3))

            W = torch.zeros(model_context.n_clients, dtype=torch.int32)

            A = torch.zeros((len(Resource.available_resources()),
                            model_context.n_nodes), dtype=resource_dtype)

            def compute_A(X_res, Y_res):
                a = torch.zeros_like(A)
                a[Resource.X_resources()] += X_res.sum(dim=(1, 2),
                                                       dtype=resource_dtype)
                a[Resource.Y_resources()] += Y_res.sum(dim=(1, 2, 4),
                                                       dtype=resource_dtype)
                return a

            def assert_no_overflow(A, Cx, Cy, x_actions, ys_actions):
                Cx = Cx.clone()
                Cy = Cy.clone()
                Cx[:, x_actions == ClientXAction.ABORT] = 0.

                # Paused X tasks may consume only storage
                x_non_storage_resources = Resource.X_resources()
                x_non_storage_resources.remove(Resource.STORAGE)
                paused_mask = x_actions == ClientXAction.PAUSE
                Cx_paused_non_storage = Cx[x_non_storage_resources]
                Cx_paused_non_storage[:, paused_mask] = 0.
                Cx[x_non_storage_resources] = Cx_paused_non_storage

                # Ignore server storage, because we never abort X tasks on the server
                Cx[Resource.X_resource_idx(Resource.STORAGE), :, :, 0] = 0.
                Cy[:, ys_actions == ClientYAction.ABORT] = 0.

                assert torch.all((A >= compute_A(Cx, Cy))[:, online_mask])

            def sample_actions():
                x_actions_dist, yi_actions_dist, ys_actions_dist = client_policy(
                    O, To, Q, Tq, A, X, Tx, Cx, Yi, Ys, Ty, Cy, W)

                return x_actions_dist.sample(), yi_actions_dist.sample(), ys_actions_dist.sample()

            tol = 1e-3

            # 1) Test doing nothing

            A = compute_A(Cx + tol, Cy + tol)

            x_actions, yi_actions, ys_actions = sample_actions()

            x_resume_mask = (
                X == XState.PAUSED) & ~y_active_node_mask & online_mask.expand_as(X)

            assert torch.all(
                x_actions[~x_resume_mask] == ClientXAction.NONE)
            assert torch.all(
                x_actions[x_resume_mask] == ClientXAction.RESUME)
            assert torch.all(yi_actions == Yi)
            assert torch.all(ys_actions == ClientYAction.NONE)
            assert_no_overflow(A, Cx, Cy, x_actions, ys_actions)

            # 2) Test pausing all ongoing X downloads on clients where non-storage resources are exceeded (and keeping all the rest)

            x_bandwidth_in = Resource.X_resource_idx(Resource.BANDWIDTH_IN)
            y_bandwidth_in = Resource.Y_resource_idx(Resource.BANDWIDTH_IN)

            X_res = Cx.clone() + tol
            X_res[x_bandwidth_in, (X == XState.DOWNLOADING) & torch.randint(
                0, 2, size=X.shape, dtype=torch.bool)] = 0.

            A = compute_A(X_res, Cy + tol)

            x_actions, yi_actions, ys_actions = sample_actions()

            bandwidth_overflow = Cx[x_bandwidth_in].sum(
                dim=(0, 1)) + Cy[y_bandwidth_in].sum(dim=(0, 1, 3)) > A[Resource.BANDWIDTH_IN]
            x_paused_mask = (X == XState.DOWNLOADING) & bandwidth_overflow.expand_as(
                X) & online_mask.expand_as(X)
            x_resume_mask = (
                X == XState.PAUSED) & ~y_active_node_mask & online_mask.expand_as(X)

            assert torch.all(x_actions[x_paused_mask] == ClientXAction.PAUSE)
            assert torch.all(x_actions[x_resume_mask] == ClientXAction.RESUME)
            assert torch.all(
                x_actions[~x_resume_mask & ~x_paused_mask] == ClientXAction.NONE)
            assert torch.all(yi_actions == Yi)
            assert torch.all(ys_actions == ClientYAction.NONE)
            assert_no_overflow(A, Cx, Cy, x_actions, ys_actions)

            # 3) Test removing sufficient X downloads to restore storage (but keeping everything else) and cancelling Y for cancelled X tasks

            storage_idx = Resource.X_resource_idx(Resource.STORAGE)

            X_res = Cx.clone() + tol
            X_res[storage_idx, ((X == XState.DOWNLOADING) | (X == XState.READY)) & torch.randint(
                0, 2, size=X.shape, dtype=torch.bool)] = 0.

            A = compute_A(X_res, Cy + tol)

            x_actions, yi_actions, ys_actions = sample_actions()

            Cx_storage = Cx[storage_idx].view(-1, model_context.n_nodes)
            sort_idx = Cx_storage.argsort(dim=0)
            unsort_idx = torch.argsort(sort_idx, dim=0)
            cumulative_storage = torch.cumsum(torch.gather(
                Cx_storage, dim=0, index=sort_idx), dim=0)
            storage_overflow = cumulative_storage > A[Resource.STORAGE]

            abort_mask = torch.gather(
                storage_overflow, dim=0, index=unsort_idx)
            abort_mask = abort_mask.view_as(X)
            abort_mask[:, :, 0] = False  # never abort on server
            abort_mask &= online_mask.expand_as(X)

            x_resume_mask = (
                X == XState.PAUSED) & ~abort_mask & ~y_active_node_mask & online_mask.expand_as(X)

            # Make sure that aborts actually work to restore storage (ignoring offline and server node)
            Cx_storage = Cx[storage_idx].clone()
            Cx_storage[x_actions == ClientXAction.ABORT] = 0.
            any_aborted = (x_actions == ClientXAction.ABORT).any(dim=(0, 1))
            assert torch.all(Cx_storage.sum(dim=(0, 1))[
                             any_aborted] <= A[Resource.STORAGE][any_aborted])

            assert torch.all(x_actions[abort_mask] == ClientXAction.ABORT)
            assert torch.all(x_actions[x_resume_mask] == ClientXAction.RESUME)
            assert torch.all(
                x_actions[~abort_mask & ~x_resume_mask] == ClientXAction.NONE)
            y_abort_mask = ((Ys == YState.DOWNLOADING) | (
                Ys == YState.EXECUTING)) & abort_mask[..., None].expand_as(yi_actions)
            assert torch.all(yi_actions[y_abort_mask] == 0)
            assert torch.all(yi_actions[~y_abort_mask] == Yi[~y_abort_mask])
            assert torch.all(ys_actions[y_abort_mask] == ClientYAction.ABORT)
            assert torch.all(ys_actions[~y_abort_mask] == ClientYAction.NONE)
            assert_no_overflow(A, Cx, Cy, x_actions, ys_actions)

            # 4) Test cancelling all Y downloads (after also cancelling X downloads), but retaining executions and uploads
            #   (i.e., make sure that downloads are the first to be cancelled)

            x_bandwidth_in = Resource.X_resource_idx(Resource.BANDWIDTH_IN)
            y_bandwidth_in = Resource.Y_resource_idx(Resource.BANDWIDTH_IN)

            X_res = Cx.clone() + tol
            X_res[x_bandwidth_in, (X == XState.DOWNLOADING) & torch.randint(
                0, 2, size=X.shape, dtype=torch.bool)] = 0.

            Y_res = Cy.clone() + tol
            Y_res[y_bandwidth_in, (Ys == YState.DOWNLOADING) & torch.randint(
                0, 2, size=Ys.shape, dtype=torch.bool)] = 0.

            A = compute_A(X_res, Y_res)

            x_actions, yi_actions, ys_actions = sample_actions()

            total_bandwidth_overflow = Cx[x_bandwidth_in].sum(
                dim=(0, 1)) + Cy[y_bandwidth_in].sum(dim=(0, 1, 3)) > A[Resource.BANDWIDTH_IN]
            y_bandwidth_overflow = Cy[y_bandwidth_in].sum(
                dim=(0, 1, 3)) > A[Resource.BANDWIDTH_IN]

            x_aborted_mask = (X == XState.DOWNLOADING) & total_bandwidth_overflow.expand_as(
                X) & online_mask.expand_as(X)
            x_resume_mask = (
                X == XState.PAUSED) & ~y_active_node_mask & online_mask.expand_as(X)
            y_aborted_mask = (Ys == YState.DOWNLOADING) & y_bandwidth_overflow[:, None].expand_as(
                Ys) & online_mask[:, None].expand_as(Ys)

            # everything should be available here actually
            # y_aborted_mask |= get_y_abort_mask_for_unavailable_x(
            #     X, x_actions, Ys)

            assert torch.all(x_actions[x_aborted_mask] == ClientXAction.PAUSE)
            assert torch.all(x_actions[x_resume_mask] == ClientXAction.RESUME)
            assert torch.all(
                x_actions[~x_aborted_mask & ~x_resume_mask] == ClientXAction.NONE)
            assert torch.all(yi_actions[y_aborted_mask] == 0)
            assert torch.all(
                yi_actions[~y_aborted_mask] == Yi[~y_aborted_mask])
            assert torch.all(ys_actions[y_aborted_mask] == ClientYAction.ABORT)
            assert torch.all(ys_actions[~y_aborted_mask] == ClientYAction.NONE)
            assert_no_overflow(A, Cx, Cy, x_actions, ys_actions)

            # 5) Test cancelling Y executions exceeding resources (after cancelling X and Y downloads), but retaining uploads

            x_cpu_index = Resource.X_resource_idx(Resource.CPU)
            y_cpu_index = Resource.Y_resource_idx(Resource.CPU)
            y_memory_index = Resource.Y_resource_idx(Resource.MEMORY)

            Cy_temp = Cy.clone()
            Cy[y_memory_index] = 0.

            X_res = Cx.clone() + tol
            # X_res[x_cpu_index, (X == XState.DOWNLOADING)
            #       & torch.randint(0, 2, size=X.shape, dtype=torch.bool)
            #       ] = 0.

            Y_res = Cy.clone() + tol
            Y_res[y_cpu_index, ((Ys == YState.EXECUTING) | (Ys == YState.DOWNLOADING))
                  & torch.randint(0, 2, size=Ys.shape, dtype=torch.bool)
                  ] = 0.

            A = compute_A(X_res, Y_res)

            total_cpu_overflow = Cx[x_cpu_index].sum(
                dim=(0, 1)) + Cy[y_cpu_index].sum(dim=(0, 1, 3)) > A[Resource.CPU]

            x_aborted_mask = (X == XState.DOWNLOADING) & total_cpu_overflow.expand_as(
                X) & online_mask.expand_as(X)
            x_resume_mask = (
                X == XState.PAUSED) & ~y_active_node_mask & online_mask.expand_as(X)

            x_actions, yi_actions, ys_actions = sample_actions()

            # y_aborted_for_unavailable_x_mask = get_y_abort_mask_for_unavailable_x(
            #     X, x_actions, Ys)

            Cx_modified = Cx.clone()
            Cx_modified[:, x_actions == ClientXAction.ABORT] = 0.
            x_non_storage_resources = Resource.X_resources()
            x_non_storage_resources.remove(Resource.STORAGE)
            Cx_paused_non_storage = Cx_modified[x_non_storage_resources]
            Cx_paused_non_storage[:, x_actions == ClientXAction.PAUSE] = 0.
            Cx_modified[x_non_storage_resources] = Cx_paused_non_storage

            Cy_modified = Cy.clone()
            # Cy_modified[:, y_aborted_for_unavailable_x_mask] = 0.
            resource_overflow_mask = torch.clip(
                compute_A(Cx_modified, Cy_modified) - A,
                min=0.0,
            ) > 0
            y_downloading_aborted_mask = (Ys == YState.DOWNLOADING) & \
                resource_overflow_mask[Resource.Y_resources(state=YState.DOWNLOADING)].any(dim=0)[
                    None, None, :, None] & online_mask[:, None].expand_as(Ys)
            Cy_modified[:, y_downloading_aborted_mask] = 0.

            # ---- Build "fixed" CPU usage (what policy subtracts from A before deciding exec aborts) ----
            # X fixed CPU after applying X actions
            fixed_cpu_x = Cx_modified[x_cpu_index].sum(dim=(0, 1))  # [node]

            # Y fixed CPU: all NON-executing Y that remains after
            # unavailable-X aborts and download aborts.
            Cy_cpu_fixed = Cy_modified[y_cpu_index].clone()
            Cy_cpu_fixed[Ys == YState.EXECUTING] = 0.0
            fixed_cpu_y = Cy_cpu_fixed.sum(dim=(0, 1, 3))  # [node]

            available_cpu = torch.clamp(
                A[Resource.CPU] - fixed_cpu_x - fixed_cpu_y, min=0.0)  # [node]

            # ---- Build EXECUTING CPU tensor exactly like policy's Cy_resources (CPU only here) ----
            exec_cpu = Cy_modified[y_cpu_index].clone()        # [p,t,n,s]
            exec_cpu[Ys != YState.EXECUTING] = 0.0

            # Apply the same Ty collapsing as the policy (counts + keep first occurrence)
            Ty_exec = Ty.clone()
            Ty_exec[Ys != YState.EXECUTING] = -1
            # [p*t*n, s]
            Ty2d = Ty_exec.view(-1, model_context.n_task_exec_slots)

            Ty_unique, Ty_counts, Ty_inverse, Ty_first_occ_mask = unique_per_row(
                Ty2d)

            counts_per_pos_2d = Ty_counts.gather(
                1, Ty_inverse)       # [rows, s]
            counts_per_pos = counts_per_pos_2d.view_as(
                Ty_exec)       # [p,t,n,s]
            first_occ = Ty_first_occ_mask.view_as(
                Ty_exec)            # [p,t,n,s]

            exec_cpu = exec_cpu * counts_per_pos.to(exec_cpu.dtype)
            exec_cpu = torch.where(first_occ, exec_cpu,
                                   torch.zeros_like(exec_cpu))

            # ---- (1) Sort by normalized "consumption level" (policy-style) ----
            denom = torch.clamp(A[Resource.CPU], min=1e-12)           # [node]
            level = exec_cpu / denom[None, None,
                                     :, None]             # [p,t,n,s]

            # policy flattens over (p,t,slot) per node: transpose node<->slot then reshape(-1, n_nodes)
            level2d = level.transpose(
                2, 3).reshape(-1, model_context.n_nodes)  # [items, node]
            cons2d = exec_cpu.transpose(
                2, 3).reshape(-1, model_context.n_nodes)

            sort_idx = torch.argsort(level2d, dim=0, descending=False)
            unsort_idx = torch.argsort(sort_idx, dim=0)

            cumsums = torch.cumsum(torch.gather(
                cons2d, dim=0, index=sort_idx), dim=0)

            # ---- (2) Compare against available_cpu (A - fixed), not A ----
            cpu_overflow = cumsums > available_cpu[None, :]

            abort2d = torch.gather(cpu_overflow, dim=0, index=unsort_idx)
            abort = abort2d.reshape(level.transpose(
                2, 3).shape).transpose(2, 3)  # back to [p,t,n,s]

            y_executing_aborted_mask = abort
            y_executing_aborted_mask &= (Ys == YState.EXECUTING)
            y_executing_aborted_mask &= online_mask[:, None].expand_as(Ys)

            # keep "fill all occurrences" exactly as policy expects
            y_executing_aborted_mask = fill_all_occurrences(
                y_executing_aborted_mask.reshape_as(Ty2d),
                Ty_unique, Ty_inverse, Ty_first_occ_mask
            ).view_as(Ys)

            # y_aborted_mask = y_aborted_for_unavailable_x_mask | y_downloading_aborted_mask | y_executing_aborted_mask
            y_aborted_mask = y_downloading_aborted_mask | y_executing_aborted_mask

            Cy_uploading_stage = Cy.clone()
            Cy_uploading_stage[:, y_aborted_mask] = 0.
            resource_overflow_uploading = torch.clip(
                compute_A(Cx_modified, Cy_uploading_stage) - A,
                min=0.0,
            )[Resource.Y_resources(state=YState.UPLOADING)]
            y_uploading_aborted_mask = (Ys == YState.UPLOADING) & \
                resource_overflow_uploading.any(dim=0)[None,
                                                       None, :, None] & online_mask[:, None].expand_as(Ys)
            y_aborted_mask |= y_uploading_aborted_mask

            assert torch.all(x_actions[x_aborted_mask] == ClientXAction.PAUSE)
            assert torch.all(x_actions[x_resume_mask] == ClientXAction.RESUME)
            assert torch.all(
                x_actions[~x_aborted_mask & ~x_resume_mask] == ClientXAction.NONE)
            assert torch.all(yi_actions[y_aborted_mask] == 0)
            assert torch.all(
                yi_actions[~y_aborted_mask] == Yi[~y_aborted_mask])
            assert torch.all(ys_actions[y_aborted_mask] == ClientYAction.ABORT)
            assert torch.all(ys_actions[~y_aborted_mask] == ClientYAction.NONE)
            assert_no_overflow(A, Cx, Cy, x_actions, ys_actions)

            Cy = Cy_temp  # Restore Cy for the next test

            # 6) Test cancelling Y uploads after cancelling X and Y downloads and Y executions

            x_memory_index = Resource.X_resource_idx(Resource.MEMORY)
            y_memory_index = Resource.Y_resource_idx(Resource.MEMORY)

            X_res = Cx.clone() + tol
            X_res[x_memory_index, (X == XState.DOWNLOADING)] = 0.

            Y_res = Cy.clone() + tol
            Y_res[y_memory_index, ((Ys == YState.EXECUTING) | (
                Ys == YState.DOWNLOADING))] = 0.
            Y_res[y_memory_index, (Ys == YState.UPLOADING) & torch.randint(
                0, 2, size=Ys.shape, dtype=torch.bool)] = 0

            A = compute_A(X_res, Y_res)

            x_actions, yi_actions, ys_actions = sample_actions()

            Cy_uploading_memory = Cy[y_memory_index].clone()
            Cy_uploading_memory[Ys != YState.UPLOADING] = 0.
            uploading_memory_overflow = Cy_uploading_memory.sum(
                dim=(0, 1, 3)) > A[Resource.MEMORY]

            x_aborted_mask = (
                X == XState.DOWNLOADING) & online_mask.expand_as(X)
            x_resume_mask = (
                X == XState.PAUSED) & ~y_active_node_mask & online_mask.expand_as(X)
            y_downloading_aborted_mask = (
                Ys == YState.DOWNLOADING) & online_mask[:, None].expand_as(Ys)
            y_executing_aborted_mask = (
                Ys == YState.EXECUTING) & online_mask[:, None].expand_as(Ys)
            y_uploading_aborted_mask = (Ys == YState.UPLOADING) & uploading_memory_overflow[:, None].expand_as(
                Ys) & online_mask[:, None].expand_as(Ys)
            y_aborted_mask = y_downloading_aborted_mask | y_executing_aborted_mask | y_uploading_aborted_mask

            assert torch.all(x_actions[x_aborted_mask] == ClientXAction.PAUSE)
            assert torch.all(x_actions[x_resume_mask] == ClientXAction.RESUME)
            assert torch.all(
                x_actions[~x_aborted_mask & ~x_resume_mask] == ClientXAction.NONE)
            assert torch.all(yi_actions[y_aborted_mask] == 0)
            assert torch.all(
                yi_actions[~y_aborted_mask] == Yi[~y_aborted_mask])
            assert torch.all(ys_actions[y_aborted_mask] == ClientYAction.ABORT)
            assert torch.all(ys_actions[~y_aborted_mask] == ClientYAction.NONE)
            assert_no_overflow(A, Cx, Cy, x_actions, ys_actions)

            # 7) Test cancelling Y uploads only because they use bandwidth_out resource, which is only used for uploads

            bandwidth_out_idx = Resource.Y_resource_idx(Resource.BANDWIDTH_OUT)

            Y_res = Cy.clone() + tol
            Y_res[bandwidth_out_idx, (Ys == YState.UPLOADING) & torch.randint(
                0, 2, size=Ys.shape, dtype=torch.bool)] = 0.

            A = compute_A(Cx + tol, Y_res)

            x_actions, yi_actions, ys_actions = sample_actions()

            bandwidth_overflow = Cy[bandwidth_out_idx].sum(
                dim=(0, 1, 3)) > A[Resource.BANDWIDTH_OUT]
            x_resume_mask = (
                X == XState.PAUSED) & ~y_active_node_mask & online_mask.expand_as(X)
            y_aborted_mask = (Ys == YState.UPLOADING) & bandwidth_overflow[:, None].expand_as(
                Ys) & online_mask[:, None].expand_as(Ys)
            # y_aborted_mask |= get_y_abort_mask_for_unavailable_x(
            #     X, x_actions, Ys)

            assert torch.all(x_actions[x_resume_mask] == ClientXAction.RESUME)
            assert torch.all(x_actions[~x_resume_mask] == ClientXAction.NONE)
            assert torch.all(yi_actions[y_aborted_mask] == 0)
            assert torch.all(
                yi_actions[~y_aborted_mask] == Yi[~y_aborted_mask])
            assert torch.all(ys_actions[y_aborted_mask] == ClientYAction.ABORT)
            assert torch.all(ys_actions[~y_aborted_mask] == ClientYAction.NONE)
            assert_no_overflow(A, Cx, Cy, x_actions, ys_actions)

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
