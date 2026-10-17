from typing import Optional
import torch
import pyro.distributions as dist
from random import shuffle

from simulation.model_context import ModelContext
from simulation.policies import ServerPolicy
from simulation.types import *


class ServerPolicyDefault(ServerPolicy):
    def __init__(self, model_context: ModelContext):
        self._model_context = model_context

        # Conservative per-subtask memory requirement (bytes).
        # Used to avoid scheduling onto nodes that are likely to trigger client aborts.
        self._required_x_memory = self._estimate_required_x_memory()

        # Fallback scalar Y-memory requirement when model-specific lookup is unavailable.
        self._required_y_memory = self._estimate_required_y_memory()

        # Batch- and model-aware Y executing memory requirements (bytes).
        self._y_exec_memory_k = 2.0
        self._y_exec_required_memory_by_model_batch = self._build_y_exec_required_memory_by_model_batch(
            k=self._y_exec_memory_k
        )

    def _estimate_required_x_memory(self) -> float:
        def mean_std(m):
            means = list(getattr(m, "original_means", {}).values())
            stds = list(getattr(m, "original_stds", {}).values())
            mean = float(sum(means) / max(len(means), 1))
            std = float(sum(stds) / max(len(stds), 1))
            return mean, std

        k = 2.0

        try:
            x_mem = self._model_context.x_resource_models[
                Resource.X_resource_idx(Resource.MEMORY)
            ]
        except Exception:
            return 0.0

        mu, sigma = mean_std(x_mem)
        return max(0.0, float(mu + k * sigma))

    def _estimate_required_y_memory(self) -> float:
        def mean_std(m):
            means = list(getattr(m, "original_means", {}).values())
            stds = list(getattr(m, "original_stds", {}).values())
            mean = float(sum(means) / max(len(means), 1))
            std = float(sum(stds) / max(len(stds), 1))
            return mean, std

        # Prefer a high-quantile proxy using mean + k*std across stages.
        # Keeping this a scalar keeps the policy fast and avoids per-(p,t) model lookups.
        k = 2.0

        try:
            y_mem_down = self._model_context.y_downloading_resource_models[
                Resource.Y_resource_idx(
                    Resource.MEMORY, state=YState.DOWNLOADING)
            ]
            y_mem_exec = self._model_context.y_executing_resource_models[
                Resource.Y_resource_idx(
                    Resource.MEMORY, state=YState.EXECUTING)
            ]
            y_mem_up = self._model_context.y_uploading_resource_models[
                Resource.Y_resource_idx(
                    Resource.MEMORY, state=YState.UPLOADING)
            ]
        except Exception:
            return 0.0

        reqs = []
        for m in (y_mem_down, y_mem_exec, y_mem_up):
            mu, sigma = mean_std(m)
            reqs.append(mu + k * sigma)

        return max(0.0, float(max(reqs)))

    def _build_y_exec_required_memory_by_model_batch(self, k: float) -> torch.Tensor | None:
        """Precompute required Y-executing memory (bytes) per model and batch size.

        The simulator's executing memory model is keyed by (task_count, model_name). We build
        a lookup table: required_mem[model_idx, batch_size] ~= mu + k*sigma, using the same
        inverse-distance interpolation over batch sizes as `CTransitionFunctionDefault`.
        """
        try:
            y_mem_exec = self._model_context.y_executing_resource_models[
                Resource.Y_resource_idx(
                    Resource.MEMORY, state=YState.EXECUTING)
            ]
        except Exception:
            return None

        keys = list(getattr(y_mem_exec, "keys", []) or [])
        if not keys:
            return None

        model_names = list(self._model_context.model_list)
        if not model_names:
            return None

        # Extract available batch sizes from keys: (task_count, model_name)
        task_counts = sorted({int(task_count)
                             for (task_count, _model_name) in keys})
        if not task_counts:
            return None

        device = self._model_context.device
        dtype = self._model_context.float_type

        counts_t = torch.as_tensor(
            task_counts, dtype=torch.int64, device=device)
        n_models = len(model_names)
        n_counts = len(task_counts)

        mu_table = torch.zeros((n_models, n_counts),
                               dtype=dtype, device=device)
        sigma_table = torch.zeros_like(mu_table)

        model_idx_map = {name: i for i, name in enumerate(model_names)}
        count_idx_map = {c: i for i, c in enumerate(task_counts)}

        for (task_count, model_name), mu in getattr(y_mem_exec, "original_means", {}).items():
            mi = model_idx_map.get(model_name)
            ci = count_idx_map.get(int(task_count))
            if mi is not None and ci is not None:
                mu_table[mi, ci] = float(mu)

        for (task_count, model_name), sigma in getattr(y_mem_exec, "original_stds", {}).items():
            mi = model_idx_map.get(model_name)
            ci = count_idx_map.get(int(task_count))
            if mi is not None and ci is not None:
                sigma_table[mi, ci] = float(sigma)

        max_batch = max(
            int(self._model_context.n_task_exec_slots), int(max(task_counts)))
        required = torch.zeros((n_models, max_batch + 1),
                               dtype=dtype, device=device)
        required[:, 0] = 0.0

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

            required[:, b] = torch.clamp(mu + float(k) * sigma, min=0.0)

        return required

    def _get_free_memory(self, A, Cx, Cy) -> torch.Tensor:
        """Compute free memory per node (bytes), based on current consumptions."""
        mem_avail = A[Resource.MEMORY]
        mem_used = torch.zeros_like(mem_avail)

        try:
            x_mem_idx = Resource.X_resource_idx(Resource.MEMORY)
            mem_used += Cx[x_mem_idx].sum(dim=(0, 1))
        except Exception:
            pass

        try:
            y_mem_idx = Resource.Y_resource_idx(Resource.MEMORY)
            mem_used += Cy[y_mem_idx].sum(dim=(0, 1, 3))
        except Exception:
            pass

        return mem_avail - mem_used

    def _has_memory_headroom(self, A, Cx, Cy, required_memory: float | None = None) -> torch.Tensor:
        # Returns a boolean mask of shape [node] indicating if node has enough free memory.
        required_memory = self._required_y_memory if required_memory is None else required_memory
        if required_memory <= 0:
            return torch.ones((A.shape[-1],), dtype=torch.bool, device=A.device)

        mem_free = self._get_free_memory(A, Cx, Cy)
        return mem_free >= required_memory

    def __call__(self, O, To, Q, Tq, A, X, Tx, Cx, Yi, Ys, Ty, Cy, W):
        return (self._do_action_x(O, To, Q, Tq, A, X, Tx, Cx, Yi, Ys, Ty, Cy, W),
                *self._do_action_y(O, To, Q, Tq, A, X, Tx, Cx, Yi, Ys, Ty, Cy, W))

    def _do_action_x(self, O, To, Q, Tq, A, X, Tx, Cx, Yi, Ys, Ty, Cy, W):
        # Basically, here we download partitions one by one for each client
        # (many clients in parallel, but each client downloads only one partition at a time) while
        # those clients have enough storage and are online.
        X_actions = torch.full_like(X, fill_value=ServerXAction.NONE)

        storage_idx = Resource.X_resource_idx(Resource.STORAGE)

        offline_mask = O == 0
        inactive_mask = ((X != XState.DOWNLOADING) &
                         (X != XState.PAUSED)).all(dim=(0, 1))
        available_storage = A[Resource.STORAGE] - \
            Cx[storage_idx].sum(dim=(0, 1))

        memory_ok_node_mask = self._has_memory_headroom(
            A, Cx, Cy, required_memory=self._required_x_memory)

        n_partitions = self._model_context.n_partitions
        n_tasks = self._model_context.n_tasks
        partition_sizes = self._model_context.partition_sizes
        partition_mask = self._model_context.partition_mask

        # Candidate partitions to download: those that require <= storage than available,
        # are not already present, actually exist (partition mask) and belong to clients that
        # are not downloading anything currently.
        candidate_partitions = available_storage >= partition_sizes[:, :, None]
        candidate_partitions &= (
            X == XState.EMPTY) & partition_mask[:, :, None] & inactive_mask[None, None, :]
        candidate_partitions &= memory_ok_node_mask[None, None, :]

        random_indices = torch.randint_like(
            candidate_partitions, low=0, high=10000, dtype=torch.int32)
        # Make sure False candidates are not selected
        random_indices[~candidate_partitions] = -1
        selected_indices = random_indices.reshape(
            n_partitions * n_tasks, -1).argmax(dim=0)

        X_action_mask = torch.full(
            (n_partitions * n_tasks, candidate_partitions.shape[-1]), fill_value=False)
        X_action_mask[selected_indices, torch.arange(
            len(selected_indices))] = True
        X_action_mask = X_action_mask.reshape(n_partitions, n_tasks, -1)
        X_action_mask &= candidate_partitions

        X_actions[X_action_mask] = ServerXAction.DOWNLOAD

        # Make sure that server node does nothing!
        X_actions[:, :, 0] = ServerXAction.NONE
        # Make sure nothing happens to offline clients
        X_actions[:, :, 1:][:, :, offline_mask] = ServerXAction.NONE

        return dist.Delta(X_actions)

    def _do_action_y(self, O, To, Q, Tq, A, X, Tx, Cx, Yi, Ys, Ty, Cy, W):
        Ys_actions = torch.full_like(Ys, fill_value=ServerYAction.NONE)
        Yi_actions = Yi.clone()

        online_mask = O == 1
        partition_mask = self._model_context.partition_mask

        use_exec_lookup = self._y_exec_required_memory_by_model_batch is not None
        if use_exec_lookup:
            # Track remaining free memory as we assign new tasks this step.
            mem_free = self._get_free_memory(A, Cx, Cy).clone()
        else:
            # Fallback: scalar per-subtask memory gate.
            memory_ok_node_mask = self._has_memory_headroom(A, Cx, Cy)

        partition, task_type, node, client_slot = Yi.shape
        client = Q.shape[0]
        flat = Yi.reshape(partition, task_type, node * client_slot)
        valid = (flat >= 1) & (flat <= client)

        idx = (flat - 1).to(torch.long).clamp_(0,
                                               client - 1)  # in-bounds for all
        # CUDA scatter_reduce_ does not support bool tensors on some PyTorch builds.
        existing_task_mask_u8 = torch.zeros(
            (partition, task_type, client), device=Yi.device, dtype=torch.uint8)
        existing_task_mask_u8.scatter_reduce_(
            2, idx, valid.to(torch.uint8), reduce="amax", include_self=True)
        existing_task_mask = existing_task_mask_u8.bool()

        # Requested tasks: [1,K,C]
        q_mask = torch.zeros((1, task_type, client),
                             device=Yi.device, dtype=torch.bool)

        client_idx = (Q > 0).nonzero(as_tuple=True)[0]              # shape [n]
        # shape [n], 0..K-1
        task_idx = (Q[client_idx] - 1).to(torch.long)

        q_mask[0, task_idx, client_idx] = True

        # [P,K,C]
        tasks_to_add_mask = ~existing_task_mask & q_mask & partition_mask[..., None]
        # [P,K,C]
        tasks_to_cancel_mask = existing_task_mask & ~q_mask & partition_mask[..., None]

        existing_task_mask = None  # free memory

        # Only consider done tasks for finishing
        done = Ys.reshape(partition, task_type, node *
                          client_slot) == YState.DONE
        tasks_to_finish_mask_u8 = torch.zeros(
            (partition, task_type, client), device=Yi.device, dtype=torch.uint8)
        tasks_to_finish_mask_u8.scatter_reduce_(
            2, idx, (valid & done).to(torch.uint8), reduce="amax", include_self=True)
        tasks_to_finish_mask = tasks_to_finish_mask_u8.bool()

        tasks_to_finish_mask &= q_mask & partition_mask[..., None]
        q_task_idx = (Q.to(torch.long) -
                      1).clamp(min=0).to(partition_mask.device)
        tasks_to_finish_mask &= torch.where(
            Q >= 1,
            (tasks_to_finish_mask.sum(dim=(0, 1)) ==
             partition_mask[:, q_task_idx].sum(dim=0))[None, None, :],
            torch.zeros_like(tasks_to_finish_mask, dtype=torch.bool),
        )

        # Step 1: cancel finished tasks
        cancel_flat = tasks_to_cancel_mask.gather(
            dim=2,
            index=(Yi - 1).view(partition, task_type, -1).clamp(min=0),
        ).bool()
        finish_flat = tasks_to_finish_mask.gather(
            dim=2,
            index=(Yi - 1).view(partition, task_type, -1).clamp(min=0),
        ).bool()
        # TODO: Distinguish between CANCEL and FINISH actions!
        Ys_actions.view(partition, task_type, -
                        1)[cancel_flat] = ServerYAction.CANCEL
        Ys_actions.view(partition, task_type, -
                        1)[finish_flat] = ServerYAction.FINISH
        Ys_actions[Yi_actions == 0] = ServerYAction.NONE
        Yi_actions[Ys_actions == ServerYAction.CANCEL] = 0
        Yi_actions[Ys_actions == ServerYAction.FINISH] = 0

        # Step 2: assign new tasks to available slots
        waiting_client_ids = torch.arange(
            1, client + 1)[tasks_to_add_mask.any(dim=(0, 1))]
        torch.randperm(waiting_client_ids.size(0), out=waiting_client_ids)

        shuffle(list())

        partitions = list(range(self._model_context.n_partitions))
        tasks = list(range(self._model_context.n_tasks))
        slots = list(range(self._model_context.n_task_exec_slots))
        shuffle(partitions)
        shuffle(tasks)
        shuffle(slots)

        for p in partitions:
            for t in tasks:
                if not self._model_context.partition_mask[p, t]:
                    continue

                if use_exec_lookup:
                    model_idx = int(
                        self._model_context.partition_model_indices[p, t].item())
                    required_by_batch = self._y_exec_required_memory_by_model_batch[model_idx]
                    planned_batch = torch.zeros(
                        (node,), dtype=torch.long, device=A.device)
                waiting = torch.nonzero(
                    tasks_to_add_mask[p, t]).flatten().to(Yi_actions.dtype)
                if waiting.numel() == 0:
                    continue

                waiting = waiting[torch.randperm(waiting.numel())]

                wpos = 0
                for s in slots:
                    if wpos >= waiting.numel():
                        break

                    recv_mask = Ys[p, t, :, s] == YState.IDLE
                    assert torch.all(Yi[p, t, :, s][recv_mask] == 0)

                    # server node is always online
                    recv_mask[1:] &= online_mask
                    recv_mask &= X[p, t] == XState.READY

                    if not use_exec_lookup:
                        recv_mask &= memory_ok_node_mask

                    recv = torch.nonzero(recv_mask, as_tuple=False).flatten()
                    if recv.numel() == 0:
                        continue

                    if use_exec_lookup:
                        old_counts = planned_batch[recv]
                        new_counts = torch.clamp(
                            old_counts + 1, max=required_by_batch.shape[0] - 1)
                        delta_req = required_by_batch[new_counts] - \
                            required_by_batch[old_counts]
                        delta_req = torch.clamp(delta_req, min=0.0)

                        mem_ok = mem_free[recv] >= delta_req
                        recv = recv[mem_ok]
                        delta_req = delta_req[mem_ok]
                        if recv.numel() == 0:
                            continue

                    # Randomize eligible receiver nodes to avoid deterministic
                    # preference for server / low-index clients.
                    perm = torch.randperm(recv.numel(), device=recv.device)
                    recv = recv[perm]
                    if use_exec_lookup:
                        delta_req = delta_req[perm]

                    n = min(recv.numel(), waiting.numel() - wpos)

                    recv_sel = recv[:n]
                    send_sel = waiting[wpos:wpos+n]
                    wpos += n

                    if use_exec_lookup and n > 0:
                        delta_sel = delta_req[:n]
                        planned_batch[recv_sel] += 1
                        mem_free[recv_sel] -= delta_sel

                    # SAFE writes (no chained boolean indexing)
                    Ys_actions[p, t, recv_sel, s] = ServerYAction.EXECUTE
                    Yi_actions[p, t, recv_sel, s] = send_sel + 1

        assert torch.all(Yi_actions[Ys_actions == ServerYAction.EXECUTE] != 0)

        return dist.Delta(Yi_actions), dist.Delta(Ys_actions)
