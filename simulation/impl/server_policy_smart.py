import torch
import pyro.distributions as dist
from random import shuffle

from simulation.impl.server_policy_default import ServerPolicyDefault
from simulation.types import *
# from utils.unique import unique_per_row


class ServerPolicySmart(ServerPolicyDefault):
    def __init__(self, model_context, server_priority_scale, randomize=True):
        """
        server_priority_scale: how much to prioritize the server node (node 0) compared to other nodes when assigning tasks. 
        Higher value means more priority to the server node.
        """
        super().__init__(model_context)
        self.server_priority_scale = server_priority_scale
        self.randomize = randomize

    def _do_action_y(self, O, To, Q, Tq, A, X, Tx, Cx, Yi, Ys, Ty, Cy, W):
        Ys_actions = torch.full_like(Ys, fill_value=ServerYAction.NONE)
        Yi_actions = Yi.clone()

        use_exec_lookup = self._y_exec_required_memory_by_model_batch is not None
        if use_exec_lookup:
            mem_free = self._get_free_memory(A, Cx, Cy).clone()
        else:
            memory_ok_node_mask = self._has_memory_headroom(A, Cx, Cy)

        online_node_mask = torch.ones(
            (O.shape[0] + 1,), dtype=torch.bool, device=A.device)
        online_node_mask[1:] = O == 1
        if not use_exec_lookup:
            online_node_mask &= memory_ok_node_mask

        partition_mask = self._model_context.partition_mask

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
            dim=2,
            index=idx,
            src=valid.to(torch.uint8),
            reduce="amax",
            include_self=True,
        )
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
        tasks_to_remove_mask = existing_task_mask & ~q_mask & partition_mask[..., None]

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
        cancel_flat = tasks_to_remove_mask.gather(
            dim=2,
            index=(Yi - 1).view(partition, task_type, -1).clamp(min=0),
        ).bool()
        finish_flat = tasks_to_finish_mask.gather(
            dim=2,
            index=(Yi - 1).view(partition, task_type, -1).clamp(min=0),
        ).bool()
        
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

        partitions = list(range(self._model_context.n_partitions))
        tasks = list(range(self._model_context.n_tasks))

        if self.randomize:
            shuffle(partitions)
            shuffle(tasks)

        for p in partitions:
            for t in tasks:
                if not self._model_context.partition_mask[p, t]:
                    continue

                if use_exec_lookup:
                    model_idx = int(
                        self._model_context.partition_model_indices[p, t].item())
                    required_by_batch = self._y_exec_required_memory_by_model_batch[model_idx]

                node_priority = ((Ys != YState.IDLE) & (Ys != YState.DONE) & (
                    Ys_actions != ServerYAction.EXECUTE)).sum(dim=(0, 1, 3)).to(torch.float32)

                node_priority[~online_node_mask] = float("inf")

                node_priority[0] *= 1 / max(self.server_priority_scale, 1e-12)
                nodes = torch.argsort(node_priority, stable=not self.randomize)

                waiting = torch.nonzero(
                    tasks_to_add_mask[p, t]).flatten().to(Yi_actions.dtype)
                if waiting.numel() == 0:
                    continue
                if self.randomize:
                    waiting = waiting[torch.randperm(waiting.numel())]

                sorted_ys = Ys[p, t, nodes]  # [N, S]
                idle_mask = (
                    sorted_ys == YState.IDLE) & online_node_mask[nodes, None] & (X[p, t, nodes, None] == XState.READY)

                idle_counts = idle_mask.sum(dim=1).to(torch.long)
                if int(idle_counts.sum().item()) == 0:
                    continue

                cap = idle_counts
                if use_exec_lookup:
                    # Make sure it's monotone so `searchsorted` is valid.
                    req = torch.cummax(required_by_batch.to(
                        mem_free.dtype), dim=0).values
                    mem_sorted = mem_free[nodes]
                    max_by_mem = torch.searchsorted(
                        req, mem_sorted, right=True) - 1
                    max_by_mem = torch.clamp(max_by_mem, min=0)
                    cap = torch.minimum(cap, max_by_mem)

                # Take the first `cap[i]` idle slots per node.
                assign_mask = idle_mask & (
                    idle_mask.cumsum(dim=1) <= cap[:, None])
                assign_indices = assign_mask.nonzero(as_tuple=False)
                n_assign = min(
                    int(assign_indices.shape[0]), int(waiting.numel()))
                if n_assign <= 0:
                    continue
                assign_indices = assign_indices[:n_assign]

                actual_nodes = nodes[assign_indices[:, 0]]
                actual_slots = assign_indices[:, 1]

                Ys_actions[p, t, actual_nodes,
                           actual_slots] = ServerYAction.EXECUTE
                Yi_actions[p, t, actual_nodes,
                           actual_slots] = waiting[:n_assign] + 1
                tasks_to_add_mask[p, t, waiting[:n_assign].to(
                    torch.long)] = False

                if use_exec_lookup:
                    batch_used_sorted = torch.bincount(
                        assign_indices[:, 0], minlength=nodes.shape[0]
                    ).to(torch.long)
                    mem_free[nodes] -= req[batch_used_sorted]

        assert torch.all(Yi_actions[Ys_actions == ServerYAction.EXECUTE] != 0)

        return dist.Delta(Yi_actions), dist.Delta(Ys_actions)
