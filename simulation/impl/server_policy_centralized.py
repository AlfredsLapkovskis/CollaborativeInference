import torch
import pyro.distributions as dist
from random import shuffle

from simulation.impl.server_policy_default import ServerPolicyDefault
from simulation.types import *


class ServerPolicyCentralized(ServerPolicyDefault):
    def _do_action_x(self, O, To, Q, Tq, A, X, Tx, Cx, Yi, Ys, Ty, Cy, W):
        X_actions = torch.full_like(X, fill_value=ServerXAction.NONE)
        return dist.Delta(X_actions)

    def _do_action_y(self, O, To, Q, Tq, A, X, Tx, Cx, Yi, Ys, Ty, Cy, W):
        Ys_actions = torch.full_like(Ys, fill_value=ServerYAction.NONE)
        Yi_actions = Yi.clone()

        ys_server = Ys[:, :, 0, :]
        yi_server = Yi[:, :, 0, :]

        partition_mask = self._model_context.partition_mask

        partition, task_type, _ = yi_server.shape
        client = Q.shape[0]
        valid = (yi_server >= 1) & (yi_server <= client)
        idx = (yi_server - 1).to(torch.long).clamp_(0,
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
        tasks_to_remove_mask = existing_task_mask & ~q_mask & partition_mask[..., None]
        existing_task_mask = None  # free memory

        # Only consider done tasks for finishing
        done = ys_server == YState.DONE
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
        Ys_actions[Yi_actions == 0] = ServerYAction.NONE  # fix
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
                waiting = torch.nonzero(
                    tasks_to_add_mask[p, t]).flatten().to(Yi_actions.dtype)
                if waiting.numel() == 0:
                    continue

                waiting = waiting[torch.randperm(waiting.numel())]

                recv_mask = Ys[p, t, 0, :] == YState.IDLE
                assert torch.all(Yi[p, t, 0, :][recv_mask] == 0)

                recv = torch.nonzero(recv_mask, as_tuple=False).flatten()
                if recv.numel() == 0:
                    continue

                n = min(recv.numel(), waiting.numel())

                recv_sel = recv[:n]
                send_sel = waiting[:n]

                Ys_actions[p, t, 0, recv_sel] = ServerYAction.EXECUTE
                Yi_actions[p, t, 0, recv_sel] = send_sel + 1

        assert torch.all(Yi_actions[Ys_actions == ServerYAction.EXECUTE] != 0)

        return dist.Delta(Yi_actions), dist.Delta(Ys_actions)
