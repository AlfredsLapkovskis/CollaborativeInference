import pyro.distributions as dist
import torch
from simulation.types import *

from simulation.transition_functions import YihatTransitionFunction


class YihatTransitionFunctionDefault(YihatTransitionFunction):
    def __call__(self, Yi_prev, Ys_prev, O_curr, server_actions, client_actions):
        # All tensors have shape [partition, task_type, node, client_slot]

        Yihat_curr = Yi_prev.clone()

        online_mask = torch.ones(
            (1, 1, Yihat_curr.shape[-2], 1), dtype=torch.bool)
        online_mask[0, 0, 1:, 0] = O_curr == 1

        active_mask = Yi_prev != 0

        actions_for_active_yi = server_actions[active_mask]
        assert torch.all((actions_for_active_yi == 0) | (
            actions_for_active_yi == Yi_prev[active_mask]))

        # Clients don't initiate new Yi tasks
        assert torch.all((client_actions == 0) | (client_actions == Yi_prev))

        server_update_mask = online_mask | (
            (Ys_prev == YState.DONE) & (server_actions == 0))
        Yihat_curr[server_update_mask] = server_actions[server_update_mask]
        Yihat_curr[online_mask & (client_actions == 0) & (
            Yi_prev != 0)] = 0  # Client aborts

        Yihat_curr[~online_mask & (Ys_prev != YState.DONE)] = 0

        return dist.Delta(Yihat_curr)
