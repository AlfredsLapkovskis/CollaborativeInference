import pyro.distributions as dist
import torch
from simulation.types import *

from simulation.transition_functions import YshatTransitionFunction


class YshatTransitionFunctionDefault(YshatTransitionFunction):
    def __call__(self, Ys_prev, Yi_prev, O_curr, server_actions, client_actions):
        # All tensors have shape [partition, task_type, node, client_slot]

        Yshat_curr = Ys_prev.clone()

        online_mask = torch.ones(
            (1, 1, Yshat_curr.shape[-2], 1), dtype=torch.bool)
        online_mask[0, 0, 1:, 0] = O_curr == 1

        assert torch.all(Yshat_curr[online_mask & (
            server_actions == ServerYAction.EXECUTE)] == YState.IDLE)

        Yshat_cancelled = Yshat_curr[online_mask & (
            server_actions == ServerYAction.CANCEL)]
        assert torch.all((Yshat_cancelled == YState.DOWNLOADING) | (
            Yshat_cancelled == YState.EXECUTING) | (Yshat_cancelled == YState.UPLOADING) | (Yshat_cancelled == YState.DONE))

        Yshat_aborted = Yshat_curr[online_mask & (
            client_actions == ClientYAction.ABORT)]
        assert torch.all((Yshat_aborted == YState.DOWNLOADING) | (
            Yshat_aborted == YState.EXECUTING) | (Yshat_aborted == YState.UPLOADING))

        Yshat_curr[online_mask & (
            server_actions == ServerYAction.EXECUTE)] = YState.DOWNLOADING

        # Server doesn't have uploading/downloading stages
        Yshat_curr[:, :, 0, :][online_mask[:, :, 0, :] & (
            server_actions == ServerYAction.EXECUTE)[:, :, 0, :]] = YState.EXECUTING

        # Can be finished offline
        Yshat_curr[server_actions == ServerYAction.FINISH] = YState.IDLE
        Yshat_curr[(online_mask | (Yshat_curr == YState.DONE)) & (
            server_actions == ServerYAction.CANCEL)] = YState.IDLE
        Yshat_curr[online_mask & (
            client_actions == ClientYAction.ABORT)] = YState.IDLE

        Yshat_curr[~online_mask & (Ys_prev != YState.DONE)] = YState.IDLE

        return dist.Delta(Yshat_curr)
