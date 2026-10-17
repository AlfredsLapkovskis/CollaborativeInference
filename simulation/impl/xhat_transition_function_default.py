import pyro.distributions as dist
import torch
from simulation.types import *

from simulation.transition_functions import XhatTransitionFunction


class XhatTransitionFunctionDefault(XhatTransitionFunction):
    def __call__(self, X_prev, O_curr, server_actions, client_actions):
        # All tensors have shape [partition, task_type, node]

        Xhat_curr = X_prev.clone()

        assert torch.all(Xhat_curr[server_actions ==
                                   ServerXAction.DOWNLOAD] == XState.EMPTY)

        Xhat_cancelled = Xhat_curr[server_actions == ServerXAction.CANCEL]
        assert torch.all((Xhat_cancelled == XState.DOWNLOADING)
                         | (Xhat_cancelled == XState.READY))

        Xhat_aborted = Xhat_curr[client_actions == ClientXAction.ABORT]
        assert torch.all((Xhat_aborted == XState.DOWNLOADING)
                         | (Xhat_aborted == XState.READY)
                         | (Xhat_aborted == XState.PAUSED))

        Xhat_paused = Xhat_curr[client_actions == ClientXAction.PAUSE]
        assert torch.all(Xhat_paused == XState.DOWNLOADING)

        Xhat_resumed = Xhat_curr[client_actions == ClientXAction.RESUME]
        assert torch.all(Xhat_resumed == XState.PAUSED)

        online_mask = torch.ones(
            (1, 1, Xhat_curr.shape[-1]), dtype=torch.bool)
        online_mask[0, 0, 1:] = O_curr == 1

        Xhat_curr[(server_actions == ServerXAction.DOWNLOAD)
                  & online_mask] = XState.DOWNLOADING
        Xhat_curr[(server_actions == ServerXAction.CANCEL)
                  & online_mask] = XState.EMPTY

        Xhat_curr[(client_actions == ClientXAction.PAUSE)
                  & online_mask] = XState.PAUSED
        Xhat_curr[(client_actions == ClientXAction.RESUME)
                  & online_mask] = XState.DOWNLOADING

        Xhat_curr[(client_actions == ClientXAction.ABORT)
                  & online_mask] = XState.EMPTY

        return dist.Delta(Xhat_curr)
