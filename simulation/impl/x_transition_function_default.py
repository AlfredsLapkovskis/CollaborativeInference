import pyro.distributions as dist
import torch
from simulation.types import *

from simulation.transition_functions import XTransitionFunction


class XTransitionFunctionDefault(XTransitionFunction):
    def __call__(self, Xhat_curr, Tx_curr):
        # All tensors have shape [partition, task_type, client]

        assert torch.all((Xhat_curr[Tx_curr != 0] == XState.DOWNLOADING)
                         | (Xhat_curr[Tx_curr != 0] == XState.PAUSED))

        X_curr = Xhat_curr.clone()
        X_curr[(Tx_curr == 0) & (Xhat_curr == XState.DOWNLOADING)] = XState.READY

        return dist.Delta(X_curr)
