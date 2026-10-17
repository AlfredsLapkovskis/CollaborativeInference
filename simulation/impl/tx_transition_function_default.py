import pyro.distributions as dist
import torch

from simulation.model_context import ModelContext
from simulation.types import *
from simulation.transition_functions import TxTransitionFunction


class TxTransitionFunctionDefault(TxTransitionFunction):
    def __init__(self, model_context: ModelContext):
        # partition_sizes has shape [partition, task_type]

        self._model_context = model_context

    def __call__(self, Tx_prev, Xhat_curr, Cx_curr):
        # Tx_prev, Xhat_curr have shape [partition, task_type, node]
        # Cx_curr has shape [x_resource, partition, task_type, node]
        # A_curr has shape [resource, node]

        downloading_mask: torch.Tensor = Xhat_curr == XState.DOWNLOADING

        storage_idx = Resource.X_resource_idx(Resource.STORAGE)

        Tx_curr = Tx_prev.clone()

        continuing_mask = (
            Cx_curr[storage_idx] < self._model_context.partition_sizes.unsqueeze(-1)) & downloading_mask
        paused_mask = Xhat_curr == XState.PAUSED

        Tx_curr[continuing_mask] += 1
        Tx_curr[~(continuing_mask | paused_mask)] = 0

        return dist.Delta(Tx_curr)
