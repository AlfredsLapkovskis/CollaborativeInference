import pyro.distributions as dist
import torch
from simulation.types import *

from simulation.transition_functions import YsTransitionFunction


class YsTransitionFunctionDefault(YsTransitionFunction):
    def __call__(self, Yshat_curr, Yihat_curr, Ty_curr):
        Ys_curr = Yshat_curr.clone()

        Ty_zero = Ty_curr == 0

        Ys_curr[Ty_zero & (Yshat_curr == YState.DOWNLOADING)
                ] = YState.EXECUTING
        Ys_curr[Ty_zero & (Yshat_curr == YState.EXECUTING)] = YState.UPLOADING

        # Server doesn't have uploading/downloading stages, so it can be finished directly from executing
        Ys_curr[:, :, 0, :][Ty_zero[:, :, 0, :] & (
            Yshat_curr == YState.EXECUTING)[:, :, 0, :]] = YState.DONE

        Ys_curr[Ty_zero & (Yshat_curr == YState.UPLOADING)] = YState.DONE

        return dist.Delta(Ys_curr)
