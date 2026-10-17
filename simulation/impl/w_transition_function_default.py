import pyro.distributions as dist
import torch
from simulation.types import *

from simulation.transition_functions import WTransitionFunction


class WTransitionFunctionDefault(WTransitionFunction):
    def __call__(self, Xhat_curr, Tx_curr):
        return dist.Delta(torch.zeros(1))
