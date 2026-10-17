import pyro.distributions as dist
import torch
from simulation.types import *

from simulation.transition_functions import YiTransitionFunction


class YiTransitionFunctionDefault(YiTransitionFunction):
    def __call__(self, Yihat_curr, Yshat_curr, Ty_curr):
        return dist.Delta(Yihat_curr)
