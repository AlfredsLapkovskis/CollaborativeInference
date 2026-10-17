import pyro.distributions as dist

from simulation.transition_functions import OTransitionFunction


class OTransitionFunctionDefault(OTransitionFunction):
    def __call__(self, O_prev, To_curr):
        mask = To_curr > 0
        return dist.MaskedMixture(
            mask=mask,
            component0=dist.Delta(1 - O_prev),
            component1=dist.Delta(O_prev)
        )
