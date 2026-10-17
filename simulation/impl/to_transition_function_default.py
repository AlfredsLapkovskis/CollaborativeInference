import numpy as np
import torch
import pyro.distributions as dist

from simulation.transition_functions import ToTransitionFunction


class ToTransitionFunctionDefault(ToTransitionFunction):
    # OFF state: time spent offline (inter-session time) ~ LogNormal
    # Based on Benevenuto et al. (2009, IMC): inter-session times fit LogNormal.
    # Median offline ~1.46h, mean ~4.5h, calibrated so that together with the ON
    # distribution the steady-state P(online) converges to ~10%.
    _OFF_LOC = np.log(5259.)
    _OFF_SCALE = 1.5

    # ON state: session duration (time online) ~ Weibull
    # Based on Gyarmati & Trinh (2010, IEEE Network): "users' online time spending
    # can be modeled with Weibull distributions".
    # Concentration < 1 gives decreasing hazard: most sessions end quickly, long tail.
    # Calibrated to median=10min, mean=30min; yields ~5 on/off cycles per user per day.
    _ON_SCALE = 1133.1
    _ON_CONCENTRATION = 0.5764

    def __call__(self, To_prev, O_prev):
        To_next = To_prev + 1.0

        def p_inc(d):
            S_prev = 1.0 - d.cdf(To_prev)
            S_next = 1.0 - d.cdf(To_next)
            return S_next / S_prev

        def two_point_mix(p):
            probs = torch.stack([1.0 - p, p], dim=-1)
            comps = torch.stack([torch.zeros_like(To_prev), To_next], dim=-1)
            return dist.MixtureSameFamily(
                mixture_distribution=dist.Categorical(probs=probs),
                component_distribution=dist.Delta(comps),
            )

        p_off = p_inc(dist.LogNormal(loc=self._OFF_LOC, scale=self._OFF_SCALE))
        p_on = p_inc(dist.Weibull(scale=self._ON_SCALE,
                     concentration=self._ON_CONCENTRATION))

        return dist.MaskedMixture(
            mask=O_prev == 1,
            component0=two_point_mix(p_off),
            component1=two_point_mix(p_on),
        )
