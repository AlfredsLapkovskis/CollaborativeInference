import torch
import pyro.distributions as dist

from simulation.model_context import ModelContext
from simulation.transition_functions import TqTransitionFunction
from simulation.types import *


class TqTransitionFunctionDefault(TqTransitionFunction):
    # OFF state: idle time between requests (think time) ~ Pareto
    # Based on Benevenuto et al. (2009, IMC): inter-request times are heavy-tailed.
    # scale=30s sets the minimum think time; median ~48s, mean=90s — consistent
    # with a chat/LLM app where composing a prompt takes at least half a minute.
    # alpha=1.5 from self-similarity literature (Crovella & Bestavros 1997).
    _OFF_SCALE = 30.
    _OFF_ALPHA = 1.5

    # ON state: request patience / time until user cancels ~ Pareto
    # scale=30s ensures all users wait at least 30s before cancelling — reasonable
    # for a streaming LLM where tokens appear within seconds of submission.
    # alpha=1.5 gives median ~48s, mean=90s patience, with a heavy tail for
    # patient users. Together with OFF, P(requesting | online) = 50%.
    _ON_SCALE = 30.
    _ON_ALPHA = 1.5

    def __init__(self, model_context: ModelContext):
        self._model_context = model_context

    def __call__(self, Tq_prev, Q_prev, O_curr, Yi_prev, Ys_prev):
        online_mask = O_curr == 1

        done_Yi = Yi_prev.clone()
        done_Yi[Ys_prev != YState.DONE] = 0

        partition_counts = self._model_context.partition_mask.sum(dim=0)
        Yi_unique, Yi_counts = torch.unique(done_Yi, return_counts=True)
        Yi_counts = Yi_counts[(Yi_unique > 0) & (Q_prev[Yi_unique - 1] > 0)]
        Yi_unique = Yi_unique[(Yi_unique > 0) & (Q_prev[Yi_unique - 1] > 0)]

        finished_mask = torch.zeros_like(Q_prev, dtype=torch.bool)
        total_task_partition_counts = partition_counts[Q_prev[Yi_unique - 1] - 1]
        # TODO: ASSUMING EACH CLIENT TASK'S PARTITION IS SCHEDULED ONLY ONCE SIMULTANEOUSLY!!!
        finished_mask[Yi_unique -
                      1] = Yi_counts == total_task_partition_counts
        assert torch.all(Yi_counts <= total_task_partition_counts)

        Tq_next = Tq_prev + 1.0

        def p_inc(d: dist.Distribution, min_val=None):
            Tq_next_mask = Tq_next >= min_val
            Tq_prev_mask = Tq_prev >= min_val

            S_prev = torch.ones_like(
                Tq_prev, dtype=self._model_context.float_type)
            S_next = torch.ones_like(
                Tq_next, dtype=self._model_context.float_type)

            S_prev[Tq_prev_mask] = 1. - d.cdf(Tq_prev[Tq_prev_mask])
            S_next[Tq_next_mask] = 1. - d.cdf(Tq_next[Tq_next_mask])

            return S_next / S_prev

        def two_point_mix(p):
            probs = torch.stack([1.0 - p, p], dim=-1)
            comps = torch.stack([torch.zeros_like(Tq_prev), Tq_next], dim=-1)
            return dist.MixtureSameFamily(
                mixture_distribution=dist.Categorical(probs=probs),
                component_distribution=dist.Delta(comps),
            )

        p_off = p_inc(dist.Pareto(scale=self._OFF_SCALE,
                      alpha=self._OFF_ALPHA), min_val=self._OFF_SCALE)
        p_on = p_inc(dist.Pareto(scale=self._ON_SCALE,
                     alpha=self._ON_ALPHA), min_val=self._ON_SCALE)

        return dist.MaskedMixture(
            mask=online_mask & ~finished_mask,
            component0=dist.Delta(torch.tensor([0])),
            component1=dist.MaskedMixture(
                mask=Q_prev != 0,
                component0=two_point_mix(p_off),
                component1=two_point_mix(p_on),
            ),
        )
