import torch
import pyro.distributions as dist

from simulation.transition_functions import QTransitionFunction
from simulation.model_context import ModelContext
from simulation.types import *


class QTransitionFunctionDefault(QTransitionFunction):
    def __init__(self, model_context: ModelContext):
        self._model_context = model_context

    def __call__(self, Q_prev, O_curr, Tq_curr):
        online_mask = O_curr == 1
        positive_duration_mask = Tq_curr > 0
        was_querying_mask = Q_prev == 0

        return dist.MaskedMixture(
            mask=online_mask,
            component0=dist.Delta(torch.tensor([0])),
            component1=dist.MaskedMixture(
                mask=positive_duration_mask,
                component0=dist.MaskedMixture(
                    mask=was_querying_mask,
                    component0=dist.Delta(torch.tensor([0])),
                    component1=dist.MixtureSameFamily(
                        mixture_distribution=dist.Categorical(
                            probs=self._model_context.task_query_probabilities),
                        component_distribution=dist.Delta(torch.arange(
                            1, self._model_context.task_query_probabilities.size(0) + 1)),
                    ),
                ),
                component1=dist.Delta(Q_prev),
            ),
        )
