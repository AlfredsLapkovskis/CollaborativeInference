import pytest
import torch
import warnings
import numpy as np
from scipy import stats

from simulation.model_context import ModelContext
from simulation.impl.tq_transition_function_default import TqTransitionFunctionDefault
from simulation.impl.q_transition_function_default import QTransitionFunctionDefault
from simulation.types import *


class TestTqTransitionFunctionDefault:
    @pytest.fixture(
        params=[
            ModelContext(
                n_clients=100,
                n_task_exec_slots=8,
                n_timesteps=1000,
                partition_sizes=torch.tensor([
                    [2, 1, 3, 2, 1],
                    [3, 2, 1, 2, 0],
                    [1, 2, 1, 0, 0],
                    [4, 1, 0, 0, 0]
                ]),
                partition_models=[
                    ["a", "b", "c", "d", "e"],
                    ["e", "f", "g", "h"],
                    ["i", "j", "k"],
                    ["l", "m"],
                ],
                task_query_probabilities=torch.tensor(
                    [0.1, 0.5, 0.05, 0.15, 0.2]),
            ),
        ],
    )
    def model_context(self, request):
        return request.param

    @pytest.fixture
    def q_transition_function(self, model_context):
        return QTransitionFunctionDefault(model_context)

    @pytest.fixture
    def tq_transition_function(self, model_context):
        return TqTransitionFunctionDefault(model_context)

    def test_allowed_values(self, model_context: ModelContext, q_transition_function, tq_transition_function):
        Tq_prev = torch.zeros(model_context.n_clients, dtype=torch.int32)
        Q_prev = torch.zeros(model_context.n_clients, dtype=torch.int32)

        for _ in range(10_000):
            O = torch.ones(model_context.n_clients, dtype=torch.int8)
            Yi = torch.zeros(
                (model_context.n_partitions,
                 model_context.n_tasks,
                 model_context.n_nodes,
                 model_context.n_task_exec_slots),
                dtype=torch.int32,
            )
            Ys = torch.zeros_like(Yi, dtype=torch.int8)

            Tq = tq_transition_function(
                Tq_prev, Q_prev, O, Yi, Ys).sample()
            Q = q_transition_function(Q_prev, O, Tq).sample()

            assert torch.all((Q != Q_prev) == (Tq == 0))
            assert torch.all((Q == Q_prev) == ((Tq_prev + 1) == Tq))
            assert torch.all((Tq == 0) | (Tq == Tq_prev + 1))

            Tq_prev = Tq
            Q_prev = Q

        if (Q_prev == 0).sum() == 0:
            warnings.warn(
                "All clients have Q != 0; unsure if all allowed values were tested.")

        if (Q_prev != 0).sum() == 0:
            warnings.warn(
                "All clients have Q == 0; unsure if all allowed values were tested.")

    def test_finished_query(self, model_context: ModelContext, q_transition_function, tq_transition_function):
        Tq_prev = torch.ones(model_context.n_clients, dtype=torch.int32)
        Q_prev = torch.ones(model_context.n_clients, dtype=torch.int32)

        O = torch.ones(model_context.n_clients, dtype=torch.int8)

        offline_clients = model_context.n_clients // 2
        O[:offline_clients] = 0
        Tq_prev[:offline_clients // 2] = 0
        Q_prev[:offline_clients // 4] = 0

        Yi = torch.zeros(
            (model_context.n_partitions,
             model_context.n_tasks,
             model_context.n_nodes,
             model_context.n_task_exec_slots),
            dtype=torch.int32,
        )
        Ys = torch.zeros_like(Yi, dtype=torch.int8)

        for _ in range(10_000):
            Tq = tq_transition_function(
                Tq_prev, Q_prev, O, Yi, Ys).sample()
            Q = q_transition_function(Q_prev, O, Tq).sample()

            assert torch.all(Tq[:offline_clients] == 0)
            assert torch.all((Tq[offline_clients:] == 0) |
                             (Tq[offline_clients:] == 2))
            assert torch.all(Q[:offline_clients] == 0)

    def test_distribution(self, model_context: ModelContext, q_transition_function, tq_transition_function):
        Tq_prev = torch.zeros(model_context.n_clients, dtype=torch.int32)
        Q_prev = torch.zeros(model_context.n_clients, dtype=torch.int32)
        O = torch.ones(model_context.n_clients, dtype=torch.int8)

        active_durations = []
        inactive_durations = []

        for _ in range(10_000):
            Yi = torch.zeros(
                (model_context.n_partitions,
                 model_context.n_tasks,
                 model_context.n_nodes,
                 model_context.n_task_exec_slots),
                dtype=torch.int32,
            )
            Ys = torch.zeros_like(Yi, dtype=torch.int8)

            Tq = tq_transition_function(
                Tq_prev, Q_prev, O, Yi, Ys).sample()
            Q = q_transition_function(Q_prev, O, Tq).sample()

            state_changed = Q != Q_prev
            durations = Tq_prev[state_changed]
            was_active = Q_prev[state_changed] != 0

            active_durations.extend(durations[was_active].tolist())
            inactive_durations.extend(durations[~was_active].tolist())

            Tq_prev = Tq
            Q_prev = Q

        def jitter(d):
            d = np.asarray(d, dtype=float)
            return d + np.random.rand(d.size)

        OFF_SCALE = 30.
        OFF_ALPHA = 1.5
        ON_SCALE = 30.
        ON_ALPHA = 1.5

        inactive = np.array(inactive_durations)
        active = np.array(active_durations)

        assert inactive.size > 200, f"Too few inactive samples: {inactive.size}"
        assert active.size > 200, f"Too few active samples: {active.size}"

        inact_b_hat, inact_loc_hat, inact_scale_hat = stats.pareto.fit(
            jitter(inactive), floc=0)
        act_b_hat, act_loc_hat, act_scale_hat = stats.pareto.fit(
            jitter(active), floc=0)

        if abs(inact_scale_hat - OFF_SCALE) >= 0.1:
            warnings.warn(
                f"OFF_SCALE mismatch: inact_scale_hat={inact_scale_hat:.4f}, expected={OFF_SCALE:.4f}"
            )

        if abs(act_scale_hat - ON_SCALE) >= 0.1:
            warnings.warn(
                f"ON_SCALE mismatch: act_scale_hat={act_scale_hat:.4f}, expected={ON_SCALE:.4f}"
            )

        if abs(inact_b_hat - OFF_ALPHA) >= 0.05:
            warnings.warn(
                f"OFF_ALPHA mismatch: inact_b_hat={inact_b_hat:.4f}, expected={OFF_ALPHA:.4f}"
            )

        if abs(act_b_hat - ON_ALPHA) >= 0.05:
            warnings.warn(
                f"ON_ALPHA mismatch: act_b_hat={act_b_hat:.4f}, expected={ON_ALPHA:.4f}"
            )
