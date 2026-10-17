import pytest
import torch
import warnings

from simulation.model_context import ModelContext
from simulation.impl.q_transition_function_default import QTransitionFunctionDefault
from simulation.impl.tq_transition_function_default import TqTransitionFunctionDefault
from simulation.types import *
from simulation.variable_names import Tq


class TestQTransitionFunctionDefault:
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

    def test_switching(self, model_context: ModelContext, q_transition_function):
        online_clients = model_context.n_clients // 2
        offline_clients = model_context.n_clients - online_clients

        O = torch.ones(model_context.n_clients, dtype=torch.int8)
        O[:offline_clients] = 0

        Tq = torch.zeros(model_context.n_clients, dtype=torch.int32)
        Q = torch.zeros(model_context.n_clients, dtype=torch.int32)

        for _ in range(10_000):
            result = q_transition_function(Q, O, Tq)
            sample = result.sample()

            assert torch.all(sample[:offline_clients] == 0)
            assert torch.all(sample[offline_clients:] > 0)

        Tq = torch.ones(model_context.n_clients, dtype=torch.int32)

        for _ in range(10_000):
            result = q_transition_function(Q, O, Tq)
            sample = result.sample()

            assert torch.all(sample == 0)

        Q = torch.randint(
            low=1,
            high=model_context.n_tasks + 1,
            size=(model_context.n_clients,),
            dtype=torch.int32,
        )

        for _ in range(10_000):
            result = q_transition_function(Q, O, Tq)
            sample = result.sample()

            assert torch.all(sample[:offline_clients] == 0)
            assert torch.all(sample[offline_clients:] == Q[offline_clients:])

    def test_finished(self, model_context: ModelContext, q_transition_function, tq_transition_function):
        O = torch.ones(model_context.n_clients, dtype=torch.int8)

        Tq_prev = torch.ones(model_context.n_clients, dtype=torch.int32)

        Q = torch.randint(
            low=1,
            high=model_context.n_tasks + 1,
            size=(model_context.n_clients,),
            dtype=torch.int32,
        )

        for _ in range(10_000):
            Yi = torch.zeros(
                (model_context.n_partitions,
                 model_context.n_tasks,
                 model_context.n_nodes,
                 model_context.n_task_exec_slots),
                dtype=torch.int32,
            )
            Ys = torch.zeros_like(Yi, dtype=torch.int8)

            all_indices = Q.nonzero().to(torch.int32)
            random_indices = torch.randperm(len(all_indices))[:10]

            for idx in random_indices:
                task = Q[idx] - 1
                for part in range(model_context.n_partitions):
                    if model_context.partition_mask[part, task]:
                        while True:
                            node = torch.randint(
                                0, model_context.n_nodes, (1,)).item()
                            slot = torch.randint(
                                0, model_context.n_task_exec_slots, (1,)).item()

                            if Yi[part, task, node, slot] == 0:
                                Yi[part, task, node, slot] = idx + 1
                                Ys[part, task, node, slot] = YState.DONE
                                break

            Tq = tq_transition_function(Tq_prev, Q, O, Yi, Ys).sample()
            result = q_transition_function(Q, O, Tq)
            sample = result.sample()

            assert torch.all(sample[random_indices] == 0)

            mask = torch.ones(
                model_context.n_clients, dtype=torch.bool)
            mask[random_indices] = False

            assert torch.all(sample[mask] == Q[mask])

    def test_query_distribution(self, model_context: ModelContext, q_transition_function):
        O = torch.ones(model_context.n_clients, dtype=torch.int8)
        Tq = torch.zeros(model_context.n_clients, dtype=torch.int32)
        Q = torch.zeros(model_context.n_clients, dtype=torch.int32)

        query_counts = torch.zeros(
            model_context.n_tasks, dtype=torch.int32)

        n_samples = 100_000

        for _ in range(n_samples):
            result = q_transition_function(Q, O, Tq)
            sample = result.sample()

            assert torch.all(sample > 0)

            tasks, counts = torch.unique(sample, return_counts=True)

            query_counts[tasks - 1] += counts

        empirical_probs = query_counts.float() / query_counts.sum().float()

        expected_probs = model_context.task_query_probabilities

        if not torch.allclose(empirical_probs, expected_probs, atol=0.01):
            warnings.warn(
                f"Empirical query probabilities {empirical_probs.numpy()} differ from expected {expected_probs.numpy()}"
            )
