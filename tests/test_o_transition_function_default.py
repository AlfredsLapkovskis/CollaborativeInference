import pytest
import torch
from simulation.impl.o_transition_function_default import OTransitionFunctionDefault


class TestOTransitionFunctionDefault:
    @pytest.fixture
    def transition_function(self):
        return OTransitionFunctionDefault()

    def test_online_to_offline(self, transition_function):
        O_prev = torch.tensor(1, dtype=torch.int8)
        To_curr = torch.tensor(0, dtype=torch.int32)

        result = transition_function(O_prev, To_curr)

        sample = result.sample()

        expected = torch.tensor(0, dtype=torch.int8)
        assert torch.equal(sample, expected)

    def test_online_to_online(self, transition_function):
        O_prev = torch.tensor(1, dtype=torch.int8)
        To_curr = torch.tensor(7, dtype=torch.int32)

        result = transition_function(O_prev, To_curr)

        sample = result.sample()

        expected = torch.tensor(1, dtype=torch.int8)
        assert torch.equal(sample, expected)

    def test_offline_to_offline(self, transition_function):
        O_prev = torch.tensor(0, dtype=torch.int8)
        To_curr = torch.tensor(34, dtype=torch.int32)

        result = transition_function(O_prev, To_curr)

        sample = result.sample()

        expected = torch.tensor(0, dtype=torch.int8)
        assert torch.equal(sample, expected)

    def test_offline_to_online(self, transition_function):
        O_prev = torch.tensor(0, dtype=torch.int8)
        To_curr = torch.tensor(0, dtype=torch.int32)

        result = transition_function(O_prev, To_curr)

        sample = result.sample()

        expected = torch.tensor(1, dtype=torch.int8)
        assert torch.equal(sample, expected)

    def test_vector(self, transition_function):
        O_prev = torch.tensor(
            [1, 0, 1, 0, 1, 0, 1, 1,  1,  0], dtype=torch.int8)
        To_curr = torch.tensor(
            [0, 0, 1, 0, 0, 0, 3, 10, 11, 0], dtype=torch.int32)

        result = transition_function(O_prev, To_curr)

        sample = result.sample()

        expected = torch.tensor(
            [0, 1, 1, 1, 0, 1, 1, 1, 1, 1], dtype=torch.int8)
        assert torch.equal(sample, expected)
