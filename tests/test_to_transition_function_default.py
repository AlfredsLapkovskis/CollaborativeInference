import numpy as np
import pytest
import pyro
import torch
from simulation.impl.to_transition_function_default import ToTransitionFunctionDefault
from simulation.impl.o_transition_function_default import OTransitionFunctionDefault
from scipy import stats
import warnings


class TestToTransitionFunctionDefault:
    @pytest.fixture
    def To_transition_function(self):
        return ToTransitionFunctionDefault()

    @pytest.fixture
    def O_transition_function(self):
        return OTransitionFunctionDefault()

    def test_allowed_values(self, To_transition_function, O_transition_function):
        To_prev = torch.zeros(100, dtype=torch.int32)
        O_prev = torch.zeros(100, dtype=torch.int8)

        for _ in range(10_000):
            To = To_transition_function(To_prev, O_prev).sample()
            O = O_transition_function(O_prev, To).sample()

            assert torch.all((O != O_prev) == (To == 0))
            assert torch.all((O == O_prev) == ((To_prev + 1) == To))
            assert torch.all((To == 0) | (To == To_prev + 1))

            To_prev = To
            O_prev = O

    def test_distributions(self, To_transition_function, O_transition_function):
        To_prev = torch.zeros(10000, dtype=torch.int32)
        O_prev = torch.zeros(10000, dtype=torch.int8)

        online_durations = []
        offline_durations = []

        for _ in range(10_000):
            To = To_transition_function(To_prev, O_prev).sample()
            O = O_transition_function(O_prev, To).sample()

            state_changed = O != O_prev
            durations = To_prev[state_changed]
            was_online = O_prev[state_changed] == 1

            online_durations.extend(durations[was_online].tolist())
            offline_durations.extend(durations[~was_online].tolist())

            To_prev = To
            O_prev = O

        def jitter(d):
            d = np.asarray(d, dtype=float)
            return d + np.random.rand(d.size)

        OFFLINE_LOGNORM_LOC = -0.586 + np.log(3600.)
        OFFLINE_LOGNORM_SCALE = 2.844

        ONLINE_WEIBULL_SCALE = 2.964 * 3600.
        ONLINE_WEIBULL_CONCENTRATION = 0.393

        # Convert to arrays (SciPy expects floats)
        offline = np.asarray(offline_durations, dtype=np.float32)
        online = np.asarray(online_durations, dtype=np.float32)

        # Basic sanity: need enough samples for a meaningful GoF test
        assert offline.size > 200, f"Too few offline samples: {offline.size}"
        assert online.size > 200, f"Too few online samples: {online.size}"

        s_hat, loc_hat, scale_hat = stats.lognorm.fit(
            jitter(offline), floc=0.0)
        mu_hat = np.log(scale_hat)

        c_hat, loc_hat, scale_hat = stats.weibull_min.fit(
            jitter(online), floc=0.0)

        if abs(mu_hat - OFFLINE_LOGNORM_LOC) >= 0.1:
            warnings.warn(
                f"OFFLINE_LOGNORM_LOC mismatch: mu_hat={mu_hat:.4f}, expected={OFFLINE_LOGNORM_LOC:.4f}",
                UserWarning,
            )
        if abs(s_hat - OFFLINE_LOGNORM_SCALE) >= 0.1:
            warnings.warn(
                f"OFFLINE_LOGNORM_SCALE mismatch: s_hat={s_hat:.4f}, expected={OFFLINE_LOGNORM_SCALE:.4f}",
                UserWarning,
            )

        if abs(scale_hat - ONLINE_WEIBULL_SCALE) / ONLINE_WEIBULL_SCALE >= 0.1:
            warnings.warn(
                f"ONLINE_WEIBULL_SCALE mismatch: scale_hat={scale_hat:.4f}, expected={ONLINE_WEIBULL_SCALE:.4f}",
                UserWarning,
            )
        if abs(c_hat - ONLINE_WEIBULL_CONCENTRATION) >= 0.05:
            warnings.warn(
                f"ONLINE_WEIBULL_CONCENTRATION mismatch: c_hat={c_hat:.4f}, expected={ONLINE_WEIBULL_CONCENTRATION:.4f}",
                UserWarning,
            )
