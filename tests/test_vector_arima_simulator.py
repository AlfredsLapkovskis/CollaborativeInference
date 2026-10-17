import pytest
import torch
import warnings

from utils.vector_arima_simulator import VectorArimaSimulator, SimulatorConfig
from utils.yeo_johnson_transform import *


class TestVectorArimaSimulator:
    @pytest.fixture(scope="class", autouse=True)
    def setup(self):
        old_dtype = torch.get_default_dtype()
        # use higher precision for testing to reduce numerical issues
        torch.set_default_dtype(torch.float64)
        yield
        torch.set_default_dtype(old_dtype)

    @pytest.fixture(
        params=[
            {
                "const": 0.20032450746385005,
                "ar.L1": 0.9300496743797745,
                "ma.L1": 0.4739830000183593,
                "sigma2": 0.006728709237234101,
            },
        ],
    )
    def params(self, request):
        return request.param

    @pytest.fixture(
        params=[
            1000,
        ],
    )
    def n_series(self, request):
        return request.param

    @pytest.fixture
    def simulator(self, params, n_series):
        return VectorArimaSimulator(params=params, n_series=n_series)

    def test_simulation(self, simulator: VectorArimaSimulator, params, n_series):
        n_simulations = 1000

        config = SimulatorConfig(
            n_series=n_series,
        )

        self._seed()
        # shape [n_series, n_simulations]
        simulated_result = simulator.simulate(
            n_simulations=n_simulations, config=config)

        self._seed()
        test_result = self._calc_test_series(n_series, n_simulations, params)

        assert torch.allclose(simulated_result, test_result, atol=1e-5)

    def test_several_simulations(self, simulator: VectorArimaSimulator, params, n_series):
        n_simulations_1 = 200
        n_simulations_2 = 500
        n_simulations_3 = 300
        n_simulations = n_simulations_1 + n_simulations_2 + n_simulations_3

        config = SimulatorConfig(
            n_series=n_series,
        )

        self._seed()
        # shape [n_series, n_simulations]
        simulated_result = simulator.simulate(
            n_simulations=n_simulations_1, config=config)
        simulated_result = torch.cat([
            simulated_result,
            simulator.simulate(n_simulations=n_simulations_2, config=config),
        ], dim=1)
        simulated_result = torch.cat([
            simulated_result,
            simulator.simulate(n_simulations=n_simulations_3, config=config),
        ], dim=1)

        self._seed()
        test_result = self._calc_test_series(n_series, n_simulations, params)

        assert torch.allclose(simulated_result, test_result, atol=1e-5)

    def test_simulation_with_config(self, simulator: VectorArimaSimulator, params, n_series):
        n_simulations = 500

        config = SimulatorConfig(
            n_series=n_series,
            mean=torch.repeat_interleave(
                torch.as_tensor([924430037.5181315]), n_series),
            std=torch.repeat_interleave(
                torch.as_tensor([27432989.379645243]), n_series),
            min=torch.repeat_interleave(
                torch.as_tensor([12425717.66311004]), n_series),
            max=torch.repeat_interleave(
                torch.as_tensor([939311038.3779404]), n_series),
            power_transform_lambda=torch.tensor([3.75753751437925]),
        )

        self._seed()
        # shape [n_series, n_simulations]
        simulated_result = simulator.simulate(
            n_simulations=n_simulations, config=config)

        self._seed()
        test_result = self._calc_test_series(
            n_series, n_simulations, params, config)

        assert torch.allclose(simulated_result, test_result, atol=1e-4)

    def test_simulation_with_mask(self, simulator: VectorArimaSimulator, params, n_series):
        n_simulations = 500

        config = SimulatorConfig(
            n_series=n_series,
        )

        online_mask = torch.ones(n_series, dtype=torch.bool)
        online_mask[::2] = False  # half of the series are offline

        self._seed()
        # shape [n_series, n_simulations]
        simulated_result = simulator.simulate(
            n_simulations=n_simulations, mask=online_mask, config=config)

        self._seed()
        test_result = self._calc_test_series(n_series, n_simulations, params)
        test_result[~online_mask, :] = 0.0

        assert torch.allclose(simulated_result, test_result, atol=1e-5)

    def test_statistics(self, simulator: VectorArimaSimulator, params, n_series):
        n_simulations = 10_000

        config = SimulatorConfig(
            n_series=n_series,
            mean=torch.repeat_interleave(
                torch.as_tensor([924430037.5181315]), n_series),
            std=torch.repeat_interleave(
                torch.as_tensor([27432989.379645243]), n_series),
            min=torch.repeat_interleave(
                torch.as_tensor([12425717.66311004]), n_series),
            max=torch.repeat_interleave(
                torch.as_tensor([939311038.3779404]), n_series),
            power_transform_lambda=torch.tensor([3.75753751437925]),
        )

        self._seed()
        # shape [n_series, n_simulations]
        simulated_result = simulator.simulate(
            n_simulations=n_simulations, config=config)

        simulated_means = torch.mean(simulated_result, dim=1)
        simulated_stds = torch.std(simulated_result, dim=1)
        simulated_min = torch.min(simulated_result, dim=1).values
        simulated_max = torch.max(simulated_result, dim=1).values

        standard_errors = 4 * simulated_stds / (n_simulations ** 0.5)

        if ~torch.all(torch.abs(simulated_means -
                                config.mean) <= standard_errors):
            avg_diff = torch.mean(torch.abs(simulated_means - config.mean))
            warnings.warn(
                f"Simulated means differ from expected means more than 4 standard errors (expected difference: {standard_errors.mean()}, average difference: {avg_diff}).")
        if ~torch.all(torch.abs(simulated_stds - config.std)
                      <= standard_errors):
            avg_diff = torch.mean(torch.abs(simulated_stds - config.std))
            warnings.warn(
                f"Simulated stds differ from expected stds more than 4 standard errors (expected difference: {standard_errors.mean()}, average difference: {avg_diff}).")

        assert torch.all(simulated_max <= config.max)
        assert torch.all(simulated_min >= config.min)

    def _calc_test_series(self, n_series, n_steps, params, config: SimulatorConfig = None):
        dtype = torch.get_default_dtype()
        const, ar_coeffs, ma_coeffs, sigma = self._extract_params(params)

        const = const * (1 - sum(ar_coeffs))

        if config is not None and config.mean is not None:
            initial_obs_fill = [config.mean[i].item() for i in range(n_series)]
        else:
            denom = 1.0 - sum(ar_coeffs)
            mu = (const / denom) if denom != 0.0 else 0.0
            initial_obs_fill = [float(mu)] * n_series
        initial_obs_fill = torch.as_tensor(
            initial_obs_fill, dtype=dtype)

        if config is not None:
            if config.min is not None or config.max is not None:
                initial_obs_fill = torch.clamp(
                    initial_obs_fill,
                    min=config.min if config.min is not None else None,
                    max=config.max if config.max is not None else None,
                )

        # Preprocess the fill into normalized/transformed space, mirroring preprocess_obs
        preprocessed_obs_fill = initial_obs_fill.clone()
        if config is not None:
            if config.mean is not None and config.std is not None:
                preprocessed_obs_fill = (
                    preprocessed_obs_fill - config.mean) / config.std
            if config.power_transform is not None:
                preprocessed_obs_fill = power_transform(
                    preprocessed_obs_fill[None, :], config.power_transform
                )[0]

        test_result = [[] for _ in range(n_series)]
        test_eps = [[] for _ in range(n_series)]

        for t in range(n_steps):
            eps_t = torch.normal(mean=0.0, std=sigma, size=(n_series,))

            for series_idx in range(n_series):
                ar_term = 0.0
                for lag, coef in enumerate(ar_coeffs, start=1):
                    if t - lag >= 0:
                        ar_term += coef * test_result[series_idx][t - lag]
                    else:
                        # was: skipped (implicitly 0)
                        ar_term += coef * preprocessed_obs_fill[series_idx]
                ma_term = 0.0
                for lag, coef in enumerate(ma_coeffs, start=1):
                    if t - lag >= 0:
                        ma_term += coef * test_eps[series_idx][t - lag]
                x_t = const + ar_term + ma_term + eps_t[series_idx]

                if config is not None:
                    # Build a single-series transformer for this series_idx so
                    # that power_transform / inv_power_transform receive a tensor
                    # whose D dimension matches the number of lambdas (1, not n_series).
                    if config.power_transform is not None:
                        single_transformer = power_fit(
                            [[0.0]])  # placeholder; lambdas overwritten below
                        single_transformer.lambdas_ = \
                            config.power_transform.lambdas_[
                                series_idx:series_idx + 1]

                    if config.power_transform is not None:
                        x_t = inv_power_transform(
                            x_t[None, None], single_transformer)[0, 0]
                    if config.mean is not None and config.std is not None:
                        x_t = x_t * config.std[series_idx] + \
                            config.mean[series_idx]
                    # 1. Make sure that x_t lie withing min/max bounds
                    if config.min is not None or config.max is not None:
                        x_t = torch.clamp(
                            x_t,
                            min=config.min[series_idx] if config.min is not None else None,
                            max=config.max[series_idx] if config.max is not None else None,
                        )
                    # 2. Transform back for generating the next step
                    if config.mean is not None and config.std is not None:
                        x_t = (x_t - config.mean[series_idx]) / \
                            config.std[series_idx]
                    if config.power_transform is not None:
                        x_t = power_transform(
                            x_t[None, None], single_transformer)[0, 0]

                test_result[series_idx].append(x_t)
                test_eps[series_idx].append(eps_t[series_idx])

        test_result = torch.as_tensor(
            test_result, dtype=dtype)

        if config is not None:
            if config.power_transform is not None:
                test_result = inv_power_transform(
                    test_result.transpose(0, 1), config.power_transform).transpose(0, 1)
            if config.mean is not None and config.std is not None:
                test_result = test_result * \
                    config.std[:, None] + config.mean[:, None]
            if config.min is not None or config.max is not None:
                test_result = torch.clamp(
                    test_result,
                    min=config.min[:,
                                   None] if config.min is not None else None,
                    max=config.max[:,
                                   None] if config.max is not None else None,
                )

        return test_result

    def _seed(self, seed=42):
        torch.manual_seed(seed)

    def _extract_params(self, params):
        const = params["const"]
        ar_coeffs = []
        ma_coeffs = []

        for key, value in params.items():
            if key.startswith("ar.L"):
                lag = int(key.split(".L")[1])
                ar_coeffs.append((lag, value))
            elif key.startswith("ma.L"):
                lag = int(key.split(".L")[1])
                ma_coeffs.append((lag, value))
        ar_coeffs.sort(key=lambda x: x[0])
        ma_coeffs.sort(key=lambda x: x[0])

        ar_coeffs = [coef for _, coef in ar_coeffs]
        ma_coeffs = [coef for _, coef in ma_coeffs]

        sigma = params["sigma2"] ** 0.5

        return const, ar_coeffs, ma_coeffs, sigma
