import re
import torch

from utils.yeo_johnson_transform import *
from utils.quantization import *


class VectorArimaSimulator:
    def __init__(
        self,
        params,
        n_series,
    ):
        self.n_series = n_series
        self.const = 0.
        self.sigma = 0.

        phis = {}
        thetas = {}

        ar_prefix = "ar.L"
        ma_prefix = "ma.L"
        const_prefix = "const"
        sigma2_prefix = "sigma2"

        for name, value in params.items():
            if name.startswith(ar_prefix):
                idx = int(name[len(ar_prefix):])
                phis[idx] = value
            elif name.startswith(ma_prefix):
                idx = int(name[len(ma_prefix):])
                thetas[idx] = value
            elif name.startswith(const_prefix):
                self.const = torch.as_tensor(value)
            elif name.startswith(sigma2_prefix):
                self.sigma = torch.sqrt(torch.as_tensor(value))
            else:
                raise ValueError(f"Unexpected parameter name: {name}")

        def sort_and_get_values(value_dict):
            return torch.as_tensor([item[1] for item in sorted(value_dict.items(), key=lambda x: x[0])])

        self.phis = sort_and_get_values(phis)
        self.thetas = sort_and_get_values(thetas)

        self.const = self.const * (1 - self.phis.sum())

        self.prev_obs = torch.full((n_series, len(self.phis)), torch.nan)
        self.prev_eps = torch.full((n_series, len(self.thetas)), torch.nan)
        # self.prev_obs = np.zeros((n_series, len(self.phis)))
        # self.prev_eps = np.zeros((n_series, len(self.thetas)))

    def simulate(self, n_simulations, mask=None, config: "SimulatorConfig" = None):
        series = torch.zeros((self.n_series, n_simulations))

        def _initial_obs_fill_values():
            # Fill value for missing AR history in observation space.
            # Prefer per-series mean when available; otherwise use stationary mean.
            if config is not None and config.mean is not None:
                fill = config.mean.clone().to(series.dtype)
            else:
                if self.phis.numel() > 0:
                    denom = 1.0 - self.phis.sum()
                    mu = torch.nan_to_num(
                        self.const / denom,
                        nan=0.0,
                        posinf=0.0,
                        neginf=0.0,
                    )
                else:
                    mu = self.const
                fill = torch.full((self.n_series,),
                                  float(mu), dtype=series.dtype)

            if config is not None:
                if config.min is not None or config.max is not None:
                    fill = torch.clamp(
                        fill,
                        min=config.min if config.min is not None else None,
                        max=config.max if config.max is not None else None,
                    )
                if config.discrete_thresholds is not None:
                    fill = vector_digitize(
                        fill[:, None], config.discrete_thresholds)[:, 0].to(series.dtype)

            return fill

        def _initial_eps_fill_values():
            # Fill value for missing MA innovations history.
            fill = torch.zeros((self.n_series,), dtype=series.dtype)
            if config is not None and config.eps_discrete_thresholds is not None:
                fill = torch.bucketize(
                    fill, boundaries=config.eps_discrete_thresholds, right=True).to(series.dtype)
            return fill

        initial_obs_fill = _initial_obs_fill_values()
        initial_eps_fill = _initial_eps_fill_values()

        # TODO: For each observation, preprocessing should use the config which
        # was relevant at the time of generating this observation.

        def preprocess_obs(obs):
            obs = obs.clone()
            nan_mask = torch.isnan(obs)
            if nan_mask.any():
                # Use torch.where instead of boolean-index assignment to avoid
                # materializing a full (n_series, n_lags) expanded tensor, which
                # caused OOM when n_series is large.
                obs = torch.where(nan_mask, initial_obs_fill[:, None], obs)

            if config:
                if config.discrete_levels is not None:
                    # TODO: - how about the case when all levels are the same for each series?
                    obs = stochastic_vector_restore(
                        obs.to(int), config.discrete_thresholds, min_val=config.min, max_val=config.max,
                    )
                if config.min is not None or config.max is not None:
                    obs = torch.clamp(
                        obs,
                        min=config.min[:,
                                       None] if config.min is not None else None,
                        max=config.max[:,
                                       None] if config.max is not None else None,
                    )
                if config.mean is not None and config.std is not None:
                    obs = (obs - config.mean[:, None]) / config.std[:, None]
                if config.power_transform is not None:
                    x = obs.transpose(0, 1)  # .detach().cpu().numpy()
                    x = power_transform(x, config.power_transform)
                    x = x.transpose(0, 1)
                    obs = x
            return obs

        def preprocess_eps(eps):
            eps = eps.clone()
            nan_mask = torch.isnan(eps)
            if nan_mask.any():
                # Same fix as preprocess_obs: avoid materializing expanded tensor.
                eps = torch.where(nan_mask, initial_eps_fill[:, None], eps)

            if config:
                if config.eps_discrete_levels is not None:
                    eps = config.eps_discrete_levels[eps.to(int)]
            return eps

        def process_obs(obs):
            if config:
                if config.power_transform is not None:
                    x = obs[None, :]  # .detach().cpu().numpy()
                    x = inv_power_transform(x, config.power_transform)
                    obs = x[0]
                if config.mean is not None and config.std is not None:
                    obs = obs * config.std + \
                        config.mean
                if config.min is not None or config.max is not None:
                    obs = torch.clamp(
                        obs,
                        min=config.min if config.min is not None else None,
                        max=config.max if config.max is not None else None,
                    )
                if config.discrete_thresholds is not None:
                    # TODO: - how about the case when all levels are the same for each series?
                    obs = vector_digitize(
                        obs[:, None], config.discrete_thresholds)[:, 0]
            return obs

        def process_eps(eps):
            if config:
                if config.eps_discrete_thresholds is not None:
                    eps = torch.bucketize(
                        eps, boundaries=config.eps_discrete_thresholds, right=True)
            return eps

        for t in range(n_simulations):
            eps = torch.normal(
                mean=0., std=self.sigma, size=(self.n_series,), device=torch.get_default_device())
            obs = self.const + \
                preprocess_obs(self.prev_obs) @ self.phis + \
                preprocess_eps(self.prev_eps) @ self.thetas + eps

            eps = process_eps(eps)
            obs = process_obs(obs)

            if self.prev_obs.numel() > 0:
                self.prev_obs[:, 1:] = self.prev_obs[:, :-1].clone()
                self.prev_obs[:, 0] = obs
            if self.prev_eps.numel() > 0:
                self.prev_eps[:, 1:] = self.prev_eps[:, :-1].clone()
                self.prev_eps[:, 0] = eps

            series[:, t] = obs

            if mask is not None:
                self.prev_obs[~mask, :] = torch.nan
                self.prev_eps[~mask, :] = torch.nan
                series[~mask, t] = 0

        return series


class SimulatorConfig:
    @property
    def n_series(self):
        return self._n_series

    @property
    def discrete_levels(self):
        return self._discrete_levels

    @property
    def discrete_thresholds(self):
        return self._discrete_thresholds

    @property
    def eps_discrete_levels(self):
        return self._eps_discrete_levels

    @property
    def eps_discrete_thresholds(self):
        return self._eps_discrete_thresholds

    @property
    def min(self):
        return self._min

    @property
    def max(self):
        return self._max

    @property
    def mean(self):
        return self._mean

    @property
    def std(self):
        return self._std

    @property
    def power_transform(self):
        return self._power_transform

    def __init__(
        self,
        n_series=None,
        discrete_levels=None,  # Must have shape (n_series, k)
        discrete_thresholds=None,  # Must have shape (n_series, k - 1)
        eps_discrete_levels=None,  # Must have shape (m,)
        eps_discrete_thresholds=None,  # Must have shape (m - 1,)
        min=None,  # Must have shape (n_series,)
        max=None,  # Must have shape (n_series,)
        mean=None,  # Must have shape (n_series,)
        std=None,  # Must have shape (n_series,)
        power_transform_lambda=None,  # Must have shape (1,) or (n_series,)
    ):
        # TODO: revise assertions
        # assert ((mean is None) == (std is None))
        # assert ((mean is None) or n_series is not None)
        # assert ((mean is None) or (power_transform_lambda is not None))
        # assert ((discrete_levels is None) == (discrete_thresholds is None) == (
        #     eps_discrete_levels is None) == (eps_discrete_thresholds is None))
        # assert ((discrete_levels is None) or ((discrete_levels.shape[0] == n_series) and (discrete_thresholds.shape[0] == n_series) and (
        #     discrete_levels.shape[1] - 1 == discrete_thresholds.shape[1])))
        # assert ((eps_discrete_levels is None) or (
        #     eps_discrete_levels.shape[0] - 1 == eps_discrete_thresholds.shape[0]))
        # assert ((discrete_levels is None) or (mean is not None))

        self._n_series = n_series
        self._discrete_levels = torch.as_tensor(
            discrete_levels) if discrete_levels is not None else None
        self._discrete_thresholds = torch.as_tensor(
            discrete_thresholds) if discrete_thresholds is not None else None
        self._eps_discrete_levels = torch.as_tensor(
            eps_discrete_levels) if eps_discrete_levels is not None else None
        self._eps_discrete_thresholds = torch.as_tensor(
            eps_discrete_thresholds) if eps_discrete_thresholds is not None else None
        self._min = min
        self._max = max
        self._mean = mean
        self._std = std
        self._setup_power_transform(power_transform_lambda)

    def update(self, **kwargs):
        if 'n_series' in kwargs:
            self._n_series = kwargs['n_series']
        if 'discrete_levels' in kwargs:
            self._discrete_levels = kwargs['discrete_levels']
            if self._discrete_levels is not None:
                self._discrete_levels = torch.as_tensor(self._discrete_levels)
        if 'discrete_thresholds' in kwargs:
            self._discrete_thresholds = kwargs['discrete_thresholds']
            if self._discrete_thresholds is not None:
                self._discrete_thresholds = torch.as_tensor(
                    self._discrete_thresholds)
        if 'eps_discrete_levels' in kwargs:
            self._eps_discrete_levels = kwargs['eps_discrete_levels']
            if self._eps_discrete_levels is not None:
                self._eps_discrete_levels = torch.as_tensor(
                    self._eps_discrete_levels)
        if 'eps_discrete_thresholds' in kwargs:
            self._eps_discrete_thresholds = kwargs['eps_discrete_thresholds']
            if self._eps_discrete_thresholds is not None:
                self._eps_discrete_thresholds = torch.as_tensor(
                    self._eps_discrete_thresholds)
        if 'min' in kwargs:
            self._min = torch.as_tensor(kwargs['min'])
        if 'max' in kwargs:
            self._max = torch.as_tensor(kwargs['max'])
        if 'mean' in kwargs:
            self._mean = torch.as_tensor(kwargs['mean'])
        if 'std' in kwargs:
            self._std = torch.as_tensor(kwargs['std'])
        if 'power_transform_lambda' in kwargs:
            power_transform_lambda = kwargs['power_transform_lambda']
            self._setup_power_transform(power_transform_lambda)

    def _setup_power_transform(self, power_transform_lambda):
        assert self._n_series is not None

        if power_transform_lambda is not None:
            self._power_transform = power_fit(np.zeros((1, self._n_series)))
            power_transform_lambda = np.atleast_1d(power_transform_lambda)
            if power_transform_lambda.shape == (1,):
                power_transform_lambda = np.full(
                    (self._n_series,), power_transform_lambda)
            self._power_transform.lambdas_ = power_transform_lambda
        else:
            self._power_transform = None
