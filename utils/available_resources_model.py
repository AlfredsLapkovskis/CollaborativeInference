import os
import statsmodels.api as sm
import json
import typing
from scipy.optimize import minimize
import torch
import numpy as np

import utils.yeo_johnson_transform as yeo
import utils.quantization as quant


# Forward declaration
AvailableResourcesModel = typing.NewType("AvailableResourcesModel", None)


class AvailableResourcesModelTrainer:

    _DISCRETE_LEVELS = 10

    def __init__(
        self,
        series,
        order=(1, 0, 1),
    ):
        self._order = order
        self._min = np.min(series)
        self._max = np.max(series)
        self._mean = np.mean(series)
        self._std = np.std(series)
        self._zero_level = (0. - self._mean) / self._std

        series = (series - self._mean) / self._std
        self._z_series = series

        self._transformer = yeo.power_fit(series[:, None])

        series = yeo.power_transform(torch.as_tensor(
            series[:, None]), self._transformer)[:, 0].numpy()

        self._model = sm.tsa.ARIMA(series, order=order)
        self._config = None

    def fit(self) -> AvailableResourcesModel:
        if self._config is not None:
            return self._config

        z_discrete_levels, z_discrete_thresholds, z_discrete_mse = quant.lloyd_max(
            self._z_series, self._DISCRETE_LEVELS - 1,
            standardize=False,
        )
        zero = np.atleast_1d(self._zero_level)
        z_discrete_levels = np.concatenate([zero, z_discrete_levels])
        z_discrete_thresholds = np.concatenate([zero, z_discrete_thresholds])

        # params, mse = quant.get_optimal_reconstruction_values(
        # self._z_series, z_discrete_thresholds, min_val=self._zero_level, max_val=(self._max - self._mean) / self._std)

        res = self._model.fit()

        self._config = AvailableResourcesModel(
            model_order=self._order,
            model_param_dict=dict(zip(res.param_names, res.params)),
            model_aic=res.aic,
            model_bic=res.bic,
            model_llf=res.llf,
            z_discrete_levels=z_discrete_levels,
            z_discrete_thresholds=z_discrete_thresholds,
            z_discrete_mse=z_discrete_mse,
            original_min=self._min,
            original_max=self._max,
            original_mean=self._mean,
            original_std=self._std,
            power_transform_lambda=self._transformer.lambdas_[0],
        )

        return self._config


class AvailableResourcesModel:
    @property
    def model_order(self):
        return self._model_order

    @property
    def model_param_dict(self):
        return self._model_param_dict

    @property
    def model_aic(self):
        return self._model_aic

    @property
    def model_bic(self):
        return self._model_bic

    @property
    def model_llf(self):
        return self._model_llf

    @property
    def z_discrete_levels(self):
        return self._z_discrete_levels

    @property
    def z_discrete_thresholds(self):
        return self._z_discrete_thresholds

    @property
    def discrete_mse(self):
        return self._z_discrete_mse

    @property
    def original_min(self):
        return self._original_min

    @property
    def original_max(self):
        return self._original_max

    @property
    def original_mean(self):
        return self._original_mean

    @property
    def original_std(self):
        return self._original_std

    @property
    def power_transform_lambda(self):
        return self._power_transform_lambda

    def __init__(
        self,
        model_order: tuple,
        model_param_dict: dict,
        model_aic: float,
        model_bic: float,
        model_llf: float,
        z_discrete_levels: np.ndarray,
        z_discrete_thresholds: np.ndarray,
        z_discrete_mse: float,
        original_min: float,
        original_max: float,
        original_mean: float,
        original_std: float,
        power_transform_lambda: float,
    ):
        self._model_order = model_order
        self._model_param_dict = model_param_dict
        self._model_aic = model_aic
        self._model_bic = model_bic
        self._model_llf = model_llf
        self._z_discrete_levels = z_discrete_levels
        self._z_discrete_thresholds = z_discrete_thresholds
        self._z_discrete_mse = z_discrete_mse
        self._original_min = original_min
        self._original_max = original_max
        self._original_mean = original_mean
        self._original_std = original_std
        self._power_transform_lambda = power_transform_lambda

    @staticmethod
    def deserialize(serialized_str):
        data = json.loads(serialized_str)
        return AvailableResourcesModel(
            model_order=tuple(data['model_order']),
            model_param_dict=data['model_param_dict'],
            model_aic=data['model_aic'],
            model_bic=data['model_bic'],
            model_llf=data['model_llf'],
            z_discrete_levels=np.array(data['z_discrete_levels']),
            z_discrete_thresholds=np.array(data['z_discrete_thresholds']),
            z_discrete_mse=data['z_discrete_mse'],
            original_min=data['original_min'],
            original_max=data['original_max'],
            original_mean=data['original_mean'],
            original_std=data['original_std'],
            power_transform_lambda=data['power_transform_lambda'],
        )

    @staticmethod
    def load(path):
        with open(path, 'r') as f:
            serialized_str = f.read()
        return AvailableResourcesModel.deserialize(serialized_str)

    def serialize(self, pretty=False):
        return json.dumps({
            'model_order': list(self._model_order),
            'model_param_dict': self._model_param_dict,
            'model_aic': self._model_aic,
            'model_bic': self._model_bic,
            'model_llf': self._model_llf,
            'z_discrete_levels': self._z_discrete_levels.tolist(),
            'z_discrete_thresholds': self._z_discrete_thresholds.tolist(),
            'z_discrete_mse': self._z_discrete_mse,
            'original_min': self._original_min,
            'original_max': self._original_max,
            'original_mean': self._original_mean,
            'original_std': self._original_std,
            'power_transform_lambda': self._power_transform_lambda,
        }, indent=2 if pretty else None)

    def save(self, path, pretty=False):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            f.write(self.serialize(pretty=pretty))
