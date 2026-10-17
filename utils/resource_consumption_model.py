import os
import numpy as np
import statsmodels.api as sm
import json
from scipy.optimize import minimize
import typing
import torch

import utils.yeo_johnson_transform as yeo


# Forward declaration
ResourceConsumptionModel = typing.NewType("ResourceConsumptionModel", None)


class ResourceConsumptionModelTrainer:
    def __init__(
        self,
        series_groups,
        order=(1, 0, 1),
        trend=None,
    ):
        self._order = order
        self._trend = trend
        self._mins = {}
        self._maxs = {}
        self._means = {}
        self._stds = {}

        self._preprocessed_series_array = []

        # self._transformer = yeo.power_fit(
        #     np.concat([np.concat(g) for g in series_groups.values()])[:, None])

        for key, series_group in series_groups.items():
            concat_series = np.concatenate(series_group)

            self._mins[key] = np.min(concat_series)
            self._maxs[key] = np.max(concat_series)
            self._means[key] = mean = np.mean(concat_series)
            self._stds[key] = std = np.std(concat_series)

            for series in series_group:
                series = (series - mean) / std
                # series = yeo.power_transform(
                #     torch.as_tensor(series[:, None]), self._transformer)[:, 0].numpy()

                self._preprocessed_series_array.append(series)

        self._transformer = yeo.power_fit(
            np.concatenate(self._preprocessed_series_array)[:, None])

        for idx, series in enumerate(self._preprocessed_series_array):
            series = yeo.power_transform(
                torch.as_tensor(series[:, None]), self._transformer)[:, 0].numpy()
            self._preprocessed_series_array[idx] = series

        self._config = None

    def fit(self) -> ResourceConsumptionModel:
        if self._config is not None:
            return self._config

        def create_arima(series):
            return sm.tsa.ARIMA(series, order=self._order, trend=self._trend)

        # The example model needs nobs - d > max(3*q, p) observations to
        # estimate ARIMA starting params via conditional sum of squares
        # (statsmodels SARIMAX._conditional_sum_squares), so pick any series
        # long enough instead of assuming the first one qualifies.
        p, d, q = self._order
        min_length = max(3 * q, p) + d + 1
        example_series = next(
            (s for s in self._preprocessed_series_array if len(s) >= min_length),
            self._preprocessed_series_array[0],
        )
        example_model = create_arima(example_series)
        models = [create_arima(series)
                  for series in self._preprocessed_series_array]

        def neg_log_likelihood(params):
            params = example_model.transform_params(params)
            return -sum(m.filter(params).llf for m in models)

        initial_res = example_model.fit()
        initial_params = example_model.untransform_params(initial_res.params)
        param_names = initial_res.param_names

        # TODO: - some params should have bounds (e.g., sigma2 >= 0)!!!!!!!!!!!

        res = minimize(
            neg_log_likelihood,
            initial_params,
            method="L-BFGS-B",
            options={"maxiter": 100, "ftol": 1e-2},
        )
        total_llf = -res.fun
        res_params = example_model.transform_params(res.x)
        sample_model = example_model.filter(res_params)

        self._config = ResourceConsumptionModel(
            model_order=self._order,
            model_trend=self._trend,
            model_param_dict=dict(zip(param_names, res_params)),
            model_aic=sample_model.aic,
            model_bic=sample_model.bic,
            model_llf=sample_model.llf,
            total_llf=total_llf,
            keys=list(self._mins.keys()),
            original_mins=self._mins,
            original_maxs=self._maxs,
            original_means=self._means,
            original_stds=self._stds,
            power_transform_lambda=self._transformer.lambdas_[0],
        )

        return self._config


class ResourceConsumptionModel:
    @property
    def model_order(self):
        return self._model_order

    @property
    def model_trend(self):
        return self._model_trend

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
    def total_llf(self):
        return self._total_llf

    @property
    def keys(self):
        return self._keys

    @property
    def original_mins(self):
        return self._original_mins

    @property
    def original_maxs(self):
        return self._original_maxs

    @property
    def original_means(self):
        return self._original_means

    @property
    def original_stds(self):
        return self._original_stds

    @property
    def power_transform_lambda(self):
        return self._power_transform_lambda

    def __init__(
        self,
        model_order: tuple,
        model_trend: str | None,
        model_param_dict: dict,
        model_aic: float,
        model_bic: float,
        model_llf: float,
        total_llf: float,
        keys: list[tuple],
        original_mins: dict,
        original_maxs: dict,
        original_means: dict,
        original_stds: dict,
        power_transform_lambda: float,
    ):
        self._model_order = model_order
        self._model_trend = model_trend
        self._model_param_dict = model_param_dict
        self._model_aic = model_aic
        self._model_bic = model_bic
        self._model_llf = model_llf
        self._total_llf = total_llf
        self._keys = keys
        self._original_mins = original_mins
        self._original_maxs = original_maxs
        self._original_means = original_means
        self._original_stds = original_stds
        self._power_transform_lambda = power_transform_lambda

    @staticmethod
    def deserialize(serialized_str):
        def try_int(value):
            try:
                return int(value)
            except:
                return value

        def deserialize_key_values(d):
            return {tuple(map(try_int, k.split('_'))): v for k, v in d.items()}

        data = json.loads(serialized_str)
        return ResourceConsumptionModel(
            model_order=tuple(data['model_order']),
            model_trend=data.get('model_trend'),
            model_param_dict=data['model_param_dict'],
            model_aic=data['model_aic'],
            model_bic=data['model_bic'],
            model_llf=data['model_llf'],
            total_llf=data['total_llf'],
            keys=[tuple(k) for k in data['keys']],
            original_mins=deserialize_key_values(data['original_mins']),
            original_maxs=deserialize_key_values(data['original_maxs']),
            original_means=deserialize_key_values(data['original_means']),
            original_stds=deserialize_key_values(data['original_stds']),
            power_transform_lambda=data['power_transform_lambda'],
        )

    @staticmethod
    def load(path):
        with open(path, 'r') as f:
            serialized_str = f.read()
        return ResourceConsumptionModel.deserialize(serialized_str)

    def serialize(self, pretty=False):
        def serialize_key_values(d):
            return {'_'.join(map(str, k)): v for k, v in d.items()}

        return json.dumps({
            'model_order': list(self._model_order),
            'model_trend': self._model_trend,
            'model_param_dict': self._model_param_dict,
            'model_aic': self._model_aic,
            'model_bic': self._model_bic,
            'model_llf': self._model_llf,
            'total_llf': self._total_llf,
            'keys': [list(k) for k in self._keys],
            'original_mins': serialize_key_values(self._original_mins),
            'original_maxs': serialize_key_values(self._original_maxs),
            'original_means': serialize_key_values(self._original_means),
            'original_stds': serialize_key_values(self._original_stds),
            'power_transform_lambda': self._power_transform_lambda,
        }, indent=2 if pretty else None)

    def save(self, path, pretty=False):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            f.write(self.serialize(pretty=pretty))
