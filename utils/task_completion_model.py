import json
import os
import numpy as np
import pandas as pd
import torch
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler, PolynomialFeatures


class TaskCompletionModelTrainer:
    def __init__(
        self,
        data: pd.DataFrame,
        conf_columns: list[str],
        stratification_columns: list[str],
        constant_feature_columns: list[str],
        variable_feature_columns: list[str],
        # Useful workaround to store models with identical set of parameters,
        # while keeping non-zero values only for useful features.
        ignored_feature_columns: list[str] = [],
        standardize: bool = True,
        polynomial_degree: int = 1,
        penalizer: float = 0.0,
        l1_ratio: float = 0.0,
    ):
        data = data.copy()

        # Build person-period format: one row per (task, t), event=1 only at final row.
        dfs = []
        for idx, (_, df) in enumerate(data.groupby(conf_columns)):
            df = df.copy()
            df["id"] = idx
            df["event"] = 0
            df["t"] = range(1, len(df) + 1)
            df.loc[df.index[-1], "event"] = 1
            dfs.append(df)

        df_all = pd.concat(dfs)

        # log(t) captures the baseline hazard shape flexibly with a single term.
        # It is prepended to feature_columns so it participates in polynomial
        # expansions together with the other covariates if desired.
        df_all["log_t"] = np.log(df_all["t"])

        feature_columns = (
            ["log_t"] + constant_feature_columns + variable_feature_columns)

        self._n_constant_features = len(constant_feature_columns)
        self._n_variable_features = len(variable_feature_columns)

        df_all[feature_columns] = df_all[feature_columns].apply(pd.to_numeric)

        self._means = []
        self._stds = []

        if standardize:
            scaler = StandardScaler()
            scaler.fit(df_all[feature_columns])
            df_all[feature_columns] = scaler.transform(df_all[feature_columns])
            self._means = scaler.mean_
            self._stds = scaler.scale_

        self._means = torch.as_tensor(self._means, dtype=torch.float64)
        self._stds = torch.as_tensor(self._stds, dtype=torch.float64)

        assert polynomial_degree >= 1

        self._poly = PolynomialFeatures(
            degree=polynomial_degree, include_bias=False)
        X_poly = self._poly.fit_transform(df_all[feature_columns])
        poly_feature_names = self._poly.get_feature_names_out(feature_columns)
        df_poly = pd.DataFrame(
            X_poly, columns=poly_feature_names, index=df_all.index)

        self._df_all = pd.concat([
            df_all[[*stratification_columns, "event", "t", "id"]],
            df_poly,
        ], axis=1)

        self.poly_feature_names = poly_feature_names.tolist()
        ignored_feature_names = set()
        for col in ignored_feature_columns:
            for fname in poly_feature_names:
                if col in fname:
                    ignored_feature_names.add(fname)

        self.ignored_feature_names = list(ignored_feature_names)
        self.stratification_columns = stratification_columns
        self.constant_feature_columns = constant_feature_columns
        self.variable_feature_columns = variable_feature_columns
        self.ignored_feature_columns = ignored_feature_columns
        self.feature_columns = feature_columns
        self.standardize = standardize
        self.polynomial_degree = polynomial_degree
        self.penalizer = penalizer
        self.l1_ratio = l1_ratio

        self._config = None

    def fit(self):
        if self._config is not None:
            return self._config

        # Identify pure time term indices: involve log_t only, no other covariates.
        log_t_idx = self.feature_columns.index("log_t")
        other_indices = [i for i in range(
            len(self.feature_columns)) if i != log_t_idx]
        powers = self._poly.powers_  # (n_poly_terms, n_base_features)
        pure_time_mask = (
            (powers[:, log_t_idx] > 0) &
            (powers[:, other_indices].sum(axis=1) == 0)
        )
        pure_time_indices = np.where(pure_time_mask)[0]
        pure_time_names = {
            self.poly_feature_names[i] for i in pure_time_indices}

        # Active poly features: exclude ignored and pure time terms.
        # Pure time terms are fully absorbed by stratum x time interactions below.
        active_feature_names = [
            c for c in self.poly_feature_names
            if c not in self.ignored_feature_names
            and c not in pure_time_names
        ]
        active_feature_indices = [
            i for i, c in enumerate(self.poly_feature_names)
            if c not in self.ignored_feature_names
            and c not in pure_time_names
        ]

        # Combine multiple stratification columns into one composite tuple key.
        fitted_df = self._df_all.copy()
        if len(self.stratification_columns) == 0:
            fitted_df["_stratum"] = "_all"
            strata_col = "_stratum"
        elif len(self.stratification_columns) == 1:
            strata_col = self.stratification_columns[0]
        else:
            strata_col = "_stratum"
            fitted_df[strata_col] = list(
                zip(*[fitted_df[c] for c in self.stratification_columns])
            )

        strata_labels = sorted(fitted_df[strata_col].unique(), key=str)

        # K dummies, no drop_first — each dummy is its own stratum intercept.
        # No separate const term; the K dummies span the intercept space fully.
        stratum_dummies = pd.get_dummies(
            fitted_df[strata_col], prefix="_s", drop_first=False, dtype=float
        )
        dummy_names = stratum_dummies.columns.tolist()
        n_dummies = len(dummy_names)

        # Stratum x time interactions: each stratum gets its own time coefficients.
        # Naming: "_s_vit-h-14 log_t", "_s_vit-h-14 log_t^2", etc.
        interaction_names = []
        interaction_arrays = []
        for j, dummy_name in enumerate(dummy_names):
            d = stratum_dummies[dummy_name].to_numpy()
            for i in pure_time_indices:
                interaction_names.append(
                    f"{dummy_name} {self.poly_feature_names[i]}")
                interaction_arrays.append(
                    d * fitted_df[self.poly_feature_names[i]].to_numpy())

        # Build X as numpy to avoid pandas concat copies.
        n_rows = len(fitted_df)
        X_numpy = np.column_stack(
            [stratum_dummies.to_numpy()]
            + ([fitted_df[active_feature_names].to_numpy()]
               if active_feature_names else [])
            + (interaction_arrays if interaction_arrays else [np.empty((n_rows, 0))])
        )
        X_col_names = dummy_names + active_feature_names + interaction_names
        X = pd.DataFrame(X_numpy, columns=X_col_names, index=fitted_df.index)
        y = fitted_df["event"]

        model = sm.GLM(
            endog=y,
            exog=X,
            family=sm.families.Binomial(),
        )

        if self.penalizer > 0:
            # Stratum dummies and stratum x time interactions are structural —
            # not penalized. Only covariate terms are regularized.
            alpha = np.zeros(X.shape[1])
            n_unpenalized = n_dummies + len(interaction_names)
            alpha[n_unpenalized:] = self.penalizer
            result = model.fit_regularized(
                method="elastic_net",
                alpha=alpha,
                L1_wt=self.l1_ratio,
                refit=False,
            )
        else:
            result = model.fit()

        # param order: stratum dummies | non-pure-time poly features | stratum x time interactions
        # Non-pure-time poly features = active features + ignored features (with zero params).
        # Pure time terms are excluded entirely — absorbed by stratum x time interactions.
        non_pure_time_names = [
            c for c in self.poly_feature_names if c not in pure_time_names]
        non_pure_time_indices = [
            i for i, c in enumerate(self.poly_feature_names) if c not in pure_time_names]

        all_param_names = dummy_names + non_pure_time_names + interaction_names
        param_map = dict(zip(X.columns, result.params))
        model_params = torch.zeros(len(all_param_names), dtype=torch.float64)
        for i, name in enumerate(all_param_names):
            if name in param_map:
                model_params[i] = param_map[name]

        # Powers matrix: one row per parameter, same order as all_param_names:
        #   dummy rows | non-pure-time poly rows | interaction rows
        # Column layout: [base feature columns... | dummy columns...]
        n_base = len(self.feature_columns)

        # Dummy rows: power 1 for own dummy column, 0 everywhere else.
        dummy_powers = np.zeros(
            (n_dummies, n_base + n_dummies), dtype=np.int64)
        for j in range(n_dummies):
            dummy_powers[j, n_base + j] = 1

        # Non-pure-time poly rows: includes active and ignored features.
        # Ignored features have zero coefficients but their power rows are present
        # so models with different ignored_feature_columns share the same layout.
        non_pure_time_powers = np.pad(
            powers[non_pure_time_indices], ((0, 0), (0, n_dummies)), constant_values=0)

        # Interaction rows: time_i base powers + dummy column j set to 1.
        interaction_powers = []
        for j in range(n_dummies):
            for i in pure_time_indices:
                row = np.zeros(n_base + n_dummies, dtype=np.int64)
                row[:n_base] = powers[i]
                row[n_base + j] = 1
                interaction_powers.append(row)

        all_powers = np.vstack(
            [dummy_powers, non_pure_time_powers]
            + ([interaction_powers] if interaction_powers else [])
        )

        self._config = TaskCompletionModel(
            stratification_columns=self.stratification_columns,
            constant_feature_columns=self.constant_feature_columns,
            variable_feature_columns=self.variable_feature_columns,
            ignored_feature_columns=self.ignored_feature_columns,
            standardize=self.standardize,
            means=self._means,
            stds=self._stds,
            polynomial_degree=self.polynomial_degree,
            polynomial_powers=torch.as_tensor(all_powers),
            model_strata=strata_labels,
            model_stratum_dummy_names=dummy_names,
            model_param_names=all_param_names,
            model_params=model_params,
            model_aic=result.aic,
        )

        return self._config


class TaskCompletionModel:
    @property
    def stratification_columns(self):
        return self._stratification_columns

    @property
    def constant_feature_columns(self):
        return self._constant_feature_columns

    @property
    def variable_feature_columns(self):
        return self._variable_feature_columns

    @property
    def ignored_feature_columns(self):
        return self._ignored_feature_columns

    @property
    def standardize(self):
        return self._standardize

    @property
    def means(self):
        return self._means

    @property
    def stds(self):
        return self._stds

    @property
    def polynomial_degree(self):
        return self._polynomial_degree

    @property
    def polynomial_powers(self):
        return self._polynomial_powers

    @property
    def model_strata(self):
        return self._model_strata

    @property
    def model_stratum_dummy_names(self):
        return self._model_stratum_dummy_names

    @property
    def model_param_names(self):
        return self._model_param_names

    @property
    def model_params(self):
        return self._model_params

    @property
    def model_aic(self):
        return self._model_aic

    def __init__(
        self,
        stratification_columns: list[str],
        constant_feature_columns: list[str],
        variable_feature_columns: list[str],
        ignored_feature_columns: list[str],
        standardize: bool,
        means: torch.Tensor,
        stds: torch.Tensor,
        polynomial_degree: int,
        polynomial_powers: torch.Tensor,
        model_strata: list,
        model_stratum_dummy_names: list[str],
        model_param_names: list[str],
        model_params: torch.Tensor,
        model_aic: float,
    ):
        self._stratification_columns = stratification_columns
        self._constant_feature_columns = constant_feature_columns
        self._variable_feature_columns = variable_feature_columns
        self._ignored_feature_columns = ignored_feature_columns
        self._standardize = standardize
        self._means = means
        self._stds = stds
        self._polynomial_degree = polynomial_degree
        self._polynomial_powers = polynomial_powers
        self._model_strata = model_strata
        self._model_stratum_dummy_names = model_stratum_dummy_names
        self._model_param_names = model_param_names
        self._model_params = model_params
        self._model_aic = model_aic

    @staticmethod
    def deserialize(serialized_str, device=None):
        data = json.loads(serialized_str)
        return TaskCompletionModel(
            stratification_columns=data["stratification_columns"],
            constant_feature_columns=data["constant_feature_columns"],
            variable_feature_columns=data["variable_feature_columns"],
            ignored_feature_columns=data["ignored_feature_columns"],
            standardize=data["standardize"],
            means=torch.as_tensor(data["means"], device=device),
            stds=torch.as_tensor(data["stds"], device=device),
            polynomial_degree=data["polynomial_degree"],
            polynomial_powers=torch.as_tensor(
                data["polynomial_powers"], device=device),
            model_strata=[tuple(s) if isinstance(s, list) else s
                          for s in data["model_strata"]],
            model_stratum_dummy_names=data["model_stratum_dummy_names"],
            model_param_names=data["model_param_names"],
            model_params=torch.as_tensor(data["model_params"], device=device),
            model_aic=data["model_aic"],
        )

    @staticmethod
    def load(path, device=None):
        with open(path, "r") as f:
            serialized_str = f.read()
        return TaskCompletionModel.deserialize(serialized_str, device=device)

    def serialize(self, pretty=False):
        return json.dumps({
            "stratification_columns": self._stratification_columns,
            "constant_feature_columns": self._constant_feature_columns,
            "variable_feature_columns": self._variable_feature_columns,
            "ignored_feature_columns": self._ignored_feature_columns,
            "standardize": self.standardize,
            "means": self.means.tolist(),
            "stds": self.stds.tolist(),
            "polynomial_degree": self.polynomial_degree,
            "polynomial_powers": self.polynomial_powers.tolist(),
            "model_strata": [list(s) if isinstance(s, tuple) else s
                             for s in self.model_strata],
            "model_stratum_dummy_names": self._model_stratum_dummy_names,
            "model_param_names": self._model_param_names,
            "model_params": self.model_params.tolist(),
            "model_aic": self.model_aic,
        }, indent=2 if pretty else None)

    def save(self, path, pretty=False):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(self.serialize(pretty=pretty))
