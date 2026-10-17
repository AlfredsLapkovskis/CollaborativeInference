import torch
from utils.task_completion_model import TaskCompletionModel


class VectorPLRSimulator:
    def __init__(
        self,
        task_completion_models: list[TaskCompletionModel],
    ):
        self._models = task_completion_models
        self._example_model = self._models[0]

        # Assert all models have the same properties
        assert len(self._models) > 0

        assert all(m.stratification_columns ==
                   self._models[0].stratification_columns for m in self._models)
        assert all(m.variable_feature_columns ==
                   self._models[0].variable_feature_columns for m in self._models)
        assert all(m.constant_feature_columns ==
                   self._models[0].constant_feature_columns for m in self._models)
        assert all(m.standardize ==
                   self._models[0].standardize for m in self._models)
        # assert all(torch.equal(m.means, self._models[0].means) for m in self._models)
        # assert all(torch.equal(m.stds, self._models[0].stds) for m in self._models)
        assert all(m.polynomial_degree ==
                   self._models[0].polynomial_degree for m in self._models)
        assert all(m.polynomial_powers.shape ==
                   self._models[0].polynomial_powers.shape for m in self._models)
        assert all(m.model_strata ==
                   self._models[0].model_strata for m in self._models)
        assert all(m.model_param_names ==
                   self._models[0].model_param_names for m in self._models)

        assert len(self._example_model.means) == len(self._example_model.stds)

        # Number of base features (log_t + covariates), before dummy columns.
        # polynomial_powers has shape [n_params, n_base + n_dummies], so:
        self._n_dummies = len(self._example_model.model_stratum_dummy_names)
        self._n_strata = len(self._example_model.model_strata)
        self._n_base = self._example_model.polynomial_powers.shape[1] - \
            self._n_dummies

        # Cache per-model tensors once; these are reused at every timestep.
        self._all_powers = torch.stack(
            [m.polynomial_powers for m in self._models])
        self._all_params = torch.stack([m.model_params for m in self._models])
        self._all_means = torch.stack([m.means for m in self._models])
        self._all_stds = torch.stack([m.stds for m in self._models])
        self._strata_eye = torch.eye(
            self._n_strata,
            dtype=self._all_params.dtype,
            device=self._all_params.device,
        )
        self._sample_chunk_size = 8192

    def _compute_poly_features_chunk(
        self,
        features: torch.Tensor,
        powers: torch.Tensor,
    ) -> torch.Tensor:
        # Compute polynomial terms without materializing [n_samples, n_params, n_features].
        n_features, n_samples = features.shape
        n_params = powers.shape[0]

        poly_features = torch.ones(
            (n_samples, n_params),
            dtype=features.dtype,
            device=features.device,
        )

        for feature_idx in range(n_features):
            exponents = powers[:, feature_idx]
            non_zero = exponents != 0
            if not torch.any(non_zero):
                continue

            poly_features[:, non_zero] = poly_features[:, non_zero] * (
                features[feature_idx].unsqueeze(
                    1) ** exponents[non_zero].unsqueeze(0)
            )

        return poly_features

    def _compute_linear_predictor(
        self,
        features: torch.Tensor,
        model_idx: torch.Tensor,
    ) -> torch.Tensor:
        linear_predictor = torch.empty(
            (features.shape[1],),
            dtype=self._all_params.dtype,
            device=features.device,
        )

        for model_id in torch.unique(model_idx):
            model_id_int = int(model_id.item())
            sample_indices = (model_idx == model_id).nonzero(
                as_tuple=False).squeeze(1)

            powers = self._all_powers[model_id_int]
            params = self._all_params[model_id_int]

            for start in range(0, sample_indices.numel(), self._sample_chunk_size):
                chunk_indices = sample_indices[start:start +
                                               self._sample_chunk_size]
                chunk_features = features[:, chunk_indices]
                poly_features = self._compute_poly_features_chunk(
                    chunk_features, powers)
                linear_predictor[chunk_indices] = torch.sum(
                    poly_features * params.unsqueeze(0),
                    dim=1,
                )

        return linear_predictor

    def simulate(
        self,
        const_covariates: torch.tensor,
        var_covariates: torch.tensor,
        time: torch.tensor,
        model_idx: torch.tensor,
        strata_idx: torch.tensor,
        mask: torch.tensor,
    ) -> torch.tensor:
        # covariates shape: [n_features, n_samples]
        device = self._all_params.device
        model_idx = model_idx.long().to(device)
        strata_idx = strata_idx.long().to(device)
        time = time.to(device)
        mask = mask.to(device)

        active_mask = mask & (time > 0)
        survival_probabilities = torch.ones(
            time.shape,
            dtype=self._all_params.dtype,
            device=device,
        )
        if not torch.any(active_mask):
            return survival_probabilities

        active_indices = active_mask.nonzero(as_tuple=False).squeeze(1)
        const_covariates = const_covariates[:, active_indices.to(
            const_covariates.device)].to(device)
        var_covariates = var_covariates[:, active_indices.to(
            var_covariates.device)].to(device)
        time = time[active_indices]
        model_idx = model_idx[active_indices]
        strata_idx = strata_idx[active_indices]

        # Build base features [log_t, const_covariates, var_covariates],
        # shape: [n_base, n_samples]
        base_features = torch.cat([const_covariates, var_covariates])
        if base_features.shape[0] + 1 == self._n_base:
            # log_t not yet included — prepend it
            # t <= 1 is handled by masking survival_probabilities to 1
            log_t = torch.log(time.clamp(min=1).to(
                base_features.dtype))[None, :]
            base_features = torch.cat([log_t, base_features])
        else:
            assert base_features.shape[0] == self._n_base, (
                f"Expected {self._n_base} base features, got {base_features.shape[0]}")

        # Standardize base features per model (means/stds cover base features only)
        if self._example_model.standardize:
            base_features = (
                base_features - self._all_means[model_idx].T) / self._all_stds[model_idx].T

        # Append one-hot stratum dummy columns so the full feature vector matches
        # the column layout of polynomial_powers: [base features | dummy columns]
        strata_dummies = self._strata_eye[strata_idx].T.to(
            base_features.dtype)                           # [n_strata, n_samples]
        # [n_base+n_strata, n_samples]
        features = torch.cat([base_features, strata_dummies])

        linear_predictor = self._compute_linear_predictor(features, model_idx)

        event_probabilities = torch.sigmoid(linear_predictor)
        survival_probabilities[active_indices] = 1. - event_probabilities

        return survival_probabilities
