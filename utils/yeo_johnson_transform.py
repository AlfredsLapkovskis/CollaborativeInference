# import numpy as np
import torch

from sklearn.preprocessing import PowerTransformer


def power_fit(series):
    transformer = PowerTransformer("yeo-johnson", standardize=False)
    return transformer.fit(series)


def power_transform(series: torch.Tensor, transformer: PowerTransformer):
    # Normalize to 2-D so masks work uniformly; squeeze back at end.
    squeezed = series.dim() == 1
    if squeezed:
        series = series.unsqueeze(0)

    # lambdas: (D,) on same device/dtype as series
    lambdas = torch.as_tensor(
        transformer.lambdas_,
        dtype=series.dtype,
        device=series.device,
    )  # shape: (D,)

    eps = torch.finfo(series.dtype).eps

    # Lambda masks are (D,) — constant across rows, so index columns not elements.
    # This avoids large (N, D) boolean-index gathers when N (n_series) is huge.
    lambda_lt_eps = torch.abs(lambdas) < eps      # (D,)
    lambda_gt_2 = torch.abs(lambdas - 2) > eps  # (D,)

    out = torch.empty_like(series)

    # Columns where λ ≈ 0: pos → log1p(x), neg → -log1p(-x)
    col = lambda_lt_eps.nonzero(as_tuple=True)[0]
    if col.numel() > 0:
        s = series[:, col]
        out[:, col] = torch.where(s >= 0, torch.log1p(s), -torch.log1p(-s))

    # Columns where λ != 0 and λ <= 2: pos → ((x+1)^λ-1)/λ, neg → -log1p(-x)
    col = (~lambda_lt_eps & ~lambda_gt_2).nonzero(as_tuple=True)[0]
    if col.numel() > 0:
        s = series[:, col]
        l = lambdas[col]
        pos_mask = s >= 0
        # Use in-place pow_ to avoid allocating a second (N, col_count) intermediate.
        t = (s + 1).clamp_min_(0)
        t.pow_(l)
        out[:, col] = torch.where(pos_mask, (t - 1) / l, -torch.log1p(-s))

    # Columns where λ != 0 and λ > 2: pos → ((x+1)^λ-1)/λ, neg → -(((-x+1)^(2-λ)-1)/(2-λ))
    col = (~lambda_lt_eps & lambda_gt_2).nonzero(as_tuple=True)[0]
    if col.numel() > 0:
        s = series[:, col]
        l = lambdas[col]
        two_minus_l = 2 - l
        pos_mask = s >= 0
        # Use in-place pow_ for both branches.
        t_pos = (s + 1).clamp_min_(0)
        t_pos.pow_(l)
        t_neg = (-s + 1).clamp_min_(0)
        t_neg.pow_(two_minus_l)
        out[:, col] = torch.where(
            pos_mask, (t_pos - 1) / l, -(t_neg - 1) / two_minus_l)

    return out.squeeze(0) if squeezed else out


def inv_power_transform(series: torch.Tensor, transformer: PowerTransformer):
    # Normalize to 2-D so masks work uniformly; squeeze back at end.
    squeezed = series.dim() == 1
    if squeezed:
        series = series.unsqueeze(0)

    # lambdas: (D,) on same device/dtype as series
    lambdas = torch.as_tensor(
        transformer.lambdas_,
        dtype=series.dtype,
        device=series.device,
    )  # shape: (D,)

    eps = torch.finfo(series.dtype).eps

    # Lambda masks are (D,) — constant across rows, so index columns not elements.
    # This avoids large (N, D) boolean-index gathers when N (n_series) is huge.
    lambda_lt_eps = torch.abs(lambdas) < eps      # (D,)
    lambda_gt_2 = torch.abs(lambdas - 2) > eps  # (D,)

    out = torch.empty_like(series)

    # Columns where λ ≈ 0: pos → exp(x)-1, neg → 1-exp(-x)
    col = lambda_lt_eps.nonzero(as_tuple=True)[0]
    if col.numel() > 0:
        s = series[:, col]
        out[:, col] = torch.where(s >= 0, torch.exp(s) - 1, 1 - torch.exp(-s))

    # Columns where λ != 0 and λ <= 2: pos → (λx+1)^(1/λ)-1, neg → 1-exp(-x)
    col = (~lambda_lt_eps & ~lambda_gt_2).nonzero(as_tuple=True)[0]
    if col.numel() > 0:
        s = series[:, col]
        l = lambdas[col]
        pos_mask = s >= 0
        # Use in-place pow_ to avoid allocating a second (N, col_count) intermediate.
        t = (l * s + 1).clamp_min_(0)
        t.pow_(1.0 / l)
        out[:, col] = torch.where(pos_mask, t - 1, 1 - torch.exp(-s))

    # Columns where λ != 0 and λ > 2: pos → (λx+1)^(1/λ)-1, neg → 1-(-(2-λ)x+1)^(1/(2-λ))
    col = (~lambda_lt_eps & lambda_gt_2).nonzero(as_tuple=True)[0]
    if col.numel() > 0:
        s = series[:, col]
        l = lambdas[col]
        two_minus_l = 2 - l
        pos_mask = s >= 0
        # Use in-place pow_ for both branches.
        t_pos = (l * s + 1).clamp_min_(0)
        t_pos.pow_(1.0 / l)
        t_neg = (-two_minus_l * s + 1).clamp_min_(0)
        t_neg.pow_(1.0 / two_minus_l)
        out[:, col] = torch.where(pos_mask, t_pos - 1, 1 - t_neg)

    return out.squeeze(0) if squeezed else out
