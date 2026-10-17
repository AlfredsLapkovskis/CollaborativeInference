import numpy as np
from sklearn.neighbors import KernelDensity
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.stats import entropy


def lloyd_max(
    series: np.array,
    n_levels,
    mc_samples=100_000,
    mse_tolerance=0.0001,
    random_state=42,
    kernel="gaussian",
    bandwidth="silverman",
    standardize=True,
):
    rng = np.random.default_rng(random_state)

    def randint():
        return rng.integers(0, 1 << 31, 1)[0]

    mean = series.mean() if standardize else 0.
    std = series.std() if standardize else 1.
    z_series = (series - mean) / std

    kernel = KernelDensity(bandwidth=bandwidth, kernel=kernel).fit(
        z_series.reshape(-1, 1))

    levels = np.linspace(z_series.min(), z_series.max(), n_levels)

    prev_mse = np.inf

    while True:
        thresholds = np.array([(levels[th - 1] + levels[th]) /
                              2 for th in range(1, n_levels)])

        for i in range(len(levels)):
            a = thresholds[i - 1] if i > 0 else -np.inf
            b = thresholds[i] if i < len(thresholds) else np.inf

            levels[i] = _kde_interval_mean(kernel, a, b)

        levels = np.sort(levels)

        samples = kernel.sample(mc_samples * len(levels),
                                random_state=randint()).ravel()
        idx = np.digitize(samples, thresholds, right=True)

        mse = ((samples - levels[idx]) ** 2).mean()
        if prev_mse - mse < mse_tolerance:
            break
        prev_mse = mse

    print(f"Lloyd-Max converged with MSE={mse}")
    return levels * std + mean, thresholds * std + mean, mse


def get_optimal_reconstruction_values(
    series: np.array,
    thresholds,
    mc_samples=100_000,
    mse_tolerance=0.0001,
    random_state=42,
    kernel="gaussian",
    bandwidth="silverman",
    min_val=None,
    max_val=None,
):
    rng = np.random.default_rng(random_state)
    random_states = rng.integers(0, 1 << 31, 10)

    kernel = KernelDensity(bandwidth=bandwidth, kernel=kernel).fit(
        series.reshape(-1, 1))
    samples = kernel.sample(mc_samples * len(thresholds + 1),
                            random_state=random_state).ravel()
    idx = np.digitize(samples.ravel(), thresholds, right=True)

    def reconstruction_mse(params):
        beta_a, beta_b = params

        mse = 0
        for state in random_states:
            restored = stochastic_vector_restore(
                idx[np.newaxis, :],
                thresholds[np.newaxis, :],
                np.atleast_1d(min_val),
                np.atleast_1d(max_val),
                beta_a,
                beta_b,
                random_state=state,
            )[0, :]
            mse += ((samples - restored) ** 2).mean()

        mse /= len(random_states)
        print(f"Params {beta_a=}, {beta_b=}, {mse=}")

        return mse

    result = minimize(
        reconstruction_mse,
        x0=(1., 1.),
        method="Powell",
        bounds=[(1e-9, None), (1e-9, None)],
        options={
            "xtol": 1e-6,
            "ftol": 1e-6,
        },
    )

    print(f"x={result.x}, fun={result.fun}")
    return result.x, result.fun


def vector_restore(series, levels):
    # series: (m, n) indices
    # levels: (m, k) levels (each row ascending)
    # Result: (m, n) values

    return levels[np.arange(series.shape[0])[:, None], series]


def stochastic_vector_restore(series, thresholds, min_val=None, max_val=None, beta_a=None, beta_b=None, random_state=None):
    rng = np.random.default_rng(random_state)

    if min_val is None or max_val is None:
        avg_diff = np.mean(np.diff(thresholds, axis=1), axis=1)
        min_val = thresholds[:, 0] - avg_diff if min_val is None else min_val
        max_val = thresholds[:, -1] + avg_diff if max_val is None else max_val

    low_thresholds = np.concatenate(
        [min_val[:, np.newaxis], thresholds], axis=1)
    high_thresholds = np.concatenate(
        [thresholds, max_val[:, np.newaxis]], axis=1)

    low_levels = vector_restore(series, low_thresholds)
    high_levels = vector_restore(series, high_thresholds)

    if beta_a is not None and beta_b is not None:
        u = rng.beta(beta_a, beta_b, size=series.shape)
    else:
        u = rng.uniform(size=series.shape)

    return low_levels + (high_levels - low_levels) * u


def vector_digitize(series, thresholds):
    # series: (m, n) values
    # bins: (m, k) thresholds (each row ascending)
    # Result: (m, n) indices v

    # bins[i-1] < x <= bins[i]

    return (thresholds[:, None, :] < series[..., None]).sum(
        axis=-1)   # counts per (i,j)


def _kde_interval_mean(kde: KernelDensity, a, b):
    pdf = _make_pdf(kde)

    fx, _ = quad(pdf, a, b)
    if fx <= 0:
        raise RuntimeError("Failed to compute an integral")

    xfx, _ = quad(lambda x: x * pdf(x), a, b)

    return xfx / fx


def _make_pdf(kde: KernelDensity):
    def pdf(x):
        x = np.atleast_1d(x).astype(float)
        x = x.reshape(-1, 1)
        return np.exp(kde.score_samples(x))
    return pdf
