from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats
from scipy.optimize import minimize

from core.weibull_math import EPS

BOOTSTRAP_RESAMPLES = 50
SUPPORTED_DISTRIBUTIONS = ("Weibull", "Lognormal", "Gamma", "Gumbel", "Exponential", "Normal", "Rayleigh")


@dataclass(frozen=True)
class DistributionFit:
    name: str
    params: tuple[float, ...]
    param_labels: tuple[str, ...]
    log_likelihood: float
    aic: float
    bic: float
    fit_score: float = float("nan")
    rmse: float = float("nan")
    ks_statistic: float = float("nan")
    ks_pvalue: float = float("nan")
    ad_statistic: float = float("nan")
    ad_pvalue: float = float("nan")


def neg_log_likelihood(params: np.ndarray, data: np.ndarray) -> float:
    beta, eta = params
    if beta <= 0 or eta <= 0:
        return 1e20

    n = len(data)
    log_likelihood = (
        n * np.log(beta)
        - n * beta * np.log(eta)
        + (beta - 1.0) * np.sum(np.log(data))
        - np.sum((data / eta) ** beta)
    )
    return float(-log_likelihood)


def optimization_starts(data: np.ndarray) -> list[np.ndarray]:
    mean = max(float(np.mean(data)), EPS)
    median = max(float(np.median(data)), EPS)
    p63 = max(float(np.quantile(data, 0.632)), EPS)
    starts = []
    for beta_guess in (0.8, 1.2, 1.5, 2.5, 4.0):
        for eta_guess in (mean, median, p63):
            starts.append(np.array([beta_guess, eta_guess], dtype=float))
    return starts


def estimate_weibull_mle(data: np.ndarray) -> tuple[float, float]:
    if len(data) < 2:
        raise ValueError("Need at least 2 valid positive observations.")

    cleaned = np.asarray(data, dtype=float)
    best_result = None
    best_score = float("inf")

    for start in optimization_starts(cleaned):
        result = minimize(
            neg_log_likelihood,
            x0=start,
            args=(cleaned,),
            method="L-BFGS-B",
            bounds=[(1e-6, None), (1e-6, None)],
        )
        if result.success and float(result.fun) < best_score:
            best_result = result
            best_score = float(result.fun)

    if best_result is None:
        fallback = minimize(
            neg_log_likelihood,
            x0=np.array([1.5, max(float(np.mean(cleaned)), EPS)]),
            args=(cleaned,),
            method="Powell",
            bounds=[(1e-6, None), (1e-6, None)],
        )
        if not fallback.success:
            raise RuntimeError("Weibull parameter estimation failed for this dataset.")
        best_result = fallback

    beta, eta = best_result.x
    return float(beta), float(eta)


def distribution_pdf(name: str, params: tuple[float, ...], values: np.ndarray | float) -> np.ndarray:
    if name == "Weibull":
        shape, scale = params
        return stats.weibull_min.pdf(values, shape, loc=0.0, scale=scale)
    if name == "Lognormal":
        sigma, scale = params
        return stats.lognorm.pdf(values, sigma, loc=0.0, scale=scale)
    if name == "Gamma":
        shape, scale = params
        return stats.gamma.pdf(values, shape, loc=0.0, scale=scale)
    if name == "Gumbel":
        loc, scale = params
        return stats.gumbel_r.pdf(values, loc=loc, scale=scale)
    if name == "Exponential":
        (scale,) = params
        return stats.expon.pdf(values, loc=0.0, scale=scale)
    if name == "Normal":
        mean, std = params
        return stats.norm.pdf(values, loc=mean, scale=std)
    if name == "Rayleigh":
        (scale,) = params
        return stats.rayleigh.pdf(values, loc=0.0, scale=scale)
    raise ValueError(f"Unsupported distribution: {name}")


def distribution_cdf(name: str, params: tuple[float, ...], values: np.ndarray | float) -> np.ndarray:
    if name == "Weibull":
        shape, scale = params
        return stats.weibull_min.cdf(values, shape, loc=0.0, scale=scale)
    if name == "Lognormal":
        sigma, scale = params
        return stats.lognorm.cdf(values, sigma, loc=0.0, scale=scale)
    if name == "Gamma":
        shape, scale = params
        return stats.gamma.cdf(values, shape, loc=0.0, scale=scale)
    if name == "Gumbel":
        loc, scale = params
        return stats.gumbel_r.cdf(values, loc=loc, scale=scale)
    if name == "Exponential":
        (scale,) = params
        return stats.expon.cdf(values, loc=0.0, scale=scale)
    if name == "Normal":
        mean, std = params
        return stats.norm.cdf(values, loc=mean, scale=std)
    if name == "Rayleigh":
        (scale,) = params
        return stats.rayleigh.cdf(values, loc=0.0, scale=scale)
    raise ValueError(f"Unsupported distribution: {name}")


def distribution_ppf(name: str, params: tuple[float, ...], probabilities: np.ndarray | float) -> np.ndarray:
    p = np.clip(probabilities, EPS, 1.0 - EPS)
    if name == "Weibull":
        shape, scale = params
        return stats.weibull_min.ppf(p, shape, loc=0.0, scale=scale)
    if name == "Lognormal":
        sigma, scale = params
        return stats.lognorm.ppf(p, sigma, loc=0.0, scale=scale)
    if name == "Gamma":
        shape, scale = params
        return stats.gamma.ppf(p, shape, loc=0.0, scale=scale)
    if name == "Gumbel":
        loc, scale = params
        return stats.gumbel_r.ppf(p, loc=loc, scale=scale)
    if name == "Exponential":
        (scale,) = params
        return stats.expon.ppf(p, loc=0.0, scale=scale)
    if name == "Normal":
        mean, std = params
        return stats.norm.ppf(p, loc=mean, scale=std)
    if name == "Rayleigh":
        (scale,) = params
        return stats.rayleigh.ppf(p, loc=0.0, scale=scale)
    raise ValueError(f"Unsupported distribution: {name}")


def distribution_mean(name: str, params: tuple[float, ...]) -> float:
    if name == "Weibull":
        shape, scale = params
        return float(stats.weibull_min.mean(shape, loc=0.0, scale=scale))
    if name == "Lognormal":
        sigma, scale = params
        return float(stats.lognorm.mean(sigma, loc=0.0, scale=scale))
    if name == "Gamma":
        shape, scale = params
        return float(stats.gamma.mean(shape, loc=0.0, scale=scale))
    if name == "Gumbel":
        loc, scale = params
        return float(stats.gumbel_r.mean(loc=loc, scale=scale))
    if name == "Exponential":
        (scale,) = params
        return float(stats.expon.mean(loc=0.0, scale=scale))
    if name == "Normal":
        mean, std = params
        return float(stats.norm.mean(loc=mean, scale=std))
    if name == "Rayleigh":
        (scale,) = params
        return float(stats.rayleigh.mean(loc=0.0, scale=scale))
    raise ValueError(f"Unsupported distribution: {name}")


def distribution_hazard(name: str, params: tuple[float, ...], values: np.ndarray | float) -> np.ndarray:
    pdf = np.asarray(distribution_pdf(name, params, values), dtype=float)
    survival = np.clip(1.0 - np.asarray(distribution_cdf(name, params, values), dtype=float), EPS, None)
    return pdf / survival


def _anderson_darling_statistic(data: np.ndarray, name: str, params: tuple[float, ...]) -> float:
    sorted_data = np.sort(np.asarray(data, dtype=float))
    n = len(sorted_data)
    cdf_values = np.clip(distribution_cdf(name, params, sorted_data), EPS, 1.0 - EPS)
    i = np.arange(1, n + 1, dtype=float)
    statistic = -n - np.mean((2.0 * i - 1.0) * (np.log(cdf_values) + np.log(1.0 - cdf_values[::-1])))
    return float(statistic)


def _ad_test_for_distribution(data: np.ndarray, distribution_name: str, params: tuple[float, ...]) -> tuple[float, float]:
    try:
        if distribution_name == "Weibull":
            result = stats.anderson(data, dist="weibull_min", method="interpolate")
            return _anderson_darling_statistic(data, distribution_name, params), float(result.pvalue)
        if distribution_name == "Lognormal":
            result = stats.anderson(np.log(np.maximum(data, EPS)), dist="norm", method="interpolate")
            return _anderson_darling_statistic(data, distribution_name, params), float(result.pvalue)
        if distribution_name == "Normal":
            result = stats.anderson(data, dist="norm", method="interpolate")
            return _anderson_darling_statistic(data, distribution_name, params), float(result.pvalue)
        if distribution_name == "Gumbel":
            result = stats.anderson(data, dist="gumbel_r", method="interpolate")
            return _anderson_darling_statistic(data, distribution_name, params), float(result.pvalue)
        if distribution_name == "Rayleigh":
            (scale,) = params
            transformed = (np.asarray(data, dtype=float) / max(scale, EPS)) ** 2 / 2.0
            result = stats.anderson(transformed, dist="expon", method="interpolate")
            return _anderson_darling_statistic(data, distribution_name, params), float(result.pvalue)
        if distribution_name == "Exponential":
            result = stats.anderson(data, dist="expon", method="interpolate")
            return _anderson_darling_statistic(data, distribution_name, params), float(result.pvalue)
        return _anderson_darling_statistic(data, distribution_name, params), float("nan")
    except Exception:
        return _anderson_darling_statistic(data, distribution_name, params), float("nan")


def fit_single_distribution(
    name: str,
    data: np.ndarray,
    weibull_params: tuple[float, float] | None = None,
) -> DistributionFit:
    cleaned = np.asarray(data, dtype=float)
    if len(cleaned) < 2:
        raise ValueError("Need at least 2 valid positive observations.")

    if name == "Weibull":
        shape, scale = weibull_params if weibull_params is not None else estimate_weibull_mle(cleaned)
        params = (float(shape), float(scale))
        param_labels = ("beta", "eta")
    elif name == "Lognormal":
        sigma, _, scale = stats.lognorm.fit(cleaned, floc=0.0)
        params = (float(sigma), float(scale))
        param_labels = ("sigma", "scale")
    elif name == "Gamma":
        shape, _, scale = stats.gamma.fit(cleaned, floc=0.0)
        params = (float(shape), float(scale))
        param_labels = ("shape", "scale")
    elif name == "Gumbel":
        loc, scale = stats.gumbel_r.fit(cleaned)
        params = (float(loc), float(max(scale, EPS)))
        param_labels = ("loc", "scale")
    elif name == "Exponential":
        _, scale = stats.expon.fit(cleaned, floc=0.0)
        params = (float(scale),)
        param_labels = ("scale",)
    elif name == "Normal":
        mean, std = stats.norm.fit(cleaned)
        params = (float(mean), float(max(std, EPS)))
        param_labels = ("mean", "std")
    elif name == "Rayleigh":
        _, scale = stats.rayleigh.fit(cleaned, floc=0.0)
        params = (float(scale),)
        param_labels = ("scale",)
    else:
        raise ValueError(f"Unsupported distribution: {name}")

    log_likelihood = float(np.sum(np.log(np.clip(distribution_pdf(name, params, cleaned), EPS, None))))
    n = len(cleaned)
    parameter_count = len(params)
    aic = float(2 * parameter_count - 2 * log_likelihood)
    bic = float(parameter_count * np.log(n) - 2 * log_likelihood)
    ks_result = stats.kstest(cleaned, lambda value: distribution_cdf(name, params, value))
    median_ranks = (np.arange(1, n + 1, dtype=float) - 0.3) / (n + 0.4)
    fitted_probabilities = np.asarray(distribution_cdf(name, params, np.sort(cleaned)), dtype=float)
    residuals = fitted_probabilities - median_ranks
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((median_ranks - np.mean(median_ranks)) ** 2))
    fit_score = float(np.clip(1.0 - (ss_res / ss_tot if ss_tot > EPS else 0.0), 0.0, 1.0))
    rmse = float(np.sqrt(np.mean(residuals**2)))
    ad_statistic, ad_pvalue = _ad_test_for_distribution(cleaned, name, params)

    return DistributionFit(
        name=name,
        params=params,
        param_labels=param_labels,
        log_likelihood=log_likelihood,
        aic=aic,
        bic=bic,
        fit_score=fit_score,
        rmse=rmse,
        ks_statistic=float(ks_result.statistic),
        ks_pvalue=float(ks_result.pvalue),
        ad_statistic=ad_statistic,
        ad_pvalue=ad_pvalue,
    )


def fit_distribution_models(
    data: np.ndarray,
    weibull_params: tuple[float, float] | None = None,
) -> tuple[DistributionFit, ...]:
    fits = tuple(fit_single_distribution(name, data, weibull_params=weibull_params) for name in SUPPORTED_DISTRIBUTIONS)
    return tuple(sorted(fits, key=lambda item: (item.aic, item.bic, item.rmse, item.name)))


def bootstrap_distribution_parameters(
    data: np.ndarray,
    distribution_name: str,
    resamples: int = BOOTSTRAP_RESAMPLES,
    random_seed: int | None = None,
) -> np.ndarray:
    cleaned = np.asarray(data, dtype=float)
    if len(cleaned) < 2:
        raise ValueError("Need at least 2 valid positive observations.")

    rng = np.random.default_rng(random_seed)
    parameter_samples: list[tuple[float, ...]] = []

    for _ in range(max(int(resamples), 1)):
        sample = np.sort(rng.choice(cleaned, size=len(cleaned), replace=True))
        try:
            fit = fit_single_distribution(distribution_name, sample)
        except Exception:
            continue
        if all(np.isfinite(value) for value in fit.params):
            parameter_samples.append(tuple(float(value) for value in fit.params))

    if not parameter_samples:
        fallback_fit = fit_single_distribution(distribution_name, cleaned)
        parameter_samples.append(tuple(float(value) for value in fallback_fit.params))

    return np.asarray(parameter_samples, dtype=float)


def bootstrap_weibull_parameters(
    data: np.ndarray,
    resamples: int = BOOTSTRAP_RESAMPLES,
    random_seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    samples = bootstrap_distribution_parameters(data, "Weibull", resamples=resamples, random_seed=random_seed)
    return samples[:, 0], samples[:, 1]
