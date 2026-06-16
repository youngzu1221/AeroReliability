from __future__ import annotations

from dataclasses import dataclass
from zlib import crc32

import numpy as np
from scipy.integrate import cumulative_trapezoid

from core.optimization import (
    DistributionFit,
    bootstrap_distribution_parameters,
    estimate_weibull_mle,
    fit_distribution_models,
    fit_single_distribution,
    distribution_cdf,
    distribution_mean,
    distribution_ppf,
)
from core.weibull_math import EPS, failure_mode_from_beta

RISK_THRESHOLDS = {
    "high": 0.60,
    "medium": 0.30,
}

DISTRIBUTION_BEHAVIOR = {
    "Weibull": "Beta-based wear-out model",
    "Lognormal": "Positively skewed life model",
    "Gamma": "Cumulative damage model",
    "Gumbel": "Extreme-value stress model",
    "Exponential": "Constant failure-rate model",
    "Normal": "Symmetric variation model",
    "Rayleigh": "Rising-rate special-case model",
}


@dataclass(frozen=True)
class ConfidenceInterval:
    lower: float
    upper: float


@dataclass(frozen=True)
class WeibullResult:
    component: str
    data: np.ndarray
    selected_distribution: str
    selected_fit: DistributionFit
    selected_param_labels: tuple[str, ...]
    selected_param_values: tuple[float, ...]
    selected_param_cis: tuple[ConfidenceInterval, ...]
    characteristic_value: float
    beta: float
    eta: float
    beta_ci: ConfidenceInterval
    eta_ci: ConfidenceInterval
    mttf: float
    mtbf: float
    current_reliability: float
    mission_reliability: float
    conditional_reliability: float
    conditional_reliability_ci: ConfidenceInterval
    conditional_failure_probability: float
    conditional_failure_probability_ci: ConfidenceInterval
    rul: float
    rul_ci: ConfidenceInterval
    optimal_replacement: float
    min_cost_rate: float
    failure_mode: str
    risk: str
    best_distribution: str
    distribution_fits: tuple[DistributionFit, ...]
    severity: int
    occurrence: int
    detectability: int
    rpn: int
    bootstrap_param_samples: np.ndarray
    bootstrap_metric_samples: dict[str, np.ndarray]


@dataclass(frozen=True)
class MaintenanceDecision:
    level: str
    message: str


def confidence_interval(values: np.ndarray, level: float = 0.95) -> ConfidenceInterval:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return ConfidenceInterval(float("nan"), float("nan"))

    if len(arr) == 1:
        value = float(arr[0])
        return ConfidenceInterval(value, value)

    tail = max((1.0 - float(level)) / 2.0, 0.0)
    lower, upper = np.quantile(arr, [tail, 1.0 - tail])
    return ConfidenceInterval(float(lower), float(upper))


def risk_label(probability: float) -> str:
    if probability >= RISK_THRESHOLDS["high"]:
        return "HIGH"
    if probability >= RISK_THRESHOLDS["medium"]:
        return "MEDIUM"
    return "LOW"


def clamp_fmea_rating(value: int | float) -> int:
    return int(np.clip(int(round(float(value))), 1, 10))


def occurrence_rating(probability: float) -> int:
    return int(np.clip(np.ceil(float(probability) * 10.0), 1, 10))


def decision_from_metrics(conditional_failure_probability: float, optimal_replacement: float) -> MaintenanceDecision:
    if conditional_failure_probability >= 0.60:
        return MaintenanceDecision(
            level="HIGH",
            message=f"Immediate maintenance planning is recommended before about {optimal_replacement:,.2f}.",
        )
    if conditional_failure_probability >= 0.30:
        return MaintenanceDecision(
            level="MEDIUM",
            message=f"Schedule maintenance soon. Recommended replacement around {optimal_replacement:,.2f}.",
        )
    return MaintenanceDecision(
        level="LOW",
        message=f"Continue monitoring. Recommended replacement around {optimal_replacement:,.2f}.",
    )


def distribution_reliability(distribution_name: str, params: tuple[float, ...], time_values: np.ndarray | float) -> np.ndarray:
    return 1.0 - np.asarray(distribution_cdf(distribution_name, params, time_values), dtype=float)


def distribution_failure_mode(distribution_name: str, weibull_beta: float) -> str:
    if distribution_name == "Weibull":
        return failure_mode_from_beta(weibull_beta)
    return DISTRIBUTION_BEHAVIOR.get(distribution_name, distribution_name)


def cost_based_optimal_replacement(
    data: np.ndarray,
    distribution_name: str,
    params: tuple[float, ...],
    preventive_cost: float,
    failure_cost: float,
    characteristic_value: float,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    start = max(float(np.min(data) * 0.05), 1.0)
    end = max(float(np.max(data) * 2.5), characteristic_value * 2.0, start + 1.0)
    t_range = np.linspace(0.0, end, 1800)
    cost_rate = np.zeros_like(t_range)
    idx = 1

    for _ in range(5):
        t_range = np.linspace(0.0, end, 1800)
        reliability_curve = np.clip(distribution_reliability(distribution_name, params, t_range), 0.0, 1.0)
        failure_probability = 1.0 - reliability_curve
        expected_cycle_length = np.maximum(cumulative_trapezoid(reliability_curve, t_range, initial=0.0), EPS)
        numerator = preventive_cost * reliability_curve + failure_cost * failure_probability
        cost_rate = numerator / expected_cycle_length
        idx = 1 + int(np.argmin(cost_rate[1:]))
        if idx < int(0.97 * len(t_range)):
            break
        end *= 1.75

    return float(t_range[idx]), float(cost_rate[idx]), t_range[1:], cost_rate[1:]


def _find_distribution_fit(distribution_fits: tuple[DistributionFit, ...], distribution_name: str) -> DistributionFit:
    for fit in distribution_fits:
        if fit.name == distribution_name:
            return fit
    raise ValueError(f"{distribution_name} fit is not available.")


def _metrics_from_params(
    distribution_name: str,
    params: tuple[float, ...],
    current_age: float,
    mission_time: float,
) -> dict[str, float]:
    current_rel = float(np.clip(distribution_reliability(distribution_name, params, current_age), 0.0, 1.0))
    mission_rel = float(np.clip(distribution_reliability(distribution_name, params, current_age + mission_time), 0.0, 1.0))
    conditional_rel = mission_rel / current_rel if current_rel > EPS else 0.0
    conditional_rel = float(np.clip(conditional_rel, 0.0, 1.0))
    conditional_fail = float(np.clip(1.0 - conditional_rel, 0.0, 1.0))
    mttf = float(distribution_mean(distribution_name, params))
    characteristic_value = float(distribution_ppf(distribution_name, params, 0.632))
    return {
        "current_reliability": current_rel,
        "mission_reliability": mission_rel,
        "conditional_reliability": conditional_rel,
        "conditional_failure_probability": conditional_fail,
        "mttf": mttf,
        "rul": float(mttf - current_age),
        "characteristic_value": characteristic_value,
    }


def analyze_component(
    component: str,
    data: np.ndarray,
    mttr: float,
    current_age: float,
    mission_time: float,
    preventive_cost: float,
    failure_cost: float,
    severity: int = 5,
    detectability: int = 5,
    selected_distribution: str = "Weibull",
) -> WeibullResult:
    distribution_fits = fit_distribution_models(data)
    selected_fit = _find_distribution_fit(distribution_fits, selected_distribution)
    weibull_fit = _find_distribution_fit(distribution_fits, "Weibull")
    beta, eta = weibull_fit.params

    seed = crc32(f"{component}:{selected_distribution}".encode("utf-8")) & 0xFFFFFFFF
    bootstrap_param_samples = bootstrap_distribution_parameters(data, selected_distribution, random_seed=seed)
    bootstrap_weibull_samples = (
        bootstrap_param_samples
        if selected_distribution == "Weibull"
        else bootstrap_distribution_parameters(data, "Weibull", random_seed=seed)
    )
    selected_param_cis = tuple(
        confidence_interval(bootstrap_param_samples[:, idx])
        for idx in range(bootstrap_param_samples.shape[1])
    )

    selected_metrics = _metrics_from_params(selected_distribution, selected_fit.params, current_age, mission_time)
    bootstrap_metric_rows = [_metrics_from_params(selected_distribution, tuple(sample), current_age, mission_time) for sample in bootstrap_param_samples]
    bootstrap_metrics = {
        "current_reliability": np.asarray([row["current_reliability"] for row in bootstrap_metric_rows], dtype=float),
        "mission_reliability": np.asarray([row["mission_reliability"] for row in bootstrap_metric_rows], dtype=float),
        "conditional_reliability": np.asarray([row["conditional_reliability"] for row in bootstrap_metric_rows], dtype=float),
        "conditional_failure_probability": np.asarray([row["conditional_failure_probability"] for row in bootstrap_metric_rows], dtype=float),
        "mttf": np.asarray([row["mttf"] for row in bootstrap_metric_rows], dtype=float),
        "rul": np.asarray([row["rul"] for row in bootstrap_metric_rows], dtype=float),
        "characteristic_value": np.asarray([row["characteristic_value"] for row in bootstrap_metric_rows], dtype=float),
    }

    optimal_replacement, min_cost_rate, _, _ = cost_based_optimal_replacement(
        data,
        selected_distribution,
        selected_fit.params,
        preventive_cost,
        failure_cost,
        selected_metrics["characteristic_value"],
    )

    clipped_severity = clamp_fmea_rating(severity)
    occurrence = occurrence_rating(selected_metrics["conditional_failure_probability"])
    clipped_detectability = clamp_fmea_rating(detectability)

    beta_ci = confidence_interval(bootstrap_weibull_samples[:, 0])
    eta_ci = confidence_interval(bootstrap_weibull_samples[:, 1])

    return WeibullResult(
        component=component,
        data=np.asarray(data, dtype=float),
        selected_distribution=selected_distribution,
        selected_fit=selected_fit,
        selected_param_labels=selected_fit.param_labels,
        selected_param_values=selected_fit.params,
        selected_param_cis=selected_param_cis,
        characteristic_value=selected_metrics["characteristic_value"],
        beta=float(beta),
        eta=float(eta),
        beta_ci=beta_ci,
        eta_ci=eta_ci,
        mttf=selected_metrics["mttf"],
        mtbf=float(selected_metrics["mttf"] + mttr),
        current_reliability=selected_metrics["current_reliability"],
        mission_reliability=selected_metrics["mission_reliability"],
        conditional_reliability=selected_metrics["conditional_reliability"],
        conditional_reliability_ci=confidence_interval(bootstrap_metrics["conditional_reliability"]),
        conditional_failure_probability=selected_metrics["conditional_failure_probability"],
        conditional_failure_probability_ci=confidence_interval(bootstrap_metrics["conditional_failure_probability"]),
        rul=selected_metrics["rul"],
        rul_ci=confidence_interval(bootstrap_metrics["rul"]),
        optimal_replacement=optimal_replacement,
        min_cost_rate=min_cost_rate,
        failure_mode=distribution_failure_mode(selected_distribution, float(beta)),
        risk=risk_label(selected_metrics["conditional_failure_probability"]),
        best_distribution=distribution_fits[0].name,
        distribution_fits=distribution_fits,
        severity=clipped_severity,
        occurrence=occurrence,
        detectability=clipped_detectability,
        rpn=clipped_severity * occurrence * clipped_detectability,
        bootstrap_param_samples=bootstrap_param_samples,
        bootstrap_metric_samples=bootstrap_metrics,
    )


def analyze_datasets(
    datasets: dict[str, np.ndarray],
    mttr: float,
    current_age: float,
    mission_time: float,
    preventive_cost: float,
    failure_cost: float,
    severity: int = 5,
    detectability: int = 5,
    selected_distribution: str = "Weibull",
) -> tuple[dict[str, WeibullResult], tuple[str, ...]]:
    analysis: dict[str, WeibullResult] = {}
    skipped: list[str] = []

    for component, data in datasets.items():
        try:
            analysis[component] = analyze_component(
                component,
                data,
                mttr,
                current_age,
                mission_time,
                preventive_cost,
                failure_cost,
                severity,
                detectability,
                selected_distribution,
            )
        except Exception as exc:
            skipped.append(f"{component}: {exc}")

    return analysis, tuple(skipped)


def results_to_frame(results: dict[str, WeibullResult]):
    import pandas as pd

    rows = []
    for result in results.values():
        best_fit = result.distribution_fits[0]
        row = {
            "Component": result.component,
            "Selected Distribution": result.selected_distribution,
            "Characteristic Value": result.characteristic_value,
            "MTTF": result.mttf,
            "MTBF": result.mtbf,
            "Conditional Reliability": result.conditional_reliability,
            "Conditional Reliability CI Lower": result.conditional_reliability_ci.lower,
            "Conditional Reliability CI Upper": result.conditional_reliability_ci.upper,
            "Conditional Probability of Failure": result.conditional_failure_probability,
            "Failure Probability CI Lower": result.conditional_failure_probability_ci.lower,
            "Failure Probability CI Upper": result.conditional_failure_probability_ci.upper,
            "RUL": result.rul,
            "RUL CI Lower": result.rul_ci.lower,
            "RUL CI Upper": result.rul_ci.upper,
            "Optimal Replacement": result.optimal_replacement,
            "Min Cost Rate": result.min_cost_rate,
            "Failure Mode": result.failure_mode,
            "Risk": result.risk,
            "Best Fit Distribution": result.best_distribution,
            "Best Fit AIC": best_fit.aic,
            "Best Fit BIC": best_fit.bic,
            "Best Fit RMSE": best_fit.rmse,
            "Severity": result.severity,
            "Occurrence": result.occurrence,
            "Detectability": result.detectability,
            "RPN": result.rpn,
        }
        for idx, label in enumerate(result.selected_param_labels, start=1):
            row[f"Parameter {idx}"] = result.selected_param_values[idx - 1]
            row[f"Parameter {idx} Label"] = label
            row[f"Parameter {idx} CI Lower"] = result.selected_param_cis[idx - 1].lower
            row[f"Parameter {idx} CI Upper"] = result.selected_param_cis[idx - 1].upper
        rows.append(row)

    columns = [
        "Component",
        "Selected Distribution",
        "Parameter 1 Label",
        "Parameter 1",
        "Parameter 1 CI Lower",
        "Parameter 1 CI Upper",
        "Parameter 2 Label",
        "Parameter 2",
        "Parameter 2 CI Lower",
        "Parameter 2 CI Upper",
        "Characteristic Value",
        "MTTF",
        "MTBF",
        "Conditional Reliability",
        "Conditional Reliability CI Lower",
        "Conditional Reliability CI Upper",
        "Conditional Probability of Failure",
        "Failure Probability CI Lower",
        "Failure Probability CI Upper",
        "RUL",
        "RUL CI Lower",
        "RUL CI Upper",
        "Optimal Replacement",
        "Min Cost Rate",
        "Failure Mode",
        "Risk",
        "Best Fit Distribution",
        "Best Fit AIC",
        "Best Fit BIC",
        "Best Fit RMSE",
        "Severity",
        "Occurrence",
        "Detectability",
        "RPN",
    ]

    if not rows:
        return pd.DataFrame(columns=columns)

    return pd.DataFrame(rows).reindex(columns=columns).sort_values("Conditional Probability of Failure", ascending=False).reset_index(drop=True)
