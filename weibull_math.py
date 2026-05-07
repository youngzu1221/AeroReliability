from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.optimize import minimize
from scipy.special import gamma

EPS = 1e-12
RISK_THRESHOLDS = {"high": 0.60, "medium": 0.30}


@dataclass(frozen=True)
class WeibullResult:
    component: str
    data: np.ndarray
    beta: float
    eta: float
    mttf: float
    mtbf: float
    current_reliability: float
    mission_reliability: float
    conditional_reliability: float
    conditional_failure_probability: float
    rul: float
    optimal_replacement: float
    min_cost_rate: float
    failure_mode: str
    risk: str
    decision: str


def clean_life_data(values: Iterable[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(list(values) if not isinstance(values, np.ndarray) else values, dtype=object)
    arr = np.array([x for x in arr if x is not None], dtype=object)
    numeric = []
    for item in arr:
        try:
            value = float(item)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value) and value > 0:
            numeric.append(value)
    return np.sort(np.asarray(numeric, dtype=float))


def reliability(t: np.ndarray | float, beta: float, eta: float) -> np.ndarray:
    t_arr = np.maximum(np.asarray(t, dtype=float), EPS)
    return np.exp(-((t_arr / eta) ** beta))


def hazard(t: np.ndarray | float, beta: float, eta: float) -> np.ndarray:
    t_arr = np.maximum(np.asarray(t, dtype=float), EPS)
    return (beta / eta) * (t_arr / eta) ** (beta - 1.0)


def weibull_pdf(t: np.ndarray | float, beta: float, eta: float) -> np.ndarray:
    return hazard(t, beta, eta) * reliability(t, beta, eta)


def weibull_cdf(t: np.ndarray | float, beta: float, eta: float) -> np.ndarray:
    return 1.0 - reliability(t, beta, eta)


def weibull_quantile(probability: float, beta: float, eta: float) -> float:
    p = float(np.clip(probability, EPS, 1.0 - EPS))
    return float(eta * (-np.log(1.0 - p)) ** (1.0 / beta))


def conditional_reliability_between(start_time: float, end_time: float, beta: float, eta: float) -> float:
    start_rel = float(reliability(start_time, beta, eta))
    if start_rel <= EPS:
        return 0.0
    end_rel = float(reliability(end_time, beta, eta))
    return float(np.clip(end_rel / start_rel, 0.0, 1.0))


def conditional_failure_probability_between(start_time: float, end_time: float, beta: float, eta: float) -> float:
    return float(np.clip(1.0 - conditional_reliability_between(start_time, end_time, beta, eta), 0.0, 1.0))


def expected_remaining_life(current_age: float, beta: float, eta: float) -> float:
    current_age = max(float(current_age), 0.0)
    survival_now = float(reliability(current_age, beta, eta))
    if survival_now <= EPS:
        return 0.0

    upper = max(current_age + 10.0 * eta, weibull_quantile(0.999999, beta, eta), current_age + 1.0)
    grid = np.linspace(current_age, upper, 4000)
    surv = reliability(grid, beta, eta)
    area = float(getattr(np, "trapezoid", np.trapz)(surv, grid))
    return max(0.0, area / survival_now)


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


def estimate_weibull_mle(data: np.ndarray) -> tuple[float, float]:
    if len(data) < 2:
        raise ValueError("Need at least 2 valid positive observations.")

    guesses = [
        np.array([1.2, max(float(np.mean(data)), EPS)]),
        np.array([1.8, max(float(np.median(data)), EPS)]),
        np.array([3.0, max(float(np.max(data) / 2.0), EPS)]),
    ]

    best_result = None
    best_value = float("inf")
    for guess in guesses:
        result = minimize(
            neg_log_likelihood,
            x0=guess,
            args=(data,),
            method="L-BFGS-B",
            bounds=[(1e-6, None), (1e-6, None)],
        )
        if result.success and float(result.fun) < best_value:
            best_result = result
            best_value = float(result.fun)

    if best_result is None:
        raise RuntimeError("Weibull MLE failed for all initial guesses.")

    beta, eta = best_result.x
    return float(beta), float(eta)


def cost_based_optimal_replacement(
    data: np.ndarray,
    beta: float,
    eta: float,
    preventive_cost: float,
    failure_cost: float,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    start = max(EPS, float(np.min(data) * 0.25))
    end = max(float(np.max(data) * 3.0), eta * 4.0, start + 1.0)
    t_range = np.linspace(start, end, 1600)

    t_full = np.concatenate(([0.0], t_range))
    survival_full = reliability(t_full, beta, eta)
    cumulative_survival = np.concatenate(
        ([0.0], np.cumsum((survival_full[:-1] + survival_full[1:]) * 0.5 * np.diff(t_full)))
    )[1:]
    survival = survival_full[1:]
    expected_cycle_cost = preventive_cost * survival + failure_cost * (1.0 - survival)
    cost_rate = expected_cycle_cost / np.maximum(cumulative_survival, EPS)
    idx = int(np.argmin(cost_rate))
    return float(t_range[idx]), float(cost_rate[idx]), t_range, cost_rate


def risk_label(probability: float) -> str:
    if probability >= RISK_THRESHOLDS["high"]:
        return "HIGH"
    if probability >= RISK_THRESHOLDS["medium"]:
        return "MEDIUM"
    return "LOW"


def failure_mode_from_beta(beta: float) -> str:
    if beta < 0.95:
        return "Infant mortality"
    if beta <= 1.05:
        return "Random failure"
    if beta < 3.0:
        return "Wear-out"
    return "Wear-out (sudden failure)"


def decision_from_metrics(
    current_age: float,
    optimal_replacement: float,
    conditional_failure_probability: float,
    beta: float,
) -> str:
    if current_age >= optimal_replacement or conditional_failure_probability >= 0.70:
        return "OVERDUE"
    if conditional_failure_probability >= 0.35:
        return "PLAN REPLACEMENT"
    if beta >= 5.0 and current_age >= 0.75 * optimal_replacement:
        return "PLAN REPLACEMENT"
    if current_age >= 0.90 * optimal_replacement:
        return "PLAN REPLACEMENT"
    return "SAFE"


def fmt_pct(value: float) -> str:
    return f"{float(value):.2%}"


def fmt_money(value: float) -> str:
    return f"${float(value):,.2f}"


def safe_filename(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value.strip())
    return safe or "component"


def analyze_component(
    component: str,
    data: np.ndarray,
    mttr: float,
    current_age: float,
    mission_time: float,
    preventive_cost: float,
    failure_cost: float,
) -> WeibullResult:
    beta, eta = estimate_weibull_mle(data)
    mttf = float(eta * gamma(1.0 + 1.0 / beta))
    mtbf = float(mttf + mttr)

    current_rel = float(reliability(current_age, beta, eta))
    mission_rel = float(reliability(current_age + mission_time, beta, eta))
    conditional_rel = conditional_reliability_between(current_age, current_age + mission_time, beta, eta)
    conditional_fail = float(np.clip(1.0 - conditional_rel, 0.0, 1.0))

    optimal_replacement, min_cost_rate, _, _ = cost_based_optimal_replacement(
        data, beta, eta, preventive_cost, failure_cost
    )

    rul = expected_remaining_life(current_age, beta, eta)
    decision = decision_from_metrics(current_age, optimal_replacement, conditional_fail, beta)

    return WeibullResult(
        component=component,
        data=data,
        beta=beta,
        eta=eta,
        mttf=mttf,
        mtbf=mtbf,
        current_reliability=current_rel,
        mission_reliability=mission_rel,
        conditional_reliability=conditional_rel,
        conditional_failure_probability=conditional_fail,
        rul=max(0.0, rul),
        optimal_replacement=optimal_replacement,
        min_cost_rate=min_cost_rate,
        failure_mode=failure_mode_from_beta(beta),
        risk=risk_label(conditional_fail),
        decision=decision,
    )
