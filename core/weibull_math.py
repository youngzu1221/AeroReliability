from __future__ import annotations

import numpy as np

EPS = 1e-12


def reliability(t: np.ndarray | float, beta: float, eta: float) -> np.ndarray:
    t_arr = np.maximum(np.asarray(t, dtype=float), EPS)
    return np.exp(-((t_arr / eta) ** beta))


def hazard(t: np.ndarray | float, beta: float, eta: float) -> np.ndarray:
    t_arr = np.maximum(np.asarray(t, dtype=float), EPS)
    return (beta / eta) * (t_arr / eta) ** (beta - 1.0)


def weibull_pdf(t: np.ndarray | float, beta: float, eta: float) -> np.ndarray:
    t_arr = np.maximum(np.asarray(t, dtype=float), EPS)
    return hazard(t_arr, beta, eta) * reliability(t_arr, beta, eta)


def weibull_cdf(t: np.ndarray | float, beta: float, eta: float) -> np.ndarray:
    return 1.0 - reliability(t, beta, eta)


def weibull_quantile(probability: float, beta: float, eta: float) -> float:
    p = float(np.clip(probability, EPS, 1.0 - EPS))
    return float(eta * (-np.log(1.0 - p)) ** (1.0 / beta))


def distribution_time_limits(data: np.ndarray) -> tuple[float, float]:
    data_min = float(np.min(data))
    data_max = float(np.max(data))
    low = max(EPS, data_min * 0.5)
    high = max(data_max * 1.2, low + 1.0)
    return low, high


def failure_mode_from_beta(beta: float) -> str:
    if beta < 0.95:
        return "Infant mortality"
    if beta <= 1.05:
        return "Random failure"
    return "Wear-out"
