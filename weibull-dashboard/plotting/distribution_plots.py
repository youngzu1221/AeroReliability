from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

from core.optimization import DistributionFit, distribution_cdf, distribution_hazard, distribution_pdf
from core.reliability import WeibullResult
from core.weibull_math import EPS, distribution_time_limits
from plotting.styling import apply_axis_bounds_and_units, axis_limits, axis_limits_with_margin, plot_axis_config, plot_padding

MODEL_COLORS = {
    "Weibull": "#0b7285",
    "Lognormal": "tab:orange",
    "Gamma": "tab:purple",
    "Gumbel": "tab:red",
    "Exponential": "tab:green",
    "Normal": "tab:brown",
    "Rayleigh": "tab:pink",
}
OBSERVATION_COLOR = "#1d3557"
CHARACTERISTIC_COLOR = "#e63946"


def _ordered_distribution_fits(result: WeibullResult) -> list[DistributionFit]:
    ranking = {fit.name: idx for idx, fit in enumerate(result.distribution_fits)}
    return sorted(result.distribution_fits, key=lambda fit: ranking.get(fit.name, len(ranking)))


def _distribution_plot_context(result: WeibullResult) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    distribution_start, distribution_end = distribution_time_limits(result.data)
    distribution_range = np.linspace(distribution_start, distribution_end, 1200)
    ranks = np.arange(1, len(result.data) + 1)
    median_ranks = (ranks - 0.3) / (len(result.data) + 0.4)
    return result.data, distribution_range, median_ranks


def _observation_pdf_points(result: WeibullResult, fit: DistributionFit) -> np.ndarray:
    return np.asarray(distribution_pdf(fit.name, fit.params, result.data), dtype=float)


def build_distribution_fit_figure(
    result: WeibullResult,
    adjustment_target: str,
    x_pad: float,
    y_pad: float,
    axis_config: dict[str, float | bool],
    for_pdf: bool = False,
) -> plt.Figure:
    data, distribution_range, median_ranks = _distribution_plot_context(result)
    fit = result.selected_fit
    color = MODEL_COLORS.get(fit.name, "#1f77b4")
    characteristic_value = float(result.characteristic_value)

    pdf_curve = np.asarray(distribution_pdf(fit.name, fit.params, distribution_range), dtype=float)
    cdf_curve = np.asarray(distribution_cdf(fit.name, fit.params, distribution_range), dtype=float)
    pdf_points = _observation_pdf_points(result, fit)

    pdf_x_pad, pdf_y_pad = plot_padding(adjustment_target, "PDF", x_pad, y_pad)
    cdf_x_pad, cdf_y_pad = plot_padding(adjustment_target, "CDF", x_pad, y_pad)
    pdf_axis_config = plot_axis_config(adjustment_target, "PDF", axis_config)
    cdf_axis_config = plot_axis_config(adjustment_target, "CDF", axis_config)

    x_low, x_high = axis_limits(float(distribution_range.min()), float(distribution_range.max()), pdf_x_pad)
    pdf_y_low, pdf_y_high = axis_limits_with_margin(0.0, float(max(np.max(pdf_curve), np.max(pdf_points))), pdf_y_pad, margin=0.04)
    cdf_y_margin = 0.03 + cdf_y_pad

    fig, axes = plt.subplots(1, 2, figsize=(11.7, 4.9 if for_pdf else 4.6))

    axes[0].plot(distribution_range, pdf_curve, color=color, linewidth=2.0, label=f"{fit.name} fit")
    axes[0].scatter(data, pdf_points, s=28, color=OBSERVATION_COLOR, label="Observations", zorder=3)
    axes[0].axvline(characteristic_value, color=CHARACTERISTIC_COLOR, linestyle="--", linewidth=1.6, label="Characteristic value")
    axes[0].set_title("PDF")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Density")
    apply_axis_bounds_and_units(axes[0], pdf_axis_config, (x_low, x_high), (pdf_y_low, pdf_y_high))
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend(fontsize=9)

    x_low_cdf, x_high_cdf = axis_limits(float(distribution_range.min()), float(distribution_range.max()), cdf_x_pad)
    axes[1].plot(distribution_range, cdf_curve, color=color, linewidth=2.0, label=f"{fit.name} fit")
    axes[1].scatter(data, median_ranks, s=28, color=OBSERVATION_COLOR, label="Observations", zorder=3)
    axes[1].axvline(characteristic_value, color=CHARACTERISTIC_COLOR, linestyle="--", linewidth=1.6, label="Characteristic value")
    axes[1].set_title("CDF")
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Probability")
    apply_axis_bounds_and_units(axes[1], cdf_axis_config, (x_low_cdf, x_high_cdf), (-cdf_y_margin, 1.0 + cdf_y_margin))
    axes[1].grid(True, linestyle="--", alpha=0.4)
    axes[1].legend(fontsize=9)

    fig.suptitle(f"Distribution Fit - {result.component} ({fit.name})", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    return fig


def build_distribution_comparison_figure(
    result: WeibullResult,
    adjustment_target: str,
    x_pad: float,
    y_pad: float,
    axis_config: dict[str, float | bool],
    for_pdf: bool = False,
) -> plt.Figure:
    data, distribution_range, median_ranks = _distribution_plot_context(result)
    ordered_fits = _ordered_distribution_fits(result)
    selected_fit = result.selected_fit
    selected_points = _observation_pdf_points(result, selected_fit)

    pdf_x_pad, pdf_y_pad = plot_padding(adjustment_target, "PDF", x_pad, y_pad)
    cdf_x_pad, cdf_y_pad = plot_padding(adjustment_target, "CDF", x_pad, y_pad)
    pdf_axis_config = plot_axis_config(adjustment_target, "PDF", axis_config)
    cdf_axis_config = plot_axis_config(adjustment_target, "CDF", axis_config)

    pdf_peak = max(float(np.max(distribution_pdf(fit.name, fit.params, distribution_range))) for fit in ordered_fits)
    x_low, x_high = axis_limits(float(distribution_range.min()), float(distribution_range.max()), pdf_x_pad)
    pdf_y_low, pdf_y_high = axis_limits_with_margin(0.0, float(max(pdf_peak, np.max(selected_points))), pdf_y_pad, margin=0.04)
    cdf_y_margin = 0.03 + cdf_y_pad

    fig, axes = plt.subplots(1, 2, figsize=(11.7, 4.9 if for_pdf else 4.6))

    for fit in ordered_fits:
        color = MODEL_COLORS.get(fit.name, None)
        axes[0].plot(distribution_range, distribution_pdf(fit.name, fit.params, distribution_range), color=color, linewidth=1.8, label=fit.name)
    axes[0].scatter(data, selected_points, s=24, color=OBSERVATION_COLOR, label="Observations", zorder=3)
    axes[0].set_title("PDF Comparison")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Density")
    apply_axis_bounds_and_units(axes[0], pdf_axis_config, (x_low, x_high), (pdf_y_low, pdf_y_high))
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend(fontsize=8)

    x_low_cdf, x_high_cdf = axis_limits(float(distribution_range.min()), float(distribution_range.max()), cdf_x_pad)
    for fit in ordered_fits:
        color = MODEL_COLORS.get(fit.name, None)
        axes[1].plot(distribution_range, distribution_cdf(fit.name, fit.params, distribution_range), color=color, linewidth=1.8, label=fit.name)
    axes[1].scatter(data, median_ranks, s=24, color=OBSERVATION_COLOR, label="Observations", zorder=3)
    axes[1].set_title("CDF Comparison")
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Probability")
    apply_axis_bounds_and_units(axes[1], cdf_axis_config, (x_low_cdf, x_high_cdf), (-cdf_y_margin, 1.0 + cdf_y_margin))
    axes[1].grid(True, linestyle="--", alpha=0.4)
    axes[1].legend(fontsize=8)

    fig.suptitle(f"Distribution Comparison - {result.component}", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    return fig


def build_hazard_function_figure(
    result: WeibullResult,
    adjustment_target: str,
    x_pad: float,
    y_pad: float,
    axis_config: dict[str, float | bool],
) -> plt.Figure:
    fit = result.selected_fit
    start = max(EPS, float(np.min(result.data)) * 0.5)
    end = max(float(np.max(result.data)) * 1.8, result.characteristic_value * 1.3, start + 1.0)
    hazard_range = np.linspace(start, end, 1000)
    hazard_values = np.asarray(distribution_hazard(fit.name, fit.params, hazard_range), dtype=float)

    haz_x_pad, haz_y_pad = plot_padding(adjustment_target, "Hazard Function", x_pad, y_pad)
    haz_axis_config = plot_axis_config(adjustment_target, "Hazard Function", axis_config)
    x_low, x_high = axis_limits(float(hazard_range.min()), float(hazard_range.max()), haz_x_pad)
    y_low, y_high = axis_limits_with_margin(float(np.min(hazard_values)), float(np.max(hazard_values)), haz_y_pad, margin=0.04)

    fig, ax = plt.subplots(figsize=(11.7, 4.0))
    ax.plot(hazard_range, hazard_values, color=MODEL_COLORS.get(fit.name, "#1f77b4"), linewidth=2.0, label=f"{fit.name} hazard")
    ax.axvline(result.characteristic_value, color=CHARACTERISTIC_COLOR, linestyle="--", linewidth=1.6, label="Characteristic value")
    ax.set_title("Hazard Function")
    ax.set_xlabel("Time")
    ax.set_ylabel("Failure rate")
    apply_axis_bounds_and_units(ax, haz_axis_config, (x_low, x_high), (y_low, y_high))
    if not haz_axis_config.get("use_major_units"):
        ax.xaxis.set_major_locator(MultipleLocator(500.0))
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


def build_forward_risk_figure(
    result: WeibullResult,
    current_age: float,
    horizon: float,
    adjustment_target: str,
    x_pad: float,
    y_pad: float,
    axis_config: dict[str, float | bool],
) -> plt.Figure:
    requested_horizon = max(float(horizon), 0.0)
    plot_horizon = max(requested_horizon, 1.0)
    future_times = np.linspace(max(EPS, current_age), current_age + plot_horizon, 250)
    fit = result.selected_fit

    current_rel = float(distribution_reliability(fit.name, fit.params, current_age))
    if requested_horizon <= 0:
        conditional_rel = np.ones_like(future_times)
        conditional_fail = np.zeros_like(future_times)
        haz = np.full_like(future_times, float(distribution_hazard(fit.name, fit.params, current_age)))
        rel_low = rel_high = conditional_rel
        fail_low = fail_high = conditional_fail
    else:
        rel = distribution_reliability(fit.name, fit.params, future_times)
        conditional_rel = rel / current_rel if current_rel > EPS else np.zeros_like(rel)
        conditional_rel = np.clip(conditional_rel, 0.0, 1.0)
        conditional_fail = 1.0 - conditional_rel
        haz = distribution_hazard(fit.name, fit.params, future_times)

        param_samples = result.bootstrap_param_samples
        bootstrap_rel = np.asarray(
            [
                distribution_reliability(fit.name, tuple(sample), future_times)
                for sample in param_samples
            ],
            dtype=float,
        )
        bootstrap_current_rel = np.asarray(
            [float(distribution_reliability(fit.name, tuple(sample), current_age)) for sample in param_samples],
            dtype=float,
        )
        bootstrap_conditional_rel = np.divide(
            bootstrap_rel,
            bootstrap_current_rel[:, None],
            out=np.zeros_like(bootstrap_rel),
            where=bootstrap_current_rel[:, None] > EPS,
        )
        bootstrap_conditional_rel = np.clip(bootstrap_conditional_rel, 0.0, 1.0)
        bootstrap_conditional_fail = 1.0 - bootstrap_conditional_rel
        rel_low, rel_high = np.quantile(bootstrap_conditional_rel, [0.025, 0.975], axis=0)
        fail_low, fail_high = np.quantile(bootstrap_conditional_fail, [0.025, 0.975], axis=0)

    rel_x_pad, rel_y_pad = plot_padding(adjustment_target, "Conditional Reliability", x_pad, y_pad)
    fail_x_pad, fail_y_pad = plot_padding(adjustment_target, "Conditional Failure Probability", x_pad, y_pad)
    haz_x_pad, haz_y_pad = plot_padding(adjustment_target, "Hazard Trend", x_pad, y_pad)
    rel_axis_config = plot_axis_config(adjustment_target, "Conditional Reliability", axis_config)
    fail_axis_config = plot_axis_config(adjustment_target, "Conditional Failure Probability", axis_config)
    haz_axis_config = plot_axis_config(adjustment_target, "Hazard Trend", axis_config)

    rel_x_low, rel_x_high = axis_limits(float(future_times.min()), float(future_times.max()), rel_x_pad)
    fail_x_low, fail_x_high = axis_limits(float(future_times.min()), float(future_times.max()), fail_x_pad)
    haz_x_low, haz_x_high = axis_limits(float(future_times.min()), float(future_times.max()), haz_x_pad)
    haz_y_low, haz_y_high = axis_limits_with_margin(float(np.min(haz)), float(np.max(haz)), haz_y_pad, margin=0.04)

    fig, axes = plt.subplots(1, 3, figsize=(11.7, 4.2))
    rel_y_margin = 0.03 + rel_y_pad
    fail_y_margin = 0.03 + fail_y_pad

    axes[0].fill_between(future_times, rel_low, rel_high, color="tab:blue", alpha=0.18, label="95% CI")
    axes[0].plot(future_times, conditional_rel, color="tab:blue", label="Estimate")
    axes[0].axvline(current_age, color="black", linestyle="--", alpha=0.65)
    axes[0].set_title("Conditional Reliability")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Reliability")
    apply_axis_bounds_and_units(axes[0], rel_axis_config, (rel_x_low, rel_x_high), (-rel_y_margin, 1.0 + rel_y_margin))
    if not rel_axis_config.get("use_major_units"):
        axes[0].yaxis.set_major_locator(MultipleLocator(0.1))
    axes[0].grid(True, linestyle="--", alpha=0.45)
    axes[0].legend(fontsize=8)

    axes[1].fill_between(future_times, fail_low, fail_high, color="tab:red", alpha=0.18, label="95% CI")
    axes[1].plot(future_times, conditional_fail, color="tab:red", label="Estimate")
    axes[1].axvline(current_age, color="black", linestyle="--", alpha=0.65)
    axes[1].set_title("Conditional Failure Probability")
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Probability")
    apply_axis_bounds_and_units(axes[1], fail_axis_config, (fail_x_low, fail_x_high), (-fail_y_margin, 1.0 + fail_y_margin))
    if not fail_axis_config.get("use_major_units"):
        axes[1].yaxis.set_major_locator(MultipleLocator(0.1))
    axes[1].grid(True, linestyle="--", alpha=0.45)
    axes[1].legend(fontsize=8)

    axes[2].plot(future_times, haz, color=MODEL_COLORS.get(fit.name, "#1f77b4"))
    axes[2].axvline(current_age, color="black", linestyle="--", alpha=0.65)
    axes[2].set_title("Hazard Trend")
    axes[2].set_xlabel("Time")
    axes[2].set_ylabel("Failure rate")
    apply_axis_bounds_and_units(axes[2], haz_axis_config, (haz_x_low, haz_x_high), (haz_y_low, haz_y_high))
    axes[2].grid(True, linestyle="--", alpha=0.45)

    fig.suptitle(f"Forward Risk Trend - {result.component}", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


def distribution_reliability(distribution_name: str, params: tuple[float, ...], time_values: np.ndarray | float) -> np.ndarray:
    return 1.0 - np.asarray(distribution_cdf(distribution_name, params, time_values), dtype=float)
