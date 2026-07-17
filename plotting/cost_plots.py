from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from core.reliability import WeibullResult, cost_based_optimal_replacement
from plotting.styling import (
    apply_axis_bounds_and_units,
    axis_limits,
    axis_limits_with_margin,
    nice_major_unit,
    plot_axis_config,
    plot_padding,
)


def build_cost_curve_figure(
    result: WeibullResult,
    preventive_cost: float,
    failure_cost: float,
    adjustment_target: str,
    x_pad: float,
    y_pad: float,
    axis_config: dict[str, float | bool],
) -> plt.Figure:
    t_range, cost_rate = cost_based_optimal_replacement(
        result.data,
        result.selected_distribution,
        result.selected_fit.params,
        preventive_cost,
        failure_cost,
        result.characteristic_value,
    )[2:]

    display_end = min(
        float(t_range.max()),
        max(
            result.optimal_replacement * 2.25,
            result.characteristic_value * 1.75,
            float(result.data.max()) * 1.35,
            1.0,
        ),
    )
    display_mask = t_range <= display_end
    plot_times = t_range[display_mask]
    plot_cost_rate = cost_rate[display_mask]

    cost_x_pad, cost_y_pad = plot_padding(adjustment_target, "Cost Rate", x_pad, y_pad)
    cost_axis_config = plot_axis_config(adjustment_target, "Cost Rate", axis_config)
    x_low, x_high = axis_limits(float(plot_times.min()), float(plot_times.max()), cost_x_pad)
    x_low = 0.0
    y_low, y_high = axis_limits_with_margin(float(plot_cost_rate.min()), float(plot_cost_rate.max()), cost_y_pad, margin=0.04)

    fig, ax = plt.subplots(figsize=(11.7, 4.2))
    ax.plot(plot_times, plot_cost_rate, linewidth=2.0)
    ax.scatter([result.optimal_replacement], [result.min_cost_rate], color="crimson", s=42, zorder=3, label="Optimal replacement")
    ax.axvline(result.optimal_replacement, color="crimson", linestyle="--", linewidth=1.6)
    ax.annotate(
        f"{result.optimal_replacement:,.0f}",
        xy=(result.optimal_replacement, result.min_cost_rate),
        xytext=(8, 10),
        textcoords="offset points",
        color="crimson",
        fontsize=9,
    )
    ax.set_title(f"Replacement Economics - {result.component}")
    ax.set_xlabel("Replacement time")
    ax.set_ylabel("Expected cost rate ($/time)")
    apply_axis_bounds_and_units(ax, cost_axis_config, (x_low, x_high), (y_low, y_high))
    if not cost_axis_config.get("use_major_units"):
        x_major_unit = nice_major_unit(x_high - x_low, target_ticks=6)
        ax.set_xticks(np.arange(0.0, x_high + (0.5 * x_major_unit), x_major_unit))
        ax.set_xlim(x_low, x_high)
    ax.ticklabel_format(style="plain", axis="x", useOffset=False)
    ax.grid(True, linestyle="--", alpha=0.45)
    ax.legend()
    fig.tight_layout()
    return fig
