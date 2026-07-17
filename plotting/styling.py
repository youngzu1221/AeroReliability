from __future__ import annotations

from matplotlib.ticker import MultipleLocator

PLOT_ADJUSTMENT_TARGETS = [
    "All plots",
    "PDF",
    "CDF",
    "Hazard Function",
    "Conditional Reliability",
    "Conditional Failure Probability",
    "Hazard Trend",
    "Cost Rate",
    "None",
]


def axis_limits(low: float, high: float, pad: float) -> tuple[float, float]:
    span = high - low
    if span <= 0:
        span = max(1.0, abs(high), 1.0)
    return low - span * pad, high + span * pad


def axis_limits_with_margin(low: float, high: float, pad: float, margin: float = 0.03) -> tuple[float, float]:
    return axis_limits(low, high, pad + margin)


def plot_padding(target: str, plot_name: str, x_pad: float, y_pad: float) -> tuple[float, float]:
    if target == "All plots" or target == plot_name:
        return x_pad, y_pad
    return 0.0, 0.0


def plot_axis_config(target: str, plot_name: str, axis_config: dict[str, float | bool]) -> dict[str, float | bool]:
    if target == "All plots" or target == plot_name:
        return axis_config
    return {}


def apply_axis_bounds_and_units(
    ax,
    axis_config: dict[str, float | bool],
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
) -> None:
    x_low, x_high = x_limits
    y_low, y_high = y_limits

    if axis_config.get("use_x_bounds"):
        configured_min = float(axis_config.get("x_min", x_low))
        configured_max = float(axis_config.get("x_max", x_high))
        if configured_max > configured_min:
            x_low, x_high = configured_min, configured_max

    if axis_config.get("use_y_bounds"):
        configured_min = float(axis_config.get("y_min", y_low))
        configured_max = float(axis_config.get("y_max", y_high))
        if configured_max > configured_min:
            y_low, y_high = configured_min, configured_max

    ax.set_xlim(x_low, x_high)
    ax.set_ylim(y_low, y_high)

    if axis_config.get("use_major_units"):
        x_major = float(axis_config.get("x_major_unit", 0.0))
        y_major = float(axis_config.get("y_major_unit", 0.0))
        if x_major > 0:
            ax.xaxis.set_major_locator(MultipleLocator(x_major))
        if y_major > 0:
            ax.yaxis.set_major_locator(MultipleLocator(y_major))
