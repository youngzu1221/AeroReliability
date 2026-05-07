from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MultipleLocator
from scipy.optimize import minimize
from scipy.special import gamma


# ============================================================
# PAGE SETUP
# ============================================================
st.set_page_config(
    page_title="Reliability & Weibull Pro Dashboard",
    page_icon="🔧",
    layout="wide",
)


# ============================================================
# CONSTANTS
# ============================================================
EPS = 1e-12
RISK_THRESHOLDS = {
    "high": 0.60,
    "medium": 0.30,
}


# ============================================================
# MODELS
# ============================================================
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


# ============================================================
# NUMERIC HELPERS
# ============================================================
def clean_life_data(values: pd.Series | np.ndarray) -> np.ndarray:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    arr = arr[arr > 0]
    return np.sort(arr)


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

    upper = max(
        current_age + 10.0 * eta,
        weibull_quantile(0.999999, beta, eta),
        current_age + 1.0,
    )
    upper = max(upper, current_age + EPS)
    grid = np.linspace(current_age, upper, 4000)
    surv = reliability(grid, beta, eta)
    trapezoid = getattr(np, "trapezoid", None)
    if trapezoid is not None:
        area = float(trapezoid(surv, grid))
    else:
        area = float(np.trapz(surv, grid))
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

    initial_guess = np.array([1.5, max(float(np.mean(data)), EPS)])
    result = minimize(
        neg_log_likelihood,
        x0=initial_guess,
        args=(data,),
        method="L-BFGS-B",
        bounds=[(1e-6, None), (1e-6, None)],
    )

    if not result.success:
        raise RuntimeError(str(result.message))

    beta, eta = result.x
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


def axis_limits(low: float, high: float, pad: float) -> tuple[float, float]:
    span = high - low
    if span <= 0:
        span = max(1.0, abs(high), 1.0)
    return low - span * pad, high + span * pad


def axis_limits_with_margin(low: float, high: float, pad: float, margin: float = 0.03) -> tuple[float, float]:
    return axis_limits(low, high, pad + margin)


def distribution_time_limits(data: np.ndarray) -> tuple[float, float]:
    data_min = float(np.min(data))
    data_max = float(np.max(data))
    low = max(EPS, data_min * 0.5)
    high = max(data_max * 1.2, low + 1.0)
    return low, high


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


def plot_padding(target: str, plot_name: str, x_pad: float, y_pad: float) -> tuple[float, float]:
    if target == "All plots" or target == plot_name:
        return x_pad, y_pad
    return 0.0, 0.0


def plot_axis_config(target: str, plot_name: str, axis_config: dict[str, float | bool]) -> dict[str, float | bool]:
    if target == "All plots" or target == plot_name:
        return axis_config
    return {}


def state_key(value: str) -> str:
    return value.lower().replace(" ", "_").replace("/", "_")


def default_plot_padding(target: str) -> float:
    return 0.05 if target == "All plots" else 0.0


def sync_adjustment(source_key: str, canonical_key: str, mirror_key: str) -> None:
    value = float(st.session_state[source_key])
    st.session_state[canonical_key] = value
    st.session_state[mirror_key] = value


def reset_active_axis_adjustment() -> None:
    target = st.session_state.get("axis_adjustment_target", "All plots")
    default_value = default_plot_padding(target)

    for key in (
        "axis_x_pad",
        "axis_x_pad_slider",
        "axis_x_pad_number",
        "axis_y_pad",
        "axis_y_pad_slider",
        "axis_y_pad_number",
    ):
        st.session_state[key] = default_value

    for key in (
        "axis_use_x_bounds",
        "axis_x_min",
        "axis_x_max",
        "axis_use_y_bounds",
        "axis_y_min",
        "axis_y_max",
        "axis_use_major_units",
        "axis_x_major_unit",
        "axis_y_major_unit",
    ):
        st.session_state.pop(key, None)


def adjustment_control(label: str, canonical_key: str) -> float:
    slider_key = f"{canonical_key}_slider"
    number_key = f"{canonical_key}_number"

    st.session_state.setdefault(slider_key, float(st.session_state[canonical_key]))
    st.session_state.setdefault(number_key, float(st.session_state[canonical_key]))

    slider_col, number_col = st.columns([0.68, 0.32])
    with slider_col:
        st.slider(
            label,
            0.0,
            1.0,
            step=0.01,
            key=slider_key,
            on_change=sync_adjustment,
            args=(slider_key, canonical_key, number_key),
        )
    with number_col:
        st.number_input(
            f"{label} value",
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            format="%.2f",
            key=number_key,
            label_visibility="collapsed",
            on_change=sync_adjustment,
            args=(number_key, canonical_key, slider_key),
        )

    return float(st.session_state[canonical_key])


def axis_bounds_and_units_control() -> dict[str, float | bool]:
    with st.expander("Axis bounds and units", expanded=False):
        st.caption("Excel-style manual bounds and major units for the selected plot target.")

        use_x_bounds = st.checkbox("Custom X-axis bounds", key="axis_use_x_bounds")
        x_col_1, x_col_2 = st.columns(2)
        with x_col_1:
            x_min = st.number_input("X minimum", value=0.0, step=100.0, disabled=not use_x_bounds, key="axis_x_min")
        with x_col_2:
            x_max = st.number_input("X maximum", value=1000.0, step=100.0, disabled=not use_x_bounds, key="axis_x_max")

        use_y_bounds = st.checkbox("Custom Y-axis bounds", key="axis_use_y_bounds")
        y_col_1, y_col_2 = st.columns(2)
        with y_col_1:
            y_min = st.number_input("Y minimum", value=0.0, step=0.01, format="%.6f", disabled=not use_y_bounds, key="axis_y_min")
        with y_col_2:
            y_max = st.number_input("Y maximum", value=1.0, step=0.01, format="%.6f", disabled=not use_y_bounds, key="axis_y_max")

        use_major_units = st.checkbox("Custom major units", key="axis_use_major_units")
        u_col_1, u_col_2 = st.columns(2)
        with u_col_1:
            x_major_unit = st.number_input(
                "X major unit",
                min_value=0.0,
                value=100.0,
                step=100.0,
                disabled=not use_major_units,
                key="axis_x_major_unit",
            )
        with u_col_2:
            y_major_unit = st.number_input(
                "Y major unit",
                min_value=0.0,
                value=0.10,
                step=0.01,
                format="%.6f",
                disabled=not use_major_units,
                key="axis_y_major_unit",
            )

    return {
        "use_x_bounds": use_x_bounds,
        "x_min": float(x_min),
        "x_max": float(x_max),
        "use_y_bounds": use_y_bounds,
        "y_min": float(y_min),
        "y_max": float(y_max),
        "use_major_units": use_major_units,
        "x_major_unit": float(x_major_unit),
        "y_major_unit": float(y_major_unit),
    }


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


# ============================================================
# DATA IO
# ============================================================
@st.cache_data(show_spinner=False)
def parse_excel(uploaded_file) -> dict[str, np.ndarray]:
    df = pd.read_excel(uploaded_file)
    datasets: dict[str, np.ndarray] = {}

    for col in df.columns:
        cleaned = clean_life_data(df[col])
        if len(cleaned) > 1:
            datasets[str(col)] = cleaned

    return datasets


def build_template_workbook() -> bytes:
    example = pd.DataFrame(
        {
            "Pump A": [1200, 1450, 1680, 1710, 2100, 2380],
            "Motor B": [800, 950, 1010, 1250, 1360, 1510],
            "Bearing C": [300, 420, 510, 740, 860, 930],
        }
    )
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        example.to_excel(writer, index=False, sheet_name="failure_times")
    return buffer.getvalue()


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
        data,
        beta,
        eta,
        preventive_cost,
        failure_cost,
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


def results_to_frame(results: dict[str, WeibullResult]) -> pd.DataFrame:
    rows = []
    for result in results.values():
        rows.append(
            {
                "Component": result.component,
                "Beta": result.beta,
                "Eta": result.eta,
                "MTTF": result.mttf,
                "MTBF": result.mtbf,
                "Conditional Reliability": result.conditional_reliability,
                "Conditional Probability of Failure": result.conditional_failure_probability,
                "RUL": result.rul,
                "Optimal Replacement": result.optimal_replacement,
                "Min Cost Rate": result.min_cost_rate,
                "Failure Mode": result.failure_mode,
                "Risk": result.risk,
                "Decision": result.decision,
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values("Conditional Probability of Failure", ascending=False)
        .reset_index(drop=True)
    )


def display_results_table(df: pd.DataFrame) -> None:
    display_df = df.copy()
    pct_cols = [
        "Conditional Reliability",
        "Conditional Probability of Failure",
    ]
    numeric_cols = ["Beta", "Eta", "MTTF", "MTBF", "RUL", "Optimal Replacement"]
    money_cols = ["Min Cost Rate"]

    for col in pct_cols:
        display_df[col] = display_df[col].map(fmt_pct)
    for col in numeric_cols:
        display_df[col] = display_df[col].map(lambda x: f"{x:,.2f}")
    for col in money_cols:
        display_df[col] = display_df[col].map(fmt_money)

    st.dataframe(display_df, use_container_width=True, hide_index=True)


def formatted_results_frame(df: pd.DataFrame) -> pd.DataFrame:
    display_df = df.copy()
    pct_cols = [
        "Conditional Reliability",
        "Conditional Probability of Failure",
    ]
    numeric_cols = ["Beta", "Eta", "MTTF", "MTBF", "RUL", "Optimal Replacement"]
    money_cols = ["Min Cost Rate"]

    for col in pct_cols:
        display_df[col] = display_df[col].map(fmt_pct)
    for col in numeric_cols:
        display_df[col] = display_df[col].map(lambda x: f"{x:,.2f}")
    for col in money_cols:
        display_df[col] = display_df[col].map(fmt_money)

    return display_df


def highest_risk_component_text(df: pd.DataFrame) -> str:
    high_risk_components = df.loc[df["Risk"] == "HIGH", "Component"].astype(str).tolist()
    if high_risk_components:
        return ", ".join(high_risk_components)

    highest_probability = float(df["Conditional Probability of Failure"].max())
    highest_components = df.loc[
        np.isclose(df["Conditional Probability of Failure"], highest_probability),
        "Component",
    ].astype(str)
    return ", ".join(highest_components.tolist())


def safe_filename(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value.strip())
    return safe or "component"


def add_dataframe_page(pdf: PdfPages, title: str, df: pd.DataFrame, rows_per_page: int = 18) -> None:
    page_count = max(1, int(np.ceil(len(df) / rows_per_page)))

    for page_index in range(page_count):
        page_df = df.iloc[page_index * rows_per_page : (page_index + 1) * rows_per_page]
        fig, ax = plt.subplots(figsize=(11.7, 8.3))
        ax.axis("off")
        ax.set_title(
            f"{title} ({page_index + 1}/{page_count})" if page_count > 1 else title,
            fontsize=16,
            fontweight="bold",
            pad=18,
        )
        table = ax.table(
            cellText=page_df.values,
            colLabels=page_df.columns,
            loc="center",
            cellLoc="center",
            colLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.scale(1.0, 1.35)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def build_fleet_summary_pdf(df_results: pd.DataFrame) -> bytes:
    buffer = BytesIO()
    with PdfPages(buffer) as pdf:
        fig, ax = plt.subplots(figsize=(11.7, 8.3))
        ax.axis("off")
        top_row = df_results.iloc[0]
        summary_lines = [
            "Reliability & Weibull Fleet Summary",
            "",
            f"Components analyzed: {len(df_results)}",
            f"Highest failure probability: {fmt_pct(top_row['Conditional Probability of Failure'])}",
            f"Highest risk component(s): {highest_risk_component_text(df_results)}",
            f"Average beta: {df_results['Beta'].mean():.2f}",
            f"Average MTTF: {df_results['MTTF'].mean():,.2f}",
        ]
        ax.text(0.05, 0.90, summary_lines[0], fontsize=20, fontweight="bold", transform=ax.transAxes)
        ax.text(0.05, 0.78, "\n".join(summary_lines[2:]), fontsize=13, va="top", transform=ax.transAxes)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        add_dataframe_page(pdf, "Full Reliability Results", formatted_results_frame(df_results))

    buffer.seek(0)
    return buffer.getvalue()


def build_component_deep_dive_pdf(
    result: WeibullResult,
    current_age: float,
    horizon: float,
    preventive_cost: float,
    failure_cost: float,
    adjustment_target: str,
    x_pad: float,
    y_pad: float,
    axis_config: dict[str, float | bool] | None = None,
) -> bytes:
    axis_config = axis_config or {}
    buffer = BytesIO()
    with PdfPages(buffer) as pdf:
        fig, ax = plt.subplots(figsize=(11.7, 8.3))
        ax.axis("off")
        details = [
            f"Component: {result.component}",
            f"Risk: {result.risk}",
            f"Conditional failure probability: {fmt_pct(result.conditional_failure_probability)}",
            f"Conditional reliability: {fmt_pct(result.conditional_reliability)}",
            f"Beta: {result.beta:.3f}",
            f"Eta: {result.eta:,.2f}",
            f"MTTF: {result.mttf:,.2f}",
            f"MTBF: {result.mtbf:,.2f}",
            f"RUL: {result.rul:,.2f}",
            f"Optimal replacement: {result.optimal_replacement:,.2f}",
            f"Preventive cost: {fmt_money(preventive_cost)}",
            f"Failure cost: {fmt_money(failure_cost)}",
            f"Minimum cost rate: {fmt_money(result.min_cost_rate)}",
            f"Failure mode: {result.failure_mode}",
        ]
        ax.text(0.05, 0.90, "Component Deep Dive", fontsize=20, fontweight="bold", transform=ax.transAxes)
        ax.text(0.05, 0.80, "\n".join(details), fontsize=13, va="top", transform=ax.transAxes)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        distribution_fig = build_distribution_fit_figure(result, current_age, adjustment_target, x_pad, y_pad, axis_config, for_pdf=True)
        pdf.savefig(distribution_fig, bbox_inches="tight")
        plt.close(distribution_fig)

        forward_fig = build_forward_risk_figure(result, current_age, horizon, adjustment_target, x_pad, y_pad, axis_config)
        pdf.savefig(forward_fig, bbox_inches="tight")
        plt.close(forward_fig)

        cost_fig = build_cost_curve_figure(result, preventive_cost, failure_cost, adjustment_target, x_pad, y_pad, axis_config)
        pdf.savefig(cost_fig, bbox_inches="tight")
        plt.close(cost_fig)

    buffer.seek(0)
    return buffer.getvalue()


def build_reliability_report_pdf(
    df_results: pd.DataFrame,
    result: WeibullResult,
    current_age: float,
    horizon: float,
    preventive_cost: float,
    failure_cost: float,
    adjustment_target: str,
    x_pad: float,
    y_pad: float,
    axis_config: dict[str, float | bool],
) -> bytes:
    buffer = BytesIO()
    with PdfPages(buffer) as pdf:
        fig, ax = plt.subplots(figsize=(11.7, 8.3))
        ax.axis("off")
        top_row = df_results.iloc[0]
        summary_lines = [
            "Reliability & Weibull Report",
            "",
            "Fleet Summary",
            f"Components analyzed: {len(df_results)}",
            f"Highest failure probability: {fmt_pct(top_row['Conditional Probability of Failure'])}",
            f"Highest risk component(s): {highest_risk_component_text(df_results)}",
            f"Average beta: {df_results['Beta'].mean():.2f}",
            f"Average MTTF: {df_results['MTTF'].mean():,.2f}",
            "",
            f"Selected component: {result.component}",
            f"Preventive cost: {fmt_money(preventive_cost)}",
            f"Failure cost: {fmt_money(failure_cost)}",
            f"Minimum cost rate: {fmt_money(result.min_cost_rate)}",
        ]
        ax.text(0.05, 0.90, summary_lines[0], fontsize=20, fontweight="bold", transform=ax.transAxes)
        ax.text(0.05, 0.80, "\n".join(summary_lines[2:]), fontsize=13, va="top", transform=ax.transAxes)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        add_dataframe_page(pdf, "Full Reliability Results", formatted_results_frame(df_results))

        fig, ax = plt.subplots(figsize=(11.7, 8.3))
        ax.axis("off")
        details = [
            f"Component: {result.component}",
            f"Risk: {result.risk}",
            f"Conditional failure probability: {fmt_pct(result.conditional_failure_probability)}",
            f"Conditional reliability: {fmt_pct(result.conditional_reliability)}",
            f"Beta: {result.beta:.3f}",
            f"Eta: {result.eta:,.2f}",
            f"MTTF: {result.mttf:,.2f}",
            f"MTBF: {result.mtbf:,.2f}",
            f"RUL: {result.rul:,.2f}",
            f"Optimal replacement: {result.optimal_replacement:,.2f}",
            f"Preventive cost: {fmt_money(preventive_cost)}",
            f"Failure cost: {fmt_money(failure_cost)}",
            f"Minimum cost rate: {fmt_money(result.min_cost_rate)}",
            f"Failure mode: {result.failure_mode}",
        ]
        ax.text(0.05, 0.90, "Component Deep Dive", fontsize=20, fontweight="bold", transform=ax.transAxes)
        ax.text(0.05, 0.80, "\n".join(details), fontsize=13, va="top", transform=ax.transAxes)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        distribution_fig = build_distribution_fit_figure(result, current_age, adjustment_target, x_pad, y_pad, axis_config, for_pdf=True)
        pdf.savefig(distribution_fig, bbox_inches="tight")
        plt.close(distribution_fig)

        forward_fig = build_forward_risk_figure(result, current_age, horizon, adjustment_target, x_pad, y_pad, axis_config)
        pdf.savefig(forward_fig, bbox_inches="tight")
        plt.close(forward_fig)

        cost_fig = build_cost_curve_figure(result, preventive_cost, failure_cost, adjustment_target, x_pad, y_pad, axis_config)
        pdf.savefig(cost_fig, bbox_inches="tight")
        plt.close(cost_fig)

    buffer.seek(0)
    return buffer.getvalue()


# ============================================================
# PLOTTING
# ============================================================
def plot_distribution_fit(
    result: WeibullResult,
    current_age: float,
    adjustment_target: str,
    x_pad: float,
    y_pad: float,
    axis_config: dict[str, float | bool],
) -> None:
    data = result.data
    beta = result.beta
    eta = result.eta

    n = len(data)
    ranks = np.arange(1, n + 1)
    median_ranks = (ranks - 0.3) / (n + 0.4)

    distribution_start, distribution_end = distribution_time_limits(data)
    distribution_range = np.linspace(distribution_start, distribution_end, 1200)

    hazard_start = max(EPS, float(np.min(data)) * 0.5)
    hazard_end = max(float(np.max(data)) * 1.75, eta * 1.5, hazard_start + 1.0)
    hazard_range = np.linspace(hazard_start, hazard_end, 1000)

    pdf_curve = weibull_pdf(distribution_range, beta, eta)
    cdf_curve = weibull_cdf(distribution_range, beta, eta)
    haz_curve = hazard(hazard_range, beta, eta)
    pdf_points = weibull_pdf(data, beta, eta)

    pdf_x_pad, pdf_y_pad = plot_padding(adjustment_target, "PDF", x_pad, y_pad)
    cdf_x_pad, cdf_y_pad = plot_padding(adjustment_target, "CDF", x_pad, y_pad)
    haz_x_pad, haz_y_pad = plot_padding(adjustment_target, "Hazard Function", x_pad, y_pad)
    pdf_axis_config = plot_axis_config(adjustment_target, "PDF", axis_config)
    cdf_axis_config = plot_axis_config(adjustment_target, "CDF", axis_config)
    haz_axis_config = plot_axis_config(adjustment_target, "Hazard Function", axis_config)

    pdf_x_low, pdf_x_high = axis_limits(distribution_start, distribution_end, pdf_x_pad)
    cdf_x_low, cdf_x_high = axis_limits(distribution_start, distribution_end, cdf_x_pad)
    haz_x_low, haz_x_high = axis_limits(hazard_start, hazard_end, haz_x_pad)
    pdf_y_high_value = float(max(np.max(pdf_curve), np.max(pdf_points)))
    pdf_y_low, pdf_y_high = axis_limits_with_margin(0.0, pdf_y_high_value, pdf_y_pad, margin=0.04)
    haz_y_low, haz_y_high = axis_limits_with_margin(
        float(np.min(haz_curve)),
        float(np.max(haz_curve)),
        haz_y_pad,
        margin=0.04,
    )

    col_pdf, col_cdf = st.columns(2)

    fig_pdf, ax_pdf = plt.subplots(figsize=(8, 4))
    ax_pdf.plot(distribution_range, pdf_curve, label="Weibull fit")
    ax_pdf.scatter(data, pdf_points, s=28, label="Observations", clip_on=False)
    ax_pdf.axvline(eta, color="crimson", linestyle="--", alpha=0.75, label="η characteristic life")
    if current_age > 0:
        ax_pdf.axvline(current_age, color="black", linestyle=":", alpha=0.7, label="Current time")
    ax_pdf.set_title("PDF")
    ax_pdf.set_xlabel("Time")
    ax_pdf.set_ylabel("Density")
    apply_axis_bounds_and_units(ax_pdf, pdf_axis_config, (pdf_x_low, pdf_x_high), (pdf_y_low, pdf_y_high))
    ax_pdf.grid(True, linestyle="--", alpha=0.45)
    ax_pdf.legend()
    col_pdf.pyplot(fig_pdf, clear_figure=True)

    fig_cdf, ax_cdf = plt.subplots(figsize=(8, 4))
    ax_cdf.plot(distribution_range, cdf_curve, label="Weibull fit")
    ax_cdf.scatter(data, median_ranks, s=28, label="Median ranks", clip_on=False)
    ax_cdf.axvline(eta, color="crimson", linestyle="--", alpha=0.75, label="η characteristic life")
    if current_age > 0:
        ax_cdf.axvline(current_age, color="black", linestyle=":", alpha=0.7, label="Current time")
    ax_cdf.set_title("CDF / Unreliability")
    ax_cdf.set_xlabel("Time")
    ax_cdf.set_ylabel("Probability")
    cdf_y_margin = 0.03 + cdf_y_pad
    apply_axis_bounds_and_units(ax_cdf, cdf_axis_config, (cdf_x_low, cdf_x_high), (-cdf_y_margin, 1.0 + cdf_y_margin))
    ax_cdf.grid(True, linestyle="--", alpha=0.45)
    ax_cdf.legend()
    col_cdf.pyplot(fig_cdf, clear_figure=True)

    fig_haz, ax_haz = plt.subplots(figsize=(9, 3.2))
    ax_haz.plot(hazard_range, haz_curve, label="Hazard rate")
    ax_haz.set_title("Hazard Function (Failure Rate)")
    ax_haz.set_xlabel("Time")
    ax_haz.set_ylabel("Failure rate (per hour), not probability")
    apply_axis_bounds_and_units(ax_haz, haz_axis_config, (haz_x_low, haz_x_high), (haz_y_low, haz_y_high))
    ax_haz.grid(True, linestyle="--", alpha=0.45)
    ax_haz.legend()
    _, hazard_col, _ = st.columns([0.1, 0.8, 0.1])
    hazard_col.pyplot(fig_haz, clear_figure=True)


def plot_forward_risk(
    result: WeibullResult,
    current_age: float,
    horizon: float,
    adjustment_target: str,
    x_pad: float,
    y_pad: float,
    axis_config: dict[str, float | bool],
) -> None:
    beta = result.beta
    eta = result.eta
    optimal_replacement = result.optimal_replacement
    requested_horizon = max(float(horizon), 0.0)
    plot_horizon = max(requested_horizon, 1.0)
    future_times = np.linspace(max(EPS, current_age), current_age + plot_horizon, 250)

    current_rel = float(reliability(current_age, beta, eta))
    if requested_horizon <= 0:
        conditional_rel = np.ones_like(future_times)
        conditional_fail = np.zeros_like(future_times)
        haz = np.full_like(future_times, float(hazard(current_age, beta, eta)))
    else:
        conditional_rel = np.array(
            [conditional_reliability_between(current_age, t, beta, eta) for t in future_times],
            dtype=float,
        )
        conditional_fail = np.array(
            [conditional_failure_probability_between(current_age, t, beta, eta) for t in future_times],
            dtype=float,
        )
        haz = hazard(future_times, beta, eta)

    rel_x_pad, rel_y_pad = plot_padding(adjustment_target, "Conditional Reliability", x_pad, y_pad)
    fail_x_pad, fail_y_pad = plot_padding(adjustment_target, "Conditional Failure Probability", x_pad, y_pad)
    haz_x_pad, haz_y_pad = plot_padding(adjustment_target, "Hazard Trend", x_pad, y_pad)
    rel_axis_config = plot_axis_config(adjustment_target, "Conditional Reliability", axis_config)
    fail_axis_config = plot_axis_config(adjustment_target, "Conditional Failure Probability", axis_config)
    haz_axis_config = plot_axis_config(adjustment_target, "Hazard Trend", axis_config)

    rel_x_low, rel_x_high = axis_limits(float(future_times.min()), float(future_times.max()), rel_x_pad)
    fail_x_low, fail_x_high = axis_limits(float(future_times.min()), float(future_times.max()), fail_x_pad)
    haz_x_low, haz_x_high = axis_limits(float(future_times.min()), float(future_times.max()), haz_x_pad)
    haz_y_low, haz_y_high = axis_limits_with_margin(float(haz.min()), float(haz.max()), haz_y_pad, margin=0.04)

    col_rel, col_fail, col_haz = st.columns(3)

    fig_rel, ax_rel = plt.subplots(figsize=(6, 4))
    ax_rel.plot(future_times, conditional_rel, label="Conditional reliability")
    ax_rel.axvline(current_age, color="black", linestyle="--", alpha=0.65, label="Current time")
    ax_rel.axvline(optimal_replacement, color="crimson", linestyle=":", alpha=0.75, label="Optimal replacement")
    ax_rel.axvline(eta, color="purple", linestyle="-.", alpha=0.7, label="η characteristic life")
    ax_rel.set_title("Conditional Reliability")
    ax_rel.set_xlabel("Time")
    ax_rel.set_ylabel("Reliability")
    rel_y_margin = 0.03 + rel_y_pad
    apply_axis_bounds_and_units(ax_rel, rel_axis_config, (rel_x_low, rel_x_high), (-rel_y_margin, 1.0 + rel_y_margin))
    ax_rel.grid(True, linestyle="--", alpha=0.45)
    col_rel.pyplot(fig_rel, clear_figure=True)

    fig_fail, ax_fail = plt.subplots(figsize=(6, 4))
    ax_fail.plot(future_times, conditional_fail, label="P(fail in mission)")
    ax_fail.axvline(current_age, color="black", linestyle="--", alpha=0.65, label="Current time")
    ax_fail.axvline(optimal_replacement, color="crimson", linestyle=":", alpha=0.75, label="Optimal replacement")
    ax_fail.axvline(eta, color="purple", linestyle="-.", alpha=0.7, label="η characteristic life")
    ax_fail.set_title("Conditional Failure Probability")
    ax_fail.set_xlabel("Time")
    ax_fail.set_ylabel("Probability")
    fail_y_margin = 0.03 + fail_y_pad
    apply_axis_bounds_and_units(ax_fail, fail_axis_config, (fail_x_low, fail_x_high), (-fail_y_margin, 1.0 + fail_y_margin))
    ax_fail.grid(True, linestyle="--", alpha=0.45)
    col_fail.pyplot(fig_fail, clear_figure=True)

    fig_haz, ax_haz = plt.subplots(figsize=(6, 4))
    ax_haz.plot(future_times, haz)
    ax_haz.axvline(current_age, color="black", linestyle="--", alpha=0.65)
    ax_haz.set_title("Hazard Trend")
    ax_haz.set_xlabel("Time")
    ax_haz.set_ylabel("Failure rate (per hour), not probability")
    apply_axis_bounds_and_units(ax_haz, haz_axis_config, (haz_x_low, haz_x_high), (haz_y_low, haz_y_high))
    ax_haz.grid(True, linestyle="--", alpha=0.45)
    ax_haz.legend(fontsize=8)
    ax_haz.tick_params(axis="y", labelleft=True, pad=4)
    col_haz.pyplot(fig_haz, clear_figure=True)

    horizon_failure = float(conditional_fail[-1])
    if horizon_failure >= 0.70:
        st.error(f"Very high conditional risk within the next {requested_horizon:,.0f} cycles: {fmt_pct(horizon_failure)}")
    elif horizon_failure >= 0.40:
        st.warning(f"Moderate conditional risk within the next {requested_horizon:,.0f} cycles: {fmt_pct(horizon_failure)}")
    else:
        st.success(f"Low conditional risk within the next {requested_horizon:,.0f} cycles: {fmt_pct(horizon_failure)}")


def plot_cost_curve(
    result: WeibullResult,
    preventive_cost: float,
    failure_cost: float,
    adjustment_target: str,
    x_pad: float,
    y_pad: float,
    axis_config: dict[str, float | bool],
) -> None:
    t_range, cost_rate = cost_based_optimal_replacement(
        result.data,
        result.beta,
        result.eta,
        preventive_cost,
        failure_cost,
    )[2:]

    cost_x_pad, cost_y_pad = plot_padding(adjustment_target, "Cost Rate", x_pad, y_pad)
    cost_axis_config = plot_axis_config(adjustment_target, "Cost Rate", axis_config)
    x_low, x_high = axis_limits(float(t_range.min()), float(t_range.max()), cost_x_pad)
    y_low, y_high = axis_limits_with_margin(float(cost_rate.min()), float(cost_rate.max()), cost_y_pad, margin=0.04)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t_range, cost_rate)
    ax.axvline(result.optimal_replacement, color="crimson", linestyle="--", label="Optimal replacement")
    ax.set_title("Cost Rate by Replacement Time")
    ax.set_xlabel("Replacement time")
    ax.set_ylabel("Expected cost rate ($/time)")
    apply_axis_bounds_and_units(ax, cost_axis_config, (x_low, x_high), (y_low, y_high))
    ax.grid(True, linestyle="--", alpha=0.45)
    ax.legend()
    st.pyplot(fig, clear_figure=True)


def build_distribution_fit_figure(
    result: WeibullResult,
    current_age: float,
    adjustment_target: str,
    x_pad: float,
    y_pad: float,
    axis_config: dict[str, float | bool],
    for_pdf: bool = False,
) -> plt.Figure:
    data = result.data
    beta = result.beta
    eta = result.eta

    n = len(data)
    ranks = np.arange(1, n + 1)
    median_ranks = (ranks - 0.3) / (n + 0.4)

    distribution_start, distribution_end = distribution_time_limits(data)
    distribution_range = np.linspace(distribution_start, distribution_end, 1200)

    hazard_start = max(EPS, float(np.min(data)) * 0.5)
    hazard_end = max(float(np.max(data)) * 1.75, eta * 1.5, hazard_start + 1.0)
    hazard_range = np.linspace(hazard_start, hazard_end, 1000)

    pdf_curve = weibull_pdf(distribution_range, beta, eta)
    cdf_curve = weibull_cdf(distribution_range, beta, eta)
    haz_curve = hazard(hazard_range, beta, eta)
    pdf_points = weibull_pdf(data, beta, eta)

    pdf_x_pad, pdf_y_pad = plot_padding(adjustment_target, "PDF", x_pad, y_pad)
    cdf_x_pad, cdf_y_pad = plot_padding(adjustment_target, "CDF", x_pad, y_pad)
    haz_x_pad, haz_y_pad = plot_padding(adjustment_target, "Hazard Function", x_pad, y_pad)
    pdf_axis_config = plot_axis_config(adjustment_target, "PDF", axis_config)
    cdf_axis_config = plot_axis_config(adjustment_target, "CDF", axis_config)
    haz_axis_config = plot_axis_config(adjustment_target, "Hazard Function", axis_config)

    pdf_x_low, pdf_x_high = axis_limits(distribution_start, distribution_end, pdf_x_pad)
    cdf_x_low, cdf_x_high = axis_limits(distribution_start, distribution_end, cdf_x_pad)
    haz_x_low, haz_x_high = axis_limits(hazard_start, hazard_end, haz_x_pad)
    pdf_y_low, pdf_y_high = axis_limits_with_margin(
        0.0,
        float(max(np.max(pdf_curve), np.max(pdf_points))),
        pdf_y_pad,
        margin=0.04,
    )
    haz_y_low, haz_y_high = axis_limits_with_margin(
        float(np.min(haz_curve)),
        float(np.max(haz_curve)),
        haz_y_pad,
        margin=0.04,
    )

    fig = plt.figure(figsize=(11.7, 8.3 if for_pdf else 7.5))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.85], hspace=0.42, wspace=0.25)
    ax_pdf = fig.add_subplot(grid[0, 0])
    ax_cdf = fig.add_subplot(grid[0, 1])
    ax_haz = fig.add_subplot(grid[1, :])

    ax_pdf.plot(distribution_range, pdf_curve, label="Weibull fit")
    ax_pdf.scatter(data, pdf_points, s=22, label="Observations", clip_on=False)
    ax_pdf.axvline(eta, color="crimson", linestyle="--", alpha=0.75, label="η characteristic life")
    ax_pdf.axvline(current_age, color="black", linestyle=":", alpha=0.7, label="Current time")
    ax_pdf.set_title("PDF")
    ax_pdf.set_xlabel("Time")
    ax_pdf.set_ylabel("Density")
    apply_axis_bounds_and_units(ax_pdf, pdf_axis_config, (pdf_x_low, pdf_x_high), (pdf_y_low, pdf_y_high))
    ax_pdf.grid(True, linestyle="--", alpha=0.45)
    ax_pdf.legend(fontsize=8)
    ax_pdf.tick_params(axis="y", labelleft=True, pad=4)

    ax_cdf.plot(distribution_range, cdf_curve, label="Weibull fit")
    ax_cdf.scatter(data, median_ranks, s=22, label="Median ranks", clip_on=False)
    ax_cdf.axvline(eta, color="crimson", linestyle="--", alpha=0.75, label="η characteristic life")
    ax_cdf.axvline(current_age, color="black", linestyle=":", alpha=0.7, label="Current time")
    ax_cdf.set_title("CDF / Unreliability")
    ax_cdf.set_xlabel("Time")
    ax_cdf.set_ylabel("Probability")
    cdf_y_margin = 0.03 + cdf_y_pad
    apply_axis_bounds_and_units(ax_cdf, cdf_axis_config, (cdf_x_low, cdf_x_high), (-cdf_y_margin, 1.0 + cdf_y_margin))
    ax_cdf.grid(True, linestyle="--", alpha=0.45)
    ax_cdf.legend(fontsize=8)
    ax_cdf.tick_params(axis="y", labelleft=True, pad=4)

    ax_haz.plot(hazard_range, haz_curve, label="Hazard rate")
    ax_haz.set_title("Hazard Function (Failure Rate)")
    ax_haz.set_xlabel("Time")
    ax_haz.set_ylabel("Failure rate (per hour), not probability")
    apply_axis_bounds_and_units(ax_haz, haz_axis_config, (haz_x_low, haz_x_high), (haz_y_low, haz_y_high))
    ax_haz.grid(True, linestyle="--", alpha=0.45)
    ax_haz.legend(fontsize=8)

    fig.suptitle(f"Distribution Fit - {result.component}", fontsize=14, fontweight="bold")
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
    beta = result.beta
    eta = result.eta
    optimal_replacement = result.optimal_replacement
    requested_horizon = max(float(horizon), 0.0)
    plot_horizon = max(requested_horizon, 1.0)
    future_times = np.linspace(max(EPS, current_age), current_age + plot_horizon, 250)

    current_rel = float(reliability(current_age, beta, eta))
    if requested_horizon <= 0:
        conditional_rel = np.ones_like(future_times)
        conditional_fail = np.zeros_like(future_times)
        haz = np.full_like(future_times, float(hazard(current_age, beta, eta)))
    else:
        conditional_rel = np.array(
            [conditional_reliability_between(current_age, t, beta, eta) for t in future_times],
            dtype=float,
        )
        conditional_fail = np.array(
            [conditional_failure_probability_between(current_age, t, beta, eta) for t in future_times],
            dtype=float,
        )
        haz = hazard(future_times, beta, eta)

    rel_x_pad, rel_y_pad = plot_padding(adjustment_target, "Conditional Reliability", x_pad, y_pad)
    fail_x_pad, fail_y_pad = plot_padding(adjustment_target, "Conditional Failure Probability", x_pad, y_pad)
    haz_x_pad, haz_y_pad = plot_padding(adjustment_target, "Hazard Trend", x_pad, y_pad)
    rel_axis_config = plot_axis_config(adjustment_target, "Conditional Reliability", axis_config)
    fail_axis_config = plot_axis_config(adjustment_target, "Conditional Failure Probability", axis_config)
    haz_axis_config = plot_axis_config(adjustment_target, "Hazard Trend", axis_config)

    rel_x_low, rel_x_high = axis_limits(float(future_times.min()), float(future_times.max()), rel_x_pad)
    fail_x_low, fail_x_high = axis_limits(float(future_times.min()), float(future_times.max()), fail_x_pad)
    haz_x_low, haz_x_high = axis_limits(float(future_times.min()), float(future_times.max()), haz_x_pad)
    haz_y_low, haz_y_high = axis_limits_with_margin(float(haz.min()), float(haz.max()), haz_y_pad, margin=0.04)

    fig, axes = plt.subplots(1, 3, figsize=(11.7, 4.2))
    rel_y_margin = 0.03 + rel_y_pad
    fail_y_margin = 0.03 + fail_y_pad

    axes[0].plot(future_times, conditional_rel)
    axes[0].axvline(current_age, color="black", linestyle="--", alpha=0.65)
    axes[0].set_title("Conditional Reliability")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Reliability")
    apply_axis_bounds_and_units(axes[0], rel_axis_config, (rel_x_low, rel_x_high), (-rel_y_margin, 1.0 + rel_y_margin))
    axes[0].grid(True, linestyle="--", alpha=0.45)

    axes[1].plot(future_times, conditional_fail)
    axes[1].axvline(current_age, color="black", linestyle="--", alpha=0.65)
    axes[1].set_title("Conditional Failure Probability")
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Probability")
    apply_axis_bounds_and_units(axes[1], fail_axis_config, (fail_x_low, fail_x_high), (-fail_y_margin, 1.0 + fail_y_margin))
    axes[1].grid(True, linestyle="--", alpha=0.45)

    axes[2].plot(future_times, haz)
    axes[2].axvline(current_age, color="black", linestyle="--", alpha=0.65)
    axes[2].set_title("Hazard Trend")
    axes[2].set_xlabel("Time")
    axes[2].set_ylabel("Failure rate")
    apply_axis_bounds_and_units(axes[2], haz_axis_config, (haz_x_low, haz_x_high), (haz_y_low, haz_y_high))
    axes[2].grid(True, linestyle="--", alpha=0.45)

    fig.suptitle(f"Forward Risk Trend - {result.component}", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.subplots_adjust(left=0.10)
    return fig


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
        result.beta,
        result.eta,
        preventive_cost,
        failure_cost,
    )[2:]

    cost_x_pad, cost_y_pad = plot_padding(adjustment_target, "Cost Rate", x_pad, y_pad)
    cost_axis_config = plot_axis_config(adjustment_target, "Cost Rate", axis_config)
    x_low, x_high = axis_limits(float(t_range.min()), float(t_range.max()), cost_x_pad)
    y_low, y_high = axis_limits_with_margin(float(cost_rate.min()), float(cost_rate.max()), cost_y_pad, margin=0.04)

    fig, ax = plt.subplots(figsize=(11.7, 4.2))
    ax.plot(t_range, cost_rate, label="Cost rate")
    ax.axvline(result.optimal_replacement, color="crimson", linestyle="--", label="Optimal replacement")
    ax.axvline(result.eta, color="purple", linestyle=":", label="η characteristic life")
    ax.axvline(result.rul, color="black", linestyle="-.", label="RUL")
    ax.set_title(f"Replacement Economics - {result.component}")
    ax.set_xlabel("Replacement time")
    ax.set_ylabel("Expected cost rate ($/time)")
    apply_axis_bounds_and_units(ax, cost_axis_config, (x_low, x_high), (y_low, y_high))
    ax.grid(True, linestyle="--", alpha=0.45)
    ax.legend()
    ax.tick_params(axis="y", labelleft=True, pad=4)
    fig.tight_layout()
    fig.subplots_adjust(left=0.10)
    return fig


# ============================================================
# UI
# ============================================================
st.title("🔧 Reliability & Weibull Predictive Dashboard Pro")
st.caption("Upload an Excel workbook where each column is one component and each value is a positive failure/runtime observation.")

with st.sidebar:
    st.header("Inputs")
    mttr = st.number_input("MTTR", min_value=0.0, value=0.0, step=1.0)
    current_age = st.number_input("Current in-service time", min_value=0.0, value=0.0, step=100.0)
    mission_time = st.number_input("Future mission/runtime", min_value=0.0, value=0.0, step=100.0)
    preventive_cost = st.number_input("Preventive cost (Cp) $", min_value=0.0, value=0.0, step=100.0)
    failure_cost = st.number_input("Failure cost (Cf) $", min_value=0.0, value=0.0, step=500.0)
    prediction_horizon = st.slider("Prediction horizon", 0, 10000, 0, 100)

    st.divider()
    st.subheader("Plot padding")
    st.session_state.setdefault("axis_adjustment_target", "All plots")
    st.session_state.setdefault("axis_x_pad", default_plot_padding(st.session_state["axis_adjustment_target"]))
    st.session_state.setdefault("axis_y_pad", default_plot_padding(st.session_state["axis_adjustment_target"]))
    adjustment_target = st.selectbox(
        "Apply axis adjustment to",
        PLOT_ADJUSTMENT_TARGETS,
        key="axis_adjustment_target",
        on_change=reset_active_axis_adjustment,
    )
    x_pad = adjustment_control("Horizontal adjustment", "axis_x_pad")
    y_pad = adjustment_control("Vertical adjustment", "axis_y_pad")
    axis_config = axis_bounds_and_units_control()
    if st.button("Reset axis bounds and units"):
        reset_active_axis_adjustment()
        st.rerun()

    st.divider()
    st.markdown("**Beta interpretation**")
    st.markdown("- Beta < 0.95: infant mortality")
    st.markdown("- 0.95 to 1.05: random failure")
    st.markdown("- Beta > 1.05: wear-out")
    st.markdown("- High beta: sudden failure tendency")
    st.markdown("**Hazard interpretation:** failure rate (per hour), not probability")

template_col, upload_col = st.columns([1, 2])
with template_col:
    st.download_button(
        "Download Excel Template",
        data=build_template_workbook(),
        file_name="weibull_input_template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
with upload_col:
    uploaded_file = st.file_uploader(
        "Upload Excel file",
        type=["xlsx", "xls"],
        help="Each column should contain failure times or life observations for one component.",
    )

if uploaded_file is None:
    st.info("Upload an Excel file to begin the analysis.")
    st.stop()

try:
    datasets = parse_excel(uploaded_file)
except Exception as exc:
    st.error(f"Could not read the workbook: {exc}")
    st.stop()

if not datasets:
    st.error("No valid component columns were found. Each usable column needs at least two positive numeric values.")
    st.stop()

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
        )
    except Exception as exc:
        skipped.append(f"{component}: {exc}")

if skipped:
    with st.expander("Skipped components"):
        for item in skipped:
            st.warning(item)

if not analysis:
    st.error("No component could be analyzed after parameter estimation.")
    st.stop()

df_results = results_to_frame(analysis)
top_row = df_results.iloc[0]

metric_1, metric_2, metric_3, metric_4, metric_5 = st.columns(5)
metric_1.metric("Components", f"{len(df_results)}")
metric_2.metric("Highest Failure Probability", fmt_pct(top_row["Conditional Probability of Failure"]))
metric_3.metric("Highest Risk Component", highest_risk_component_text(df_results))
metric_4.metric("Average Beta", f"{df_results['Beta'].mean():.2f}")
metric_5.metric("Average MTTF", f"{df_results['MTTF'].mean():,.2f}")

summary_tab, detail_tab, export_tab = st.tabs(["Fleet Summary", "Component Deep Dive", "Export"])

with summary_tab:
    st.subheader("Full Reliability Results")
    display_results_table(df_results)

    high_risk = df_results[df_results["Risk"] == "HIGH"]
    if not high_risk.empty:
        st.error(f"{len(high_risk)} component(s) are currently classified as HIGH risk.")
    else:
        st.success("No component is currently classified as HIGH risk.")

with detail_tab:
    selected_component = st.selectbox("Select component", list(analysis.keys()))
    result = analysis[selected_component]

    st.markdown(f"### {selected_component}")
    left, middle, right = st.columns(3)
    left.metric("Beta", f"{result.beta:.3f}", result.failure_mode)
    middle.metric("Eta", f"{result.eta:,.2f}")
    right.metric("Risk", result.risk, fmt_pct(result.conditional_failure_probability))

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("MTTF", f"{result.mttf:,.2f}")
    m2.metric("MTBF", f"{result.mtbf:,.2f}")
    m3.metric("RUL", f"{result.rul:,.2f}")
    m4.metric("Optimal Replacement", f"{result.optimal_replacement:,.2f}")
    m5.metric("Min Cost Rate", fmt_money(result.min_cost_rate))

    st.metric("Conditional Reliability", fmt_pct(result.conditional_reliability))
    st.metric("Decision", result.decision)

    if result.decision == "OVERDUE":
        st.error(f"OVERDUE: replacement should happen now or at the next available maintenance stop.")
    elif result.decision == "PLAN REPLACEMENT":
        st.warning(f"PLAN REPLACEMENT: recommended around {result.optimal_replacement:,.2f}.")
    else:
        st.success(f"SAFE: continue monitoring. Recommended replacement around {result.optimal_replacement:,.2f}.")

    q10, q50, q90 = (
        weibull_quantile(0.10, result.beta, result.eta),
        weibull_quantile(0.50, result.beta, result.eta),
        weibull_quantile(0.90, result.beta, result.eta),
    )
    st.info(f"B10 life: {q10:,.2f} | Median life: {q50:,.2f} | B90 life: {q90:,.2f}")

    st.subheader("Distribution Fit")
    plot_distribution_fit(result, current_age, adjustment_target, x_pad, y_pad, axis_config)

    st.subheader("Forward Risk Trend")
    plot_forward_risk(result, current_age, prediction_horizon, adjustment_target, x_pad, y_pad, axis_config)

    st.subheader("Replacement Economics")
    plot_cost_curve(result, preventive_cost, failure_cost, adjustment_target, x_pad, y_pad, axis_config)

with export_tab:
    st.subheader("Download Outputs")
    raw_csv = df_results.to_csv(index=False).encode("utf-8")
    st.download_button("Download Results CSV", raw_csv, "weibull_results.csv", "text/csv")

    report_pdf_key = f"reliability_report_pdf_{safe_filename(result.component)}"
    if st.button("Prepare Reliability Report PDF"):
        st.session_state[report_pdf_key] = build_reliability_report_pdf(
            df_results,
            result,
            current_age,
            prediction_horizon,
            preventive_cost,
            failure_cost,
            adjustment_target,
            x_pad,
            y_pad,
            axis_config,
        )
    if report_pdf_key in st.session_state:
        st.download_button(
            "Download Reliability Report PDF",
            st.session_state[report_pdf_key],
            f"reliability_report_{safe_filename(result.component)}.pdf",
            "application/pdf",
        )

    cleaned_input = pd.DataFrame({name: pd.Series(data) for name, data in datasets.items()})
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df_results.to_excel(writer, index=False, sheet_name="results")
        cleaned_input.to_excel(writer, index=False, sheet_name="cleaned_input")

    st.download_button(
        "Download Analysis Workbook",
        buffer.getvalue(),
        "weibull_analysis.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

st.caption("All future risk metrics use conditional reliability: P(fail in mission)=1−R(t+Δt)/R(t).")
