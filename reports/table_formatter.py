from __future__ import annotations

import numpy as np
import pandas as pd

from core.optimization import DistributionFit
from core.reliability import ConfidenceInterval, WeibullResult, confidence_interval


def fmt_num(value: float, decimals: int = 2) -> str:
    if not np.isfinite(float(value)):
        return "N/A"
    return f"{float(value):,.{decimals}f}"


def fmt_pct(value: float) -> str:
    if not np.isfinite(float(value)):
        return "N/A"
    return f"{float(value):.2%}"


def fmt_money(value: float) -> str:
    if not np.isfinite(float(value)):
        return "N/A"
    return f"${float(value):,.2f}"


def fmt_interval(interval: ConfidenceInterval, formatter) -> str:
    return f"{formatter(interval.lower)} to {formatter(interval.upper)}"


def fmt_bounds(lower: float, upper: float, formatter) -> str:
    return f"{formatter(lower)} to {formatter(upper)}"


def distribution_parameter_text(fit: DistributionFit) -> str:
    return ", ".join(
        f"{label}={fmt_num(value, 3)}"
        for label, value in zip(fit.param_labels, fit.params, strict=False)
    )


def formatted_results_frame(df: pd.DataFrame) -> pd.DataFrame:
    display_df = df.copy()

    interval_specs = [
        ("Parameter 1 95% CI", "Parameter 1 CI Lower", "Parameter 1 CI Upper", fmt_num),
        ("Parameter 2 95% CI", "Parameter 2 CI Lower", "Parameter 2 CI Upper", fmt_num),
        (
            "Conditional Reliability 95% CI",
            "Conditional Reliability CI Lower",
            "Conditional Reliability CI Upper",
            fmt_pct,
        ),
        (
            "Failure Probability 95% CI",
            "Failure Probability CI Lower",
            "Failure Probability CI Upper",
            fmt_pct,
        ),
        ("RUL 95% CI", "RUL CI Lower", "RUL CI Upper", fmt_num),
    ]
    for label, lower_col, upper_col, formatter in interval_specs:
        if lower_col in display_df.columns and upper_col in display_df.columns:
            display_df[label] = [
                fmt_bounds(lower, upper, formatter)
                for lower, upper in zip(display_df[lower_col], display_df[upper_col], strict=False)
            ]
            display_df = display_df.drop(columns=[lower_col, upper_col])

    pct_cols = [
        "Conditional Reliability",
        "Conditional Probability of Failure",
    ]
    numeric_cols = [
        "Parameter 1",
        "Parameter 2",
        "Characteristic Value",
        "MTTF",
        "MTBF",
        "RUL",
        "Optimal Replacement",
        "Best Fit AIC",
        "Best Fit BIC",
        "Best Fit RMSE",
    ]
    precise_numeric_cols: list[str] = []
    money_cols = ["Min Cost Rate"]
    int_cols = ["Severity", "Occurrence", "Detectability", "RPN"]

    for col in pct_cols:
        if col in display_df.columns:
            display_df[col] = display_df[col].map(fmt_pct)
    for col in numeric_cols:
        if col in display_df.columns:
            display_df[col] = display_df[col].map(fmt_num)
    for col in precise_numeric_cols:
        if col in display_df.columns:
            display_df[col] = display_df[col].map(lambda value: fmt_num(value, 3))
    for col in money_cols:
        if col in display_df.columns:
            display_df[col] = display_df[col].map(fmt_money)
    for col in int_cols:
        if col in display_df.columns:
            display_df[col] = display_df[col].map(lambda value: f"{int(round(float(value)))}")

    return display_df


def distribution_comparison_frame(distribution_fits: tuple[DistributionFit, ...]) -> pd.DataFrame:
    rows = []
    for rank, fit in enumerate(distribution_fits, start=1):
        rows.append(
            {
                "Rank": rank,
                "Distribution": fit.name,
                "Parameters": distribution_parameter_text(fit),
                "Best Fit": "YES" if rank == 1 else "NO",
                "Fit % (R^2-like)": fmt_pct(fit.fit_score),
                "AIC": fmt_num(fit.aic),
                "BIC": fmt_num(fit.bic),
                "RMSE": fmt_num(fit.rmse, 4),
                "KS Statistic": fmt_num(fit.ks_statistic, 4),
                "KS p-value": fmt_num(fit.ks_pvalue, 4),
                "AD Statistic": fmt_num(fit.ad_statistic, 4),
                "AD p-value": fmt_num(fit.ad_pvalue, 4),
            }
        )
    return pd.DataFrame(rows)


def metric_descriptions_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Term": "RUL",
                "Meaning": "Remaining useful life estimated from the selected life model at the current in-service time.",
                "Why it matters": "Helps you judge how much usable life is left before maintenance should be planned.",
            },
            {
                "Term": "MTTF",
                "Meaning": "Mean time to failure for the selected fitted distribution.",
                "Why it matters": "Gives an average expected life for the component population.",
            },
            {
                "Term": "MTBF",
                "Meaning": "Mean time between failures, calculated here as MTTF plus MTTR.",
                "Why it matters": "Useful for maintenance planning and high-level availability discussions.",
            },
            {
                "Term": "Conditional Reliability",
                "Meaning": "Probability of surviving the future mission given the component has already survived to the current age.",
                "Why it matters": "This is the most operationally relevant survival metric for in-service assets.",
            },
            {
                "Term": "Failure Probability",
                "Meaning": "Probability of failing during the future mission window, conditional on surviving to now.",
                "Why it matters": "Directly supports maintenance urgency and risk communication.",
            },
            {
                "Term": "B10 / B50 / B90",
                "Meaning": "Life values at which 10%, 50%, and 90% of the population are expected to have failed.",
                "Why it matters": "Useful for maintenance policy, warranty thinking, and conservative replacement decisions.",
            },
            {
                "Term": "Characteristic value",
                "Meaning": "The 63.2% life point from the selected distribution.",
                "Why it matters": "For Weibull this equals eta, and for the other models it gives a comparable reference life marker.",
            },
            {
                "Term": "RPN",
                "Meaning": "Risk Priority Number = Severity x Occurrence x Detectability.",
                "Why it matters": "Gives a simple FMEA-style priority score for attention and action.",
            },
        ]
    )


def distribution_descriptions_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Distribution": "Weibull",
                "Shape / Behavior": "Flexible wear-out model; can show early-life, random, or wear-out behavior through beta.",
                "Main Use": "Fatigue failures, brake life, and mechanical degradation.",
                "Failure Rate Behavior": "Decreasing, constant, or increasing depending on beta.",
                "Data Requirement": "Positive continuous life data with enough spread to estimate shape reliably.",
            },
            {
                "Distribution": "Lognormal",
                "Shape / Behavior": "Right-skewed; suitable when degradation variability expands over time.",
                "Main Use": "Skewed wear and degradation processes.",
                "Failure Rate Behavior": "Usually rises after an early low-risk region.",
                "Data Requirement": "Positive continuous data with visible right skew.",
            },
            {
                "Distribution": "Gamma",
                "Shape / Behavior": "Positive and flexible for accumulated damage patterns.",
                "Main Use": "Cumulative stress, thermal damage, and cycle-based wear.",
                "Failure Rate Behavior": "Often increasing as damage accumulates.",
                "Data Requirement": "Positive continuous data tied to cumulative degradation.",
            },
            {
                "Distribution": "Gumbel",
                "Shape / Behavior": "Extreme-value model that emphasizes tail events.",
                "Main Use": "Hard landings, peak brake temperatures, and stress spikes.",
                "Failure Rate Behavior": "Driven by extreme-event tail shape rather than classic wear-out.",
                "Data Requirement": "Continuous data where extremes matter more than average behavior.",
            },
            {
                "Distribution": "Exponential",
                "Shape / Behavior": "Single-parameter random-failure model.",
                "Main Use": "Baseline benchmark for constant-rate failures.",
                "Failure Rate Behavior": "Constant over time.",
                "Data Requirement": "Positive continuous life data with no clear aging effect.",
            },
            {
                "Distribution": "Normal",
                "Shape / Behavior": "Symmetric around the mean.",
                "Main Use": "Stable operational variation and non-life process measures.",
                "Failure Rate Behavior": "Not ideal for classic wear-out hazard interpretation.",
                "Data Requirement": "Approximately symmetric continuous data.",
            },
            {
                "Distribution": "Rayleigh",
                "Shape / Behavior": "Special-case positive distribution with a single scale parameter.",
                "Main Use": "Vibration-like and rotating-system style patterns.",
                "Failure Rate Behavior": "Rising with time.",
                "Data Requirement": "Positive continuous data matching a narrow special-case shape.",
            },
        ]
    )


def fit_stat_descriptions_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Statistic": "AIC",
                "Meaning": "Akaike Information Criterion. Lower is better.",
                "How to read it": "Rewards fit quality but penalizes unnecessary complexity.",
            },
            {
                "Statistic": "BIC",
                "Meaning": "Bayesian Information Criterion. Lower is better.",
                "How to read it": "Penalizes extra complexity more strongly than AIC.",
            },
            {
                "Statistic": "RMSE",
                "Meaning": "Root mean squared error between fitted CDF values and empirical median ranks.",
                "How to read it": "Lower means the fitted distribution sits closer to the observed probability pattern.",
            },
            {
                "Statistic": "Fit % (R^2-like)",
                "Meaning": "A bounded fit score derived from the same empirical-vs-fitted CDF comparison.",
                "How to read it": "Higher is better; it behaves like an easy-to-read fit percentage rather than a literal regression R^2.",
            },
            {
                "Statistic": "KS test",
                "Meaning": "Kolmogorov-Smirnov goodness-of-fit test.",
                "How to read it": "Smaller statistic and larger p-value usually indicate closer agreement.",
            },
            {
                "Statistic": "AD test",
                "Meaning": "Anderson-Darling goodness-of-fit test.",
                "How to read it": "Puts more weight on the tails, which is useful when rare high-stress events matter.",
            },
        ]
    )


def plot_descriptions_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Plot": "Distribution Fit",
                "What it shows": "PDF and CDF for the user-selected distribution, plus the characteristic value marker.",
                "How to use it": "Check whether the chosen model follows the observed life pattern cleanly and where the 63.2% life point sits.",
            },
            {
                "Plot": "PDF / CDF Comparison",
                "What it shows": "How all candidate distributions compare against the same data.",
                "How to use it": "Use this when you want to challenge the best-fit recommendation and inspect alternatives.",
            },
            {
                "Plot": "Hazard Function",
                "What it shows": "Failure rate over time for the selected distribution.",
                "How to use it": "Use it to see whether risk is increasing, decreasing, or roughly steady with age.",
            },
            {
                "Plot": "Forward Risk Trend",
                "What it shows": "Conditional reliability, failure probability, and hazard trend into the future mission window.",
                "How to use it": "Use it to see how risk evolves from the current age, not from brand-new condition.",
            },
            {
                "Plot": "Replacement Economics",
                "What it shows": "Expected cost rate over replacement time.",
                "How to use it": "Look for the minimum point to balance preventive cost against failure cost.",
            },
        ]
    )


def confidence_summary_frame(result: WeibullResult) -> pd.DataFrame:
    rows = []
    for label, value, interval in zip(result.selected_param_labels, result.selected_param_values, result.selected_param_cis, strict=False):
        rows.append(
            {
                "Metric": label,
                "Estimate": fmt_num(value, 3),
                "95% CI": fmt_interval(interval, lambda entry: fmt_num(entry, 3)),
            }
        )

    rows.extend(
        [
            {
                "Metric": "Characteristic value",
                "Estimate": fmt_num(result.characteristic_value),
                "95% CI": fmt_interval(
                    confidence_interval(result.bootstrap_metric_samples["characteristic_value"]),
                    fmt_num,
                ),
            },
            {
                "Metric": "Conditional Reliability",
                "Estimate": fmt_pct(result.conditional_reliability),
                "95% CI": fmt_interval(result.conditional_reliability_ci, fmt_pct),
            },
            {
                "Metric": "Failure Probability",
                "Estimate": fmt_pct(result.conditional_failure_probability),
                "95% CI": fmt_interval(result.conditional_failure_probability_ci, fmt_pct),
            },
            {
                "Metric": "RUL",
                "Estimate": fmt_num(result.rul),
                "95% CI": fmt_interval(result.rul_ci, fmt_num),
            },
        ]
    )
    return pd.DataFrame(rows)


def highest_risk_component_text(df: pd.DataFrame) -> str:
    if df.empty:
        return "None"

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
