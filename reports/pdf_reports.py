from __future__ import annotations

from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from core.reliability import WeibullResult
from plotting.cost_plots import build_cost_curve_figure
from plotting.distribution_plots import (
    build_distribution_comparison_figure,
    build_distribution_fit_figure,
    build_forward_risk_figure,
    build_hazard_function_figure,
)
from reports.table_formatter import (
    confidence_summary_frame,
    distribution_comparison_frame,
    fmt_money,
    fmt_pct,
    formatted_results_frame,
    highest_risk_component_text,
)


def add_dataframe_page(pdf: PdfPages, title: str, df: pd.DataFrame, rows_per_page: int = 18) -> None:
    page_count = max(1, int(np.ceil(len(df) / rows_per_page)))

    for page_index in range(page_count):
        page_df = df.iloc[page_index * rows_per_page : (page_index + 1) * rows_per_page]
        fig, ax = plt.subplots(figsize=(11.7, 8.3))
        fig.subplots_adjust(left=0.03, right=0.97, top=0.92, bottom=0.04)
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
        table.set_fontsize(6.4)
        table.scale(0.95, 1.25)
        try:
            table.auto_set_column_width(col=list(range(len(page_df.columns))))
        except Exception:
            pass
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


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
    if df_results.empty:
        raise ValueError("No results are available to export.")

    buffer = BytesIO()
    with PdfPages(buffer) as pdf:
        fig, ax = plt.subplots(figsize=(11.7, 8.3))
        ax.axis("off")
        top_row = df_results.iloc[0]
        summary_lines = [
            "Reliability Report",
            "",
            "Fleet Summary",
            f"Components analyzed: {len(df_results)}",
            f"Highest failure probability: {fmt_pct(top_row['Conditional Probability of Failure'])}",
            f"Highest risk component(s): {highest_risk_component_text(df_results)}",
            f"Average MTTF: {df_results['MTTF'].mean():,.2f}",
            "",
            f"Selected component: {result.component}",
            f"Selected distribution for calculations: {result.selected_distribution}",
            f"Best distribution by AIC / BIC / RMSE: {result.best_distribution}",
            f"Preventive cost: {fmt_money(preventive_cost)}",
            f"Failure cost: {fmt_money(failure_cost)}",
            f"Minimum cost rate: {fmt_money(result.min_cost_rate)}",
            f"FMEA severity / occurrence / detectability: {result.severity} / {result.occurrence} / {result.detectability}",
            f"RPN: {result.rpn}",
        ]
        ax.text(0.05, 0.90, summary_lines[0], fontsize=20, fontweight="bold", transform=ax.transAxes)
        ax.text(0.05, 0.80, "\n".join(summary_lines[2:]), fontsize=13, va="top", transform=ax.transAxes)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        add_dataframe_page(pdf, "Full Reliability Results", formatted_results_frame(df_results), rows_per_page=16)
        add_dataframe_page(pdf, "Distribution Comparison", distribution_comparison_frame(result.distribution_fits), rows_per_page=18)
        add_dataframe_page(pdf, "Confidence Intervals", confidence_summary_frame(result), rows_per_page=12)

        fig, ax = plt.subplots(figsize=(11.7, 8.3))
        ax.axis("off")
        details = [
            f"Component: {result.component}",
            f"Selected distribution: {result.selected_distribution}",
            f"Risk: {result.risk}",
            f"Conditional failure probability: {fmt_pct(result.conditional_failure_probability)}",
            f"Conditional reliability: {fmt_pct(result.conditional_reliability)}",
            f"Characteristic value: {result.characteristic_value:,.2f}",
            f"RUL 95% CI: {result.rul_ci.lower:,.2f} to {result.rul_ci.upper:,.2f}",
            f"Failure probability 95% CI: {fmt_pct(result.conditional_failure_probability_ci.lower)} to {fmt_pct(result.conditional_failure_probability_ci.upper)}",
            f"Optimal replacement: {result.optimal_replacement:,.2f}",
            f"Preventive cost: {fmt_money(preventive_cost)}",
            f"Failure cost: {fmt_money(failure_cost)}",
            f"Minimum cost rate: {fmt_money(result.min_cost_rate)}",
            f"Failure mode: {result.failure_mode}",
            f"Best-fit distribution: {result.best_distribution}",
            f"Selected fit RMSE: {result.selected_fit.rmse:,.4f}",
            f"RPN: {result.rpn}",
        ]
        ax.text(0.05, 0.90, "Component Deep Dive", fontsize=20, fontweight="bold", transform=ax.transAxes)
        ax.text(0.05, 0.80, "\n".join(details), fontsize=13, va="top", transform=ax.transAxes)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        distribution_fig = build_distribution_fit_figure(result, adjustment_target, x_pad, y_pad, axis_config, for_pdf=True)
        pdf.savefig(distribution_fig, bbox_inches="tight")
        plt.close(distribution_fig)

        comparison_fig = build_distribution_comparison_figure(result, adjustment_target, x_pad, y_pad, axis_config, for_pdf=True)
        pdf.savefig(comparison_fig, bbox_inches="tight")
        plt.close(comparison_fig)

        hazard_fig = build_hazard_function_figure(result, adjustment_target, x_pad, y_pad, axis_config)
        pdf.savefig(hazard_fig, bbox_inches="tight")
        plt.close(hazard_fig)

        forward_fig = build_forward_risk_figure(result, current_age, horizon, adjustment_target, x_pad, y_pad, axis_config)
        pdf.savefig(forward_fig, bbox_inches="tight")
        plt.close(forward_fig)

        cost_fig = build_cost_curve_figure(result, preventive_cost, failure_cost, adjustment_target, x_pad, y_pad, axis_config)
        pdf.savefig(cost_fig, bbox_inches="tight")
        plt.close(cost_fig)

    buffer.seek(0)
    return buffer.getvalue()
