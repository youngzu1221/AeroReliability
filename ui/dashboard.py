from __future__ import annotations

from io import BytesIO

import pandas as pd
import streamlit as st

from core.optimization import SUPPORTED_DISTRIBUTIONS, distribution_ppf
from core.reliability import analyze_datasets, decision_from_metrics, results_to_frame
from data.excel_parser import deserialize_datasets, parse_excel, serialize_datasets
from data.templates import build_template_workbook
from plotting.cost_plots import build_cost_curve_figure
from plotting.distribution_plots import (
    build_distribution_comparison_figure,
    build_distribution_fit_figure,
    build_forward_risk_figure,
    build_hazard_function_figure,
)
from reports.pdf_reports import build_reliability_report_pdf
from reports.table_formatter import (
    confidence_summary_frame,
    distribution_comparison_frame,
    distribution_descriptions_frame,
    fmt_money,
    fmt_pct,
    formatted_results_frame,
    highest_risk_component_text,
    metric_descriptions_frame,
    plot_descriptions_frame,
    safe_filename,
)
from ui.sidebar import render_sidebar

ANALYSIS_CACHE_VERSION = "2026-05-14-analysis-v2"
REPORT_CACHE_VERSION = "2026-05-14-report-v2"


@st.cache_data(show_spinner=False)
def parse_excel_cached(file_bytes: bytes):
    return parse_excel(file_bytes)


@st.cache_data(show_spinner=False)
def analyze_datasets_cached(
    serialized_datasets: tuple[tuple[str, tuple[float, ...]], ...],
    mttr: float,
    current_age: float,
    mission_time: float,
    preventive_cost: float,
    failure_cost: float,
    severity: int,
    detectability: int,
    selected_distribution: str,
    cache_version: str,
):
    datasets = deserialize_datasets(serialized_datasets)
    return analyze_datasets(
        datasets,
        mttr,
        current_age,
        mission_time,
        preventive_cost,
        failure_cost,
        severity,
        detectability,
        selected_distribution,
    )


@st.cache_data(show_spinner=False)
def build_report_cached(
    df_results: pd.DataFrame,
    result,
    current_age: float,
    prediction_horizon: float,
    preventive_cost: float,
    failure_cost: float,
    adjustment_target: str,
    x_pad: float,
    y_pad: float,
    axis_config: dict[str, float | bool],
    cache_version: str,
) -> bytes:
    return build_reliability_report_pdf(
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


def render_dashboard() -> None:
    st.set_page_config(page_title="Reliability & Weibull Pro Dashboard", page_icon="R", layout="wide")
    st.title("Reliability & Weibull Predictive Dashboard Pro")
    st.caption("Upload an Excel workbook where each column is one component and each value is a positive failure/runtime observation.")

    sidebar = render_sidebar()
    st.session_state.setdefault("selected_distribution_method", "Weibull")
    selected_distribution = str(st.session_state["selected_distribution_method"])

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
            help="Expected format: one component per column, with at least two positive numeric observations in each usable column.",
        )

    if uploaded_file is None:
        st.info("Upload an Excel file to begin the analysis.")
        return

    with st.spinner("Loading workbook and fitting distributions..."):
        parse_result = parse_excel_cached(uploaded_file.getvalue())

    if parse_result.error:
        st.error(parse_result.error)
        if parse_result.warnings:
            with st.expander("Validation details"):
                for warning in parse_result.warnings:
                    st.warning(warning)
        return

    if parse_result.warnings:
        with st.expander("Validation details"):
            for warning in parse_result.warnings:
                st.warning(warning)

    serialized_datasets = serialize_datasets(parse_result.datasets)
    with st.spinner(f"Loading analysis using the {selected_distribution} distribution..."):
        analysis, skipped = analyze_datasets_cached(
            serialized_datasets,
            sidebar.mttr,
            sidebar.current_age,
            sidebar.mission_time,
            sidebar.preventive_cost,
            sidebar.failure_cost,
            sidebar.severity,
            sidebar.detectability,
            selected_distribution,
            ANALYSIS_CACHE_VERSION,
        )

    if skipped:
        with st.expander("Skipped components"):
            for item in skipped:
                st.warning(item)

    if not analysis:
        st.error("No component could be analyzed after parameter estimation.")
        return

    df_results = results_to_frame(analysis)
    if df_results.empty:
        st.error("No reportable results were produced from the uploaded workbook.")
        return

    top_row = df_results.iloc[0]
    trend_horizon = max(float(sidebar.mission_time), float(sidebar.prediction_horizon))
    metric_1, metric_2, metric_3, metric_4, metric_5 = st.columns(5)
    metric_1.metric("Components", f"{len(df_results)}")
    metric_2.metric("Highest Failure Probability", fmt_pct(top_row["Conditional Probability of Failure"]))
    metric_3.metric("Highest Risk Component", highest_risk_component_text(df_results))
    metric_4.metric("Selected Distribution", selected_distribution)
    metric_5.metric("Average MTTF", f"{df_results['MTTF'].mean():,.2f}")

    distribution_tab, summary_tab, detail_tab, export_tab = st.tabs(
        ["Distribution Selection", "Fleet Summary", "Component Deep Dive", "Export"]
    )

    with distribution_tab:
        st.subheader("Choose Distribution Method")
        st.selectbox(
            "Distribution method for reliability calculations",
            list(SUPPORTED_DISTRIBUTIONS),
            key="selected_distribution_method",
            help="Changing this will rerun the dashboard using that distribution for reliability, RUL, hazard, and replacement economics calculations.",
        )

        comparison_component = st.selectbox(
            "Component for distribution comparison",
            list(analysis.keys()),
            index=0,
            key="comparison_component",
        )
        comparison_result = analysis[comparison_component]
        st.info(
            f"For {comparison_component}, the best-fit recommendation by AIC / BIC / RMSE is "
            f"{comparison_result.best_distribution}. The dashboard is currently using {selected_distribution} "
            "because that is the distribution you selected."
        )

        st.subheader("Distribution Comparison")
        st.dataframe(distribution_comparison_frame(comparison_result.distribution_fits), use_container_width=True, hide_index=True)

        st.subheader("Distribution Reference Guide")
        st.dataframe(distribution_descriptions_frame(), use_container_width=True, hide_index=True)

    with summary_tab:
        st.subheader("Full Reliability Results")
        st.dataframe(formatted_results_frame(df_results), use_container_width=True, hide_index=True)
        distribution_mix = (
            df_results["Best Fit Distribution"]
            .value_counts()
            .rename_axis("Distribution")
            .reset_index(name="Components")
        )
        left, right = st.columns([1.2, 1.0])
        with left:
            high_risk = df_results[df_results["Risk"] == "HIGH"]
            if not high_risk.empty:
                st.error(f"{len(high_risk)} component(s) are currently classified as HIGH risk.")
            else:
                st.success("No component is currently classified as HIGH risk.")
            st.caption("Fleet metrics on this tab are calculated using the distribution you selected on the first tab.")
        with right:
            st.subheader("Best-Fit Distribution Mix")
            st.dataframe(distribution_mix, use_container_width=True, hide_index=True)

    with detail_tab:
        selected_component = st.selectbox("Select component", list(analysis.keys()), key="selected_component")
        result = analysis[selected_component]
        decision = decision_from_metrics(result.conditional_failure_probability, result.optimal_replacement)

        st.markdown(f"### {selected_component}")
        head_1, head_2, head_3, head_4, head_5 = st.columns(5)
        head_1.metric("Distribution", result.selected_distribution)
        head_2.metric("Characteristic Value", f"{result.characteristic_value:,.2f}")
        head_3.metric("Conditional Reliability", fmt_pct(result.conditional_reliability))
        head_4.metric("Failure Probability", fmt_pct(result.conditional_failure_probability))
        head_5.metric("Risk", result.risk, fmt_pct(result.conditional_failure_probability))

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("MTTF", f"{result.mttf:,.2f}")
        m2.metric("MTBF", f"{result.mtbf:,.2f}")
        m3.metric("RUL", f"{result.rul:,.2f}")
        m4.metric("Optimal Replacement", f"{result.optimal_replacement:,.2f}")
        m5.metric("Min Cost Rate", fmt_money(result.min_cost_rate))

        if decision.level == "HIGH":
            st.error(decision.message)
        elif decision.level == "MEDIUM":
            st.warning(decision.message)
        else:
            st.success(decision.message)

        b10 = float(distribution_ppf(result.selected_distribution, result.selected_fit.params, 0.10))
        b50 = float(distribution_ppf(result.selected_distribution, result.selected_fit.params, 0.50))
        b90 = float(distribution_ppf(result.selected_distribution, result.selected_fit.params, 0.90))
        st.info(f"B10 life: {b10:,.2f} | Median life: {b50:,.2f} | B90 life: {b90:,.2f}")

        st.subheader("Confidence Intervals")
        st.dataframe(confidence_summary_frame(result), use_container_width=True, hide_index=True)

        st.subheader("FMEA / RPN")
        f1, f2, f3, f4 = st.columns(4)
        f1.metric("Severity", f"{result.severity}")
        f2.metric("Occurrence", f"{result.occurrence}")
        f3.metric("Detectability", f"{result.detectability}")
        f4.metric("RPN", f"{result.rpn}")
        st.caption("Occurrence is derived automatically from the current conditional failure probability on a 1-10 scale.")

        st.subheader("Distribution Fit")
        st.pyplot(
            build_distribution_fit_figure(
                result,
                sidebar.adjustment_target,
                sidebar.x_pad,
                sidebar.y_pad,
                sidebar.axis_config,
            ),
            clear_figure=True,
        )

        st.subheader("PDF / CDF Comparison")
        st.pyplot(
            build_distribution_comparison_figure(
                result,
                sidebar.adjustment_target,
                sidebar.x_pad,
                sidebar.y_pad,
                sidebar.axis_config,
            ),
            clear_figure=True,
        )

        st.subheader("Hazard Function")
        st.pyplot(
            build_hazard_function_figure(
                result,
                sidebar.adjustment_target,
                sidebar.x_pad,
                sidebar.y_pad,
                sidebar.axis_config,
            ),
            clear_figure=True,
        )

        st.subheader("Forward Risk Trend")
        st.pyplot(
            build_forward_risk_figure(
                result,
                sidebar.current_age,
                trend_horizon,
                sidebar.adjustment_target,
                sidebar.x_pad,
                sidebar.y_pad,
                sidebar.axis_config,
            ),
            clear_figure=True,
        )

        st.subheader("Replacement Economics")
        st.pyplot(
            build_cost_curve_figure(
                result,
                sidebar.preventive_cost,
                sidebar.failure_cost,
                sidebar.adjustment_target,
                sidebar.x_pad,
                sidebar.y_pad,
                sidebar.axis_config,
            ),
            clear_figure=True,
        )

        st.subheader("Interpretation Guide")
        st.dataframe(metric_descriptions_frame(), use_container_width=True, hide_index=True)
        st.dataframe(plot_descriptions_frame(), use_container_width=True, hide_index=True)

    with export_tab:
        st.subheader("Download Outputs")
        selected_component = st.session_state.get("selected_component", list(analysis.keys())[0])
        result = analysis[selected_component]
        report_pdf_key = f"reliability_report_pdf_{safe_filename(selected_component)}_{selected_distribution}"
        if st.button("Prepare Reliability Report PDF"):
            with st.spinner("Preparing PDF report..."):
                st.session_state[report_pdf_key] = build_report_cached(
                    df_results,
                    result,
                    sidebar.current_age,
                    trend_horizon,
                    sidebar.preventive_cost,
                    sidebar.failure_cost,
                    sidebar.adjustment_target,
                    sidebar.x_pad,
                    sidebar.y_pad,
                    sidebar.axis_config,
                    REPORT_CACHE_VERSION,
                )
        if report_pdf_key in st.session_state:
            st.download_button(
                "Download Reliability Report PDF",
                st.session_state[report_pdf_key],
                f"reliability_report_{safe_filename(selected_component)}.pdf",
                "application/pdf",
            )

        cleaned_input = pd.DataFrame({name: pd.Series(values) for name, values in parse_result.datasets.items()})
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

    st.caption("All future risk metrics are conditional on surviving to the current in-service time.")


def main() -> None:
    render_dashboard()
