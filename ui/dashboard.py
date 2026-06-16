from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st

from core.optimization import SUPPORTED_DISTRIBUTIONS, distribution_ppf
from core.reliability import analyze_datasets, decision_from_metrics, results_to_frame
from core.weibull_math import failure_mode_from_beta
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
    fleet_summary_table_frame,
    fit_stat_descriptions_frame,
    fmt_money,
    fmt_pct,
    formatted_results_frame,
    highest_risk_component_text,
    metric_descriptions_frame,
    plot_descriptions_frame,
    safe_filename,
)
from ui.sidebar import render_sidebar

ANALYSIS_CACHE_VERSION = "2026-06-11-analysis-v3"
REPORT_CACHE_VERSION = "2026-05-14-report-v2"
LOGO_PATH = Path(__file__).resolve().parents[1] / "assets" / "ethiopian_airlines_logo.png"


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
    title_col, logo_col = st.columns([5.0, 1.7])
    with title_col:
        st.title("Reliability & Weibull Predictive Dashboard Pro")
        st.caption("Upload an Excel workbook where each column is one component and each value is a positive failure/runtime observation.")
    with logo_col:
        if LOGO_PATH.exists():
            st.image(str(LOGO_PATH), width=230)

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
        ["Distribution Selection", "Component Summary", "Component Deep Dive", "Export"]
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

        st.subheader("Goodness-of-Fit Statistic Guide")
        st.dataframe(fit_stat_descriptions_frame(), use_container_width=True, hide_index=True)

    with summary_tab:
        st.subheader("Full Reliability Results")
        filter_1, filter_2, filter_3, filter_4 = st.columns([1.4, 1.0, 1.0, 1.0])
        component_options = ["All", *sorted(df_results["Component"].astype(str).tolist())]
        selected_component_filter = filter_1.selectbox("Component", component_options, key="summary_component_filter")
        risk_options = sorted(df_results["Risk"].dropna().unique().tolist())
        selected_risk = filter_2.selectbox("Risk", ["", *risk_options], key="summary_risk_filter")
        failure_mode_options = sorted(df_results["Failure Mode"].dropna().unique().tolist())
        selected_failure_mode = filter_3.selectbox(
            "Failure mode",
            ["", *failure_mode_options],
            key="summary_failure_mode_filter",
        )
        sort_labels = {
            "Failure Probability": "Conditional Probability of Failure",
            "RUL": "RUL",
            "MTTF": "MTTF",
            "Optimal Replacement": "Optimal Replacement",
            "Min Cost Rate": "Min Cost Rate",
        }
        selected_sort = filter_4.selectbox("Sort by", ["", *list(sort_labels)], key="summary_sort_by")
        descending = st.toggle("Descending sort", value=True, key="summary_sort_desc", disabled=selected_sort == "")

        filtered_df = df_results.copy()
        if selected_component_filter != "All":
            filtered_df = filtered_df[filtered_df["Component"].astype(str) == selected_component_filter]
        if selected_risk:
            filtered_df = filtered_df[filtered_df["Risk"] == selected_risk]
        if selected_failure_mode:
            filtered_df = filtered_df[filtered_df["Failure Mode"] == selected_failure_mode]

        if filtered_df.empty:
            st.warning("No components match the current summary filters.")
        else:
            if selected_sort:
                filtered_df = filtered_df.sort_values(sort_labels[selected_sort], ascending=not descending)
            filtered_df = filtered_df.reset_index(drop=True)
            fleet_1, fleet_2, fleet_3, fleet_4 = st.columns(4)
            fleet_1.metric("Components Shown", f"{len(filtered_df)}")
            fleet_2.metric(
                "Highest Failure Probability",
                fmt_pct(float(filtered_df["Conditional Probability of Failure"].max())),
            )
            fleet_3.metric("Average MTTF", f"{filtered_df['MTTF'].mean():,.2f}")
            fleet_4.metric("High-Risk Components", f"{int((filtered_df['Risk'] == 'HIGH').sum())}")
            st.caption("Fleet metrics on this tab are calculated using the distribution you selected on the first tab.")
            summary_table = fleet_summary_table_frame(filtered_df)
            st.dataframe(
                summary_table,
                use_container_width=True,
                hide_index=True,
                height=min(640, 76 + 35 * max(len(summary_table), 1)),
                column_config={
                    "Component": st.column_config.TextColumn("Component"),
                    "Distribution": st.column_config.TextColumn("Distribution"),
                    "Primary Parameter": st.column_config.TextColumn("Primary Parameter", width="medium"),
                    "Secondary Parameter": st.column_config.TextColumn("Secondary Parameter", width="medium"),
                    "Characteristic Value": st.column_config.NumberColumn("Characteristic Value", format="%.2f"),
                    "MTTF": st.column_config.NumberColumn("MTTF", format="%.2f"),
                    "MTBF": st.column_config.NumberColumn("MTBF", format="%.2f"),
                    "Conditional Reliability": st.column_config.ProgressColumn(
                        "Conditional Reliability",
                        help="Probability of surviving the mission window given survival to the current in-service time.",
                        min_value=0.0,
                        max_value=100.0,
                        format="%.2f%%",
                    ),
                    "Failure Probability": st.column_config.ProgressColumn(
                        "Failure Probability",
                        help="Probability of failing in the mission window given survival to the current in-service time.",
                        min_value=0.0,
                        max_value=100.0,
                        format="%.2f%%",
                    ),
                    "RUL": st.column_config.NumberColumn("RUL", format="%.2f"),
                    "Optimal Replacement": st.column_config.NumberColumn("Optimal Replacement", format="%.2f"),
                    "Min Cost Rate": st.column_config.NumberColumn("Min Cost Rate", format="$%.2f"),
                    "Failure Mode": st.column_config.TextColumn("Failure Mode"),
                    "Risk": st.column_config.TextColumn("Risk"),
                    "RPN": st.column_config.NumberColumn("RPN", format="%d"),
                },
            )

    with detail_tab:
        selected_component = st.selectbox("Select component", list(analysis.keys()), key="selected_component")
        result = analysis[selected_component]
        decision = decision_from_metrics(result.conditional_failure_probability, result.optimal_replacement)
        beta_interpretation = failure_mode_from_beta(result.beta)

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

        w1, w2 = st.columns([1.0, 1.2])      
        st.caption("Weibull beta guide: Beta < 0.95 = infant mortality, 0.95 to 1.05 = random failure, Beta > 1.05 = wear-out.")

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
    st.markdown(
        "<div style='text-align:center; font-weight:600; margin-top:1.5rem;'>"
        "&copy;2026 - Ethiopian Airlines.<br/>"
        "Developed by Zelalem Geremew and Daniel Jobrie"
        "</div>",
        unsafe_allow_html=True,
    )


def main() -> None:
    render_dashboard()
