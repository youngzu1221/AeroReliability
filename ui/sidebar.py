from __future__ import annotations

from dataclasses import dataclass

import streamlit as st

from plotting.styling import PLOT_ADJUSTMENT_TARGETS
from ui.controls import adjustment_control, axis_bounds_and_units_control, default_plot_padding, reset_active_axis_adjustment


@dataclass(frozen=True)
class SidebarState:
    mttr: float
    current_age: float
    mission_time: float
    preventive_cost: float
    failure_cost: float
    severity: int
    detectability: int
    prediction_horizon: int
    adjustment_target: str
    x_pad: float
    y_pad: float
    axis_config: dict[str, float | bool]


def render_sidebar() -> SidebarState:
    with st.sidebar:
        st.header("Inputs")
        mttr = st.number_input("MTTR", min_value=0.0, value=0.0, step=1.0)
        current_age = st.number_input("Current in-service time", min_value=0.0, value=0.0, step=100.0)
        mission_time = st.number_input("Future mission/runtime", min_value=0.0, value=0.0, step=100.0)
        preventive_cost = st.number_input("Preventive cost (Cp) $", min_value=0.0, value=500.0, step=100.0)
        failure_cost = st.number_input("Failure cost (Cf) $", min_value=0.0, value=5000.0, step=500.0)
        st.caption("FMEA inputs")
        severity = st.number_input("Severity (1-10)", min_value=1, max_value=10, value=5, step=1)
        detectability = st.number_input("Detectability (1-10)", min_value=1, max_value=10, value=5, step=1)
        st.caption("Occurrence is derived automatically from conditional failure probability.")
        prediction_horizon = st.slider("Prediction horizon", 0, 30000, 0, 100)

        st.divider()
        st.subheader("Plot spacing")
        st.session_state.setdefault("axis_adjustment_target", "All plots")
        st.session_state.setdefault("axis_x_pad", default_plot_padding(st.session_state["axis_adjustment_target"]))
        st.session_state.setdefault("axis_y_pad", default_plot_padding(st.session_state["axis_adjustment_target"]))
        adjustment_target = st.selectbox(
            "Apply adjustment to",
            PLOT_ADJUSTMENT_TARGETS,
            key="axis_adjustment_target",
            on_change=reset_active_axis_adjustment,
        )
        x_pad = adjustment_control("Horizontal adjustment", "axis_x_pad")
        y_pad = adjustment_control("Vertical adjustment", "axis_y_pad")
        axis_config = axis_bounds_and_units_control()

        st.divider()
        st.markdown("**Beta interpretation**")
        st.markdown("- Beta < 0.95: infant mortality")
        st.markdown("- 0.95 to 1.05: random failure")
        st.markdown("- Beta > 1.05: wear-out")

    return SidebarState(
        mttr=float(mttr),
        current_age=float(current_age),
        mission_time=float(mission_time),
        preventive_cost=float(preventive_cost),
        failure_cost=float(failure_cost),
        severity=int(severity),
        detectability=int(detectability),
        prediction_horizon=int(prediction_horizon),
        adjustment_target=str(adjustment_target),
        x_pad=float(x_pad),
        y_pad=float(y_pad),
        axis_config=axis_config,
    )
