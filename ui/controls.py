from __future__ import annotations

import streamlit as st


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
    with st.expander("Advanced axis bounds and units", expanded=False):
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
            x_major_unit = st.number_input("X major unit", min_value=0.0, value=100.0, step=100.0, disabled=not use_major_units, key="axis_x_major_unit")
        with u_col_2:
            y_major_unit = st.number_input("Y major unit", min_value=0.0, value=0.10, step=0.01, format="%.6f", disabled=not use_major_units, key="axis_y_major_unit")

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
