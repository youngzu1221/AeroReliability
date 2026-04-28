import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.special import gamma

# ============================================================
# PAGE SETUP
# ============================================================
st.set_page_config(page_title="Reliability & Weibull Dashboard", layout="wide")
st.title("🔧 Reliability & Weibull Predictive Dashboard")

# ============================================================
# FUNCTIONS
# ============================================================
def reliability(t, beta, eta):
    t = np.asarray(t, dtype=float)
    t = np.maximum(t, 1e-12)
    return np.exp(-((t / eta) ** beta))

def hazard(t, beta, eta):
    t = np.asarray(t, dtype=float)
    t = np.maximum(t, 1e-12)
    return (beta / eta) * (t / eta) ** (beta - 1)

def weibull_pdf(t, beta, eta):
    t = np.asarray(t, dtype=float)
    t = np.maximum(t, 1e-12)
    return (beta / eta) * (t / eta) ** (beta - 1) * np.exp(-((t / eta) ** beta))

def neg_log_likelihood(params, data):
    beta, eta = params
    if beta <= 0 or eta <= 0:
        return 1e20

    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]
    data = data[data > 0]

    if len(data) < 2:
        return 1e20

    n = len(data)
    ll = (
        n * np.log(beta)
        - n * beta * np.log(eta)
        + (beta - 1) * np.sum(np.log(data))
        - np.sum((data / eta) ** beta)
    )
    return -ll

def estimate_weibull_mle(data):
    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]
    data = data[data > 0]
    data = np.sort(data)

    if len(data) < 2:
        raise ValueError("Need at least 2 valid positive data points.")

    initial_guess = np.array([1.5, float(np.mean(data)) if np.mean(data) > 0 else 1.0])

    result = minimize(
        neg_log_likelihood,
        x0=initial_guess,
        args=(data,),
        method="L-BFGS-B",
        bounds=[(1e-6, None), (1e-6, None)],
    )

    if not result.success:
        raise RuntimeError(result.message)

    beta, eta = result.x
    return float(beta), float(eta)

def fmt_pct(x):
    return f"{float(x):.2%}"

def risk_label(p):
    if p >= 0.60:
        return "HIGH"
    elif p >= 0.30:
        return "MEDIUM"
    return "LOW"

def parse_excel(uploaded_file):
    df = pd.read_excel(uploaded_file)
    datasets = {}
    for col in df.columns:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        arr = np.sort(s.to_numpy(dtype=float))
        if len(arr) > 0:
            datasets[str(col)] = arr
    return datasets

def sample_datasets():
    return {
        "Component_A": np.array([
            928.9936, 1675.1729, 5354.2334, 5554.634, 5574.3295,
            5579.8798, 5625.0248, 5647.0735, 5705.0101, 5779.6506,
            5786.7997, 5806.2507, 5809.2236, 5934.8736, 5941.1176,
            5965.4527, 5968.6011, 5981.9664, 6070.8261, 6087.7153
        ]),
        "Component_B": np.array([
            6089.3006, 6139.7166, 6144.9501, 6188.6527, 6229.9567,
            6269.3167, 6311.3682, 6321.6174, 6331.5658, 6345.3501,
            6360.1913, 6365.9809, 6393.4409, 6395.3338, 6411.2999
        ]),
        "Component_C": np.array([
            6420.254, 6426.0969, 6453.4502, 6458.4883, 6468.7205,
            6473.452, 6484.9734, 6502.7477, 6518.1449, 6520.6828,
            6522.0342, 6522.3946, 6523.9841, 6528.3539, 6530.8394
        ]),
    }

# ============================================================
# SIDEBAR INPUTS
# ============================================================
st.sidebar.header("Inputs")

mttr = st.sidebar.number_input("MTTR", min_value=0.0, value=10.0, step=1.0)
t_current = st.sidebar.number_input("In Service Time", min_value=0.0, value=5000.0, step=100.0)
t_future = st.sidebar.number_input("Future Runtime", min_value=0.0, value=500.0, step=100.0)
Cp = st.sidebar.number_input("Preventive Cost (Cp)", min_value=0.0, value=1000.0, step=100.0)
Cf = st.sidebar.number_input("Failure Cost (Cf)", min_value=0.0, value=10000.0, step=100.0)
prediction_horizon = st.sidebar.slider("Prediction Horizon (Cycles)", 100, 10000, 3000, 100)

uploaded_file = st.file_uploader("Upload Excel File (each column = one component)", type=["xlsx", "xls"])

# ============================================================
# DATA LOADING
# ============================================================
if uploaded_file is not None:
    try:
        datasets = parse_excel(uploaded_file)
        if not datasets:
            st.error("No valid numeric columns were found in the uploaded file.")
            st.stop()
    except Exception as e:
        st.error(f"Could not read file: {e}")
        st.stop()
else:
    st.info("No file uploaded. Using sample data.")
    datasets = sample_datasets()

if not datasets:
    st.error("No data found.")
    st.stop()

# ============================================================
# ANALYSIS
# ============================================================
results = []

for name, data in datasets.items():
    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]
    data = data[data > 0]
    data = np.sort(data)

    if len(data) < 2:
        continue

    beta, eta = estimate_weibull_mle(data)

    mttf = eta * gamma(1.0 + 1.0 / beta)
    mtbf = mttf + mttr

    r_current = float(reliability(t_current, beta, eta))
    r_future = float(reliability(t_current + t_future, beta, eta))

    conditional_reliability = r_future / r_current if r_current > 0 else 0.0
    conditional_reliability = float(np.clip(conditional_reliability, 0.0, 1.0))
    conditional_failure_probability = 1.0 - conditional_reliability
    conditional_failure_probability = float(np.clip(conditional_failure_probability, 0.0, 1.0))

    rul = mttf - t_current

    t_range = np.linspace(
        max(1e-6, np.min(data) * 0.5),
        max(np.max(data) * 1.5, t_current + t_future + 1),
        500,
    )
    cost_rate = (Cp + Cf * (1 - reliability(t_range, beta, eta))) / t_range
    optimal_t = float(t_range[np.argmin(cost_rate)])

    future_times = np.linspace(max(1e-6, t_current), t_current + prediction_horizon, 200)
    rel_trend = reliability(future_times, beta, eta)
    fail_trend = 1 - rel_trend
    haz_trend = hazard(future_times, beta, eta)

    results.append({
        "Component": name,
        "Beta": beta,
        "Eta": eta,
        "MTTF": mttf,
        "MTBF": mtbf,
        "R(T+t)": r_future,
        "R(T)": r_current,
        "Conditional Reliability": conditional_reliability,
        "Conditional Probability of Failure": conditional_failure_probability,
        "RUL": rul,
        "Optimal Replacement": optimal_t,
        "Future Failure at Horizon": float(fail_trend[-1]),
        "Risk": risk_label(conditional_failure_probability),
    })

if not results:
    st.error("No component could be analyzed.")
    st.stop()

df_results = pd.DataFrame(results).sort_values(
    by="Conditional Probability of Failure",
    ascending=False
).reset_index(drop=True)

# ============================================================
# DASHBOARD SUMMARY
# ============================================================
top_row = df_results.iloc[0]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Components Analyzed", f"{len(df_results)}")
c2.metric("Highest Failure Probability", fmt_pct(top_row["Conditional Probability of Failure"]))
c3.metric("Highest Risk Component", str(top_row["Component"]))
c4.metric("Risk Level", str(top_row["Risk"]))

# ============================================================
# RESULTS TABLE
# ============================================================
st.subheader("📊 Full Reliability Results")

display_df = df_results.copy()

for col in ["R(T+t)", "R(T)", "Conditional Reliability", "Conditional Probability of Failure", "Future Failure at Horizon"]:
    display_df[col] = display_df[col].apply(fmt_pct)

for col in ["Beta", "Eta", "MTTF", "MTBF", "RUL", "Optimal Replacement"]:
    display_df[col] = display_df[col].map(lambda x: f"{x:,.2f}")

st.dataframe(display_df, use_container_width=True)

csv = df_results.to_csv(index=False).encode("utf-8")
st.download_button("Download Results CSV", csv, "weibull_results.csv", "text/csv")

# ============================================================
# COMPONENT DETAIL VIEW
# ============================================================
component_names = list(datasets.keys())
selected_component = st.selectbox("Select a component", component_names)

selected_data = np.asarray(datasets[selected_component], dtype=float)
selected_data = selected_data[np.isfinite(selected_data)]
selected_data = selected_data[selected_data > 0]
selected_data = np.sort(selected_data)

selected_row = df_results[df_results["Component"] == selected_component].iloc[0]

beta = float(selected_row["Beta"])
eta = float(selected_row["Eta"])
mttf = float(selected_row["MTTF"])
mtbf = float(selected_row["MTBF"])
r_current = float(selected_row["R(T)"])
r_future = float(selected_row["R(T+t)"])
conditional_reliability = float(selected_row["Conditional Reliability"])
conditional_failure_probability = float(selected_row["Conditional Probability of Failure"])
rul = float(selected_row["RUL"])
optimal_t = float(selected_row["Optimal Replacement"])

st.subheader(f"📌 {selected_component} Key Metrics")

m1, m2, m3 = st.columns(3)
m1.metric("MTTF", f"{mttf:,.2f}")
m2.metric("MTBF", f"{mtbf:,.2f}")
m3.metric("RUL", f"{rul:,.2f}")

n1, n2, n3 = st.columns(3)
n1.metric("Unconditional Reliability R(T+t)", fmt_pct(r_future))
n2.metric("Current Reliability R(T)", fmt_pct(r_current))
n3.metric("Conditional Reliability", fmt_pct(conditional_reliability))

st.metric("Conditional Probability of Failure", fmt_pct(conditional_failure_probability))

a1, a2 = st.columns(2)
with a1:
    if conditional_failure_probability >= 0.60:
        st.error(f"🚨 HIGH RISK: {selected_component}")
    elif conditional_failure_probability >= 0.30:
        st.warning(f"⚠️ MEDIUM RISK: {selected_component}")
    else:
        st.success(f"✅ LOW RISK: {selected_component}")

with a2:
    st.info(f"Optimal Replacement Time: {optimal_t:,.2f}")

# ============================================================
# ORIGINAL STYLE PLOTS
# ============================================================
n = len(selected_data)
ranks = np.arange(1, n + 1)
median_ranks = (ranks - 0.3) / (n + 0.4)

t_range = np.linspace(
    max(1e-6, np.min(selected_data) * 0.5),
    max(np.max(selected_data) * 1.2, t_current + t_future + 1),
    1000,
)

pdf_curve = weibull_pdf(t_range, beta, eta)
cdf_curve = 1 - np.exp(-((t_range / eta) ** beta))
pdf_points = weibull_pdf(selected_data, beta, eta)

st.subheader(f"📈 Distribution Fit - {selected_component}")
col_pdf, col_cdf = st.columns(2)

fig_pdf, ax_pdf = plt.subplots(figsize=(8, 4))
ax_pdf.plot(t_range, pdf_curve, label="MLE Model")
ax_pdf.scatter(selected_data, pdf_points, s=20, label="Actual Data")
ax_pdf.set_title("PDF (Probability Density)")
ax_pdf.set_xlabel("Time")
ax_pdf.set_ylabel("Density")
ax_pdf.grid(True, linestyle="--", alpha=0.5)
ax_pdf.legend()
col_pdf.pyplot(fig_pdf)

fig_cdf, ax_cdf = plt.subplots(figsize=(8, 4))
ax_cdf.plot(t_range, cdf_curve, label="MLE Model")
ax_cdf.scatter(selected_data, median_ranks, s=20, label="Actual Data")
ax_cdf.set_title("CDF (Unreliability)")
ax_cdf.set_xlabel("Time")
ax_cdf.set_ylabel("Probability")
ax_cdf.grid(True, linestyle="--", alpha=0.5)
ax_cdf.legend()
col_cdf.pyplot(fig_cdf)

fig_haz, ax_haz = plt.subplots(figsize=(8, 4))
ax_haz.plot(t_range, hazard(t_range, beta, eta), label="Hazard Rate")
ax_haz.set_title("Hazard Function")
ax_haz.set_xlabel("Time")
ax_haz.set_ylabel("Failure Rate")
ax_haz.grid(True, linestyle="--", alpha=0.5)
ax_haz.legend()
st.pyplot(fig_haz)

# ============================================================
# TREND OVER TIME
# ============================================================
st.subheader(f"📉 Degradation Trend - {selected_component}")

future_times = np.linspace(max(1e-6, t_current), t_current + prediction_horizon, 200)
rel_trend = reliability(future_times, beta, eta)
fail_trend = 1 - rel_trend
haz_trend = hazard(future_times, beta, eta)

trend1, trend2, trend3 = st.columns(3)

fig_t1, ax_t1 = plt.subplots(figsize=(6, 4))
ax_t1.plot(future_times, rel_trend)
ax_t1.axvline(t_current, linestyle="--")
ax_t1.set_title("Reliability Decay")
ax_t1.set_xlabel("Time")
ax_t1.set_ylabel("Reliability")
ax_t1.grid(True, linestyle="--", alpha=0.5)
trend1.pyplot(fig_t1)

fig_t2, ax_t2 = plt.subplots(figsize=(6, 4))
ax_t2.plot(future_times, fail_trend)
ax_t2.axvline(t_current, linestyle="--")
ax_t2.set_title("Failure Probability Growth")
ax_t2.set_xlabel("Time")
ax_t2.set_ylabel("Probability")
ax_t2.grid(True, linestyle="--", alpha=0.5)
trend2.pyplot(fig_t2)

fig_t3, ax_t3 = plt.subplots(figsize=(6, 4))
ax_t3.plot(future_times, haz_trend)
ax_t3.axvline(t_current, linestyle="--")
ax_t3.set_title("Hazard Rate Trend")
ax_t3.set_xlabel("Time")
ax_t3.set_ylabel("Failure Rate")
ax_t3.grid(True, linestyle="--", alpha=0.5)
trend3.pyplot(fig_t3)

future_failure_at_horizon = float(fail_trend[-1])

st.subheader("🔮 Future Risk Insight")
if future_failure_at_horizon >= 0.70:
    st.error(f"🚨 Very High Risk within the next {prediction_horizon} cycles: {fmt_pct(future_failure_at_horizon)}")
elif future_failure_at_horizon >= 0.40:
    st.warning(f"⚠️ Moderate Risk within the next {prediction_horizon} cycles: {fmt_pct(future_failure_at_horizon)}")
else:
    st.success(f"✅ Low Risk within the next {prediction_horizon} cycles: {fmt_pct(future_failure_at_horizon)}")

st.caption("Failure probability is displayed as a percentage everywhere in the dashboard.")
