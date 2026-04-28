import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.special import gamma

st.set_page_config(layout="wide")

st.title("🔧 Reliability & Weibull Analysis Dashboard")

# ==========================================
# SIDEBAR INPUTS
# ==========================================
st.sidebar.header("Inputs")

mttr = st.sidebar.number_input("MTTR", value=10.0)
t_current = st.sidebar.number_input("In Service Time", value=5000.0)
t_future = st.sidebar.number_input("Future Runtime", value=500.0)
Cp = st.sidebar.number_input("Preventive Cost (Cp)", value=1000.0)
Cf = st.sidebar.number_input("Failure Cost (Cf)", value=10000.0)

# ==========================================
# FILE UPLOAD
# ==========================================
uploaded_file = st.file_uploader("Upload Excel File (Each column = component)")

datasets = {}

if uploaded_file:
    df = pd.read_excel(uploaded_file)

    for col in df.columns:
        data = df[col].dropna().values
        if len(data) > 0:
            datasets[col] = data

else:
    st.info("Using sample data")
    datasets = {
        "Component_A": np.array([928, 1675, 5354, 5554, 5705, 5779]),
        "Component_B": np.array([5806, 5934, 5965, 6070, 6089]),
    }

# ==========================================
# FUNCTIONS
# ==========================================
def neg_ll(params, t):
    b, e = params
    if b <= 0 or e <= 0:
        return 1e10
    n = len(t)
    ll = n*np.log(b) - n*b*np.log(e) + (b-1)*np.sum(np.log(t)) - np.sum((t/e)**b)
    return -ll

def R(t, b, e):
    return np.exp(-(t/e)**b)

def hazard(t, b, e):
    return (b/e)*(t/e)**(b-1)

# ==========================================
# MAIN ANALYSIS
# ==========================================
results = []

for name, data in datasets.items():

    data = np.sort(data)

    res = minimize(neg_ll, [1.0, np.mean(data)], args=(data,), method='Nelder-Mead')
    beta, eta = res.x

    mttf = eta * gamma(1 + 1/beta)
    cond_rel = R(t_current + t_future, beta, eta) / R(t_current, beta, eta)
    prob_fail = 1 - cond_rel
    RUL = mttf - t_current

    T_range = np.linspace(100, max(data)*1.5, 300)
    cost = (Cp + Cf*(1 - R(T_range, beta, eta))) / T_range
    opt_T = T_range[np.argmin(cost)]

    results.append([name, beta, eta, prob_fail, RUL, opt_T])

    # ======================================
    # PLOTS (SCATTER STYLE)
    # ======================================
    col1, col2, col3 = st.columns(3)

    n = len(data)
    ranks = np.arange(1, n+1)
    median_ranks = (ranks - 0.3) / (n + 0.4)

    t_range = np.linspace(0, max(data)*1.2, 500)

    pdf = (beta/eta)*(t_range/eta)**(beta-1)*np.exp(-(t_range/eta)**beta)
    pdf_points = (beta/eta)*(data/eta)**(beta-1)*np.exp(-(data/eta)**beta)
    cdf = 1 - np.exp(-(t_range/eta)**beta)

    # PDF
    fig1, ax1 = plt.subplots()
    ax1.plot(t_range, pdf)
    ax1.scatter(data, pdf_points)
    ax1.set_title(f"{name} PDF")
    ax1.grid()
    col1.pyplot(fig1)

    # CDF
    fig2, ax2 = plt.subplots()
    ax2.plot(t_range, cdf)
    ax2.scatter(data, median_ranks)
    ax2.set_title(f"{name} CDF")
    ax2.grid()
    col2.pyplot(fig2)

    # Hazard
    fig3, ax3 = plt.subplots()
    ax3.plot(t_range, hazard(t_range, beta, eta))
    ax3.set_title(f"{name} Hazard")
    ax3.grid()
    col3.pyplot(fig3)

# ==========================================
# RESULTS TABLE
# ==========================================
df_results = pd.DataFrame(results, columns=[
    "Component", "Beta", "Eta", "Failure Probability", "RUL", "Optimal Replacement"
])

df_results = df_results.sort_values(by="Failure Probability", ascending=False)

st.subheader("📊 Component Risk Ranking")
st.dataframe(df_results)

# ==========================================
# ALERTS
# ==========================================
st.subheader("🚨 Alerts")

for _, row in df_results.iterrows():
    if row["Failure Probability"] > 0.6:
        st.error(f"{row['Component']} → HIGH RISK")
    elif row["Failure Probability"] > 0.3:
        st.warning(f"{row['Component']} → MEDIUM RISK")
    else:
        st.success(f"{row['Component']} → LOW RISK")

# ==========================================
# DOWNLOAD
# ==========================================
csv = df_results.to_csv(index=False).encode('utf-8')
st.download_button("Download Results", csv, "results.csv", "text/csv")