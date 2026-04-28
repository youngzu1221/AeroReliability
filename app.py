results = []

for name, data in datasets.items():

    data = np.sort(data)

    # Weibull MLE
    res = minimize(neg_ll, [1.0, np.mean(data)], args=(data,), method='Nelder-Mead')
    beta, eta = res.x

    # Core metrics
    mttf = eta * gamma(1 + 1/beta)
    mtbf = mttf + mttr

    R_current = R(t_current, beta, eta)
    R_uncond = R(t_current + t_future, beta, eta)

    cond_rel = R_uncond / R_current if R_current > 0 else 0
    prob_fail = 1 - cond_rel

    RUL = mttf - t_current

    # Optimization
    T_range = np.linspace(100, max(data)*1.5, 300)
    cost = (Cp + Cf*(1 - R(T_range, beta, eta))) / T_range
    opt_T = T_range[np.argmin(cost)]

    # Store results
    results.append([
        name, beta, eta,
        mttf, mtbf,
        R_uncond, R_current,
        cond_rel, prob_fail,
        RUL, opt_T
    ])

    # ======================================
    # DISPLAY KEY METRICS (PER COMPONENT)
    # ======================================
    st.subheader(f"📌 {name} Key Metrics")

    c1, c2, c3 = st.columns(3)
    c1.metric("MTTF", f"{mttf:.2f}")
    c2.metric("MTBF", f"{mtbf:.2f}")
    c3.metric("RUL", f"{RUL:.2f}")

    c4, c5, c6 = st.columns(3)
    c4.metric("R(T)", f"{R_current:.2%}")
    c5.metric("R(T+t)", f"{R_uncond:.2%}")
    c6.metric("Conditional Reliability", f"{cond_rel:.2%}")

    st.metric("Conditional Probability of Failure", f"{prob_fail:.2%}")

    # ======================================
    # PLOTS (KEEP YOUR STYLE)
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
