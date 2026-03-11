import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go

st.set_page_config(layout="wide")

st.title("🏦 AI Credit Risk Decision Dashboard")

# =============================
# Load Model
# =============================
model = joblib.load("models/xgboost_model.pkl")

# =============================
# Sidebar Controls
# =============================
st.sidebar.header("Business Policy Controls")

threshold = st.sidebar.slider(
    "Approval Threshold",
    0.1, 0.9, 0.45, 0.05
)

st.sidebar.write(
    "Approve loan if default probability < threshold"
)

# =============================
# Customer Input Panel
# =============================
st.header("Customer Application")

col1, col2, col3 = st.columns(3)

with col1:
    income = st.number_input("Annual Income", 0, 10000000, 500000)

with col2:
    credit = st.number_input("Credit Amount", 0, 10000000, 200000)

with col3:
    goods_price = st.number_input("Goods Price", 0, 10000000, 180000)

predict = st.button("Run Risk Assessment")

# =============================
# Prediction
# =============================
if predict:

    template = pd.read_csv("data/processed/X_test.csv").iloc[:1].copy()

    template[:] = 0

    template["AMT_INCOME_TOTAL"] = income
    template["AMT_CREDIT"] = credit
    template["AMT_GOODS_PRICE"] = goods_price

    prob = model.predict_proba(template)[0][1]

    st.header("Customer Risk Score")

    col1, col2 = st.columns(2)

    # =============================
    # Risk Gauge Meter
    # =============================
    with col1:

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            title={"text": "Default Risk (%)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "red"},
                "steps": [
                    {"range": [0, 40], "color": "green"},
                    {"range": [40, 70], "color": "yellow"},
                    {"range": [70, 100], "color": "red"}
                ],
            }
        ))

        st.plotly_chart(fig)

    # =============================
    # Decision Panel
    # =============================
    with col2:

        st.metric("Default Probability", f"{prob:.2%}")

        if prob < threshold:
            st.success("Loan Approved")
        else:
            st.error("Loan Rejected")

    # =============================
    # Customer Risk Profile
    # =============================
    st.header("Customer Risk Profile")

    profile = pd.DataFrame({
        "Feature": [
            "Annual Income",
            "Credit Amount",
            "Goods Price",
            "Risk Score"
        ],
        "Value": [
            income,
            credit,
            goods_price,
            f"{prob*100:.1f}%"
        ]
    })

    st.table(profile)

    # =============================
    # SHAP Local Explanation
    # =============================
    st.header("Prediction Explanation")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(template)

    fig, ax = plt.subplots()

    shap.plots.waterfall(
        shap_values[0],
        show=False
    )

    st.pyplot(fig)

# =============================
# Global Feature Importance
# =============================
st.header("Global Feature Importance")

try:

    X_sample = pd.read_csv("data/processed/X_test.csv").sample(500)

    explainer = shap.TreeExplainer(model)

    shap_values = explainer(X_sample)

    fig, ax = plt.subplots()

    shap.plots.bar(
        shap_values,
        show=False
    )

    st.pyplot(fig)

except:
    st.warning("Feature importance requires processed dataset")

# =============================
# Portfolio Risk Simulator
# =============================
st.header("Portfolio Risk Simulation")

try:

    X_test = pd.read_csv("data/processed/X_test.csv")
    y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()

    probs = model.predict_proba(X_test)[:,1]

    thresholds = np.arange(0.1,0.9,0.05)

    results = []

    for t in thresholds:

        approved = probs < t

        approval_rate = approved.mean()

        default_rate = y_test[approved].mean()

        results.append((t, approval_rate, default_rate))

    results_df = pd.DataFrame(
        results,
        columns=["threshold","approval_rate","default_rate"]
    )

    fig, ax = plt.subplots()

    ax.plot(
        results_df["threshold"],
        results_df["approval_rate"],
        label="Approval Rate"
    )

    ax.plot(
        results_df["threshold"],
        results_df["default_rate"],
        label="Default Rate"
    )

    ax.set_xlabel("Risk Threshold")
    ax.set_ylabel("Rate")

    ax.set_title("Approval vs Default Tradeoff")

    ax.legend()

    st.pyplot(fig)

except:
    st.warning("Portfolio simulation requires test dataset")