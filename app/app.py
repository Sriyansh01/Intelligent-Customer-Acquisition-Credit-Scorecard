import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import plotly.graph_objects as go
import plotly.express as px

# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------

st.set_page_config(layout="wide")
st.title("🏦 Intelligent Customer Acquisition Credit Scorecard")

# ------------------------------------------------
# LOAD MODEL
# ------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load("models/xgboost_model.pkl")

model = load_model()

# ------------------------------------------------
# LOAD TEMPLATE FEATURES
# ------------------------------------------------

@st.cache_data
def load_template():
    return pd.read_csv("data/processed/X_test.csv")

X_template = load_template()

# ------------------------------------------------
# HUMAN READABLE FEATURE NAME
# ------------------------------------------------

def humanize(name):

    name = name.replace("AMT_", "Amount ")
    name = name.replace("EXT_SOURCE_", "External Credit Score ")
    name = name.replace("DAYS_BIRTH", "Customer Age")
    name = name.replace("FLAG_", "")
    name = name.replace("NAME_", "")
    name = name.replace("CODE_", "")
    name = name.replace("_", " ")

    return name.title()

# ------------------------------------------------
# SIDEBAR POLICY
# ------------------------------------------------

st.sidebar.header("Credit Policy")

threshold = st.sidebar.slider(
    "Approval Threshold",
    0.1,
    0.9,
    0.45,
    0.05
)

# ------------------------------------------------
# TABS
# ------------------------------------------------

tab1, tab2 = st.tabs([
    "Customer Decision",
    "Portfolio Analysis"
])

# =================================================
# CUSTOMER DECISION
# =================================================

with tab1:

    st.header("Customer Loan Application")

    col1, col2, col3 = st.columns(3)

    with col1:
        income = st.number_input("Annual Income",0,10000000,500000)
        age = st.slider("Customer Age",18,70,35)
        ext1 = st.slider("External Credit Score A",0.0,1.0,0.5)

    with col2:
        credit = st.number_input("Loan Amount",0,10000000,200000)
        ext2 = st.slider("External Credit Score B",0.0,1.0,0.5)
        owns_car = st.selectbox("Owns Car",[0,1])

    with col3:
        goods_price = st.number_input("Goods Price",0,10000000,180000)
        ext3 = st.slider("External Credit Score C",0.0,1.0,0.5)
        married = st.selectbox("Married",[0,1])

    run_model = st.button("Run Credit Risk Assessment")

    if run_model:

        # IMPORTANT: copy template row
        template = X_template.iloc[[0]].copy()

        # modify features
        template.loc[:, "AMT_INCOME_TOTAL"] = income
        template.loc[:, "AMT_CREDIT"] = credit
        template.loc[:, "AMT_GOODS_PRICE"] = goods_price

        if "AMT_ANNUITY" in template.columns:
            template.loc[:, "AMT_ANNUITY"] = credit / 12

        if "EXT_SOURCE_1" in template.columns:
            template.loc[:, "EXT_SOURCE_1"] = ext1

        if "EXT_SOURCE_2" in template.columns:
            template.loc[:, "EXT_SOURCE_2"] = ext2

        if "EXT_SOURCE_3" in template.columns:
            template.loc[:, "EXT_SOURCE_3"] = ext3

        if "DAYS_BIRTH" in template.columns:
            template.loc[:, "DAYS_BIRTH"] = -age*365

        if "FLAG_OWN_CAR_Y" in template.columns:
            template.loc[:, "FLAG_OWN_CAR_Y"] = owns_car

        if "NAME_FAMILY_STATUS_Married" in template.columns:
            template.loc[:, "NAME_FAMILY_STATUS_Married"] = married

        # ensure column order
        template = template[X_template.columns]

        # prediction
        prob = model.predict_proba(template)[0][1]

        credit_score = int(300 + (1-prob)*600)

        st.subheader("Prediction Result")

        col1, col2 = st.columns(2)

        with col1:

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob*100,
                title={"text":"Default Risk (%)"},
                gauge={
                    "axis":{"range":[0,100]},
                    "steps":[
                        {"range":[0,40],"color":"green"},
                        {"range":[40,70],"color":"yellow"},
                        {"range":[70,100],"color":"red"}
                    ]
                }
            ))

            st.plotly_chart(fig)

        with col2:

            st.metric("Default Probability",f"{prob:.2%}")
            st.metric("AI Credit Score",credit_score)

            if prob < threshold:
                st.success("Loan Approved")
            else:
                st.error("Loan Rejected")

        # SHAP explanation

        st.subheader("Top Risk Drivers")

        explainer = shap.TreeExplainer(model)
        shap_values = explainer(template)

        shap_importance = np.abs(shap_values.values[0])

        top_idx = np.argsort(shap_importance)[-3:][::-1]

        for i in top_idx:

            feature = humanize(template.columns[i])

            if shap_values.values[0][i] > 0:
                st.error(f"{feature} increases risk")
            else:
                st.success(f"{feature} reduces risk")

# =================================================
# PORTFOLIO ANALYSIS
# =================================================

with tab2:

    st.header("Customer Portfolio Risk Analyzer")

    file = st.file_uploader("Upload CSV",type=["csv"])

    if file:

        df = pd.read_csv(file)

        st.subheader("Uploaded Data")
        st.dataframe(df)

        # create portfolio dataset with correct features

        portfolio = pd.DataFrame(
            np.tile(X_template.iloc[0].values,(len(df),1)),
            columns=X_template.columns
        )

        if "AMT_INCOME_TOTAL" in df.columns:
            portfolio["AMT_INCOME_TOTAL"] = df["AMT_INCOME_TOTAL"]

        if "AMT_CREDIT" in df.columns:
            portfolio["AMT_CREDIT"] = df["AMT_CREDIT"]

        if "AMT_GOODS_PRICE" in df.columns:
            portfolio["AMT_GOODS_PRICE"] = df["AMT_GOODS_PRICE"]

        probs = model.predict_proba(portfolio)[:,1]

        df["default_probability"] = probs

        df["decision"] = np.where(
            probs < threshold,
            "Approve",
            "Reject"
        )

        # SHAP reasons

        explainer = shap.TreeExplainer(model)
        shap_values = explainer(portfolio)

        reasons = []

        for i in range(len(portfolio)):

            shap_row = shap_values.values[i]

            idx = np.argmax(np.abs(shap_row))

            feature = humanize(portfolio.columns[idx])

            if shap_row[idx] > 0:
                reason = f"High {feature}"
            else:
                reason = f"Strong {feature}"

            reasons.append(reason)

        df["reason"] = reasons

        st.subheader("Prediction Results")
        st.dataframe(df)

        total=len(df)
        approved=(df["decision"]=="Approve").sum()
        rejected=(df["decision"]=="Reject").sum()

        col1,col2,col3 = st.columns(3)

        col1.metric("Total Customers",total)
        col2.metric("Approved Loans",approved)
        col3.metric("Rejected Loans",rejected)

        # risk distribution

        st.subheader("Portfolio Risk Distribution")

        df["risk_category"]=pd.cut(
            df["default_probability"],
            bins=[0,0.3,0.6,1],
            labels=["Low Risk","Medium Risk","High Risk"]
        )

        risk_counts=df["risk_category"].value_counts()

        fig = px.bar(
            x=risk_counts.index,
            y=risk_counts.values,
            labels={"x":"Risk Level","y":"Customers"}
        )

        st.plotly_chart(fig)

        st.info("""
🟢 Low Risk (0–30%) → Safe borrowers  
🟡 Medium Risk (30–60%) → Moderate risk  
🔴 High Risk (60–100%) → High probability of default
""")