
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import io

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(layout="wide")
st.title("🏦 AI Credit Risk Decision Dashboard")

# =============================
# LOAD MODEL
# =============================
model = joblib.load("models/xgboost_model.pkl")

# =============================
# SIDEBAR CONTROLS
# =============================
st.sidebar.header("Business Policy Controls")

threshold = st.sidebar.slider(
    "Approval Threshold",
    0.1,
    0.9,
    0.45,
    0.05
)

st.sidebar.write("Approve loan if default probability < threshold")

# =============================
# INPUT MODE
# =============================
st.sidebar.header("Input Mode")

input_mode = st.sidebar.radio(
    "Select Input Type",
    [
        "Single Customer Prediction",
        "Portfolio Analysis (CSV Upload)"
    ]
)

# =============================
# PDF REPORT FUNCTION
# =============================
def generate_pdf_report(income, credit, goods_price, prob, decision):

    buffer = io.BytesIO()

    doc = SimpleDocTemplate(buffer, pagesize=letter)

    styles = getSampleStyleSheet()
    content = []

    title = Paragraph("AI Credit Risk Assessment Report", styles["Title"])
    content.append(title)
    content.append(Spacer(1,20))

    data = [
        ["Feature","Value"],
        ["Annual Income",income],
        ["Credit Amount",credit],
        ["Goods Price",goods_price],
        ["Default Probability",f"{prob:.2%}"],
        ["Decision",decision]
    ]

    table = Table(data)

    table.setStyle([
        ("BACKGROUND",(0,0),(-1,0),colors.grey),
        ("GRID",(0,0),(-1,-1),1,colors.black)
    ])

    content.append(table)
    doc.build(content)

    buffer.seek(0)

    return buffer


# ====================================================
# SINGLE CUSTOMER MODE
# ====================================================
if input_mode == "Single Customer Prediction":

    st.header("Customer Loan Application")

    col1, col2, col3 = st.columns(3)

    with col1:
        income = st.number_input("Annual Income",0,10000000,500000)

    with col2:
        credit = st.number_input("Credit Amount",0,10000000,200000)

    with col3:
        goods_price = st.number_input("Goods Price",0,10000000,180000)

    predict = st.button("Run Credit Risk Assessment")

    if predict:

        template = pd.read_csv("data/processed/X_test.csv").iloc[:1].copy()
        template[:] = 0

        template["AMT_INCOME_TOTAL"] = income
        template["AMT_CREDIT"] = credit
        template["AMT_GOODS_PRICE"] = goods_price

        prob = model.predict_proba(template)[0][1]

        risk_score = int((1-prob)*100)

        st.header("Customer Risk Score")

        col1, col2 = st.columns(2)

        # =============================
        # RISK GAUGE
        # =============================
        with col1:

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob*100,
                title={"text":"Default Risk (%)"},
                gauge={
                    "axis":{"range":[0,100]},
                    "bar":{"color":"red"},
                    "steps":[
                        {"range":[0,40],"color":"green"},
                        {"range":[40,70],"color":"yellow"},
                        {"range":[70,100],"color":"red"}
                    ]
                }
            ))

            st.plotly_chart(fig)

        # =============================
        # RISK SCORE PANEL
        # =============================
        with col2:

            st.metric("Default Probability",f"{prob:.2%}")
            st.metric("Credit Risk Score",f"{risk_score}/100")

            if prob < threshold:
                st.success("Loan Approved")
                decision = "Loan Approved"
            else:
                st.error("Loan Rejected")
                decision = "Loan Rejected"

                # =============================
                # AI RECOMMENDATION ENGINE
                # =============================
                st.subheader("AI Recommendation")

                recommendations = []

                if credit > income*0.5:
                    rec_credit = int(income*0.5)
                    recommendations.append(
                        f"Reduce credit amount to around ₹{rec_credit}"
                    )

                if goods_price > credit:
                    recommendations.append(
                        "Ensure goods price does not exceed loan amount"
                    )

                if income < 300000:
                    recommendations.append(
                        "Increase income declaration or provide guarantor"
                    )

                if len(recommendations) == 0:
                    recommendations.append(
                        "Improve financial stability or credit history"
                    )

                for r in recommendations:
                    st.write("•",r)

        # =============================
        # SHAP EXPLANATION
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
        # PDF REPORT
        # =============================
        pdf = generate_pdf_report(
            income,
            credit,
            goods_price,
            prob,
            decision
        )

        st.download_button(
            label="Download Credit Risk Report (PDF)",
            data=pdf,
            file_name="credit_risk_report.pdf",
            mime="application/pdf"
        )


# ====================================================
# PORTFOLIO ANALYSIS MODE
# ====================================================
if input_mode == "Portfolio Analysis (CSV Upload)":

    st.header("Customer Portfolio Risk Analyzer")

    uploaded_file = st.file_uploader(
        "Upload Customer Dataset (CSV)",
        type=["csv"]
    )

    if uploaded_file is not None:

        df_portfolio = pd.read_csv(uploaded_file)

        st.subheader("Uploaded Data Preview")
        st.dataframe(df_portfolio.head())

        template = pd.read_csv("data/processed/X_test.csv")

        portfolio_template = pd.DataFrame(
            0,
            index=df_portfolio.index,
            columns=template.columns
        )

        if "AMT_INCOME_TOTAL" in df_portfolio.columns:
            portfolio_template["AMT_INCOME_TOTAL"] = df_portfolio["AMT_INCOME_TOTAL"]

        if "AMT_CREDIT" in df_portfolio.columns:
            portfolio_template["AMT_CREDIT"] = df_portfolio["AMT_CREDIT"]

        if "AMT_GOODS_PRICE" in df_portfolio.columns:
            portfolio_template["AMT_GOODS_PRICE"] = df_portfolio["AMT_GOODS_PRICE"]

        probs = model.predict_proba(portfolio_template)[:,1]

        df_portfolio["default_probability"] = probs

        df_portfolio["decision"] = np.where(
            probs < threshold,
            "Approve",
            "Reject"
        )

        # =============================
        # SHAP EXPLANATIONS
        # =============================
        explainer = shap.TreeExplainer(model)

        shap_values = explainer(portfolio_template)

        top_features = []

        for i in range(len(df_portfolio)):

            values = shap_values.values[i]

            feature_importance = pd.Series(
                values,
                index=portfolio_template.columns
            ).abs().sort_values(ascending=False)

            top3 = feature_importance.head(3).index.tolist()

            top_features.append(", ".join(top3))

        df_portfolio["top_risk_factors"] = top_features

        st.subheader("Portfolio Prediction Results")
        st.dataframe(df_portfolio)

        # =============================
        # PORTFOLIO STATS
        # =============================
        total = len(df_portfolio)
        approved = (df_portfolio["decision"]=="Approve").sum()
        rejected = (df_portfolio["decision"]=="Reject").sum()

        col1,col2,col3 = st.columns(3)

        col1.metric("Total Customers",total)
        col2.metric("Approved Loans",approved)
        col3.metric("Rejected Loans",rejected)

        # =============================
        # RISK DISTRIBUTION
        # =============================
        fig,ax = plt.subplots()

        ax.hist(df_portfolio["default_probability"],bins=20)

        ax.set_xlabel("Default Probability")
        ax.set_ylabel("Customers")
        ax.set_title("Portfolio Risk Distribution")

        st.pyplot(fig)

