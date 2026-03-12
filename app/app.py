
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

# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------

st.set_page_config(layout="wide")
st.title("🏦 AI Credit Risk Decision Platform")

# ------------------------------------------------
# LOAD MODEL
# ------------------------------------------------

model = joblib.load("models/xgboost_model.pkl")

# ------------------------------------------------
# BUSINESS FRIENDLY FEATURE NAMES
# ------------------------------------------------

feature_names_map = {
    "AMT_CREDIT": "Loan Amount Requested",
    "AMT_GOODS_PRICE": "Purchase Price",
    "EXT_SOURCE_1": "External Credit Score A",
    "EXT_SOURCE_2": "External Credit Score B",
    "EXT_SOURCE_3": "External Credit Score C",
    "DAYS_BIRTH": "Customer Age",
    "AMT_ANNUITY": "Monthly Loan Payment",
    "CODE_GENDER_M": "Applicant Gender",
    "FLAG_DOCUMENT_3": "Identity Verification",
}

# ------------------------------------------------
# FORMAT FEATURE VALUES
# ------------------------------------------------

def format_feature_value(feature,value):

    if feature in ["Loan Amount Requested","Purchase Price","Monthly Loan Payment"]:
        return f"₹{int(value):,}"

    if feature=="Customer Age":
        age=int(abs(value)/365)
        return f"{age} years"

    if "Credit Score" in feature:
        return f"{value:.2f}"

    if feature=="Identity Verification":
        return "Verified" if value>0.5 else "Not Verified"

    return f"{value:.2f}"

# ------------------------------------------------
# SIDEBAR
# ------------------------------------------------

st.sidebar.header("Business Policy Controls")

threshold = st.sidebar.slider(
    "Approval Threshold",
    0.1,
    0.9,
    0.45,
    0.05
)

st.sidebar.write("Approve loan if default probability < threshold")

# ------------------------------------------------
# TABS
# ------------------------------------------------

tab1,tab2,tab3 = st.tabs([
    "Customer Decision",
    "Portfolio Analysis",
    "Policy Simulator"
])

# ------------------------------------------------
# PDF REPORT
# ------------------------------------------------

def generate_pdf(income,credit,goods,prob,decision):

    buffer=io.BytesIO()

    doc=SimpleDocTemplate(buffer,pagesize=letter)

    styles=getSampleStyleSheet()

    elements=[]

    elements.append(Paragraph("Credit Risk Assessment Report",styles["Title"]))
    elements.append(Spacer(1,20))

    table_data=[
        ["Feature","Value"],
        ["Annual Income",income],
        ["Credit Amount",credit],
        ["Goods Price",goods],
        ["Default Probability",f"{prob:.2%}"],
        ["Decision",decision]
    ]

    table=Table(table_data)

    table.setStyle([
        ("BACKGROUND",(0,0),(-1,0),colors.grey),
        ("GRID",(0,0),(-1,-1),1,colors.black)
    ])

    elements.append(table)

    doc.build(elements)

    buffer.seek(0)

    return buffer

# ------------------------------------------------
# TAB 1 : SINGLE CUSTOMER
# ------------------------------------------------

with tab1:

    st.header("Customer Loan Application")

    c1,c2,c3=st.columns(3)

    with c1:
        income=st.number_input("Annual Income",0,10000000,500000)
        age=st.number_input("Age",18,80,30)

    with c2:
        credit=st.number_input("Credit Amount",0,10000000,200000)
        employment=st.number_input("Employment Years",0,40,5)

    with c3:
        goods_price=st.number_input("Goods Price",0,10000000,180000)
        children=st.number_input("Number of Children",0,10,0)

    run_model=st.button("Run Credit Risk Assessment")

    if run_model:

        template=pd.read_csv("data/processed/X_test.csv").mean().to_frame().T

        template["AMT_INCOME_TOTAL"]=income
        template["AMT_CREDIT"]=credit
        template["AMT_GOODS_PRICE"]=goods_price

        prob=model.predict_proba(template)[0][1]

        risk_score=int((1-prob)*100)

        st.subheader("Customer Risk Score")

        col1,col2=st.columns(2)

        with col1:

            fig=go.Figure(go.Indicator(
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

        with col2:

            st.metric("Default Probability",f"{prob:.2%}")
            st.metric("Credit Risk Score",f"{risk_score}/100")

            if prob<threshold:
                decision="Loan Approved"
                st.success(decision)
            else:
                decision="Loan Rejected"
                st.error(decision)

        # SHAP EXPLANATION

        st.subheader("Prediction Explanation")

        explainer=shap.TreeExplainer(model)

        shap_values=explainer(template)

        feature_names=[
            feature_names_map.get(col,col)
            for col in template.columns
        ]

        labels=[
            f"{format_feature_value(feature_names[i],template.iloc[0][i])} — {feature_names[i]}"
            for i in range(len(feature_names))
        ]

        fig,ax=plt.subplots()

        shap.plots.waterfall(
            shap.Explanation(
                values=shap_values.values[0],
                base_values=shap_values.base_values[0],
                data=template.iloc[0],
                feature_names=labels
            ),
            show=False
        )

        st.pyplot(fig)

        # TOP DRIVERS

        st.subheader("Key Risk Drivers")

        contributions=pd.Series(
            shap_values.values[0],
            index=feature_names
        )

        top=contributions.abs().sort_values(ascending=False).head(3)

        for f in top.index:

            impact=contributions[f]

            if impact>0:
                st.write(f"🔴 {f} increases risk")
            else:
                st.write(f"🟢 {f} lowers risk")

        pdf=generate_pdf(income,credit,goods_price,prob,decision)

        st.download_button(
            "Download Credit Report",
            pdf,
            file_name="credit_report.pdf"
        )

# ------------------------------------------------
# TAB 2 : PORTFOLIO ANALYZER
# ------------------------------------------------

with tab2:

    st.header("Customer Portfolio Risk Analyzer")

    file=st.file_uploader("Upload CSV",type=["csv"])

    if file:

        df=pd.read_csv(file)

        st.subheader("Uploaded Data")

        st.dataframe(df.head())

        template_cols=pd.read_csv("data/processed/X_test.csv").columns

        portfolio=pd.DataFrame(
            0,
            index=df.index,
            columns=template_cols
        )

        if "AMT_INCOME_TOTAL" in df.columns:
            portfolio["AMT_INCOME_TOTAL"]=df["AMT_INCOME_TOTAL"]

        if "AMT_CREDIT" in df.columns:
            portfolio["AMT_CREDIT"]=df["AMT_CREDIT"]

        if "AMT_GOODS_PRICE" in df.columns:
            portfolio["AMT_GOODS_PRICE"]=df["AMT_GOODS_PRICE"]

        probs=model.predict_proba(portfolio)[:,1]

        df["default_probability"]=probs

        df["decision"]=np.where(
            probs<threshold,
            "Approve",
            "Reject"
        )

        st.subheader("Prediction Results")

        st.dataframe(df)

        total=len(df)
        approved=(df["decision"]=="Approve").sum()
        rejected=(df["decision"]=="Reject").sum()

        c1,c2,c3=st.columns(3)

        c1.metric("Total Customers",total)
        c2.metric("Approved Loans",approved)
        c3.metric("Rejected Loans",rejected)

        # HISTOGRAM

        fig,ax=plt.subplots()

        ax.hist(df["default_probability"],bins=20)

        ax.set_xlabel("Default Probability")
        ax.set_ylabel("Customers")

        st.pyplot(fig)

        # SHAP ANALYSIS FOR PORTFOLIO

        st.subheader("Top Portfolio Risk Drivers")

        explainer=shap.TreeExplainer(model)

        shap_values=explainer(portfolio)

        shap_importance=np.abs(shap_values.values).mean(axis=0)

        importance=pd.Series(
            shap_importance,
            index=portfolio.columns
        ).sort_values(ascending=False).head(10)

        importance.index=[
            feature_names_map.get(i,i)
            for i in importance.index
        ]

        st.bar_chart(importance)

# ------------------------------------------------
# TAB 3 : POLICY SIMULATOR
# ------------------------------------------------

with tab3:

    st.header("Credit Policy Simulator")

    X_test=pd.read_csv("data/processed/X_test.csv")

    probs=model.predict_proba(X_test)[:,1]

    thresholds=np.arange(0.1,0.9,0.05)

    approval_rates=[]

    for t in thresholds:

        approval_rates.append((probs<t).mean())

    sim_df=pd.DataFrame({
        "threshold":thresholds,
        "approval_rate":approval_rates
    })

    st.line_chart(sim_df.set_index("threshold"))
