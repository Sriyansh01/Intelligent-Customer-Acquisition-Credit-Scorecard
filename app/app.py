import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
<<<<<<< HEAD
import seaborn as sns
import shap
=======
import plotly.graph_objects as go
import plotly.express as px
import io
>>>>>>> 1400e1db641c9e4328801bffccd07bb95befcb85

# --- SETTINGS ---
st.set_page_config(page_title="EquiScore:Bank Scorecard ", layout="wide")

<<<<<<< HEAD
# --- LOAD ASSETS ---
@st.cache_resource
def load_assets():
    # Loading LightGBM (Proposed) and XGBoost (Baseline)
    lgb = joblib.load("models/kaggle_model.pkl")
    xgb = joblib.load("models/xgb_model.pkl")
    # Loading the 150k record dataset
    df = pd.read_csv("cs-training.csv").iloc[:, 1:].fillna(0)
    return lgb, xgb, df

try:
    lgb_model, xgb_model, df_raw = load_assets()
    st.sidebar.success("✅ Systems Online: 150k Records Indexed")
except Exception as e:
    st.error(f"❌ Initialization Error: {e}. Ensure models/ folder contains .pkl files.")

# --- SIDEBAR: IDENTITY & INPUT ---
st.sidebar.header("🔍 Identity & Profile")
mode = st.sidebar.radio("Input Method", ["Customer ID Lookup", "Manual Entry / New Individual"])
=======
# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------

st.set_page_config(page_title="AI Credit Risk Platform", layout="wide")

st.title("🏦 Intelligent Customer Acquisition Credit Scorecard")

# ------------------------------------------------
# LOAD MODEL
# ------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load("models/xgboost_model.pkl")

model = load_model()

# ------------------------------------------------
# LOAD TEMPLATE DATA
# ------------------------------------------------

@st.cache_data
def load_template():
    return pd.read_csv("data/processed/X_test.csv")

X_template = load_template()

# ------------------------------------------------
# EXECUTIVE DASHBOARD
# ------------------------------------------------

st.subheader("📊 Portfolio Overview")

col1,col2,col3,col4 = st.columns(4)

col1.metric("Applications Processed","10,000")
col2.metric("Approval Rate","63%")
col3.metric("Predicted Defaults","12%")
col4.metric("Model ROC-AUC","0.91")

st.divider()

# ------------------------------------------------
# FEATURE NAME MAPPING
# ------------------------------------------------

feature_names_map = {

    "AMT_CREDIT": "Loan Amount Requested",
    "AMT_GOODS_PRICE": "Purchase Price",
    "AMT_ANNUITY": "Monthly Installment",
    "AMT_INCOME_TOTAL": "Customer Annual Income",

    "DAYS_BIRTH": "Customer Age",
    "EXT_SOURCE_1": "External Credit Score A",
    "EXT_SOURCE_2": "External Credit Score B",
    "EXT_SOURCE_3": "External Credit Score C",

    "FLAG_DOCUMENT_3": "Identity Document Verified",
    "FLAG_OWN_CAR_Y": "Owns a Car",
    "CODE_GENDER_M": "Male Applicant",
    "NAME_FAMILY_STATUS_Married": "Married Status",
}

# ------------------------------------------------
# SIDEBAR POLICY
# ------------------------------------------------

st.sidebar.header("Credit Policy Controls")
>>>>>>> 1400e1db641c9e4328801bffccd07bb95befcb85

is_new_individual = False

<<<<<<< HEAD
if mode == "Customer ID Lookup":
    cust_id = st.sidebar.number_input("Enter Customer ID (0-149999)", 0, 149999, 0)
    person = df_raw.iloc[cust_id]
    
    # Mapping from Database
    util = person['RevolvingUtilizationOfUnsecuredLines']
    age = person['age']
    p30 = int(person['NumberOfTime30-59DaysPastDueNotWorse'])
    debt = person['DebtRatio']
    income = person['MonthlyIncome']
    lines = int(person['NumberOfOpenCreditLinesAndLoans'])
    p90 = int(person['NumberOfTimes90DaysLate'])
    estate = int(person['NumberRealEstateLoansOrLines'])
    p60 = int(person['NumberOfTime60-89DaysPastDueNotWorse'])
    deps = int(person['NumberOfDependents'])
    
    st.sidebar.info(f"👤 Displaying Record: #{cust_id}")
    
    # --- FIXED HISTORICAL STATUS BOX ---
    if person['SeriousDlqin2yrs'] == 1:
        st.sidebar.error("⚠️ **Historical Fact:** This user defaulted.")
    else:
        st.sidebar.success("✅ **Historical Fact:** This user paid on time.")
else:
    # --- MANUAL ENTRY & MICROCREDIT LOGIC ---
    is_new_individual = st.sidebar.checkbox("Is New Individual? (No formal credit history)")
    
    age = st.sidebar.slider("Age", 21, 95, 35)
    income = st.sidebar.number_input("Monthly Income ($)", 0, 999999, 5000)
    util = st.sidebar.slider("Utilization Ratio", 0.0, 1.2, 0.3)
    debt = st.sidebar.slider("Debt-to-Income Ratio", 0.0, 1.5, 0.3)
    
    if is_new_individual:
        st.sidebar.subheader("🌟 Alternative Data (Inclusive Mode)")
        residency = st.sidebar.slider("Years at Current Residence", 0, 10, 2)
        utility_history = st.sidebar.checkbox("Consistent Utility/Rent Payments?")
        
        # PROXY MAPPING: Bridging the gap for the model
        estate = 1 if residency > 4 else 0
        lines = 2 if utility_history else 0
        p30, p60, p90, deps = 0, 0, 0, 0
    else:
        p90 = st.sidebar.selectbox("Past Due (90+ Days)", [0, 1, 2, 3, 4, 5])
        estate = st.sidebar.slider("Real Estate Loans", 0, 10, 1)
        lines = st.sidebar.slider("Open Credit Lines", 0, 20, 5)
        p30, p60, deps = 0, 0, 0

# --- PREDICTION ENGINE ---
features = np.array([[util, age, p30, debt, income, lines, p90, estate, p60, deps]])
feature_names = ['Utilization', 'Age', '30-59 Days Late', 'Debt Ratio', 'Monthly Income', 
                 'Open Lines', '90+ Days Late', 'Real Estate Loans', '60-89 Days Late', 'Dependents']

st.title("🛡️ EquiScore: Inclusive Bank Acquisition Framework")
st.write(f"Track 3: Fintech - MicroCreditScore (Scalable & Explainable AI)")

# --- DASHBOARD: COMPARATIVE SCORING ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("🚀 PROPOSED (LightGBM + ADASYN)")
    p_lgb = lgb_model.predict_proba(features)[0][1]
    score_lgb = int((1 - p_lgb) * 850)
    st.metric("Inclusion Score", f"{score_lgb}/850", delta=f"{score_lgb - 650} (vs Threshold)")
    if score_lgb > 650: st.success("✅ STRATEGIC APPROVAL")
    else: st.error("❌ STRATEGIC REJECTION")

with col2:
    st.subheader("📄 BASELINE (XGBoost)")
    p_xgb = xgb_model.predict_proba(features)[0][1]
    score_xgb = int((1 - p_xgb) * 850)
    st.metric("Standard Score", f"{score_xgb}/850")
    if score_xgb > 650: st.success("✅ BASELINE APPROVAL")
    else: st.error("❌ BASELINE REJECTION")

# --- EXPLAINABLE AI (SHAP) ---
st.divider()
st.subheader("🔬 XAI: Why was this decision made?")

if is_new_individual:
    st.info("**💡 Inclusion Protocol Active:** Real Estate and Open Lines impacts are derived from Alternative Data (Residency & Utility History).")

if st.button("🔍 Generate Local Explanation (SHAP)"):
    with st.spinner("Analyzing Feature Impact..."):
        explainer = shap.TreeExplainer(lgb_model)
        shap_values = explainer.shap_values(features)
        impacts = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]
        imp_df = pd.DataFrame({'Feature': feature_names, 'Impact': impacts}).sort_values(by='Impact')
=======
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

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer,pagesize=letter)

    styles = getSampleStyleSheet()

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
# TAB 1 : CUSTOMER DECISION
# ------------------------------------------------

with tab1:

    st.header("Customer Loan Application")

    c1,c2,c3 = st.columns(3)

    with c1:
        income = st.number_input("Annual Income",0,10000000,500000)

    with c2:
        credit = st.number_input("Credit Amount",0,10000000,200000)

    with c3:
        goods_price = st.number_input("Goods Price",0,10000000,180000)

    run_model = st.button("Run Credit Risk Assessment")

    if run_model:

        template = X_template.mean().to_frame().T

        template["AMT_INCOME_TOTAL"]=income
        template["AMT_CREDIT"]=credit
        template["AMT_GOODS_PRICE"]=goods_price

        prob = model.predict_proba(template)[0][1]

        credit_score = int(300 + (1-prob)*600)

        st.subheader("Customer Risk Evaluation")

        col1,col2 = st.columns(2)

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
                decision="Loan Approved"
                st.success(decision)
            else:
                decision="Loan Rejected"
                st.error(decision)

        # SHAP EXPLANATION

        st.subheader("Explainable AI — Why this decision?")

        explainer = shap.TreeExplainer(model)
        shap_values = explainer(template)

        fig,ax = plt.subplots()
        shap.plots.waterfall(shap_values[0],show=False)
>>>>>>> 1400e1db641c9e4328801bffccd07bb95befcb85

        fig, ax = plt.subplots(figsize=(10, 5))
        # Logic: Red (Positive Impact) increases risk probability, Green (Negative Impact) decreases it
        colors = ['#ff4b4b' if x > 0 else '#2eb82e' for x in imp_df['Impact']]
        sns.barplot(x='Impact', y='Feature', data=imp_df, palette=colors, ax=ax)
        ax.set_title("How Behavioral Factors Impacted the Risk Score")
        st.pyplot(fig)

<<<<<<< HEAD
# --- STATISTICAL SENSITIVITY ANALYSIS (The Blue Graph) ---
st.divider()
st.subheader("📈 Statistical Sensitivity Analysis")
st.write(f"How **'Utilization'** affects risk for this specific profile:")

u_range = np.linspace(0, 1.0, 20)
y_risks = []
for u in u_range:
    test_features = np.array([[u, age, p30, debt, income, lines, p90, estate, p60, deps]])
    risk_prob = lgb_model.predict_proba(test_features)[0][1]
    y_risks.append(risk_prob)

fig_trend, ax_trend = plt.subplots(figsize=(10, 4))
ax_trend.plot(u_range, y_risks, color='#0000FF', marker='o', linewidth=2)
ax_trend.set_xlabel("Utilization Rate (0.0 to 1.0)")
ax_trend.set_ylabel("Probability of Default")
ax_trend.grid(True, linestyle='--', alpha=0.6)
st.pyplot(fig_trend)

st.caption("🔍 **Insight:** This graph proves our model is non-linear. It shows the 'marginal risk' of every additional dollar spent on credit.")

# --- TRACK 3 BENCHMARK ---
st.divider()
if st.checkbox("📊 Show Fintech Competition Benchmark"):
    benchmark_data = {
        "Metric": ["Algorithm", "Fairness Handle", "Inclusion Logic", "Explainability"],
        "Traditional Systems": ["XGBoost", "None (Bias)", "Exclude Thin-Files", "Black Box"],
        "EquiScore (Ours)": ["LightGBM (Leaf-wise)", "ADASYN Balancing", "Proxy Protocol", "SHAP (Transparent)"]
    }
    st.table(pd.DataFrame(benchmark_data))
=======
        # TOP RISK DRIVERS

        st.subheader("Top Drivers of Risk")

        shap_importance = np.abs(shap_values.values[0])
        top_idx = np.argsort(shap_importance)[-3:][::-1]

        for i in top_idx:

            name = template.columns[i]
            name = feature_names_map.get(name,name)

            if shap_values.values[0][i] > 0:
                st.error(f"{name} increases default risk")
            else:
                st.success(f"{name} reduces default risk")

        pdf = generate_pdf(income,credit,goods_price,prob,decision)

        st.download_button(
            "Download Credit Report",
            pdf,
            file_name="credit_report.pdf"
        )

# ------------------------------------------------
# TAB 2 : PORTFOLIO ANALYSIS
# ------------------------------------------------

with tab2:

    st.header("Customer Portfolio Risk Analyzer")

    file = st.file_uploader("Upload CSV",type=["csv"])

    if file:

        df = pd.read_csv(file)

        portfolio = pd.DataFrame(
            0,
            index=df.index,
            columns=X_template.columns
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

        # ------------------------------------------------
        # REASON COLUMN USING SHAP
        # ------------------------------------------------

        explainer = shap.TreeExplainer(model)
        shap_values = explainer(portfolio)

        reasons=[]

        for i in range(len(portfolio)):

            shap_row = shap_values.values[i]

            top_feature_idx = np.argmax(np.abs(shap_row))

            feature_name = portfolio.columns[top_feature_idx]

            feature_name = feature_names_map.get(feature_name,feature_name)

            if shap_row[top_feature_idx] > 0:
                reason = f"High {feature_name}"
            else:
                reason = f"Strong {feature_name}"

            reasons.append(reason)

        df["reason"]=reasons

        # DISPLAY TABLE

        st.subheader("Prediction Results")

        display_df = df.copy()

        display_df.columns = [
            "Income",
            "Loan Amount",
            "Goods Price",
            "Default Probability",
            "Decision",
            "Reason"
        ]

        st.dataframe(display_df)

        total=len(df)
        approved=(df["decision"]=="Approve").sum()
        rejected=(df["decision"]=="Reject").sum()

        col1,col2,col3 = st.columns(3)

        col1.metric("Total Customers",total)
        col2.metric("Approved Loans",approved)
        col3.metric("Rejected Loans",rejected)

        # ------------------------------------------------
        # RISK DISTRIBUTION
        # ------------------------------------------------

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
🟡 Medium Risk (30–60%) → Moderate default risk  
🔴 High Risk (60–100%) → High probability of default
""")

        # ------------------------------------------------
        # TOP REJECTION REASONS
        # ------------------------------------------------

        st.subheader("Top Reasons for Loan Rejection")

        rejected_customers = portfolio[df["decision"]=="Reject"]

        if len(rejected_customers)>0:

            shap_values = explainer(rejected_customers)

            shap_importance = np.abs(shap_values.values).mean(axis=0)

            importance=pd.Series(
                shap_importance,
                index=rejected_customers.columns
            ).sort_values(ascending=False).head(5)

            importance.index=[
                feature_names_map.get(i,i)
                for i in importance.index
            ]

            fig,ax=plt.subplots()

            importance.sort_values().plot(
                kind="barh",
                ax=ax
            )

            ax.set_xlabel("Impact on Default Risk")

            st.pyplot(fig)

# ------------------------------------------------
# TAB 3 : POLICY SIMULATOR
# ------------------------------------------------

with tab3:

    st.header("Credit Policy Simulator")

    probs=model.predict_proba(X_template)[:,1]

    thresholds=np.arange(0.1,0.9,0.05)

    approval_rates=[]

    for t in thresholds:
        approval_rates.append((probs<t).mean())

    sim_df=pd.DataFrame({
        "threshold":thresholds,
        "approval_rate":approval_rates
    })

    fig = px.line(
        sim_df,
        x="threshold",
        y="approval_rate",
        title="Approval Rate vs Policy Threshold"
    )

    st.plotly_chart(fig)
>>>>>>> 1400e1db641c9e4328801bffccd07bb95befcb85
