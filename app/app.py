import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import shap

# --- SETTINGS ---
st.set_page_config(page_title="EquiScore:Bank Scorecard ", layout="wide")

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

is_new_individual = False

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

        fig, ax = plt.subplots(figsize=(10, 5))
        # Logic: Red (Positive Impact) increases risk probability, Green (Negative Impact) decreases it
        colors = ['#ff4b4b' if x > 0 else '#2eb82e' for x in imp_df['Impact']]
        sns.barplot(x='Impact', y='Feature', data=imp_df, palette=colors, ax=ax)
        ax.set_title("How Behavioral Factors Impacted the Risk Score")
        st.pyplot(fig)

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
