import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.subplots as sp
import joblib

# 1. LOAD BOTH MODELS
# Make sure these filenames match exactly what is in your /models folder
lgb_model = joblib.load("models/german_model.pkl") 
xgb_model = joblib.load("models/baseline_xgb.pkl")

def plot_german_comparison():
    # Grid: Duration (months) vs Credit Amount (DM)
    duration_range = np.linspace(4, 72, 30)
    amount_range = np.linspace(250, 20000, 30)
    D, A = np.meshgrid(duration_range, amount_range)
    
    lgb_probs = []
    xgb_probs = []

    for d, a in zip(D.ravel(), A.ravel()):
        # Dummy profile for German Features: 
        # [Checking, Duration, History, Amount, Savings, Employment, Install_rate, Status, Debtors, Residence, Age, Plans, Housing, Credits, Job, Dependents, Phone, Foreign]
        # We hold most values at 'typical' safe levels (1 or 2)
        # Updated sample with 20 features to match your trained model
# Adding two extra 'placeholder' 1s at the end to hit the 20-count
        sample = np.array([[1, d, 2, a, 1, 3, 2, 1, 1, 2, 35, 1, 2, 1, 2, 1, 0, 1, 1, 1]])
        
        lgb_probs.append(lgb_model.predict_proba(sample)[0][1])
        xgb_probs.append(xgb_model.predict_proba(sample)[0][1])

    Z_lgb = np.array(lgb_probs).reshape(D.shape)
    Z_xgb = np.array(xgb_probs).reshape(D.shape)

    # Create Side-by-Side 3D Plot
    fig = sp.make_subplots(
        rows=1, cols=2, 
        specs=[[{'type': 'surface'}, {'type': 'surface'}]],
        subplot_titles=("2025 Baseline (XGBoost)", "Proposed Framework (LightGBM)")
    )

    # XGBoost Surface (Red/Yellow - Strict)
    fig.add_trace(go.Surface(z=Z_xgb, x=D, y=A, colorscale='Reds', showscale=False), row=1, col=1)
    
    # LightGBM Surface (Green/Blue - Inclusive)
    fig.add_trace(go.Surface(z=Z_lgb, x=D, y=A, colorscale='Viridis', showscale=False), row=1, col=2)

    fig.update_layout(
        title='Comparative 3D Risk Analysis: German Credit Dataset',
        height=600,
        scene=dict(xaxis_title='Duration', yaxis_title='Amount', zaxis_title='Risk'),
        scene2=dict(xaxis_title='Duration', yaxis_title='Amount', zaxis_title='Risk')
    )
    return fig

# 2. RUN APP
st.set_page_config(layout="wide")
st.title("🔬 Research Lab: Decision Boundary Comparison")
st.write("Comparing the rigid 2025 XGBoost baseline against our proposed LightGBM Framework.")

st.plotly_chart(plot_german_comparison(), use_container_width=True)