import streamlit as st
import numpy as np
import plotly.graph_objects as go
import joblib

# Load YOUR specific trained model
lgb_model = joblib.load("models/german_model.pkl")

def plot_my_model_surface():
    # We'll plot Duration vs. Credit Amount—the two biggest 'Stress' factors
    duration_range = np.linspace(4, 72, 40)
    amount_range = np.linspace(250, 20000, 40)
    D, A = np.meshgrid(duration_range, amount_range)
    
    probs = []
    for d, a in zip(D.ravel(), A.ravel()):
        # Fill in a typical profile for the other 18 features (using 1 or 2 as safe defaults)
        # [Checking, Duration, History, Amount, Savings, Employment, Install_rate, Status, Debtors, Residence, Age, Plans, Housing, Credits, Job, Dependents, Phone, Foreign, placeholder, placeholder]
        sample = np.array([[1, d, 2, a, 1, 3, 2, 1, 1, 2, 35, 1, 2, 1, 2, 1, 0, 1, 1, 1]])
        probs.append(lgb_model.predict_proba(sample)[0][1])

    Z = np.array(probs).reshape(D.shape)

    # Creating the Surface
    fig = go.Figure(data=[go.Surface(
        z=Z, x=D, y=A, 
        colorscale='Viridis', # 'Viridis' is a great 'Scientific' color map
        colorbar_title='Prob. of Default'
    )])

    fig.update_layout(
        title='EquiScore Decision Surface: Risk Probability Map',
        scene=dict(
            xaxis_title='Loan Duration (Months)',
            yaxis_title='Credit Amount (DM)',
            zaxis_title='Default Probability'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    return fig

st.title("🔬 EquiScore Reality Lab")
st.write("This 3D surface is generated directly from your trained LightGBM weights.")
st.plotly_chart(plot_my_model_surface(), use_container_width=True)