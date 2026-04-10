import streamlit as st
import numpy as np
import pandas as pd
import plotly.subplots as sp
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import ADASYN

# 1. SETUP & DATA LOADING
st.set_page_config(page_title="150k Industrial Validation", layout="wide")
st.title("🔬 Research Lab: 150k Industrial Dataset Validation")
st.write("Generating Confusion Matrices for **cs-training.csv**")

@st.cache_data
def load_and_train_150k():
    # Load the specific filename you provided
    df = pd.read_csv("cs-training.csv")
    
    # Kaggle dataset often has an 'Unnamed: 0' column as index
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)
    
    # DATA CLEANING: Essential for Kaggle 150k data
    # Filling missing Income with Median and Dependents with 0
    df['MonthlyIncome'] = df['MonthlyIncome'].fillna(df['MonthlyIncome'].median())
    df['NumberOfDependents'] = df['NumberOfDependents'].fillna(0)

    # Define features and target (SeriousDlqin2yrs is the default label)
    X = df.drop('SeriousDlqin2yrs', axis=1)
    y = df['SeriousDlqin2yrs']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Apply ADASYN to training data (The EquiScore Innovation)
    st.write("⏳ Balancing data with ADASYN... (This may take a minute for 150k rows)")
    ada = ADASYN(random_state=42)
    X_resampled, y_resampled = ada.fit_resample(X_train, y_train)

    # Train Proposed Model (LightGBM)
    lgb = LGBMClassifier(n_estimators=100, learning_rate=0.1, random_state=42, verbosity=-1)
    lgb.fit(X_resampled, y_resampled)

    # Train Baseline Model (XGBoost)
    xgb = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    xgb.fit(X_train, y_train)

    return lgb, xgb, X_test, y_test

# 2. GENERATE AND PLOT
try:
    lgb_model, xgb_model, X_test, y_test = load_and_train_150k()

    # Get Predictions
    lgb_preds = lgb_model.predict(X_test)
    xgb_preds = xgb_model.predict(X_test)

    # Generate Confusion Matrix Data
    cm_lgb = confusion_matrix(y_test, lgb_preds).ravel()
    cm_xgb = confusion_matrix(y_test, xgb_preds).ravel()

    categories = ['Non-Default (Good)', 'Default (Bad)']

    fig = sp.make_subplots(
        rows=1, cols=2, 
        subplot_titles=("2025 Baseline (XGBoost Standard)", "Proposed EquiScore (LGBM + ADASYN)"),
        horizontal_spacing = 0.15
    )

    def create_heatmap_trace(cm_data, color):
        z_values = [[cm_data[0], cm_data[1]], [cm_data[2], cm_data[3]]]
        return go.Heatmap(
            z=z_values, x=categories, y=categories,
            colorscale=color, showscale=False,
            text=[[f"TN: {cm_data[0]}", f"FP: {cm_data[1]}"], 
                  [f"FN: {cm_data[2]}", f"TP: {cm_data[3]}"]],
            texttemplate="%{text}", textfont={"size":14},
            ygap = 5, xgap = 5
        )

    fig.add_trace(create_heatmap_trace(cm_xgb, 'Reds'), row=1, col=1)
    fig.add_trace(create_heatmap_trace(cm_lgb, 'Blues'), row=1, col=2)

    fig.update_layout(height=550, template="plotly_dark", title_text="Industrial Confusion Matrix Comparison")
    st.plotly_chart(fig, use_container_width=True)

    st.success("🎯 Analysis Complete! Use the screenshot above for your Research Paper.")

except FileNotFoundError:
    st.error("Error: 'cs-training.csv' not found. Please ensure it is in the same folder as this script.")