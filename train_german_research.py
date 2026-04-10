import pandas as pd
from ucimlrepo import fetch_ucirepo 
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import joblib
import os

# Create folder if it doesn't exist
if not os.path.exists('models'):
    os.makedirs('models')

print("📥 Fetching German Credit Dataset...")
german_credit = fetch_ucirepo(id=144) 
X = german_credit.data.features 
y = german_credit.data.targets['class'] - 1 # Convert 1/2 to 0/1

# Basic Preprocessing (Convert categorical to codes for the 3D grid)
X_numeric = X.copy()
for col in X_numeric.select_dtypes(include=['object']).columns:
    X_numeric[col] = X_numeric[col].astype('category').cat.codes

print("🧠 Training Proposed LightGBM...")
lgb = LGBMClassifier(n_estimators=100, learning_rate=0.1, num_leaves=25)
lgb.fit(X_numeric, y)
joblib.dump(lgb, "models/german_model.pkl")

print("🧠 Training Baseline XGBoost...")
xgb = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=8)
xgb.fit(X_numeric, y)
joblib.dump(xgb, "models/baseline_xgb.pkl")

print("✅ SUCCESS! Models saved in /models folder.")