import pandas as pd
from lightgbm import LGBMClassifier
from imblearn.over_sampling import ADASYN
import joblib
import os

# 1. Load data
df = pd.read_csv("app/data/german_credit_clean.csv")

# 2. Select specific features to match the UI
features = ['checking_status', 'duration', 'credit_history', 'credit_amount', 'savings_status', 'age']
X = df[features]
y = df['class']

# 3. Convert text to categories (LightGBM loves this)
for col in X.select_dtypes(include=['object']).columns:
    X[col] = X[col].astype('category')

# 4. ADASYN Balancing (To beat the 2025 paper)
# For sampling, we temporarily use codes
X_codes = X.copy()
for col in X_codes.select_dtypes(include=['category']).columns:
    X_codes[col] = X_codes[col].cat.codes

ada = ADASYN(random_state=42)
X_res, y_res = ada.fit_resample(X_codes, y)

# 5. Train and Save
model = LGBMClassifier(n_estimators=100, learning_rate=0.05)
model.fit(X_res, y_res)

os.makedirs('models', exist_ok=True)
joblib.dump(model, "models/xgboost_model.pkl")
print("🏆 Model synchronized with UI features!")