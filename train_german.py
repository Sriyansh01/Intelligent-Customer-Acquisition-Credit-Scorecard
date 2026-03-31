import pandas as pd
from lightgbm import LGBMClassifier
from imblearn.over_sampling import ADASYN
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# 1. Load data
df = pd.read_csv("app/data/german_credit_clean.csv")

# 2. SELECT ONLY THE 6 FEATURES USED IN THE APP
# These must match the keys in your german_app.py dictionary exactly
selected_features = ['checking_status', 'duration', 'credit_history', 'credit_amount', 'savings_status', 'age']

X = df[selected_features].copy()
y = df['class']

# 3. Encoding only the categorical columns in our subset
le = LabelEncoder()
cat_cols = ['checking_status', 'credit_history', 'savings_status']
for col in cat_cols:
    X[col] = le.fit_transform(X[col].astype(str))

# 4. Apply ADASYN (The Research Upgrade)
ada = ADASYN(random_state=42)
X_resampled, y_resampled = ada.fit_resample(X, y)

# 5. Train LightGBM
model = LGBMClassifier(
    n_estimators=100,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42,
    verbose=-1 # Keeps the console clean
)
model.fit(X_resampled, y_resampled)

# 6. Save
os.makedirs('models', exist_ok=True)
# Note: In your app you called it xgboost_model.pkl but used LightGBM. 
# Let's keep the name for now so your app finds it.
joblib.dump(model, "models/xgboost_model.pkl") 

print("✅ SUCCESS: Model trained on 6 features and saved!")