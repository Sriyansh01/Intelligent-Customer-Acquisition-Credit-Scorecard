import pandas as pd
import joblib
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import ADASYN
from sklearn.model_selection import train_test_split

# 1. Load 150,000 Bank Records
print("📂 Loading 150,000 rows...")
df = pd.read_csv("cs-training.csv").iloc[:, 1:]

# 2. Bank-Level Cleaning (Impute Medians)
df = df.fillna(df.median())

X = df.drop('SeriousDlqin2yrs', axis=1)
y = df['SeriousDlqin2yrs']

# 3. Apply ADASYN (The Intelligence Factor)
print("⚖️ Balancing the dataset with ADASYN...")
X_resampled, y_resampled = ADASYN(random_state=42).fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2)

# 4. Train Proposed Model (LightGBM)
print("🚀 Training Proposed Model (LightGBM)...")
lgb = LGBMClassifier(n_estimators=100, learning_rate=0.05, verbose=-1)
lgb.fit(X_train, y_train)
joblib.dump(lgb, "models/kaggle_model.pkl")

# 5. Train 2025 Baseline Model (XGBoost)
print("📄 Training 2025 Baseline (XGBoost)...")
xgb = XGBClassifier(n_estimators=100, learning_rate=0.05)
xgb.fit(X_train, y_train)
joblib.dump(xgb, "models/xgb_model.pkl")

print("✅ SUCCESS! Both 'brains' are saved in the models folder.")