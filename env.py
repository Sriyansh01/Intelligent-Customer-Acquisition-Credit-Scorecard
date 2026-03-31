import lightgbm as lgb
import imblearn
import shap
import pandas as pd

print(f"LGBM Version: {lgb.__version__}")
print(f"ADASYN Ready: {imblearn.over_sampling.ADASYN}")
print("--- All Research Libraries Loaded Successfully ---")