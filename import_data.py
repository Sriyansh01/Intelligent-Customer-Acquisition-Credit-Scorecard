import pandas as pd

# Direct URL to the UCI German Credit Dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"

# Standard column names for this dataset
columns = [
    'checking_status', 'duration', 'credit_history', 'purpose', 'credit_amount',
    'savings_status', 'employment', 'installment_commitment', 'personal_status',
    'other_parties', 'residence_since', 'property_magnitude', 'age',
    'other_payment_plans', 'housing', 'existing_credits', 'job',
    'num_dependents', 'own_telephone', 'foreign_worker', 'class'
]

print("Fetching data from UCI Repository...")
# Note: 'sep' is a single space, and 'header=None' because raw data has no titles
df = pd.read_csv(url, sep=' ', header=None, names=columns)

# Convert class: 1 (Good) -> 0, 2 (Bad) -> 1 for standard binary modeling
df['class'] = df['class'] - 1

# Save to your app's data folder
import os
os.makedirs('app/data', exist_ok=True)
df.to_csv('app/data/german_credit_clean.csv', index=False)

print("✅ Success! Data imported and saved to 'app/data/german_credit_clean.csv'")
print(df.head())