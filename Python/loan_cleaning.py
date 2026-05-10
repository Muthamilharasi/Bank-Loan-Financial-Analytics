import pandas as pd
import numpy as np

# ── Load Dataset ──────────────────────────────────────────────────
df = pd.read_csv('loan_data.csv')

print("Shape:", df.shape)
print("\nColumns:\n", df.columns.tolist())
print("\nMissing values:\n", df.isnull().sum())
print("\nData types:\n", df.dtypes)

# ── Step 1: Clean Column Names ───────────────────────────────────
df.columns = (
    df.columns
    .str.strip()
    .str.replace(' ', '_')
    .str.lower()
)

print("\nCleaned columns:\n", df.columns.tolist())

# ── Step 2: Handle Missing Values ────────────────────────────────

# Numeric columns → fill with median
num_cols = df.select_dtypes(include=[np.number]).columns

for col in num_cols:
    median_val = df[col].median()
    df[col] = df[col].fillna(median_val)

# Text columns → fill with 'Unknown'
cat_cols = df.select_dtypes(include=['object']).columns

for col in cat_cols:
    df[col] = df[col].fillna('Unknown')

# ── Step 3: Fix Data Types ───────────────────────────────────────

# Current Loan Amount cleaning
if 'current_loan_amount' in df.columns:

    df['current_loan_amount'] = (
        df['current_loan_amount']
        .astype(str)
        .str.replace('[^0-9.]', '', regex=True)
    )

    df['current_loan_amount'] = pd.to_numeric(
        df['current_loan_amount'],
        errors='coerce'
    ).fillna(0)

# Annual Income cleaning
if 'annual_income' in df.columns:

    df['annual_income'] = (
        df['annual_income']
        .astype(str)
        .str.replace('[^0-9.]', '', regex=True)
    )

    df['annual_income'] = pd.to_numeric(
        df['annual_income'],
        errors='coerce'
    ).fillna(0)

# Monthly Debt cleaning
if 'monthly_debt' in df.columns:

    df['monthly_debt'] = (
        df['monthly_debt']
        .astype(str)
        .str.replace('[^0-9.]', '', regex=True)
    )

    df['monthly_debt'] = pd.to_numeric(
        df['monthly_debt'],
        errors='coerce'
    ).fillna(0)

# Credit Score cleaning
if 'credit_score' in df.columns:

    df['credit_score'] = pd.to_numeric(
        df['credit_score'],
        errors='coerce'
    ).fillna(0)

# ── Step 4: Create Calculated Columns ────────────────────────────


# 2. Risk Category based on credit score
def risk_category(score):

    if score >= 750:
        return 'Low Risk'

    elif score >= 650:
        return 'Medium Risk'

    elif score >= 550:
        return 'High Risk'

    else:
        return 'Very High Risk'

if 'credit_score' in df.columns:
    df['risk_category'] = df['credit_score'].apply(risk_category)

# 3. Loan Size Category
def loan_size(amount):

    if amount < 5000:
        return 'Small'

    elif amount < 15000:
        return 'Medium'

    elif amount < 30000:
        return 'Large'

    else:
        return 'Very Large'

if 'current_loan_amount' in df.columns:
    df['loan_size_category'] = df['current_loan_amount'].apply(loan_size)

# 4. Income Band
def income_band(income):

    if income < 30000:
        return 'Low Income'

    elif income < 60000:
        return 'Middle Income'

    elif income < 100000:
        return 'Upper Middle'

    else:
        return 'High Income'

if 'annual_income' in df.columns:
    df['income_band'] = df['annual_income'].apply(income_band)

# 5. Debt-to-Income Ratio
if 'annual_income' in df.columns and 'monthly_debt' in df.columns:

    df['monthly_income'] = df['annual_income'] / 12

    df['dti_ratio'] = (
        df['monthly_debt'] /
        df['monthly_income'].replace(0, np.nan)
    ).fillna(0).round(2)

# 6. EMI Estimate
if 'current_loan_amount' in df.columns:

    df['emi_estimate'] = (
        df['current_loan_amount'] / 36
    ).round(2)

# ── Step 5: Remove Duplicates ────────────────────────────────────

before = len(df)

df.drop_duplicates(inplace=True)

after = len(df)

print(f"\nRemoved {before - after} duplicates")

# ── Step 6: Final Dataset Info ───────────────────────────────────

print("\nFinal Shape:", df.shape)

print("\nNew Columns Added:")

new_cols = [
    'default_flag',
    'risk_category',
    'loan_size_category',
    'income_band',
    'dti_ratio',
    'emi_estimate'
]

for col in new_cols:

    if col in df.columns:
        print("✔", col)

# ── Step 7: Save Cleaned File ────────────────────────────────────

df.to_csv('cleaned_loan_data.csv', index=False)

print("\n✅ Cleaned data saved successfully!")
print("File name: cleaned_loan_data.csv")