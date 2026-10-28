"""
Script to generate test files for dashboard verification.
Generates 11 test CSVs in data/test_files/ plus answers_2016_100.csv.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
TEST_FILES_DIR = DATA_DIR / "test_files"
TEST_FILES_DIR.mkdir(parents=True, exist_ok=True)

# Load template columns
template_df = pd.read_csv(DATA_DIR / "template_applicants.csv")
TEMPLATE_COLS = list(template_df.columns)
print(f"Loaded {len(TEMPLATE_COLS)} template columns: {TEMPLATE_COLS}")

# Load sample loans source
sample_df = pd.read_csv(DATA_DIR / "sample_loans.csv")
sample_df["issue_year"] = sample_df["issue_d"].astype(str).str.extract(r"(\d{4})")
sample_df["term_num"] = sample_df["term"].astype(str).str.extract(r"(\d+)").astype(float)

# Set random seed for reproducibility
RNG = np.random.RandomState(42)

# ==============================================================================
# 1. fresh_2016_100.csv and answers_2016_100.csv
# ==============================================================================
# 100 random 36-month applicants issued in 2016 with known outcome
pool_2016_36m = sample_df[
    (sample_df["issue_year"] == "2016")
    & (sample_df["term_num"] == 36)
    & (sample_df["dti"] >= 0)
    & (sample_df["dti"] <= 100)
    & (sample_df["loan_status"].isin(["Fully Paid", "Charged Off", "Default"]))
].copy()

fresh_2016_100_full = pool_2016_36m.sample(n=100, random_state=42).copy().reset_index(drop=True)

# Save fresh_2016_100.csv (template columns only)
fresh_2016_100 = fresh_2016_100_full[TEMPLATE_COLS].copy()
fresh_2016_100.to_csv(TEST_FILES_DIR / "fresh_2016_100.csv", index=False)
print(f"Generated fresh_2016_100.csv ({len(fresh_2016_100)} rows, {len(TEMPLATE_COLS)} cols)")

# Save answers_2016_100.csv (DEMO ONLY - NOT EMPIRICAL EVIDENCE)
# Note: 36-month loans issued in 2016 had not fully matured by the Dec 2018 LendingClub snapshot
# (right-censored: active loans were still 'Current', only early resolutions are present).
# Furthermore, N=100 is too small for statistical significance. This file is provided solely
# as a mechanical demonstration for manual dashboard testing, not as validation evidence.
answers_2016_100 = fresh_2016_100.copy()
answers_2016_100["actual_loan_status"] = fresh_2016_100_full["loan_status"].values
answers_2016_100["actual_default"] = fresh_2016_100_full["loan_status"].apply(
    lambda s: 1 if s in ("Charged Off", "Default") else 0
).values
answers_2016_100["defaulted_text"] = answers_2016_100["actual_default"].apply(
    lambda d: "Defaulted" if d == 1 else "Paid Off"
).values
answers_2016_100.to_csv(TEST_FILES_DIR / "answers_2016_100.csv", index=False)
print(f"Generated answers_2016_100.csv (DEMO ONLY: {answers_2016_100['actual_default'].sum()}/100 early defaults, right-censored cohort)")

# ==============================================================================
# 2. fresh_2017_1000.csv
# ==============================================================================
# 1000 random applicants from 2017, including some 60-month loans
pool_2017 = sample_df[
    (sample_df["issue_year"] == "2017")
    & (sample_df["dti"] >= 0)
    & (sample_df["dti"] <= 100)
].copy()
fresh_2017_1000 = pool_2017.sample(n=1000, random_state=42)[TEMPLATE_COLS].copy().reset_index(drop=True)
fresh_2017_1000.to_csv(TEST_FILES_DIR / "fresh_2017_1000.csv", index=False)
n_60m_2017 = (fresh_2017_1000["term"].astype(str).str.contains("60")).sum()
print(f"Generated fresh_2017_1000.csv (1000 rows, {n_60m_2017} 60-month loans)")

# ==============================================================================
# 3. single_applicant.csv
# ==============================================================================
single_applicant = pool_2016_36m.sample(n=1, random_state=99)[TEMPLATE_COLS].copy().reset_index(drop=True)
single_applicant.to_csv(TEST_FILES_DIR / "single_applicant.csv", index=False)
print("Generated single_applicant.csv (1 row)")

# ==============================================================================
# 4. high_risk_50.csv
# ==============================================================================
# 50 applicants with low FICO (under 660) and high DTI (over 30)
# Find candidates with DTI > 30 and FICO <= 665
pool_high_risk = sample_df[(sample_df["dti"] > 30) & (sample_df["term_num"] == 36)].copy()
high_risk_50 = pool_high_risk.sample(n=50, random_state=42)[TEMPLATE_COLS].copy().reset_index(drop=True)

# Assign low FICO values (580 to 650) to strictly satisfy FICO under 660
ficos_low = RNG.choice([580, 600, 610, 620, 630, 640, 650], size=50)
high_risk_50["fico_range_low"] = ficos_low
high_risk_50["fico_range_high"] = ficos_low + 4
# Ensure DTI is strictly over 30
dtis = 30.5 + RNG.uniform(0.5, 9.5, size=50).round(1)
high_risk_50["dti"] = dtis
high_risk_50.to_csv(TEST_FILES_DIR / "high_risk_50.csv", index=False)
print(f"Generated high_risk_50.csv (FICO: {high_risk_50['fico_range_low'].min()}-{high_risk_50['fico_range_low'].max()}, DTI: {high_risk_50['dti'].min()}-{high_risk_50['dti'].max()})")

# ==============================================================================
# 5. prime_50.csv
# ==============================================================================
# 50 applicants with FICO over 760 and DTI under 10
pool_prime = sample_df[
    (sample_df["fico_range_low"] > 760)
    & (sample_df["dti"] > 0)
    & (sample_df["dti"] < 10)
    & (sample_df["term_num"] == 36)
].copy()
prime_50 = pool_prime.sample(n=50, random_state=42)[TEMPLATE_COLS].copy().reset_index(drop=True)
prime_50.to_csv(TEST_FILES_DIR / "prime_50.csv", index=False)
print(f"Generated prime_50.csv (FICO: {prime_50['fico_range_low'].min()}-{prime_50['fico_range_low'].max()}, DTI: {prime_50['dti'].min()}-{prime_50['dti'].max()})")

# ==============================================================================
# 6. missing_columns.csv
# ==============================================================================
# Valid rows (20 applicants) but with fico_range_low and annual_inc removed
base_20 = pool_2016_36m.sample(n=20, random_state=123)[TEMPLATE_COLS].copy().reset_index(drop=True)
missing_cols_df = base_20.drop(columns=["fico_range_low", "annual_inc"])
missing_cols_df.to_csv(TEST_FILES_DIR / "missing_columns.csv", index=False)
print(f"Generated missing_columns.csv ({len(missing_cols_df)} rows, {len(missing_cols_df.columns)} cols)")

# ==============================================================================
# 7. blank_values.csv
# ==============================================================================
# 20 applicants with some blank cells in optional columns
blank_values_df = pool_2016_36m.sample(n=20, random_state=456)[TEMPLATE_COLS].copy().reset_index(drop=True)
optional_cols = ["emp_length", "mort_acc", "pub_rec_bankruptcies", "bc_util", "percent_bc_gt_75", "tot_cur_bal", "tot_hi_cred_lim", "revol_util"]
# Randomly blank out ~25-30% of cells across optional columns
for col in optional_cols:
    mask = RNG.rand(len(blank_values_df)) < 0.30
    blank_values_df.loc[mask, col] = ""
blank_values_df.to_csv(TEST_FILES_DIR / "blank_values.csv", index=False)
print(f"Generated blank_values.csv (20 rows with blanks in {optional_cols})")

# ==============================================================================
# 8. bad_values.csv
# ==============================================================================
# text in number columns ("abc" in loan_amnt), negative income, DTI of 999, FICO of 1200
bad_values_df = pool_2016_36m.sample(n=4, random_state=789)[TEMPLATE_COLS].copy().reset_index(drop=True)
# Convert relevant columns to object type to support string assignment
bad_values_df["loan_amnt"] = bad_values_df["loan_amnt"].astype(object)
bad_values_df["annual_inc"] = bad_values_df["annual_inc"].astype(object)
bad_values_df["dti"] = bad_values_df["dti"].astype(object)
bad_values_df["fico_range_low"] = bad_values_df["fico_range_low"].astype(object)
bad_values_df["fico_range_high"] = bad_values_df["fico_range_high"].astype(object)

# Row 1: "abc" in loan_amnt
bad_values_df.at[0, "loan_amnt"] = "abc"
# Row 2: negative income (-50000)
bad_values_df.at[1, "annual_inc"] = -50000
# Row 3: DTI of 999
bad_values_df.at[2, "dti"] = 999
# Row 4: FICO of 1200
bad_values_df.at[3, "fico_range_low"] = 1200
bad_values_df.at[3, "fico_range_high"] = 1204

bad_values_df.to_csv(TEST_FILES_DIR / "bad_values.csv", index=False)
print("Generated bad_values.csv (4 rows with specific bad values)")

# ==============================================================================
# 8b. mixed_good_bad.csv (Point 1: Partial Acceptance verification)
# ==============================================================================
# 18 valid rows + 2 bad rows (Row 5 has 'corrupted_amnt' in loan_amnt; Row 12 has -45000 in annual_inc)
mixed_df = pool_2016_36m.sample(n=20, random_state=314)[TEMPLATE_COLS].copy().reset_index(drop=True)
mixed_df["loan_amnt"] = mixed_df["loan_amnt"].astype(object)
mixed_df["annual_inc"] = mixed_df["annual_inc"].astype(object)

# Inject 2 bad values into specific rows:
mixed_df.at[4, "loan_amnt"] = "corrupted_amnt"  # Row 5 (1-indexed)
mixed_df.at[11, "annual_inc"] = -45000          # Row 12 (1-indexed)

mixed_df.to_csv(TEST_FILES_DIR / "mixed_good_bad.csv", index=False)
print("Generated mixed_good_bad.csv (20 rows: 18 valid rows, 2 invalid rows)")

# ==============================================================================
# 9. all_60_month.csv
# ==============================================================================
# 10 applicants, all 60-month terms
pool_60m = sample_df[sample_df["term_num"] == 60].copy()
all_60_month = pool_60m.sample(n=10, random_state=42)[TEMPLATE_COLS].copy().reset_index(drop=True)
all_60_month["term"] = " 60 months"
all_60_month.to_csv(TEST_FILES_DIR / "all_60_month.csv", index=False)
print("Generated all_60_month.csv (10 rows, all 60-month term)")

# ==============================================================================
# 10. empty_file.csv
# ==============================================================================
# only the header row, no applicants
empty_file = pd.DataFrame(columns=TEMPLATE_COLS)
empty_file.to_csv(TEST_FILES_DIR / "empty_file.csv", index=False)
print(f"Generated empty_file.csv (header only, 0 rows)")

# ==============================================================================
# 11. wrong_dataset.csv
# ==============================================================================
# 20 rows with completely different columns (e.g. age, credit_amount, duration, purpose) like German Credit
german_cols = [
    "age",
    "sex",
    "job",
    "housing",
    "saving_accounts",
    "checking_account",
    "credit_amount",
    "duration_months",
    "purpose",
]
german_data = [
    [67, "male", 2, "own", "little", "little", 1169, 6, "radio/TV"],
    [22, "female", 2, "own", "little", "moderate", 5951, 48, "radio/TV"],
    [49, "male", 1, "own", "little", "none", 2096, 12, "education"],
    [45, "male", 2, "free", "little", "little", 7882, 42, "furniture/equipment"],
    [53, "male", 2, "free", "little", "little", 4870, 24, "car"],
    [35, "male", 1, "free", "none", "none", 9055, 36, "education"],
    [53, "male", 2, "own", "quite rich", "none", 2835, 24, "furniture/equipment"],
    [35, "male", 3, "rent", "little", "moderate", 6948, 36, "car"],
    [61, "male", 1, "own", "rich", "none", 3059, 12, "radio/TV"],
    [28, "male", 3, "own", "little", "moderate", 5234, 30, "car"],
    [25, "female", 2, "rent", "little", "moderate", 1295, 12, "car"],
    [24, "female", 2, "rent", "little", "none", 4308, 48, "business"],
    [22, "female", 2, "own", "little", "moderate", 1567, 12, "radio/TV"],
    [60, "male", 1, "own", "little", "little", 1199, 24, "car"],
    [28, "female", 2, "rent", "little", "little", 1403, 12, "car"],
    [32, "female", 1, "own", "moderate", "little", 1282, 15, "radio/TV"],
    [53, "male", 2, "own", "none", "none", 2424, 24, "radio/TV"],
    [35, "male", 2, "own", "little", "little", 8072, 30, "car"],
    [44, "female", 3, "free", "little", "moderate", 12579, 24, "car"],
    [31, "male", 2, "own", "quite rich", "none", 3430, 24, "radio/TV"],
]
wrong_dataset = pd.DataFrame(german_data, columns=german_cols)
wrong_dataset.to_csv(TEST_FILES_DIR / "wrong_dataset.csv", index=False)
print("Generated wrong_dataset.csv (20 rows, German Credit format)")

print("\n--- Summary of Generated Files ---")
for f in sorted(TEST_FILES_DIR.glob("*.csv")):
    size_kb = f.stat().st_size / 1024
    lines = len(f.read_text(encoding="utf-8").splitlines())
    print(f"  {f.name:25s}: {lines:4d} lines, {size_kb:6.2f} KB")
