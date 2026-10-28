"""
Central configuration for the Credit Risk Scoring project.

Everything that a teammate might want to tweak lives here so that no one has to
go hunting through the pipeline modules.
"""

from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
ARTIFACT_DIR = ROOT / "artifacts"

DATA_DIR.mkdir(exist_ok=True)
ARTIFACT_DIR.mkdir(exist_ok=True)

# Point this at whatever you downloaded from Kaggle/mirror.
# Works with either .csv or .csv.gz or falls back to sample_loans.csv.
if (DATA_DIR / "accepted_2007_to_2018Q4.csv").exists():
    RAW_DATA_PATH = DATA_DIR / "accepted_2007_to_2018Q4.csv"
elif (DATA_DIR / "accepted_2007_to_2018Q4.csv.gz").exists():
    RAW_DATA_PATH = DATA_DIR / "accepted_2007_to_2018Q4.csv.gz"
else:
    RAW_DATA_PATH = DATA_DIR / "sample_loans.csv"

CLEAN_DATA_PATH = DATA_DIR / "clean_loans.parquet"
MODEL_PATH = ARTIFACT_DIR / "best_model.joblib"
METRICS_PATH = ARTIFACT_DIR / "model_metrics.json"
QUALITY_BEFORE_PATH = ARTIFACT_DIR / "quality_before.csv"
QUALITY_AFTER_PATH = ARTIFACT_DIR / "quality_after.csv"
CLEANING_LOG_PATH = ARTIFACT_DIR / "cleaning_log.json"
SCORED_SAMPLE_PATH = ARTIFACT_DIR / "scored_sample.csv"

# --------------------------------------------------------------------------
# Sampling — the full file is ~2.26M rows x 151 cols (~1.6 GB unzipped).
# --------------------------------------------------------------------------
# None = load everything. Set to e.g. 100_000 for fast debugging runs.
NROWS = None

# Keep only loans issued on/after this year. Older Lending Club vintages have
# very different underwriting, and many columns are empty before 2012.
MIN_ISSUE_YEAR = 2013

# --------------------------------------------------------------------------
# Maturity and Censoring Controls (Step 1)
# --------------------------------------------------------------------------
DATA_SNAPSHOT_DATE = "2018-12-01"
MATURITY_BUFFER_MONTHS = 3  # Maximum maturity cutoff: 2018-09-01
LOAN_TERM_MONTHS = 36       # Restrict pipeline strictly to 36-month loans for cross-split consistency

# Strict Chronological Split Boundaries (non-overlapping month boundaries)
# Active fully-matured window: 2013-01 to 2015-09 (33 months)
TRAIN_END_DATE = "2014-12-31"        # Jan 2013 - Dec 2014 (24 months)
CALIBRATION_END_DATE = "2015-03-31"  # Jan 2015 - Mar 2015 (3 months)
TEST_END_DATE = "2015-09-30"         # Apr 2015 - Sep 2015 (6 months)

RANDOM_STATE = 42

# --------------------------------------------------------------------------
# Target
# --------------------------------------------------------------------------
TARGET = "default_flag"
RAW_TARGET = "loan_status"

# 1 = the loan went bad. 0 = the borrower paid it off.
# Anything not in this map (Current, Late, In Grace Period) gets dropped,
# because the outcome is not yet known.
TARGET_MAP = {
    "Fully Paid": 0,
    "Charged Off": 1,
    "Default": 1,
    "Does not meet the credit policy. Status:Fully Paid": 0,
    "Does not meet the credit policy. Status:Charged Off": 1,
}

# --------------------------------------------------------------------------
# LEAKAGE — read this before you touch anything else.
# --------------------------------------------------------------------------
# These columns are only known AFTER the loan has been issued and has started
# repaying. If you train on them the model effectively gets to see the answer,
# your ROC-AUC jumps to ~0.99, and any examiner who knows the dataset will
# immediately ask why. Realistic AUC on this problem is 0.68 - 0.73.
LEAKAGE_COLS = [
    "total_pymnt",
    "total_pymnt_inv",
    "total_rec_prncp",
    "total_rec_int",
    "total_rec_late_fee",
    "recoveries",
    "collection_recovery_fee",
    "last_pymnt_d",
    "last_pymnt_amnt",
    "next_pymnt_d",
    "out_prncp",
    "out_prncp_inv",
    "last_credit_pull_d",
    "last_fico_range_high",
    "last_fico_range_low",
    "debt_settlement_flag",
    "debt_settlement_flag_date",
    "settlement_status",
    "settlement_date",
    "settlement_amount",
    "settlement_percentage",
    "settlement_term",
    "hardship_flag",
    "hardship_type",
    "hardship_reason",
    "hardship_status",
    "hardship_amount",
    "hardship_start_date",
    "hardship_end_date",
    "hardship_length",
    "hardship_dpd",
    "hardship_loan_status",
    "hardship_payoff_balance_amount",
    "hardship_last_payment_amount",
    "deferral_term",
    "payment_plan_start_date",
    "orig_projected_additional_accrued_interest",
    "pymnt_plan",
    "funded_amnt",       # equals loan_amnt once funded; not known at application
    "funded_amnt_inv",
]

# --------------------------------------------------------------------------
# Columns we actually read. Using usecols keeps memory sane.
# Every column here is knowable at the moment of application.
# --------------------------------------------------------------------------
USECOLS = [
    "id",
    "loan_amnt",
    "term",
    "int_rate",
    "installment",
    "grade",
    "sub_grade",
    "emp_length",
    "home_ownership",
    "annual_inc",
    "verification_status",
    "issue_d",
    "loan_status",
    "purpose",
    "addr_state",
    "dti",
    "delinq_2yrs",
    "earliest_cr_line",
    "fico_range_low",
    "fico_range_high",
    "inq_last_6mths",
    "mths_since_last_delinq",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
    "initial_list_status",
    "application_type",
    "mort_acc",
    "pub_rec_bankruptcies",
    "tot_cur_bal",
    "acc_open_past_24mths",
    "avg_cur_bal",
    "bc_open_to_buy",
    "bc_util",
    "num_actv_bc_tl",
    "num_tl_op_past_12m",
    "percent_bc_gt_75",
    "tot_hi_cred_lim",
    "total_bc_limit",
]

# --------------------------------------------------------------------------
# Model feature sets (populated after feature engineering)
# --------------------------------------------------------------------------
CATEGORICAL_FEATURES = [
    "term",
    "grade",
    "home_ownership",
    "verification_status",
    "purpose",
    "initial_list_status",
    "application_type",
    "emp_length_band",
]

NUMERIC_FEATURES = [
    "loan_amnt",
    "int_rate",
    "installment",
    "annual_inc_log",
    "dti",
    "delinq_2yrs",
    "fico_avg",
    "inq_last_6mths",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
    "mort_acc",
    "pub_rec_bankruptcies",
    "acc_open_past_24mths",
    "bc_util",
    "percent_bc_gt_75",
    "credit_history_months",
    "loan_to_income",
    "installment_to_monthly_income",
    "emp_length_years",
    "has_delinq_history",
    "credit_utilisation",
    "dti_x_int_rate",
    "installment_to_loan",
    "revol_util_high",
    "delinq_severity_score",
    "inq_velocity",
    "credit_limit_util",
    "open_acc_ratio",
]

# True = pure borrower underwriting. grade / sub_grade / int_rate / installment
# are Lending Club's own pricing of the loan, so using them partly means
# predicting another model's output. `--with-lc-signals` turns them back on.
EXCLUDE_LC_RISK_SIGNALS = True

# --------------------------------------------------------------------------
# Champion selection
# --------------------------------------------------------------------------
# The champion is the model with the best ROC-AUC on the CALIBRATION window,
# never the test window (picking on test would leak test information into a
# modelling decision). Only models in this list are eligible, because the
# scoring API must be able to produce exact per-applicant reason codes
# (TreeSHAP for tree models, coefficient x value for logistic regression).
CHAMPION_ELIGIBLE = ["XGBoost", "LightGBM", "CatBoost", "Random Forest", "Logistic Regression"]

# --------------------------------------------------------------------------
# Decision policy: predicted probability of default -> band -> decision
# --------------------------------------------------------------------------
# ONE source of truth. The pipeline, the API, the dashboard and the reports all
# read these. Low Risk = APPROVE, Medium Risk = REVIEW, High Risk = DECLINE.
RISK_THRESHOLDS = {"low_max": 0.10, "medium_max": 0.20}
RISK_LABELS = ["Low Risk", "Medium Risk", "High Risk"]
DECISIONS = {"Low Risk": "APPROVE", "Medium Risk": "REVIEW", "High Risk": "DECLINE"}

# Loss Given Default for unsecured consumer credit (Basel convention ~0.55-0.65)
LGD = 0.60

# Scorecard scaling: PD 20% (odds 4:1) -> 660, every 40 points halves the odds
SCORE_BASE = 660
SCORE_PDO = 40
SCORE_BASE_ODDS = 4.0

# Winsorisation bounds learned on the training window only (inside the model
# pipeline), so the scoring API never computes statistics from an upload batch.
CLIP_QUANTILES = (0.005, 0.995)

# --------------------------------------------------------------------------
# Data quality scoring weights (must sum to 1.0)
# --------------------------------------------------------------------------
QUALITY_WEIGHTS = {
    "completeness": 0.35,
    "validity": 0.25,
    "consistency": 0.20,
    "uniqueness": 0.20,
}

# Columns missing more than this fraction get dropped during cleaning.
MAX_MISSING_FRACTION = 0.50
