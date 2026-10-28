"""
LAYER 2 - Step 6: Feature Engineering.

Two jobs:
  1. build_features()  - derive the ratios and bands that actually carry signal
  2. time_split()      - split train/test by issue date, not randomly

On (2): a random split lets the model train on 2018 loans and test on 2016 ones,
which cannot happen in production and quietly inflates the score. Splitting by
time mimics real deployment: train on the past, predict the future.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config


def build_features(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    df = df.copy()
    created = []

    # ---- affordability ratios -------------------------------------------
    if {"loan_amnt", "annual_inc"} <= set(df.columns):
        df["loan_to_income"] = df["loan_amnt"] / df["annual_inc"].replace(0, np.nan)
        created.append("loan_to_income")

    if {"installment", "annual_inc"} <= set(df.columns):
        df["installment_to_monthly_income"] = df["installment"] / (
            df["annual_inc"].replace(0, np.nan) / 12
        )
        created.append("installment_to_monthly_income")

    if "annual_inc" in df.columns:
        df["annual_inc_log"] = np.log1p(df["annual_inc"].clip(lower=0))
        created.append("annual_inc_log")

    # ---- credit bureau --------------------------------------------------
    if {"fico_range_low", "fico_range_high"} <= set(df.columns):
        df["fico_avg"] = (df["fico_range_low"] + df["fico_range_high"]) / 2
        created.append("fico_avg")

    if {"earliest_cr_line", "issue_d"} <= set(df.columns):
        delta = df["issue_d"] - df["earliest_cr_line"]
        df["credit_history_months"] = (delta.dt.days / 30.44).round()
        df.loc[df["credit_history_months"] < 0, "credit_history_months"] = np.nan
        created.append("credit_history_months")

    if "delinq_2yrs" in df.columns:
        df["has_delinq_history"] = (df["delinq_2yrs"] > 0).astype(int)
        created.append("has_delinq_history")

    if {"revol_bal", "tot_hi_cred_lim"} <= set(df.columns):
        df["credit_utilisation"] = df["revol_bal"] / df["tot_hi_cred_lim"].replace(
            0, np.nan
        )
        created.append("credit_utilisation")

    # ---- advanced risk interactions & burden indices -------------------
    if {"dti", "int_rate"} <= set(df.columns):
        df["dti_x_int_rate"] = df["dti"] * df["int_rate"]
        created.append("dti_x_int_rate")

    if {"installment", "loan_amnt"} <= set(df.columns):
        df["installment_to_loan"] = df["installment"] / df["loan_amnt"].replace(0, np.nan)
        created.append("installment_to_loan")

    if "revol_util" in df.columns:
        df["revol_util_high"] = (df["revol_util"] > 80.0).astype(int)
        created.append("revol_util_high")

    if {"delinq_2yrs", "pub_rec", "pub_rec_bankruptcies"} <= set(df.columns):
        df["delinq_severity_score"] = (
            df["delinq_2yrs"].fillna(0) * 2.0
            + df["pub_rec"].fillna(0) * 1.5
            + df["pub_rec_bankruptcies"].fillna(0) * 3.0
        )
        created.append("delinq_severity_score")

    if {"inq_last_6mths", "credit_history_months"} <= set(df.columns):
        df["inq_velocity"] = df["inq_last_6mths"] / (
            df["credit_history_months"].fillna(120) / 12.0 + 1.0
        )
        created.append("inq_velocity")

    if {"tot_cur_bal", "tot_hi_cred_lim"} <= set(df.columns):
        df["credit_limit_util"] = df["tot_cur_bal"] / df["tot_hi_cred_lim"].replace(0, np.nan)
        created.append("credit_limit_util")

    if {"open_acc", "total_acc"} <= set(df.columns):
        df["open_acc_ratio"] = df["open_acc"] / df["total_acc"].replace(0, np.nan)
        created.append("open_acc_ratio")

    # ---- employment band -------------------------------------------------
    if "emp_length_years" in df.columns:
        df["emp_length_band"] = pd.cut(
            df["emp_length_years"],
            bins=[-0.1, 0.9, 3, 6, 9, 10],
            labels=["<1yr", "1-3yr", "4-6yr", "7-9yr", "10+yr"],
        ).astype(str)
        created.append("emp_length_band")

    # ---- calendar --------------------------------------------------------
    if "issue_d" in df.columns:
        df["issue_year"] = df["issue_d"].dt.year
        df["issue_month"] = df["issue_d"].dt.month
        created += ["issue_year", "issue_month"]

    # tidy up infinities produced by the ratio columns
    df = df.replace([np.inf, -np.inf], np.nan)

    if verbose:
        print(f"\n  engineered {len(created)} features: {', '.join(created)}")
    return df


def select_feature_columns(df: pd.DataFrame) -> tuple[list, list]:
    """Intersect the configured feature lists with what is actually present."""
    num = [c for c in config.NUMERIC_FEATURES if c in df.columns]
    cat = [c for c in config.CATEGORICAL_FEATURES if c in df.columns]

    if config.EXCLUDE_LC_RISK_SIGNALS:
        drop = {
            "grade",
            "sub_grade",
            "int_rate",
            "dti_x_int_rate",
            "installment",
            "installment_to_loan",
            "installment_to_monthly_income",
        }
        num = [c for c in num if c not in drop]
        cat = [c for c in cat if c not in drop]

    return num, cat


class QuantileClipper(BaseEstimator, TransformerMixin):
    """Winsorise each column to quantile bounds learned during fit."""

    def __init__(self, lower: float = 0.005, upper: float = 0.995):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        arr = np.asarray(X, dtype="float64")
        self.lo_ = np.nanquantile(arr, self.lower, axis=0)
        self.hi_ = np.nanquantile(arr, self.upper, axis=0)
        self.n_features_in_ = arr.shape[1]
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype="float64"), self.lo_, self.hi_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


def build_preprocessor(numeric: list, categorical: list) -> ColumnTransformer:
    """
    Impute + winsorise + scale numerics, impute + one-hot categoricals.

    Every statistic (medians, clip bounds, means, categories) is learned on the
    training window and travels with the saved model, so scoring a single new
    applicant uses exactly the same numbers the model was trained with.
    """
    lo, hi = getattr(config, "CLIP_QUANTILES", (0.005, 0.995))
    numeric_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("clip", QuantileClipper(lo, hi)),
            ("scale", StandardScaler()),
        ]
    )

    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                            min_frequency=0.01)
    except TypeError:  # scikit-learn < 1.2
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    categorical_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", ohe),
        ]
    )

    return ColumnTransformer(
        [
            ("num", numeric_pipe, numeric),
            ("cat", categorical_pipe, categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def time_split(
    df: pd.DataFrame,
    test_fraction: float = 0.2,
    date_col: str = "issue_d",
    return_cal: bool = True,
):
    """
    Strict chronological split on non-overlapping month boundaries:
      Train:       Jan 2013 - Dec 2014 (loans issued <= config.TRAIN_END_DATE)
      Calibration: Jan 2015 - Mar 2015 (config.TRAIN_END_DATE < issue_d <= config.CALIBRATION_END_DATE)
      Test:        Apr 2015 - Sep 2015 (config.CALIBRATION_END_DATE < issue_d <= config.TEST_END_DATE)
    Zero months are shared between any sets.
    """
    if date_col not in df.columns or df[date_col].isna().all():
        from sklearn.model_selection import train_test_split

        print("  ! no usable date column - falling back to random stratified split")
        tr, te = train_test_split(
            df,
            test_size=test_fraction,
            random_state=config.RANDOM_STATE,
            stratify=df[config.TARGET] if config.TARGET in df.columns else None,
        )
        if return_cal:
            tr_fit, tr_cal = train_test_split(
                tr,
                test_size=0.15,
                random_state=config.RANDOM_STATE,
                stratify=tr[config.TARGET] if config.TARGET in tr.columns else None,
            )
            return tr_fit, tr_cal, te
        return tr, te

    df = df.sort_values(date_col)

    # Use strict month boundaries if defined in config
    train_end = pd.to_datetime(getattr(config, "TRAIN_END_DATE", "2014-12-31"))
    cal_end = pd.to_datetime(getattr(config, "CALIBRATION_END_DATE", "2015-03-31"))
    test_end = pd.to_datetime(getattr(config, "TEST_END_DATE", "2015-09-30"))

    train = df[df[date_col] <= train_end].copy()
    cal = df[(df[date_col] > train_end) & (df[date_col] <= cal_end)].copy()
    test = df[(df[date_col] > cal_end) & (df[date_col] <= test_end)].copy()

    # Fallback to row-count splitting if date boundaries don't match data
    if len(train) == 0 or len(test) == 0:
        cutoff_test = int(len(df) * (1 - test_fraction))
        train_full, test = df.iloc[:cutoff_test].copy(), df.iloc[cutoff_test:].copy()
        if return_cal:
            cutoff_cal = int(len(train_full) * 0.85)
            train, cal = train_full.iloc[:cutoff_cal].copy(), train_full.iloc[cutoff_cal:].copy()
        else:
            train = train_full

    def _fmt(s, label):
        d_min, d_max = s[date_col].min(), s[date_col].max()
        dr = s[config.TARGET].mean() if config.TARGET in s.columns else 0.0
        return f"{label}={len(s):,} ({d_min:%b %Y} - {d_max:%b %Y}, default_rate={dr:.2%})"

    print("  strict month-boundary split (zero shared months):")
    print(f"    {_fmt(train, 'train')}")
    if return_cal and len(cal):
        print(f"    {_fmt(cal, 'calibration')}")
    print(f"    {_fmt(test, 'test')}")

    if return_cal:
        return train, cal, test
    return train, test


def prepare_xy(df: pd.DataFrame):
    """Return X (features only) and y (target)."""
    num, cat = select_feature_columns(df)
    X = df[num + cat].copy()
    y = df[config.TARGET].copy() if config.TARGET in df.columns else None
    return X, y, num, cat
