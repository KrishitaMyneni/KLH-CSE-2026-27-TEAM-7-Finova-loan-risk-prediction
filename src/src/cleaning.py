"""
LAYER 1 - Step 3: Data Cleaning.

Every transformation appends to a log so we can report exactly what was
done and why.

Order matters here:
  1. drop leakage columns        (before anything else, so they never influence a decision)
  2. define the target           (and drop loans whose outcome is unknown)
  3. parse messy string columns  (term, int_rate, revol_util, emp_length, dates)
  4. drop high-missing columns
  5. remove duplicates
  6. null values that are impossible (negative DTI, zero income)
  7. fill missing categoricals with 'Unknown'

Only row-independent rules live here. Anything that needs a statistic
(winsorisation quantiles, median imputation) is learned inside the model
pipeline on the training window, see features.build_preprocessor. Computing
those statistics here would leak test-period data into training, and at
scoring time would make an applicant's score depend on who else was in the
uploaded file.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config


class CleaningLog:
    def __init__(self, verbose: bool = True):
        self.actions = []
        self.verbose = verbose

    def add(self, step, detail, rows_before=None, rows_after=None, n_affected=None):
        entry = {"step": step, "detail": detail}
        if rows_before is not None:
            entry["rows_before"] = int(rows_before)
            entry["rows_after"] = int(rows_after)
            entry["rows_removed"] = int(rows_before - rows_after)
        if n_affected is not None:
            entry["values_affected"] = int(n_affected)
        self.actions.append(entry)
        msg = f"  [{step}] {detail}"
        if n_affected is not None:
            msg += f" ({n_affected:,} values)"
        if rows_before is not None:
            msg += f" ({rows_before:,} -> {rows_after:,} rows)"
        if self.verbose:
            print(msg)

    def to_list(self):
        return self.actions


# --------------------------------------------------------------------------
# Individual parsers
# --------------------------------------------------------------------------
def parse_term(s: pd.Series) -> pd.Series:
    """' 36 months' -> 36"""
    return pd.to_numeric(
        s.astype(str).str.extract(r"(\d+)", expand=False), errors="coerce"
    )


def parse_percent(s: pd.Series) -> pd.Series:
    """'13.56%' or ' 13.56' -> 13.56"""
    if pd.api.types.is_numeric_dtype(s):
        return s
    return pd.to_numeric(
        s.astype(str).str.replace("%", "", regex=False).str.strip(), errors="coerce"
    )


def parse_emp_length(s: pd.Series) -> pd.Series:
    """'< 1 year' -> 0, '10+ years' -> 10, 'n/a' -> NaN"""
    txt = s.astype(str).str.strip().str.lower()
    out = pd.Series(np.nan, index=s.index, dtype="float64")
    out[txt.str.startswith("< 1")] = 0.0
    out[txt.str.startswith("10+")] = 10.0
    mid = txt.str.extract(r"^(\d+)\s*year", expand=False)
    out = out.fillna(pd.to_numeric(mid, errors="coerce"))
    return out


def parse_lc_date(s: pd.Series) -> pd.Series:
    """'Dec-2015' -> Timestamp. Falls back to generic parsing."""
    out = pd.to_datetime(s, format="%b-%Y", errors="coerce")
    if out.isna().mean() > 0.5:
        out = pd.to_datetime(s, errors="coerce")
    return out


# --------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------
def clean(df: pd.DataFrame, verbose: bool = True, is_training: bool = True) -> tuple[pd.DataFrame, list]:
    log = CleaningLog(verbose)
    df = df.copy()
    if verbose:
        print("\n" + "=" * 62)
        print("CLEANING")
        print("=" * 62)

    # ---- 1. leakage -----------------------------------------------------
    present_leaks = [c for c in config.LEAKAGE_COLS if c in df.columns]
    if present_leaks:
        df = df.drop(columns=present_leaks)
        log.add(
            "leakage",
            f"dropped {len(present_leaks)} post-origination columns: "
            f"{', '.join(present_leaks[:6])}"
            + ("..." if len(present_leaks) > 6 else ""),
        )

    # ---- 2. target (training only) --------------------------------------
    if is_training and config.RAW_TARGET in df.columns:
        before = len(df)
        df[config.TARGET] = df[config.RAW_TARGET].map(config.TARGET_MAP)
        kept_statuses = df[config.RAW_TARGET].isin(config.TARGET_MAP)
        dropped_statuses = (
            df.loc[~kept_statuses, config.RAW_TARGET].value_counts().to_dict()
        )
        df = df[df[config.TARGET].notna()].copy()
        df[config.TARGET] = df[config.TARGET].astype(int)
        df = df.drop(columns=[config.RAW_TARGET])
        log.add(
            "target",
            f"kept resolved loans only; dropped unresolved statuses "
            f"{list(dropped_statuses)[:4]}",
            before,
            len(df),
        )
        rate = df[config.TARGET].mean()
        log.add("target", f"default rate = {rate:.2%}")

    # ---- 3. parse string columns ---------------------------------------
    if "term" in df.columns:
        df["term"] = parse_term(df["term"])
        log.add("parse", "term -> numeric months")

    for col in ["int_rate", "revol_util", "bc_util", "percent_bc_gt_75"]:
        if col in df.columns:
            df[col] = parse_percent(df[col])
            log.add("parse", f"{col} -> numeric", n_affected=int(df[col].notna().sum()))

    if "emp_length" in df.columns:
        df["emp_length_years"] = parse_emp_length(df["emp_length"])
        df = df.drop(columns=["emp_length"])
        log.add("parse", "emp_length -> emp_length_years (0-10)")

    for col in ["issue_d", "earliest_cr_line"]:
        if col in df.columns:
            df[col] = parse_lc_date(df[col])
            log.add("parse", f"{col} -> datetime")

    # ---- 3b. vintage filter (training only) ----------------------------
    if is_training and "issue_d" in df.columns and config.MIN_ISSUE_YEAR:
        before = len(df)
        df = df[df["issue_d"].dt.year >= config.MIN_ISSUE_YEAR]
        log.add(
            "filter",
            f"kept loans issued from {config.MIN_ISSUE_YEAR} onwards",
            before,
            len(df),
        )

    # ---- 3b2. term filter (cohort consistency, training only) ----------
    if is_training and "term" in df.columns and getattr(config, "LOAN_TERM_MONTHS", None):
        before = len(df)
        df = df[df["term"] == config.LOAN_TERM_MONTHS].copy()
        log.add(
            "filter",
            f"kept {config.LOAN_TERM_MONTHS}-month term loans only",
            before,
            len(df),
        )

    # ---- 3c. maturity filter (censor elimination, training only) -------
    if is_training and {"issue_d", "term"} <= set(df.columns) and getattr(config, "DATA_SNAPSHOT_DATE", None):
        before = len(df)
        snapshot_dt = pd.to_datetime(config.DATA_SNAPSHOT_DATE)
        buffer_months = getattr(config, "MATURITY_BUFFER_MONTHS", 3)
        max_maturity_dt = snapshot_dt - pd.DateOffset(months=buffer_months)

        # term is parsed to numeric months (36 or 60)
        maturity_d = pd.Series(pd.NaT, index=df.index)
        for t in df["term"].dropna().unique():
            mask = df["term"] == t
            maturity_d[mask] = df.loc[mask, "issue_d"] + pd.DateOffset(months=int(t))

        df = df[maturity_d <= max_maturity_dt].copy()
        log.add(
            "maturity_filter",
            f"kept loans matured by {max_maturity_dt:%Y-%m-%d} "
            f"(snapshot {config.DATA_SNAPSHOT_DATE} - {buffer_months}m buffer)",
            before,
            len(df),
        )

        if config.TARGET in df.columns:
            df["_year"] = df["issue_d"].dt.year
            cohort = df.groupby(["_year", "term"]).agg(
                n_loans=(config.TARGET, "count"),
                defaults=(config.TARGET, "sum"),
                default_rate=(config.TARGET, "mean"),
            )
            print("\n  [maturity_filter] Default rate by issue year and term (matured only):")
            for idx, r in cohort.iterrows():
                print(f"    Year {idx[0]} ({int(idx[1])}m): {int(r['n_loans']):,} loans, default rate = {r['default_rate']:.2%}")
            df.drop(columns=["_year"], inplace=True)

    # ---- 4. drop high-missing columns (training only) -------------------
    if is_training:
        miss = df.isna().mean()
        to_drop = [
            c
            for c in miss[miss > config.MAX_MISSING_FRACTION].index
            if c not in (config.TARGET,)
        ]
        if to_drop:
            df = df.drop(columns=to_drop)
            log.add(
                "missing",
                f"dropped {len(to_drop)} columns over "
                f"{config.MAX_MISSING_FRACTION:.0%} missing: {', '.join(to_drop[:6])}",
            )

    # ---- 5. duplicates (training only) ----------------------------------
    if is_training:
        before = len(df)
        df = df.drop_duplicates()
        log.add("duplicates", "removed exact duplicate rows", before, len(df))

        try:
            from src.quality import BUSINESS_KEY
        except ImportError:
            from quality import BUSINESS_KEY

        key = [c for c in BUSINESS_KEY if c in df.columns]
        if key:
            before = len(df)
            df = df.drop_duplicates(subset=key, keep="first")
            log.add(
                "duplicates",
                f"removed near-duplicates on business key ({len(key)} cols)",
                before,
                len(df),
            )

        if "id" in df.columns:
            before = len(df)
            df = df.drop_duplicates(subset=["id"], keep="first")
            log.add("duplicates", "removed duplicate loan ids", before, len(df))

    # ---- 6. impossible values -------------------------------------------
    # Negative or absurd DTI values exist in the raw file.
    if "dti" in df.columns:
        bad = (df["dti"] < 0) | (df["dti"] > 100)
        if bad.any():
            df.loc[bad, "dti"] = np.nan
            log.add("outliers", "nulled invalid dti values", n_affected=int(bad.sum()))

    if "annual_inc" in df.columns:
        zero_inc = df["annual_inc"] <= 0
        if zero_inc.any():
            df.loc[zero_inc, "annual_inc"] = np.nan
            log.add(
                "outliers",
                "nulled non-positive annual income",
                n_affected=int(zero_inc.sum()),
            )

    # ---- 7. categorical gaps ---------------------------------------------
    # Numeric gaps are left as NaN: the model pipeline imputes them with
    # training-window medians.
    num_cols = df.select_dtypes(include=[np.number]).columns.difference([config.TARGET])
    n_num_missing = int(df[num_cols].isna().sum().sum())
    if n_num_missing:
        log.add("impute", "numeric gaps left for the pipeline's training-median imputer",
                n_affected=n_num_missing)

    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    n_cat_missing = int(df[cat_cols].isna().sum().sum())
    if n_cat_missing:
        for c in cat_cols:
            df[c] = df[c].fillna("Unknown")
        log.add("impute", "categorical columns -> 'Unknown'", n_affected=n_cat_missing)

    if verbose:
        print(f"\n  final shape: {df.shape[0]:,} rows x {df.shape[1]} cols")
        if config.TARGET in df.columns:
            print(f"  class balance: {df[config.TARGET].value_counts().to_dict()}")

    return df, log.to_list()


if __name__ == "__main__":
    from data_loader import load_raw

    raw = load_raw()
    cleaned, actions = clean(raw)
    print(cleaned.head())
