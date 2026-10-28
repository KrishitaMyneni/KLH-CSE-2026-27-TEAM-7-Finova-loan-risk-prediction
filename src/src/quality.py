"""
LAYER 1 - Step 4 & 5: Data Quality Assessment and Duplicate Detection.

Produces a per-column quality profile and an overall 0-100 score built from four
dimensions, which is exactly what the "Measure completeness, consistency,
validity and duplication" objective asks for.

  completeness  - what fraction of values are present
  validity      - what fraction of present values satisfy a domain rule
  consistency   - do cross-field relationships hold (dataset-level)
  uniqueness    - fraction of rows that are not duplicates (dataset-level)

Run it before cleaning and again after, and show both in the report. The
before/after delta is the single most convincing slide in the review.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config

# --------------------------------------------------------------------------
# Validity rules: column -> callable returning a boolean Series (True = valid)
# Applied only to non-null values.
# --------------------------------------------------------------------------
VALIDITY_RULES = {
    "loan_amnt": lambda s: (s >= 500) & (s <= 40_000),
    "annual_inc": lambda s: (s >= 0) & (s <= 2_000_000),
    "dti": lambda s: (s >= 0) & (s <= 100),
    "int_rate": lambda s: (s > 0) & (s <= 40),
    "revol_util": lambda s: (s >= 0) & (s <= 200),
    "fico_range_low": lambda s: (s >= 300) & (s <= 850),
    "fico_range_high": lambda s: (s >= 300) & (s <= 850),
    "open_acc": lambda s: (s >= 0) & (s <= 100),
    "total_acc": lambda s: (s >= 0) & (s <= 200),
    "delinq_2yrs": lambda s: (s >= 0) & (s <= 50),
    "pub_rec": lambda s: (s >= 0) & (s <= 50),
    "installment": lambda s: (s > 0) & (s <= 2_000),
    "bc_util": lambda s: (s >= 0) & (s <= 200),
    "percent_bc_gt_75": lambda s: (s >= 0) & (s <= 100),
    "term": lambda s: s.astype(str).str.contains("36|60", na=False),
    "home_ownership": lambda s: s.isin(
        ["RENT", "OWN", "MORTGAGE", "OTHER", "NONE", "ANY"]
    ),
    "grade": lambda s: s.isin(list("ABCDEFG")),
    "emp_length": lambda s: s.astype(str).str.contains(
        r"year|Unknown|n/a", case=False, na=False
    ),
}

# --------------------------------------------------------------------------
# Consistency rules: name -> callable(df) returning boolean Series (True = OK)
# --------------------------------------------------------------------------
def _rule_fico_order(df):
    if {"fico_range_low", "fico_range_high"} <= set(df.columns):
        return df["fico_range_low"] <= df["fico_range_high"]
    return None


def _rule_open_le_total(df):
    if {"open_acc", "total_acc"} <= set(df.columns):
        return df["open_acc"] <= df["total_acc"]
    return None


def _rule_installment_sane(df):
    """Monthly installment should not exceed the whole loan amount."""
    if {"installment", "loan_amnt"} <= set(df.columns):
        return df["installment"] <= df["loan_amnt"]
    return None


def _rule_credit_before_issue(df):
    """Earliest credit line must predate the loan issue date."""
    if {"earliest_cr_line", "issue_d"} <= set(df.columns):
        cr = pd.to_datetime(df["earliest_cr_line"], format="%b-%Y", errors="coerce")
        iss = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce")
        ok = cr <= iss
        return ok.where(cr.notna() & iss.notna(), True)
    return None


def _rule_income_supports_installment(df):
    """Annual installments should not exceed annual income."""
    if {"installment", "annual_inc"} <= set(df.columns):
        inc = pd.to_numeric(df["annual_inc"], errors="coerce")
        inst = pd.to_numeric(df["installment"], errors="coerce")
        ok = (inst * 12) <= inc
        return ok.where(inc.notna() & inst.notna() & (inc > 0), True)
    return None


CONSISTENCY_RULES = {
    "fico_low <= fico_high": _rule_fico_order,
    "open_acc <= total_acc": _rule_open_le_total,
    "installment <= loan_amnt": _rule_installment_sane,
    "earliest_cr_line <= issue_d": _rule_credit_before_issue,
    "annual installments <= income": _rule_income_supports_installment,
}

# Columns that together identify one real application. Two rows matching on all
# of these are almost certainly the same loan entered twice.
BUSINESS_KEY = [
    "loan_amnt",
    "term",
    "int_rate",
    "installment",
    "grade",
    "annual_inc",
    "issue_d",
    "addr_state",
    "purpose",
]


# --------------------------------------------------------------------------
# Column profiling
# --------------------------------------------------------------------------
def column_profile(df: pd.DataFrame) -> pd.DataFrame:
    """One row per column: type, missingness, cardinality, validity."""
    rows = []
    n = len(df)

    for col in df.columns:
        s = df[col]
        n_missing = int(s.isna().sum())
        completeness = 1 - (n_missing / n) if n else 0.0

        # validity
        rule = VALIDITY_RULES.get(col)
        if rule is not None and n_missing < n:
            present = s.dropna()
            try:
                valid_mask = rule(present)
                validity = float(valid_mask.mean())
                n_invalid = int((~valid_mask).sum())
            except Exception:
                validity, n_invalid = np.nan, 0
        else:
            validity, n_invalid = np.nan, 0

        rec = {
            "column": col,
            "dtype": str(s.dtype),
            "n_missing": n_missing,
            "pct_missing": round(100 * n_missing / n, 2) if n else 0.0,
            "completeness": round(completeness, 4),
            "n_unique": int(s.nunique(dropna=True)),
            "validity": round(validity, 4) if pd.notna(validity) else np.nan,
            "n_invalid": n_invalid,
            "has_rule": rule is not None,
        }

        if pd.api.types.is_numeric_dtype(s) and n_missing < n:
            desc = s.describe()
            rec.update(
                {
                    "min": desc.get("min"),
                    "median": desc.get("50%"),
                    "max": desc.get("max"),
                    "n_outliers_iqr": int(_iqr_outlier_mask(s).sum()),
                }
            )
        else:
            top = s.mode(dropna=True)
            rec["top_value"] = top.iloc[0] if len(top) else None

        rows.append(rec)

    return pd.DataFrame(rows).sort_values("pct_missing", ascending=False)


def _iqr_outlier_mask(s: pd.Series, k: float = 3.0) -> pd.Series:
    """Boolean mask of values beyond k IQRs. k=3 flags extreme values only."""
    s = pd.to_numeric(s, errors="coerce")
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    if pd.isna(iqr) or iqr == 0:
        return pd.Series(False, index=s.index)
    return (s < q1 - k * iqr) | (s > q3 + k * iqr)


# --------------------------------------------------------------------------
# Duplicate detection
# --------------------------------------------------------------------------
def detect_duplicates(df: pd.DataFrame) -> dict:
    """Exact, ID-level and business-key (near) duplicates."""
    result = {}

    exact = df.duplicated(keep="first")
    result["exact_duplicates"] = int(exact.sum())
    result["exact_duplicate_index"] = df.index[exact].tolist()[:1000]

    if "id" in df.columns:
        id_dupes = df["id"].notna() & df.duplicated(subset=["id"], keep="first")
        result["id_duplicates"] = int(id_dupes.sum())
    else:
        result["id_duplicates"] = 0

    key = [c for c in BUSINESS_KEY if c in df.columns]
    if key:
        near = df.duplicated(subset=key, keep="first")
        result["near_duplicates"] = int(near.sum())
        result["near_duplicate_key"] = key
        result["near_duplicate_index"] = df.index[near].tolist()[:1000]
    else:
        result["near_duplicates"] = 0
        result["near_duplicate_key"] = []

    n = len(df)
    total_dupes = max(result["exact_duplicates"], result["near_duplicates"])
    result["uniqueness_score"] = round(1 - total_dupes / n, 4) if n else 1.0
    return result


def duplicate_examples(df: pd.DataFrame, limit: int = 20) -> pd.DataFrame:
    """Return duplicate groups for data quality audit evidence."""
    key = [c for c in BUSINESS_KEY if c in df.columns]
    if not key:
        return pd.DataFrame()
    dup_mask = df.duplicated(subset=key, keep=False)
    if not dup_mask.any():
        return pd.DataFrame()
    return df.loc[dup_mask].sort_values(key).head(limit)


# --------------------------------------------------------------------------
# Overall score
# --------------------------------------------------------------------------
def assess(df: pd.DataFrame, verbose: bool = True) -> dict:
    """
    Full quality assessment.

    Returns a dict with the four dimension scores, the weighted overall score
    out of 100, the per-column profile, and the consistency rule breakdown.
    """
    profile = column_profile(df)

    completeness = float(profile["completeness"].mean())

    scored = profile.loc[profile["has_rule"], "validity"].dropna()
    validity = float(scored.mean()) if len(scored) else 1.0

    consistency_detail = {}
    passed = []
    for name, fn in CONSISTENCY_RULES.items():
        try:
            mask = fn(df)
        except Exception:
            mask = None
        if mask is None:
            continue
        rate = float(mask.mean())
        consistency_detail[name] = {
            "pass_rate": round(rate, 4),
            "n_violations": int((~mask).sum()),
        }
        passed.append(rate)
    consistency = float(np.mean(passed)) if passed else 1.0

    dupes = detect_duplicates(df)
    uniqueness = dupes["uniqueness_score"]

    w = config.QUALITY_WEIGHTS
    overall = (
        w["completeness"] * completeness
        + w["validity"] * validity
        + w["consistency"] * consistency
        + w["uniqueness"] * uniqueness
    ) * 100

    report = {
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "completeness": round(completeness * 100, 2),
        "validity": round(validity * 100, 2),
        "consistency": round(consistency * 100, 2),
        "uniqueness": round(uniqueness * 100, 2),
        "overall_score": round(overall, 2),
        "grade": _letter_grade(overall),
        "duplicates": {
            k: v for k, v in dupes.items() if not k.endswith("_index")
        },
        "consistency_detail": consistency_detail,
        "profile": profile,
    }

    if verbose:
        print_report(report)
    return report


def _letter_grade(score: float) -> str:
    if score >= 90:
        return "A - Excellent"
    if score >= 80:
        return "B - Good"
    if score >= 70:
        return "C - Acceptable"
    if score >= 60:
        return "D - Poor"
    return "E - Unusable without cleaning"


def print_report(report: dict) -> None:
    print("\n" + "=" * 62)
    print("DATA QUALITY REPORT")
    print("=" * 62)
    print(f"  Rows: {report['n_rows']:,}   Columns: {report['n_cols']}")
    print(f"  Completeness : {report['completeness']:6.2f}%")
    print(f"  Validity     : {report['validity']:6.2f}%")
    print(f"  Consistency  : {report['consistency']:6.2f}%")
    print(f"  Uniqueness   : {report['uniqueness']:6.2f}%")
    print("-" * 62)
    print(f"  OVERALL      : {report['overall_score']:6.2f} / 100  ({report['grade']})")
    print("=" * 62)

    d = report["duplicates"]
    print(f"  exact duplicates : {d['exact_duplicates']:,}")
    print(f"  id duplicates    : {d['id_duplicates']:,}")
    print(f"  near duplicates  : {d['near_duplicates']:,}")

    worst = report["profile"].head(8)
    if worst["pct_missing"].max() > 0:
        print("\n  Most incomplete columns:")
        for _, r in worst.iterrows():
            if r["pct_missing"] > 0:
                print(f"    {r['column']:<28} {r['pct_missing']:6.2f}% missing")
    print()


def save_report(report: dict, path) -> None:
    report["profile"].to_csv(path, index=False)
    print(f"  saved column profile -> {path}")


if __name__ == "__main__":
    from data_loader import load_raw

    assess(load_raw())
