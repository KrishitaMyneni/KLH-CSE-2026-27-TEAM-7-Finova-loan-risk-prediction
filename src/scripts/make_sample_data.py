"""
Generate a synthetic dataset with the same schema, dtypes and mess as the real
Lending Club file — including the string formatting quirks ('36 months',
'13.56%', '< 1 year'), missing values, duplicate rows and out-of-range outliers.

Use it to develop and test the whole pipeline before (or instead of) downloading
the 1.6 GB original, and to let all three teammates run the code on day one.

    python scripts/make_sample_data.py --rows 50000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config

RNG = np.random.default_rng(42)

GRADES = list("ABCDEFG")
GRADE_P = [0.17, 0.29, 0.27, 0.16, 0.07, 0.03, 0.01]
GRADE_BASE_RATE = dict(zip(GRADES, [0.05, 0.09, 0.15, 0.22, 0.30, 0.38, 0.45]))
GRADE_RATE = dict(zip(GRADES, [7.0, 10.5, 13.8, 17.5, 21.0, 25.0, 28.5]))

PURPOSES = [
    "debt_consolidation", "credit_card", "home_improvement", "other",
    "major_purchase", "medical", "small_business", "car", "vacation",
    "moving", "house", "renewable_energy", "wedding",
]
PURPOSE_P = [0.58, 0.23, 0.06, 0.04, 0.02, 0.015, 0.015, 0.012,
             0.009, 0.008, 0.006, 0.003, 0.002]

STATES = ["CA", "NY", "TX", "FL", "IL", "NJ", "PA", "OH", "GA", "VA",
          "NC", "MI", "MD", "AZ", "MA", "WA", "CO", "MN", "MO", "IN"]

EMP_LENGTHS = ["< 1 year", "1 year", "2 years", "3 years", "4 years", "5 years",
               "6 years", "7 years", "8 years", "9 years", "10+ years"]


def generate(n: int) -> pd.DataFrame:
    grade = RNG.choice(GRADES, size=n, p=GRADE_P)

    loan_amnt = np.round(RNG.gamma(4, 3500, n).clip(1000, 40000), -2)
    term_months = RNG.choice([36, 60], size=n, p=[0.72, 0.28])
    int_rate = np.array([GRADE_RATE[g] for g in grade]) + RNG.normal(0, 1.2, n)
    int_rate = int_rate.clip(5.3, 31.0)

    monthly_r = int_rate / 100 / 12
    installment = (
        loan_amnt * monthly_r / (1 - (1 + monthly_r) ** -term_months)
    ).round(2)

    annual_inc = np.round(RNG.lognormal(11.05, 0.62, n).clip(4000, 800000), -2)
    dti = (RNG.gamma(3.2, 5.8, n)).clip(0, 55).round(2)
    emp_length = RNG.choice(EMP_LENGTHS, size=n)

    fico_low = (RNG.normal(698, 32, n)).clip(660, 845)
    fico_low = (np.floor(fico_low / 5) * 5).astype(int)

    issue_dates = pd.to_datetime("2013-01-01") + pd.to_timedelta(
        RNG.integers(0, 6 * 365, n), unit="D"
    )
    hist_years = RNG.gamma(4, 4, n).clip(1, 45)
    earliest_cr = issue_dates - pd.to_timedelta((hist_years * 365).astype(int), unit="D")

    revol_util = RNG.beta(2.4, 2.6, n) * 100
    open_acc = RNG.poisson(11, n).clip(1, 60)
    total_acc = open_acc + RNG.poisson(13, n)
    delinq_2yrs = RNG.poisson(0.3, n).clip(0, 20)
    pub_rec = RNG.poisson(0.16, n).clip(0, 15)
    inq_6m = RNG.poisson(0.7, n).clip(0, 20)
    revol_bal = (annual_inc * RNG.beta(1.6, 8, n)).round(0)

    df = pd.DataFrame(
        {
            "id": np.arange(1_000_000, 1_000_000 + n),
            "loan_amnt": loan_amnt,
            "funded_amnt": loan_amnt,
            "term": [f" {t} months" for t in term_months],
            "int_rate": [f" {r:.2f}%" for r in int_rate],
            "installment": installment,
            "grade": grade,
            "sub_grade": [f"{g}{RNG.integers(1, 6)}" for g in grade],
            "emp_length": emp_length,
            "home_ownership": RNG.choice(
                ["MORTGAGE", "RENT", "OWN", "ANY"], n, p=[0.49, 0.40, 0.105, 0.005]
            ),
            "annual_inc": annual_inc,
            "verification_status": RNG.choice(
                ["Verified", "Source Verified", "Not Verified"], n, p=[0.30, 0.38, 0.32]
            ),
            "issue_d": issue_dates.strftime("%b-%Y"),
            "purpose": RNG.choice(PURPOSES, n, p=PURPOSE_P),
            "addr_state": RNG.choice(STATES, n),
            "dti": dti,
            "delinq_2yrs": delinq_2yrs,
            "earliest_cr_line": earliest_cr.strftime("%b-%Y"),
            "fico_range_low": fico_low,
            "fico_range_high": fico_low + 4,
            "inq_last_6mths": inq_6m,
            "mths_since_last_delinq": np.where(
                delinq_2yrs > 0, RNG.integers(1, 60, n), np.nan
            ),
            "open_acc": open_acc,
            "pub_rec": pub_rec,
            "revol_bal": revol_bal,
            "revol_util": [f"{u:.1f}%" for u in revol_util],
            "total_acc": total_acc,
            "initial_list_status": RNG.choice(["w", "f"], n, p=[0.72, 0.28]),
            "application_type": RNG.choice(
                ["Individual", "Joint App"], n, p=[0.94, 0.06]
            ),
            "mort_acc": RNG.poisson(1.6, n).clip(0, 25),
            "pub_rec_bankruptcies": (pub_rec * RNG.binomial(1, 0.6, n)).clip(0, 10),
            "tot_cur_bal": (annual_inc * RNG.gamma(2, 1.1, n)).round(0),
            "acc_open_past_24mths": RNG.poisson(4.3, n).clip(0, 40),
            "avg_cur_bal": (revol_bal / open_acc).round(0),
            "bc_open_to_buy": (RNG.gamma(2, 6000, n)).round(0),
            "bc_util": np.round(RNG.beta(2.6, 2.2, n) * 100, 1),
            "num_actv_bc_tl": RNG.poisson(3.6, n).clip(0, 30),
            "num_tl_op_past_12m": RNG.poisson(2.1, n).clip(0, 25),
            "percent_bc_gt_75": np.round(RNG.beta(1.4, 2.2, n) * 100, 1),
            "tot_hi_cred_lim": (annual_inc * RNG.gamma(2.4, 1.3, n)).round(0),
            "total_bc_limit": (RNG.gamma(2.2, 9000, n)).round(0),
        }
    )

    # ---- target with genuine, learnable signal --------------------------
    logit = (
        -3.30
        + 3.1 * np.array([GRADE_BASE_RATE[g] for g in grade])
        + 0.030 * dti
        + 0.55 * (term_months == 60)
        - 0.42 * np.log1p(annual_inc / 50000)
        + 0.012 * revol_util
        + 0.22 * delinq_2yrs
        + 0.17 * inq_6m
        - 0.0055 * (fico_low - 660)
        + 0.9 * (loan_amnt / annual_inc)
        + RNG.normal(0, 0.85, n)
    )
    p_default = 1 / (1 + np.exp(-logit))
    defaulted = RNG.binomial(1, p_default).astype(bool)

    df["loan_status"] = np.where(defaulted, "Charged Off", "Fully Paid")

    # ~8% of loans are still running - outcome unknown, must be dropped
    still_running = RNG.random(n) < 0.08
    df.loc[still_running, "loan_status"] = RNG.choice(
        ["Current", "Late (31-120 days)", "In Grace Period"],
        still_running.sum(),
        p=[0.88, 0.08, 0.04],
    )

    # ---- post-origination columns: the leakage trap ---------------------
    paid_ratio = np.where(defaulted, RNG.beta(2, 5, n), 1.0)
    df["total_pymnt"] = (loan_amnt * paid_ratio * 1.13).round(2)
    df["total_rec_prncp"] = (loan_amnt * paid_ratio).round(2)
    df["recoveries"] = np.where(defaulted, loan_amnt * RNG.beta(1, 9, n), 0).round(2)
    df["out_prncp"] = np.where(still_running, loan_amnt * 0.4, 0).round(2)
    df["last_pymnt_amnt"] = (installment * RNG.uniform(0.5, 3, n)).round(2)
    df["last_pymnt_d"] = issue_dates.strftime("%b-%Y")
    df["debt_settlement_flag"] = np.where(defaulted & (RNG.random(n) < 0.1), "Y", "N")

    return _add_realistic_mess(df)


def _add_realistic_mess(df: pd.DataFrame) -> pd.DataFrame:
    """Inject the specific kinds of dirt the real file contains."""
    n = len(df)

    # missing values, at rates close to the real dataset
    for col, rate in {
        "emp_length": 0.06,
        "dti": 0.002,
        "revol_util": 0.001,
        "mort_acc": 0.05,
        "bc_util": 0.03,
        "percent_bc_gt_75": 0.03,
        "avg_cur_bal": 0.02,
        "tot_cur_bal": 0.02,
        "annual_inc": 0.0005,
    }.items():
        mask = RNG.random(n) < rate
        df.loc[mask, col] = np.nan

    # a column that is almost entirely empty - should get dropped by cleaning
    df["mths_since_last_record"] = np.where(
        RNG.random(n) < 0.12, RNG.integers(0, 120, n), np.nan
    )

    # 'n/a' as a string, not a null
    na_mask = RNG.random(n) < 0.02
    df.loc[na_mask, "emp_length"] = "n/a"

    # impossible values
    bad = RNG.random(n) < 0.004
    df.loc[bad, "dti"] = -1.0
    huge = RNG.random(n) < 0.0015
    df.loc[huge, "annual_inc"] = RNG.uniform(3e6, 9.5e6, huge.sum())
    zero = RNG.random(n) < 0.0008
    df.loc[zero, "annual_inc"] = 0

    # duplicate rows: exact copies and near-copies with a different id
    n_dupes = max(int(n * 0.012), 1)
    dupe_idx = RNG.choice(n, n_dupes, replace=False)
    exact = df.iloc[dupe_idx].copy()
    near = df.iloc[dupe_idx[: n_dupes // 2]].copy()
    near["id"] = near["id"] + 5_000_000

    out = pd.concat([df, exact, near], ignore_index=True)
    return out.sample(frac=1, random_state=7).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=50_000)
    ap.add_argument("--out", default=str(config.DATA_DIR / "sample_loans.csv"))
    args = ap.parse_args()

    print(f"generating {args.rows:,} synthetic loans ...")
    df = generate(args.rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    resolved = df["loan_status"].isin(["Fully Paid", "Charged Off"])
    rate = (df.loc[resolved, "loan_status"] == "Charged Off").mean()
    print(f"  wrote {len(df):,} rows x {df.shape[1]} cols -> {args.out}")
    print(f"  default rate among resolved loans: {rate:.2%}")
    print(f"\nNow run:  python run_pipeline.py --data {args.out}")


if __name__ == "__main__":
    main()
