"""
LAYER 2 - Step 8: Risk Prediction / Risk Classification.

Turns a raw probability of default into something a lending officer can act on:
a Low / Medium / High band, plus an expected-loss figure in rupees or dollars.

Two banding strategies:

  absolute  - fixed probability thresholds (config.RISK_THRESHOLDS). Interpretable,
              stable over time, and tied to a stated risk appetite.
  quantile  - tertiles of the predicted distribution. Always produces balanced
              buckets, but the meaning of "High Risk" drifts with the portfolio.

Absolute is the default, because a band should mean the same thing every quarter.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config

# Loss Given Default: industry convention for unsecured consumer credit is that
# roughly 55-65% of the outstanding balance is unrecoverable after charge-off.
DEFAULT_LGD = config.LGD


def assign_band(prob, thresholds: dict | None = None, strategy: str = "absolute"):
    """
    prob : float, list, ndarray or Series of predicted default probabilities
    returns the same shape, as risk-band labels
    """
    thresholds = thresholds or config.RISK_THRESHOLDS
    low, med, high = config.RISK_LABELS

    scalar = np.isscalar(prob)
    s = pd.Series([prob] if scalar else prob, dtype="float64")

    if strategy == "quantile":
        q1, q2 = s.quantile([1 / 3, 2 / 3])
        out = pd.Series(
            np.select([s <= q1, s <= q2], [low, med], default=high), index=s.index
        )
    else:
        out = pd.Series(
            np.select(
                [s <= thresholds["low_max"], s <= thresholds["medium_max"]],
                [low, med],
                default=high,
            ),
            index=s.index,
        )

    return out.iloc[0] if scalar else out.to_numpy()


def score_dataframe(df: pd.DataFrame, pipeline, strategy: str = "absolute"):
    """Attach default_probability, risk_band and credit_score to a DataFrame."""
    from src.models import model_feature_columns

    cols = [c for c in model_feature_columns(pipeline) if c in df.columns]
    X = df[cols] if cols else df
    prob = pipeline.predict_proba(X)[:, 1]

    out = df.copy()
    out["default_probability"] = prob
    out["risk_band"] = assign_band(prob, strategy=strategy)
    out["credit_score"] = probability_to_score(prob)
    return out


def probability_to_score(
    prob,
    base: int = config.SCORE_BASE,
    pdo: int = config.SCORE_PDO,
    base_odds: float = config.SCORE_BASE_ODDS,
):
    """
    Convert probability to a familiar 300-850 style scorecard number.

    Standard points-to-double-the-odds scaling, which is how FICO-style scores
    are actually built. A higher score means a safer borrower, so it reads the
    right way round in production scorecards.

    The anchors are set to this portfolio: base_odds=4 is a 20% default rate,
    which is roughly the book average, and it is pinned to 660. Every 40 points
    above that halves the odds of default. So ~20% PD scores 660, ~5% scores
    about 750, and ~40% scores about 600.
    """
    prob = np.clip(np.asarray(prob, dtype=float), 1e-6, 1 - 1e-6)
    odds = (1 - prob) / prob
    factor = pdo / np.log(2)
    offset = base - factor * np.log(base_odds)
    score = offset + factor * np.log(odds)
    return np.clip(score, 300, 850).round().astype(int)


def expected_loss(prob, exposure, lgd: float = DEFAULT_LGD):
    """EL = PD x LGD x EAD. The standard Basel formulation."""
    return np.asarray(prob) * lgd * np.asarray(exposure)


def portfolio_summary(scored: pd.DataFrame, amount_col: str = "loan_amnt") -> pd.DataFrame:
    """Per-band aggregation: counts, exposure, expected loss, actual default rate."""
    df = scored.copy()
    if amount_col in df.columns:
        df["expected_loss"] = expected_loss(
            df["default_probability"], df[amount_col]
        )

    agg = {
        "default_probability": ["count", "mean"],
    }
    if amount_col in df.columns:
        agg[amount_col] = "sum"
        agg["expected_loss"] = "sum"
    if config.TARGET in df.columns:
        agg[config.TARGET] = "mean"

    summary = df.groupby("risk_band", observed=True).agg(agg)
    summary.columns = ["_".join(c).strip("_") for c in summary.columns]

    rename = {
        "default_probability_count": "n_loans",
        "default_probability_mean": "avg_predicted_pd",
        f"{amount_col}_sum": "total_exposure",
        "expected_loss_sum": "total_expected_loss",
        f"{config.TARGET}_mean": "actual_default_rate",
    }
    summary = summary.rename(columns=rename)

    if "n_loans" in summary.columns:
        summary["pct_of_portfolio"] = (
            100 * summary["n_loans"] / summary["n_loans"].sum()
        ).round(2)

    order = [b for b in config.RISK_LABELS if b in summary.index]
    return summary.reindex(order)


RATIONALES = {
    "APPROVE": "Default probability is within standard risk appetite.",
    "REVIEW": "Elevated risk. Forward to an underwriter: consider a higher rate, "
              "a smaller principal, or additional income verification.",
    "DECLINE": "Default probability exceeds risk appetite.",
}


def decision(prob: float, thresholds: dict | None = None) -> dict:
    """Band, score and lending recommendation for one applicant."""
    band = assign_band(prob, thresholds=thresholds)
    action = config.DECISIONS[band]
    return {
        "probability_of_default": round(float(prob), 4),
        "risk_band": band,
        "credit_score": int(probability_to_score(prob)),
        "recommendation": action,
        "rationale": RATIONALES[action],
    }


def decision_zone_default_rates(scored: pd.DataFrame) -> dict:
    """Observed default rate inside each decision zone (the policy backtest)."""
    if config.TARGET not in scored.columns:
        return {}
    out = {}
    for band, grp in scored.groupby("risk_band", observed=True):
        out[config.DECISIONS[band]] = {
            "n_loans": int(len(grp)),
            "share": round(len(grp) / len(scored), 4),
            "actual_default_rate": round(float(grp[config.TARGET].mean()), 4),
            "avg_predicted_pd": round(float(grp["default_probability"].mean()), 4),
        }
    return out


def gains_table(scored: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    """
    Decile lift table — rank borrowers by predicted risk, cut into ten buckets,
    show the actual default rate in each. If the model works, the top decile
    should default several times more often than the bottom. This is the single
    most persuasive table you can show a non-technical examiner.
    """
    if config.TARGET not in scored.columns:
        return pd.DataFrame()

    df = scored.sort_values("default_probability", ascending=False).copy()
    df["decile"] = pd.qcut(
        df["default_probability"].rank(method="first", ascending=False),
        n_bins,
        labels=range(1, n_bins + 1),
    )

    table = df.groupby("decile", observed=True).agg(
        n_loans=(config.TARGET, "size"),
        n_defaults=(config.TARGET, "sum"),
        actual_default_rate=(config.TARGET, "mean"),
        avg_predicted_pd=("default_probability", "mean"),
    )

    overall = df[config.TARGET].mean()
    table["lift"] = (table["actual_default_rate"] / overall).round(2)
    table["cumulative_defaults_pct"] = (
        100 * table["n_defaults"].cumsum() / table["n_defaults"].sum()
    ).round(1)
    return table.reset_index()


def calculate_psi(
    expected: np.ndarray | pd.Series,
    actual: np.ndarray | pd.Series,
    num_buckets: int = 10,
    epsilon: float = 1e-4,
) -> dict:
    """
    Population Stability Index (PSI) between expected (train) and actual (test).
    Uses quantile bins based on the expected distribution.
    """
    exp = pd.Series(expected).dropna().to_numpy()
    act = pd.Series(actual).dropna().to_numpy()
    if len(exp) == 0 or len(act) == 0:
        return {"psi": 0.0, "status": "Stable", "n_expected": 0, "n_actual": 0}

    percentiles = np.linspace(0, 100, num_buckets + 1)
    bins = np.percentile(exp, percentiles)
    bins = np.unique(bins)
    if len(bins) < 2:
        return {"psi": 0.0, "status": "Stable", "n_expected": len(exp), "n_actual": len(act)}

    bins[0] = -np.inf
    bins[-1] = np.inf

    exp_counts, _ = np.histogram(exp, bins=bins)
    act_counts, _ = np.histogram(act, bins=bins)

    exp_pct = exp_counts / len(exp)
    act_pct = act_counts / len(act)

    exp_pct = np.where(exp_pct == 0, epsilon, exp_pct)
    act_pct = np.where(act_pct == 0, epsilon, act_pct)

    psi_vals = (act_pct - exp_pct) * np.log(act_pct / exp_pct)
    total_psi = float(np.sum(psi_vals))

    if total_psi < 0.10:
        status = "Stable (PSI < 0.10)"
    elif total_psi < 0.25:
        status = "Moderate Drift (0.10 <= PSI < 0.25)"
    else:
        status = "Significant Drift (PSI >= 0.25)"

    return {
        "psi": round(total_psi, 4),
        "status": status,
        "n_expected": len(exp),
        "n_actual": len(act),
    }


FEATURE_HUMAN_NAMES = {
    "loan_amnt": "Loan amount",
    "term": "Loan term",
    "dti": "Debt-to-income ratio",
    "annual_inc_log": "Annual income",
    "revol_bal": "Revolving credit balance",
    "acc_open_past_24mths": "Accounts opened in the past 24 months",
    "credit_history_months": "Length of credit history",
    "bc_util": "Bankcard utilization",
    "fico_avg": "FICO score",
    "credit_utilisation": "Total credit utilization",
    "credit_limit_util": "Balance relative to total credit limit",
    "open_acc_ratio": "Share of credit lines still open",
    "open_acc": "Open credit lines",
    "total_acc": "Total credit lines",
    "revol_util": "Revolving line utilization",
    "revol_util_high": "Revolving utilization above 80%",
    "loan_to_income": "Loan amount relative to income",
    "inq_velocity": "Credit inquiries relative to history length",
    "inq_last_6mths": "Credit inquiries in the past 6 months",
    "mort_acc": "Mortgage accounts",
    "pub_rec": "Public derogatory records",
    "pub_rec_bankruptcies": "Public bankruptcy records",
    "delinq_2yrs": "Delinquencies in the past 2 years",
    "has_delinq_history": "Any delinquency in the past 2 years",
    "delinq_severity_score": "Delinquency and public-record severity",
    "percent_bc_gt_75": "Bankcards above 75% of limit",
    "emp_length_years": "Employment length",
    "emp_length_band": "Employment length",
    "home_ownership": "Home ownership",
    "verification_status": "Income verification",
    "purpose": "Loan purpose",
    "initial_list_status": "Listing status",
    "application_type": "Application type",
}

# Which raw upload column to show next to each model feature
_VALUE_SOURCE = {
    "annual_inc_log": "annual_inc",
    "emp_length_years": "emp_length",
    "emp_length_band": "emp_length",
    "revol_util_high": "revol_util",
    "has_delinq_history": "delinq_2yrs",
}

_PERCENT = {"dti", "revol_util", "bc_util", "percent_bc_gt_75", "revol_util_high"}
_MONEY = {"annual_inc_log", "loan_amnt", "revol_bal"}


def _display_value(feature: str, row: dict | None):
    if not row:
        return "N/A"
    if feature == "fico_avg":
        lo, hi = row.get("fico_range_low"), row.get("fico_range_high")
        try:
            return int(round((float(lo) + float(hi)) / 2))
        except (TypeError, ValueError):
            return lo if lo is not None and pd.notna(lo) else "N/A"
    val = row.get(_VALUE_SOURCE.get(feature, feature))
    if val is None or (not isinstance(val, str) and pd.isna(val)) or str(val).strip() == "":
        return "N/A"
    try:
        num = float(str(val).replace("%", "").replace(",", ""))
    except ValueError:
        return str(val).strip()
    if feature in _MONEY:
        return f"${num:,.0f}"
    if feature in _PERCENT:
        return f"{num:.1f}%"
    return int(num) if num.is_integer() else round(num, 2)


def generate_reason_codes(feature_names: list, shap_values: np.ndarray,
                          original_row: dict | None = None, top_k: int = 4) -> list[dict]:
    """
    Plain-English reason codes from per-feature contributions.
    Positive contribution -> pushed default risk up; negative -> pulled it down.
    `feature_names` are source features (one-hot columns already summed).
    """
    items = []
    for name, val in zip(feature_names, shap_values, strict=True):
        label = FEATURE_HUMAN_NAMES.get(name, name.replace("_", " ").capitalize())
        shown = _display_value(name, original_row)
        suffix = f" ({shown})" if shown != "N/A" else ""
        if val > 0:
            impact, desc = "increased risk", f"{label}{suffix} increased default risk"
        else:
            impact, desc = "decreased risk", f"{label}{suffix} reduced default risk"
        items.append({
            "feature": name,
            "label": label,
            "shap_value": round(float(val), 4),
            "abs_impact": float(abs(val)),
            "impact": impact,
            "description": desc,
            "value": shown,
        })

    items.sort(key=lambda x: x["abs_impact"], reverse=True)
    return items[:top_k]
