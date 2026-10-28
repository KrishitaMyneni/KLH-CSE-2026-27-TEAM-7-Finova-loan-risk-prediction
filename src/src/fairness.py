"""
Segment-level fairness audit of the champion's decisions on the test window.

Lending Club publishes no protected attributes (race, sex, age), so this audit
cannot test protected classes directly. It checks the segments that are
available and that commonly act as proxies in fair-lending reviews: income
band, region, home ownership and income verification.

For each segment:
  approval_rate           share the policy would APPROVE (Low Risk band)
  adverse_impact_ratio    approval_rate / best segment's approval_rate.
                          Below 0.80 fails the "four-fifths" screening rule.
  good_approval_rate      approval rate among borrowers who actually repaid
                          (equal-opportunity view: are good borrowers in this
                          segment approved as often as elsewhere?)
  calibration_gap         avg predicted PD - actual default rate. Positive means
                          the model overstates this segment's risk.
  auc                     ranking quality inside the segment
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config

STATE_REGION = {
    **dict.fromkeys(["CT", "ME", "MA", "NH", "RI", "VT", "NJ", "NY", "PA"], "Northeast"),
    **dict.fromkeys(["IL", "IN", "MI", "OH", "WI", "IA", "KS", "MN", "MO", "NE", "ND", "SD"], "Midwest"),
    **dict.fromkeys(["DE", "DC", "FL", "GA", "MD", "NC", "SC", "VA", "WV", "AL", "KY", "MS",
                     "TN", "AR", "LA", "OK", "TX"], "South"),
    **dict.fromkeys(["AZ", "CO", "ID", "MT", "NV", "NM", "UT", "WY", "AK", "CA", "HI", "OR",
                     "WA"], "West"),
}

AIR_THRESHOLD = 0.80


def _segments(df: pd.DataFrame) -> dict[str, pd.Series]:
    seg = {}
    if "annual_inc" in df.columns:
        seg["income_band"] = pd.qcut(
            df["annual_inc"], 4, labels=["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"],
            duplicates="drop",
        ).astype(str)
    if "addr_state" in df.columns:
        seg["region"] = df["addr_state"].map(STATE_REGION).fillna("Other")
    for col in ["home_ownership", "verification_status"]:
        if col in df.columns:
            seg[col] = df[col].astype(str)
    return seg


def audit(scored: pd.DataFrame, min_group: int = 500) -> pd.DataFrame:
    """`scored` needs default_probability, risk_band and the target."""
    y = scored[config.TARGET]
    approved = scored["risk_band"] == config.RISK_LABELS[0]
    rows = []
    for seg_name, groups in _segments(scored).items():
        for g in groups.dropna().unique():
            m = groups == g
            if m.sum() < min_group:
                continue
            yy, pp = y[m], scored.loc[m, "default_probability"]
            goods = m & (y == 0)
            rows.append({
                "segment": seg_name,
                "group": g,
                "n_loans": int(m.sum()),
                "actual_default_rate": round(float(yy.mean()), 4),
                "avg_predicted_pd": round(float(pp.mean()), 4),
                "calibration_gap": round(float(pp.mean() - yy.mean()), 4),
                "approval_rate": round(float(approved[m].mean()), 4),
                "good_approval_rate": round(float(approved[goods].mean()), 4) if goods.any() else np.nan,
                "approved_default_rate": round(float(yy[approved[m]].mean()), 4) if approved[m].any() else np.nan,
                "auc": round(float(roc_auc_score(yy, pp)), 4) if yy.nunique() == 2 else np.nan,
            })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    best = out.groupby("segment")["approval_rate"].transform("max")
    out["adverse_impact_ratio"] = (out["approval_rate"] / best).round(3)
    out["passes_four_fifths"] = out["adverse_impact_ratio"] >= AIR_THRESHOLD
    return out


def summarise(report: pd.DataFrame) -> dict:
    if report.empty:
        return {}
    summary = {}
    for seg, grp in report.groupby("segment"):
        summary[seg] = {
            "min_adverse_impact_ratio": float(grp["adverse_impact_ratio"].min()),
            "groups_failing_four_fifths": grp.loc[~grp["passes_four_fifths"], "group"].tolist(),
            "max_abs_calibration_gap": float(grp["calibration_gap"].abs().max()),
            "auc_range": [float(grp["auc"].min()), float(grp["auc"].max())],
        }
    return summary
