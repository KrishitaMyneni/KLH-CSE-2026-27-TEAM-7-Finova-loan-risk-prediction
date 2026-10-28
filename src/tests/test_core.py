"""Unit tests for the data, feature, risk, explainability and fairness layers."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config
from src import cleaning, explain, fairness, features, models, risk


# --------------------------------------------------------------- cleaning
def test_parsers_handle_lending_club_formats():
    assert cleaning.parse_term(pd.Series([" 36 months", "60 months"])).tolist() == [36, 60]
    assert cleaning.parse_percent(pd.Series(["13.56%", " 7.1"])).tolist() == [13.56, 7.1]
    emp = cleaning.parse_emp_length(pd.Series(["< 1 year", "10+ years", "3 years", "n/a"]))
    assert emp.iloc[:3].tolist() == [0.0, 10.0, 3.0] and np.isnan(emp.iloc[3])
    assert cleaning.parse_lc_date(pd.Series(["Dec-2015"])).iloc[0] == pd.Timestamp("2015-12-01")


def test_leakage_columns_are_dropped():
    df = pd.DataFrame({"loan_amnt": [1000], "total_pymnt": [1100], "recoveries": [0]})
    out, _ = cleaning.clean(df, verbose=False, is_training=False)
    assert not set(config.LEAKAGE_COLS) & set(out.columns)


def _applicant(**over):
    row = {"loan_amnt": 10000, "term": " 36 months", "annual_inc": 60000, "dti": 18.0,
           "revol_util": "50%", "emp_length": "5 years", "issue_d": "Jan-2016",
           "earliest_cr_line": "Jan-2005", "home_ownership": "RENT"}
    row.update(over)
    return row


def test_inference_cleaning_does_not_depend_on_the_batch():
    """An applicant's cleaned values must not change with who else is uploaded."""
    alone = pd.DataFrame([_applicant()])
    batch = pd.DataFrame([_applicant(), _applicant(annual_inc=5_000_000, dti=None)])
    a, _ = cleaning.clean(alone, verbose=False, is_training=False)
    b, _ = cleaning.clean(batch, verbose=False, is_training=False)
    pd.testing.assert_series_equal(a.iloc[0], b.iloc[0], check_names=False)
    assert np.isnan(b.iloc[1]["dti"])  # left for the pipeline's training-median imputer


def test_impossible_values_are_nulled():
    df = pd.DataFrame([_applicant(dti=-5), _applicant(annual_inc=0)])
    out, _ = cleaning.clean(df, verbose=False, is_training=False)
    assert np.isnan(out.loc[0, "dti"]) and np.isnan(out.loc[1, "annual_inc"])


# --------------------------------------------------------------- features
def test_quantile_clipper_uses_fit_bounds_only():
    clip = features.QuantileClipper(0.0, 1.0).fit(np.array([[1.0], [2.0], [3.0]]))
    assert clip.transform(np.array([[100.0], [-5.0]])).ravel().tolist() == [3.0, 1.0]


def test_time_split_has_no_shared_months():
    dates = pd.date_range("2013-01-01", "2015-09-01", freq="MS")
    df = pd.DataFrame({"issue_d": np.repeat(dates, 5), config.TARGET: 0})
    tr, cal, te = features.time_split(df, return_cal=True)
    assert tr["issue_d"].max() < cal["issue_d"].min()
    assert cal["issue_d"].max() < te["issue_d"].min()
    assert te["issue_d"].max() <= pd.Timestamp(config.TEST_END_DATE)


def test_lc_pricing_signals_excluded_by_default():
    df = pd.DataFrame(columns=config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES)
    num, cat = features.select_feature_columns(df)
    assert not {"int_rate", "grade", "installment"} & set(num + cat)


# --------------------------------------------------------------- risk
def test_score_anchor_and_monotonicity():
    assert risk.probability_to_score(0.20) == config.SCORE_BASE
    s = risk.probability_to_score(np.array([0.02, 0.1, 0.3, 0.6]))
    assert all(np.diff(s) < 0)
    # doubling the good:bad odds (4:1 -> 8:1, i.e. PD 1/9) adds PDO points
    assert risk.probability_to_score(1 / 9) == config.SCORE_BASE + config.SCORE_PDO


def test_bands_and_decisions_follow_config():
    lo, mid = config.RISK_THRESHOLDS["low_max"], config.RISK_THRESHOLDS["medium_max"]
    bands = risk.assign_band(np.array([lo, lo + 1e-3, mid, mid + 1e-3]))
    assert bands.tolist() == ["Low Risk", "Medium Risk", "Medium Risk", "High Risk"]
    assert risk.decision(lo / 2)["recommendation"] == "APPROVE"
    assert risk.decision(mid + 0.05)["recommendation"] == "DECLINE"


def test_psi_zero_for_identical_and_large_for_shifted():
    rng = np.random.default_rng(0)
    x = rng.normal(size=5000)
    assert risk.calculate_psi(x, x)["psi"] == 0.0
    assert risk.calculate_psi(x, x + 2)["psi"] > 0.25


def test_reason_codes_sorted_and_readable():
    codes = risk.generate_reason_codes(
        ["dti", "fico_avg", "purpose"], np.array([0.3, -0.5, 0.1]),
        {"dti": 31.2, "fico_range_low": 700, "fico_range_high": 704, "purpose": "credit_card"},
        top_k=2,
    )
    assert [c["feature"] for c in codes] == ["fico_avg", "dti"]
    assert codes[0]["value"] == 702 and "reduced" in codes[0]["description"]
    assert codes[1]["value"] == "31.2%" and "increased" in codes[1]["description"]


# --------------------------------------------------------------- explain
@pytest.fixture(scope="module")
def toy_data():
    rng = np.random.default_rng(1)
    n = 3000
    X = pd.DataFrame({
        "a": rng.normal(size=n),
        "b": rng.normal(size=n),
        "purpose": rng.choice(["car", "credit_card", "other"], size=n),
    })
    logit = 1.2 * X["a"] - 0.8 * X["b"] + (X["purpose"] == "car") * 0.7 - 1.5
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return X, y


@pytest.mark.parametrize("name", ["Logistic Regression", "XGBoost"])
def test_explainer_is_additive_and_groups_one_hot(toy_data, name):
    from sklearn.pipeline import Pipeline

    X, y = toy_data
    est = models.get_model_zoo(1.0)[name]
    pipe = Pipeline([("prep", features.build_preprocessor(["a", "b"], ["purpose"])),
                     ("model", est)]).fit(X, y)
    ex = explain.Explainer(pipe, ["a", "b"], ["purpose"], background=X)
    assert ex.supported
    contrib = ex.contributions(X.head(50))
    assert list(contrib.columns) == ["a", "b", "purpose"]
    # driver signs match the generating process
    assert np.corrcoef(X["a"].head(50), contrib["a"])[0, 1] > 0.8
    assert np.corrcoef(X["b"].head(50), contrib["b"])[0, 1] < -0.8


# --------------------------------------------------------------- fairness
def test_fairness_flags_four_fifths_failures():
    n = 2000
    scored = pd.DataFrame({
        "home_ownership": ["RENT"] * n + ["OWN"] * n,
        config.TARGET: [0, 1] * n,
        "default_probability": [0.05] * n + [0.30] * n,
    })
    scored["risk_band"] = risk.assign_band(scored["default_probability"])
    rep = fairness.audit(scored, min_group=100)
    own = rep[rep["group"] == "OWN"].iloc[0]
    assert own["adverse_impact_ratio"] == 0.0 and not own["passes_four_fifths"]
    assert fairness.summarise(rep)["home_ownership"]["groups_failing_four_fifths"] == ["OWN"]
