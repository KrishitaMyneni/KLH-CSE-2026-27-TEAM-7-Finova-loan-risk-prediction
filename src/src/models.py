"""
LAYER 2 - Step 7: Machine Learning.

Trains and compares Logistic Regression, Random Forest, gradient boosters, an
MLP and a soft-voting ensemble, and evaluates them with metrics that make sense
for an imbalanced credit problem.

On metrics: accuracy is useless here. With roughly 20% defaults, a model that
predicts "nobody defaults" scores 80% accuracy and is worthless. Use:

  ROC-AUC   - overall ranking ability; the headline number
  PR-AUC    - precision/recall trade-off on the minority class
  KS        - max separation between good and bad score distributions.
              The standard credit-scoring statistic. Above 0.30 is deployable.
  Recall    - of borrowers who actually defaulted, how many did we flag
  Brier     - are the probabilities themselves calibrated
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config
from src.features import build_preprocessor


def ks_statistic(y_true, y_prob) -> float:
    """Kolmogorov-Smirnov: max gap between TPR and FPR across all cut-offs."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(tpr - fpr))


def bootstrap_auc_ci(
    y_true, y_prob, n_bootstraps: int = 1000, alpha: float = 0.05, seed: int = 42
) -> tuple[float, float]:
    """Calculate 95% bootstrap confidence interval for ROC-AUC on the test set."""
    from joblib import Parallel, delayed

    n = len(y_true)
    y_true_arr = np.asarray(y_true)
    y_prob_arr = np.asarray(y_prob)

    def _one_boot(s):
        rng = np.random.default_rng(s)
        idx = rng.integers(0, n, size=n)
        sample_y = y_true_arr[idx]
        if sample_y.sum() == 0 or sample_y.sum() == n:
            return np.nan
        return roc_auc_score(sample_y, y_prob_arr[idx])

    scores = Parallel(n_jobs=-1, prefer="threads")(
        delayed(_one_boot)(seed + i) for i in range(n_bootstraps)
    )
    valid = [sc for sc in scores if not np.isnan(sc)]
    lower = float(np.percentile(valid, 100 * (alpha / 2)))
    upper = float(np.percentile(valid, 100 * (1 - alpha / 2)))
    return round(lower, 4), round(upper, 4)


def _xgb(scale_pos_weight: float):
    from xgboost import XGBClassifier

    return XGBClassifier(
        n_estimators=250,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=30,
        gamma=0.3,
        reg_lambda=4.0,
        reg_alpha=0.5,
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc",
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",
    )


def _lgbm(scale_pos_weight: float):
    from lightgbm import LGBMClassifier

    return LGBMClassifier(
        n_estimators=300,
        max_depth=6,
        num_leaves=31,
        learning_rate=0.035,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        min_child_samples=25,
        reg_alpha=0.4,
        reg_lambda=2.5,
        scale_pos_weight=scale_pos_weight,
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1,
    )


def _catboost(scale_pos_weight: float):
    from catboost import CatBoostClassifier

    return CatBoostClassifier(
        iterations=300,
        depth=5,
        learning_rate=0.05,
        l2_leaf_reg=4,
        auto_class_weights="Balanced",
        random_seed=config.RANDOM_STATE,
        verbose=0,
        thread_count=-1,
        allow_writing_files=False,
    )


def _mlp(scale_pos_weight: float):
    from sklearn.neural_network import MLPClassifier

    return MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        alpha=0.001,
        batch_size=256,
        learning_rate_init=0.002,
        max_iter=35,
        early_stopping=True,
        n_iter_no_change=5,
        random_state=config.RANDOM_STATE,
    )


def get_model_zoo(scale_pos_weight: float = 1.0) -> dict:
    """
    Baselines, gradient boosters, an MLP, and a soft-voting ensemble of
    XGBoost + CatBoost + MLP with fixed 0.40 / 0.35 / 0.25 weights.
    Optional libraries that are not installed are skipped.
    """
    zoo = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=config.RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=250,
            max_depth=12,
            min_samples_leaf=40,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    factories = {
        "XGBoost": _xgb,
        "LightGBM": _lgbm,
        "CatBoost": _catboost,
        "Tabular Neural Net (MLP)": _mlp,
    }
    for name, make in factories.items():
        try:
            zoo[name] = make(scale_pos_weight)
        except ImportError:
            print(f"  ! {name} unavailable (library not installed) - skipping")

    members = [
        (key, make, w)
        for key, name, make, w in [
            ("xgb", "XGBoost", _xgb, 0.40),
            ("cat", "CatBoost", _catboost, 0.35),
            ("mlp", "Tabular Neural Net (MLP)", _mlp, 0.25),
        ]
        if name in zoo
    ]
    if len(members) >= 2:
        from sklearn.ensemble import VotingClassifier

        total = sum(w for _, _, w in members)
        zoo["Soft-Voting Ensemble"] = VotingClassifier(
            estimators=[(key, make(scale_pos_weight)) for key, make, _ in members],
            voting="soft",
            weights=[w / total for _, _, w in members],
        )

    return zoo


def _calibrate(fitted_pipe, X_cal, y_cal, method: str = "isotonic"):
    """
    Re-calibrate the probabilities of an already-fitted pipeline on held-out data.

    Why this matters: class_weight='balanced' and scale_pos_weight deliberately
    distort the class prior to improve recall on defaulters. That helps ranking,
    but it also inflates every predicted probability — a borrower with a true 8%
    default risk might come out at 0.35. If you then apply a 0.10 / 0.25 risk
    band cut-off, the entire portfolio lands in "High Risk" and your expected-loss
    numbers are nonsense.

    Calibration maps those distorted scores back onto real probabilities without
    touching the ranking (so ROC-AUC and KS are unchanged, while Brier improves).
    This is a genuinely good thing to be able to explain in the viva.
    """
    from sklearn.calibration import CalibratedClassifierCV

    try:  # scikit-learn >= 1.6
        from sklearn.frozen import FrozenEstimator

        calibrated = CalibratedClassifierCV(FrozenEstimator(fitted_pipe), method=method)
    except ImportError:  # scikit-learn < 1.6
        calibrated = CalibratedClassifierCV(fitted_pipe, cv="prefit", method=method)

    calibrated.fit(X_cal, y_cal)
    return calibrated


def evaluate(y_true, y_prob, threshold: float = 0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    total = max(tn + fp + fn + tp, 1)
    acc = (tp + tn) / total
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    bal_acc = 0.5 * (sens + spec)

    return {
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
        "pr_auc": round(float(average_precision_score(y_true, y_prob)), 4),
        "ks": round(ks_statistic(y_true, y_prob), 4),
        "brier": round(float(brier_score_loss(y_true, y_prob)), 4),
        "accuracy": round(float(acc), 4),
        "balanced_accuracy": round(float(bal_acc), 4),
        "recall_default": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "precision_default": round(
            float(precision_score(y_true, y_pred, zero_division=0)), 4
        ),
        "f1_default": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "threshold": threshold,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "actual_default_rate": round(float(np.mean(y_true)), 4),
        "predicted_default_rate": round(float(np.mean(y_pred)), 4),
    }


def find_best_threshold(y_true, y_prob, metric: str = "f1") -> float:
    """
    0.5 is arbitrary for an imbalanced problem. Sweep for the cut-off that
    maximises F1 (balanced) or Youden's J (equal weight to sensitivity and
    specificity). Report both the default and the tuned threshold.
    """
    grid = np.arange(0.05, 0.95, 0.01)
    if metric == "youden":
        fpr, tpr, thr = roc_curve(y_true, y_prob)
        return float(thr[np.argmax(tpr - fpr)])
    scores = [f1_score(y_true, (y_prob >= t).astype(int), zero_division=0) for t in grid]
    return float(grid[int(np.argmax(scores))])


def train_and_compare(
    X_train,
    y_train,
    X_cal,
    y_cal,
    X_test,
    y_test,
    numeric,
    categorical,
    verbose: bool = True,
    calibrate: bool = True,
):
    """
    Fit every model in the zoo inside its own preprocessing pipeline.

    X_train/y_train: Fitting partition (earliest loans).
    X_cal/y_cal: Calibration partition (next chronological slice).
                 Isotonic calibration and decision threshold tuning are fit here.
    X_test/y_test: Out-of-time evaluation partition (latest loans).
    """
    pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))

    X_fit, y_fit = X_train, y_train
    if X_cal is None or len(X_cal) == 0:
        calibrate = False

    if verbose:
        print("\n" + "=" * 62)
        print("MODEL TRAINING")
        print("=" * 62)
        print(f"  fit={len(X_fit):,}  "
              f"calibration={len(X_cal) if X_cal is not None else 0:,}  test={len(X_test):,}")
        print(f"  train default rate={y_train.mean():.2%}  "
              f"scale_pos_weight={pos_weight:.2f}")
        print(f"  probability calibration: {'isotonic' if calibrate else 'off'}")

    results, fitted = {}, {}

    for name, estimator in get_model_zoo(pos_weight).items():
        print(f"\n  -> {name}")
        t0 = time.time()

        pipe = Pipeline(
            [
                ("prep", build_preprocessor(numeric, categorical)),
                ("model", estimator),
            ]
        )
        pipe.fit(X_fit, y_fit)

        raw_brier = None
        if calibrate and X_cal is not None and len(X_cal) > 0:
            raw_brier = brier_score_loss(y_test, pipe.predict_proba(X_test)[:, 1])
            pipe = _calibrate(pipe, X_cal, y_cal)
            cal_brier = brier_score_loss(y_test, pipe.predict_proba(X_test)[:, 1])
            print(f"     calibrated: Brier {raw_brier:.4f} -> {cal_brier:.4f}")

        elapsed = time.time() - t0

        prob_test = pipe.predict_proba(X_test)[:, 1]
        prob_train = pipe.predict_proba(X_fit)[:, 1]

        metrics = evaluate(y_test, prob_test)
        metrics["train_roc_auc"] = round(float(roc_auc_score(y_fit, prob_train)), 4)
        metrics["overfit_gap"] = round(metrics["train_roc_auc"] - metrics["roc_auc"], 4)
        metrics["fit_seconds"] = round(elapsed, 1)
        if raw_brier is not None:
            metrics["brier_uncalibrated"] = round(float(raw_brier), 4)

        # STEP 2 FIX: Tune threshold on CALIBRATION set (y_cal), never on y_test!
        if y_cal is not None and len(y_cal) > 0:
            prob_cal = pipe.predict_proba(X_cal)[:, 1]
            tuned = find_best_threshold(y_cal, prob_cal)
            # Champion selection uses this. Isotonic calibration was fit on the
            # same window, which does not move AUC (it is a monotone map).
            metrics["cal_roc_auc"] = round(float(roc_auc_score(y_cal, prob_cal)), 4)
        else:
            tuned = 0.5
        metrics["tuned"] = evaluate(y_test, prob_test, tuned)
        metrics["tuned_threshold"] = round(float(tuned), 4)

        # STEP 3: Bootstrap 95% Confidence Interval for ROC-AUC (1,000 resamples)
        ci_low, ci_high = bootstrap_auc_ci(y_test, prob_test, n_bootstraps=1000)
        metrics["roc_auc_ci_lower"] = ci_low
        metrics["roc_auc_ci_upper"] = ci_high
        metrics["roc_auc_ci"] = f"[{ci_low:.4f}, {ci_high:.4f}]"

        results[name] = metrics
        fitted[name] = pipe

        print(
            f"     ROC-AUC {metrics['roc_auc']:.4f} (95% CI {metrics['roc_auc_ci']}) "
            f"| PR-AUC {metrics['pr_auc']:.4f} | KS {metrics['ks']:.4f} "
            f"| recall {metrics['tuned']['recall_default']:.3f} @ thr {tuned:.2f} | {elapsed:.1f}s"
        )
        if metrics["overfit_gap"] > 0.05:
            print(
                f"     ! train/test AUC gap {metrics['overfit_gap']:.3f} "
                "- this model is overfitting"
            )
        if metrics["roc_auc"] > 0.90:
            print(
                "     !! AUC above 0.90 on this dataset almost always means "
                "leakage. Check your feature list against config.LEAKAGE_COLS."
            )

    return results, fitted


def pick_best(results: dict, key: str = "cal_roc_auc") -> str:
    """
    Champion = best calibration-window AUC among models the API can explain.
    The test window is only ever used to report, never to choose.
    """
    eligible = [n for n in results if n in getattr(config, "CHAMPION_ELIGIBLE", results)]
    pool = eligible or list(results)
    return max(pool, key=lambda name: results[name].get(key, results[name]["roc_auc"]))


def summary_table(results: dict) -> pd.DataFrame:
    """
    Threshold-free metrics (ROC-AUC, PR-AUC, KS, Brier) plus recall/precision/F1
    measured at each model's tuned operating point.

    Recall at a fixed 0.5 cut-off is reported nowhere, on purpose. Once
    probabilities are properly calibrated on a portfolio that defaults 20% of
    the time, almost nothing scores above 0.5, so a 0.5 threshold flags nobody
    and recall collapses to ~0.02. That is an artefact of the cut-off, not of
    the model. Choosing the operating point is a business decision about how
    many good borrowers you are willing to reject to catch one bad one.
    """
    rows = []
    for name, m in results.items():
        t = m.get("tuned", m)
        rows.append(
            {
                "Model": name,
                "ROC-AUC": m["roc_auc"],
                "95% CI": m.get("roc_auc_ci", ""),
                "KS": m["ks"],
                "PR-AUC": m["pr_auc"],
                "Accuracy": t.get("accuracy", 0.0),
                "Balanced Acc": t.get("balanced_accuracy", 0.0),
                "Brier": m["brier"],
                "Threshold": round(t["threshold"], 2),
                "Recall (default)": t["recall_default"],
                "Precision (default)": t["precision_default"],
                "F1 (default)": t["f1_default"],
                "Train AUC": m["train_roc_auc"],
                "Overfit gap": m["overfit_gap"],
                "Fit (s)": m["fit_seconds"],
            }
        )
    return pd.DataFrame(rows).sort_values("ROC-AUC", ascending=False)


def unwrap_pipeline(estimator):
    """
    Dig the raw sklearn Pipeline out of a CalibratedClassifierCV (and out of the
    FrozenEstimator inside it on scikit-learn >= 1.6). Returns the estimator
    unchanged if it is already a Pipeline.
    """
    if isinstance(estimator, Pipeline):
        return estimator

    inner = getattr(estimator, "estimator", None)
    if inner is None:
        cals = getattr(estimator, "calibrated_classifiers_", None)
        if cals:
            inner = getattr(cals[0], "estimator", None)

    # FrozenEstimator wraps the real thing one more level down
    while inner is not None and not isinstance(inner, Pipeline):
        nxt = getattr(inner, "estimator", None)
        if nxt is None:
            break
        inner = nxt

    return inner if isinstance(inner, Pipeline) else estimator


def model_feature_columns(estimator) -> list:
    """The exact column names the fitted pipeline expects as input."""
    pipe = unwrap_pipeline(estimator)
    prep = pipe.named_steps.get("prep") if hasattr(pipe, "named_steps") else None
    if prep is None:
        return []
    cols = []
    for _, _, columns in prep.transformers_:
        if isinstance(columns, (list, tuple)):
            cols.extend(columns)
    return cols


def feature_importance(estimator, top_n: int = 25) -> pd.DataFrame:
    """Works for all model types, calibrated or not, including voting ensembles."""
    pipe = unwrap_pipeline(estimator)
    if not hasattr(pipe, "named_steps"):
        return pd.DataFrame()
    prep = pipe.named_steps["prep"]
    model = pipe.named_steps["model"]

    try:
        names = list(prep.get_feature_names_out())
    except Exception:
        names = [f"f{i}" for i in range(getattr(model, "n_features_in_", 0))]

    if hasattr(model, "feature_importances_"):
        vals = model.feature_importances_
    elif hasattr(model, "coef_"):
        vals = np.abs(model.coef_.ravel())
    elif hasattr(model, "estimators_"):
        # Aggregate sub-estimator importances for VotingClassifier
        sub_imps = []
        for est in model.estimators_:
            if hasattr(est, "feature_importances_"):
                sub_imps.append(est.feature_importances_)
            elif hasattr(est, "coef_"):
                sub_imps.append(np.abs(est.coef_.ravel()))
            elif hasattr(est, "coefs_"):
                sub_imps.append(np.mean(np.abs(est.coefs_[0]), axis=1))
        if sub_imps:
            vals = np.mean(sub_imps, axis=0)
        else:
            return pd.DataFrame()
    else:
        return pd.DataFrame()

    n = min(len(names), len(vals))
    return (
        pd.DataFrame({"feature": names[:n], "importance": vals[:n]})
        .sort_values("importance", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def save(pipe, results, best_name, numeric, categorical, extra: dict | None = None):
    """
    Everything the scoring API needs travels in one bundle, so the API never
    hardcodes a threshold, benchmark or model type.
    """
    bundle = {
        "pipeline": pipe,
        "model_name": best_name,
        "numeric_features": numeric,
        "categorical_features": categorical,
        "metrics": results[best_name],
        "risk_thresholds": dict(config.RISK_THRESHOLDS),
        "lgd": config.LGD,
        "trained_at": pd.Timestamp.now().isoformat(),
        **(extra or {}),
    }
    joblib.dump(bundle, config.MODEL_PATH)
    with open(config.METRICS_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  saved model  -> {config.MODEL_PATH}")
    print(f"  saved metrics-> {config.METRICS_PATH}")


def load():
    if not Path(config.MODEL_PATH).exists():
        raise FileNotFoundError(
            f"No trained model at {config.MODEL_PATH}. Run: python run_pipeline.py"
        )
    return joblib.load(config.MODEL_PATH)
