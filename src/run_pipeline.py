"""
End-to-end pipeline: raw file in, trained model and every evaluation artifact out.

    python run_pipeline.py                          # uses config.RAW_DATA_PATH
    python run_pipeline.py --data data/sample_loans.csv
    python run_pipeline.py --nrows 200000           # fast development run
    python run_pipeline.py --with-lc-signals        # add grade / int_rate (ablation)

Every file in artifacts/ is written by this script from one run. Nothing in
there should be edited by hand; rerun the pipeline instead.

  LAYER 1  collect -> assess -> clean -> re-assess
  LAYER 2  engineer -> split -> train -> select champion -> band
  LAYER 3  explain, monitor (PSI), fairness audit, save
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import config
from src import cleaning, explain, fairness, features, models, quality, risk
from src.data_loader import load_raw

PSI_FEATURES = [
    "dti", "annual_inc_log", "revol_bal", "acc_open_past_24mths",
    "credit_history_months", "bc_util", "fico_avg", "credit_utilisation",
    "loan_to_income", "revol_util",
]


def banner(text: str) -> None:
    print("\n" + "=" * 62)
    print(text)
    print("=" * 62)


def _period(df: pd.DataFrame) -> dict:
    return {
        "n_loans": int(len(df)),
        "start": f"{df['issue_d'].min():%Y-%m}",
        "end": f"{df['issue_d'].max():%Y-%m}",
        "default_rate": round(float(df[config.TARGET].mean()), 4),
    }


def _save_json(obj, name: str) -> None:
    path = config.ARTIFACT_DIR / name
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    print(f"  saved {path.relative_to(ROOT)}")


def _importance_plot(imp: pd.DataFrame, model_name: str) -> None:
    top = imp.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top["feature"], top["importance"], color="#3b6ea5")
    ax.set_xlabel("Mean |contribution| to log-odds of default")
    ax.set_title(f"Top drivers of default risk - {model_name} (test window)")
    fig.tight_layout()
    out = config.ARTIFACT_DIR / "figures" / "shap_summary.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"  saved {out.relative_to(ROOT)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="path to the raw CSV")
    ap.add_argument("--nrows", type=int, default=None, help="row limit for dev runs")
    ap.add_argument("--with-lc-signals", action="store_true",
                    help="include grade / sub_grade / int_rate / installment (ablation run)")
    ap.add_argument("--skip-save", action="store_true")
    args = ap.parse_args()

    config.EXCLUDE_LC_RISK_SIGNALS = not args.with_lc_signals
    if args.with_lc_signals:
        print("  ! Lending Club's own pricing signals INCLUDED (ablation run)")

    t_start = time.time()

    # ================= LAYER 1: DATA PREPARATION =====================
    banner("LAYER 1 - DATA PREPARATION")

    print("\n[1/5] Collection")
    raw = load_raw(path=args.data, nrows=args.nrows)

    print("\n[2/5] Quality assessment (BEFORE cleaning)")
    q_before = quality.assess(raw)

    print("\n[3/5] Duplicate detection")
    dupes = quality.detect_duplicates(raw)
    print(f"  exact={dupes['exact_duplicates']:,}  "
          f"id={dupes['id_duplicates']:,}  near={dupes['near_duplicates']:,}")

    print("\n[4/5] Cleaning")
    clean_df, cleaning_log = cleaning.clean(raw)
    del raw

    print("\n[5/5] Quality assessment (AFTER cleaning)")
    q_after = quality.assess(clean_df)

    # ================= LAYER 2: DATA SCIENCE / ML ====================
    banner("LAYER 2 - DATA SCIENCE / ML")

    print("\n[1/4] Feature engineering")
    feat_df = features.build_features(clean_df)

    print("\n[2/4] Train / calibration / test split (strict month boundaries)")
    train_df, cal_df, test_df = features.time_split(feat_df, return_cal=True)

    X_train, y_train, num, cat = features.prepare_xy(train_df)
    X_cal, y_cal, _, _ = features.prepare_xy(cal_df)
    X_test, y_test, _, _ = features.prepare_xy(test_df)
    print(f"  {len(num)} numeric + {len(cat)} categorical features")

    print("\n[3/4] Train & compare models")
    results, fitted = models.train_and_compare(
        X_train, y_train, X_cal, y_cal, X_test, y_test, num, cat
    )

    banner("MODEL COMPARISON (test window - reporting only)")
    table = models.summary_table(results)
    print(table.to_string(index=False))

    best_name = models.pick_best(results)
    best_pipe = fitted[best_name]
    best_test = max(results, key=lambda n: results[n]["roc_auc"])
    print(f"\n  CHAMPION: {best_name}  (selected on calibration-window AUC "
          f"{results[best_name].get('cal_roc_auc')})")
    print(f"  test ROC-AUC {results[best_name]['roc_auc']:.4f} "
          f"{results[best_name]['roc_auc_ci']}, KS {results[best_name]['ks']:.4f}")
    if best_test != best_name:
        print(f"  (highest test AUC was {best_test} at {results[best_test]['roc_auc']:.4f};"
              " not used for selection)")

    print("\n[4/4] Risk banding")
    scored = risk.score_dataframe(test_df, best_pipe)
    summary = risk.portfolio_summary(scored)
    print(summary.to_string())
    gains = risk.gains_table(scored)
    zones = risk.decision_zone_default_rates(scored)
    print(f"\n  Riskiest decile defaults {gains.iloc[0]['lift']:.2f}x the portfolio average.")

    # ================= LAYER 3: EXPLAIN / MONITOR / AUDIT ============
    banner("LAYER 3 - EXPLAINABILITY, STABILITY, FAIRNESS")

    explainer = explain.Explainer(best_pipe, num, cat,
                                  background=X_train.sample(min(5000, len(X_train)),
                                                            random_state=config.RANDOM_STATE))
    imp_sample = X_test.sample(min(5000, len(X_test)), random_state=config.RANDOM_STATE)
    imp = explain.global_importance(explainer, imp_sample)
    print("  Top drivers (mean |contribution|):")
    for _, r in imp.head(10).iterrows():
        print(f"    {r['feature']:<30} {r['importance']:.4f}")

    prob_train = best_pipe.predict_proba(X_train)[:, 1]
    psi = {
        "reference": "train window", "monitored": "test window",
        "credit_score_psi": risk.calculate_psi(risk.probability_to_score(prob_train),
                                               scored["credit_score"]),
        "default_probability_psi": risk.calculate_psi(prob_train, scored["default_probability"]),
        "features_psi": {f: risk.calculate_psi(X_train[f], X_test[f])
                         for f in PSI_FEATURES if f in X_train.columns},
    }
    print(f"  score PSI {psi['credit_score_psi']['psi']:.4f} ({psi['credit_score_psi']['status']})")

    fair = fairness.audit(scored)
    fair_summary = fairness.summarise(fair)
    for seg, s in fair_summary.items():
        flag = "ok" if not s["groups_failing_four_fifths"] else f"below 0.80: {s['groups_failing_four_fifths']}"
        print(f"  fairness - {seg:<20} min AIR {s['min_adverse_impact_ratio']:.2f}  {flag}")

    # Applicant benchmarks shown in the dashboard: medians of the training book
    benchmarks = {
        "fico_avg": float(train_df["fico_avg"].median()),
        "dti": float(train_df["dti"].median()),
        "annual_inc": float(train_df["annual_inc"].median()),
        "revol_util": float(train_df["revol_util"].median()),
        "avg_pd": round(float(scored["default_probability"].mean()), 4),
        "tau_low": config.RISK_THRESHOLDS["low_max"],
        "tau_high": config.RISK_THRESHOLDS["medium_max"],
    }

    if args.skip_save:
        banner(f"DONE in {time.time() - t_start:.1f}s (nothing saved)")
        return

    banner("SAVING ARTIFACTS")
    run_info = {
        "trained_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "data_file": str(Path(args.data or config.RAW_DATA_PATH).name),
        "nrows_limit": args.nrows,
        "lc_signals_included": not config.EXCLUDE_LC_RISK_SIGNALS,
        "champion": best_name,
        "champion_selection": "highest calibration-window ROC-AUC among CHAMPION_ELIGIBLE",
        "periods": {"train": _period(train_df), "calibration": _period(cal_df),
                    "test": _period(test_df)},
        "features": {"numeric": num, "categorical": cat},
        "risk_thresholds": config.RISK_THRESHOLDS,
        "decision_zones_test": zones,
        "python": platform.python_version(),
    }
    models.save(best_pipe, results, best_name, num, cat, extra={
        "benchmarks": benchmarks,
        "explain_background": X_train.sample(min(2000, len(X_train)),
                                             random_state=config.RANDOM_STATE),
        "run_info": run_info,
    })
    _save_json(run_info, "run_info.json")
    _save_json(benchmarks, "underwriting_benchmarks.json")
    _save_json(psi, "psi.json")
    _save_json(fair_summary, "fairness_summary.json")
    _save_json(cleaning_log, "cleaning_log.json")
    for name, rep, path in [
        ("before", q_before, config.QUALITY_BEFORE_PATH),
        ("after", q_after, config.QUALITY_AFTER_PATH),
    ]:
        quality.save_report(rep, path)
        _save_json({k: v for k, v in rep.items() if k != "profile"},
                   f"quality_summary_{name}.json")

    imp.to_csv(config.ARTIFACT_DIR / "feature_importance.csv", index=False)
    table.to_csv(config.ARTIFACT_DIR / "model_comparison.csv", index=False)
    summary.to_csv(config.ARTIFACT_DIR / "portfolio_summary.csv")
    gains.to_csv(config.ARTIFACT_DIR / "gains_table.csv", index=False)
    fair.to_csv(config.ARTIFACT_DIR / "fairness_report.csv", index=False)
    _importance_plot(imp, best_name)

    keep = [c for c in scored.columns if c != "data_source"]
    scored[keep].sample(min(20_000, len(scored)), random_state=config.RANDOM_STATE) \
        .to_csv(config.SCORED_SAMPLE_PATH, index=False)

    try:
        if "id" in feat_df.columns:
            feat_df["id"] = feat_df["id"].astype(str)
        feat_df.to_parquet(config.CLEAN_DATA_PATH, index=False)
    except Exception as e:
        print(f"  parquet unavailable ({e}); clean dataset not saved")

    banner(f"DONE in {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
