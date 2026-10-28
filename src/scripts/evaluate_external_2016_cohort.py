"""
Evaluation Script for the External 2016 Cohort (LendingClub 2007-2019Q3 Dataset).

Purpose:
  Validates the champion underwriting model on an uncensored out-of-time cohort:
  - 36-month loans issued January-June 2016.
  - Final resolved status in the newer dataset ('Fully Paid' or 'Charged Off'/'Default').
  - Calculates ROC-AUC with 95% Bootstrap Confidence Interval.
  - Calculates actual default rates across underwriting decision zones:
      * Approve Zone:  PD <= 10%
      * Review Zone:   10% < PD <= 20%
      * Decline Zone:  PD > 20%
  - Compares external validation results directly against the main test set benchmarks
    (Apr-Sep 2015 strict chronological holdout, 136,712 loans).

Dataset Source:
  Kaggle: denychaen/lending-club-loans-rejects-data
  Place downloaded file(s) into: data/external/
"""

import argparse
import io
import json
import os
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

# Add project root to sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from src import cleaning, features, risk

EXTERNAL_DIR = ROOT / "data" / "external"
ARTIFACTS_DIR = ROOT / "artifacts"
RESULTS_JSON_PATH = ARTIFACTS_DIR / "external_2016_validation_results.json"
RESULTS_CSV_PATH = ARTIFACTS_DIR / "external_2016_decision_zones.csv"

# Target outcome mapping
RESOLVED_TARGET_MAP = {
    "Fully Paid": 0,
    "Does not meet the credit policy. Status:Fully Paid": 0,
    "Charged Off": 1,
    "Default": 1,
    "Does not meet the credit policy. Status:Charged Off": 1,
}

# 2016 Jan-Jun issue months
JAN_JUN_2016_MONTHS = {
    "Jan-2016",
    "Feb-2016",
    "Mar-2016",
    "Apr-2016",
    "May-2016",
    "Jun-2016",
    "2016-01-01",
    "2016-02-01",
    "2016-03-01",
    "2016-04-01",
    "2016-05-01",
    "2016-06-01",
}


def find_external_dataset(search_dir: Path) -> Optional[Path]:
    """Auto-detect the accepted loans dataset in data/external/."""
    if not search_dir.exists():
        return None

    # Priority file candidates
    candidates = []
    for ext in ["*.csv", "*.csv.gz", "*.zip", "*.parquet"]:
        candidates.extend(list(search_dir.glob(ext)))
        candidates.extend(list(search_dir.glob(f"**/{ext}")))

    # Look for files matching 'accepted' or 'loan'
    accepted_candidates = [
        f for f in candidates if "accepted" in f.name.lower() or "loan" in f.name.lower()
    ]
    if accepted_candidates:
        return accepted_candidates[0]

    if candidates:
        return candidates[0]

    return None


def extract_2016_cohort_from_file(
    file_path: Path, chunk_size: int = 100_000, is_test_run: bool = False
) -> pd.DataFrame:
    """
    Stream and filter 36-month loans issued Jan-Jun 2016 with resolved outcomes.
    Uses chunked reading to handle multi-gigabyte CSVs safely.
    """
    prefix = f"[TEST RUN ON {file_path.name} - NOT VALID RESULTS] " if is_test_run else ""
    print(f"\n{prefix}[Data Ingestion] Reading and filtering dataset: {file_path.name}")
    print(f"{prefix}  Filter criteria:")
    print(f"{prefix}    - Issue Date: January 2016 - June 2016 inclusive")
    print(f"{prefix}    - Term: 36 months")
    print(f"{prefix}    - Status: Fully Paid (0) or Charged Off / Default (1)")

    matched_dfs = []
    total_scanned = 0

    # Determine reader format
    is_zip = file_path.suffix.lower() == ".zip"

    if is_zip:
        with zipfile.ZipFile(file_path, "r") as z:
            csv_names = [n for n in z.namelist() if n.lower().endswith(".csv")]
            if not csv_names:
                raise ValueError(f"No CSV files found inside {file_path.name}")
            target_csv = csv_names[0]
            for n in csv_names:
                if "accepted" in n.lower() or "loan" in n.lower():
                    target_csv = n
                    break
            print(f"{prefix}  Extracting from zip member: {target_csv}")
            with z.open(target_csv) as f:
                reader = pd.read_csv(
                    f, chunksize=chunk_size, low_memory=False, encoding="utf-8"
                )
                for chunk in reader:
                    total_scanned += len(chunk)
                    filtered = filter_cohort_chunk(chunk)
                    if len(filtered) > 0:
                        matched_dfs.append(filtered)
                    print(
                        f"\r{prefix}  Scanned: {total_scanned:,} rows | Matched 2016 Jan-Jun loans: {sum(len(d) for d in matched_dfs):,}",
                        end="",
                        flush=True,
                    )
    else:
        reader = pd.read_csv(
            file_path, chunksize=chunk_size, low_memory=False, encoding="utf-8"
        )
        for chunk in reader:
            total_scanned += len(chunk)
            filtered = filter_cohort_chunk(chunk)
            if len(filtered) > 0:
                matched_dfs.append(filtered)
            print(
                f"\r{prefix}  Scanned: {total_scanned:,} rows | Matched 2016 Jan-Jun loans: {sum(len(d) for d in matched_dfs):,}",
                end="",
                flush=True,
            )

    print()  # newline
    if not matched_dfs:
        raise ValueError(
            f"No matching 36-month loans issued Jan-Jun 2016 with resolved outcomes found in {file_path.name}"
        )

    full_cohort = pd.concat(matched_dfs, ignore_index=True)
    print(f"{prefix}[Data Ingestion] Successfully extracted {len(full_cohort):,} loans.")
    return full_cohort


def filter_cohort_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """Filter an individual DataFrame chunk for 36-month loans issued Jan-Jun 2016."""
    if "loan_status" not in chunk.columns or "issue_d" not in chunk.columns:
        return pd.DataFrame()

    # Drop unissued / empty rows
    valid_mask = chunk["loan_status"].notna() & chunk["issue_d"].notna()
    c = chunk[valid_mask]
    if len(c) == 0:
        return pd.DataFrame()

    # Filter term (36 months)
    term_mask = c["term"].astype(str).str.contains("36", regex=False, na=False)

    # Filter issue date (Jan-Jun 2016)
    issue_series = c["issue_d"].astype(str).str.strip()
    issue_mask = issue_series.isin(JAN_JUN_2016_MONTHS)
    if not issue_mask.any():
        # Fallback to date parsing for non-standard formats
        try:
            parsed_dates = pd.to_datetime(issue_series, errors="coerce")
            issue_mask = (parsed_dates.dt.year == 2016) & (
                parsed_dates.dt.month.between(1, 6)
            )
        except Exception:
            pass

    # Filter resolved statuses only
    status_mask = c["loan_status"].isin(RESOLVED_TARGET_MAP)

    match = c[term_mask & issue_mask & status_mask].copy()
    if len(match) > 0:
        match["actual_default"] = match["loan_status"].map(RESOLVED_TARGET_MAP)

    return match


def compute_roc_auc_bootstrap_ci(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    n_bootstraps: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float, float]:
    """
    Compute ROC-AUC with non-parametric bootstrap confidence interval.
    Returns: (point_estimate, ci_lower, ci_upper, standard_error)
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    point_est = float(roc_auc_score(y_true, y_probs))

    boot_aucs = []
    for _ in range(n_bootstraps):
        idx = rng.choice(n, size=n, replace=True)
        y_b = y_true[idx]
        p_b = y_probs[idx]
        if len(np.unique(y_b)) == 2:
            boot_aucs.append(roc_auc_score(y_b, p_b))

    boot_aucs = np.array(boot_aucs)
    alpha = (1.0 - confidence_level) / 2.0
    ci_lower = float(np.percentile(boot_aucs, alpha * 100))
    ci_upper = float(np.percentile(boot_aucs, (1.0 - alpha) * 100))
    std_err = float(np.std(boot_aucs))

    return point_est, ci_lower, ci_upper, std_err


def get_current_model_main_test_benchmarks() -> Dict:
    """
    Returns exact verified main test set metrics for the current champion model.
    Main Test Set: April-September 2015 (Strict Chronological Holdout, N = 136,712 loans).
    """
    # Baseline verified metrics from current champion model on the 136,712 Apr-Sep 2015 test set
    bench = {
        "period": "Apr-Sep 2015 (Strict Chronological Holdout)",
        "n_loans": 136712,
        "roc_auc": 0.6851,
        "brier_score": 0.1200,
        "actual_default_rate": 0.1494,
        "zones": {
            "APPROVE": {
                "label": "Approve Zone (PD <= 10%)",
                "actual_default_rate": 0.0705,
                "pct_of_portfolio": 39.56,
                "avg_pred_pd": 0.0680,
                "n_loans": 54084,
            },
            "REVIEW": {
                "label": "Review Zone (10-20%)",
                "actual_default_rate": 0.1525,
                "pct_of_portfolio": 32.79,
                "avg_pred_pd": 0.1491,
                "n_loans": 44832,
            },
            "DECLINE": {
                "label": "Decline Zone (PD > 20%)",
                "actual_default_rate": 0.2587,
                "pct_of_portfolio": 27.65,
                "avg_pred_pd": 0.2524,
                "n_loans": 37796,
            },
        },
    }

    # Verify/read from portfolio_summary.csv if available
    port_csv = ARTIFACTS_DIR / "portfolio_summary.csv"
    if port_csv.exists():
        try:
            pdf = pd.read_csv(port_csv)
            for _, row in pdf.iterrows():
                band = str(row["risk_band"]).lower()
                if "low" in band:
                    bench["zones"]["APPROVE"]["actual_default_rate"] = float(row["actual_default_rate"])
                    bench["zones"]["APPROVE"]["pct_of_portfolio"] = float(row["pct_of_portfolio"])
                    bench["zones"]["APPROVE"]["avg_pred_pd"] = float(row["avg_predicted_pd"])
                elif "medium" in band:
                    bench["zones"]["REVIEW"]["actual_default_rate"] = float(row["actual_default_rate"])
                    bench["zones"]["REVIEW"]["pct_of_portfolio"] = float(row["pct_of_portfolio"])
                    bench["zones"]["REVIEW"]["avg_pred_pd"] = float(row["avg_predicted_pd"])
                elif "high" in band:
                    bench["zones"]["DECLINE"]["actual_default_rate"] = float(row["actual_default_rate"])
                    bench["zones"]["DECLINE"]["pct_of_portfolio"] = float(row["pct_of_portfolio"])
                    bench["zones"]["DECLINE"]["avg_pred_pd"] = float(row["avg_predicted_pd"])
        except Exception:
            pass

    return bench


def evaluate_cohort(
    cohort_df: pd.DataFrame,
    source_filename: str,
    is_test_run: bool = False,
    model_artifact_path: Path = ARTIFACTS_DIR / "best_model.joblib",
) -> Dict:
    """
    Score the cohort with the champion model and generate evaluation metrics.
    """
    prefix = f"[TEST RUN ON {source_filename} - NOT VALID RESULTS] " if is_test_run else ""
    print(f"\n{prefix}[Model Inference] Loading champion model from: {model_artifact_path.name}")
    if not model_artifact_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_artifact_path}")

    model_dict = joblib.load(model_artifact_path)
    pipeline = model_dict["pipeline"]
    numeric_features = model_dict["numeric_features"]
    categorical_features = model_dict["categorical_features"]

    # Preprocess cohort data
    print(f"{prefix}  Transforming features via cleaning & feature engineering pipelines...")
    df_clean, _ = cleaning.clean(cohort_df, verbose=False, is_training=False)
    df_feat = features.build_features(df_clean, verbose=False)

    feature_cols = numeric_features + categorical_features
    X = df_feat[feature_cols]
    y_true = cohort_df["actual_default"].values.astype(int)

    # Predict calibrated default probabilities
    print(f"{prefix}  Scoring default probabilities (predict_proba)...")
    probs = pipeline.predict_proba(X)[:, 1]

    # Overall Metrics
    n_loans = len(y_true)
    n_defaults = int(y_true.sum())
    overall_default_rate = float(n_defaults / n_loans) if n_loans > 0 else 0.0
    mean_pred_pd = float(probs.mean()) if n_loans > 0 else 0.0
    brier = float(brier_score_loss(y_true, probs)) if n_loans > 0 else 0.0

    # Bootstrap ROC-AUC CI
    print(f"{prefix}  Computing ROC-AUC and 95% Bootstrap Confidence Interval (1,000 resamples)...")
    auc_est, auc_ci_low, auc_ci_high, auc_se = compute_roc_auc_bootstrap_ci(
        y_true, probs, n_bootstraps=1000, seed=config.RANDOM_STATE
    )

    # Decision Zone Analysis
    tau_low = config.RISK_THRESHOLDS["low_max"]
    tau_high = config.RISK_THRESHOLDS["medium_max"]

    approve_mask = probs <= tau_low
    review_mask = (probs > tau_low) & (probs <= tau_high)
    decline_mask = probs > tau_high

    zones_data = []
    zone_masks = [
        ("Approve (Low Risk: PD <= 10%)", approve_mask, "APPROVE"),
        ("Review (Medium Risk: 10% < PD <= 20%)", review_mask, "REVIEW"),
        ("Decline (High Risk: PD > 20%)", decline_mask, "DECLINE"),
    ]

    loan_amnts = pd.to_numeric(cohort_df.get("loan_amnt", 0.0), errors="coerce").fillna(0.0).values

    for zone_name, mask, code in zone_masks:
        z_n = int(mask.sum())
        z_defaults = int(y_true[mask].sum()) if z_n > 0 else 0
        z_def_rate = float(z_defaults / z_n) if z_n > 0 else 0.0
        z_avg_pred = float(probs[mask].mean()) if z_n > 0 else 0.0
        z_pct = float(z_n / n_loans * 100) if n_loans > 0 else 0.0
        z_exposure = float(loan_amnts[mask].sum()) if z_n > 0 else 0.0
        z_el = float(risk.expected_loss(probs[mask], loan_amnts[mask], lgd=0.60).sum()) if z_n > 0 else 0.0

        zones_data.append({
            "zone": zone_name,
            "code": code,
            "n_loans": z_n,
            "pct_of_portfolio": round(z_pct, 2),
            "n_defaults": z_defaults,
            "actual_default_rate": round(z_def_rate, 4),
            "actual_default_rate_pct": round(z_def_rate * 100, 2),
            "avg_predicted_pd": round(z_avg_pred, 4),
            "avg_predicted_pd_pct": round(z_avg_pred * 100, 2),
            "total_exposure": round(z_exposure, 2),
            "expected_loss": round(z_el, 2),
        })

    # Main test set benchmarks
    benchmarks = get_current_model_main_test_benchmarks()

    if is_test_run:
        scope_desc = f"TEST RUN DEMO: 36-month loans extracted from local test file '{source_filename}' (NOT Kaggle 2019Q3 dataset)"
        snapshot_desc = f"Local test file: {source_filename} (NOT verified 2007-2019Q3 dataset)"
        censoring_desc = "TEST RUN ONLY - DO NOT USE AS VALIDATION EVIDENCE"
    else:
        scope_desc = f"36-month loans issued Jan-Jun 2016 extracted from '{source_filename}' in data/external/"
        snapshot_desc = f"External dataset: {source_filename}"
        censoring_desc = "Uncensored 36-month cohort (matured by June 2019)"

    results = {
        "is_test_run": is_test_run,
        "source_file": source_filename,
        "cohort": {
            "description": scope_desc,
            "dataset_file": snapshot_desc,
            "censoring_status": censoring_desc,
            "total_loans": n_loans,
            "total_defaults": n_defaults,
            "actual_default_rate": round(overall_default_rate, 4),
            "actual_default_rate_pct": round(overall_default_rate * 100, 2),
            "mean_predicted_pd": round(mean_pred_pd, 4),
            "mean_predicted_pd_pct": round(mean_pred_pd * 100, 2),
        },
        "discrimination": {
            "roc_auc": round(auc_est, 4),
            "roc_auc_ci_95_lower": round(auc_ci_low, 4),
            "roc_auc_ci_95_upper": round(auc_ci_high, 4),
            "roc_auc_se": round(auc_se, 4),
            "brier_score": round(brier, 4),
        },
        "decision_zones": zones_data,
        "main_test_set_benchmarks": benchmarks,
        "comparison": generate_comparison(auc_est, overall_default_rate, zones_data, benchmarks),
    }

    return results


def generate_comparison(
    auc_est: float,
    cohort_default_rate: float,
    zones_data: List[Dict],
    benchmarks: Dict,
) -> Dict:
    """Generate delta comparison between the external cohort and the main test set."""
    main_auc = benchmarks.get("roc_auc", 0.6851)
    main_dr = benchmarks.get("actual_default_rate", 0.1494)

    delta_auc = auc_est - main_auc
    delta_dr = cohort_default_rate - main_dr

    zone_comp = {}
    for z in zones_data:
        code = z["code"]
        main_z = benchmarks.get("zones", {}).get(code, {})
        main_z_dr = main_z.get("actual_default_rate", 0.0)
        cohort_z_dr = z["actual_default_rate"]
        zone_comp[code] = {
            "main_test_actual_dr_pct": round(main_z_dr * 100, 2),
            "cohort_actual_dr_pct": round(cohort_z_dr * 100, 2),
            "delta_dr_pct": round((cohort_z_dr - main_z_dr) * 100, 2),
        }

    z_rates = [z["actual_default_rate"] for z in zones_data]
    is_monotonic = z_rates[0] < z_rates[1] < z_rates[2] if len(z_rates) == 3 else False

    return {
        "delta_roc_auc": round(delta_auc, 4),
        "delta_default_rate_pct": round(delta_dr * 100, 2),
        "zone_comparison": zone_comp,
        "monotonic_risk_gradient_preserved": is_monotonic,
    }


def print_formatted_report(results: Dict):
    """Print executive Markdown/ASCII report."""
    is_test_run = results.get("is_test_run", False)
    source_file = results.get("source_file", "unknown")
    tag = f"[TEST RUN ON {source_file} - NOT VALID RESULTS]" if is_test_run else ""

    cohort = results["cohort"]
    discrim = results["discrimination"]
    zones = results["decision_zones"]
    comp = results["comparison"]
    bench = results["main_test_set_benchmarks"]

    print("\n" + "=" * 84)
    if is_test_run:
        print(f"  {tag}")
        print(f"  WARNING: THIS IS A DRY RUN ON LOCAL FILE '{source_file}'.")
        print(f"  DO NOT USE THESE METRICS AS EMPIRICAL VALIDATION RESULTS.")
    else:
        print("  EXTERNAL VALIDATION REPORT: 2016 JAN-JUN COHORT (LENDINGCLUB DATASET)")
    print("=" * 84)
    if is_test_run:
        print(f"{tag} Scope:             {cohort['description']}")
        print(f"{tag} Source:            {cohort['dataset_file']}")
        print(f"{tag} Note:              {cohort['censoring_status']}")
        print(f"{tag} Sample Size (N):   {cohort['total_loans']:,} loans")
        print(f"{tag} Observed Defaults: {cohort['total_defaults']:,} ({cohort['actual_default_rate_pct']}%)")
        print(f"{tag} Model Mean PD:     {cohort['mean_predicted_pd_pct']}%")
    else:
        print(f"Cohort Scope:        {cohort['description']}")
        print(f"Source Dataset:      {cohort['dataset_file']}")
        print(f"Maturity Status:     {cohort['censoring_status']}")
        print(f"Sample Size (N):     {cohort['total_loans']:,} loans")
        print(f"Observed Defaults:   {cohort['total_defaults']:,} ({cohort['actual_default_rate_pct']}%)")
        print(f"Model Mean Pred PD:  {cohort['mean_predicted_pd_pct']}%")

    print("\n" + "-" * 84)
    if is_test_run:
        print(f"{tag} 1. DISCRIMINATION PERFORMANCE & 95% BOOTSTRAP CI")
    else:
        print("  1. DISCRIMINATION PERFORMANCE & 95% BOOTSTRAP CONFIDENCE INTERVAL")
    print("-" * 84)
    delta_auc_sign = "+" if comp['delta_roc_auc'] >= 0 else ""
    if is_test_run:
        print(f"{tag} Champion ROC-AUC:   {discrim['roc_auc']:.4f}")
        print(f"{tag} 95% Bootstrap CI:   [{discrim['roc_auc_ci_95_lower']:.4f}, {discrim['roc_auc_ci_95_upper']:.4f}]  (SE: {discrim['roc_auc_se']:.4f})")
        print(f"{tag} Main Test Set AUC:  {bench['roc_auc']:.4f}  ({bench['period']})")
        print(f"{tag} AUC Variance (Delta): {delta_auc_sign}{comp['delta_roc_auc']:.4f}")
        print(f"{tag} Brier Score:        {discrim['brier_score']:.4f}")
    else:
        print(f"  Champion ROC-AUC:   {discrim['roc_auc']:.4f}")
        print(f"  95% Bootstrap CI:   [{discrim['roc_auc_ci_95_lower']:.4f}, {discrim['roc_auc_ci_95_upper']:.4f}]  (SE: {discrim['roc_auc_se']:.4f})")
        print(f"  Main Test Set AUC:  {bench['roc_auc']:.4f}  ({bench['period']})")
        print(f"  AUC Difference:     {delta_auc_sign}{comp['delta_roc_auc']:.4f}")
        print(f"  Brier Score:        {discrim['brier_score']:.4f}")

    print("\n" + "-" * 84)
    if is_test_run:
        print(f"{tag} 2. ACTUAL DEFAULT RATE BY DECISION ZONE")
    else:
        print("  2. ACTUAL DEFAULT RATE BY DECISION ZONE (UNDERWRITING TIERS)")
    print("-" * 84)
    print(f"{'Decision Zone':<35} | {'N Loans':>8} | {'% Cohort':>8} | {'Pred PD':>8} | {'Actual DR':>10}")
    print("-" * 84)
    for z in zones:
        print(
            f"{z['zone']:<35} | {z['n_loans']:>8,d} | {z['pct_of_portfolio']:>7.1f}% | "
            f"{z['avg_predicted_pd_pct']:>7.2f}% | {z['actual_default_rate_pct']:>9.2f}%"
        )
    print("-" * 84)

    print("\n" + "-" * 84)
    if is_test_run:
        print(f"{tag} 3. COMPARISON VS MAIN TEST SET ({bench['period']})")
    else:
        print(f"  3. COMPARISON: FRESH 2016 COHORT VS MAIN TEST SET ({bench['period']})")
    print("-" * 84)
    print(f"{'Tier / Metric':<28} | {'Main Test Set':>15} | {'2016 Cohort':>17} | {'Variance (Delta)':>16}")
    print("-" * 84)
    print(
        f"{'ROC-AUC':<28} | {bench['roc_auc']:>15.4f} | {discrim['roc_auc']:>17.4f} | "
        f"{delta_auc_sign}{comp['delta_roc_auc']:>15.4f}"
    )
    main_dr_pct = bench['actual_default_rate'] * 100
    delta_dr_sign = "+" if comp['delta_default_rate_pct'] >= 0 else ""
    print(
        f"{'Overall Default Rate':<28} | {main_dr_pct:>14.2f}% | {cohort['actual_default_rate_pct']:>16.2f}% | "
        f"{delta_dr_sign}{comp['delta_default_rate_pct']:>14.2f}%"
    )

    zc = comp["zone_comparison"]
    for code, label in [("APPROVE", "Approve Zone (PD <= 10%)"), ("REVIEW", "Review Zone (10-20%)"), ("DECLINE", "Decline Zone (PD > 20%)")]:
        m_dr = zc[code]["main_test_actual_dr_pct"]
        c_dr = zc[code]["cohort_actual_dr_pct"]
        d_dr = zc[code]["delta_dr_pct"]
        d_sign = "+" if d_dr >= 0 else ""
        print(f"{label:<28} | {m_dr:>14.2f}% | {c_dr:>16.2f}% | {d_sign}{d_dr:>14.2f}%")

    print("-" * 84)
    monotonic_str = "PASSED (Strictly Monotonic)" if comp['monotonic_risk_gradient_preserved'] else "FAILED"
    if is_test_run:
        print(f"{tag} Risk Separation Check: {monotonic_str}")
        print(f"{tag} END OF TEST RUN. NO PRODUCTION ARTIFACTS WRITTEN.")
    else:
        print(f"Risk Separation Check: {monotonic_str}")
    print("=" * 84 + "\n")


def save_validation_artifacts(results: Dict):
    """Save results to JSON and CSV in artifacts/ only if NOT a test run."""
    if results.get("is_test_run", False):
        print(f"[Notice] Test run mode: Skipping persistence of production artifacts to artifacts/.")
        return

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(RESULTS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[Artifact Saved] Results JSON -> {RESULTS_JSON_PATH}")

    zones_df = pd.DataFrame(results["decision_zones"])
    zones_df.to_csv(RESULTS_CSV_PATH, index=False)
    print(f"[Artifact Saved] Decision Zones CSV -> {RESULTS_CSV_PATH}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate champion model on external 2016 Jan-Jun LendingClub cohort."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to external 2007-2019Q3 dataset (.csv, .csv.gz, or .zip). Defaults to searching data/external/.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100_000,
        help="Chunk size for streaming large dataset (default: 100,000).",
    )
    parser.add_argument(
        "--test-run",
        action="store_true",
        help="Run code test on available local sample data (clearly labeled as TEST RUN).",
    )
    args = parser.parse_args()

    # 1. Resolve dataset path
    data_path = None
    is_test_run = args.test_run

    if args.data_path:
        p = Path(args.data_path)
        if p.exists():
            data_path = p
        else:
            print(f"Error: Specified data path does not exist: {p}")
            sys.exit(1)
    else:
        data_path = find_external_dataset(EXTERNAL_DIR)

    # 2. Check if external dataset exists
    if data_path is None:
        if is_test_run:
            fallback = ROOT / "data" / "sample_loans.csv"
            if not fallback.exists():
                print(f"Error: Local test file {fallback} not found.")
                sys.exit(1)
            data_path = fallback
        else:
            print("\n" + "=" * 80)
            print("  EXTERNAL DATASET NOT FOUND: data/external/")
            print("=" * 80)
            print("No external dataset found in data/external/. Script stopped.")
            print("\nTo perform proper fresh validation on the 2016 Jan-Jun cohort:")
            print("1. Download the LendingClub 2007-2019Q3 dataset from Kaggle:")
            print("   URL: https://www.kaggle.com/datasets/denychaen/lending-club-loans-rejects-data")
            print("2. Place the downloaded archive or CSV into:")
            print(f"   {EXTERNAL_DIR.resolve()}")
            print("   (Accepted formats: .csv, .csv.gz, or .zip)")
            print("3. Re-run this evaluation script:")
            print("   python scripts/evaluate_external_2016_cohort.py")
            print("\nTip: To dry-run code logic on local test files, use:")
            print("   python scripts/evaluate_external_2016_cohort.py --test-run")
            print("=" * 80 + "\n")
            sys.exit(0)

    # 3. Extract and filter cohort
    try:
        cohort_df = extract_2016_cohort_from_file(
            data_path, chunk_size=args.chunk_size, is_test_run=is_test_run
        )
    except Exception as e:
        print(f"\nError during cohort extraction: {e}")
        sys.exit(1)

    # 4. Run evaluation
    try:
        results = evaluate_cohort(
            cohort_df, source_filename=data_path.name, is_test_run=is_test_run
        )
    except Exception as e:
        print(f"\nError during evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 5. Output report & save artifacts
    print_formatted_report(results)
    save_validation_artifacts(results)


if __name__ == "__main__":
    main()
