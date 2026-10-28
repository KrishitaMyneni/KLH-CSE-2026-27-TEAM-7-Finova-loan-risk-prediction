"""
Upload each generated test file to the running dashboard API and report results.
"""

import json
from pathlib import Path
import requests
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TEST_DIR = ROOT / "data" / "test_files"
API_URL = "http://127.0.0.1:8000/api/score"

test_files = [
    # Good files
    ("fresh_2016_100.csv", "GOOD"),
    ("fresh_2017_1000.csv", "GOOD"),
    ("single_applicant.csv", "GOOD"), 
    ("high_risk_50.csv", "GOOD"),
    ("prime_50.csv", "GOOD"),
    # Tricky files
    ("missing_columns.csv", "TRICKY"),
    ("blank_values.csv", "TRICKY"),
    ("bad_values.csv", "TRICKY"),
    ("mixed_good_bad.csv", "TRICKY"),
    ("all_60_month.csv", "TRICKY"),
    ("empty_file.csv", "TRICKY"),
    ("wrong_dataset.csv", "TRICKY"),
]

results = []

print("=" * 80)
print("TESTING FILE UPLOADS AGAINST DASHBOARD SCORING ENGINE (http://127.0.0.1:8000/api/score)")
print("=" * 80)

for filename, category in test_files:
    filepath = TEST_DIR / filename
    if not filepath.exists():
        print(f"Error: {filepath} does not exist!")
        continue

    print(f"\n---> Uploading: {filename} ({category})")
    
    with open(filepath, "rb") as f:
        try:
            res = requests.post(API_URL, files={"file": (filename, f, "text/csv")})
            status_code = res.status_code

            if status_code == 200:
                data = res.json()
                summary = data.get("summary", {})
                global_warning = data.get("global_warning")
                could_not_score = data.get("could_not_score", [])
                
                outcome = "Scored normally"
                details = (
                    f"Total: {summary.get('total_applicants')}, "
                    f"Valid Scored: {summary.get('valid_count', summary.get('total_applicants'))}, "
                    f"Could Not Score: {len(could_not_score)}, "
                    f"Eligible: {summary.get('eligible_count')}, "
                    f"Approve: {summary.get('approved_count')}, "
                    f"Review: {summary.get('review_count')}, "
                    f"Decline: {summary.get('declined_count')}, "
                    f"Ineligible: {summary.get('ineligible_count')}, "
                    f"Avg Score: {summary.get('avg_credit_score')}, "
                    f"Avg PD: {summary.get('avg_default_probability_pct')}"
                )
                if global_warning:
                    details += f" | Warning: {global_warning}"

                print(f"  Result: {outcome} (HTTP 200)")
                print(f"  Details: {details}")

                if could_not_score:
                    print(f"  Could Not Score Items ({len(could_not_score)}):")
                    for cns in could_not_score:
                        print(f"    - Row {cns['row_number']}: {cns['reason_text']}")

                results.append({
                    "file": filename,
                    "category": category,
                    "status_code": status_code,
                    "classification": outcome,
                    "summary": details,
                    "could_not_score": could_not_score,
                    "raw_response": data
                })

            elif status_code in (400, 422):
                data = res.json() if res.headers.get("content-type") == "application/json" else {"detail": res.text}
                error_msg = data.get("detail", res.text)
                outcome = "Clear error"
                print(f"  Result: {outcome} (HTTP {status_code})")
                print(f"  Message: {error_msg.replace(chr(10), ' | ')}")
                results.append({
                    "file": filename,
                    "category": category,
                    "status_code": status_code,
                    "classification": outcome,
                    "summary": error_msg,
                    "raw_response": data
                })

            else:
                outcome = "Crash (Unexpected HTTP status)"
                print(f"  Result: {outcome} (HTTP {status_code})")
                print(f"  Response: {res.text[:300]}")
                results.append({
                    "file": filename,
                    "category": category,
                    "status_code": status_code,
                    "classification": outcome,
                    "summary": res.text[:300],
                    "raw_response": None
                })

        except Exception as e:
            outcome = "Crash (Exception)"
            print(f"  Result: {outcome} - {str(e)}")
            results.append({
                "file": filename,
                "category": category,
                "status_code": None,
                "classification": outcome,
                "summary": str(e),
                "raw_response": None
            })

# Save evaluation results
with open(ROOT / "scratch" / "test_results.json", "w", encoding="utf-8") as f:
    serializable = []
    for r in results:
        copy_r = {k: v for k, v in r.items() if k != "raw_response"}
        serializable.append(copy_r)
    json.dump(serializable, f, indent=2)

# ==============================================================================
# Model Demonstration: fresh_2016_100 vs answers_2016_100 (DEMO ONLY)
# ==============================================================================
print("\n" + "=" * 80)
print("DEMO ONLY: fresh_2016_100 PREDICTIONS VS answers_2016_100.csv")
print("NOTE: This is strictly a manual testing demonstration, NOT empirical evidence.")
print("The 2016 cohort in the Dec 2018 snapshot is right-censored (36m loans had not fully")
print("matured; active loans remained 'Current' and are missing here), and N=100 is too small.")
print("Do not claim alignment or benchmark validity from this demo.")
print("=" * 80)

# Find 2016 predictions
pred_res = next((r["raw_response"] for r in results if r["file"] == "fresh_2016_100.csv"), None)
if pred_res and "applicants" in pred_res:
    pred_df = pd.DataFrame(pred_res["applicants"])
    answers_df = pd.read_csv(TEST_DIR / "answers_2016_100.csv")
    
    merged = pd.concat([answers_df, pred_df[["default_probability", "credit_score", "risk_band", "decision", "expected_loss"]]], axis=1)
    
    print(f"Total Applicants Demonstrated: {len(merged)}")
    print(f"Resolved Defaults in Snapshot: {merged['actual_default'].sum()} ({(merged['actual_default'].mean()*100):.1f}%)")
    print(f"Resolved Paid Off in Snapshot: {(merged['actual_default'] == 0).sum()} ({((1 - merged['actual_default'].mean())*100):.1f}%)")
    
    print("\nDecision Breakdown in Snapshot Sample:")
    for dec in ["APPROVE", "REVIEW", "DECLINE"]:
        sub = merged[merged["decision"] == dec]
        n = len(sub)
        n_def = sub["actual_default"].sum() if n > 0 else 0
        def_rate = (n_def / n * 100) if n > 0 else 0.0
        print(f"  {dec:8s}: {n:2d} applicants | Early Defaults: {n_def:2d} | Snapshot Sample Default Rate: {def_rate:5.1f}%")
        
    print("\nRisk Band Breakdown in Snapshot Sample:")
    for band in ["Low Risk", "Medium Risk", "High Risk"]:
        sub = merged[merged["risk_band"] == band]
        n = len(sub)
        n_def = sub["actual_default"].sum() if n > 0 else 0
        def_rate = (n_def / n * 100) if n > 0 else 0.0
        print(f"  {band:12s}: {n:2d} applicants | Early Defaults: {n_def:2d} | Snapshot Sample Default Rate: {def_rate:5.1f}%")

    good_scores = merged[merged["actual_default"] == 0]["credit_score"]
    bad_scores = merged[merged["actual_default"] == 1]["credit_score"]
    print(f"\nAverage Score for Snapshot Non-Defaulters: {good_scores.mean():.1f}")
    print(f"Average Score for Snapshot Defaulters:     {bad_scores.mean():.1f}")
    print(f"Score Separation in Demo Cohort:           {good_scores.mean() - bad_scores.mean():.1f} points")

print("\nAll testing completed successfully.")
