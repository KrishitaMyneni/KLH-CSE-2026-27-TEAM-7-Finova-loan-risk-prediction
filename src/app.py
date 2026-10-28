"""
FastAPI Backend for Credit Risk Scoring Console.
Provides real-time institutional underwriting, probability of default (PD) estimation,
credit score generation (300-850), Basel expected loss, and plain-English Adverse Action SHAP reason codes.

Zero PII disk logging. Reuses src/ cleaning and feature engineering directly.
"""

from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import config
from src import cleaning, explain, features, risk

# Initialize FastAPI App
app = FastAPI(
    title="Institutional Credit Risk Scoring Engine",
    description="Upload-and-score underwriting engine: calibrated PD, scorecard, expected loss and per-applicant reason codes.",
    version="2.1.0",
)

# The dashboard is served from this same origin, so cross-origin access is only
# needed when another front end is pointed at the API. Set ALLOWED_ORIGINS
# (comma-separated) to open it up.
_origins = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]
if _origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

# Load Champion Model and artifacts at startup
MODEL_PATH = ROOT / "artifacts" / "best_model.joblib"
if not MODEL_PATH.exists():
    raise RuntimeError(f"Champion model artifact not found at {MODEL_PATH}")

model_artifact: dict[str, Any] = joblib.load(MODEL_PATH)
pipeline = model_artifact["pipeline"]
model_name = model_artifact.get("model_name", "champion")
numeric_features = model_artifact["numeric_features"]
categorical_features = model_artifact["categorical_features"]
model_metrics = model_artifact.get("metrics", {})

# Policy and benchmarks come from the training run, never from this file
RISK_THRESHOLDS = model_artifact.get("risk_thresholds", config.RISK_THRESHOLDS)
LGD = float(model_artifact.get("lgd", config.LGD))
TAU_LOW = float(RISK_THRESHOLDS["low_max"])
TAU_HIGH = float(RISK_THRESHOLDS["medium_max"])

_bench_path = ROOT / "artifacts" / "underwriting_benchmarks.json"
BENCHMARKS = model_artifact.get("benchmarks") or (
    json.loads(_bench_path.read_text()) if _bench_path.exists() else {}
)
BENCHMARKS = {**BENCHMARKS, "tau_low": TAU_LOW, "tau_high": TAU_HIGH}

explainer = explain.Explainer(
    pipeline, numeric_features, categorical_features,
    background=model_artifact.get("explain_background"),
)

REQUIRED_COLUMNS = [
    "loan_amnt",
    "term",
    "emp_length",
    "home_ownership",
    "annual_inc",
    "verification_status",
    "purpose",
    "dti",
    "delinq_2yrs",
    "earliest_cr_line",
    "fico_range_low",
    "fico_range_high",
    "inq_last_6mths",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
    "initial_list_status",
    "application_type",
    "mort_acc",
    "pub_rec_bankruptcies",
    "tot_cur_bal",
    "tot_hi_cred_lim",
    "bc_util",
    "acc_open_past_24mths",
]

HUMAN_FEATURE_NAMES = {
    "loan_amnt": "Loan Amount",
    "term": "Term",
    "emp_length": "Employment Length",
    "home_ownership": "Home Ownership",
    "annual_inc": "Annual Income",
    "verification_status": "Income Verification",
    "purpose": "Loan Purpose",
    "dti": "Debt-to-Income (DTI)",
    "delinq_2yrs": "Delinquencies (Past 2 Yrs)",
    "earliest_cr_line": "Earliest Credit Line",
    "issue_d": "Issue Date",
    "fico_range_low": "FICO Bureau Score",
    "fico_range_high": "FICO High",
    "inq_last_6mths": "Credit Inquiries (Past 6 Mths)",
    "open_acc": "Open Credit Accounts",
    "pub_rec": "Public Derogatory Records",
    "revol_bal": "Revolving Balance",
    "revol_util": "Revolving Line Utilization",
    "total_acc": "Total Credit Lines",
    "initial_list_status": "Listing Status",
    "application_type": "Application Type",
    "mort_acc": "Mortgage Accounts",
    "pub_rec_bankruptcies": "Bankruptcy Records",
    "tot_cur_bal": "Total Current Balance",
    "tot_hi_cred_lim": "Total High Credit Limit",
    "bc_util": "Bankcard Utilization",
    "percent_bc_gt_75": "Bankcards Exceeding 75% Limit",
    "acc_open_past_24mths": "Accounts Opened (Past 24 Mths)",
}

# Mount static files from dashboard/
DASHBOARD_DIR = ROOT / "dashboard"
if DASHBOARD_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(DASHBOARD_DIR)), name="static")


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_file = DASHBOARD_DIR / "index.html"
    if not index_file.exists():
        return HTMLResponse("<h3>Dashboard index.html not found.</h3>", status_code=404)
    with open(index_file, encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/api/health")
async def health():
    info = model_artifact.get("run_info", {})
    return {
        "status": "ok",
        "model": model_name,
        "trained_at": model_artifact.get("trained_at"),
        "test_roc_auc": model_metrics.get("roc_auc"),
        "test_ks": model_metrics.get("ks"),
        "risk_thresholds": RISK_THRESHOLDS,
        "test_period": info.get("periods", {}).get("test"),
        "reason_codes": explainer.supported,
    }


@app.get("/api/template")
async def download_template():
    template_file = ROOT / "data" / "template_applicants.csv"
    if not template_file.exists():
        raise HTTPException(status_code=404, detail="Template CSV not found.")
    return FileResponse(
        path=str(template_file),
        media_type="text/csv",
        filename="applicant_scoring_template.csv",
    )


@app.post("/api/score")
async def score_applicants(file: UploadFile = File(...)):
    # 1. Validation: file format
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Invalid file format. Only CSV files (.csv) are accepted.",
        )

    # 2. Validation: file size limit (5 MB)
    content = await file.read()
    if len(content) > 5 * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail="File size exceeds the 5MB upload limit.",
        )

    # 3. Parse CSV in-memory (Zero disk persistence)
    try:
        raw_df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV file: {e}") from e

    if raw_df.empty:
        raise HTTPException(status_code=400, detail="The uploaded CSV file is empty.")

    # 4. Check required columns
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in raw_df.columns]
    if missing_cols:
        raise HTTPException(
            status_code=422,
            detail=f"Uploaded CSV is missing {len(missing_cols)} required column(s): {', '.join(missing_cols)}",
        )

    # 4b. Validate data sanity (types and permissible ranges per row)
    numeric_check_cols = [
        "loan_amnt", "annual_inc", "dti", "delinq_2yrs", "fico_range_low", "fico_range_high",
        "inq_last_6mths", "open_acc", "pub_rec", "revol_bal", "total_acc", "mort_acc",
        "pub_rec_bankruptcies", "tot_cur_bal", "tot_hi_cred_lim", "acc_open_past_24mths"
    ]
    valid_indices = []
    invalid_rows = []
    all_validation_errors = []

    for idx in range(len(raw_df)):
        row_num = idx + 1
        row_errors = []
        for col in numeric_check_cols:
            if col in raw_df.columns:
                val = raw_df.at[idx, col]
                if pd.notna(val) and str(val).strip() != "":
                    try:
                        num = float(str(val).replace(",", "").strip())
                    except (ValueError, TypeError):
                        row_errors.append(
                            f"Invalid non-numeric value '{val}' in column '{col}'"
                        )
                        continue

                    if col == "loan_amnt" and num <= 0:
                        row_errors.append(
                            f"Loan amount must be positive, got {val}"
                        )
                    elif col == "annual_inc" and num < 0:
                        row_errors.append(
                            f"Negative annual income ({val}) is invalid"
                        )
                    elif col == "dti" and ((num < 0 and num != -1.0) or num > 100):
                        row_errors.append(
                            f"DTI ratio of {val} is outside permissible range (0%–100%)"
                        )
                    elif col in ("fico_range_low", "fico_range_high") and (num < 300 or num > 850):
                        row_errors.append(
                            f"FICO credit score ({val}) is outside valid credit bureau range (300–850)"
                        )

        if row_errors:
            invalid_rows.append({
                "row_number": row_num,
                "reasons": row_errors,
                "raw_data": raw_df.iloc[idx].to_dict(),
            })
            for err in row_errors:
                all_validation_errors.append(f"Row {row_num}: {err}")
        else:
            valid_indices.append(idx)

    # Only reject the whole file if ALL rows fail or required columns are missing
    if len(valid_indices) == 0:
        preview = "\n• " + "\n• ".join(all_validation_errors[:8])
        if len(all_validation_errors) > 8:
            preview += f"\n... and {len(all_validation_errors) - 8} more issue(s)"
        raise HTTPException(
            status_code=422,
            detail=f"Data validation failed: All {len(raw_df)} row(s) contained invalid or out-of-bounds data:{preview}",
        )

    # Slice valid rows for scoring pipeline
    df_scoring = raw_df.iloc[valid_indices].copy().reset_index(drop=True)

    # Ensure numeric columns are properly typed (in case rejected string values caused object dtype)
    for col in numeric_check_cols:
        if col in df_scoring.columns:
            df_scoring[col] = pd.to_numeric(df_scoring[col], errors="coerce")

    # Check for issue_d (optional, defaults to current date if missing or blank)
    if "issue_d" not in df_scoring.columns or df_scoring["issue_d"].isna().all():
        df_scoring["issue_d"] = pd.Timestamp.now().strftime("%b-%Y")
    else:
        df_scoring["issue_d"] = df_scoring["issue_d"].fillna(pd.Timestamp.now().strftime("%b-%Y"))

    # Check for 60-month term loans warning
    parsed_term = cleaning.parse_term(df_scoring["term"])
    has_60m_loans = (parsed_term == 60).any()
    notice_parts = []
    if has_60m_loans:
        n_60m = int((parsed_term == 60).sum())
        notice_parts.append(
            f"{_plural(n_60m, 'applicant')} asked for a 60-month term, so {'it was' if n_60m == 1 else 'they were'} not scored (only 36-month loans are)"
        )
    if invalid_rows:
        notice_parts.append(
            f"{_plural(len(invalid_rows), 'row')} failed validation and could not be scored (see Input issues)"
        )
    global_warning = ("; ".join(notice_parts) + ".").capitalize() if notice_parts else None

    # 5. Reuse exact cleaning & feature engineering pipelines
    try:
        df_clean, _ = cleaning.clean(df_scoring, verbose=False, is_training=False)
        df_feat = features.build_features(df_clean, verbose=False)
        # Optional columns the upload left out are scored as missing: the
        # pipeline imputes them with training medians instead of erroring.
        X = df_feat.reindex(columns=numeric_features + categorical_features)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Feature transformation error: {e}",
        ) from e

    # 6. Predict probabilities of default and scores
    probs = pipeline.predict_proba(X)[:, 1]
    scores = risk.probability_to_score(probs)
    bands = risk.assign_band(probs, thresholds=RISK_THRESHOLDS)

    # Safe loan amount for expected loss calculation
    loan_amnts = pd.to_numeric(df_scoring["loan_amnt"], errors="coerce").fillna(0.0).values
    expected_losses = risk.expected_loss(probs, loan_amnts, lgd=LGD)

    # 7. Per-applicant contributions (TreeSHAP or exact linear, by model type)
    if explainer.supported:
        shap_vals = explainer.contributions(X).to_numpy()
    else:
        shap_vals = np.zeros((len(X), len(explainer.features)))

    applicants_out = []
    n_approved = 0
    n_review = 0
    n_declined = 0
    n_ineligible = 0

    tau_low, tau_high = TAU_LOW, TAU_HIGH

    eligible_scores = []
    eligible_probs = []
    eligible_exposure = 0.0
    eligible_el = 0.0

    def safe_float(val: Any, default: float = 0.0) -> float:
        try:
            num = float(val)
            return default if np.isnan(num) else num
        except (ValueError, TypeError):
            return default

    def safe_int(val: Any, default: int = 700) -> int:
        try:
            num = float(val)
            return default if np.isnan(num) else int(round(num))
        except (ValueError, TypeError):
            return default

    for i in range(len(df_scoring)):
        orig_row = df_scoring.iloc[i].to_dict()
        orig_row_num = int(valid_indices[i] + 1)
        p_val = float(probs[i])
        score_val = int(scores[i])
        band_val = str(bands[i])
        amt_val = float(loan_amnts[i])
        el_val = float(expected_losses[i])
        term_val = parsed_term.iloc[i]

        # Top-4 Reason codes
        reasons = risk.generate_reason_codes(
            explainer.features,
            shap_vals[i],
            original_row=orig_row,
            top_k=4,
        ) if explainer.supported else []

        # Check which required fields were missing / blank in original upload (for Point 2)
        imputed_fields = []
        for col in REQUIRED_COLUMNS:
            v = orig_row.get(col)
            if pd.isna(v) or v is None or str(v).strip() == "":
                imputed_fields.append(col)
        imputed_labels = [HUMAN_FEATURE_NAMES.get(c, c.replace("_", " ").title()) for c in imputed_fields]

        # Decision logic
        if term_val == 60:
            dec = "INELIGIBLE"
            dec_rationale = "Rejected: Applicant requested 60-month term. Model strictly calibrated on 36-month loan cohorts."
            n_ineligible += 1
            row_warning = "60-month term is outside validated model scope. Not scored."

            applicants_out.append(
                {
                    "id": orig_row_num,
                    "loan_amnt": amt_val,
                    "term": "60 months",
                    "annual_inc": safe_float(orig_row.get("annual_inc"), 0.0),
                    "dti": safe_float(orig_row.get("dti"), 0.0),
                    "fico": safe_int(orig_row.get("fico_range_low"), 700),
                    "purpose": str(orig_row.get("purpose", "debt_consolidation")),
                    "default_probability": None,
                    "default_probability_pct": "Not scored",
                    "credit_score": None,
                    "risk_band": "Not scored",
                    "expected_loss": None,
                    "decision": dec,
                    "rationale": dec_rationale,
                    "warning": row_warning,
                    "imputed_fields": imputed_fields,
                    "imputed_labels": imputed_labels,
                    "reason_codes": [],
                    "raw_features": {k: str(v) for k, v in orig_row.items() if pd.notna(v)},
                }
            )
        else:
            # Eligible 36-month loan: band -> decision, both from config policy
            dec = config.DECISIONS[band_val]
            if dec == "APPROVE":
                dec_rationale = f"Low default risk ({p_val:.2%}) ≤ {tau_low:.0%}. Automated approval."
                n_approved += 1
            elif dec == "REVIEW":
                dec_rationale = f"Moderate default risk ({p_val:.2%}) in review zone ({tau_low:.0%}–{tau_high:.0%}). Forward to underwriter for manual review."
                n_review += 1
            else:
                dec_rationale = f"High default risk ({p_val:.2%}) > {tau_high:.0%}. Exceeds risk appetite."
                n_declined += 1

            eligible_scores.append(score_val)
            eligible_probs.append(p_val)
            eligible_exposure += amt_val
            eligible_el += el_val

            applicants_out.append(
                {
                    "id": orig_row_num,
                    "loan_amnt": amt_val,
                    "term": "36 months",
                    "annual_inc": safe_float(orig_row.get("annual_inc"), 0.0),
                    "dti": safe_float(orig_row.get("dti"), 0.0),
                    "fico": safe_int(orig_row.get("fico_range_low"), 700),
                    "purpose": str(orig_row.get("purpose", "debt_consolidation")),
                    "default_probability": round(p_val, 4),
                    "default_probability_pct": f"{p_val * 100:.2f}%",
                    "credit_score": score_val,
                    "risk_band": band_val,
                    "expected_loss": round(el_val, 2),
                    "decision": dec,
                    "rationale": dec_rationale,
                    "warning": None,
                    "imputed_fields": imputed_fields,
                    "imputed_labels": imputed_labels,
                    "reason_codes": reasons,
                    "raw_features": {k: str(v) for k, v in orig_row.items() if pd.notna(v)},
                }
            )

    # Format invalid rows list for partial acceptance reporting
    could_not_score_out = []
    for inv in invalid_rows:
        could_not_score_out.append(
            {
                "row_number": inv["row_number"],
                "reasons": inv["reasons"],
                "reason_text": "; ".join(inv["reasons"]),
                "loan_amnt": str(inv["raw_data"].get("loan_amnt", "N/A")),
                "annual_inc": str(inv["raw_data"].get("annual_inc", "N/A")),
                "dti": str(inv["raw_data"].get("dti", "N/A")),
                "fico": str(inv["raw_data"].get("fico_range_low", "N/A")),
                "purpose": str(inv["raw_data"].get("purpose", "N/A")),
                "raw_features": {k: str(v) for k, v in inv["raw_data"].items() if pd.notna(v)},
            }
        )

    # Summary strictly on eligible scored cohort
    n_eligible = len(eligible_scores)
    avg_score = float(np.mean(eligible_scores)) if n_eligible > 0 else 0.0
    avg_pd = float(np.mean(eligible_probs) * 100) if n_eligible > 0 else 0.0
    approval_rate = (n_approved / n_eligible * 100) if n_eligible > 0 else 0.0

    benchmarks_payload = BENCHMARKS

    return JSONResponse(
        content={
            "status": "success",
            "model": model_name,
            "tau_low": tau_low,
            "tau_high": tau_high,
            "global_warning": global_warning,
            "summary": {
                "total_applicants": len(raw_df),
                "valid_count": len(df_scoring),
                "could_not_score_count": len(invalid_rows),
                "eligible_count": n_eligible,
                "approved_count": n_approved,
                "review_count": n_review,
                "declined_count": n_declined,
                "ineligible_count": n_ineligible,
                "approval_rate_pct": f"{approval_rate:.1f}%",
                "total_exposure": round(eligible_exposure, 2),
                "total_expected_loss": round(eligible_el, 2),
                "avg_credit_score": round(avg_score, 1),
                "avg_default_probability_pct": f"{avg_pd:.2f}%",
            },
            "could_not_score": could_not_score_out,
            "benchmarks": benchmarks_payload,
            "applicants": applicants_out,
        }
    )


@app.get("/api/demo-batch")
async def download_demo_batch():
    demo_file = ROOT / "data" / "demo_batch_300.csv"
    if not demo_file.exists():
        raise HTTPException(status_code=404, detail="Demo batch CSV not found.")
    return FileResponse(
        path=str(demo_file),
        media_type="text/csv",
        filename="demo_batch_300.csv",
    )


@app.get("/api/example-batch")
async def download_example_batch():
    example_file = ROOT / "data" / "example_applicants.csv"
    if not example_file.exists():
        raise HTTPException(status_code=404, detail="Example applicants CSV not found.")
    return FileResponse(
        path=str(example_file),
        media_type="text/csv",
        filename="example_applicants.csv",
    )


@app.get("/api/mixed-batch")
async def download_mixed_batch():
    mixed_file = ROOT / "data" / "test_files" / "mixed_good_bad.csv"
    if not mixed_file.exists():
        raise HTTPException(status_code=404, detail="Mixed test CSV not found.")
    return FileResponse(
        path=str(mixed_file),
        media_type="text/csv",
        filename="mixed_good_bad.csv",
    )


@app.get("/api/ineligible-batch")
async def download_ineligible_batch():
    ineligible_file = ROOT / "data" / "test_files" / "all_60_month.csv"
    if not ineligible_file.exists():
        raise HTTPException(status_code=404, detail="Ineligible test CSV not found.")
    return FileResponse(
        path=str(ineligible_file),
        media_type="text/csv",
        filename="all_60_month.csv",
    )


if __name__ == "__main__":
    uvicorn.run("app:app", host=os.getenv("HOST", "127.0.0.1"),
                port=int(os.getenv("PORT", "8000")), reload=False)

