"""
Unit and integration tests for the Credit Risk Scoring Console and FastAPI backend.
Tests upload-and-score functionality, data validation, 60-month rejection,
300-applicant batch scoring, and SHAP reason codes.
"""

import io
from pathlib import Path

from fastapi.testclient import TestClient

from app import REQUIRED_COLUMNS, app

client = TestClient(app)
ROOT = Path(__file__).resolve().parents[1]


def test_serve_index():
    """Test that GET / returns the dashboard HTML page."""
    response = client.get("/")
    assert response.status_code == 200
    assert "Credit Risk" in response.text


def test_download_template():
    """Test that GET /api/template returns the CSV template with all required headers."""
    response = client.get("/api/template")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    content = response.text
    for col in REQUIRED_COLUMNS:
        assert col in content, f"Template is missing required column: {col}"


def test_download_demo_batch_endpoint():
    """Test that GET /api/demo-batch returns the 300-applicant demo batch CSV."""
    response = client.get("/api/demo-batch")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    assert "demo_batch_300.csv" in response.headers.get("content-disposition", "")


def test_download_example_batch_endpoint():
    """Test that GET /api/example-batch returns the 5-applicant example CSV."""
    response = client.get("/api/example-batch")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")


def test_score_example_applicants():
    """Test that POST /api/score with data/example_applicants.csv scores all applicants properly."""
    csv_path = ROOT / "data" / "example_applicants.csv"
    assert csv_path.exists(), "data/example_applicants.csv not found"

    with open(csv_path, "rb") as f:
        response = client.post(
            "/api/score",
            files={"file": ("example_applicants.csv", f, "text/csv")},
        )

    assert response.status_code == 200, f"Scoring failed: {response.text}"
    data = response.json()

    assert data["status"] == "success"
    assert "summary" in data
    summary = data["summary"]
    assert summary["total_applicants"] == 5
    assert summary["eligible_count"] == 4
    assert summary["approved_count"] >= 1
    assert summary["declined_count"] >= 1
    assert summary["ineligible_count"] == 1  # The 60-month loan

    applicants = data["applicants"]
    assert len(applicants) == 5

    # Check eligible applicants
    for app_row in applicants[:4]:
        assert app_row["default_probability"] is not None
        assert 0.0 <= app_row["default_probability"] <= 1.0
        assert 300 <= app_row["credit_score"] <= 850
        assert app_row["risk_band"] in ["Low Risk", "Medium Risk", "High Risk"]
        assert app_row["decision"] in ["APPROVE", "REVIEW", "DECLINE"]
        assert app_row["expected_loss"] >= 0.0
        assert len(app_row["reason_codes"]) == 4

    # Validate 60-month applicant was flagged as INELIGIBLE and not scored
    app_60m = applicants[4]
    assert app_60m["term"] == "60 months"
    assert app_60m["decision"] == "INELIGIBLE"
    assert app_60m["default_probability"] is None
    assert app_60m["credit_score"] is None
    assert app_60m["expected_loss"] is None
    assert "Not scored" in app_60m["default_probability_pct"]
    assert "60-month term" in app_60m["warning"]


def test_score_demo_batch_300():
    """Test that POST /api/score with data/demo_batch_300.csv scores 300 applicants."""
    csv_path = ROOT / "data" / "demo_batch_300.csv"
    assert csv_path.exists(), "data/demo_batch_300.csv not found"

    with open(csv_path, "rb") as f:
        response = client.post(
            "/api/score",
            files={"file": ("demo_batch_300.csv", f, "text/csv")},
        )

    assert response.status_code == 200, f"Scoring failed: {response.text}"
    data = response.json()

    assert data["status"] == "success"
    assert "summary" in data
    summary = data["summary"]

    assert summary["total_applicants"] == 300
    assert summary["ineligible_count"] == 6
    assert summary["eligible_count"] == 294
    assert summary["approved_count"] > 0
    assert summary["review_count"] > 0
    assert summary["declined_count"] > 0

    assert len(data["applicants"]) == 300

    # Ensure benchmarks payload is present for UI comparison bars
    assert "benchmarks" in data
    bench = data["benchmarks"]
    assert "fico_avg" in bench
    assert "dti" in bench
    assert "annual_inc" in bench
    assert "revol_util" in bench
    assert bench["tau_low"] == 0.10
    assert bench["tau_high"] == 0.20


def test_reject_non_csv_file():
    """Test that POST /api/score rejects non-CSV files."""
    response = client.post(
        "/api/score",
        files={"file": ("test.txt", io.BytesIO(b"Hello world"), "text/plain")},
    )
    assert response.status_code == 400
    assert "Only CSV files" in response.json()["detail"]


def test_reject_missing_required_columns():
    """Test that POST /api/score returns 422 if required columns are missing."""
    invalid_csv = "loan_amnt,annual_inc,fico_range_low\n10000,50000,700\n"
    response = client.post(
        "/api/score",
        files={"file": ("missing_cols.csv", io.BytesIO(invalid_csv.encode("utf-8")), "text/csv")},
    )
    assert response.status_code == 422
    assert "missing" in response.json()["detail"].lower()


def test_reject_oversized_file():
    """Test that POST /api/score rejects files exceeding 5MB."""
    large_content = b"x" * (5 * 1024 * 1024 + 1024)
    response = client.post(
        "/api/score",
        files={"file": ("large.csv", io.BytesIO(large_content), "text/csv")},
    )
    assert response.status_code == 400
    assert "5MB" in response.json()["detail"]


def test_partial_acceptance_mixed_good_bad():
    """
    Test Requirement 1 (PARTIAL ACCEPTANCE):
    If some rows fail validation, score the valid rows and show the invalid ones
    in a separate 'Could not score' list with the reason for each (row number + problem).
    Only reject the whole file if ALL rows fail or required columns are missing.
    mixed_good_bad.csv has 18 good rows and 2 bad rows (Row 5 corrupted amount, Row 12 negative income).
    """
    csv_path = ROOT / "data" / "test_files" / "mixed_good_bad.csv"
    assert csv_path.exists(), "data/test_files/mixed_good_bad.csv not found"

    with open(csv_path, "rb") as f:
        response = client.post(
            "/api/score",
            files={"file": ("mixed_good_bad.csv", f, "text/csv")},
        )

    assert response.status_code == 200, f"Expected HTTP 200 partial acceptance, got {response.status_code}: {response.text}"
    data = response.json()
    assert data["status"] == "success"

    # Verify summary reflects partial scoring
    summary = data["summary"]
    assert summary["total_applicants"] == 20
    assert summary["valid_count"] == 18
    assert summary["eligible_count"] == 18

    # Verify scored applicants
    applicants = data["applicants"]
    assert len(applicants) == 18
    for row in applicants:
        assert row["decision"] in ["APPROVE", "REVIEW", "DECLINE"]
        assert 300 <= row["credit_score"] <= 850
        assert 0.0 <= row["default_probability"] <= 1.0

    # Verify 'could_not_score' list
    could_not_score = data.get("could_not_score", [])
    assert len(could_not_score) == 2, f"Expected 2 invalid rows, got {len(could_not_score)}"

    bad_row_nums = [c["row_number"] for c in could_not_score]
    assert 5 in bad_row_nums, "Row 5 ('corrupted_amnt') was not in could_not_score"
    assert 12 in bad_row_nums, "Row 12 ('-45000' annual income) was not in could_not_score"

    # Verify reasons contain row number and problem description
    r5 = next(c for c in could_not_score if c["row_number"] == 5)
    assert any("loan_amnt" in r.lower() or "numeric" in r.lower() for r in r5["reasons"])
    r12 = next(c for c in could_not_score if c["row_number"] == 12)
    assert any("negative" in r.lower() or "annual income" in r.lower() for r in r12["reasons"])

    # Global warning should alert user about unscorable rows
    assert data.get("global_warning") is not None
    assert "2 rows failed validation and could not be scored" in data["global_warning"]


def test_optional_model_column_can_be_missing():
    """A CSV without percent_bc_gt_75 (not a required column) still scores."""
    csv_path = ROOT / "data" / "example_applicants.csv"
    import pandas as pd

    df = pd.read_csv(csv_path)
    df = df.drop(columns=[c for c in ["percent_bc_gt_75"] if c in df.columns])
    response = client.post(
        "/api/score",
        files={"file": ("no_optional.csv", io.BytesIO(df.to_csv(index=False).encode()), "text/csv")},
    )
    assert response.status_code == 200, response.text
    assert response.json()["summary"]["eligible_count"] >= 1


def test_filled_in_values_tracking():
    """
    Test Requirement 2 (FILLED-IN VALUES):
    When blank values are filled in automatically, the API returns imputed_fields and
    imputed_labels for that applicant so the UI can display a note listing which fields were filled in.
    """
    csv_path = ROOT / "data" / "test_files" / "blank_values.csv"
    assert csv_path.exists(), "data/test_files/blank_values.csv not found"

    with open(csv_path, "rb") as f:
        response = client.post(
            "/api/score",
            files={"file": ("blank_values.csv", f, "text/csv")},
        )

    assert response.status_code == 200
    data = response.json()
    applicants = data["applicants"]

    # Verify that applicants with blank values have imputed_fields populated
    imputed_apps = [a for a in applicants if len(a.get("imputed_fields", [])) > 0]
    assert len(imputed_apps) > 0, "Expected at least one applicant with imputed fields in blank_values.csv"

    for row in imputed_apps:
        assert len(row["imputed_fields"]) == len(row["imputed_labels"])
        for field, label in zip(row["imputed_fields"], row["imputed_labels"], strict=True):
            assert isinstance(field, str)
            assert isinstance(label, str)
            assert len(label) > 0


def test_reject_when_all_rows_fail():
    """
    Test Requirement 1 Edge Case:
    Only reject the whole file if ALL rows fail validation.
    bad_values.csv has 4 rows that ALL contain fatal validation issues.
    """
    csv_path = ROOT / "data" / "test_files" / "bad_values.csv"
    assert csv_path.exists(), "data/test_files/bad_values.csv not found"

    with open(csv_path, "rb") as f:
        response = client.post(
            "/api/score",
            files={"file": ("bad_values.csv", f, "text/csv")},
        )

    assert response.status_code == 422
    err_detail = response.json()["detail"]
    assert "All 4 row(s) contained invalid or out-of-bounds data" in err_detail

