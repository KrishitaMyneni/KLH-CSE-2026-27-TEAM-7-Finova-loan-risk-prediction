import sys
from pathlib import Path
import time
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
SCREENSHOT_DIR = ROOT / "docs" / "screenshots"
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
DEMO_CSV = ROOT / "data" / "demo_batch_300.csv"

def capture():
    print(f"Starting upgraded dashboard screenshot capture...")
    print(f"Target directory: {SCREENSHOT_DIR}")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)

        # -------------------------------------------------------------
        # VIEWPORT 1: 1920 x 1080 (High-Definition Desktop)
        # -------------------------------------------------------------
        print("\n--- VIEWPORT 1: 1920 x 1080 Desktop ---")
        context_1080 = browser.new_context(viewport={"width": 1920, "height": 1080}, device_scale_factor=1.5)
        page = context_1080.new_page()

        # 1. EMPTY START SCREEN
        print("\n[1/4] Capturing empty start screen...")
        page.goto("http://127.0.0.1:8000")
        page.wait_for_selector("#uploadSection", state="visible")
        time.sleep(1.0)
        p1 = SCREENSHOT_DIR / "01_empty_start_screen.png"
        page.screenshot(path=str(p1))
        print(f"  [SAVED] {p1}")

        # 2. SCORE DEMO BATCH (300 APPLICANTS)
        print("\n[2/4] Scoring 300-applicant demo batch...")
        file_input = page.locator("#fileInput")
        file_input.set_input_files(str(DEMO_CSV))

        page.wait_for_selector("#resultsSection", state="visible", timeout=15000)
        page.wait_for_selector("#applicantTableBody tr", state="visible", timeout=15000)
        time.sleep(2.5)  # Allow Chart.js animations to complete

        # Capture Portfolio Overview (top half)
        page.evaluate("window.scrollTo(0, 0)")
        time.sleep(0.5)
        p2 = SCREENSHOT_DIR / "02_portfolio_overview.png"
        page.screenshot(path=str(p2))
        print(f"  [SAVED] {p2}")

        # Capture Underwriting Table (scroll down)
        table_container = page.locator(".table-container")
        table_container.scroll_into_view_if_needed()
        time.sleep(0.5)
        p3 = SCREENSHOT_DIR / "03_results_table.png"
        page.screenshot(path=str(p3))
        print(f"  [SAVED] {p3}")

        # 3. APPLICANT DETAIL DOSSIER (MODAL)
        print("\n[3/4] Opening applicant detail dossier modal...")
        first_row = page.locator("#applicantTableBody tr").first
        first_row.click()
        page.wait_for_selector("#applicantModal", state="visible", timeout=5000)
        time.sleep(1.5)  # Allow canvas gauge and bars to render

        p4 = SCREENSHOT_DIR / "04_applicant_detail_view.png"
        page.screenshot(path=str(p4))
        print(f"  [SAVED] {p4}")

        context_1080.close()

        # -------------------------------------------------------------
        # VIEWPORT 2: 1366 x 768 (Standard Business Laptop)
        # -------------------------------------------------------------
        print("\n--- VIEWPORT 2: 1366 x 768 Standard Laptop ---")
        context_768 = browser.new_context(viewport={"width": 1366, "height": 768}, device_scale_factor=1.5)
        page_laptop = context_768.new_page()

        page_laptop.goto("http://127.0.0.1:8000")
        page_laptop.wait_for_selector("#uploadSection", state="visible")
        file_input_laptop = page_laptop.locator("#fileInput")
        file_input_laptop.set_input_files(str(DEMO_CSV))

        page_laptop.wait_for_selector("#resultsSection", state="visible", timeout=15000)
        page_laptop.wait_for_selector("#applicantTableBody tr", state="visible", timeout=15000)
        time.sleep(2.0)

        # Check table does not overflow horizontally
        table_box = page_laptop.locator(".table-container").bounding_box()
        print(f"  Table container width on 1366px screen: {table_box['width']:.1f}px")

        p5 = SCREENSHOT_DIR / "05_laptop_1366_results.png"
        page_laptop.screenshot(path=str(p5))
        print(f"  [SAVED] {p5}")
        context_768.close()

        # -------------------------------------------------------------
        # VIEWPORT 3: Partial Acceptance Audit View
        # -------------------------------------------------------------
        print("\n--- VIEWPORT 3: Partial Acceptance & Audit Trail ---")
        context_audit = browser.new_context(viewport={"width": 1920, "height": 1080}, device_scale_factor=1.5)
        page_audit = context_audit.new_page()
        page_audit.goto("http://127.0.0.1:8000")
        page_audit.wait_for_selector("#btnDemoMixed", state="visible")
        page_audit.click("#btnDemoMixed")

        page_audit.wait_for_selector("#couldNotScoreSection", state="visible", timeout=15000)
        time.sleep(2.0)

        could_not_score = page_audit.locator("#couldNotScoreSection")
        could_not_score.scroll_into_view_if_needed()
        time.sleep(0.5)

        p6 = SCREENSHOT_DIR / "06_partial_acceptance_audit.png"
        page_audit.screenshot(path=str(p6))
        print(f"  [SAVED] {p6}")

        context_audit.close()
        browser.close()

        # Copy key presentation console assets to presentation_artifacts/
        import shutil
        present_dir = ROOT / "presentation_artifacts"
        present_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p2, present_dir / "console_01_portfolio_overview.png")
        shutil.copy2(p3, present_dir / "console_02_results_table.png")
        shutil.copy2(p4, present_dir / "console_03_applicant_dossier.png")
        shutil.copy2(p6, present_dir / "console_04_partial_acceptance.png")
        print("\nCopied console presentation screenshots to presentation_artifacts/")
        print("All upgraded screenshots captured successfully!")

if __name__ == "__main__":
    capture()
