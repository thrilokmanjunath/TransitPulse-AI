"""Smoke Tests for Streamlit Dashboard Pages."""

from transitpulse.config import settings


def test_dashboard_files_exist():
    dash_root = settings.PROJECT_ROOT / "dashboard"
    app_file = dash_root / "app.py"
    assert app_file.exists(), "Root dashboard app.py missing!"

    pages_dir = dash_root / "pages"
    assert pages_dir.exists(), "Dashboard pages directory missing!"

    expected_pages = [
        "1_Mobility_Overview.py",
        "2_Route_Performance.py",
        "3_Delay_Prediction.py",
        "4_Weather_Impact.py",
        "5_Event_Impact.py",
        "6_Scenario_Simulator.py",
        "7_Explainable_AI.py",
        "8_Model_Benchmarking.py",
    ]

    for p in expected_pages:
        page_path = pages_dir / p
        assert page_path.exists(), f"Expected dashboard page {p} does not exist!"
        # Smoke test content length
        content = page_path.read_text(encoding="utf-8")
        assert len(content) > 100, f"Page {p} is empty or incomplete!"
