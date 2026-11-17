"""Tests for FastAPI REST Endpoints."""

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_api_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["city"] == "Bengaluru"


def test_api_routes():
    resp = client.get("/api/v1/routes")
    assert resp.status_code == 200
    routes = resp.json()
    assert len(routes) >= 5
    route_ids = [r["route_id"] for r in routes]
    assert "335-E" in route_ids
    assert "500-D" in route_ids


def test_api_stops_success():
    resp = client.get("/api/v1/stops?route_id=335-E")
    assert resp.status_code == 200
    stops = resp.json()
    assert len(stops) >= 10
    assert stops[0]["stop_sequence"] == 1


def test_api_stops_not_found():
    resp = client.get("/api/v1/stops?route_id=INVALID-ROUTE-999")
    assert resp.status_code == 404


def test_api_predict():
    payload = {
        "route_id": "335-E",
        "stop_sequence": 3,
        "prev_stop_delay_min": 2.5,
        "precip_mm": 4.0,
        "temperature_c": 25.0,
        "is_event_active": False,
    }
    resp = client.post("/api/v1/predict", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "predicted_delay_min" in data
    assert "delay_category" in data
    assert "multi_horizon_delays" in data
    assert "horizon_15m" in data["multi_horizon_delays"]
    assert "attribution_summary" in data


def test_api_scenario():
    payload = {
        "route_id": "500-D",
        "scenario_name": "Test Downpour",
        "rain_intensity_mm": 12.0,
        "peak_commute_active": True,
        "stadium_event_active": False,
    }
    resp = client.post("/api/v1/scenario", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["route_id"] == "500-D"
    assert data["net_delay_increase_min"] >= 0.0
    assert "contributing_factors" in data


def test_api_model_info():
    resp = client.get("/api/v1/model/info")
    assert resp.status_code == 200
    data = resp.json()
    assert "features_used" in data
    assert data["framework"].startswith("LightGBM")


def test_api_model_metrics():
    resp = client.get("/api/v1/model/metrics")
    assert resp.status_code == 200
    data = resp.json()
    assert data["test_mae"] > 0.0
    assert data["test_r2"] > 0.8
