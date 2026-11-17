"""Scenario Simulator Endpoints for What-If Policy Analysis."""

import pandas as pd
from fastapi import APIRouter, HTTPException

from api.routes.predict import get_loaded_models
from api.schemas import ScenarioRequest, ScenarioResponse
from transitpulse.config import settings
from transitpulse.features.builder import FeatureBuilder
from transitpulse.ingestion.event_connector import EventConnector
from transitpulse.ingestion.transit_connector import TransitConnector

router = APIRouter(prefix="/api/v1", tags=["Scenario Simulation"])
transit_conn = TransitConnector()
event_conn = EventConnector()


@router.post("/scenario", response_model=ScenarioResponse)
def evaluate_scenario(req: ScenarioRequest) -> ScenarioResponse:
    """Evaluates the impact of exogenous shocks (monsoon downpours, stadium crowds)."""
    best_model, _, _, feature_names = get_loaded_models()

    df_stops = transit_conn.get_stops()
    matching_stops = df_stops[df_stops["route_id"] == req.route_id].sort_values("stop_sequence")
    if matching_stops.empty:
        raise HTTPException(status_code=404, detail=f"Route '{req.route_id}' not found.")

    mid_stop = matching_stops.iloc[len(matching_stops) // 2]
    eval_time = pd.Timestamp.now(tz=settings.TIMEZONE).replace(
        hour=18 if req.peak_commute_active else 14, minute=0
    )

    # 1. Baseline Run (Clear weather, zero events, off-peak or normal)
    base_record = pd.DataFrame(
        [
            {
                "trip_id": "SCENARIO-BASE",
                "route_id": req.route_id,
                "route_short_name": mid_stop["route_short_name"],
                "stop_id": mid_stop["stop_id"],
                "stop_sequence": int(mid_stop["stop_sequence"]),
                "stop_name": mid_stop["stop_name"],
                "stop_lat": mid_stop["stop_lat"],
                "stop_lon": mid_stop["stop_lon"],
                "scheduled_time": eval_time,
                "scheduled_travel_time_sec": mid_stop["scheduled_travel_time_sec"],
                "prev_stop_id": "PREV",
                "prev_stop_delay_min": 1.5,
                "observed_delay_min": 0.0,
                "delay_category": "On-time",
                "precip_mm": 0.0,
                "temperature_c": 26.0,
                "relative_humidity": 60.0,
                "wind_gusts_kmh": 8.0,
                "weather_code": 0,
                "dist_nearest_event_km": 15.0,
                "min_to_nearest_event": 999.0,
                "dataset_mode": "simulated",
            }
        ]
    )

    # 2. Scenario Run (User-defined shocks)
    scen_record = base_record.copy()
    scen_record["precip_mm"] = req.rain_intensity_mm
    scen_record["weather_code"] = (
        65 if req.rain_intensity_mm >= 10.0 else (61 if req.rain_intensity_mm > 0 else 0)
    )
    scen_record["dist_nearest_event_km"] = 0.8 if req.stadium_event_active else 15.0
    scen_record["min_to_nearest_event"] = 20.0 if req.stadium_event_active else 999.0

    df_events = event_conn.get_events()
    fb = FeatureBuilder()

    X_base = fb.build_features(base_record, df_stops, df_events)[feature_names]
    X_scen = fb.build_features(scen_record, df_stops, df_events)[feature_names]

    base_pred = float(best_model.predict_reg(X_base)[0])
    scen_pred = float(best_model.predict_reg(X_scen)[0])
    scen_category = str(best_model.predict_cls(X_scen)[0])

    net_increase = round(max(0.0, scen_pred - base_pred), 2)

    # Contributing factors
    rain_contrib = (
        round(req.rain_intensity_mm * 1.4 + (0.3 * req.rain_intensity_mm**1.2), 2)
        if req.rain_intensity_mm > 0
        else 0.0
    )
    event_contrib = (
        round((req.expected_event_attendance / 10000.0) * 3.5, 2)
        if req.stadium_event_active
        else 0.0
    )
    peak_contrib = 4.2 if req.peak_commute_active else 0.0

    return ScenarioResponse(
        scenario_name=req.scenario_name,
        route_id=req.route_id,
        baseline_delay_min=round(base_pred, 2),
        scenario_delay_min=round(scen_pred, 2),
        net_delay_increase_min=net_increase,
        predicted_delay_category=scen_category,
        contributing_factors={
            "rainfall_friction_delay_min": rain_contrib,
            "stadium_event_shock_min": event_contrib,
            "commuter_peak_delay_min": peak_contrib,
        },
    )
