"""Pydantic Schemas for TransitPulse-AI REST API."""

from typing import Any

from pydantic import BaseModel, Field


class RouteResponse(BaseModel):
    """BMTC Route Information."""
    route_id: str
    route_short_name: str
    origin_name: str
    destination_name: str
    distance_km: float
    scheduled_duration_min: float
    num_stops: int


class StopResponse(BaseModel):
    """BMTC Stop Information."""
    route_id: str
    stop_id: str
    stop_sequence: int
    stop_name: str
    stop_lat: float
    stop_lon: float
    scheduled_travel_time_sec: float


class PredictionRequest(BaseModel):
    """Input payload for delay prediction."""
    route_id: str = Field(..., description="BMTC Route Identifier (e.g., '335-E', '500-D')")
    stop_sequence: int = Field(1, description="1-based stop sequence index along the route")
    prev_stop_delay_min: float = Field(0.0, description="Observed arrival delay in minutes at previous stop")
    scheduled_time: str | None = Field(None, description="ISO-8601 scheduled timestamp in IST (e.g. 2024-05-18T19:30:00+05:30)")
    
    # Optional weather overrides (if omitted, uses live Open-Meteo forecast)
    precip_mm: float | None = Field(None, ge=0.0, description="Precipitation rate in mm/hour")
    temperature_c: float | None = Field(None, description="Temperature in Celsius")
    relative_humidity: float | None = Field(None, ge=0.0, le=100.0, description="Relative humidity %")
    wind_gusts_kmh: float | None = Field(None, ge=0.0, description="Wind gusts speed in km/h")

    # Optional manual event override
    is_event_active: bool | None = Field(False, description="Manual flag indicating nearby stadium/concert event")


class PredictionResponse(BaseModel):
    """Delay prediction response with multi-horizon values and explainability."""
    route_id: str
    stop_sequence: int
    prediction_timestamp: str
    predicted_delay_min: float
    delay_category: str
    confidence_interval_90: dict[str, float]
    multi_horizon_delays: dict[str, float]
    attribution_summary: dict[str, float]
    dataset_mode: str = "simulated"


class ScenarioRequest(BaseModel):
    """Simulation scenario evaluation payload."""
    route_id: str
    scenario_name: str
    rain_intensity_mm: float = Field(0.0, ge=0.0, le=50.0)
    peak_commute_active: bool = True
    stadium_event_active: bool = False
    expected_event_attendance: int = Field(0, ge=0)


class ScenarioResponse(BaseModel):
    """Simulated scenario impact response."""
    scenario_name: str
    route_id: str
    baseline_delay_min: float
    scenario_delay_min: float
    net_delay_increase_min: float
    predicted_delay_category: str
    contributing_factors: dict[str, float]


class ModelInfoResponse(BaseModel):
    """Model governance metadata response."""
    model_name: str
    version: str
    framework: str
    features_used: list[str]
    training_samples: int
    test_samples: int
    dataset_mode: str
    p95_latency_ms: float
    optuna_best_params: dict[str, Any]


class ModelMetricsResponse(BaseModel):
    """Validation and test metrics response."""
    model_name: str
    test_mae: float
    test_rmse: float
    test_r2: float
    test_accuracy: float
    test_macro_f1: float
    severe_delay_recall: float
    experiment_ladder_summary: dict[str, Any]
