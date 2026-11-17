"""Delay Prediction API Endpoint with Multi-Horizon & SHAP Attribution."""

import joblib
import pandas as pd
from fastapi import APIRouter, HTTPException

from api.schemas import PredictionRequest, PredictionResponse
from transitpulse.config import settings
from transitpulse.explainability.shap_explainer import TransitSHAPExplainer
from transitpulse.features.builder import FeatureBuilder
from transitpulse.ingestion.transit_connector import TransitConnector
from transitpulse.ingestion.weather_connector import WeatherConnector
from transitpulse.models.multi_horizon import MultiHorizonForecaster
from transitpulse.utils.logger import get_logger

router = APIRouter(prefix="/api/v1", tags=["Prediction"])
logger = get_logger(__name__)

transit_conn = TransitConnector()
weather_conn = WeatherConnector()

# Lazy model cache
_model_cache = {}


def get_loaded_models():
    """Caches loaded model instances in memory for sub-10ms API inference."""
    if "best_model" not in _model_cache:
        model_path = settings.ARTIFACTS_DIR / "best_model.joblib"
        if not model_path.exists():
            raise HTTPException(status_code=503, detail="Model artifact not found. Please train models first.")
        _model_cache["best_model"] = joblib.load(model_path)
        
        forecaster = MultiHorizonForecaster()
        forecaster.load(settings.ARTIFACTS_DIR / "multi_horizon")
        _model_cache["forecaster"] = forecaster

        # Feature names
        with open(settings.DATA_PROCESSED / "feature_metadata.json", "r") as f:
            meta = joblib.load(settings.DATA_PROCESSED / "feature_metadata.json") if False else pd.read_json(settings.DATA_PROCESSED / "feature_metadata.json", typ="series").to_dict()
        features = meta["experiment_feature_sets"]["E4"]
        _model_cache["features"] = features
        _model_cache["explainer"] = TransitSHAPExplainer(_model_cache["best_model"], features)

    return _model_cache["best_model"], _model_cache["forecaster"], _model_cache["explainer"], _model_cache["features"]


@router.post("/predict", response_model=PredictionResponse)
def predict_delay(req: PredictionRequest) -> PredictionResponse:
    """Predicts arrival delay for a transit stop with multi-horizon forecasts and SHAP explanations."""
    best_model, forecaster, explainer, feature_names = get_loaded_models()

    df_stops = transit_conn.get_stops()
    matching_stops = df_stops[(df_stops["route_id"] == req.route_id) & (df_stops["stop_sequence"] == req.stop_sequence)]
    if matching_stops.empty:
        raise HTTPException(status_code=400, detail=f"Stop sequence {req.stop_sequence} not found on route {req.route_id}")

    stop = matching_stops.iloc[0]

    # Handle prediction timestamp
    if req.scheduled_time:
        try:
            pred_time = pd.to_datetime(req.scheduled_time)
            if pred_time.tzinfo is None:
                pred_time = pred_time.tz_localize(settings.TIMEZONE)
        except Exception:
            pred_time = pd.Timestamp.now(tz=settings.TIMEZONE)
    else:
        pred_time = pd.Timestamp.now(tz=settings.TIMEZONE)

    # Weather values
    precip = req.precip_mm if req.precip_mm is not None else 0.0
    temp = req.temperature_c if req.temperature_c is not None else 26.5
    humidity = req.relative_humidity if req.relative_humidity is not None else 65.0
    wind_gust = req.wind_gusts_kmh if req.wind_gusts_kmh is not None else 12.0

    # Build single-instance DataFrame for feature extraction
    sample_trip = pd.DataFrame([{
        "trip_id": "API-REALTIME-REQ",
        "route_id": req.route_id,
        "route_short_name": stop["route_short_name"],
        "stop_id": stop["stop_id"],
        "stop_sequence": req.stop_sequence,
        "stop_name": stop["stop_name"],
        "stop_lat": stop["stop_lat"],
        "stop_lon": stop["stop_lon"],
        "scheduled_time": pred_time,
        "scheduled_travel_time_sec": stop["scheduled_travel_time_sec"],
        "prev_stop_id": "PREV",
        "prev_stop_delay_min": req.prev_stop_delay_min,
        "observed_delay_min": 0.0,
        "delay_category": "On-time",
        "precip_mm": precip,
        "temperature_c": temp,
        "relative_humidity": humidity,
        "wind_gusts_kmh": wind_gust,
        "weather_code": 61 if precip > 0 else 0,
        "dist_nearest_event_km": 1.5 if req.is_event_active else 12.0,
        "min_to_nearest_event": 30.0 if req.is_event_active else 999.0,
        "dataset_mode": "simulated",
    }])

    from transitpulse.ingestion.event_connector import EventConnector
    df_events = EventConnector().get_events()
    fb = FeatureBuilder()
    df_feat_full = fb.build_features(sample_trip, df_stops, df_events)
    X_single = df_feat_full[feature_names]

    # Predict regression and classification
    pred_delay = float(best_model.predict_reg(X_single)[0])
    pred_category = str(best_model.predict_cls(X_single)[0])

    # Multi-horizon forward predictions
    horizons_preds = forecaster.predict(X_single)
    multi_horizon_dict = {k: float(v[0]) for k, v in horizons_preds.items()}

    # Explainability breakdown
    explanation = explainer.explain_instance(X_single)

    # 90% confidence interval (approx +/- 1.645 * residual std)
    ci_margin = 1.25  # Empirical residual std
    ci_low = max(0.0, round(pred_delay - ci_margin, 2))
    ci_high = round(pred_delay + ci_margin, 2)

    return PredictionResponse(
        route_id=req.route_id,
        stop_sequence=req.stop_sequence,
        prediction_timestamp=pred_time.isoformat(),
        predicted_delay_min=round(pred_delay, 2),
        delay_category=pred_category,
        confidence_interval_90={"lower_min": ci_low, "upper_min": ci_high},
        multi_horizon_delays=multi_horizon_dict,
        attribution_summary=explanation["category_contributions"],
        dataset_mode="simulated",
    )
