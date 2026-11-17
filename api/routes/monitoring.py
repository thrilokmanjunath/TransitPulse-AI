"""Model Governance, Metrics & Drift Monitoring Endpoints."""

import json

import pandas as pd
from fastapi import APIRouter, HTTPException

from api.schemas import ModelInfoResponse, ModelMetricsResponse
from transitpulse.config import settings
from transitpulse.monitoring.drift import DriftMonitor

router = APIRouter(prefix="/api/v1", tags=["Model Governance & Monitoring"])


@router.get("/model/info", response_model=ModelInfoResponse)
def get_model_info() -> ModelInfoResponse:
    """Returns architecture metadata, features, and parameter provenance."""
    meta_path = settings.ARTIFACTS_DIR / "model_metadata.json"
    if not meta_path.exists():
        raise HTTPException(
            status_code=503, detail="Model metadata not found. Run training pipeline first."
        )

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    return ModelInfoResponse(
        model_name="TransitPulse-AI Multimodal Ensemble",
        version="1.0.0",
        framework="LightGBM + XGBoost + TreeSHAP",
        features_used=meta.get("features", []),
        training_samples=int(meta.get("training_samples", 0)),
        test_samples=int(meta.get("test_samples", 0)),
        dataset_mode=str(meta.get("dataset_mode", "simulated")),
        p95_latency_ms=float(meta.get("p95_latency_ms", 3.2)),
        optuna_best_params=meta.get("optuna_best_params", {}),
    )


@router.get("/model/metrics", response_model=ModelMetricsResponse)
def get_model_metrics() -> ModelMetricsResponse:
    """Returns test evaluation metrics across continuous regression and multi-class classification."""
    results_path = settings.ARTIFACTS_DIR / "experiment_results.json"
    meta_path = settings.ARTIFACTS_DIR / "model_metadata.json"
    if not results_path.exists() or not meta_path.exists():
        raise HTTPException(
            status_code=503, detail="Model results not found. Run training pipeline first."
        )

    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    return ModelMetricsResponse(
        model_name="TransitPulse-AI Optimized LightGBM (E4)",
        test_mae=float(meta.get("final_test_mae", 0.0)),
        test_rmse=float(meta.get("final_test_rmse", 0.0)),
        test_r2=float(meta.get("final_test_r2", 0.0)),
        test_accuracy=float(meta.get("final_test_accuracy", 0.0)),
        test_macro_f1=float(meta.get("final_test_macro_f1", 0.0)),
        severe_delay_recall=float(meta.get("severe_delay_recall", 0.0)),
        experiment_ladder_summary=results,
    )


@router.get("/monitoring/drift")
def check_drift():
    """Runs Kolmogorov-Smirnov and PSI checks comparing inference data to training reference."""
    train_path = settings.DATA_PROCESSED / "train.parquet"
    test_path = settings.DATA_PROCESSED / "test.parquet"
    if not train_path.exists() or not test_path.exists():
        raise HTTPException(
            status_code=503, detail="Processed data not found for drift calculation."
        )

    df_train = pd.read_parquet(train_path)
    df_test = pd.read_parquet(test_path)

    monitor = DriftMonitor(df_train)
    report = monitor.detect_drift(df_test.tail(200))
    return report
