"""Pytest Configuration and Hermetic Test Fixtures for TransitPulse-AI."""

import json

import joblib
import numpy as np
import pandas as pd
import pytest

from transitpulse.config import settings
from transitpulse.models.multi_horizon import MultiHorizonForecaster
from transitpulse.models.tree_models import LightGBMModel


@pytest.fixture(scope="session", autouse=True)
def ensure_test_artifacts():
    """Ensures minimal test datasets, metadata, and model artifacts exist so tests run hermetically."""
    settings.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    settings.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Feature metadata
    meta_path = settings.DATA_PROCESSED / "feature_metadata.json"
    features_e0 = [
        "prev_stop_delay_min",
        "route_code",
        "route_prior_mean_delay",
        "scheduled_travel_time_sec",
    ]
    features_e1 = features_e0 + [
        "hour_of_day",
        "day_of_week",
        "is_weekend",
        "is_morning_peak",
        "is_evening_peak",
        "is_peak_hour",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
    ]
    features_e2 = features_e1 + [
        "precip_rate_mm",
        "is_raining",
        "is_heavy_rain",
        "temperature_c",
        "relative_humidity",
        "wind_gusts_kmh",
        "weather_severity_idx",
        "precip_rolling_3h_mm",
    ]
    features_e3 = features_e2 + [
        "stop_lat",
        "stop_lon",
        "dist_to_cbd_km",
        "stop_sequence",
        "route_progression_pct",
        "road_hierarchy_level",
        "stop_density_1km",
        "event_count_500m",
        "event_count_1km",
        "event_count_3km",
        "dist_nearest_event_km",
        "min_to_nearest_event",
        "event_capacity_shock",
        "has_nearby_event",
        "rain_x_peak_hour",
        "rain_x_road_hierarchy",
        "event_x_peak_hour",
        "weather_x_progression",
    ]
    features_e4 = features_e3

    if not meta_path.exists():
        metadata = {
            "total_records": 100,
            "train_records": 70,
            "val_records": 15,
            "test_records": 15,
            "experiment_feature_sets": {
                "E0": features_e0,
                "E1": features_e1,
                "E2": features_e2,
                "E3": features_e3,
                "E4": features_e4,
            },
            "all_features": features_e4,
            "target_regression": "observed_delay_min",
            "target_classification": "delay_category",
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    # 2. Processed Parquets (train, val, test) if missing
    train_path = settings.DATA_PROCESSED / "train.parquet"
    val_path = settings.DATA_PROCESSED / "val.parquet"
    test_path = settings.DATA_PROCESSED / "test.parquet"

    if not (train_path.exists() and val_path.exists() and test_path.exists()):
        np.random.seed(42)
        n_rows = 100
        data = {col: np.random.randn(n_rows).astype(np.float32) for col in features_e4}
        data["prev_stop_delay_min"] = np.random.uniform(0, 10, n_rows).astype(np.float32)
        data["route_code"] = np.random.choice([1, 2, 3, 4, 5], n_rows).astype(np.int32)
        data["route_id"] = np.random.choice(["335-E", "500-D", "KIA-9", "G-2", "201-R"], n_rows)
        data["stop_id"] = [f"STOP_{i}" for i in range(n_rows)]
        data["trip_id"] = [f"TRIP_{i // 10}" for i in range(n_rows)]
        data["stop_sequence"] = (np.arange(n_rows) % 10 + 1).astype(np.int32)

        # Monotonic chronological timestamps
        base_time = pd.Timestamp("2024-05-01 06:00:00")
        data["scheduled_time"] = [base_time + pd.Timedelta(hours=i) for i in range(n_rows)]
        data["observed_delay_min"] = (
            (data["prev_stop_delay_min"] * 0.8 + np.random.normal(0, 1, n_rows))
            .clip(-2, 60)
            .astype(np.float32)
        )
        data["delay_category"] = pd.cut(
            data["observed_delay_min"],
            bins=[-np.inf, 5.0, 10.0, 20.0, np.inf],
            labels=["On-time", "Minor", "Moderate", "Severe"],
        ).astype(str)
        data["delay_horizon_15m"] = data["observed_delay_min"] * 0.95
        data["delay_horizon_30m"] = data["observed_delay_min"] * 0.90
        data["delay_horizon_60m"] = data["observed_delay_min"] * 0.85
        data["dataset_mode"] = "simulated"

        df_synthetic = pd.DataFrame(data)
        df_synthetic.iloc[:70].to_parquet(train_path, index=False)
        df_synthetic.iloc[70:85].to_parquet(val_path, index=False)
        df_synthetic.iloc[85:].to_parquet(test_path, index=False)

    # 3. Model artifact (best_model.joblib) if missing
    model_path = settings.ARTIFACTS_DIR / "best_model.joblib"
    if not model_path.exists():
        df_tr = pd.read_parquet(train_path)
        X_tr = df_tr[features_e4]
        y_tr_reg = df_tr["observed_delay_min"].values
        y_tr_cls = df_tr["delay_category"].values

        lgb_test = LightGBMModel(
            params={"n_estimators": 10, "max_depth": 3, "verbose": -1, "random_state": 42}
        )
        lgb_test.fit(X_tr, y_tr_reg, y_tr_cls)
        joblib.dump(lgb_test, model_path)

    # 4. Multi-horizon models if missing
    h_dir = settings.ARTIFACTS_DIR / "multi_horizon"
    if not (h_dir / "model_h15.joblib").exists():
        forecaster = MultiHorizonForecaster()
        df_tr = pd.read_parquet(train_path)
        df_vl = pd.read_parquet(val_path)
        forecaster.fit(
            X_train=df_tr[features_e4],
            df_train_full=df_tr,
            X_val=df_vl[features_e4],
            df_val_full=df_vl,
            best_params={"n_estimators": 10, "max_depth": 3, "verbose": -1, "random_state": 42},
        )
        forecaster.save(h_dir)

    # 5. Sample test records if missing
    sample_path = settings.ARTIFACTS_DIR / "sample_test_records.parquet"
    if not sample_path.exists():
        pd.read_parquet(test_path).head(20).to_parquet(sample_path, index=False)
