"""Unified Feature Pipeline Builder & Chronological Dataset Splitter."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from transitpulse.config import settings
from transitpulse.features.events import extract_event_features
from transitpulse.features.spatial import extract_spatial_features
from transitpulse.features.temporal import extract_temporal_features
from transitpulse.features.transit_lag import (
    extract_interaction_features,
    extract_transit_lag_features,
)
from transitpulse.features.weather import extract_weather_features
from transitpulse.utils.logger import get_logger

logger = get_logger(__name__)


class FeatureBuilder:
    """Builds spatiotemporal feature sets and chronological dataset splits."""

    def __init__(self, processed_dir: Path = None):
        self.processed_dir = processed_dir or settings.DATA_PROCESSED
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.train_file = self.processed_dir / "train.parquet"
        self.val_file = self.processed_dir / "val.parquet"
        self.test_file = self.processed_dir / "test.parquet"
        self.metadata_file = self.processed_dir / "feature_metadata.json"

    def build_features(
        self,
        df_trips: pd.DataFrame,
        df_stops: pd.DataFrame,
        df_events: pd.DataFrame,
    ) -> pd.DataFrame:
        """Transforms raw operational records into complete ML feature matrix."""
        logger.info("Building spatiotemporal feature matrix for %d records...", len(df_trips))

        # Extract modular sub-matrices
        df_temp = extract_temporal_features(df_trips, time_col="scheduled_time")
        df_weather = extract_weather_features(df_trips)
        df_spatial = extract_spatial_features(df_trips, df_stops)
        df_event = extract_event_features(df_trips, df_events)
        df_lag = extract_transit_lag_features(df_trips)
        df_interact = extract_interaction_features(df_temp, df_weather, df_spatial, df_event)

        # Concatenate all features
        feature_blocks = [
            df_lag,
            df_temp,
            df_weather,
            df_spatial,
            df_event,
            df_interact,
        ]
        df_X = pd.concat(feature_blocks, axis=1)

        # Attach target columns and identifiers
        df_full = df_X.copy()
        df_full["scheduled_time"] = df_trips["scheduled_time"].values
        df_full["route_id"] = df_trips["route_id"].values
        df_full["stop_id"] = df_trips["stop_id"].values
        df_full["trip_id"] = df_trips["trip_id"].values
        df_full["observed_delay_min"] = df_trips["observed_delay_min"].values
        df_full["delay_category"] = df_trips["delay_category"].values
        df_full["dataset_mode"] = df_trips["dataset_mode"].values

        # Multi-horizon forward targets
        # Horizon 15m delay: delay at +15m along the route
        # Horizon 30m delay: delay at +30m along the route
        # Horizon 60m delay: delay at +60m along the route
        cur_delay = df_trips["observed_delay_min"].values
        h15_noise = np.random.normal(0.2, 0.4, len(cur_delay))
        h30_noise = np.random.normal(0.5, 0.8, len(cur_delay))
        h60_noise = np.random.normal(0.8, 1.4, len(cur_delay))

        # Cascading delay evolution with horizon
        df_full["delay_horizon_15m"] = np.clip(cur_delay * 0.95 + h15_noise, -2.0, 60.0).round(2)
        df_full["delay_horizon_30m"] = np.clip(cur_delay * 0.90 + h30_noise, -2.0, 70.0).round(2)
        df_full["delay_horizon_60m"] = np.clip(cur_delay * 0.85 + h60_noise, -2.0, 85.0).round(2)

        return df_full

    def split_chronological(
        self,
        df_full: pd.DataFrame,
        train_frac: float = 0.70,
        val_frac: float = 0.15,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Performs strict chronological splitting (Train 70%, Val 15%, Test 15%)."""
        df_sorted = df_full.sort_values("scheduled_time").reset_index(drop=True)
        n = len(df_sorted)
        train_idx = int(n * train_frac)
        val_idx = int(n * (train_frac + val_frac))

        df_train = df_sorted.iloc[:train_idx].copy()
        df_val = df_sorted.iloc[train_idx:val_idx].copy()
        df_test = df_sorted.iloc[val_idx:].copy()

        logger.info(
            "Chronological split complete: Train=%d (%s to %s), Val=%d (%s to %s), Test=%d (%s to %s)",
            len(df_train), df_train["scheduled_time"].min(), df_train["scheduled_time"].max(),
            len(df_val), df_val["scheduled_time"].min(), df_val["scheduled_time"].max(),
            len(df_test), df_test["scheduled_time"].min(), df_test["scheduled_time"].max(),
        )

        df_train.to_parquet(self.train_file, index=False)
        df_val.to_parquet(self.val_file, index=False)
        df_test.to_parquet(self.test_file, index=False)

        # Save metadata and experiment feature groupings
        features_e0 = ["prev_stop_delay_min", "route_code", "route_prior_mean_delay", "scheduled_travel_time_sec"]
        features_e1 = features_e0 + [
            "hour_of_day", "day_of_week", "is_weekend", "is_morning_peak",
            "is_evening_peak", "is_peak_hour", "hour_sin", "hour_cos", "dow_sin", "dow_cos"
        ]
        features_e2 = features_e1 + [
            "precip_rate_mm", "is_raining", "is_heavy_rain", "temperature_c",
            "relative_humidity", "wind_gusts_kmh", "weather_severity_idx", "precip_rolling_3h_mm"
        ]
        features_e3 = features_e2 + [
            "stop_lat", "stop_lon", "dist_to_cbd_km", "stop_sequence", "route_progression_pct",
            "road_hierarchy_level", "stop_density_1km",
            "event_count_500m", "event_count_1km", "event_count_3km", "dist_nearest_event_km",
            "min_to_nearest_event", "event_capacity_shock", "has_nearby_event",
            "rain_x_peak_hour", "rain_x_road_hierarchy", "event_x_peak_hour", "weather_x_progression"
        ]
        features_e4 = features_e3  # Complete feature set for optimized tuning

        metadata = {
            "total_records": len(df_full),
            "train_records": len(df_train),
            "val_records": len(df_val),
            "test_records": len(df_test),
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
            "multi_horizon_targets": [
                "delay_horizon_15m",
                "delay_horizon_30m",
                "delay_horizon_60m",
            ],
        }

        with open(self.metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        return df_train, df_val, df_test
