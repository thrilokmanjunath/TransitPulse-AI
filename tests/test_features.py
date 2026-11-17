"""Tests for Spatiotemporal Feature Engineering."""

import pandas as pd

from transitpulse.config import settings
from transitpulse.features.temporal import extract_temporal_features
from transitpulse.features.weather import extract_weather_features


def test_temporal_feature_extraction():
    df = pd.DataFrame({
        "scheduled_time": pd.to_datetime([
            "2024-05-18 09:30:00",  # Morning peak
            "2024-05-18 14:00:00",  # Midday off-peak
            "2024-05-18 18:45:00",  # Evening peak
        ]).tz_localize(settings.TIMEZONE)
    })
    feat = extract_temporal_features(df)
    assert "is_peak_hour" in feat.columns
    assert "hour_sin" in feat.columns
    assert "hour_cos" in feat.columns
    assert feat["is_peak_hour"].tolist() == [1, 0, 1]
    assert feat["is_morning_peak"].tolist() == [1, 0, 0]
    assert feat["is_evening_peak"].tolist() == [0, 0, 1]


def test_weather_feature_extraction():
    df = pd.DataFrame({
        "precip_mm": [0.0, 2.5, 12.0],
        "temperature_c": [28.0, 24.0, 21.0],
        "relative_humidity": [50.0, 80.0, 95.0],
        "wind_gusts_kmh": [8.0, 22.0, 35.0],
        "weather_code": [0, 61, 95],
    })
    feat = extract_weather_features(df)
    assert "is_heavy_rain" in feat.columns
    assert "weather_severity_idx" in feat.columns
    assert feat["is_heavy_rain"].tolist() == [0, 0, 1]
    assert feat["weather_severity_idx"].tolist() == [0, 1, 3]
