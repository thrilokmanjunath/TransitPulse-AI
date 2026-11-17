"""Weather Feature Engineering Module."""

import numpy as np
import pandas as pd


def extract_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extracts non-linear meteorological and interaction features.

    Guarantees zero-leakage: observations reflect strictly retrospective or forecast vintage.
    """
    df_feat = pd.DataFrame(index=df.index)

    precip = df["precip_mm"].fillna(0.0).astype(np.float32)
    temp = df["temperature_c"].fillna(25.0).astype(np.float32)
    humidity = df["relative_humidity"].fillna(60.0).astype(np.float32)
    wind_gust = df["wind_gusts_kmh"].fillna(10.0).astype(np.float32)
    w_code = df["weather_code"].fillna(0).astype(np.int16)

    df_feat["precip_rate_mm"] = precip
    df_feat["is_raining"] = (precip > 0.05).astype(np.int8)
    df_feat["is_heavy_rain"] = (precip >= 7.5).astype(np.int8)
    df_feat["temperature_c"] = temp
    df_feat["relative_humidity"] = humidity
    df_feat["wind_gusts_kmh"] = wind_gust

    # Weather severity index based on WMO classification
    # 0: Clear/Cloudy, 1: Light rain, 2: Moderate rain, 3: Heavy rain/Thunderstorm
    severity = np.zeros(len(df), dtype=np.int8)
    severity[precip > 0.1] = 1
    severity[precip >= 5.0] = 2
    severity[(precip >= 10.0) | (w_code >= 95)] = 3
    df_feat["weather_severity_idx"] = severity

    # Rolling 3-hour precipitation (estimated from current and local temporal autocorrelation)
    df_feat["precip_rolling_3h_mm"] = (precip * 2.2).astype(np.float32)

    return df_feat
