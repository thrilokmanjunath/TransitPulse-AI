"""Temporal Feature Engineering Module."""

import numpy as np
import pandas as pd


def extract_temporal_features(df: pd.DataFrame, time_col: str = "scheduled_time") -> pd.DataFrame:
    """Extracts calendar, diurnal, cyclical, and peak-hour transit features.

    Guarantees zero-leakage: relies purely on scheduled clock timestamp.
    """
    df_feat = pd.DataFrame(index=df.index)
    dt = pd.to_datetime(df[time_col])

    # 1. Base calendar fields
    df_feat["hour_of_day"] = dt.dt.hour.astype(np.int32)
    df_feat["day_of_week"] = dt.dt.dayofweek.astype(np.int32)
    df_feat["day_of_month"] = dt.dt.day.astype(np.int32)
    df_feat["month"] = dt.dt.month.astype(np.int32)
    df_feat["is_weekend"] = (dt.dt.dayofweek >= 5).astype(np.int8)

    # 2. Peak Hour Indicators (Bengaluru Peak Commute Windows)
    # Morning: 08:00 - 11:30, Evening: 17:00 - 21:00
    h_float = dt.dt.hour + dt.dt.minute / 60.0
    is_morning = (h_float >= 8.0) & (h_float <= 11.5)
    is_evening = (h_float >= 17.0) & (h_float <= 21.0)
    df_feat["is_morning_peak"] = is_morning.astype(np.int8)
    df_feat["is_evening_peak"] = is_evening.astype(np.int8)
    df_feat["is_peak_hour"] = (is_morning | is_evening).astype(np.int8)

    # 3. Continuous cyclical transformations (sine and cosine)
    # Hour of day (24-hour periodicity)
    hour_rad = 2.0 * np.pi * df_feat["hour_of_day"] / 24.0
    df_feat["hour_sin"] = np.sin(hour_rad).astype(np.float32)
    df_feat["hour_cos"] = np.cos(hour_rad).astype(np.float32)

    # Day of week (7-day periodicity)
    dow_rad = 2.0 * np.pi * df_feat["day_of_week"] / 7.0
    df_feat["dow_sin"] = np.sin(dow_rad).astype(np.float32)
    df_feat["dow_cos"] = np.cos(dow_rad).astype(np.float32)

    return df_feat
