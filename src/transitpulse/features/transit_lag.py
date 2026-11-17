"""Transit Lag and Cross-Modal Interaction Feature Engineering."""

import numpy as np
import pandas as pd


def extract_transit_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extracts upstream stop delay and route identifier encodings.

    Strictly zero-leakage: only observed delay at the PREVIOUS upstream stop
    is used as a feature for predicting the current stop.
    """
    df_feat = pd.DataFrame(index=df.index)

    # Previous stop delay
    df_feat["prev_stop_delay_min"] = df["prev_stop_delay_min"].fillna(0.0).astype(np.float32)

    # Route identity categorical codes
    route_mapping = {
        "335-E": 1,
        "500-D": 2,
        "KIA-9": 3,
        "G-2": 4,
        "201-R": 5,
    }
    df_feat["route_code"] = df["route_id"].map(route_mapping).fillna(0).astype(np.int32)

    # Historical average route delay baseline (empirical priors from dispatch schedules)
    route_priors = {
        "335-E": 4.5,
        "500-D": 6.8,
        "KIA-9": 3.2,
        "G-2": 5.1,
        "201-R": 3.9,
    }
    df_feat["route_prior_mean_delay"] = (
        df["route_id"].map(route_priors).fillna(4.0).astype(np.float32)
    )

    return df_feat


def extract_interaction_features(
    df_temp: pd.DataFrame,
    df_weather: pd.DataFrame,
    df_spatial: pd.DataFrame,
    df_event: pd.DataFrame,
) -> pd.DataFrame:
    """Computes cross-modal physical interaction terms."""
    df_feat = pd.DataFrame(index=df_temp.index)

    # 1. Rain x Peak Hour Interaction (Major urban bottleneck amplifier)
    df_feat["rain_x_peak_hour"] = (df_weather["precip_rate_mm"] * df_temp["is_peak_hour"]).astype(
        np.float32
    )

    # 2. Rain x Road Hierarchy (Drainage & speed drop on high-speed ring roads)
    df_feat["rain_x_road_hierarchy"] = (
        df_weather["precip_rate_mm"] * df_spatial["road_hierarchy_level"]
    ).astype(np.float32)

    # 3. Event Shock x Peak Hour (Stadium exits colliding with evening office traffic)
    df_feat["event_x_peak_hour"] = (
        df_event["event_capacity_shock"] * df_temp["is_evening_peak"]
    ).astype(np.float32)

    # 4. Weather Severity x Stop Progression (Cumulative cascade along route)
    df_feat["weather_x_progression"] = (
        df_weather["weather_severity_idx"] * df_spatial["route_progression_pct"]
    ).astype(np.float32)

    return df_feat
