"""Geospatial Feature Engineering Module."""

import numpy as np
import pandas as pd

from transitpulse.config import settings


def haversine_vec(lat1, lon1, lat2, lon2):
    """Vectorized haversine distance in kilometers."""
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return 6371.0 * c


def extract_spatial_features(df: pd.DataFrame, df_stops: pd.DataFrame) -> pd.DataFrame:
    """Extracts geographic proximity, road hierarchy, and stop progression features."""
    df_feat = pd.DataFrame(index=df.index)

    lats = df["stop_lat"].astype(np.float64)
    lons = df["stop_lon"].astype(np.float64)

    df_feat["stop_lat"] = lats.astype(np.float32)
    df_feat["stop_lon"] = lons.astype(np.float32)

    # 1. Distance to Bengaluru Central Business District (MG Road / Vidhana Soudha Centroid)
    df_feat["dist_to_cbd_km"] = haversine_vec(
        lats, lons, settings.CENTROID_LAT, settings.CENTROID_LON
    ).astype(np.float32)

    # 2. Stop sequence progression fraction
    # Sequence relative to route length
    route_max_seq = df.groupby("route_id")["stop_sequence"].transform("max")
    df_feat["stop_sequence"] = df["stop_sequence"].astype(np.int32)
    df_feat["route_progression_pct"] = (df["stop_sequence"] / route_max_seq).astype(np.float32)

    # 3. Scheduled travel time (sec)
    df_feat["scheduled_travel_time_sec"] = (
        df["scheduled_travel_time_sec"].fillna(0.0).astype(np.float32)
    )

    # 4. Road hierarchy proxy (Arterial / Outer Ring Road vs Radial)
    # ORR (route 500-D) and Airport Road (KIA-9) are high-capacity express corridors (Hierarchy 4)
    hierarchy = np.full(len(df), 2, dtype=np.int8)  # Secondary arterial default
    hierarchy[df["route_id"] == "500-D"] = 4  # Trunk / Ring highway
    hierarchy[df["route_id"] == "KIA-9"] = 4  # Airport Expressway
    hierarchy[df["route_id"] == "335-E"] = 3  # Major Radial IT Corridor
    df_feat["road_hierarchy_level"] = hierarchy

    # 5. Local stop density proxy (stops within 1.5 km in BMTC topology)
    all_lats = df_stops["stop_lat"].values
    all_lons = df_stops["stop_lon"].values
    densities = []
    for lat, lon in zip(lats, lons):
        dists = haversine_vec(lat, lon, all_lats, all_lons)
        densities.append(int((dists <= 1.5).sum()))
    df_feat["stop_density_1km"] = np.array(densities, dtype=np.int32)

    return df_feat
