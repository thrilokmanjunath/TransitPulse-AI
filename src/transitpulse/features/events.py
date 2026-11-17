"""Urban Event Feature Engineering Module."""

import numpy as np
import pandas as pd

from transitpulse.features.spatial import haversine_vec


def extract_event_features(df: pd.DataFrame, df_events: pd.DataFrame) -> pd.DataFrame:
    """Extracts spatial event proximity rings and temporal countdown features."""
    df_feat = pd.DataFrame(index=df.index)

    lats = df["stop_lat"].values
    lons = df["stop_lon"].values
    sched_times = pd.to_datetime(df["scheduled_time"])

    event_lats = df_events["venue_lat"].values
    event_lons = df_events["venue_lon"].values
    event_starts = pd.to_datetime(df_events["start_time"]).values
    event_ends = pd.to_datetime(df_events["end_time"]).values
    attendances = df_events["expected_attendance"].fillna(10000).values

    counts_500m = np.zeros(len(df), dtype=np.int32)
    counts_1km = np.zeros(len(df), dtype=np.int32)
    counts_3km = np.zeros(len(df), dtype=np.int32)
    min_dist_km = np.full(len(df), 999.0, dtype=np.float32)
    min_to_start = np.full(len(df), 9999.0, dtype=np.float32)
    active_capacity_shock = np.zeros(len(df), dtype=np.float32)

    for i in range(len(df)):
        cur_t = sched_times.iloc[i].to_datetime64()
        cur_lat = lats[i]
        cur_lon = lons[i]

        dists = haversine_vec(cur_lat, cur_lon, event_lats, event_lons)
        min_dist_km[i] = np.min(dists)

        # Check events within ±4 hours of scheduled stop time
        for ev_idx, dist in enumerate(dists):
            ev_st = event_starts[ev_idx]
            ev_en = event_ends[ev_idx]
            dt_hours = (cur_t - ev_st) / np.timedelta64(1, "h")

            # Within window of influence: -2 hours before to +1 hour after
            if -2.0 <= dt_hours <= 4.0:
                if dist <= 0.5:
                    counts_500m[i] += 1
                if dist <= 1.0:
                    counts_1km[i] += 1
                if dist <= 3.0:
                    counts_3km[i] += 1
                    # Capacity gravity shock
                    decay = 1.0 / (1.0 + (dist / 1.0) ** 2)
                    active_capacity_shock[i] += decay * (attendances[ev_idx] / 10000.0)

                diff_min = (ev_st - cur_t) / np.timedelta64(1, "m")
                if abs(diff_min) < abs(min_to_start[i]):
                    min_to_start[i] = float(diff_min)

    df_feat["event_count_500m"] = counts_500m
    df_feat["event_count_1km"] = counts_1km
    df_feat["event_count_3km"] = counts_3km
    df_feat["dist_nearest_event_km"] = min_dist_km
    df_feat["min_to_nearest_event"] = min_to_start
    df_feat["event_capacity_shock"] = active_capacity_shock.astype(np.float32)
    df_feat["has_nearby_event"] = (counts_3km > 0).astype(np.int8)

    return df_feat
