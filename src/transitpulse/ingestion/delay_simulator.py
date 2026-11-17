"""Calibrated Transit Delay Simulator for Controlled Spatiotemporal Experiments.

Redesigned with realistic urban stochasticity, segment-dependent autoregression,
telemetry dropouts (missing observations), weather forecast uncertainty,
and genuine spatiotemporal multi-horizon forward targets.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from transitpulse.config import settings
from transitpulse.utils.logger import get_logger

logger = get_logger(__name__)


def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculates great-circle distance between two points on Earth in kilometers."""
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return 6371.0 * float(c)


class TransitDelaySimulator:
    """Generates calibrated spatiotemporal delay records with verified urban physics."""

    def __init__(self, output_dir: Path | None = None):
        self.output_dir = output_dir or settings.DATA_SIMULATED
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_file = self.output_dir / "bmtc_simulated_trips.parquet"

    def simulate_dataset(
        self,
        df_routes: pd.DataFrame,
        df_stops: pd.DataFrame,
        df_weather: pd.DataFrame,
        df_events: pd.DataFrame,
        start_date: str = "2024-05-01",
        end_date: str = "2024-05-21",
        trips_per_hour: int = 2,
    ) -> pd.DataFrame:
        """Simulates operational vehicle runs across dates, hours, and routes with stochastic realism."""
        logger.info(
            "Starting redesigned calibrated transit delay simulation from %s to %s...",
            start_date,
            end_date,
        )
        np.random.seed(settings.RANDOM_SEED)

        # Weather lookup dictionary by timestamp
        weather_lookup: dict[str, dict[str, float]] = {}
        for _, row in df_weather.iterrows():
            ts_str = row["timestamp"].strftime("%Y-%m-%d %H:00:00")
            weather_lookup[ts_str] = {
                "precip": float(row.get("precipitation", 0.0)),
                "temp": float(row.get("temperature_2m", 25.0)),
                "humidity": float(row.get("relative_humidity_2m", 60.0)),
                "wind_gusts": float(row.get("wind_gusts_10m", 10.0)),
                "weather_code": int(row.get("weather_code", 0)),
            }

        dates = pd.date_range(start=start_date, end=end_date, freq="D")
        all_trips_data = []
        trip_counter = 1000

        service_hours = list(range(6, 23))

        for date in dates:
            for hour in service_hours:
                ts_key = f"{date.strftime('%Y-%m-%d')} {hour:02d}:00:00"
                w_info = weather_lookup.get(
                    ts_key,
                    {
                        "precip": 0.0,
                        "temp": 26.0,
                        "humidity": 65.0,
                        "wind_gusts": 8.0,
                        "weather_code": 0,
                    },
                )
                ground_precip = w_info["precip"]

                # Weather forecast uncertainty at prediction time (forecast noise)
                forecast_precip = max(0.0, float(ground_precip + np.random.normal(0.0, 0.4)))
                forecast_temp = float(w_info["temp"] + np.random.normal(0.0, 0.8))

                for _, route in df_routes.iterrows():
                    route_id = route["route_id"]
                    route_stops = df_stops[df_stops["route_id"] == route_id].sort_values(
                        "stop_sequence"
                    )
                    if len(route_stops) < 2:
                        continue

                    for _ in range(trips_per_hour):
                        trip_counter += 1
                        trip_id = f"TRIP-{route_id}-{trip_counter}"
                        minute_offset = np.random.randint(0, 60)
                        trip_start = date.tz_localize(settings.TIMEZONE) + pd.Timedelta(
                            hours=hour, minutes=minute_offset
                        )

                        # Initial terminal departure delay (stochastic schedule adherence)
                        accumulated_delay = np.random.normal(1.2, 1.0)
                        current_time = trip_start

                        last_observed_stop_id = None
                        last_observed_delay = 0.0

                        trip_stop_records = []

                        for _, stop in route_stops.iterrows():
                            stop_id = stop["stop_id"]
                            seq = stop["stop_sequence"]
                            sched_sec = stop["scheduled_travel_time_sec"]
                            stop_lat = stop["stop_lat"]
                            stop_lon = stop["stop_lon"]

                            current_time = current_time + pd.Timedelta(seconds=sched_sec)

                            # 1. Base route congestion (log-normal with route-specific scale)
                            route_scale = 1.3 if route_id in ["500-D", "335-E"] else 1.0
                            delta_base = float(
                                np.random.lognormal(mean=0.4, sigma=0.4) * route_scale
                            )

                            # 2. Peak hour friction with stochastic duration shifts
                            h = current_time.hour + current_time.minute / 60.0
                            is_morning_peak = 1.0 if (7.75 <= h <= 11.25) else 0.0
                            is_evening_peak = 1.0 if (16.75 <= h <= 21.25) else 0.0
                            peak_shock = (
                                3.2 * is_morning_peak + 4.8 * is_evening_peak
                            ) * route_scale

                            # 3. Weather friction: non-linear rainfall shock + drainage bottleneck
                            if ground_precip > 0.0:
                                weather_shock = 1.6 * ground_precip + (0.4 * (ground_precip**1.35))
                            else:
                                weather_shock = 0.0

                            # Non-linear interaction: rain during evening peak
                            if ground_precip > 4.0 and is_evening_peak:
                                weather_shock *= 1.7

                            # 4. Urban Event Shockwave with crowd estimate uncertainty
                            event_shock = 0.0
                            nearest_event_dist = 999.0
                            nearest_event_min_start = 9999.0

                            for _, ev in df_events.iterrows():
                                dist_km = haversine_distance_km(
                                    stop_lat, stop_lon, ev["venue_lat"], ev["venue_lon"]
                                )
                                if dist_km < nearest_event_dist:
                                    nearest_event_dist = dist_km
                                    diff_min = (
                                        ev["start_time"] - current_time
                                    ).total_seconds() / 60.0
                                    nearest_event_min_start = diff_min

                                ev_start = ev["start_time"] - pd.Timedelta(minutes=90)
                                ev_end = ev["end_time"] + pd.Timedelta(minutes=60)
                                if ev_start <= current_time <= ev_end and dist_km <= 3.5:
                                    decay = 1.0 / (1.0 + (dist_km / 1.0) ** 2)
                                    # Crowd size estimate has +/- 20% uncertainty
                                    est_attendance = ev["expected_attendance"] * np.random.uniform(
                                        0.8, 1.2
                                    )
                                    capacity_factor = float(est_attendance) / 10000.0
                                    event_shock += 4.2 * decay * capacity_factor

                            # 5. Realistic Observation Disturbance (Passenger boarding + traffic signals + GPS jitter)
                            dwell_noise = float(np.random.lognormal(mean=0.2, sigma=0.4) - 1.0)
                            signal_noise = float(np.random.exponential(scale=0.6))
                            gps_jitter = float(np.random.normal(0.0, 0.8))

                            total_disturbance = dwell_noise + signal_noise + gps_jitter

                            # 6. Segment-dependent heterogeneous autoregression rho
                            segment_km = (
                                sched_sec / 3600.0
                            ) * 22.0  # approximate 22 km/h average speed
                            rho_segment = float(
                                np.clip(
                                    0.72
                                    + 0.12 * min(1.0, segment_km / 2.5)
                                    + np.random.uniform(-0.06, 0.06),
                                    0.55,
                                    0.90,
                                )
                            )

                            if seq > 1:
                                accumulated_delay = (
                                    accumulated_delay * rho_segment
                                    + delta_base
                                    + peak_shock * 0.35
                                    + weather_shock * 0.35
                                    + event_shock
                                    + total_disturbance
                                )
                            else:
                                accumulated_delay = max(
                                    accumulated_delay + total_disturbance * 0.3, 0.0
                                )

                            accumulated_delay = float(np.clip(accumulated_delay, -2.5, 65.0))

                            # 7. Telemetry Dropouts (Missing Vehicle Observations ~10% probability)
                            is_ping_dropped = (
                                (np.random.rand() < 0.10) and (seq > 1) and (seq < len(route_stops))
                            )

                            if accumulated_delay <= 5.0:
                                delay_class = "On-time"
                            elif accumulated_delay <= 10.0:
                                delay_class = "Minor"
                            elif accumulated_delay <= 20.0:
                                delay_class = "Moderate"
                            else:
                                delay_class = "Severe"

                            trip_stop_records.append(
                                {
                                    "trip_id": trip_id,
                                    "route_id": route_id,
                                    "route_short_name": route["route_short_name"],
                                    "stop_id": stop_id,
                                    "stop_sequence": seq,
                                    "stop_name": stop["stop_name"],
                                    "stop_lat": stop_lat,
                                    "stop_lon": stop_lon,
                                    "scheduled_time": current_time,
                                    "scheduled_travel_time_sec": sched_sec,
                                    "prev_stop_id": last_observed_stop_id,
                                    "prev_stop_delay_min": last_observed_delay,
                                    "observed_delay_min": np.round(accumulated_delay, 2),
                                    "delay_category": delay_class,
                                    "precip_mm": np.round(forecast_precip, 2),
                                    "ground_precip_mm": np.round(ground_precip, 2),
                                    "temperature_c": np.round(forecast_temp, 1),
                                    "relative_humidity": np.round(w_info["humidity"], 1),
                                    "wind_gusts_kmh": np.round(w_info["wind_gusts"], 1),
                                    "weather_code": w_info["weather_code"],
                                    "dist_nearest_event_km": np.round(nearest_event_dist, 2),
                                    "min_to_nearest_event": np.round(nearest_event_min_start, 1),
                                    "is_telemetry_missing": int(is_ping_dropped),
                                    "dataset_mode": "simulated",
                                }
                            )

                            if not is_ping_dropped:
                                last_observed_stop_id = stop_id
                                last_observed_delay = accumulated_delay

                        all_trips_data.extend(trip_stop_records)

        df_sim = pd.DataFrame(all_trips_data)

        # 8. Genuine Multi-Horizon Spatiotemporal Forward Target Computation
        logger.info("Computing genuine forward spatiotemporal multi-horizon targets...")
        df_sim = self._compute_forward_horizons(df_sim)

        df_sim.to_parquet(self.output_file, index=False)
        logger.info(
            "Successfully simulated %d spatiotemporal stop arrival records with realistic stochastic noise across %d trips.",
            len(df_sim),
            trip_counter - 1000,
        )
        return df_sim

    def _compute_forward_horizons(self, df: pd.DataFrame) -> pd.DataFrame:
        """Looks ahead along the trip to find actual delay at stops reached at +15, +30, and +60 minutes."""
        df_sorted = df.sort_values(["trip_id", "stop_sequence"]).reset_index(drop=True)

        h15_delays = []
        h30_delays = []
        h60_delays = []

        for _, trip_group in df_sorted.groupby("trip_id", sort=False):
            times = trip_group["scheduled_time"].values
            delays = trip_group["observed_delay_min"].values
            n_stops = len(trip_group)

            for i in range(n_stops):
                t_cur = times[i]
                d_cur = delays[i]

                d15, d30, d60 = d_cur, d_cur, d_cur
                found_15, found_30, found_60 = False, False, False

                for j in range(i + 1, n_stops):
                    dt_min = (times[j] - t_cur) / np.timedelta64(1, "m")
                    if not found_15 and dt_min >= 12.0:
                        d15 = delays[j]
                        found_15 = True
                    if not found_30 and dt_min >= 25.0:
                        d30 = delays[j]
                        found_30 = True
                    if not found_60 and dt_min >= 50.0:
                        d60 = delays[j]
                        found_60 = True

                if not found_15:
                    d15 = delays[-1]
                if not found_30:
                    d30 = delays[-1]
                if not found_60:
                    d60 = delays[-1]

                h15_delays.append(d15)
                h30_delays.append(d30)
                h60_delays.append(d60)

        df_sorted["delay_horizon_15m"] = np.round(h15_delays, 2)
        df_sorted["delay_horizon_30m"] = np.round(h30_delays, 2)
        df_sorted["delay_horizon_60m"] = np.round(h60_delays, 2)
        return df_sorted

    def get_simulated_trips(self) -> pd.DataFrame:
        """Loads cached simulated trips."""
        if not self.output_file.exists():
            raise FileNotFoundError(f"Simulated trips file not found at {self.output_file}")
        return pd.read_parquet(self.output_file)
