"""Calibrated Transit Delay Simulator for Controlled Spatiotemporal Experiments.

Simulates transit operational delay distributions across real BMTC routes and stops,
conditioned on real Open-Meteo precipitation, empirical diurnal traffic curves,
and spatial event shockwaves.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from transitpulse.config import settings
from transitpulse.utils.logger import get_logger

logger = get_logger(__name__)


def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculates great-circle distance between two points on the Earth (km)."""
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return 6371.0 * float(c)


class TransitDelaySimulator:
    """Generates calibrated spatiotemporal delay records with verified physics."""

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
        """Simulates operational vehicle runs across dates, hours, and routes."""
        logger.info(
            "Starting calibrated transit delay simulation from %s to %s...",
            start_date, end_date
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
        simulated_records = []
        trip_counter = 1000

        # Hours of operation: 06:00 to 22:00
        service_hours = list(range(6, 23))

        for date in dates:
            for hour in service_hours:
                # Retrieve weather at current hour
                ts_key = f"{date.strftime('%Y-%m-%d')} {hour:02d}:00:00"
                w_info = weather_lookup.get(ts_key, {
                    "precip": 0.0,
                    "temp": 26.0,
                    "humidity": 65.0,
                    "wind_gusts": 8.0,
                    "weather_code": 0,
                })
                precip = w_info["precip"]

                for _, route in df_routes.iterrows():
                    route_id = route["route_id"]
                    route_stops = df_stops[df_stops["route_id"] == route_id].sort_values("stop_sequence")
                    if len(route_stops) < 2:
                        continue

                    for _ in range(trips_per_hour):
                        trip_counter += 1
                        trip_id = f"TRIP-{route_id}-{trip_counter}"
                        minute_offset = np.random.randint(0, 60)
                        trip_start = date.tz_localize(settings.TIMEZONE) + pd.Timedelta(hours=hour, minutes=minute_offset)

                        accumulated_delay = np.random.normal(1.0, 0.5)  # initial terminal departure delay
                        current_time = trip_start

                        prev_stop_id = None
                        prev_stop_delay = 0.0

                        for _, stop in route_stops.iterrows():
                            stop_id = stop["stop_id"]
                            seq = stop["stop_sequence"]
                            sched_sec = stop["scheduled_travel_time_sec"]
                            stop_lat = stop["stop_lat"]
                            stop_lon = stop["stop_lon"]

                            # Advance time by scheduled traversal time
                            current_time = current_time + pd.Timedelta(seconds=sched_sec)

                            # 1. Base route congestion (log-normal)
                            delta_base = np.random.lognormal(mean=0.3, sigma=0.25)

                            # 2. Peak hour friction (Morning: 08-11, Evening: 17-21)
                            h = current_time.hour + current_time.minute / 60.0
                            is_morning_peak = 1.0 if 8.0 <= h <= 11.0 else 0.0
                            is_evening_peak = 1.0 if 17.0 <= h <= 21.0 else 0.0
                            peak_shock = 3.5 * is_morning_peak + 5.0 * is_evening_peak

                            # Route 500-D (ORR IT Corridor) and 335-E experience heavier peak congestion
                            if route_id in ["500-D", "335-E"]:
                                peak_shock *= 1.4

                            # 3. Weather Friction (Non-linear rainfall shock)
                            # Light rain (<2mm): mild delay. Heavy rain (>10mm): severe waterlogging
                            if precip > 0.0:
                                weather_shock = 1.5 * precip + (0.5 * precip ** 1.3)
                            else:
                                weather_shock = 0.0

                            # Interaction: rain during evening peak causes gridlock
                            if precip > 5.0 and is_evening_peak:
                                weather_shock *= 1.8

                            # 4. Urban Event Shockwave
                            event_shock = 0.0
                            nearest_event_dist = 999.0
                            nearest_event_min_start = 9999.0

                            for _, ev in df_events.iterrows():
                                dist_km = haversine_distance_km(stop_lat, stop_lon, ev["venue_lat"], ev["venue_lon"])
                                if dist_km < nearest_event_dist:
                                    nearest_event_dist = dist_km
                                    diff_min = (ev["start_time"] - current_time).total_seconds() / 60.0
                                    nearest_event_min_start = diff_min

                                # Shock applies if event is active or starting within 90 minutes
                                ev_start = ev["start_time"] - pd.Timedelta(minutes=90)
                                ev_end = ev["end_time"] + pd.Timedelta(minutes=60)
                                if ev_start <= current_time <= ev_end:
                                    if dist_km <= 3.0:
                                        # Inverse distance gravity decay
                                        decay = 1.0 / (1.0 + (dist_km / 1.0) ** 2)
                                        capacity_factor = float(ev["expected_attendance"]) / 10000.0
                                        event_shock += 4.0 * decay * capacity_factor

                            # 5. Stochastic noise
                            epsilon = np.random.normal(0.0, 0.6)

                            # 6. Autoregressive propagation along stops
                            if seq > 1:
                                rho = 0.85  # 85% delay carried over from upstream stop
                                accumulated_delay = (
                                    accumulated_delay * rho +
                                    delta_base +
                                    peak_shock * 0.4 +
                                    weather_shock * 0.4 +
                                    event_shock +
                                    epsilon
                                )
                            else:
                                accumulated_delay = max(accumulated_delay, 0.0)

                            accumulated_delay = float(np.clip(accumulated_delay, -2.0, 55.0))

                            # Delay classification
                            if accumulated_delay <= 5.0:
                                delay_class = "On-time"
                            elif accumulated_delay <= 10.0:
                                delay_class = "Minor"
                            elif accumulated_delay <= 20.0:
                                delay_class = "Moderate"
                            else:
                                delay_class = "Severe"

                            simulated_records.append({
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
                                "prev_stop_id": prev_stop_id,
                                "prev_stop_delay_min": prev_stop_delay,
                                "observed_delay_min": np.round(accumulated_delay, 2),
                                "delay_category": delay_class,
                                "precip_mm": np.round(precip, 2),
                                "temperature_c": np.round(w_info["temp"], 1),
                                "relative_humidity": np.round(w_info["humidity"], 1),
                                "wind_gusts_kmh": np.round(w_info["wind_gusts"], 1),
                                "weather_code": w_info["weather_code"],
                                "dist_nearest_event_km": np.round(nearest_event_dist, 2),
                                "min_to_nearest_event": np.round(nearest_event_min_start, 1),
                                "dataset_mode": "simulated",
                            })

                            prev_stop_id = stop_id
                            prev_stop_delay = accumulated_delay

        df_sim = pd.DataFrame(simulated_records)
        df_sim.to_parquet(self.output_file, index=False)
        logger.info(
            "Successfully simulated %d spatiotemporal stop arrival records across %d trips.",
            len(df_sim), trip_counter - 1000
        )
        return df_sim

    def get_simulated_trips(self) -> pd.DataFrame:
        """Loads cached simulated trips."""
        if not self.output_file.exists():
            raise FileNotFoundError(f"Simulated trips file not found at {self.output_file}")
        return pd.read_parquet(self.output_file)
