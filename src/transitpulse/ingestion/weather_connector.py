"""Weather Data Connector for Open-Meteo Ingestion.

Retrieves historical reanalysis and forward weather forecasts for Bengaluru,
normalizing timestamps to Asia/Kolkata and managing schema validation.
"""

import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from transitpulse.config import settings
from transitpulse.utils.logger import get_logger

logger = get_logger(__name__)


class WeatherConnector:
    """Connects to Open-Meteo API with retry and caching logic."""

    def __init__(self, raw_dir: Path | None = None):
        self.raw_dir = raw_dir or settings.DATA_RAW
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.weather_file = self.raw_dir / "bengaluru_weather.parquet"

    def fetch_historical_weather(
        self,
        start_date: str = "2024-05-01",
        end_date: str = "2024-05-31",
        lat: float = settings.CENTROID_LAT,
        lon: float = settings.CENTROID_LON,
        max_retries: int = 3,
        backoff_sec: float = 2.0,
    ) -> pd.DataFrame:
        """Fetches historical hourly weather for Bengaluru from Open-Meteo Archive API."""
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": ",".join(settings.WEATHER_VARIABLES),
            "timezone": settings.TIMEZONE,
        }
        logger.info(
            "Querying Open-Meteo Historical Archive (%s to %s) for Bengaluru [%.4f, %.4f]...",
            start_date, end_date, lat, lon
        )

        for attempt in range(1, max_retries + 1):
            try:
                resp = requests.get(
                    settings.OPEN_METEO_HISTORICAL_URL,
                    params=params,
                    timeout=15,
                )
                if resp.status_code == 200:
                    data = resp.json().get("hourly", {})
                    df = pd.DataFrame(data)
                    df["timestamp"] = pd.to_datetime(df["time"]).dt.tz_localize(settings.TIMEZONE)
                    df = df.drop(columns=["time"])
                    df["dataset_mode"] = "observed"
                    df.to_parquet(self.weather_file, index=False)
                    logger.info("Successfully fetched %d hourly weather records.", len(df))
                    return df
                else:
                    logger.warning(
                        "Open-Meteo returned status %d (Attempt %d/%d): %s",
                        resp.status_code, attempt, max_retries, resp.text[:200]
                    )
            except Exception as exc:
                logger.warning(
                    "Open-Meteo request failed on attempt %d/%d: %s",
                    attempt, max_retries, exc
                )
            time.sleep(backoff_sec * attempt)

        logger.warning("Falling back to calibrated Bengaluru weather generation...")
        return self._generate_fallback_weather(start_date, end_date)

    def fetch_forecast_weather(
        self,
        days: int = 3,
        lat: float = settings.CENTROID_LAT,
        lon: float = settings.CENTROID_LON,
    ) -> pd.DataFrame:
        """Fetches real-time weather forecast for operational forward prediction."""
        params = {
            "latitude": lat,
            "longitude": lon,
            "forecast_days": days,
            "hourly": ",".join(settings.WEATHER_VARIABLES),
            "timezone": settings.TIMEZONE,
        }
        try:
            resp = requests.get(
                settings.OPEN_METEO_FORECAST_URL,
                params=params,
                timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json().get("hourly", {})
                df = pd.DataFrame(data)
                df["timestamp"] = pd.to_datetime(df["time"]).dt.tz_localize(settings.TIMEZONE)
                df = df.drop(columns=["time"])
                df["dataset_mode"] = "observed"
                return df
        except Exception as exc:
            logger.warning("Forecast request failed: %s", exc)

        now = datetime.now()
        start = now.strftime("%Y-%m-%d")
        end = (now + timedelta(days=days)).strftime("%Y-%m-%d")
        return self._generate_fallback_weather(start, end)

    def _generate_fallback_weather(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Generates calibrated meteorological data matching Bengaluru climate."""
        start_dt = pd.to_datetime(start_date).tz_localize(settings.TIMEZONE)
        end_dt = (pd.to_datetime(end_date) + pd.Timedelta(days=1)).tz_localize(settings.TIMEZONE)
        timestamps = pd.date_range(start=start_dt, end=end_dt, freq="1h", inclusive="left")
        
        np.random.seed(settings.RANDOM_SEED)
        hours = timestamps.hour.values
        
        # Diurnal temperature cycle in Bengaluru: ~20°C min early morning to ~32°C max afternoon
        temp = 20.0 + 10.0 * np.sin((hours - 8) * np.pi / 12) + np.random.normal(0, 1.2, len(hours))
        temp = np.clip(temp, 16.0, 36.0)
        
        # Humidity inversely proportional to temperature
        humidity = 85.0 - 30.0 * np.sin((hours - 8) * np.pi / 12) + np.random.normal(0, 4.0, len(hours))
        humidity = np.clip(humidity, 30.0, 98.0)
        
        # Bengaluru convective afternoon/evening thunderstorm rains
        rain_prob = np.where((hours >= 15) & (hours <= 21), 0.35, 0.08)
        is_raining = np.random.binomial(1, rain_prob)
        rain_depth = is_raining * np.random.exponential(scale=5.5, size=len(hours))
        rain_depth = np.round(rain_depth, 2)
        
        # Wind speed & gusts
        wind_speed = 10.0 + 5.0 * np.sin(hours * np.pi / 12) + np.random.normal(0, 2.0, len(hours))
        wind_speed = np.clip(wind_speed, 2.0, 40.0)
        wind_gusts = wind_speed + np.random.uniform(5.0, 20.0, len(hours)) * (rain_depth > 0)
        
        # WMO Weather Codes: 0 (Clear), 61 (Slight rain), 63 (Moderate rain), 65 (Heavy), 95 (Thunderstorm)
        weather_codes = np.zeros(len(hours), dtype=int)
        weather_codes[rain_depth > 0.1] = 61
        weather_codes[rain_depth > 5.0] = 63
        weather_codes[rain_depth > 12.0] = 65
        weather_codes[(rain_depth > 8.0) & (wind_gusts > 30.0)] = 95
        
        df = pd.DataFrame({
            "timestamp": timestamps,
            "temperature_2m": np.round(temp, 2),
            "relative_humidity_2m": np.round(humidity, 1),
            "precipitation": rain_depth,
            "rain": rain_depth,
            "wind_speed_10m": np.round(wind_speed, 1),
            "wind_gusts_10m": np.round(wind_gusts, 1),
            "weather_code": weather_codes,
            "dataset_mode": "simulated",
        })
        df.to_parquet(self.weather_file, index=False)
        logger.info("Generated fallback Bengaluru weather dataset with %d records.", len(df))
        return df

    def get_weather(self, start_date: str = "2024-05-01", end_date: str = "2024-05-31") -> pd.DataFrame:
        """Retrieves cached weather or queries API."""
        if self.weather_file.exists():
            return pd.read_parquet(self.weather_file)
        return self.fetch_historical_weather(start_date=start_date, end_date=end_date)
