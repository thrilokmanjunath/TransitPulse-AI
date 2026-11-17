"""TransitPulse-AI Configuration Module.

Central configuration settings, geographic constants, file paths,
and operational parameters for Bengaluru transit delay prediction.
"""

from pathlib import Path
from typing import Any

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Platform-wide settings and directory paths."""

    # Base paths
    PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent
    DATA_DIR: Path = PROJECT_ROOT / "data"
    DATA_RAW: Path = DATA_DIR / "raw"
    DATA_INTERIM: Path = DATA_DIR / "interim"
    DATA_PROCESSED: Path = DATA_DIR / "processed"
    DATA_SIMULATED: Path = DATA_DIR / "simulated"
    DATA_EXTERNAL: Path = DATA_DIR / "external"

    MODELS_DIR: Path = PROJECT_ROOT / "models"
    ARTIFACTS_DIR: Path = MODELS_DIR / "artifacts"
    REPORTS_DIR: Path = PROJECT_ROOT / "reports"
    LOGS_DIR: Path = PROJECT_ROOT / "logs"

    # Environment & Reproducibility
    ENVIRONMENT: str = "production"
    RANDOM_SEED: int = 42
    TIMEZONE: str = "Asia/Kolkata"

    # Geographic Scope: Bengaluru, Karnataka, India
    CITY_NAME: str = "Bengaluru"
    LAT_MIN: float = 12.75
    LAT_MAX: float = 13.25
    LON_MIN: float = 77.40
    LON_MAX: float = 77.85
    CENTROID_LAT: float = 12.9716
    CENTROID_LON: float = 77.5946

    # Pipeline Latency & Leakage Prevention
    INGESTION_LATENCY_MINUTES: int = 2

    # Multi-Horizon Targets (Minutes)
    FORECAST_HORIZONS: list[int] = [15, 30, 60]

    # Delay Classification Thresholds (Minutes)
    # On-time: <= 5m, Minor: (5, 10], Moderate: (10, 20], Severe: > 20m
    DELAY_CLASSES: list[str] = ["On-time", "Minor", "Moderate", "Severe"]
    CLASS_THRESHOLDS: list[float] = [5.0, 10.0, 20.0]

    # Open-Meteo Configuration
    OPEN_METEO_HISTORICAL_URL: str = "https://archive-api.open-meteo.com/v1/archive"
    OPEN_METEO_FORECAST_URL: str = "https://api.open-meteo.com/v1/forecast"
    WEATHER_VARIABLES: list[str] = [
        "temperature_2m",
        "relative_humidity_2m",
        "precipitation",
        "rain",
        "wind_speed_10m",
        "wind_gusts_10m",
        "weather_code",
    ]

    # Major Public Event Venues in Bengaluru
    BENGALURU_VENUES: dict[str, dict[str, Any]] = {
        "chinnaswamy_stadium": {
            "name": "M. Chinnaswamy Stadium (Cricket)",
            "lat": 12.9788,
            "lon": 77.5996,
            "capacity": 38000,
            "category": "sports",
        },
        "palace_grounds": {
            "name": "Bengaluru Palace Grounds",
            "lat": 13.0034,
            "lon": 77.5891,
            "capacity": 50000,
            "category": "concert",
        },
        "biec": {
            "name": "Bangalore Int. Exhibition Centre (BIEC)",
            "lat": 13.0631,
            "lon": 77.4735,
            "capacity": 30000,
            "category": "exhibition",
        },
        "kanteerava_stadium": {
            "name": "Sree Kanteerava Outdoor Stadium",
            "lat": 12.9698,
            "lon": 77.5926,
            "capacity": 25000,
            "category": "sports",
        },
        "ktpo_whitefield": {
            "name": "KTPO Exhibition Complex Whitefield",
            "lat": 12.9782,
            "lon": 77.7289,
            "capacity": 12000,
            "category": "exhibition",
        },
        "koramangala_indoor": {
            "name": "Koramangala Indoor Stadium",
            "lat": 12.9352,
            "lon": 77.6245,
            "capacity": 6000,
            "category": "sports",
        },
    }

    # API Settings (Default loopback for secure local execution; override to 0.0.0.0 in Docker)
    API_HOST: str = "127.0.0.1"
    API_PORT: int = 8000

    # Streamlit Dashboard Settings
    DASHBOARD_PORT: int = 8501

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()

# Ensure required runtime directories exist
for directory in [
    settings.DATA_RAW,
    settings.DATA_INTERIM,
    settings.DATA_PROCESSED,
    settings.DATA_SIMULATED,
    settings.DATA_EXTERNAL,
    settings.ARTIFACTS_DIR,
    settings.REPORTS_DIR,
    settings.LOGS_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)
