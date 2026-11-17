"""End-to-End Data Ingestion Pipeline for TransitPulse-AI."""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from transitpulse.config import settings
from transitpulse.ingestion.delay_simulator import TransitDelaySimulator
from transitpulse.ingestion.event_connector import EventConnector
from transitpulse.ingestion.transit_connector import TransitConnector
from transitpulse.ingestion.weather_connector import WeatherConnector
from transitpulse.utils.logger import get_logger

logger = get_logger(__name__)


def compute_file_sha256(filepath: Path) -> str:
    """Computes SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


class IngestionPipeline:
    """Orchestrates multi-modal data ingestion and provenance tracking."""

    def __init__(self):
        self.transit_conn = TransitConnector()
        self.weather_conn = WeatherConnector()
        self.event_conn = EventConnector()
        self.simulator = TransitDelaySimulator()
        self.manifest_file = settings.DATA_RAW / "manifest.json"

    def run(self, start_date: str = "2024-05-01", end_date: str = "2024-05-21") -> dict[str, Any]:
        """Executes full multi-modal ingestion pipeline."""
        logger.info("Executing TransitPulse-AI multi-modal ingestion pipeline...")
        run_timestamp = datetime.now().isoformat()

        # 1. Ingest BMTC Topology
        df_routes, df_stops = self.transit_conn.fetch_and_save_topology()

        # 2. Ingest Weather
        df_weather = self.weather_conn.fetch_historical_weather(
            start_date=start_date, end_date=end_date
        )

        # 3. Ingest Events
        df_events = self.event_conn.fetch_and_save_events()

        # 4. Run Calibrated Delay Simulator
        df_sim = self.simulator.simulate_dataset(
            df_routes=df_routes,
            df_stops=df_stops,
            df_weather=df_weather,
            df_events=df_events,
            start_date=start_date,
            end_date=end_date,
        )

        # 5. Build Provenance Manifest
        manifest = {
            "ingestion_timestamp": run_timestamp,
            "bengaluru_centroid": [settings.CENTROID_LAT, settings.CENTROID_LON],
            "timezone": settings.TIMEZONE,
            "artifacts": {
                "bmtc_routes": {
                    "path": str(self.transit_conn.routes_file),
                    "records": len(df_routes),
                    "mode": "observed",
                    "sha256": compute_file_sha256(self.transit_conn.routes_file),
                },
                "bmtc_stops": {
                    "path": str(self.transit_conn.stops_file),
                    "records": len(df_stops),
                    "mode": "observed",
                    "sha256": compute_file_sha256(self.transit_conn.stops_file),
                },
                "weather": {
                    "path": str(self.weather_conn.weather_file),
                    "records": len(df_weather),
                    "mode": df_weather["dataset_mode"].iloc[0],
                    "sha256": compute_file_sha256(self.weather_conn.weather_file),
                },
                "events": {
                    "path": str(self.event_conn.events_file),
                    "records": len(df_events),
                    "mode": "observed",
                    "sha256": compute_file_sha256(self.event_conn.events_file),
                },
                "simulated_trips": {
                    "path": str(self.simulator.output_file),
                    "records": len(df_sim),
                    "mode": "simulated",
                    "sha256": compute_file_sha256(self.simulator.output_file),
                },
            },
        }

        with open(self.manifest_file, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        logger.info("Ingestion complete. Provenance manifest written to %s.", self.manifest_file)
        return manifest
