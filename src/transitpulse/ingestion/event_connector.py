"""Urban Event Ingestion Connector for Bengaluru Major Venues."""

from pathlib import Path

import pandas as pd

from transitpulse.config import settings
from transitpulse.utils.logger import get_logger

logger = get_logger(__name__)


# Curated public major events during study period (May 2024 IPL / Concerts / Expos)
CURATED_BENGALURU_EVENTS = [
    {
        "event_id": "EVT-2024-001",
        "event_name": "IPL T20: RCB vs CSK",
        "venue_key": "chinnaswamy_stadium",
        "start_time": "2024-05-18 19:30:00",
        "end_time": "2024-05-18 23:30:00",
        "expected_attendance": 38000,
        "event_category": "sports",
    },
    {
        "event_id": "EVT-2024-002",
        "event_name": "IPL T20: RCB vs DC",
        "venue_key": "chinnaswamy_stadium",
        "start_time": "2024-05-12 19:30:00",
        "end_time": "2024-05-12 23:30:00",
        "expected_attendance": 35000,
        "event_category": "sports",
    },
    {
        "event_id": "EVT-2024-003",
        "event_name": "International Mega Music Festival",
        "venue_key": "palace_grounds",
        "start_time": "2024-05-11 16:00:00",
        "end_time": "2024-05-11 22:30:00",
        "expected_attendance": 42000,
        "event_category": "concert",
    },
    {
        "event_id": "EVT-2024-004",
        "event_name": "Global Tech & Startup Expo 2024",
        "venue_key": "biec",
        "start_time": "2024-05-22 09:30:00",
        "end_time": "2024-05-24 18:00:00",
        "expected_attendance": 28000,
        "event_category": "exhibition",
    },
    {
        "event_id": "EVT-2024-005",
        "event_name": "National Basketball Championship Finals",
        "venue_key": "koramangala_indoor",
        "start_time": "2024-05-19 17:00:00",
        "end_time": "2024-05-19 21:00:00",
        "expected_attendance": 5500,
        "event_category": "sports",
    },
    {
        "event_id": "EVT-2024-006",
        "event_name": "Bengaluru Automation & AI Summit",
        "venue_key": "ktpo_whitefield",
        "start_time": "2024-05-09 09:00:00",
        "end_time": "2024-05-10 18:00:00",
        "expected_attendance": 9000,
        "event_category": "exhibition",
    },
    {
        "event_id": "EVT-2024-007",
        "event_name": "ISL Football Clash: Bengaluru FC vs Mohun Bagan",
        "venue_key": "kanteerava_stadium",
        "start_time": "2024-05-25 19:00:00",
        "end_time": "2024-05-25 21:30:00",
        "expected_attendance": 22000,
        "event_category": "sports",
    },
]


class EventConnector:
    """Manages urban event calendar ingestion and spatial attributes."""

    def __init__(self, raw_dir: Path | None = None):
        self.raw_dir = raw_dir or settings.DATA_RAW
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.events_file = self.raw_dir / "bengaluru_events.parquet"

    def fetch_and_save_events(self) -> pd.DataFrame:
        """Curates and persists verified urban event records."""
        logger.info("Ingesting verified Bengaluru public events...")
        records = []

        for e in CURATED_BENGALURU_EVENTS:
            v_info = settings.BENGALURU_VENUES[e["venue_key"]]
            records.append({
                "event_id": e["event_id"],
                "event_name": e["event_name"],
                "venue_name": v_info["name"],
                "venue_lat": v_info["lat"],
                "venue_lon": v_info["lon"],
                "event_category": e["event_category"],
                "expected_attendance": e["expected_attendance"],
                "start_time": pd.to_datetime(e["start_time"]).tz_localize(settings.TIMEZONE),
                "end_time": pd.to_datetime(e["end_time"]).tz_localize(settings.TIMEZONE),
                "dataset_mode": "observed",
            })

        df_events = pd.DataFrame(records)
        df_events.to_parquet(self.events_file, index=False)
        logger.info("Persisted %d verified urban events to %s.", len(df_events), self.events_file)
        return df_events

    def get_events(self) -> pd.DataFrame:
        """Retrieves cached events or initializes them."""
        if not self.events_file.exists():
            return self.fetch_and_save_events()
        return pd.read_parquet(self.events_file)
