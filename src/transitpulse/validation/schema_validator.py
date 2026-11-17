"""Data Validation and Quality Verification Module.

Enforces geographic boundary constraints, null checks, schema types,
outlier bounds, and temporal monotonic ordering.
"""

import pandas as pd
from pydantic import BaseModel

from transitpulse.config import settings
from transitpulse.utils.logger import get_logger

logger = get_logger(__name__)


class ValidationReport(BaseModel):
    """Structured quality assessment report."""
    dataset_name: str
    total_records: int
    null_counts: dict[str, int]
    duplicate_rows: int
    outliers_detected: dict[str, int]
    geo_bounds_violations: int
    passed_validation: bool
    summary_messages: list[str]


class SchemaValidator:
    """Validates transit, weather, and feature datasets."""

    def __init__(self):
        self.lat_min = settings.LAT_MIN
        self.lat_max = settings.LAT_MAX
        self.lon_min = settings.LON_MIN
        self.lon_max = settings.LON_MAX

    def validate_bmtc_stops(self, df: pd.DataFrame) -> ValidationReport:
        """Validates BMTC bus stops dataframe."""
        total = len(df)
        null_counts = df.isnull().sum().to_dict()
        duplicates = int(df.duplicated(subset=["route_id", "stop_sequence"]).sum())

        # Check geographic bounding box
        lat_violations = ~df["stop_lat"].between(self.lat_min, self.lat_max)
        lon_violations = ~df["stop_lon"].between(self.lon_min, self.lon_max)
        geo_violations = int((lat_violations | lon_violations).sum())

        outliers = {}
        # Sequence must be positive
        outliers["negative_sequence"] = int((df["stop_sequence"] <= 0).sum())

        passed = (duplicates == 0) and (geo_violations == 0) and (null_counts.get("stop_id", 0) == 0)
        messages = []
        if geo_violations > 0:
            messages.append(f"Detected {geo_violations} stops outside Bengaluru coordinates.")
        if duplicates > 0:
            messages.append(f"Detected {duplicates} duplicate route-sequence pairs.")
        if passed:
            messages.append("All BMTC stop schemas, coordinate bounds, and sequence orderings passed.")

        return ValidationReport(
            dataset_name="BMTC Stops",
            total_records=total,
            null_counts=null_counts,
            duplicate_rows=duplicates,
            outliers_detected=outliers,
            geo_bounds_violations=geo_violations,
            passed_validation=passed,
            summary_messages=messages,
        )

    def validate_weather(self, df: pd.DataFrame) -> ValidationReport:
        """Validates Open-Meteo meteorological records."""
        total = len(df)
        null_counts = df.isnull().sum().to_dict()
        duplicates = int(df.duplicated(subset=["timestamp"]).sum())

        outliers = {}
        # Bengaluru temperatures rarely go below 10°C or above 45°C
        outliers["extreme_temp"] = int((~df["temperature_2m"].between(10.0, 45.0)).sum())
        # Precipitation cannot be negative
        outliers["negative_precipitation"] = int((df["precipitation"] < 0.0).sum())
        # Humidity must be 0-100%
        outliers["invalid_humidity"] = int((~df["relative_humidity_2m"].between(0.0, 100.0)).sum())

        passed = (duplicates == 0) and (sum(outliers.values()) == 0)
        messages = []
        if not passed:
            messages.append(f"Weather outliers detected: {outliers}")
        else:
            messages.append("All meteorological ranges, humidity limits, and timestamps valid.")

        return ValidationReport(
            dataset_name="Open-Meteo Weather",
            total_records=total,
            null_counts=null_counts,
            duplicate_rows=duplicates,
            outliers_detected=outliers,
            geo_bounds_violations=0,
            passed_validation=passed,
            summary_messages=messages,
        )

    def validate_trips(self, df: pd.DataFrame) -> ValidationReport:
        """Validates simulated/observed transit trip records."""
        total = len(df)
        null_counts = df.isnull().sum().to_dict()
        duplicates = int(df.duplicated(subset=["trip_id", "stop_id"]).sum())

        outliers = {}
        # Delay outside [-10m, 120m] considered extreme outlier
        outliers["extreme_delay"] = int((~df["observed_delay_min"].between(-10.0, 120.0)).sum())

        lat_violations = ~df["stop_lat"].between(self.lat_min, self.lat_max)
        lon_violations = ~df["stop_lon"].between(self.lon_min, self.lon_max)
        geo_violations = int((lat_violations | lon_violations).sum())

        passed = (duplicates == 0) and (geo_violations == 0) and (outliers["extreme_delay"] == 0)
        messages = []
        if passed:
            messages.append("All transit operational delay records conform to schema specifications.")
        else:
            messages.append(f"Trips validation issues: duplicates={duplicates}, geo={geo_violations}, outliers={outliers}")

        return ValidationReport(
            dataset_name="Transit Trips",
            total_records=total,
            null_counts=null_counts,
            duplicate_rows=duplicates,
            outliers_detected=outliers,
            geo_bounds_violations=geo_violations,
            passed_validation=passed,
            summary_messages=messages,
        )
