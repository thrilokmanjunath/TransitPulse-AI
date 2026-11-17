"""Tests for Schema Validation and Quality Checks."""

import pandas as pd

from transitpulse.validation.schema_validator import SchemaValidator


def test_schema_validator_bmtc_stops():
    validator = SchemaValidator()
    df_valid = pd.DataFrame(
        [
            {
                "route_id": "335-E",
                "stop_sequence": 1,
                "stop_id": "ST-01",
                "stop_lat": 12.9716,
                "stop_lon": 77.5946,
            },
            {
                "route_id": "335-E",
                "stop_sequence": 2,
                "stop_id": "ST-02",
                "stop_lat": 12.9800,
                "stop_lon": 77.6000,
            },
        ]
    )
    report = validator.validate_bmtc_stops(df_valid)
    assert report.passed_validation is True
    assert report.geo_bounds_violations == 0
    assert report.duplicate_rows == 0


def test_schema_validator_out_of_bounds_detection():
    validator = SchemaValidator()
    df_invalid = pd.DataFrame(
        [
            {
                "route_id": "335-E",
                "stop_sequence": 1,
                "stop_id": "ST-BAD",
                "stop_lat": 28.6139,  # New Delhi lat (out of Bengaluru bounds)
                "stop_lon": 77.2090,
            }
        ]
    )
    report = validator.validate_bmtc_stops(df_invalid)
    assert report.passed_validation is False
    assert report.geo_bounds_violations == 1
