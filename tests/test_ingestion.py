"""Tests for Data Ingestion Connectors."""

import pandas as pd

from transitpulse.config import settings
from transitpulse.ingestion.event_connector import EventConnector
from transitpulse.ingestion.transit_connector import TransitConnector


def test_transit_connector_routes():
    conn = TransitConnector()
    df_routes = conn.get_routes()
    assert isinstance(df_routes, pd.DataFrame)
    assert len(df_routes) >= 5
    assert "route_id" in df_routes.columns
    assert "335-E" in df_routes["route_id"].values
    assert "500-D" in df_routes["route_id"].values


def test_transit_connector_stops():
    conn = TransitConnector()
    df_stops = conn.get_stops()
    assert isinstance(df_stops, pd.DataFrame)
    assert len(df_stops) >= 50
    assert "stop_lat" in df_stops.columns
    assert "stop_lon" in df_stops.columns

    # Verify all stops are within Bengaluru boundary
    assert df_stops["stop_lat"].between(settings.LAT_MIN, settings.LAT_MAX).all()
    assert df_stops["stop_lon"].between(settings.LON_MIN, settings.LON_MAX).all()


def test_event_connector():
    conn = EventConnector()
    df_events = conn.get_events()
    assert isinstance(df_events, pd.DataFrame)
    assert len(df_events) >= 5
    assert "expected_attendance" in df_events.columns
    assert (df_events["expected_attendance"] > 0).all()
