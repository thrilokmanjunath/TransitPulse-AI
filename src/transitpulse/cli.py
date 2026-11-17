"""TransitPulse-AI Command Line Interface (CLI)."""

import argparse
import subprocess  # nosec B404
import sys

import pandas as pd
import uvicorn

from transitpulse.config import settings
from transitpulse.features.builder import FeatureBuilder
from transitpulse.ingestion.pipeline import IngestionPipeline
from transitpulse.models.trainer import ExperimentTrainer
from transitpulse.utils.logger import get_logger
from transitpulse.validation.quality_reporter import QualityReporter

logger = get_logger("transitpulse.cli")


def cmd_ingest(args):
    """Executes data ingestion pipeline."""
    pipe = IngestionPipeline()
    manifest = pipe.run(start_date=args.start_date, end_date=args.end_date)
    print("Ingestion complete. Manifest:", manifest)


def cmd_validate(args):
    """Executes data validation and quality reporting."""
    df_stops = pd.read_parquet(settings.DATA_RAW / "bmtc_stops.parquet")
    df_weather = pd.read_parquet(settings.DATA_RAW / "bengaluru_weather.parquet")
    df_trips = pd.read_parquet(settings.DATA_SIMULATED / "bmtc_simulated_trips.parquet")

    reporter = QualityReporter()
    report_path = reporter.audit_and_generate(df_stops, df_weather, df_trips)
    print(f"Validation report generated at: {report_path}")


def cmd_features(args):
    """Executes feature engineering and chronological dataset splitting."""
    df_trips = pd.read_parquet(settings.DATA_SIMULATED / "bmtc_simulated_trips.parquet")
    df_stops = pd.read_parquet(settings.DATA_RAW / "bmtc_stops.parquet")
    from transitpulse.ingestion.event_connector import EventConnector

    df_events = EventConnector().get_events()

    fb = FeatureBuilder()
    df_full = fb.build_features(df_trips, df_stops, df_events)
    train, val, test = fb.split_chronological(df_full)
    print(f"Features built: Train={len(train)}, Val={len(val)}, Test={len(test)}")


def cmd_train(args):
    """Runs experiment ladder E0 through E4 and saves production model bundle."""
    trainer = ExperimentTrainer()
    results = trainer.run_all_experiments()
    print("Training finished! Results summary saved.")


def cmd_api(args):
    """Launches FastAPI REST server."""
    print(f"Starting TransitPulse-AI API on {args.host}:{args.port}...")
    uvicorn.run("api.main:app", host=args.host, port=args.port, reload=args.reload)


def cmd_dashboard(args):
    """Launches Streamlit visual analytics dashboard."""
    dashboard_path = settings.PROJECT_ROOT / "dashboard" / "app.py"
    print(f"Launching Streamlit Dashboard on port {args.port}...")
    subprocess.run(
        [  # nosec B603
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(dashboard_path),
            "--server.port",
            str(args.port),
            "--server.address",
            args.host,
        ]
    )


def main():
    parser = argparse.ArgumentParser(
        prog="transitpulse",
        description="TransitPulse-AI: Spatiotemporal Public Transit Delay Prediction Platform",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Ingest
    p_ingest = subparsers.add_parser("ingest", help="Run multi-modal data ingestion")
    p_ingest.add_argument("--start-date", default="2024-05-01")
    p_ingest.add_argument("--end-date", default="2024-05-14")
    p_ingest.set_defaults(func=cmd_ingest)

    # Validate
    p_val = subparsers.add_parser("validate", help="Validate raw and simulated datasets")
    p_val.set_defaults(func=cmd_validate)

    # Features
    p_feat = subparsers.add_parser(
        "features", help="Extract spatiotemporal features and split data"
    )
    p_feat.set_defaults(func=cmd_features)

    # Train
    p_train = subparsers.add_parser("train", help="Train models across experiment ladder E0-E4")
    p_train.set_defaults(func=cmd_train)

    # API
    p_api = subparsers.add_parser("api", help="Launch FastAPI server")
    p_api.add_argument("--host", default=settings.API_HOST)
    p_api.add_argument("--port", type=int, default=settings.API_PORT)
    p_api.add_argument("--reload", action="store_true")
    p_api.set_defaults(func=cmd_api)

    # Dashboard
    p_dash = subparsers.add_parser("dashboard", help="Launch Streamlit dashboard")
    p_dash.add_argument("--host", default=settings.API_HOST)
    p_dash.add_argument("--port", type=int, default=settings.DASHBOARD_PORT)
    p_dash.set_defaults(func=cmd_dashboard)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
