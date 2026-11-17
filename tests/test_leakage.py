"""Strict Temporal Leakage & Cross-Contamination Test Suite."""

import pandas as pd
import pytest

from transitpulse.config import settings


def test_chronological_split_strict_monotonicity():
    """Verifies Train, Validation, and Test datasets do not overlap in time."""
    train_path = settings.DATA_PROCESSED / "train.parquet"
    val_path = settings.DATA_PROCESSED / "val.parquet"
    test_path = settings.DATA_PROCESSED / "test.parquet"

    if not (train_path.exists() and val_path.exists() and test_path.exists()):
        pytest.skip("Processed datasets not generated yet.")

    df_train = pd.read_parquet(train_path)
    df_val = pd.read_parquet(val_path)
    df_test = pd.read_parquet(test_path)

    t_train_max = pd.to_datetime(df_train["scheduled_time"]).max()
    t_val_min = pd.to_datetime(df_val["scheduled_time"]).min()
    t_val_max = pd.to_datetime(df_val["scheduled_time"]).max()
    t_test_min = pd.to_datetime(df_test["scheduled_time"]).min()

    # Monotonic boundary verification
    assert t_train_max <= t_val_min, (
        f"Temporal leakage detected: Train max ({t_train_max}) > Val min ({t_val_min})"
    )
    assert t_val_max <= t_test_min, (
        f"Temporal leakage detected: Val max ({t_val_max}) > Test min ({t_test_min})"
    )


def test_no_future_target_in_training_features():
    """Verifies that no future labels exist in the model's feature column set."""
    train_path = settings.DATA_PROCESSED / "train.parquet"
    meta_path = settings.DATA_PROCESSED / "feature_metadata.json"

    if not (train_path.exists() and meta_path.exists()):
        pytest.skip("Data not ready.")

    import json

    with open(meta_path, "r") as f:
        meta = json.load(f)

    for exp_id, features in meta["experiment_feature_sets"].items():
        assert "observed_delay_min" not in features, f"Target leaked into {exp_id} feature set!"
        assert "delay_category" not in features, (
            f"Target category leaked into {exp_id} feature set!"
        )
        assert "delay_horizon_15m" not in features, (
            f"Future horizon leaked into {exp_id} feature set!"
        )
        assert "delay_horizon_30m" not in features, (
            f"Future horizon leaked into {exp_id} feature set!"
        )
        assert "delay_horizon_60m" not in features, (
            f"Future horizon leaked into {exp_id} feature set!"
        )


def test_trip_boundary_independence():
    """Verifies trip IDs do not leak across the chronological train/test boundary."""
    train_path = settings.DATA_PROCESSED / "train.parquet"
    test_path = settings.DATA_PROCESSED / "test.parquet"

    if not (train_path.exists() and test_path.exists()):
        pytest.skip("Data not ready.")

    df_train = pd.read_parquet(train_path)
    df_test = pd.read_parquet(test_path)

    train_trips = set(df_train["trip_id"].unique())
    test_trips = set(df_test["trip_id"].unique())

    overlap = train_trips.intersection(test_trips)
    assert len(overlap) == 0, f"Trip ID contamination detected across splits: {overlap}"
