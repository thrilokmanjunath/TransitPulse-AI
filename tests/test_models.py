"""Tests for Model Loading, Baselines, and Inference."""

import joblib
import numpy as np
import pandas as pd
import pytest

from transitpulse.config import settings
from transitpulse.explainability.shap_explainer import TransitSHAPExplainer
from transitpulse.models.baselines import HistoricalMeanBaseline, PersistenceBaseline


def test_historical_mean_baseline():
    X = pd.DataFrame({"route_code": [1, 1, 2, 2]})
    y = np.array([4.0, 6.0, 10.0, 12.0])
    base = HistoricalMeanBaseline().fit(X, y)
    preds = base.predict(pd.DataFrame({"route_code": [1, 2, 3]}))
    assert preds[0] == 5.0
    assert preds[1] == 11.0
    assert preds[2] == 8.0  # fallback global mean


def test_persistence_baseline():
    X = pd.DataFrame({"prev_stop_delay_min": [3.0, 7.5, np.nan]})
    y = np.array([3.5, 7.0, 2.0])
    base = PersistenceBaseline().fit(X, y)
    preds = base.predict(X)
    assert preds[0] == 3.0
    assert preds[1] == 7.5
    assert preds[2] == base.default_delay


def test_production_model_loading_and_shap():
    model_path = settings.ARTIFACTS_DIR / "best_model.joblib"
    if not model_path.exists():
        pytest.skip("Model artifact not trained yet.")

    model = joblib.load(model_path)
    assert hasattr(model, "predict_reg")
    assert hasattr(model, "predict_cls")

    sample_path = settings.ARTIFACTS_DIR / "sample_test_records.parquet"
    if not sample_path.exists():
        pytest.skip("Sample test records not found.")

    with open(settings.DATA_PROCESSED / "feature_metadata.json", "r") as f:
        meta = pd.read_json(settings.DATA_PROCESSED / "feature_metadata.json", typ="series").to_dict()
    features = meta["experiment_feature_sets"]["E4"]

    df_sample = pd.read_parquet(sample_path).head(5)
    preds = model.predict_reg(df_sample[features])
    classes = model.predict_cls(df_sample[features])

    assert len(preds) == 5
    assert len(classes) == 5
    assert all(c in ["On-time", "Minor", "Moderate", "Severe"] for c in classes)

    # Test SHAP explainer
    explainer = TransitSHAPExplainer(model, features)
    exp_res = explainer.explain_instance(df_sample.iloc[[0]])
    assert "predicted_delay_min" in exp_res
    assert "category_contributions" in exp_res
    assert "weather_impact_min" in exp_res["category_contributions"]
