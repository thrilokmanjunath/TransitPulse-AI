"""Explainable AI Module Using TreeSHAP for Delay Attribution."""

from typing import Any

import numpy as np
import pandas as pd
import shap

from transitpulse.utils.logger import get_logger

logger = get_logger(__name__)


class TransitSHAPExplainer:
    """Computes exact Shapley values using TreeSHAP for tree ensembles."""

    def __init__(self, model, feature_names: list[str]):
        self.model = model.regressor if hasattr(model, "regressor") else model
        self.feature_names = feature_names
        # Create TreeExplainer on underlying Booster
        self.explainer = shap.TreeExplainer(self.model)

    def explain_instance(self, row: pd.DataFrame) -> dict[str, Any]:
        """Calculates Shapley additive attribution breakdown for a single prediction."""
        X_eval = row[self.feature_names].fillna(0.0)
        shap_values = self.explainer.shap_values(X_eval)

        # Handle multiple outputs if any
        if isinstance(shap_values, list):
            sv = shap_values[0]
        else:
            sv = shap_values

        base_val = float(self.explainer.expected_value) if not isinstance(self.explainer.expected_value, np.ndarray) else float(self.explainer.expected_value[0])
        val_array = sv[0] if len(sv.shape) > 1 else sv

        attributions = []
        for feat, val, feat_val in zip(self.feature_names, val_array, X_eval.iloc[0].values):
            attributions.append({
                "feature": feat,
                "value": float(feat_val),
                "shap_attribution_min": round(float(val), 3),
            })

        # Sort by absolute impact
        attributions = sorted(attributions, key=lambda x: abs(x["shap_attribution_min"]), reverse=True)

        # High-level category aggregations
        weather_features = [
            "precip_rate_mm", "is_raining", "is_heavy_rain", "temperature_c",
            "relative_humidity", "wind_gusts_kmh", "weather_severity_idx",
            "precip_rolling_3h_mm", "rain_x_peak_hour", "rain_x_road_hierarchy"
        ]
        event_features = [
            "event_count_500m", "event_count_1km", "event_count_3km",
            "dist_nearest_event_km", "min_to_nearest_event", "event_capacity_shock",
            "has_nearby_event", "event_x_peak_hour"
        ]
        peak_features = [
            "is_morning_peak", "is_evening_peak", "is_peak_hour", "hour_of_day",
            "hour_sin", "hour_cos"
        ]

        weather_contrib = sum(a["shap_attribution_min"] for a in attributions if a["feature"] in weather_features)
        event_contrib = sum(a["shap_attribution_min"] for a in attributions if a["feature"] in event_features)
        peak_contrib = sum(a["shap_attribution_min"] for a in attributions if a["feature"] in peak_features)

        predicted_delay = base_val + float(np.sum(val_array))

        return {
            "base_expected_delay_min": round(base_val, 2),
            "predicted_delay_min": round(predicted_delay, 2),
            "top_attributions": attributions[:8],
            "category_contributions": {
                "weather_impact_min": round(weather_contrib, 2),
                "event_impact_min": round(event_contrib, 2),
                "commuter_peak_impact_min": round(peak_contrib, 2),
                "route_baseline_impact_min": round(predicted_delay - (base_val + weather_contrib + event_contrib + peak_contrib), 2),
            },
        }

    def compute_global_importance(self, X_sample: pd.DataFrame) -> pd.DataFrame:
        """Calculates global mean absolute SHAP values across a sample cohort."""
        X_eval = X_sample[self.feature_names].fillna(0.0)
        shap_values = self.explainer.shap_values(X_eval)
        if isinstance(shap_values, list):
            sv = shap_values[0]
        else:
            sv = shap_values

        mean_abs_shap = np.mean(np.abs(sv), axis=0)
        df_imp = pd.DataFrame({
            "feature": self.feature_names,
            "mean_abs_shap": mean_abs_shap,
        }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

        return df_imp
