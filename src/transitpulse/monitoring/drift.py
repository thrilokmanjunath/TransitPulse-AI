"""Data and Prediction Drift Monitoring Engine."""

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def calculate_psi(expected: np.ndarray, actual: np.ndarray, num_buckets: int = 10) -> float:
    """Calculates Population Stability Index (PSI) between reference and current cohort."""
    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    # Determine quantile bins based on reference
    quantiles = np.linspace(0, 100, num_buckets + 1)
    bins = np.percentile(expected, quantiles)
    bins[0] = -np.inf
    bins[-1] = np.inf

    exp_counts, _ = np.histogram(expected, bins=bins)
    act_counts, _ = np.histogram(actual, bins=bins)

    exp_pct = np.maximum(exp_counts / len(expected), 1e-4)
    act_pct = np.maximum(act_counts / len(actual), 1e-4)

    psi_value = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))
    return float(round(psi_value, 4))


class DriftMonitor:
    """Monitors distribution drift between training reference and inference data."""

    def __init__(self, reference_df: pd.DataFrame):
        self.reference_df = reference_df

    def detect_drift(self, current_df: pd.DataFrame, features: list[str] = None) -> dict[str, Any]:
        """Runs KS-tests and PSI across feature distributions."""
        cols = features or [
            "prev_stop_delay_min", "precip_rate_mm", "temperature_c",
            "hour_of_day", "dist_nearest_event_km"
        ]
        drift_results = {}
        drift_detected_count = 0

        for col in cols:
            if col in self.reference_df.columns and col in current_df.columns:
                ref_vals = self.reference_df[col].dropna().values
                cur_vals = current_df[col].dropna().values

                if len(cur_vals) < 5 or len(ref_vals) < 5:
                    continue

                ks_stat, p_value = ks_2samp(ref_vals, cur_vals)
                psi_val = calculate_psi(ref_vals, cur_vals)

                is_drift = bool(p_value < 0.05 or psi_val > 0.25)
                if is_drift:
                    drift_detected_count += 1

                drift_results[col] = {
                    "ks_statistic": round(float(ks_stat), 4),
                    "p_value": round(float(p_value), 4),
                    "psi": psi_val,
                    "drift_status": "Drift Detected" if is_drift else "Stable",
                }

        overall_status = "Action Required" if drift_detected_count >= 2 else ("Minor Drift" if drift_detected_count == 1 else "Healthy")

        return {
            "overall_status": overall_status,
            "drifted_features_count": drift_detected_count,
            "total_features_monitored": len(drift_results),
            "feature_diagnostics": drift_results,
        }
