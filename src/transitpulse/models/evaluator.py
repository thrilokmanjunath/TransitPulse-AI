"""Model Evaluation and Scientific Metrics Module."""

import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    median_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    root_mean_squared_error,
)


class ModelEvaluator:
    """Evaluates continuous delay regression and multi-class classification."""

    @staticmethod
    def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
        """Calculates standard continuous regression error metrics."""
        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(root_mean_squared_error(y_true, y_pred))
        r2 = float(r2_score(y_true, y_pred))
        medae = float(median_absolute_error(y_true, y_pred))

        return {
            "mae": round(mae, 3),
            "rmse": round(rmse, 3),
            "r2": round(r2, 4),
            "medae": round(medae, 3),
        }

    @staticmethod
    def evaluate_classification(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
        """Calculates multi-class delay categorization metrics."""
        acc = float(accuracy_score(y_true, y_pred))
        macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        precision = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
        recall = float(recall_score(y_true, y_pred, average="macro", zero_division=0))

        # Severe delay recall (Class "Severe")
        severe_mask = y_true == "Severe"
        if np.sum(severe_mask) > 0:
            severe_recall = float(np.sum((y_pred == "Severe") & severe_mask) / np.sum(severe_mask))
        else:
            severe_recall = 0.0

        return {
            "accuracy": round(acc, 4),
            "macro_f1": round(macro_f1, 4),
            "macro_precision": round(precision, 4),
            "macro_recall": round(recall, 4),
            "severe_recall": round(severe_recall, 4),
        }

    @staticmethod
    def benchmark_latency(
        model_predict_fn, sample_X: pd.DataFrame, n_iterations: int = 500
    ) -> dict[str, float]:
        """Measures single-instance inference latency (ms)."""
        latencies_ms = []
        # Warmup
        for _ in range(10):
            _ = model_predict_fn(sample_X.iloc[[0]])

        # Timed benchmark
        row = sample_X.iloc[[0]]
        for _ in range(n_iterations):
            t0 = time.perf_counter()
            _ = model_predict_fn(row)
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)

        return {
            "latency_p50_ms": round(float(np.percentile(latencies_ms, 50)), 2),
            "latency_p95_ms": round(float(np.percentile(latencies_ms, 95)), 2),
            "latency_p99_ms": round(float(np.percentile(latencies_ms, 99)), 2),
        }

    @staticmethod
    def slice_performance(df_eval: pd.DataFrame) -> dict[str, Any]:
        """Computes stratified MAE across peak/off-peak, weather, and route corridors."""
        slices = {}

        # 1. Peak vs Off-Peak
        if "is_peak_hour" in df_eval.columns:
            peak_mae = float(
                mean_absolute_error(
                    df_eval[df_eval["is_peak_hour"] == 1]["y_true"],
                    df_eval[df_eval["is_peak_hour"] == 1]["y_pred"],
                )
            )
            offpeak_mae = float(
                mean_absolute_error(
                    df_eval[df_eval["is_peak_hour"] == 0]["y_true"],
                    df_eval[df_eval["is_peak_hour"] == 0]["y_pred"],
                )
            )
            slices["peak_hour_mae"] = round(peak_mae, 3)
            slices["offpeak_hour_mae"] = round(offpeak_mae, 3)

        # 2. Raining vs Clear
        if "is_raining" in df_eval.columns:
            rain_mask = df_eval["is_raining"] == 1
            if rain_mask.sum() > 0:
                rain_mae = float(
                    mean_absolute_error(
                        df_eval[rain_mask]["y_true"],
                        df_eval[rain_mask]["y_pred"],
                    )
                )
            else:
                rain_mae = 0.0
            clear_mae = float(
                mean_absolute_error(
                    df_eval[~rain_mask]["y_true"],
                    df_eval[~rain_mask]["y_pred"],
                )
            )
            slices["rain_condition_mae"] = round(rain_mae, 3)
            slices["clear_condition_mae"] = round(clear_mae, 3)

        # 3. Route specific MAE
        route_maes = {}
        if "route_id" in df_eval.columns:
            for rid, grp in df_eval.groupby("route_id"):
                route_maes[str(rid)] = round(
                    float(mean_absolute_error(grp["y_true"], grp["y_pred"])), 3
                )
            slices["route_level_mae"] = route_maes

        return slices
