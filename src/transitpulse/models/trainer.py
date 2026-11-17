"""Training Pipeline Orchestrator for Experiments E0 to E4."""

import json
from pathlib import Path
from typing import Any

import joblib
import mlflow
import pandas as pd

from transitpulse.config import settings
from transitpulse.models.baselines import (
    HistoricalMeanBaseline,
)
from transitpulse.models.evaluator import ModelEvaluator
from transitpulse.models.multi_horizon import MultiHorizonForecaster
from transitpulse.models.tree_models import CatBoostModel, LightGBMModel, XGBoostModel
from transitpulse.models.tuner import OptunaTuner
from transitpulse.utils.logger import get_logger

logger = get_logger(__name__)


class ExperimentTrainer:
    """Manages the full experimental ladder (E0-E4), MLflow tracking, and model serialization."""

    def __init__(self, artifacts_dir: Path = None):
        self.artifacts_dir = artifacts_dir or settings.ARTIFACTS_DIR
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.artifacts_dir / "model_metadata.json"
        self.results_file = self.artifacts_dir / "experiment_results.json"
        
        # Setup SQLite database MLflow tracking
        db_path = settings.PROJECT_ROOT / "mlflow.db"
        mlflow_uri = f"sqlite:///{db_path}"
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment("transitpulse_delay_prediction")

    def run_all_experiments(self) -> dict[str, Any]:
        """Executes E0 through E4 experiments and saves evaluation artifacts."""
        logger.info("Loading processed datasets for model training...")
        df_train = pd.read_parquet(settings.DATA_PROCESSED / "train.parquet")
        df_val = pd.read_parquet(settings.DATA_PROCESSED / "val.parquet")
        df_test = pd.read_parquet(settings.DATA_PROCESSED / "test.parquet")

        with open(settings.DATA_PROCESSED / "feature_metadata.json", "r") as f:
            feat_meta = json.load(f)

        exp_features = feat_meta["experiment_feature_sets"]
        y_train_reg = df_train["observed_delay_min"].values
        y_val_reg = df_val["observed_delay_min"].values
        y_test_reg = df_test["observed_delay_min"].values

        y_train_cls = df_train["delay_category"].values
        y_val_cls = df_val["delay_category"].values
        y_test_cls = df_test["delay_category"].values

        evaluator = ModelEvaluator()
        results: dict[str, Any] = {}

        # ==========================================
        # E0: Historical Baseline
        # ==========================================
        logger.info("Executing Experiment E0: Historical Baseline...")
        with mlflow.start_run(run_name="E0_Historical_Baseline"):
            e0_cols = exp_features["E0"]
            base_model = HistoricalMeanBaseline().fit(df_train[e0_cols], y_train_reg)
            e0_preds = base_model.predict(df_test[e0_cols])
            e0_metrics = evaluator.evaluate_regression(y_test_reg, e0_preds)
            mlflow.log_params({"experiment": "E0", "features": len(e0_cols)})
            mlflow.log_metrics(e0_metrics)
            results["E0"] = {
                "name": "E0 — Historical Baseline",
                "features_count": len(e0_cols),
                "features": e0_cols,
                "test_metrics": e0_metrics,
            }

        # ==========================================
        # E1: Transit + Temporal Features
        # ==========================================
        logger.info("Executing Experiment E1: Transit + Temporal...")
        with mlflow.start_run(run_name="E1_Transit_Temporal"):
            e1_cols = exp_features["E1"]
            e1_model = LightGBMModel()
            e1_model.fit(df_train[e1_cols], y_train_reg, y_train_cls, eval_set_reg=(df_val[e1_cols], y_val_reg))
            e1_preds = e1_model.predict_reg(df_test[e1_cols])
            e1_preds_cls = e1_model.predict_cls(df_test[e1_cols])
            e1_metrics = evaluator.evaluate_regression(y_test_reg, e1_preds)
            e1_cls_metrics = evaluator.evaluate_classification(y_test_cls, e1_preds_cls)
            mlflow.log_params({"experiment": "E1", "features": len(e1_cols)})
            mlflow.log_metrics({**e1_metrics, **e1_cls_metrics})
            results["E1"] = {
                "name": "E1 — Transit + Temporal",
                "features_count": len(e1_cols),
                "features": e1_cols,
                "test_metrics": e1_metrics,
                "classification_metrics": e1_cls_metrics,
            }

        # ==========================================
        # E2: Transit + Temporal + Weather Features
        # ==========================================
        logger.info("Executing Experiment E2: Transit + Temporal + Weather...")
        with mlflow.start_run(run_name="E2_Transit_Weather"):
            e2_cols = exp_features["E2"]
            e2_model = LightGBMModel()
            e2_model.fit(df_train[e2_cols], y_train_reg, y_train_cls, eval_set_reg=(df_val[e2_cols], y_val_reg))
            e2_preds = e2_model.predict_reg(df_test[e2_cols])
            e2_preds_cls = e2_model.predict_cls(df_test[e2_cols])
            e2_metrics = evaluator.evaluate_regression(y_test_reg, e2_preds)
            e2_cls_metrics = evaluator.evaluate_classification(y_test_cls, e2_preds_cls)
            mlflow.log_params({"experiment": "E2", "features": len(e2_cols)})
            mlflow.log_metrics({**e2_metrics, **e2_cls_metrics})
            results["E2"] = {
                "name": "E2 — Transit + Temporal + Weather",
                "features_count": len(e2_cols),
                "features": e2_cols,
                "test_metrics": e2_metrics,
                "classification_metrics": e2_cls_metrics,
            }

        # ==========================================
        # E3: Multimodal (Transit + Weather + Events + Spatial)
        # ==========================================
        logger.info("Executing Experiment E3: Complete Multimodal...")
        with mlflow.start_run(run_name="E3_Full_Multimodal"):
            e3_cols = exp_features["E3"]
            e3_model = LightGBMModel()
            e3_model.fit(df_train[e3_cols], y_train_reg, y_train_cls, eval_set_reg=(df_val[e3_cols], y_val_reg))
            e3_preds = e3_model.predict_reg(df_test[e3_cols])
            e3_preds_cls = e3_model.predict_cls(df_test[e3_cols])
            e3_metrics = evaluator.evaluate_regression(y_test_reg, e3_preds)
            e3_cls_metrics = evaluator.evaluate_classification(y_test_cls, e3_preds_cls)
            mlflow.log_params({"experiment": "E3", "features": len(e3_cols)})
            mlflow.log_metrics({**e3_metrics, **e3_cls_metrics})
            results["E3"] = {
                "name": "E3 — Full Multimodal (Weather + Events + Spatial)",
                "features_count": len(e3_cols),
                "features": e3_cols,
                "test_metrics": e3_metrics,
                "classification_metrics": e3_cls_metrics,
            }

        # ==========================================
        # E4: Optimized Ensembles (Optuna-Tuned LightGBM, XGBoost, CatBoost)
        # ==========================================
        logger.info("Executing Experiment E4: Optuna Tuning & Model Comparison...")
        e4_cols = exp_features["E4"]
        tuner = OptunaTuner(n_trials=15)
        best_lgb_params = tuner.tune_lightgbm(
            df_train[e4_cols], y_train_reg, df_val[e4_cols], y_val_reg
        )

        with mlflow.start_run(run_name="E4_Optimized_LightGBM"):
            lgb_opt = LightGBMModel(params=best_lgb_params)
            lgb_opt.fit(df_train[e4_cols], y_train_reg, y_train_cls, eval_set_reg=(df_val[e4_cols], y_val_reg))
            lgb_test_preds = lgb_opt.predict_reg(df_test[e4_cols])
            lgb_test_cls = lgb_opt.predict_cls(df_test[e4_cols])
            lgb_metrics = evaluator.evaluate_regression(y_test_reg, lgb_test_preds)
            lgb_cls_metrics = evaluator.evaluate_classification(y_test_cls, lgb_test_cls)
            lgb_latency = evaluator.benchmark_latency(lgb_opt.predict_reg, df_test[e4_cols], n_iterations=300)
            mlflow.log_params(best_lgb_params)
            mlflow.log_metrics({**lgb_metrics, **lgb_cls_metrics, **lgb_latency})
            results["E4_LightGBM"] = {
                "name": "E4 — Optimized LightGBM (Production Default)",
                "best_params": best_lgb_params,
                "test_metrics": lgb_metrics,
                "classification_metrics": lgb_cls_metrics,
                "latency_benchmark": lgb_latency,
            }

        # XGBoost Comparison on E4
        logger.info("Training XGBoost on E4 feature set...")
        with mlflow.start_run(run_name="E4_XGBoost"):
            xgb_model = XGBoostModel()
            xgb_model.fit(df_train[e4_cols], y_train_reg, y_train_cls, eval_set_reg=(df_val[e4_cols], y_val_reg))
            xgb_preds = xgb_model.predict_reg(df_test[e4_cols])
            xgb_metrics = evaluator.evaluate_regression(y_test_reg, xgb_preds)
            xgb_latency = evaluator.benchmark_latency(xgb_model.predict_reg, df_test[e4_cols], n_iterations=200)
            mlflow.log_metrics({**xgb_metrics, **xgb_latency})
            results["E4_XGBoost"] = {
                "name": "E4 — XGBoost",
                "test_metrics": xgb_metrics,
                "latency_benchmark": xgb_latency,
            }

        # CatBoost Comparison on E4
        logger.info("Training CatBoost on E4 feature set...")
        with mlflow.start_run(run_name="E4_CatBoost"):
            cb_model = CatBoostModel()
            cb_model.fit(df_train[e4_cols], y_train_reg, y_train_cls)
            cb_preds = cb_model.predict_reg(df_test[e4_cols])
            cb_metrics = evaluator.evaluate_regression(y_test_reg, cb_preds)
            results["E4_CatBoost"] = {
                "name": "E4 — CatBoost",
                "test_metrics": cb_metrics,
            }

        # Multi-Horizon Forecasting Training (15m, 30m, 60m)
        logger.info("Training Multi-Horizon Forecaster (15m, 30m, 60m)...")
        forecaster = MultiHorizonForecaster()
        forecaster.fit(
            X_train=df_train[e4_cols],
            df_train_full=df_train,
            X_val=df_val[e4_cols],
            df_val_full=df_val,
            best_params=best_lgb_params,
        )
        forecaster.save(self.artifacts_dir / "multi_horizon")

        # Stratified Slices Analysis
        df_eval_slice = df_test.copy()
        df_eval_slice["y_true"] = y_test_reg
        df_eval_slice["y_pred"] = lgb_test_preds
        slices_analysis = evaluator.slice_performance(df_eval_slice)
        results["slices_analysis"] = slices_analysis

        # Save production artifacts
        joblib.dump(lgb_opt, self.artifacts_dir / "best_model.joblib")
        joblib.dump(xgb_model, self.artifacts_dir / "xgboost_model.joblib")

        # Save test sample for dashboard / API smoke testing
        df_test.head(100).to_parquet(self.artifacts_dir / "sample_test_records.parquet", index=False)

        # Save model metadata
        metadata = {
            "model_type": "LightGBM_Optimized",
            "production_artifact": "best_model.joblib",
            "training_samples": len(df_train),
            "test_samples": len(df_test),
            "features": e4_cols,
            "feature_count": len(e4_cols),
            "optuna_best_params": best_lgb_params,
            "final_test_mae": lgb_metrics["mae"],
            "final_test_rmse": lgb_metrics["rmse"],
            "final_test_r2": lgb_metrics["r2"],
            "final_test_accuracy": lgb_cls_metrics["accuracy"],
            "final_test_macro_f1": lgb_cls_metrics["macro_f1"],
            "severe_delay_recall": lgb_cls_metrics["severe_recall"],
            "p95_latency_ms": lgb_latency["latency_p95_ms"],
            "horizons_supported": [15, 30, 60],
            "dataset_mode": "simulated",
            "timestamp": pd.Timestamp.now().isoformat(),
        }

        with open(self.metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        with open(self.results_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        logger.info("Training pipeline complete! Model metadata saved to %s.", self.metadata_file)
        return results
