"""Multi-Horizon Forecasting Engine for Forward Planning."""

from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from transitpulse.config import settings
from transitpulse.utils.logger import get_logger

logger = get_logger(__name__)


class MultiHorizonForecaster:
    """Manages horizon-specific models for 15, 30, and 60-minute forward delays."""

    def __init__(self, horizons: list[int] = None):
        self.horizons = horizons or settings.FORECAST_HORIZONS
        self.models: dict[int, lgb.LGBMRegressor] = {}

    def fit(
        self,
        X_train: pd.DataFrame,
        df_train_full: pd.DataFrame,
        X_val: pd.DataFrame,
        df_val_full: pd.DataFrame,
        best_params: dict = None,
    ) -> "MultiHorizonForecaster":
        """Trains individual horizon-specific boosting models."""
        base_params = {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "max_depth": 6,
            "random_state": settings.RANDOM_SEED,
            "n_jobs": -1,
            "verbose": -1,
        }
        if best_params:
            base_params.update(best_params)

        for h in self.horizons:
            target_col = f"delay_horizon_{h}m"
            y_train = df_train_full[target_col].values
            y_val = df_val_full[target_col].values

            logger.info("Training %d-minute forward horizon model on %d features...", h, X_train.shape[1])
            model = lgb.LGBMRegressor(**base_params)
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)],
            )
            self.models[h] = model

        return self

    def predict(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        """Predicts delay across all supported operational horizons."""
        preds = {}
        for h, model in self.models.items():
            preds[f"horizon_{h}m"] = np.round(model.predict(X), 2)
        return preds

    def save(self, output_dir: Path) -> None:
        """Serializes multi-horizon models to disk."""
        output_dir.mkdir(parents=True, exist_ok=True)
        for h, model in self.models.items():
            fp = output_dir / f"lightgbm_horizon_{h}m.joblib"
            joblib.dump(model, fp)
        logger.info("Saved multi-horizon models to %s.", output_dir)

    def load(self, model_dir: Path) -> "MultiHorizonForecaster":
        """Loads multi-horizon models from disk."""
        for h in self.horizons:
            fp = model_dir / f"lightgbm_horizon_{h}m.joblib"
            if fp.exists():
                self.models[h] = joblib.load(fp)
        return self
