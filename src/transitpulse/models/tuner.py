"""Hyperparameter Optimization with Optuna."""

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import mean_absolute_error

from transitpulse.config import settings
from transitpulse.utils.logger import get_logger

logger = get_logger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


class OptunaTuner:
    """Tunes gradient boosting hyperparameters to minimize validation MAE."""

    def __init__(self, n_trials: int = 15):
        self.n_trials = n_trials

    def tune_lightgbm(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame,
        y_val: np.ndarray,
    ) -> dict:
        """Runs Optuna study for LightGBM regressor."""
        logger.info("Initiating Optuna hyperparameter tuning (%d trials)...", self.n_trials)

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 150, 400, step=50),
                "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
                "max_depth": trial.suggest_int("max_depth", 4, 10),
                "num_leaves": trial.suggest_int("num_leaves", 15, 63),
                "subsample": trial.suggest_float("subsample", 0.65, 0.95),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 0.95),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
                "random_state": settings.RANDOM_SEED,
                "n_jobs": -1,
                "verbose": -1,
            }
            model = lgb.LGBMRegressor(**params)
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)],
            )
            preds = model.predict(X_val)
            return mean_absolute_error(y_val, preds)

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=self.n_trials)

        logger.info(
            "Optuna optimization finished. Best Validation MAE: %.4f with params: %s",
            study.best_value,
            study.best_params,
        )
        return study.best_params
