"""Gradient Boosted Tree Models: LightGBM, XGBoost, CatBoost."""

from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.preprocessing import LabelEncoder

from transitpulse.config import settings
from transitpulse.utils.logger import get_logger

logger = get_logger(__name__)


class LightGBMModel:
    """LightGBM dual regressor and multi-class classifier."""

    def __init__(self, params: dict[str, Any] | None = None):
        self.params = params or {
            "n_estimators": 250,
            "learning_rate": 0.05,
            "max_depth": 6,
            "num_leaves": 31,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "random_state": settings.RANDOM_SEED,
            "n_jobs": -1,
            "verbose": -1,
        }
        self.regressor = lgb.LGBMRegressor(**self.params)
        self.classifier = lgb.LGBMClassifier(**self.params)

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train_reg: np.ndarray,
        y_train_cls: np.ndarray | None = None,
        eval_set_reg: tuple[pd.DataFrame, np.ndarray] | None = None,
    ) -> "LightGBMModel":
        callbacks = [lgb.early_stopping(stopping_rounds=25, verbose=False)] if eval_set_reg else None
        self.regressor.fit(
            X_train,
            y_train_reg,
            eval_set=[eval_set_reg] if eval_set_reg else None,
            callbacks=callbacks,
        )
        if y_train_cls is not None:
            self.classifier.fit(X_train, y_train_cls)
        return self

    def predict_reg(self, X: pd.DataFrame) -> np.ndarray:
        return self.regressor.predict(X)

    def predict_cls(self, X: pd.DataFrame) -> np.ndarray:
        return self.classifier.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.classifier.predict_proba(X)


class XGBoostModel:
    """XGBoost dual regressor and multi-class classifier with LabelEncoder."""

    def __init__(self, params: dict[str, Any] | None = None):
        self.params = params or {
            "n_estimators": 250,
            "learning_rate": 0.05,
            "max_depth": 6,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "random_state": settings.RANDOM_SEED,
            "n_jobs": -1,
            "tree_method": "hist",
        }
        self.regressor = xgb.XGBRegressor(**self.params)
        self.classifier = xgb.XGBClassifier(**self.params)
        self.label_encoder = LabelEncoder()

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train_reg: np.ndarray,
        y_train_cls: np.ndarray | None = None,
        eval_set_reg: tuple[pd.DataFrame, np.ndarray] | None = None,
    ) -> "XGBoostModel":
        self.regressor.fit(
            X_train,
            y_train_reg,
            eval_set=[eval_set_reg] if eval_set_reg else None,
            verbose=False,
        )
        if y_train_cls is not None:
            y_encoded = self.label_encoder.fit_transform(y_train_cls)
            self.classifier.fit(X_train, y_encoded, verbose=False)
        return self

    def predict_reg(self, X: pd.DataFrame) -> np.ndarray:
        return self.regressor.predict(X)

    def predict_cls(self, X: pd.DataFrame) -> np.ndarray:
        y_pred_int = self.classifier.predict(X)
        return self.label_encoder.inverse_transform(y_pred_int)


class CatBoostModel:
    """CatBoost dual regressor and multi-class classifier with LabelEncoder."""

    def __init__(self, params: dict[str, Any] | None = None):
        self.params = params or {
            "iterations": 250,
            "learning_rate": 0.05,
            "depth": 6,
            "random_seed": settings.RANDOM_SEED,
            "verbose": False,
            "thread_count": -1,
        }
        self.regressor = CatBoostRegressor(**self.params)
        self.classifier = CatBoostClassifier(**self.params)
        self.label_encoder = LabelEncoder()

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train_reg: np.ndarray,
        y_train_cls: np.ndarray | None = None,
    ) -> "CatBoostModel":
        self.regressor.fit(X_train, np.asarray(y_train_reg, dtype=np.float64))
        if y_train_cls is not None:
            y_encoded = self.label_encoder.fit_transform(np.asarray(y_train_cls, dtype=str))
            self.classifier.fit(X_train, y_encoded)
        return self

    def predict_reg(self, X: pd.DataFrame) -> np.ndarray:
        return self.regressor.predict(X)

    def predict_cls(self, X: pd.DataFrame) -> np.ndarray:
        y_pred_int = self.classifier.predict(X).ravel().astype(int)
        return self.label_encoder.inverse_transform(y_pred_int)
