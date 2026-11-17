"""Baseline Models for Benchmark Comparison."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Ridge


class HistoricalMeanBaseline:
    """Predicts historical route-level mean delay."""

    def __init__(self):
        self.route_means: dict[int, float] = {}
        self.global_mean: float = 0.0

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "HistoricalMeanBaseline":
        self.global_mean = float(np.mean(y))
        df_tmp = X.copy()
        df_tmp["y"] = y
        self.route_means = df_tmp.groupby("route_code")["y"].mean().to_dict()
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if "route_code" not in X.columns:
            return np.full(len(X), self.global_mean)
        return X["route_code"].map(self.route_means).fillna(self.global_mean).values


class PersistenceBaseline:
    """Predicts current stop delay equals immediately preceding upstream stop delay."""

    def __init__(self):
        self.default_delay: float = 0.0

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "PersistenceBaseline":
        self.default_delay = float(np.mean(y))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if "prev_stop_delay_min" in X.columns:
            return X["prev_stop_delay_min"].fillna(self.default_delay).values
        return np.full(len(X), self.default_delay)


class LinearBaseline:
    """Linear regression baseline with Ridge L2 regularization."""

    def __init__(self, alpha: float = 1.0):
        self.model = Ridge(alpha=alpha, random_state=42)

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "LinearBaseline":
        self.model.fit(X.fillna(0.0), y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X.fillna(0.0))


class RandomForestBaseline:
    """Random Forest regressor and classifier."""

    def __init__(self, n_estimators: int = 100, max_depth: int = 10):
        self.regressor = RandomForestRegressor(
            n_estimators=n_estimators, max_depth=max_depth, n_jobs=-1, random_state=42
        )
        self.classifier = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, n_jobs=-1, random_state=42
        )

    def fit(
        self, X: pd.DataFrame, y_reg: np.ndarray, y_cls: np.ndarray | None = None
    ) -> "RandomForestBaseline":
        self.regressor.fit(X.fillna(0.0), y_reg)
        if y_cls is not None:
            self.classifier.fit(X.fillna(0.0), y_cls)
        return self

    def predict_reg(self, X: pd.DataFrame) -> np.ndarray:
        return self.regressor.predict(X.fillna(0.0))

    def predict_cls(self, X: pd.DataFrame) -> np.ndarray:
        return self.classifier.predict(X.fillna(0.0))
