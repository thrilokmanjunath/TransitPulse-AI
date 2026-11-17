"""Deep Neural Architecture for Temporal Delay Prediction."""


import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler

from transitpulse.config import settings


class TemporalMLPModel:
    """Multi-Layer Perceptron neural baseline for tabular temporal delay forecasting."""

    def __init__(self, hidden_layer_sizes=(128, 64, 32), max_iter=200):
        self.scaler = StandardScaler()
        self.regressor = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation="relu",
            max_iter=max_iter,
            random_state=settings.RANDOM_SEED,
            early_stopping=True,
            n_iter_no_change=15,
        )
        self.classifier = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation="relu",
            max_iter=max_iter,
            random_state=settings.RANDOM_SEED,
            early_stopping=True,
            n_iter_no_change=15,
        )

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train_reg: np.ndarray,
        y_train_cls: np.ndarray | None = None,
    ) -> "TemporalMLPModel":
        X_scaled = self.scaler.fit_transform(X_train.fillna(0.0))
        self.regressor.fit(X_scaled, y_train_reg)
        if y_train_cls is not None:
            self.classifier.fit(X_scaled, y_train_cls)
        return self

    def predict_reg(self, X: pd.DataFrame) -> np.ndarray:
        X_scaled = self.scaler.transform(X.fillna(0.0))
        return self.regressor.predict(X_scaled)

    def predict_cls(self, X: pd.DataFrame) -> np.ndarray:
        X_scaled = self.scaler.transform(X.fillna(0.0))
        return self.classifier.predict(X_scaled)
