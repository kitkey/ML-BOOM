import pandas as pd
import numpy as np


class Nadaray_Watson():
    def __init__(self, k: int = 3, metric: str = "euclidean", kernel: str = "gauss", h: int = 3):
        self.k = k
        self.X_train = None
        self.y_train = None
        self.train_size = None
        self.metric = metric
        self.kernel = kernel
        self.h = h

    def _get_metric_value(self, X: pd.DataFrame) -> np.array:
        X = X.to_numpy()
        X_train = self.X_train.to_numpy()
        metric_value = None
        if self.metric == "euclidean":
            metric_value = np.sqrt(np.power(X[:, np.newaxis, :] - X_train, 2).sum(axis=2))
        elif self.metric == "manhattan":
            metric_value = np.abs(X[:, np.newaxis, :] - X_train).sum(axis=2)
        elif self.metric == "chebyshev":
            metric_value = np.abs(X[:, np.newaxis, :] - X_train).max(axis=2)
        elif self.metric == "cosine":
            X_norm = np.linalg.norm(X, axis=1)
            X_train_norm = np.linalg.norm(X_train, axis=1)
            metric_value = 1 - np.dot(X, X_train.T) / (X_norm[:, np.newaxis] * X_train_norm)
        return metric_value

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        self.X_train = X_train
        self.y_train = y_train
        self.train_size = X_train.shape

    def _get_kernel_value(self, distances: np.array) -> np.array:
        kernel_value = (2 * np.pi) ** (-1 / 2) * np.exp(-distances ** 2 / self.h)
        return kernel_value

    def predict(self, X: pd.DataFrame) -> pd.Series:
        distance = self._get_metric_value(X)
        nearest_index = distance.argsort()[:, :self.k]
        nearest_distances = np.sort(distance)[:, :self.k]

        y_nearest = self.y_train.to_numpy()[nearest_index]
        kernel_values = self._get_kernel_value(nearest_distances)
        preds = (np.squeeze(y_nearest, axis=2) * kernel_values).sum(axis=1) / kernel_values.sum(axis=1)
        return preds