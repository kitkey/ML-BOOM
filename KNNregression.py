import pandas as pd
import numpy as np

class KNNreg():
    def __init__(self, k: int = 3, metric: str = "euclidean", weight: str = "uniform"):
        self.k = k
        self.X_train = None
        self.y_train = None
        self.train_size = None
        self.metric = metric
        self.weight = weight

    def __repr__(self):
        return f"KNNReg class: k={self.k}"

    def _get_metric_value(self, X: pd.DataFrame) -> np.array:
        X = X.to_numpy()
        X_train = self.X_train.to_numpy()
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

    def predict(self, X: pd.DataFrame) -> pd.Series:
        distance = self._get_metric_value(X)
        nearest_index = distance.argsort()[:, :self.k]
        y_nearest = self.y_train.to_numpy()[nearest_index]
        if self.weight == "uniform":
            prediction = np.apply_along_axis(np.mean, arr=y_nearest, axis=1)
        elif self.weight == "rank":
            count_weight = (1 / np.array(range(1, self.k + 1))).sum()
            weights = (1 / np.arange(1, X.shape[1] + 1, 1)) / count_weight
            prediction = weights[np.newaxis, :] * y_nearest
        elif self.weight == "distance":
            distances = np.sort(distance)[:, :self.k]
            count_distances = (1 / distances).sum(axis=1)
            weights = (1 / distances) / count_distances[:, np.newaxis]
            prediction = weights * y_nearest

        return prediction
