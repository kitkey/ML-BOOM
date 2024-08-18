import numpy as np
import pandas as pd

class KNNClf():
    def __init__(self, k=3, metric="euclidean", weight="uniform"):
        self.k = k
        self.X = None
        self.y = None
        self.train_size = None
        self.metric = metric
        self.weight = weight

    def __repr__(self):
        return f"KNNClf class: k={self.k}"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        X.reset_index(inplace=True, drop=True)
        y.reset_index(inplace=True, drop=True)
        self.X = X
        self.y = y
        self.train_size = (X.shape[0], X.shape[1])

    def predict(self, X_eval: pd.DataFrame) -> pd.Series:
        distances = self._get_distance_value(X_eval)
        nearest_neighbours = distances.argsort(axis=1)[:, :self.k]
        vals = self.y.to_numpy()[nearest_neighbours]

        if self.weight == "uniform":

            el_counts = np.apply_along_axis(lambda x: np.bincount(x, minlength=2), axis=1, arr=vals)
            el_counts[:, -1] = el_counts[:, -1] + 1
            pred_classes = el_counts.argmax(axis=1)

        elif self.weight == "rank":

            count_sum = (1 / np.array(range(1, vals.shape[1] + 1))).sum()
            ind1 = np.stack(np.where(vals == 1)).astype(float)
            ind1[1] = 1 / (ind1[1] + 1)
            sum_weights_one = np.bincount(ind1[0].astype(int), weights=ind1[1], minlength=vals.shape[0]) / count_sum
            pred_classes = pd.Series(data=(sum_weights_one >= 1 - sum_weights_one))

        elif self.weight == "distance":

            distances_s = np.sort(distances, axis=1)[:, :self.k]
            count_sum = (1 / distances_s).sum(axis=1)
            ind1 = np.stack(np.where(vals == 1)).astype(float)

            ind1[1] = (1 / distances_s)[vals == 1]

            sum_weights_one = np.bincount(ind1[0].astype(int), weights=ind1[1], minlength=vals.shape[0]) / count_sum

            pred_classes = pd.Series(data=(sum_weights_one >= 0.5))

        return pred_classes

    def _get_distance_value(self, X_eval: pd.DataFrame) -> np.array:
        metric_value = None
        if self.metric == "euclidean":
            metric_value = np.sqrt(((X_eval.to_numpy()[:, None, :] - self.X.to_numpy()) ** 2).sum(axis=2))

        elif self.metric == "chebyshev":
            metric_value = np.abs(X_eval.to_numpy()[:, None, :] - self.X.to_numpy()).max(axis=2)

        elif self.metric == "manhattan":
            metric_value = np.abs(X_eval.to_numpy()[:, None, :] - self.X.to_numpy()).sum(axis=2)
        elif self.metric == "cosine":
            metric_value = 1 - ((X_eval.to_numpy()[:, None, :] * self.X.to_numpy()).sum(axis=2)) / np.sqrt(
                ((X_eval.to_numpy()[:, None, :] ** 2).sum(axis=2) * (self.X.to_numpy() ** 2).sum(axis=1)))
        return metric_value

    def predict_proba(self, X_eval: pd.DataFrame) -> pd.Series:
        distances = self._get_distance_value(X_eval)

        nearest_neighbours = distances.argsort(axis=1)[:, :self.k]
        vals = self.y.to_numpy()[nearest_neighbours]
        if self.weight == "uniform":
            el_counts = np.apply_along_axis(lambda x: np.bincount(x, minlength=2), axis=1, arr=vals)
            prob_one = pd.Series(el_counts[:, 1] / (el_counts[:, 1] + el_counts[:, 0]))

        elif self.weight == "rank":
            count_sum = (1 / np.array(range(1, vals.shape[1] + 1))).sum()
            ind1 = np.stack(np.where(vals == 1)).astype(float)
            ind1[1] = 1 / (ind1[1] + 1)
            prob_one = np.bincount(ind1[0].astype(int), weights=ind1[1], minlength=vals.shape[0]) / count_sum
        elif self.weight == "distance":

            distances_s = np.sort(distances, axis=1)[:, :self.k]
            count_sum = (1 / distances_s).sum(axis=1)
            ind1 = np.stack(np.where(vals == 1)).astype(float)

            ind1[1] = (1 / distances_s)[vals == 1]
            prob_one = np.bincount(ind1[0].astype(int), weights=ind1[1], minlength=vals.shape[0]) / count_sum

        return prob_one








