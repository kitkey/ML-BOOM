import numpy as np
import pandas as pd
from collections import defaultdict
from typing import List


class Agglomerative():
    def __init__(self, n_clusters=3, metric="euclidean") -> None:
        self.n_clusters = n_clusters
        self.metric = metric

    def __repr__(self) -> str:
        return f"Agglomerative class: n_clusters={self.n_clusters}"

    def _get_distances(self, X: pd.DataFrame) -> np.array:
        distances = None
        X_dist = np.vstack(X.to_numpy().copy())

        if self.metric == "euclidean":
            distances = np.sqrt(((X_dist[:, np.newaxis] - X_dist) ** 2).sum(axis=2))
        elif self.metric == "chebyshev":
            distances = np.abs(X_dist[:, np.newaxis] - X_dist).max(axis=2)
        elif self.metric == "manhattan":
            distances = np.abs(X_dist[:, np.newaxis] - X_dist).sum(axis=2)
        elif self.metric == "cosine":
            distances = 1 - ((X_dist[:, None, :] * X_dist).sum(axis=2)) / np.sqrt(
                ((X_dist[:, None, :] ** 2).sum(axis=2) * (X_dist ** 2).sum(axis=1)))

        distances[distances == 0] = np.inf
        return distances

    def _clusterize(self,
                    distances: np.array,
                    current_clusters: pd.DataFrame):
        current_clusters.reset_index(drop=True, inplace=True)
        arg_nearest = np.argmin(distances)

        arg_current_cluster = arg_nearest // distances.shape[1]
        arg_nearest_cluster = arg_nearest % distances.shape[1]

        size_of_first_cluster = len(current_clusters["current_points"][arg_current_cluster])
        size_of_second_cluster = len(current_clusters["current_points"][arg_nearest_cluster])

        current_clusters["current_points"][arg_current_cluster].extend(
            current_clusters["current_points"][arg_nearest_cluster])

        current_clusters["centroid"][arg_current_cluster] = (current_clusters["centroid"][
                                                                 arg_current_cluster] * size_of_first_cluster +
                                                             current_clusters["centroid"][
                                                                 arg_nearest_cluster] * size_of_second_cluster) / (
                                                                        size_of_first_cluster + size_of_second_cluster)
        current_clusters.drop([arg_nearest_cluster], inplace=True)
        return current_clusters

    def fit_predict(self, X: pd.DataFrame) -> np.array:
        # dict_of_clusters: defaultdict[int, list[list[int] | np.ndarray]] = defaultdict(list)
        data_of_clusters = pd.DataFrame(data={"current_points": range(X.shape[0]), "centroid": X.values.tolist()})
        data_of_clusters["current_points"] = data_of_clusters['current_points'].apply(lambda x: [x])
        data_of_clusters["centroid"] = data_of_clusters['centroid'].apply(lambda x: np.array(x))

        while data_of_clusters.shape[0] != self.n_clusters:
            distances = self._get_distances(data_of_clusters["centroid"])

            data_of_clusters = self._clusterize(distances, data_of_clusters)
        data_of_clusters.reset_index(inplace=True, drop=True)
        y = np.zeros(X.shape[0], dtype=int)
        for i in range(self.n_clusters):
            y[np.array(data_of_clusters["current_points"][i])] = np.array(data_of_clusters["current_points"][i][0])
        return y



