import pandas as pd
import numpy as np
from typing import List


class KMeans():
    def __init__(self,
                 n_clusters: int = 3,
                 max_iter: int = 10,
                 n_init: int = 3,
                 random_state: int = 42) -> None:
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n_init = n_init
        self.random_state = random_state
        self.inertia_ = np.inf
        self.cluster_centers_ = None

    def __repr__(self) -> str:
        return f"KMeans class: n_clusters={self.n_clusters}, max_iter={self.max_iter}, n_init={self.n_init}, random_state={self.random_state}"

    def _euclidean(self, X: np.array, centroids: np.array) -> np.array:
        distances = np.sqrt(((X[:, np.newaxis] - centroids) ** 2).sum(axis=2))
        return distances

    def _calculating_point_cluster(self, distances: np.array) -> (np.array, np.array):
        distances_inside_clusters = distances.min(axis=1)
        classes = distances.argmin(axis=1)

        return distances_inside_clusters, classes

    def fit(self, X: pd.DataFrame) -> None:
        np.random.seed(self.random_state)

        columns = X.columns
        for _ in range(self.n_init):
            coordinate = [[] for i in range(self.n_clusters)]
            pred_coordinate = [[0] for i in range(self.n_clusters)]
            cnt_iter = 0
            for centroid in range(self.n_clusters):
                for i in range(columns.size):
                    coordinate[centroid].append(np.random.uniform(X[columns[i]].min(), X[columns[i]].max()))

            while cnt_iter < self.max_iter and pred_coordinate != coordinate:
                distances = self._euclidean(X.to_numpy(), np.array(coordinate))
                distances_inside_clusters, points_clusters = self._calculating_point_cluster(distances)
                # print(distances_inside_clusters.sum())
                pred_coordinate = coordinate.copy()
                for centroid in range(self.n_clusters):

                    centroid_points = X[points_clusters == centroid]
                    if centroid_points.size != 0:
                        centroid_new_coordinate = centroid_points.sum(axis=0) / centroid_points.shape[0]

                        coordinate[centroid] = centroid_new_coordinate.tolist()
                cnt_iter += 1
                # print(cnt_iter,pred_coordinate == coordinate)
            distances = self._euclidean(X.to_numpy(), np.array(coordinate))
            # print(distances)
            distances_inside_clusters, _ = self._calculating_point_cluster(distances)
            wcss = (distances_inside_clusters ** 2).sum()

            # print("x \n", X.to_numpy(),"\n", "coord \n",np.array(coordinate) )
            if wcss < self.inertia_:
                self.inertia_ = wcss
                self.cluster_centers_ = coordinate

    def predict(self, X: pd.DataFrame) -> List[int]:
        distances = self._euclidean(X.to_numpy(), np.array(self.cluster_centers_))
        _, classes = self._calculating_point_cluster(distances)
        classes = classes.tolist()
        return classes
