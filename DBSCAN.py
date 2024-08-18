from collections import deque
import pandas as pd
import numpy as np


class DBSCAN():
    def __init__(self,
                 eps=3,
                 min_samples=3,
                 metric="euclidean") -> None:
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric

    def __repr__(self) -> str:
        return f"DBSCAN class: eps={self.eps}, min_samples={self.min_samples}"

    def _get_distances(self, X: pd.DataFrame) -> np.array:
        distances = None
        X_dist = X.to_numpy()

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

    def _check_neighbours(self, distances: np.array) -> np.array:
        return ((distances < self.eps).sum(axis=1) >= self.min_samples)

    def fit_predict(self, X: pd.DataFrame) -> pd.Series:
        check = np.zeros(X.shape[0], dtype=int)
        clusters = []
        distances = self._get_distances(X)
        is_root = self._check_neighbours(distances)

        for current_point in range(X.shape[0]):
            queue = deque()
            queue.append(current_point)

            while queue:
                cur_point = queue.popleft()

                if not check[cur_point]:
                    if is_root[cur_point] and cur_point == current_point:
                        clusters.append([cur_point])

                        all_neighbours = ((distances[cur_point] < self.eps) == 1)
                        not_checked_neighbours = np.where(check[all_neighbours] == 0)

                        queue.extend(np.where(all_neighbours)[0][not_checked_neighbours])

                    elif is_root[cur_point]:
                        clusters[-1].append(cur_point)

                        all_neighbours = ((distances[cur_point] < self.eps) == 1)
                        not_checked_neighbours = np.where(check[all_neighbours] == 0)

                        queue.extend(np.where(all_neighbours)[0][not_checked_neighbours])

                    elif cur_point != current_point:
                        clusters[-1].append(cur_point)

                    else:
                        continue

                    check[cur_point] = 1

        expected_list = []
        y = pd.Series(data=len(clusters), index=range(X.shape[0]))
        for num, cluster in enumerate(clusters):
            y[cluster] = num

        return y












