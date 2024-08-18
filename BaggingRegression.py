import copy
import random
from typing import List

import numpy as np
import pandas as pd


class LineReg():
    def __init__(self, n_iter=100, learning_rate=0.1, metric=None, reg=None, l1_coef=0, l2_coef=0, sgd_sample=None,
                 random_state=42):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = None
        self.metric = metric
        self.last_metric = None
        self.reg = reg
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.sgd_sample = sgd_sample
        self.random_state = random_state

    def __str__(self):
        string = f"LineReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}"
        return string

    def _get_metric_value(self, y_target: pd.Series, y_pred: pd.Series) -> float:
        n = y_target.size
        metric_value = 0
        if self.metric == "mae":
            metric_value = 1 / n * np.abs((y_target - y_pred)).sum()
        elif self.metric == "rmse":
            metric_value = np.sqrt(1 / n * ((y_target - y_pred) ** 2).sum())
        elif self.metric == "r2":
            metric_value = 1 - ((y_target - y_pred) ** 2).sum() / ((y_target - y_target.mean()) ** 2).sum()
        elif self.metric == "mape":
            metric_value = 100 / n * np.abs((y_target - y_pred) / y_target).sum()
        elif self.metric == "mse":
            metric_value = 1 / n * ((y_target - y_pred) ** 2).sum()
        return metric_value

    def fit(self, X: pd.DataFrame, y: pd.Series, verbose: int = 0) -> None:
        random.seed(self.random_state)
        X.reset_index(inplace=True, drop=True)
        y.reset_index(inplace=True, drop=True)
        if "Bias" not in X.columns:
            X.insert(loc=0, column="Bias", value=1)
        self.weights = np.ones(X.columns.size)
        error_reg, grad_reg = 0, 0
        if type(self.sgd_sample) is float:
            sgd_sample = round(self.sgd_sample * X.shape[0])
        elif self.sgd_sample is not None:
            sgd_sample = self.sgd_sample
        for iteration in range(1, self.n_iter + 1):
            if self.sgd_sample is not None:
                sample_rows_idx = random.sample(range(X.shape[0]), sgd_sample)
                X_train = X.iloc[sample_rows_idx]
                y_train = y[sample_rows_idx]
            else:
                X_train = X.copy()
                y_train = y.copy()
            y_pred = pd.Series(data=X_train.to_numpy() @ self.weights.T, index=y_train.index)
            grad = self.mse_gradient(y_train, y_pred, X_train)
            if self.reg is not None:
                error_reg, grad_reg = self.__regularize()
            error = error_reg
            grad += grad_reg
            lr = self.__calculate_learning_rate(iteration)
            if verbose != False:
                if iteration % verbose == 0:
                    y_full_predict = self.predict(X)
                    error += self.mse_error(self, y, y_full_predict)
                    if self.metric is None:
                        print(f"{iteration} | loss: {error}")
                    else:
                        metric = self._get_metric_value(y, y_full_predict)
                        print(f"{iteration} | loss: {error} | {self.metric}: {metric}")

            self.__update_weights(grad, lr)
        y_pred = X.to_numpy() @ self.weights.T
        self.last_metric = self._get_metric_value(y, y_pred)

    def mse_gradient(self, y_target: pd.Series, y_pred: pd.Series, X: pd.DataFrame) -> np.array:
        X = X.to_numpy()
        grad = 2 / y_pred.size * (y_pred - y_target) @ X
        return grad

    def mse_error(self, y_target: pd.Series, y_pred: pd.Series) -> float:
        error = 1 / y_target.size * ((y_target - y_pred) ** 2).sum()
        return error

    def __update_weights(self, grad: pd.Series, lr: float) -> None:
        self.weights -= lr * grad

    def get_coef(self) -> np.array:
        return self.weights[1:]

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if "Bias" not in X.columns:
            X.insert(loc=0, column="Bias", value=1)
        y_pred = X.to_numpy() @ self.weights.T
        return y_pred

    def get_best_score(self) -> float:
        return self.last_metric

    def __regularize(self) -> (float, np.array):
        reg_loss, reg_grad = None, None
        if self.reg == "l1":
            reg_loss = self.l1_coef * np.abs(self.weights).sum()
            reg_grad = self.l1_coef * np.sign(self.weights)
        elif self.reg == "l2":
            reg_loss = self.l2_coef * ((self.weights) ** 2).sum()
            reg_grad = 2 * self.l2_coef * self.weights
        elif self.reg == "elasticnet":
            reg_loss = self.l1_coef * np.abs(self.weights).sum() + self.l2_coef * ((self.weights) ** 2).sum()
            reg_grad = self.l1_coef * np.sign(self.weights) + 2 * self.l2_coef * self.weights
        return reg_loss, reg_grad

    def __calculate_learning_rate(self, iteration) -> float:
        if callable(self.learning_rate):
            lr = self.learning_rate(iteration)
        else:
            lr = self.learning_rate
        return lr


class KNNReg():
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


class TreeReg():
    def __init__(self, max_depth=5, min_samples_split=2, max_leafs=20, bins=None, N_source=0):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs if max_leafs > 1 else 2
        self.leafs_cnt = 1
        self.tree = [0 for i in range(10000)]
        self.__predictions = None
        self.sum_prob_for_test = 0
        self.bins = bins
        self.bin_seps = None
        self.bin_activate = {}
        self.fi = dict()
        self.y_size = 0
        self.N_source = N_source

    def __repr__(self):
        return f"TreeReg class: max_depth={self.max_depth}, min_samples_split={self.min_samples_split}, max_leafs={self.max_leafs}"

    @staticmethod
    def get_mse_value(y):
        y_c = y.values
        mse = np.sum(np.vectorize(lambda x: x ** 2)(y_c - y_c.mean())) / y.size
        return mse

    @staticmethod
    def get_gain(y, y_left, y_right):
        y_mse, y_left_mse, y_right_mse = TreeReg.get_mse_value(y), TreeReg.get_mse_value(
            y_left), TreeReg.get_mse_value(y_right)
        current_gain = (y_mse - (y_left.size / y.size * y_left_mse + y_right.size / y.size * y_right_mse))
        return current_gain

    def get_importance(self, x, x_left, x_right):
        importance = x.size / self.y_size * (TreeReg.get_mse_value(x) - (
                    x_left.size / x.size * TreeReg.get_mse_value(
                x_left) + x_right.size / x.size * TreeReg.get_mse_value(x_right)))
        return importance

    def __get_hist(self, X: pd.DataFrame):
        for col in X.columns.tolist():
            x = np.unique(X[col].values)
            _, seps = np.histogram(x, bins=self.bins)
            self.bin_seps[col] = seps[1:-1]

    def get_best_split(self, X: pd.DataFrame, y: pd.Series):
        X.reset_index(inplace=True, drop=True)
        y.reset_index(inplace=True, drop=True)
        cols = X.columns.tolist()
        best_gain = -10 ** 9
        best_col = None
        best_separator = None
        for col in cols:
            separators = np.array([])
            if col not in self.bin_activate.keys():
                self.bin_activate[col] = 0
            X.sort_values(by=col, inplace=True)
            y = y.iloc[X.index]
            vals = X[col].values
            vals = np.unique(vals)
            if not self.bin_activate[col]:
                separators = (vals[1:] + vals[:-1]) / 2
            if self.bins is not None:
                if separators.size > self.bins or self.bin_activate[col]:
                    separators = self.bin_seps[col].values.tolist()
                    self.bin_activate[col] = 1
            for separator in separators:
                if not (X[col].min() <= separator < X[col].max()):
                    continue
                y_more, y_less = y[X[col] > separator], y[X[col] <= separator]

                gain = TreeReg.get_gain(y, y_less, y_more)
                if gain > best_gain:
                    best_gain = gain
                    best_col = col
                    best_separator = separator
        col_name, split_value, gain = best_col, best_separator, best_gain
        return col_name, split_value, gain

    def fit(self, X, y):
        X.reset_index(inplace=True, drop=True)
        y.reset_index(inplace=True, drop=True)
        self.fi = dict(zip(X.columns.tolist(), np.zeros(X.columns.size, dtype=int)))
        if self.bins is not None:
            self.bin_seps = pd.DataFrame(columns=X.columns.tolist())
            self.__get_hist(X)
        self.__get_tree(X, y)

    def __get_tree(self, X, y, depth=0, node_num=0):
        node_direct = "left" if node_num % 2 == 1 else "right"
        if node_num == 0:
            self.y_size = y.size if not self.N_source else self.N_source
        conditions_leaf = (depth >= self.max_depth) or (self.leafs_cnt >= self.max_leafs) or (
                    y.size < self.min_samples_split) or \
                          (np.unique(y.values).size <= 1)
        if conditions_leaf:
            self.tree[node_num] = {"node_direction": node_direct, "value": y.mean(), "node_num": node_num,
                                   "separator": None, "column": None}
            return

        col_name, split_value, gain = self.get_best_split(X.copy(), y.copy())
        if gain == 0:
            self.tree[node_num] = {"node_direction": node_direct, "value": y.mean(), "node_num": node_num,
                                   "separator": None, "column": None}
            return
        x_more, x_less, y_more, y_less = X[X[col_name] > split_value], X[X[col_name] <= split_value], y[
            X[col_name] > split_value], y[X[col_name] <= split_value]
        self.leafs_cnt += 1
        self.fi[col_name] += self.get_importance(y, y_less, y_more)
        self.tree[node_num] = {"node_direction": node_direct, "value": None, "node_num": node_num,
                               "separator": split_value, "column": col_name}
        self.__get_tree(x_less.copy(), y_less.copy(), depth=depth + 1, node_num=node_num * 2 + 1)
        self.__get_tree(x_more.copy(), y_more.copy(), depth=depth + 1, node_num=node_num * 2 + 2)
        return

    def print_tree(self, i=0, ots=0):
        if self.tree[i] == 0:
            return
        if self.tree[i]["value"] is None:
            print(ots * " ", self.tree[i]["column"], " > ", self.tree[i]["separator"])
        else:
            print(ots * " ", self.tree[i]["node_direction"], " = ", self.tree[i]["value"])
            self.sum_prob_for_test += self.tree[i]["value"]
        self.print_tree(2 * i + 1, ots + 4)
        self.print_tree(2 * i + 2, ots + 4)

    def predict(self, X: pd.DataFrame):
        X.reset_index(drop=True, inplace=True)
        self.search(X)
        return self.__predictions

    def search(self, X: pd.DataFrame, current_index=0):
        if current_index == 0:
            self.__predictions = pd.Series(data=np.zeros(X.shape[0]))
        if self.tree[current_index]["value"] is None:
            x_more = X[X[self.tree[current_index]["column"]] > self.tree[current_index]["separator"]]
            x_less = X[X[self.tree[current_index]["column"]] <= self.tree[current_index]["separator"]]
            self.search(x_less, current_index * 2 + 1)
            self.search(x_more, current_index * 2 + 2)
        else:
            self.__predictions[X.index] = self.tree[current_index]["value"]


class BaggingReg():
    def __init__(self,
                 estimator=None,
                 n_estimators=10,
                 max_samples=1.0,
                 random_state=42,
                 oob_score=None) -> None:
        self.estimator = estimator
        self.estimators = []
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state
        self.oob_score = oob_score
        self.oob_score_ = None

    def __repr__(self):
        return f"BaggingReg class: estimator={self.estimator}, n_estimators={self.n_estimators}, max_samples={self.max_samples}, random_state={self.random_state}"

    def _get_metric_value(self, y_target: pd.Series, y_pred: pd.Series) -> float:
        n = y_target.shape[0]
        metric_value = 0
        if self.oob_score == "mae":
            metric_value = 1 / n * np.abs((y_target - y_pred)).sum()
        elif self.oob_score == "rmse":
            metric_value = np.sqrt(1 / n * ((y_target - y_pred) ** 2).sum())
        elif self.oob_score == "r2":
            metric_value = 1 - ((y_target - y_pred) ** 2).sum() / ((y_target - y_target.mean()) ** 2).sum()
        elif self.oob_score == "mape":
            metric_value = 100 / n * np.abs((y_target - y_pred) / y_target).sum()
        elif self.oob_score == "mse":
            metric_value = 1 / n * ((y_target - y_pred) ** 2).sum()
        return metric_value

    def fit(self, X, y):
        random.seed(self.random_state)
        X.reset_index(drop=True, inplace=True)
        y.reset_index(drop=True, inplace=True)
        rows_num_list = list(range(X.shape[0]))
        rows_smpl_cnt = round(self.max_samples * X.shape[0])
        sample_rows_idx = [random.choices(rows_num_list, k=rows_smpl_cnt) for i in range(3)]

        oob_activate = self.oob_score is not None
        predictions_oob = pd.Series(data=0, index=X.index)
        count_values_oob = pd.Series(data=0, index=X.index)
        y_oob = pd.Series(data=np.inf, index=X.index)
        for i in range(self.n_estimators):
            if oob_activate:
                rows_oob = list(set(rows_num_list) - set(sample_rows_idx[i]))
                X_oob = X.iloc[rows_oob]
                y_oob[rows_oob] = y[rows_oob]

            estimator = copy.deepcopy(self.estimator)
            estimator.fit(X.iloc[sample_rows_idx[i]], y[sample_rows_idx[i]])

            if oob_activate:
                predictions_oob[rows_oob] += estimator.predict(X_oob).tolist()
                count_values_oob[rows_oob] += 1
            self.estimators.append(estimator)

        if oob_activate:
            count_values_oob[count_values_oob == 0] = np.inf
            predictions_oob /= count_values_oob
            self.oob_score_ = self._get_metric_value(y_oob[y_oob != np.inf], predictions_oob)

    def predict(self, X):
        s = pd.Series(data=0, index=range(X.shape[0]))
        for i in range(self.n_estimators):
            s += self.estimators[i].predict(X)
        s /= self.n_estimators
        return s


