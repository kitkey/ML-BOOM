import random

import pandas as pd
import numpy as np
class TreeReg():
    def __init__(self, max_depth, min_samples_split, max_leafs, bins=None, N_source=0):
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
                    y.size < self.min_samples_split) or (np.unique(y.values).size <= 1)

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


class ForestReg():
    def __init__(self, n_estimators=10, max_features=0.5, max_samples=0.5, random_state=42, max_depth=5,
                 min_samples_split=2, max_leafs=20, bins=16, oob_score=None):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_samples = max_samples
        self.random_state = random_state
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs
        self.bins = bins
        self.leafs_cnt = 0
        self.forest = []
        self.fi = {}
        self.oob_score = oob_score
        self.oob_score_ = None

    def __repr__(self):
        return f"ForestReg class: n_estimators={self.n_estimators}, max_features={self.max_features}, max_samples={self.max_samples}, max_depth={self.max_depth}, min_samples_split={self.min_samples_split}, max_leafs={self.max_leafs}, bins={self.bins}, random_state={self.random_state}"

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

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.fi = dict(zip(X.columns.values, np.zeros(X.shape[0], dtype=int)))
        random.seed(self.random_state)
        init_cols = X.columns.tolist()

        cols_smpl_cnt = round(self.max_features * len(init_cols))
        init_rows_cnt = X.shape[0]
        rows_smpl_cnt = round(self.max_samples * init_rows_cnt)

        oob_activate = self.oob_score is not None
        predictions_oob = pd.Series(data=0, index=X.index)
        count_values_oob = pd.Series(data=0, index=X.index)
        y_oob = pd.Series(data=np.inf, index=X.index)
        for num_tree in range(self.n_estimators):

            cols_idx = random.sample(init_cols, cols_smpl_cnt)

            rows_idx = random.sample(range(init_rows_cnt), rows_smpl_cnt)

            if oob_activate:
                rows_oob = list(set(range(init_rows_cnt)) - set(rows_idx))

                X_oob = X.iloc[rows_oob]
                y_oob[rows_oob] = y[rows_oob]

            x_train = X.loc[rows_idx, cols_idx]
            y_train = y[rows_idx]
            tree = TreeReg(max_depth=self.max_depth, min_samples_split=self.min_samples_split,
                           max_leafs=self.max_leafs, bins=self.bins, N_source=X.shape[0])

            tree.fit(x_train, y_train)
            if oob_activate:
                predictions_oob[rows_oob] += tree.predict(X_oob).tolist()

                count_values_oob[rows_oob] += 1

            for key, item in tree.fi.items():
                self.fi[key] += item

            self.leafs_cnt += tree.leafs_cnt
            self.forest.append(tree)
        if oob_activate:
            count_values_oob[count_values_oob == 0] = np.inf
            predictions_oob /= count_values_oob
            self.oob_score_ = self._get_metric_value(y_oob[y_oob != np.inf], predictions_oob)

    def predict(self, X: pd.DataFrame):
        y_mean = np.zeros(X.shape[0])
        for tree in self.forest:
            y = tree.predict(X).values
            y_mean += y

        y_mean = y_mean / self.n_estimators
        return y_mean










