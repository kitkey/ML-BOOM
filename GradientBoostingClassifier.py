import random
from collections import defaultdict

import numpy as np
import pandas as pd


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
        y = pd.Series(y)
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
                    y.size < self.min_samples_split)
        if conditions_leaf:
            self.tree[node_num] = {"node_direction": node_direct, "value": y.mean(), "node_num": node_num,
                                   "separator": None, "column": None, "nums": y.index.values, "y": y}
            return

        col_name, split_value, gain = self.get_best_split(X.copy(), y.copy())
        if gain == 0:
            self.tree[node_num] = {"node_direction": node_direct, "value": y.mean(), "node_num": node_num,
                                   "separator": None, "column": None, "nums": y.index.values, "y": y}

            return

        x_more, x_less, y_more, y_less = X[X[col_name] > split_value], X[X[col_name] <= split_value], y[
            X[col_name] > split_value], y[X[col_name] <= split_value]
        self.leafs_cnt += 1
        self.fi[col_name] += self.get_importance(y, y_less, y_more)
        self.tree[node_num] = {"node_direction": node_direct, "value": None, "node_num": node_num,
                               "separator": split_value, "column": col_name, "nums": None}
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


class BoostClf():
    def __init__(self,
                 n_estimators=10,
                 learning_rate=0.1,
                 max_depth=5,
                 min_samples_split=2,
                 max_leafs=20,
                 bins=16,
                 metric=None,
                 max_features=0.5,
                 max_samples=0.5,
                 random_state=42,
                 reg=0.1
                 ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs
        self.bins = bins
        self.pred_0 = None
        self.trees = []
        self.predictions = None
        self.metric = metric
        self.best_score = None
        self.max_features = max_features
        self.max_samples = max_samples
        self.random_state = random_state
        self.reg = reg
        self.leaves_count = 0
        self.fi = {}

    def __repr__(self) -> str:
        return f"BoostClf class: n_estimators={self.n_estimators}, learning_rate={self.learning_rate}, max_depth={self.max_depth}, min_samples_split={self.min_samples_split}, max_leafs={self.max_leafs}, bins={self.bins}"

    def fit(self, X: pd.DataFrame, y: pd.Series, X_eval: pd.DataFrame = None, y_eval: pd.Series = None,
            early_stopping: int = None, verbose: bool = False) -> None:
        self.fi = dict(zip(X.columns.tolist(), [0.0] * X.shape[1]))
        random.seed(self.random_state)
        y = pd.Series(data=y)
        X.reset_index(inplace=True, drop=True)
        y.reset_index(inplace=True, drop=True)

        self.pred_0 = np.log(y.mean() / (1 - y.mean()))

        self.predictions = pd.Series(data=self.pred_0, index=range(y.size))

        if X_eval is not None:
            X_eval.reset_index(inplace=True, drop=True)
            y_eval.reset_index(inplace=True, drop=True)
            self.predictions_eval = pd.Series(data=self.pred_0, index=range(y_eval.size))
            early_stopping_cnt = 0
            last_ind = 0
            best_last_score = np.inf
            pred_value = None
        init_cols = X.columns.values.tolist()
        cols_smpl_cnt = round(self.max_features * X.shape[1])
        init_rows_cnt = X.shape[0]
        rows_smpl_cnt = round(self.max_samples * init_rows_cnt)

        for num_tree in range(1, self.n_estimators + 1):
            lr = self.__calculate_learning_rate(num_tree)
            cols_idx = random.sample(init_cols, cols_smpl_cnt)
            rows_idx = random.sample(range(init_rows_cnt), rows_smpl_cnt)
            X_train = X.loc[rows_idx, cols_idx].copy()
            y_train = y[rows_idx].copy()
            X_train.reset_index(inplace=True, drop=True)
            y_train.reset_index(inplace=True, drop=True)
            grad = self.__calculate_grad(y_train, self.predictions[rows_idx].reset_index(inplace=False, drop=True))

            tree = TreeReg(max_depth=self.max_depth, min_samples_split=self.min_samples_split,
                             max_leafs=self.max_leafs, bins=self.bins, N_source=y.size)
            tree.fit(X_train, -grad)
            self.value_replace(tree.tree, y_train, self.__calculate_log_chance(
                self.predictions[rows_idx].reset_index(inplace=False, drop=True)))
            self.trees.append(tree)
            for key, value in tree.fi.items():
                self.fi[key] += value
            self.leaves_count += tree.leafs_cnt
            self.predictions += lr * tree.predict(X)

            if X_eval is not None:

                self.predictions_eval += lr * tree.predict(X_eval)
                value = 0
                if self.metric is not None:
                    value = self._get_metric_value(y_eval,
                                                   (self.__calculate_log_chance(self.predictions_eval) > 0.5).astype(
                                                       int), self.__calculate_log_chance(self.predictions_eval))
                else:
                    value = self.__calculate_loss(y_eval, self.predictions_eval)
                if num_tree == 1:
                    best_last_score = value
                    pred_value = value
                else:
                    if self.metric is None:
                        if value >= pred_value:
                            early_stopping_cnt += 1
                        else:
                            early_stopping_cnt = 0
                            best_last_score = value
                            pred_value = value
                    else:
                        if value < pred_value:
                            early_stopping_cnt += 1
                        else:
                            early_stopping_cnt = 0
                            best_last_score = value
                            pred_value = value

                if early_stopping_cnt == early_stopping:
                    last_ind = num_tree
                    break

            if verbose:
                loss = self.__calculate_loss(y, self.predictions)
                if self.metric is not None:
                    metric = self._get_metric_value(y,
                                                    (self.__calculate_log_chance(self.predictions) > 0.5).astype(int),
                                                    self.__calculate_log_chance(self.predictions))

        if X_eval is not None:
            if early_stopping_cnt == early_stopping:
                self.trees = self.trees[:last_ind - early_stopping]

                self.best_score = best_last_score

        else:
            if self.metric is not None:
                self.best_score = self._get_metric_value(y,
                                                         (self.__calculate_log_chance(self.predictions) > 0.5).astype(
                                                             int), self.__calculate_log_chance(self.predictions))
            else:
                self.best_score = self.__calculate_loss(y, self.predictions)

    def __calculate_learning_rate(self, iteration):
        if callable(self.learning_rate):
            lr = self.learning_rate(iteration)
        else:
            lr = self.learning_rate
        return lr

    def __calculate_loss(self, y_target: pd.Series, y_logits: pd.Series) -> float:
        loss = - 1 / y_target.size * (
                    y_target * np.log(self.__calculate_log_chance(y_logits)) + (1 - y_target) * np.log(
                1 - self.__calculate_log_chance(y_logits))).sum()
        return loss

    def _get_metric_value(self, y_target: pd.Series, y_predict: pd.Series, y_proba: pd.Series) -> float:
        value = 0
        if self.metric == "accuracy":
            value = (y_target == y_predict).sum() / y_target.size
        elif self.metric == "precision":
            TP = (y_target[y_predict == 1] == 1).sum()
            FP = (y_target[y_predict == 1] == 0).sum()
            value = TP / (TP + FP)
        elif self.metric == "recall":
            TP = (y_target[y_predict == 1] == 1).sum()
            FN = (y_target[y_predict == 0] == 1).sum()
            value = TP / (TP + FN)
        elif self.metric == "f1":
            TP = (y_target[y_predict == 1] == 1).sum()
            FP = (y_target[y_predict == 1] == 0).sum()
            FN = (y_target[y_predict == 0] == 1).sum()

            precision = TP / (TP + FP)
            recall = TP / (TP + FN)
            value = 2 * precision * recall / (precision + recall)
        elif self.metric == "roc_auc":
            Y = pd.DataFrame(data={"proba": y_proba, "class": y_target})
            Y.sort_values(by="proba", inplace=True, ascending=False)
            Y.reset_index(inplace=True, drop=True)
            s = 0
            for i in range(Y.shape[0]):
                if Y.loc[i]["class"] == 0:
                    s += Y[:i][(Y[:i]["proba"] > Y.loc[i]["proba"]) & (Y[:i]["class"] == 1)].shape[0]
                    s += Y[(Y["proba"] == Y.loc[i]["proba"]) & (Y["class"] == 1)].shape[0] / 2

            s /= (y_target == 1).sum() * (y_target == 0).sum()
            value = s
        return value

    def value_replace(self, tree, y_target, y_probas, i=0):
        if tree[i]["value"] is None:
            self.value_replace(tree, y_target, y_probas, 2 * i + 1)
            self.value_replace(tree, y_target, y_probas, 2 * i + 2)
        else:
            indexes = tree[i]["nums"]
            gamma = (y_target[indexes] - y_probas[indexes]).sum() / (
                        (1 - y_probas[indexes]) * y_probas[indexes]).sum() + self.reg * self.leaves_count

            tree[i]["value"] = gamma

    def __calculate_grad(self, y_target: pd.Series, y_logit_prob: pd.Series) -> float:
        grad = self.__calculate_log_chance(y_logit_prob) - y_target  # y_target - probs
        return grad

    def __calculate_log_chance(self, x: pd.Series) -> float:
        return 1 / (1 + np.exp(-x))

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        prediction = pd.Series(data=self.pred_0, index=range(X.shape[0]))

        for i in range(1, len(self.trees) + 1):
            lr = self.__calculate_learning_rate(i)
            prediction += self.trees[i - 1].predict(X) * lr
        probas = self.__calculate_log_chance(prediction)
        return probas

    def predict(self, X: pd.DataFrame) -> pd.Series:
        prediction_probas = self.predict_proba(X)
        prediction_probas = (prediction_probas > 0.5).astype(int)
        return prediction_probas


