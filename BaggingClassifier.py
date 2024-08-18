import copy
import random
from copy import deepcopy
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, Union, List, DefaultDict


class Node():
    def __init__(self,
                 node_kind: str = "leaf",
                 ) -> None:
        allowed_node_kinds = {"leaf", "node"}
        if node_kind not in allowed_node_kinds:
            raise TypeError(f"node_kind must be one of the allowed kinds : {allowed_node_kinds}")
        self.node_kind = node_kind
        self.l_node = None
        self.r_node = None
        self.node_dictionary = defaultdict(int)
        self.direction = None


class TreeClf():
    def __init__(self,
                 max_depth: int = 5,
                 min_samples_split: int = 2,
                 max_leafs: int = 20,
                 bins: int = None,
                 criterion: str = "entropy",
                 source_size_of_model_ensemble: int = None) -> None:

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs if max_leafs > 1 else 2
        self.tree = Node(node_kind="node")
        self.leafs_cnt = 1
        self.bins = bins
        self.__using_bins = defaultdict(bool)
        self.__bin_seps = defaultdict(int)
        self.criterion = criterion
        self.fi = defaultdict(int)
        self.n_source = source_size_of_model_ensemble

    def _calculate_entropy(self,
                           y: pd.Series) -> float:

        count_elements = np.bincount(y, minlength=2)
        probalities = count_elements / y.size
        probalities = probalities[probalities != 0]
        entropy = - (probalities * np.log2(probalities)).sum()
        return entropy

    def _calculate_gini(self,
                        y: pd.Series) -> float:

        count_elements = np.bincount(y, minlength=2)
        probalities = count_elements / y.size
        probalities = probalities[probalities != 0]
        gini = 1 - (probalities ** 2).sum()
        return gini

    def _calculate_importance(self,
                              y: pd.Series,
                              y_left: pd.Series,
                              y_right: pd.Series) -> float:
        information_function = self._calculate_entropy if self.criterion == "entropy" else self._calculate_gini
        initial_information = information_function(y)
        left_information_with_n_coef = information_function(y_left) * (y_left.size / y.size)
        right_information_with_n_coef = information_function(y_right) * (y_right.size / y.size)
        importance = (y.size / self.n_source) * (
                    initial_information - (left_information_with_n_coef + right_information_with_n_coef))
        return importance

    def _calculate_information_gain(self,
                                    y: pd.Series,
                                    y_left: pd.Series,
                                    y_right: pd.Series
                                    ) -> float:
        information_function = self._calculate_entropy if self.criterion == "entropy" else self._calculate_gini
        initial_information = information_function(y)
        left_information_with_n_coef = information_function(y_left) * (y_left.size / y.size)
        right_information_with_n_coef = information_function(y_right) * (y_right.size / y.size)
        information_gain = initial_information - (left_information_with_n_coef + right_information_with_n_coef)
        return information_gain

    def __precalculate_bins(self, X: pd.DataFrame) -> defaultdict:
        bins = defaultdict(float)
        for column in X.columns:
            if X[column].unique().size - 1 >= self.bins:
                self.__using_bins[column] = True
                bins[column] = np.histogram(X[column], bins=self.bins)[1]
                bins[column] = bins[column][1:-1]
        return bins

    def _get_best_split(self,
                        X: pd.DataFrame,
                        y: pd.Series) -> (str, float, float):

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not isinstance(y, pd.Series):
            y = pd.Series(y)

        X_splitting = (X).copy()
        y_splitting = (y).copy()
        X_splitting.reset_index(inplace=True, drop=True)
        y_splitting.reset_index(inplace=True, drop=True)

        col_name, split_value, ig = None, None, -np.inf

        for column in X_splitting.columns:
            values = X_splitting[column]
            sorted_unique_values = values.sort_values(inplace=False).unique()

            if sorted_unique_values.size > 1:
                separators = (sorted_unique_values[1:] + sorted_unique_values[:-1]) / 2
                if self.__using_bins[column]:
                    separators = self.__bin_seps[column]

                for separator in separators:
                    y_left = y_splitting[values <= separator]
                    y_right = y_splitting[values > separator]
                    if y_left.size == 0 or y_right.size == 0:
                        continue

                    information_gain = self._calculate_information_gain(y_splitting, y_left, y_right)

                    if information_gain > ig:
                        ig = information_gain
                        split_value = separator
                        col_name = column

        if col_name is None:
            col_name, split_value, ig = "0", 0, 0

        return col_name, split_value, ig

    def fit(self,
            X: pd.DataFrame,
            y: pd.Series) -> None:

        X_train = X.copy()
        y_train = y.copy()

        if not isinstance(X_train, pd.DataFrame):
            X_train = pd.DataFrame(X_train)
        if not isinstance(y_train, pd.Series):
            y_train = pd.Series(y_train)

        X_train.reset_index(inplace=True,
                            drop=True)

        y_train.reset_index(inplace=True,
                            drop=True)

        self.__using_bins = defaultdict(bool)
        if self.bins is not None:
            self.__bin_seps = self.__precalculate_bins(X)

        if self.n_source is None:
            self.n_source = y.size
        self.fi = defaultdict(int, dict(zip(X.columns.tolist(), [0] * X.shape[1])))

        current_depth = 0
        self.__build_tree(X_train,
                          y_train,
                          current_node=self.tree,
                          current_depth=current_depth)

    def __build_tree(self,
                     X: pd.DataFrame,
                     y: pd.Series,
                     current_node: 'Node',
                     current_depth: int = 0
                     ) -> None:

        conditions_leaf = (
                (current_depth == self.max_depth) or
                (self.leafs_cnt == self.max_leafs) or
                (y.size < self.min_samples_split)
        )
        if conditions_leaf:
            current_node.node_kind = "leaf"
            current_node.node_dictionary["value"] = y.mean()
            return

        col_name, split_value, ig = self._get_best_split(X, y)

        if ig == 0:
            current_node.node_kind = "leaf"
            current_node.node_dictionary["value"] = y.mean()
            return

        X_left = X[X[col_name] <= split_value]
        X_right = X[X[col_name] > split_value]

        y_left = y[X[col_name] <= split_value]
        y_right = y[X[col_name] > split_value]

        self.fi[col_name] += self._calculate_importance(y, y_left, y_right)

        node_left, node_right = Node(node_kind="node"), Node(node_kind="node")

        node_left.direction = "left"
        node_right.direction = "right"

        current_node.l = node_left
        current_node.r = node_right

        current_node.node_dictionary["column"] = col_name
        current_node.node_dictionary["separator"] = split_value
        current_node.node_dictionary["information_gain"] = ig

        self.leafs_cnt += 1

        self.__build_tree(X=X_left,
                          y=y_left,
                          current_node=current_node.l,
                          current_depth=current_depth + 1)

        self.__build_tree(X=X_right,
                          y=y_right,
                          current_node=current_node.r,
                          current_depth=current_depth + 1)

    def print_tree(self,
                   node: 'Node' = None,
                   i=0,
                   ) -> None:
        if i == 0:
            node = self.tree
        if node.node_kind == "node":
            print("   " * i, node.node_dictionary["column"], " > ", node.node_dictionary["separator"])

            self.print_tree(node.l, i + 1)
            self.print_tree(node.r, i + 1)
        else:
            print("   " * i, node.direction, " = ", node.node_dictionary["value"])

    def predict_proba(self,
                      X: pd.DataFrame) -> float:

        X.reset_index(inplace=True, drop=True)
        y = self.__predict_proba_recursion(X, self.tree)
        return y

    def __predict_proba_recursion(self,
                                  X: pd.DataFrame,
                                  current_node: 'Node',
                                  y: pd.Series = None) -> pd.Series:

        if current_node is self.tree:
            y = pd.Series(data=0, index=range(X.shape[0]))
        if current_node.node_kind == "node":
            separator = current_node.node_dictionary["separator"]
            column = current_node.node_dictionary["column"]

            X_left = X[X[column] <= separator]
            X_right = X[X[column] > separator]
            y = self.__predict_proba_recursion(X=X_left, current_node=current_node.l, y=y)
            y = self.__predict_proba_recursion(X=X_right, current_node=current_node.r, y=y)
        else:
            y[X.index] = current_node.node_dictionary["value"]
        return y

    def predict(self, X: pd.DataFrame) -> pd.Series:

        y = self.predict_proba(X)

        y[y > 0.5] = 1
        y[y <= 0.5] = 0
        return y.astype(int)


class LogReg():
    def __init__(self,
                 n_iter=10,
                 learning_rate=0.1,
                 metric=None,
                 reg=None,
                 l1_coef=0,
                 l2_coef=0,
                 sgd_sample=None,
                 random_state=42
                 ):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = None
        self.metric = metric
        self.metric_value = None
        self.reg = reg
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.sgd_sample = sgd_sample
        self.random_state = random_state

    def __repr__(self):
        return f"LogReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}"

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

    def fit(self, X: pd.DataFrame, y: pd.Series, verbose=False) -> None:
        random.seed(self.random_state)
        X.reset_index(inplace=True, drop=True)
        y.reset_index(inplace=True, drop=True)
        X.insert(loc=0, column="Bias", value=1)
        self.weights = np.ones(X.shape[1])
        grad_reg, reg_loss = 0, 0
        sgd_sample = None
        if type(self.sgd_sample) is float:
            sgd_sample = round(self.sgd_sample * X.shape[0])
        elif type(self.sgd_sample) is int:
            sgd_sample = self.sgd_sample

        for iteration in range(1, self.n_iter + 1):
            X_train = X.copy()
            y_train = y.copy()
            if sgd_sample is not None:
                sample_rows_idx = random.sample(range(X.shape[0]), sgd_sample)
                X_train = X.iloc[sample_rows_idx]
                y_train = y[sample_rows_idx]
                X_train.reset_index(inplace=True, drop=True)
                y_train.reset_index(inplace=True, drop=True)

            y_logits = self.__calculate_logits(X_train)
            grad = self.__calculate_gradient(y_train, y_logits, X_train)
            if self.reg is not None:
                grad_reg = self.__calculate_grad_reg()
                grad += grad_reg
            if verbose:
                y_full_logits = self.__calculate_logits(X)
                loss = self.__calculate_loss(y, y_full_logits)
                if self.reg is not None:
                    reg_loss = self.__calculate_reg_loss()
                    loss += reg_loss
                if (iteration - 1) % verbose == 0:
                    print(f"{self.iteration} | loss: {loss}")
            lr = self.__calculate_learning_rate(iteration)
            self.__update_weights(grad, lr)
        if self.metric is not None:
            y_pred = self.predict(X)
            y_proba = self.predict_proba(X)
            self.metric_value = self._get_metric_value(y, y_pred, y_proba)

    def __calculate_loss(self, y_target: pd.Series, y_logits: pd.Series) -> float:
        loss = - 1 / y_target.size * (
                    y_target * np.log(y_logits + 1e-15) + (1 - y_target) * np.log(1 - y_logits + 1e-15))
        return loss

    def __calculate_grad_reg(self) -> np.array:
        grad = 0
        if self.reg == "l1":
            grad = self.l1_coef * np.sign(self.weights)
        elif self.reg == "l2":
            grad = self.l2_coef * 2 * self.weights
        elif self.reg == "elasticnet":
            grad = self.l1_coef * np.sign(self.weights) + self.l2_coef * 2 * self.weights
        return grad

    def __calculate_learning_rate(self, iteration: int) -> float:
        lr = 0
        if callable(self.learning_rate):
            lr = self.learning_rate(iteration)
        else:
            lr = self.learning_rate
        return lr

    def __calculate_reg_loss(self) -> float:
        loss = 0
        if self.reg == "l1":
            loss = self.l1_coef * np.abs(self.weights).sum()
        elif self.reg == "l2":
            loss = self.l2_coef * (self.weights ** 2).sum()
        elif self.reg == "elasticnet":
            loss = self.l1_coef * np.abs(self.weights).sum() + self.l2_coef * (self.weights ** 2).sum()
        return loss

    def __calculate_logits(self, X: pd.DataFrame) -> pd.Series:
        values = X.to_numpy() @ self.weights.T
        logits = pd.Series(data=(1 / (1 + np.exp(-values))))
        return logits

    def __calculate_gradient(self, y_target: pd.Series, y_logits: pd.Series, X: pd.DataFrame) -> pd.Series:
        grad = 1 / y_target.size * ((y_logits - y_target).to_numpy()) @ X.to_numpy()
        return grad

    def __update_weights(self, grad: np.array, lr: float) -> None:
        self.weights -= lr * grad

    def get_coef(self):
        return self.weights[1:]

    def predict_proba(self, X):
        if "Bias" not in X.columns:
            X.insert(loc=0, column="Bias", value=1)
        probas = self.__calculate_logits(X)
        return probas

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if "Bias" not in X.columns:
            X.insert(loc=0, column="Bias", value=1)
        probas = self.predict_proba(X)
        predictions = (probas > 0.5).astype(int)
        return predictions

    def get_best_score(self):
        return self.metric_value


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
            metric_value = (1 - (row * self.X_train).sum(axis=1) / (np.power(row.pow(2).sum(), 0.5) *
                                                                    self.X_train.pow(2).sum(axis=1).pow(
                                                                        0.5))).to_numpy()
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


class BaggingClf():
    def __init__(self,
                 estimator=None,
                 n_estimators=10,
                 max_samples=1.0,
                 random_state=42,
                 oob_score=None) -> None:
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state
        self.estimators = []
        self.oob_score = oob_score
        self.oob_score_ = None

    def __repr__(self):
        return f"BaggingClf class: estimator={self.estimator}, n_estimators={self.n_estimators}, max_samples={self.max_samples}, random_state={self.random_state}"

    def _get_metric_value(self, y_target: pd.Series, y_predict: pd.Series, y_proba: pd.Series) -> float:
        value = 0
        if self.oob_score == "accuracy":
            value = (y_target == y_predict).sum() / y_target.shape[0]

        elif self.oob_score == "precision":
            TP = (y_target[y_predict == 1] == 1).sum()
            FP = (y_target[y_predict == 1] == 0).sum()
            value = TP / (TP + FP)

        elif self.oob_score == "recall":
            TP = (y_target[y_predict == 1] == 1).sum()
            FN = (y_target[y_predict == 0] == 1).sum()
            value = TP / (TP + FN)

        elif self.oob_score == "f1":
            TP = (y_target[y_predict == 1] == 1).sum()
            FP = (y_target[y_predict == 1] == 0).sum()
            FN = (y_target[y_predict == 0] == 1).sum()

            precision = TP / (TP + FP)
            recall = TP / (TP + FN)
            value = 2 * precision * recall / (precision + recall)

        elif self.oob_score == "roc_auc":
            Y = pd.DataFrame(data={"proba": y_proba.copy(), "class": y_target.copy()})
            Y.sort_values(by="proba", inplace=True, ascending=False)
            Y.reset_index(inplace=True, drop=True)
            s = 0
            Y["proba"] = round(Y["proba"], 10)
            for i in range(Y.shape[0]):
                if Y.loc[i]["class"] == 0:
                    s += Y[:i][(Y[:i]["proba"] > Y.loc[i]["proba"]) & (Y[:i]["class"] == 1)].shape[0]
                    s += Y[(Y["proba"] == Y.loc[i]["proba"]) & (Y["class"] == 1)].shape[0] / 2

            s /= (y_target == 1).sum() * (y_target == 0).sum()
            value = s
        return value

    def fit(self, X, y) -> None:
        random.seed(self.random_state)
        X.reset_index(inplace=True, drop=True)
        y.reset_index(inplace=True, drop=True)

        rows_num_list = list(range(X.shape[0]))
        rows_smpl_cnt = round(self.max_samples * X.shape[0])
        sample_rows_idx = [random.choices(rows_num_list, k=rows_smpl_cnt) for i in range(3)]

        oob_activate = self.oob_score is not None
        predictions_oob = pd.Series(data=0, index=X.index)
        count_values_oob = pd.Series(data=0, index=X.index)
        y_oob = pd.Series(data=np.inf, index=X.index)

        for i in range(self.n_estimators):
            if oob_activate:
                rows_oob = list(set(range(X.index.size)) - set(sample_rows_idx[i]))
                X_oob = X.iloc[rows_oob].copy()
                y_oob[rows_oob] = y[rows_oob].copy()

            estimator = copy.deepcopy(self.estimator)
            estimator.fit(X.iloc[sample_rows_idx[i]], y[sample_rows_idx[i]])
            self.estimators.append(estimator)

            if oob_activate:
                predictions_oob[rows_oob] += estimator.predict_proba(X_oob).tolist()
                count_values_oob[rows_oob] += 1
        if oob_activate:
            count_values_oob[count_values_oob == 0] = np.inf
            predictions_oob /= count_values_oob
            self.oob_score_ = self._get_metric_value(y_oob[y_oob != np.inf], predictions_oob[y_oob != np.inf] > 0.5,
                                                     predictions_oob[y_oob != np.inf])

    def predict_proba(self, X) -> pd.Series:
        s = pd.Series(data=0, index=range(X.shape[0]))
        for i in range(self.n_estimators):
            s += self.estimators[i].predict_proba(X)
        s /= self.n_estimators

        return s

    def predict(self, X, type) -> pd.Series:
        if type == "mean":
            s = (self.predict_proba(X) > 0.5).astype(int)

        elif type == "vote":

            s = pd.Series(data=0, index=range(X.shape[0]))
            sk = pd.DataFrame(columns=list(range(self.n_estimators)))
            for i in range(self.n_estimators):
                sk[i] = self.estimators[i].predict(X)
            t = 1 - np.apply_along_axis(lambda x: np.bincount(x, minlength=2), arr=sk, axis=1)[:, ::-1].argmax(axis=1)
            s = t

        return s


