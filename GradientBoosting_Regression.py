import random
import pandas as pd
import numpy as np

class TreeReg():
    def __init__(self, max_depth = 5, min_samples_split = 2, max_leafs = 20, bins = None, N_source = 0):
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
        mse = np.sum(np.vectorize(lambda x: x**2)(y_c-y_c.mean())) / y.size
        return mse
    
    @staticmethod
    def get_gain(y, y_left, y_right):
        y_mse, y_left_mse, y_right_mse = TreeReg.get_mse_value(y), TreeReg.get_mse_value(y_left), TreeReg.get_mse_value(y_right)
        current_gain = (y_mse - (y_left.size / y.size * y_left_mse + y_right.size / y.size * y_right_mse)) 
        return current_gain
    
    def get_importance(self, x, x_left, x_right):
        importance = x.size / self.y_size * (TreeReg.get_mse_value(x) - (x_left.size / x.size * TreeReg.get_mse_value(x_left) + x_right.size / x.size * TreeReg.get_mse_value(x_right)))
        return importance
    
    def __get_hist(self, X : pd.DataFrame):
        for col in X.columns.tolist():
            x = np.unique(X[col].values)
            _, seps = np.histogram(x, bins=self.bins)
            self.bin_seps[col] = seps[1:-1]
         
    def get_best_split(self, X : pd.DataFrame, y : pd.Series):
        X.reset_index(inplace=True, drop=True)
        y.reset_index(inplace=True, drop=True)
        cols = X.columns.tolist()
        best_gain = -10**9
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
   
    def __get_tree(self, X, y, depth = 0, node_num = 0):
        node_direct = "left" if node_num % 2 == 1 else "right"
        if node_num == 0:
            self.y_size = y.size if not self.N_source else self.N_source
        conditions_leaf = (depth >= self.max_depth) or (self.leafs_cnt >= self.max_leafs) or (y.size < self.min_samples_split)
        if conditions_leaf:
            self.tree[node_num] = {"node_direction": node_direct, "value": y.mean(), "node_num": node_num, "separator": None, "column": None, "nums": y.index.values, "y": y}
            return 
        
        col_name, split_value, gain = self.get_best_split(X.copy(), y.copy())
        if gain == 0:
            self.tree[node_num] = {"node_direction": node_direct, "value": y.mean(), "node_num": node_num, "separator": None, "column": None, "nums": y.index.values, "y": y}
           
            return 

        x_more, x_less, y_more, y_less = X[X[col_name] > split_value], X[X[col_name] <= split_value], y[X[col_name] > split_value], y[X[col_name] <= split_value]
        self.leafs_cnt += 1
        self.fi[col_name] += self.get_importance(y, y_less, y_more)
        self.tree[node_num] = {"node_direction": node_direct, "value": None, "node_num": node_num, "separator": split_value, "column": col_name, "nums": None}
        self.__get_tree(x_less.copy(), y_less.copy(), depth = depth + 1, node_num = node_num * 2 + 1)
        self.__get_tree(x_more.copy(), y_more.copy(), depth = depth + 1, node_num = node_num * 2 + 2)
        return
        
    def print_tree(self, i = 0, ots = 0):
        if self.tree[i] == 0:
            return
        if self.tree[i]["value"] is None:
            print(ots*" ", self.tree[i]["column"], " > ", self.tree[i]["separator"]) 
        else:
            print(ots*" ", self.tree[i]["node_direction"], " = ", self.tree[i]["value"])
            self.sum_prob_for_test += self.tree[i]["value"]
        self.print_tree(2*i+1, ots+4)
        self.print_tree(2*i+2, ots+4)
        
    def predict(self, X : pd.DataFrame):
        X.reset_index(drop=True, inplace=True)
        self.search(X)
        return self.__predictions
    
    def search(self, X : pd.DataFrame, current_index = 0):
        if current_index == 0:
            self.__predictions = pd.Series(data = np.zeros(X.shape[0]))
        if self.tree[current_index]["value"] is None:
            x_more = X[X[self.tree[current_index]["column"]] > self.tree[current_index]["separator"]]
            x_less = X[X[self.tree[current_index]["column"]] <= self.tree[current_index]["separator"]]
            self.search(x_less, current_index * 2 + 1)
            self.search(x_more, current_index * 2 + 2)
        else:
            self.__predictions[X.index] = self.tree[current_index]["value"]
            
            
            
            
            
            
            
            
            
            
class BoostReg():
    def __init__(self, 
                 n_estimators = 10, 
                 learning_rate = 0.1, 
                 max_depth = 5,
                 min_samples_split = 2,
                 max_leafs = 20,
                 bins = 16,
                 loss = "MSE", 
                 metric=None,
                 max_features = 0.5,
                 max_samples =0.5,
                 random_state =42,
                 reg = 0.1,
                 ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs
        self.bins = bins
        self.pred_0 = 0
        self.trees = []
        self.loss = loss
        self.metric = metric
        self.best_score = 0
        self.leaves_cnt = 0
        self.max_features = max_features
        self.max_samples = max_samples
        self.random_state = random_state
        self.reg = reg
        self.fi = {}
        
    def __repr__(self):
        return f"BoostReg class: n_estimators={self.n_estimators}, learning_rate={self.learning_rate}, max_depth={self.max_depth}, min_samples_split={self.min_samples_split}, max_leafs={self.max_leafs}, bins={self.bins}"
    
    @staticmethod
    def mse_loss(target, y):
        antigrad = 2*(target-y).values
        loss = 1/target.size *((y-target)**2).sum()
        return loss, antigrad
    
    @staticmethod
    def mae_loss(target, y):
        loss = 1/target.size * np.sum(np.abs(target - y))
        antigrad = np.sign(target-y)
        return loss, antigrad
    
    def _get_metric_value(self, y_target: pd.Series, y_pred : pd.Series) -> float:
        n = y_target.size
        metric_value = 0
        if self.metric == "MAE":
            metric_value = 1 / n * np.abs((y_target - y_pred)).sum()
        elif self.metric == "RMSE":
            metric_value = np.sqrt(1 / n * ((y_target - y_pred)**2).sum())
        elif self.metric == "R2":
            metric_value = 1 - ((y_target - y_pred)**2).sum() / ((y_target - y_target.mean())**2).sum()
        elif self.metric == "MAPE":
            metric_value = 100 / n * np.abs((y_target - y_pred) / y_target).sum()
        elif self.metric == "MSE":
            metric_value = 1 / n * ((y_target-y_pred)**2).sum()
        return metric_value
    
    def fit(self, X : pd.DataFrame, y, X_eval = None, y_eval = None, early_stopping = None, verbose=None) -> None: 
        random.seed(self.random_state)
        y = pd.Series(data=y)
        self.fi = dict(zip(X.columns, [0.0]*X.shape[1]))
        
        loss_func = BoostReg.mse_loss if self.loss == "MSE" else BoostReg.mae_loss
        self.pred_0 = y.mean() if self.loss == "MSE" else y.median()
        
    
        
        predictions = pd.Series(data=self.pred_0, index=range(X.shape[0]))
        init_cols = X.columns.tolist()
        cols_smpl_cnt = round(self.max_features * X.columns.size)
        init_rows_cnt = X.shape[0]
        rows_smpl_cnt = round(self.max_samples * X.shape[0])
        
        is_eval_metric = (X_eval is not None) and (y_eval is not None) and (early_stopping is not None)
        if is_eval_metric:
            y_eval = pd.Series(data=y_eval)
            X_eval.reset_index(inplace=True, drop=True)
            y_eval.reset_index(inplace=True, drop=True)
            pred_eval = predictions.copy()
            early_stopping_count = 0
            is_stopping = 0
            best_score = 1e20
        for num_tree in range(1, self.n_estimators+1):
            
            lr = self.__calculate_learning_rate(num_tree)
            
            cols_idx = random.sample(init_cols, cols_smpl_cnt)
            rows_idx = random.sample(range(init_rows_cnt), rows_smpl_cnt)
            X_train = X.loc[rows_idx, cols_idx].copy()
            y_train = y[rows_idx].copy()
            X_train.reset_index(inplace=True, drop=True)
            y_train.reset_index(inplace=True, drop=True)
            
            loss, antigrad = loss_func(y_train.copy(), predictions[rows_idx].copy().reset_index(inplace=False, drop=True))
            
            tree = TreeReg(max_depth = self.max_depth, min_samples_split = self.min_samples_split, max_leafs = self.max_leafs, bins = self.bins, N_source = y.size)
            tree.fit(X_train, antigrad)
            
            for key, value in tree.fi.items():
                self.fi[key] += value
            
            tree.tree = self.leaf_search(tree.tree, y_train.copy(), predictions[rows_idx].copy())
            predictions += lr * tree.predict(X)
            
            if is_eval_metric:
                pred_eval += lr * tree.predict(X_eval)
                metric_value = 0
                if self.metric is not None:
                    metric_value = self._get_metric_value(y_eval.copy(), pred_eval.copy())
                else:
                    metric_value, _ = loss_func(y_eval.copy(), pred_eval.copy())
                best_score = min(best_score, metric_value)
                if num_tree > 1:
                    if metric_value > pred_metric_value:
                        early_stopping_count += 1
                    else:
                        early_stopping_count = 0
                pred_metric_value = metric_value
            if verbose is not None:
                verbose_output = f"{num_tree}. mse loss: {loss}"
                if self.metric is not None:
                    metric_value = self._get_metric_value(y, predictions)
                    verbose_output += f"{self.metric}: {metric_value}"
                    if is_eval_metric:
                        verbose_output += f"| eval_metric: {metric_value}"
            if is_eval_metric:
                if early_stopping_count == early_stopping:
                    is_stopping = True
                    break
            if self.metric is None and num_tree == self.n_estimators - 1:
                loss, _ = loss_func(y.copy(), predictions.copy()) 
            self.leaves_cnt += tree.leafs_cnt
            self.trees.append(tree)
        if is_stopping:
            self.trees = self.trees[:early_stopping_count]
            self.best_score = best_score
        else:
            if self.metric is not None:
                self.best_score = self._get_metric_value(y, predictions)
            else:
                self.best_score = loss_func(y.copy(), predictions.copy())[0]
            
    
    def leaf_search(self, tree, y, predictions, i = 0):
        if tree[i] == 0:
            pass
        elif tree[i]["value"] is None:
            tree = self.leaf_search(tree, y.copy(), predictions, 2 * i + 1)
            tree = self.leaf_search(tree, y.copy(), predictions, 2 * i + 2)
            
        else:
            predictions.reset_index(inplace=True, drop=True)
            s = y[tree[i]["nums"]] - predictions[tree[i]["nums"]]
  
            tree[i]["value"] = s.mean() if self.loss == "MSE" else s.median()
            tree[i]["value"] += self.reg * self.leaves_cnt
        return tree
    def __calculate_learning_rate(self, iteration : int) -> float:
        lr = 0
        if callable(self.learning_rate):
            lr = self.learning_rate(iteration)
        else:
            lr = self.learning_rate
        return lr
    
    def predict(self, X : pd.DataFrame):
        prediction = pd.Series(data=0, index=range(X.shape[0]))
        num_tree = 0
        for tree in self.trees:
            num_tree += 1
            lr = self.__calculate_learning_rate(num_tree)
            prediction += tree.predict(X) * lr
        prediction += self.pred_0
        return prediction





