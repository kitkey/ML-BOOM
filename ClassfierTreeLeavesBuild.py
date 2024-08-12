from copy import deepcopy
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, Union, List, DefaultDict

class Node():
    def __init__(self, 
                 node_kind : str = "leaf",
                ) -> None:
        allowed_node_kinds = {"leaf", "node"}
        if node_kind not in allowed_node_kinds:
            raise TypeError(f"node_kind must be one of the allowed kinds : {allowed_node_kinds}")
        self.node_kind = node_kind
        self.l_node = None
        self.r_node = None
        self.node_dictionary = defaultdict(int)
        self.direction = None
        
        
class MyTreeClf():
    def __init__(self, 
                 max_depth : int = 5,
                 min_samples_split : int = 2,
                 max_leafs : int = 20,
                 bins : int = None,
                 criterion : str = "entropy",
                 source_size_of_model_ensemble : int = None) -> None:
        
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs if max_leafs > 1 else 2
        self.tree = Node(node_kind = "node")
        self.leafs_cnt = 1
        self.bins = bins 
        self.__using_bins = defaultdict(bool)
        self.__bin_seps = defaultdict(int)
        self.criterion = criterion
        self.fi = defaultdict(int)
        self.n_source = source_size_of_model_ensemble
        
    def _calculate_entropy(self, 
                           y : pd.Series) -> float:
        
        count_elements = np.bincount(y, minlength=2)
        probalities = count_elements / y.size 
        probalities = probalities[probalities != 0]
        entropy = - (probalities * np.log2(probalities)).sum()
        return entropy
    
    def _calculate_gini(self, 
                        y : pd.Series) -> float:
        
        count_elements = np.bincount(y, minlength=2)
        probalities = count_elements / y.size
        probalities = probalities[probalities != 0]
        gini = 1 - (probalities**2).sum()
        return gini
    
    def _calculate_importance(self, 
                              y : pd.Series,
                              y_left : pd.Series,
                              y_right : pd.Series) -> float:
        information_function = self._calculate_entropy if self.criterion == "entropy" else self._calculate_gini
        initial_information = information_function(y)
        left_information_with_n_coef = information_function(y_left) * (y_left.size / y.size)
        right_information_with_n_coef = information_function(y_right) * (y_right.size / y.size)
        importance = (y.size / self.n_source) * (initial_information - (left_information_with_n_coef + right_information_with_n_coef))
        return importance
    
    def _calculate_information_gain(self, 
                                    y : pd.Series, 
                                    y_left : pd.Series, 
                                    y_right : pd.Series
                                   ) -> float:
        information_function = self._calculate_entropy if self.criterion == "entropy" else self._calculate_gini
        initial_information = information_function(y)
        left_information_with_n_coef = information_function(y_left) * (y_left.size / y.size)
        right_information_with_n_coef = information_function(y_right) * (y_right.size / y.size)
        information_gain = initial_information - (left_information_with_n_coef + right_information_with_n_coef)
        return information_gain
    
    def __precalculate_bins(self, X : pd.DataFrame) -> defaultdict:
        bins = defaultdict(float)
        for column in X.columns:
            if X[column].unique().size - 1 >= self.bins:
                self.__using_bins[column] = True
                bins[column] = np.histogram(X[column], bins=self.bins)[1]
                bins[column] = bins[column][1:-1]
        return bins
    
    def _get_best_split(self,
                        X : pd.DataFrame, 
                        y : pd.Series) -> (str, float, float):
        
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
            X : pd.DataFrame, 
            y : pd.Series) -> None:
        
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
        self.fi = defaultdict(int, dict(zip(X.columns.tolist(), [0]*X.shape[1])))
        
        current_depth = 0
        self.__build_tree(X_train,
                          y_train, 
                          current_node = self.tree,
                          current_depth = current_depth)
                      

    def __build_tree(self, 
                     X : pd.DataFrame, 
                     y : pd.Series, 
                     current_node: 'Node',
                     current_depth : int = 0
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
     
        self.__build_tree(X = X_left, 
                          y = y_left, 
                          current_node = current_node.l, 
                          current_depth = current_depth + 1)
        
        self.__build_tree(X = X_right, 
                          y = y_right, 
                          current_node = current_node.r, 
                          current_depth = current_depth + 1)

            
    def print_tree(self, 
                   node : 'Node' = None, 
                   i = 0,
                  ) -> None:
        if i == 0:
            node = self.tree
        if node.node_kind == "node":
            print("   "*i, node.node_dictionary["column"], " > ", node.node_dictionary["separator"])
           
            self.print_tree(node.l, i+1)
            self.print_tree(node.r, i+1)
        else:
            print("   "*i, node.direction, " = ", node.node_dictionary["value"])
    
    def predict_proba(self, 
                      X : pd.DataFrame) -> float:
        
        X.reset_index(inplace=True, drop=True)
        y = self.__predict_proba_recursion(X, self.tree)
        return y
    
    def __predict_proba_recursion(self, 
                                  X : pd.DataFrame, 
                                  current_node : 'Node', 
                                  y : pd.Series = None) -> pd.Series:
        
        if current_node is self.tree:
            y = pd.Series(data=0, index=range(X.shape[0]))
        if current_node.node_kind == "node":
            separator = current_node.node_dictionary["separator"]
            column = current_node.node_dictionary["column"]
            
            X_left = X[X[column] <= separator]
            X_right = X[X[column] > separator]
            y = self.__predict_proba_recursion(X = X_left, current_node = current_node.l, y=y)
            y = self.__predict_proba_recursion(X = X_right, current_node = current_node.r, y=y)
        else:
            y[X.index] = current_node.node_dictionary["value"]
        return y
    
    def predict(self, X : pd.DataFrame) -> pd.Series:
        
        y = self.predict_proba(X)
       
        y[y > 0.5] = 1
        y[y <= 0.5] = 0
        return y.astype(int)


