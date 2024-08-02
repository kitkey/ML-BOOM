import random
import numpy as np
import pandas as pd


class MyLogReg():
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
        return f"MyLogReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}"
    
    def _get_metric_value(self, y_target : pd.Series, y_predict : pd.Series, y_proba : pd.Series) -> float:
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
                    s += Y[:i][(Y[:i]["proba"] >  Y.loc[i]["proba"]) & (Y[:i]["class"] == 1)].shape[0]
                    s += Y[(Y["proba"] ==  Y.loc[i]["proba"]) & (Y["class"] == 1)].shape[0] / 2
                   
            s /= (y_target == 1).sum() * (y_target == 0).sum()
            value = s
        return value
    def fit(self, X : pd.DataFrame, y : pd.Series, verbose=False) -> None:
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
        
        for iteration in range(1, self.n_iter+1):
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
            
    def __calculate_loss(self, y_target : pd.Series, y_logits : pd.Series) -> float:
        loss = - 1 / y_target.size * (y_target * np.log(y_logits + 1e-15) + (1-y_target) * np.log(1-y_logits+1e-15))
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
    
    def __calculate_learning_rate(self, iteration : int) -> float:
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
            loss = self.l2_coef * (self.weights**2).sum()
        elif self.reg == "elasticnet":
            loss = self.l1_coef * np.abs(self.weights).sum() + self.l2_coef * (self.weights**2).sum()
        return loss
    
    def __calculate_logits(self, X : pd.DataFrame) -> pd.Series:
        values = X.to_numpy() @ self.weights.T
        logits = pd.Series(data=(1/(1+np.exp(-values))))
        return logits
    
    def __calculate_gradient(self, y_target : pd.Series, y_logits : pd.Series, X : pd.DataFrame) -> pd.Series: 
        grad = 1 / y_target.size * ((y_logits - y_target).to_numpy()) @ X.to_numpy()
        return grad
    
    def __update_weights(self, grad : np.array, lr : float) -> None:
        self.weights -= lr * grad
    
    def get_coef(self):
        return self.weights[1:]
    
    def predict_proba(self, X):
        probas = self.__calculate_logits(X)
        return probas
    
    def predict(self, X : pd.DataFrame) -> pd.Series:
        if "Bias" not in X.columns:
            X.insert(loc=0, column="Bias", value=1)
        probas = self.predict_proba(X)
        predictions = (probas > 0.5).astype(int)
        return predictions
    def get_best_score(self):
        return self.metric_value



