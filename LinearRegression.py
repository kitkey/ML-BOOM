from typing import List
class LinearRegression():
    def __init__(self, n_iter=100, learning_rate=0.1, metric=None, reg=None, l1_coef=0, l2_coef=0, sgd_sample=None, random_state=42):
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
        string = f"MyLineReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}"
        return string
    
    def _get_metric_value(self, y_target: pd.Series, y_pred : pd.Series) -> float:
        n = y_target.size
        metric_value = 0
        if self.metric == "mae":
            metric_value = 1 / n * np.abs((y_target - y_pred)).sum()
        elif self.metric == "rmse":
            metric_value = np.sqrt(1 / n * ((y_target - y_pred)**2).sum())
        elif self.metric == "r2":
            metric_value = 1 - ((y_target - y_pred)**2).sum() / ((y_target - y_target.mean())**2).sum()
        elif self.metric == "mape":
            metric_value = 100 / n * np.abs((y_target - y_pred) / y_target).sum()
        elif self.metric == "mse":
            metric_value = 1 / n * ((y_target-y_pred)**2).sum()
        return metric_value
    
    def fit(self, X : pd.DataFrame, y: pd.Series, verbose: int) -> None:
        random.seed(self.random_state)
        X.reset_index(inplace=True, drop=True)
        y.reset_index(inplace=True, drop=True)
        X.insert(loc=0, column="Bias", value=1)
        self.weights = np.ones(X.columns.size)
        error_reg, grad_reg = 0, 0
        if type(self.sgd_sample) is float:
            sgd_sample = round(self.sgd_sample * X.shape[0])
        elif self.sgd_sample is not None:
            sgd_sample = self.sgd_sample
        for iteration in range(1, self.n_iter+1):
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
            
    def mse_gradient(self, y_target : pd.Series, y_pred : pd.Series, X : pd.DataFrame) -> np.array:
        X = X.to_numpy()
        grad = 2 / y_pred.size * (y_pred - y_target) @ X 
        return grad
    
    def mse_error(self, y_target : pd.Series, y_pred : pd.Series) -> float:
        error = 1/y_target.size * ((y_target-y_pred)**2).sum()
        return error
    
    def __update_weights(self, grad : pd.Series, lr : float) -> None:
        self.weights -= lr * grad
    
    def get_coef(self) -> np.array:
        return self.weights[1:]
    
    def predict(self, X: pd.DataFrame) -> pd.Series:
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
            reg_loss = self.l2_coef * ((self.weights)**2).sum()
            reg_grad = 2 * self.l2_coef * self.weights
        elif self.reg == "elasticnet":
            reg_loss = self.l1_coef * np.abs(self.weights).sum() + self.l2_coef * ((self.weights)**2).sum()
            reg_grad = self.l1_coef * np.sign(self.weights) + 2 * self.l2_coef * self.weights
        return reg_loss, reg_grad

    def __calculate_learning_rate(self, iteration) -> float:
        if callable(self.learning_rate):
            lr = self.learning_rate(iteration)
        else:
            lr = self.learning_rate
        return lr