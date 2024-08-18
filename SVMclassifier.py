import numpy as np
import pandas as pd


class SVM():
    def __init__(self,
                 alpha : float = 1.0,
                 learning_rate: float = 0.01,
                 num_epochs : int = 500,
                 kernel : str = "rbf") -> None:

        self.w = None
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.kernel = kernel
        self.X_train = None

    def _calculate_kernel_trick(self,
                                X: pd.DataFrame) -> pd.DataFrame:
        X_t = X.copy().to_numpy()

        if self.kernel == "rbf":
            lambda_value = 1 / X.shape[1]
            X_t = np.exp(-lambda_value * np.sum((X_t[:, np.newaxis] - X_t[np.newaxis, :]) ** 2, axis=1))

        X_t = pd.DataFrame(X_t)

        return X_t

    def __activation_function(self,
                              X: pd.DataFrame) -> np.array:
        X_activate = X.copy()
        prediction = np.sign(X_activate @ self.w.T)
        return prediction

    def _calculate_margin(self,
                          X: pd.DataFrame,
                          y: pd.Series) -> np.array:
        margin = y * (X.to_numpy() @ self.w.T)
        return margin

    def _calculate_loss(self,
                        X: pd.DataFrame,
                        y: pd.Series) -> float:
        margin = self._calculate_margin(X, y)
        loss = (np.maximum(0, 1 - margin)).sum() + (self.alpha * (self.w ** 2).sum() / 2)
        return loss

    def _calculate_grad(self,
                        X: pd.DataFrame,
                        y: pd.Series) -> np.array:
        margin = self._calculate_margin(X, y)
        grad = self.alpha * self.w - (X.mul(y, axis=0).mul(margin < 1, axis=0)).sum(axis=0).values
        return grad

    def fit(self,
            X: pd.DataFrame,
            y: pd.Series) -> None:
        y[y != 1] = -1

        X_activate = X.copy()
        if not "Bias" in X.columns:
            X_activate.insert(column="Bias", value=-1, loc=0)

        if self.kernel is not None:
            self.X_train = X_activate.copy()
            X_activate = self._calculate_kernel_trick(X_activate)
        self.w = np.random.normal(loc=0, scale=0.05, size=X_activate.shape[1])

        for i in range(self.num_epochs):
            self.w -= self.learning_rate * self._calculate_grad(X_activate, y)
            loss = self._calculate_loss(X_activate, y)
            print(loss)

    def predict(self,
                X: pd.DataFrame) -> pd.Series:
        X_test = X.copy()
        if not "Bias" in X_test.columns:
            X_test.insert(column="Bias", value=-1, loc=0)
        if self.kernel == "rbf":
            X_test = pd.DataFrame(np.exp(
                -1 / X_test.shape[1] * np.sum(((X_test.to_numpy())[:, np.newaxis] - self.X_train.to_numpy()) ** 2,
                                              axis=1)))

        return self.__activation_function(X_test)
