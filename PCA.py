import numpy as np
import pandas as pd


class PCA():
    def __init__(self, n_components=3):
        self.n_components = n_components
    
    def __repr__(self):
        return f"PCA class: n_components={self.n_components}"

    def fit_transform(self, X : pd.DataFrame) -> pd.DataFrame:
        for column in X.columns:
            X[column] = X[column] - X[column].mean()
        cov_ftr_matrix = np.cov(X.to_numpy().T)
        eig_values, eig_vectors = np.linalg.eigh(np.cov(X.to_numpy().T))
        vecs = (eig_vectors.T)[eig_values.argsort()[::-1]][:self.n_components]
        x = X @ vecs.T
        return x