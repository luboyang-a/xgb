import numpy as np

class PCA:

    def __init__(self, n_components):
        self.k = n_components
        self.comps = None
        self.mean = None
        self.eigenvalues = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        X_cen = X - self.mean
        cov_mat = np.cov(X_cen, rowvar=False)
        eigen_values, eigen_vectors = np.linalg.eigh(cov_mat)
        idx = np.argsort(eigen_values)[::-1]
        self.eigenvalues = eigen_values[idx]
        sorted_vectors = eigen_vectors[:, idx]
        self.comps = sorted_vectors[:, :self.k]
    
    def transform(self, X):
        X_cen = X - self.mean
        return np.dot(X_cen, self.comps)
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    
    @property
    def explained_variance_ratio(self):
        return self.eigenvalues[:self.k] / np.sum(self.eigenvalues)
    
    def inverse_transform(self, X_red):
        return np.dot(X_red, self.comps.T) + self.mean
  
class PCA_SVD:

    def __init__(self, n_components):
        self.k = n_components
        self.comps = None
        self.mean = None
        self.singular_values = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        X_cen = X - self.mean
        U, S, Vt = np.linalg.svd(X_cen, full_matrices=False)
        V = Vt.T
        self.singular_values = S
        self.comps = V[:, :self.k]

    def transform(self, X):
        X_cen = X - self.mean
        return np.dot(X_cen, self.comps)
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    
    @property
    def explained_variance_ratio(self):
        exp_var = self.singular_values**2
        return exp_var[:self.k] / np.sum(exp_var)
    
    def inverse_transform(self, X_red):
        return np.dot(X_red, self.comps.T) + self.mean