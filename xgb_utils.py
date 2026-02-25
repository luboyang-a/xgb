import numpy as np

class utils:

    @staticmethod
    def equal_width(X, n_bin=256):
        X = X.astype(np.float64)
        n_samples, n_features = X.shape[0], X.shape[1]
        X_binned = np.zeros(X.shape, dtype=np.uint8)
        bin_info = []
        for f in range(n_features):
            maxv = np.max(X[:, f])
            minv = np.min(X[:, f])
            step = (maxv - minv + 1e-9) / n_bin if maxv > minv else 1.0
            indices = ((X[:, f] - minv) / step).astype(np.int32)
            indices = np.clip(indices, a_min=0, a_max=n_bin - 1)
            X_binned[:, f] = indices.astype(np.uint8)
            bin_info.append({'min': minv, 'step': step})
        return X_binned, bin_info
    
    @staticmethod
    def transform_equal_width(X, bin_info, n_bin=256):
        X = X.astype(np.float64)
        X_binned = np.zeros(X.shape, dtype=np.uint8)
        for f, info in enumerate(bin_info):
            indices = ((X[:, f] - info['min']) / info['step']).astype(np.int32)
            indices = np.clip(indices, a_min=0, a_max=n_bin - 1)
            X_binned[:, f] = indices.astype(np.uint8)
        return X_binned
    
    @staticmethod
    def equal_frequency(X, n_bin=256):
        X = X.astype(np.float64)
        n_samples, n_features = X.shape[0], X.shape[1]
        X_binned = np.zeros(X.shape, dtype=np.uint8)
        bin_info = []
        for f in range(n_features):
            percent = np.linspace(0, 100, n_bin + 1)
            thresholds = np.percentile(X[:, f], percent[1:-1])
            thresholds = np.unique(thresholds)
            indices = np.searchsorted(thresholds, X[:, f])
            X_binned[:, f] = indices.astype(np.uint8)
            bin_info.append(thresholds)
        return X_binned, bin_info
    
    @staticmethod
    def transform_equal_frequency(X, bin_info, n_bin=256):
        X = X.astype(np.float64)
        X_binned = np.zeros(X.shape, dtype=np.uint8)
        for f, thresholds in enumerate(bin_info):
            indices = np.searchsorted(thresholds, X[:, f])
            X_binned[:, f] = indices.astype(np.uint8)
        return X_binned