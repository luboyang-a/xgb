import numpy as np

class MSE_Loss:

    def upd(self, y_true, y_pred):
        g = y_pred - y_true
        h = np.ones_like(y_true)
        return g, h

class Log_Loss:

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -15, 15)))
    
    def upd(self, y_true, y_pred):
        p = self._sigmoid(y_pred)
        g = p - y_true
        h = p * (1 - p)
        return g.astype(np.float64), np.maximum(h, 1e-16).astype(np.float64)