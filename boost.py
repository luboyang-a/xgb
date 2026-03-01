import numpy as np
from losses import Log_Loss
from tree import DecisionTree


class Boost:

    def __init__(self, loss, n_estimators=10, max_depth=3, learning_rate=0.1, l2_reg=1.0, n_bin=256, base_score=0, gamma=0):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.lr = learning_rate
        self.l2_reg = l2_reg
        self.n_bin = n_bin
        self.base_score = base_score
        self.trees = []
        self.loss = loss
        self.gamma = gamma

    def fit(self, X_binned, y):
        X_binned = np.asfortranarray(X_binned, dtype=np.uint8)
        y_pred = np.full_like(y, self.base_score, dtype=np.float64)
        for i in range(self.n_estimators):
            g, h = self.loss.upd(y, y_pred)
            tree = DecisionTree(
                max_depth=self.max_depth,
                l2_reg=self.l2_reg,
                gamma=self.gamma,
                n_bin = self.n_bin
            )
            tree.fit(X_binned, g, h)
            t = tree.predict(X_binned)
            y_pred += self.lr * t
            self.trees.append(tree)

    def predict(self, X_binned):
        X_binned = np.ascontiguousarray(X_binned, dtype=np.uint8)
        y_pred = np.full(X_binned.shape[0], self.base_score, dtype=np.float64)
        for tree in self.trees:
            y_pred += self.lr * tree.predict(X_binned)
        return y_pred
    
class OneVsRestBoost:

    def __init__(self, n_classes, **params):
        self.n_classes = n_classes
        self.params = params
        self.models = []

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -15, 15)))

    def fit(self, X_binned, y):
        for i in range(self.n_classes):
            y_bi = np.where(y == i, 1, 0).astype(np.float64)
            model = Boost(loss=Log_Loss(), **self.params)
            model.fit(X_binned, y_bi)
            self.models.append(model)

    def predict_proba(self, X_binned):
        probs = np.array([self._sigmoid(m.predict(X_binned)) for m in self.models])
        return probs.T
    
    def predict(self, X_binned):
        probs = self.predict_proba(X_binned)
        return np.argmax(probs, axis=1)
    
class OneVsOneBoost:

    def __init__(self, n_classes, **params):
        self.n_classes = n_classes
        self.params = params
        self.models = []

    def fit(self, X_binned, y):
        for i in range(self.n_classes):
            for j in range(i + 1, self.n_classes):
                mask = (y == i) | (y == j)
                X_pairs, y_pairs = X_binned[mask], y[mask]
                y_bi = np.where(y_pairs == i, 1.0, 0.0).astype(np.float64)
                model = Boost(loss=Log_Loss(), **self.params)
                model.fit(X_pairs, y_bi)
                self.models.append(((i, j), model))
    
    def predict(self, X_binned):
        n_samples = X_binned.shape[0]
        votes = np.zeros((n_samples, self.n_classes))
        for (i, j), model in self.models:
            scores = model.predict(X_binned)
            votes[scores > 0, i] += 1
            votes[scores <= 0, j] += 1
        return np.argmax(votes, axis=1)