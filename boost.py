import numpy as np
from hist import split
from losses import Log_Loss

class Node:

    def __init__(self, is_leaf=False, weight=None, feature=None, bin_idx=None, left=None, right=None):
        self.is_leaf = is_leaf
        self.weight = weight
        self.feature = feature
        self.bin_idx = bin_idx
        self.left = left
        self.right = right

class DecisionTree:

    def __init__(self, max_depth=3, l2_reg=1.0, n_bin=256, gamma=0):
        self.max_depth = max_depth
        self.l2_reg = l2_reg
        self.n_bin = n_bin
        self.root = None
        self.gamma = gamma

    def _build(self, X_binned, g, h, depth=0):
        sum_g = np.sum(g)
        sum_h = np.sum(h)
        w = -sum_g / (sum_h + self.l2_reg)
        if depth >= self.max_depth or len(g) < 2 or sum_h < 1e-6:
            return Node(is_leaf=True, weight=w)
        best_f, best_b, best_g = split(g, h, X_binned, self.l2_reg, self.n_bin)
        if best_f == -1 or best_g <= self.gamma:
            return Node(is_leaf=True, weight=w)
        mask = X_binned[:, best_f] <= best_b
        left_node = self._build(X_binned[mask], g[mask], h[mask], depth + 1)
        right_node = self._build(X_binned[~mask], g[~mask], h[~mask], depth + 1)
        return Node(
            feature=best_f,
            bin_idx=best_b,
            left=left_node,
            right=right_node
        )
    
    def fit(self, X_binned, g, h):
        self.root = self._build(X_binned, g, h, depth=0)

    def _traverse(self, x, node=None):
        curr = node if node else self.root
        while not curr.is_leaf:
            if x[curr.feature] <= curr.bin_idx:
                curr = curr.left
            else:
                curr = curr.right
        return curr.weight
        
    def predict(self, X_binned):
        return np.array([self._traverse(x) for x in X_binned])
    
    def print_tree(self, node=None, depth=0):
        if node is None: node = self.root
        indent = "|   " * depth
        if node.is_leaf:
            print(f"{indent}|--- Leaf Weight: {node.weight:.2f}")
            return
        print(f"{indent}|--- Feature {node.feature} <= {node.bin_idx}")
        self.print_tree(node.left, depth + 1)
        print(f"{indent}|--- Feature {node.feature} > {node.bin_idx}")
        self.print_tree(node.right, depth + 1)

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
        y_pred = np.full_like(y, self.base_score, dtype=np.float64)
        for i in range(self.n_estimators):
            g, h = self.loss.upd(y, y_pred)
            tree = DecisionTree(
                max_depth=self.max_depth,
                l2_reg=self.l2_reg,
                n_bin=self.n_bin,
                gamma=self.gamma
            )
            tree.fit(X_binned, g, h)
            t = tree.predict(X_binned)
            y_pred += self.lr * t
            self.trees.append(tree)

    def predict(self, X_binned):
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