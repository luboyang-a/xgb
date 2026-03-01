# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False

import numpy as np
cimport numpy as cnp
from libcpp.vector cimport vector
from hist cimport split

cdef struct Node:
    int feature
    int threshold
    int left
    int right
    double weight
    double gain

cdef class DecisionTree:
    cdef vector[Node] nodes
    cdef int max_depth
    cdef double l2_reg
    cdef double gamma
    cdef int n_bin
    cdef int root

    def __init__(self, max_depth=3, l2_reg=1.0, gamma=0, n_bin=256):
        self.max_depth = max_depth
        self.l2_reg = l2_reg
        self.nodes.reserve((2**(max_depth + 1)) - 1)
        self.gamma = gamma
        self.n_bin = n_bin
        self.root = -1

    cdef int build(self, double[:] g, double[:] h, unsigned char[:, ::1] X_binned, int[:] indices, int start, int end, int depth):
        cdef double sum_g = 0, sum_h = 0
        cdef int i, idx
        cdef int best_f, best_b
        cdef double best_g

        for i in range(start, end):
            sum_g += g[indices[i]]
            sum_h += h[indices[i]]
        cdef double w = -sum_g / (sum_h + self.l2_reg)

        cdef Node node
        node.feature = -1
        node.threshold = -1
        node.left = -1
        node.right = -1
        node.weight = 0.0
        node.gain = 0.0
        idx = self.nodes.size()
        self.nodes.push_back(node)

        if depth >= self.max_depth or (end - start) < 2 or sum_h <= 1e-6:
            self.nodes[idx].weight = w
            return idx

        best_f, best_b, best_g = split(g, h, X_binned, indices, start, end, self.l2_reg, self.n_bin)
        if best_f == -1 or best_g <= self.gamma:
            self.nodes[idx].weight = w
            return idx

        cdef int l = start, r = end - 1
        cdef int tmp
        while l <= r:
            if X_binned[indices[l], best_f] <= best_b:
                l += 1
            else:
                tmp = indices[l]
                indices[l] = indices[r]
                indices[r] = tmp
                r -= 1
        
        self.nodes[idx].feature = best_f
        self.nodes[idx].threshold = best_b
        self.nodes[idx].gain = best_g
        cdef int left_idx = self.build(g, h, X_binned, indices, start, l, depth + 1)
        cdef int right_idx = self.build(g, h, X_binned, indices, l, end, depth + 1)
        self.nodes[idx].left = left_idx
        self.nodes[idx].right = right_idx
        return idx

    def fit(self, unsigned char[:, ::1] X_binned, double[:] g, double[:] h):
        cdef int n_samples = X_binned.shape[0]
        cdef int[:] indices = np.arange(n_samples, dtype=np.int32)
        self.nodes.clear()
        self.root = self.build(g, h, X_binned, indices, 0, n_samples, 0)

    def predict(self, unsigned char[:, ::1] X_binned):
        cdef int n_samples = X_binned.shape[0]
        cdef double[:] preds = np.zeros(n_samples, dtype=np.float64)
        cdef int i, curr
        cdef Node node

        for i in range(n_samples):
            curr = self.root
            while True:
                node = self.nodes[curr]
                if node.left == -1:
                    preds[i] = node.weight
                    break
                if X_binned[i, node.feature] <= node.threshold:
                    curr = node.left
                else:
                    curr = node.right
        return np.asarray(preds)