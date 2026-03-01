# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import numpy as np
cimport numpy as cnp
from libc.string cimport memset

cdef (int, int, double) split(
    double[:] g,
    double[:] h,
    unsigned char[:, ::1] X_binned,
    int[:] indices,
    int start,
    int end,
    double l2_reg,
    int n_bin=256
):
    cdef int n_features = X_binned.shape[1]
    cdef double best_gain = 0.0
    cdef int best_feature = -1
    cdef int best_bin = -1

    cdef double total_g = 0.0, total_h = 0.0
    cdef int f, i, idx, b, bin_idx

    for i in range(start, end):
        idx = indices[i]
        total_g += g[idx]
        total_h += h[idx]

    cdef double bin_g[256]
    cdef double bin_h[256]

    cdef double g_left, h_left, g_right, h_right, gain
    cdef double gain_const = 0.5 * (total_g**2 / (total_h + l2_reg))

    for f in range(n_features):
        memset(bin_g, 0, 256 * sizeof(double))
        memset(bin_h, 0, 256 * sizeof(double))

        for i in range(start, end):
            idx = indices[i]
            bin_idx = X_binned[idx, f]
            bin_g[bin_idx] += g[idx]
            bin_h[bin_idx] += h[idx]

        g_left = 0.0
        h_left = 0.0
        for b in range(n_bin - 1):
            g_left += bin_g[b]
            h_left += bin_h[b]
            if h_left <= 0: continue
            g_right = total_g - g_left
            h_right = total_h - h_left
            if h_right <= 0: continue
            gain = 0.5 * ((g_left**2 / (h_left + l2_reg)) + (g_right**2 / (h_right + l2_reg))) - gain_const
            if gain > best_gain:
                best_gain = gain
                best_feature = f
                best_bin = b
    return best_feature, best_bin, best_gain