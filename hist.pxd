cdef (int, int, double) split(
    double[:] g,
    double[:] h,
    unsigned char[:, ::1] X_binned,
    int[:] indices,
    int start,
    int end,
    double l2_reg,
    int n_bin=*
)