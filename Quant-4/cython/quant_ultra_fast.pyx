# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
import numpy as np
cimport numpy as cnp
from libc.math cimport log, sqrt

cpdef cnp.ndarray[cnp.double_t, ndim=1] log_returns(cnp.ndarray[cnp.double_t, ndim=1] prices):
    cdef Py_ssize_t i, n = prices.shape[0]
    cdef cnp.ndarray[cnp.double_t, ndim=1] out = np.empty(max(n - 1, 0), dtype=np.float64)
    for i in range(1, n):
        out[i - 1] = log(prices[i] / prices[i - 1]) if prices[i - 1] > 0 and prices[i] > 0 else 0.0
    return out

cpdef double downside_deviation(cnp.ndarray[cnp.double_t, ndim=1] returns, double mar=0.0):
    cdef Py_ssize_t i, n = returns.shape[0]
    cdef double d, total = 0.0
    if n == 0: return 0.0
    for i in range(n):
        d = returns[i] - mar
        if d < 0: total += d * d
    return sqrt(total / n)

