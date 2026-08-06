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

cpdef cnp.ndarray[cnp.double_t, ndim=1] rolling_sum(cnp.ndarray[cnp.double_t, ndim=1] values, int window):
    cdef Py_ssize_t i, n = values.shape[0]
    cdef double running = 0.0
    cdef cnp.ndarray[cnp.double_t, ndim=1] out = np.full(n, np.nan, dtype=np.float64)
    if window <= 0: raise ValueError("window must be positive")
    for i in range(n):
        running += values[i]
        if i >= window: running -= values[i-window]
        if i >= window-1: out[i] = running
    return out

cpdef cnp.ndarray[cnp.double_t, ndim=1] momentum(cnp.ndarray[cnp.double_t, ndim=1] prices, int window):
    cdef Py_ssize_t i, n = prices.shape[0]
    cdef cnp.ndarray[cnp.double_t, ndim=1] out = np.zeros(n, dtype=np.float64)
    if window <= 0: raise ValueError("window must be positive")
    for i in range(window, n):
        if prices[i-window] > 0 and prices[i] > 0:
            out[i] = log(prices[i] / prices[i-window])
    return out

cpdef cnp.ndarray[cnp.double_t, ndim=1] garman_klass_volatility(
    cnp.ndarray[cnp.double_t, ndim=1] open_, cnp.ndarray[cnp.double_t, ndim=1] high,
    cnp.ndarray[cnp.double_t, ndim=1] low, cnp.ndarray[cnp.double_t, ndim=1] close):
    cdef Py_ssize_t i, n = close.shape[0]
    cdef double hl, co, variance
    cdef cnp.ndarray[cnp.double_t, ndim=1] out = np.zeros(n, dtype=np.float64)
    for i in range(n):
        if open_[i] > 0 and high[i] > 0 and low[i] > 0 and close[i] > 0:
            hl = log(high[i] / low[i]); co = log(close[i] / open_[i])
            variance = 0.5 * hl * hl - (2.0 * log(2.0) - 1.0) * co * co
            out[i] = sqrt(variance) if variance > 0 else 0.0
    return out

cpdef cnp.ndarray[cnp.double_t, ndim=1] capped_simplex_projection(
    cnp.ndarray[cnp.double_t, ndim=1] values, cnp.ndarray[cnp.double_t, ndim=1] caps, double budget):
    cdef int iteration
    cdef double low = -1.0, high = 1.0, middle
    cdef cnp.ndarray[cnp.double_t, ndim=1] clipped
    if budget <= 0: return np.zeros_like(values)
    clipped = np.minimum(np.maximum(values, 0.0), np.maximum(caps, 0.0))
    if clipped.sum() <= budget: return clipped
    low = float(np.min(values - caps)); high = float(np.max(values))
    for iteration in range(80):
        middle = (low + high) / 2.0
        clipped = np.minimum(np.maximum(values - middle, 0.0), np.maximum(caps, 0.0))
        if clipped.sum() > budget: low = middle
        else: high = middle
    return clipped

