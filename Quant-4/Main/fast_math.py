"""Cython hot paths with a numerically equivalent NumPy fallback."""
import numpy as np

try:
    from quant_ultra_fast import log_returns, downside_deviation
    BACKEND = "cython"
except ImportError:
    BACKEND = "numpy"
    def log_returns(prices):
        p = np.asarray(prices, dtype=np.float64)
        return np.diff(np.log(p)) if len(p) > 1 else np.array([], dtype=np.float64)
    def downside_deviation(returns, mar=0.0):
        r = np.asarray(returns, dtype=np.float64) - mar
        return float(np.sqrt(np.mean(np.minimum(r, 0.0) ** 2))) if len(r) else 0.0

