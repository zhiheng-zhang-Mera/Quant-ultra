"""Cython hot paths with a numerically equivalent NumPy fallback."""
import numpy as np

try:
    import quant_ultra_fast as _cython
except ImportError:
    _cython = None

BACKEND = "cython" if _cython is not None else "numpy"
CAPABILITIES = {name: bool(_cython is not None and hasattr(_cython, name)) for name in (
    "log_returns", "downside_deviation", "rolling_sum", "momentum", "garman_klass_volatility", "capped_simplex_projection"
)}

def log_returns(prices):
    p = np.asarray(prices, dtype=np.float64)
    if CAPABILITIES["log_returns"]: return _cython.log_returns(p)
    return np.diff(np.log(p)) if len(p) > 1 else np.array([], dtype=np.float64)

def downside_deviation(returns, mar=0.0):
    r = np.asarray(returns, dtype=np.float64)
    if CAPABILITIES["downside_deviation"]: return float(_cython.downside_deviation(r, mar))
    centered = r - mar
    return float(np.sqrt(np.mean(np.minimum(centered, 0.0) ** 2))) if len(r) else 0.0

def rolling_sum(values, window):
    values = np.asarray(values, dtype=np.float64)
    if CAPABILITIES["rolling_sum"]: return _cython.rolling_sum(values, int(window))
    out = np.full(len(values), np.nan)
    if len(values) >= window: out[window-1:] = np.convolve(values, np.ones(window), mode="valid")
    return out

def momentum(prices, window):
    prices = np.asarray(prices, dtype=np.float64)
    if CAPABILITIES["momentum"]: return _cython.momentum(prices, int(window))
    out = np.zeros(len(prices)); valid = (prices[window:] > 0) & (prices[:-window] > 0)
    out[window:][valid] = np.log(prices[window:][valid] / prices[:-window][valid])
    return out

def garman_klass_volatility(open_, high, low, close):
    arrays = [np.asarray(x, dtype=np.float64) for x in (open_, high, low, close)]
    if CAPABILITIES["garman_klass_volatility"]: return _cython.garman_klass_volatility(*arrays)
    o, h, l, c = arrays; valid = (o > 0) & (h > 0) & (l > 0) & (c > 0); out = np.zeros(len(c))
    variance = .5 * np.log(h[valid] / l[valid])**2 - (2*np.log(2)-1) * np.log(c[valid] / o[valid])**2
    out[valid] = np.sqrt(np.clip(variance, 0, None)); return out

def capped_simplex_projection(values, caps, budget):
    values = np.asarray(values, dtype=np.float64); caps = np.asarray(caps, dtype=np.float64)
    if CAPABILITIES["capped_simplex_projection"]: return _cython.capped_simplex_projection(values, caps, float(budget))
    clipped = np.minimum(np.maximum(values, 0), np.maximum(caps, 0))
    if budget <= 0: return np.zeros_like(values)
    if clipped.sum() <= budget: return clipped
    low, high = float(np.min(values-caps)), float(np.max(values))
    for _ in range(80):
        middle = (low+high)/2; clipped = np.minimum(np.maximum(values-middle, 0), np.maximum(caps, 0))
        if clipped.sum() > budget: low = middle
        else: high = middle
    return clipped

