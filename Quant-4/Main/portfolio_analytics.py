"""Auditable quantitative analytics and constrained allocation formulas."""
from __future__ import annotations
import numpy as np
import pandas as pd
from Main.fast_math import log_returns, downside_deviation

def nearest_psd(cov: np.ndarray, floor: float = 1e-8) -> np.ndarray:
    a = (np.asarray(cov, float) + np.asarray(cov, float).T) / 2
    vals, vecs = np.linalg.eigh(a)
    return (vecs * np.maximum(vals, floor)) @ vecs.T

def risk_parity_weights(cov: np.ndarray, iterations: int = 500, tol: float = 1e-10) -> np.ndarray:
    cov = nearest_psd(cov); n = cov.shape[0]; w = np.full(n, 1 / n)
    for _ in range(iterations):
        sigma = float(np.sqrt(max(w @ cov @ w, 1e-16)))
        rc = w * (cov @ w) / sigma
        target = sigma / n
        new = w * np.sqrt(target / np.maximum(rc, 1e-16)); new /= new.sum()
        if np.max(np.abs(new - w)) < tol: return new
        w = new
    return w

def black_litterman(pi, cov, P, Q, tau=0.05, omega=None):
    pi, cov, P, Q = map(lambda x: np.asarray(x, float), (pi, cov, P, Q))
    omega = np.diag(np.diag(P @ (tau * cov) @ P.T)) if omega is None else np.asarray(omega, float)
    precision = np.linalg.pinv(tau * cov)
    post_cov = np.linalg.pinv(precision + P.T @ np.linalg.pinv(omega) @ P)
    post_mean = post_cov @ (precision @ pi + P.T @ np.linalg.pinv(omega) @ Q)
    return post_mean, post_cov + cov

def metrics(prices: pd.Series, annualization=252, risk_free=0.02) -> dict:
    r = log_returns(prices.dropna().to_numpy(dtype=float)); n = len(r)
    if not n: return {"observations": 0}
    ann_ret = float(np.exp(r.mean() * annualization) - 1); ann_vol = float(r.std(ddof=1) * np.sqrt(annualization)) if n > 1 else 0.0
    wealth = np.exp(np.cumsum(r)); peak = np.maximum.accumulate(wealth); mdd = float(np.min(wealth / peak - 1))
    dd = downside_deviation(r, risk_free / annualization) * np.sqrt(annualization)
    var95 = float(np.quantile(r, .05)); cvar95 = float(r[r <= var95].mean())
    return {"observations": n, "annual_return": ann_ret, "annual_volatility": ann_vol, "sharpe": (ann_ret-risk_free)/ann_vol if ann_vol else 0.0, "sortino": (ann_ret-risk_free)/dd if dd else 0.0, "max_drawdown": mdd, "calmar": ann_ret/abs(mdd) if mdd else 0.0, "var_95_daily": var95, "cvar_95_daily": cvar95}

def trade_signal(prices: pd.Series, current_weight: float, cost: float, last_price: float) -> dict:
    m = metrics(prices); ma20 = float(prices.tail(20).mean()); ma60 = float(prices.tail(60).mean())
    momentum = float(prices.iloc[-1] / prices.iloc[max(0, len(prices)-61)] - 1)
    score = (1 if last_price > ma20 else -1) + (1 if ma20 > ma60 else -1) + (1 if momentum > 0 else -1) + (1 if m.get("sharpe", 0) > 0 else -1)
    target = 0.0 if score <= -2 else min(0.10, max(0.0, current_weight + (0.02 if score >= 2 else 0.0)))
    action = "减仓/退出" if target < current_weight else ("增持" if target > current_weight else "持有")
    return {**m, "ma20": ma20, "ma60": ma60, "momentum_60d": momentum, "unrealized_return": last_price / cost - 1 if cost > 0 else None, "score": score, "current_weight": current_weight, "target_weight": target, "action": action}

