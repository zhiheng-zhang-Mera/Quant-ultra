"""Auditable quantitative analytics and constrained allocation formulas."""
from __future__ import annotations
import numpy as np
import pandas as pd
from Main.fast_math import log_returns, downside_deviation
from Main.trading_costs import explicit_order_fees, round_trip_friction_rate


def nearest_psd(cov: np.ndarray, floor: float = 1e-8) -> np.ndarray:
    a = (np.asarray(cov, float) + np.asarray(cov, float).T) / 2
    vals, vecs = np.linalg.eigh(a)
    return (vecs * np.maximum(vals, floor)) @ vecs.T


def risk_parity_weights(cov: np.ndarray, iterations: int = 500, tol: float = 1e-10) -> np.ndarray:
    cov = nearest_psd(cov)
    n = cov.shape[0]
    w = np.full(n, 1 / n)
    for _ in range(iterations):
        sigma = float(np.sqrt(max(w @ cov @ w, 1e-16)))
        rc = w * (cov @ w) / sigma
        new = w * np.sqrt((sigma / n) / np.maximum(rc, 1e-16))
        new /= new.sum()
        if np.max(np.abs(new - w)) < tol:
            return new
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
    r = log_returns(prices.dropna().to_numpy(dtype=float))
    n = len(r)
    if not n:
        return {"observations": 0}
    ann_ret = float(np.exp(r.mean() * annualization) - 1)
    ann_vol = float(r.std(ddof=1) * np.sqrt(annualization)) if n > 1 else 0.0
    wealth = np.exp(np.cumsum(r))
    mdd = float(np.min(wealth / np.maximum.accumulate(wealth) - 1))
    dd = downside_deviation(r, risk_free / annualization) * np.sqrt(annualization)
    var95 = float(np.quantile(r, 0.05))
    cvar95 = float(r[r <= var95].mean())
    return {
        "observations": n, "annual_return": ann_ret, "annual_volatility": ann_vol,
        "sharpe": (ann_ret - risk_free) / ann_vol if ann_vol else 0.0,
        "sortino": (ann_ret - risk_free) / dd if dd else 0.0,
        "max_drawdown": mdd, "calmar": ann_ret / abs(mdd) if mdd else 0.0,
        "var_95_daily": var95, "cvar_95_daily": cvar95,
    }


def technical_snapshot(frame: pd.DataFrame) -> dict:
    """Return transparent trend, volatility and price-level inputs."""
    if len(frame) < 60:
        raise ValueError("至少需要 60 个交易日数据")
    close = pd.to_numeric(frame["close"], errors="coerce").dropna()
    high = pd.to_numeric(frame["high"], errors="coerce")
    low = pd.to_numeric(frame["low"], errors="coerce")
    prev = pd.to_numeric(frame["close"], errors="coerce").shift(1)
    tr = pd.concat([(high - low).abs(), (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    atr14 = float(tr.tail(14).mean())
    last = float(close.iloc[-1])
    ma20, ma60 = float(close.tail(20).mean()), float(close.tail(60).mean())
    momentum = float(last / close.iloc[-61] - 1) if len(close) >= 61 else float(last / close.iloc[0] - 1)
    return {"last_price": last, "ma20": ma20, "ma60": ma60, "atr14": atr14, "atr_pct": atr14 / last, "momentum_60d": momentum}


def recommendation(frame: pd.DataFrame, model_weight: float | None = None, max_weight: float = 0.08, asset_type: str = "stock", cost_config: dict | None = None) -> dict:
    """Candidate output: entry range, allocation and volatility-adaptive take profit."""
    snap = technical_snapshot(frame)
    perf = metrics(frame["close"])
    score = sum((
        1 if snap["last_price"] > snap["ma20"] else -1,
        1 if snap["ma20"] > snap["ma60"] else -1,
        1 if snap["momentum_60d"] > 0 else -1,
        1 if perf.get("sharpe", 0) > 0 else -1,
    ))
    anchor = min(snap["last_price"], snap["ma20"])
    entry_low = max(0.01, anchor - 1.00 * snap["atr14"])
    entry_high = min(snap["last_price"], snap["ma20"] + 0.15 * snap["atr14"])
    if entry_high < entry_low:
        entry_high = entry_low
    base_weight = float(model_weight) if model_weight is not None and np.isfinite(model_weight) else 0.05
    confidence = float(np.clip((score + 4) / 8, 0.25, 1.0))
    volatility_scale = float(np.clip(0.15 / max(perf.get("annual_volatility", 0.15), 0.05), 0.35, 1.0))
    stop_loss_pct = float(np.clip(max(1.25 * snap["atr_pct"], 0.035), 0.035, 0.08))
    loss_budget_weight = float(np.clip(0.0075 / stop_loss_pct, 0.0, max_weight))
    suggested_weight = float(np.clip(min(base_weight * confidence * volatility_scale, loss_budget_weight), 0.0, max_weight))
    assumed_notional = max(10000.0, suggested_weight * 1000000.0)
    friction = round_trip_friction_rate(assumed_notional, asset_type=asset_type, holding_days=20, config=cost_config)
    minimum_net_profit = 0.02
    take_profit_pct = float(np.clip(max(1.40 * snap["atr_pct"], friction + minimum_net_profit), 0.03, 0.12))
    qualified = score >= 2 and suggested_weight > 0 and entry_high >= entry_low
    return {
        **perf, **snap, "score": score, "qualified": qualified,
        "entry_price_low": round(entry_low, 4), "entry_price_high": round(entry_high, 4),
        "suggested_weight": suggested_weight, "take_profit_pct": take_profit_pct,
        "estimated_round_trip_cost_pct": friction, "minimum_net_profit_pct": minimum_net_profit,
        "net_take_profit_pct": take_profit_pct - friction, "stop_loss_pct": stop_loss_pct,
        "reward_risk_ratio": (take_profit_pct - friction) / stop_loss_pct,
        "formula_evidence": {
            "entry": "min(last, MA20)-1.0*ATR14 to min(last, MA20+0.15*ATR14); no chasing above last",
            "allocation": "min(model confidence weight * volatility scale, 0.75% loss budget / stop distance, 8% cap)",
            "take_profit": "clip(max(1.4*ATR14/last, round-trip friction + 2% net target), 3%, 12%)",
        },
    }


def holding_advice(frame: pd.DataFrame, total_capital: float, quantity: int, average_cost: float, model_weight: float | None = None, asset_type: str = "stock", cost_config: dict | None = None) -> dict:
    if total_capital <= 0 or quantity < 0 or average_cost <= 0:
        raise ValueError("总资金必须大于0、数量不得为负、平均成本必须大于0")
    rec = recommendation(frame, model_weight=model_weight, asset_type=asset_type, cost_config=cost_config)
    last = rec["last_price"]
    market_value = quantity * last
    current_weight = market_value / total_capital
    target_value = total_capital * rec["suggested_weight"]
    lot_size = 100
    target_quantity = max(0, int(target_value / last / lot_size) * lot_size)
    delta_quantity = target_quantity - quantity
    pnl_amount = quantity * (last - average_cost)
    pnl_pct = last / average_cost - 1
    take_profit_price = average_cost * (1 + rec["take_profit_pct"])
    sell_fees = explicit_order_fees(quantity * take_profit_price, "sell", asset_type=asset_type, config=cost_config)
    estimated_net_take_profit = quantity * (take_profit_price - average_cost) - sell_fees["total"]
    stop_reference = max(rec["entry_price_low"] - rec["atr14"], average_cost * 0.92)
    if last >= take_profit_price:
        action, reason = "分批止盈", "当前价达到基于持仓成本计算的止盈价"
    elif not rec["qualified"] and delta_quantity < 0:
        action, reason = "减仓", "趋势/风险评分未通过且实际仓位高于目标"
    elif delta_quantity >= lot_size and rec["entry_price_low"] <= last <= rec["entry_price_high"]:
        action, reason = "按批次买入", "当前价位于理想入仓区间且目标数量高于持仓"
    elif delta_quantity <= -lot_size:
        action, reason = "减仓至目标", "实际仓位高于模型建议仓位"
    else:
        action, reason = "持有观察", "尚未触发买入、减仓或止盈条件"
    return {
        **rec, "total_capital": total_capital, "quantity": quantity,
        "average_cost": average_cost, "market_value": market_value,
        "current_weight": current_weight, "pnl_amount": pnl_amount, "pnl_pct": pnl_pct,
        "target_value": target_value, "target_quantity": target_quantity,
        "delta_quantity": delta_quantity, "take_profit_price": take_profit_price,
        "estimated_sell_fees_at_take_profit": sell_fees["total"],
        "estimated_net_take_profit_amount": estimated_net_take_profit,
        "stop_reference_price": stop_reference, "action": action, "action_reason": reason,
    }
