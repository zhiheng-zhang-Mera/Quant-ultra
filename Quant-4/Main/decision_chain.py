"""Regime-gated nonlinear expert chain for selection, entry, holding and exit."""
from __future__ import annotations
import math
import numpy as np
import pandas as pd

STAGE_METHODS = {
    "selection": ("trend", "momentum", "mean_reversion", "risk_adjusted", "drawdown_resilience", "volatility_breakout", "liquidity_confirmation", "sentiment_confirmation"),
    "entry": ("atr_pullback", "ma_retest", "breakout_confirmation", "volatility_ladder", "liquidity_aware", "value_zone", "momentum_continuation", "risk_budget"),
    "holding": ("trend_follow", "trailing_stop", "volatility_control", "drawdown_guard", "signal_persistence", "liquidity_monitor", "time_stop", "profit_protection"),
    "take_profit": ("atr_target", "volatility_band", "trailing_exit", "risk_reward", "resistance", "time_decay", "liquidity_exit", "partial_ladder"),
}

TRANSITIONS = {
    "trend": {"breakout_confirmation", "momentum_continuation", "trend_follow"},
    "momentum": {"momentum_continuation", "volatility_ladder", "trend_follow"},
    "mean_reversion": {"atr_pullback", "ma_retest", "value_zone"},
    "risk_adjusted": {"risk_budget", "volatility_control", "drawdown_guard"},
    "drawdown_resilience": {"risk_budget", "drawdown_guard", "trailing_stop"},
    "volatility_breakout": {"breakout_confirmation", "volatility_ladder", "trailing_stop"},
    "liquidity_confirmation": {"liquidity_aware", "liquidity_monitor", "liquidity_exit"},
    "sentiment_confirmation": {"momentum_continuation", "signal_persistence", "time_stop"},
    "atr_pullback": {"trailing_stop", "profit_protection", "atr_target"},
    "ma_retest": {"trend_follow", "signal_persistence", "resistance"},
    "breakout_confirmation": {"trend_follow", "trailing_stop", "trailing_exit"},
    "volatility_ladder": {"volatility_control", "partial_ladder", "volatility_band"},
    "liquidity_aware": {"liquidity_monitor", "liquidity_exit", "partial_ladder"},
    "value_zone": {"signal_persistence", "risk_reward", "resistance"},
    "momentum_continuation": {"trend_follow", "profit_protection", "trailing_exit"},
    "risk_budget": {"volatility_control", "drawdown_guard", "risk_reward"},
    "trend_follow": {"trailing_exit", "resistance", "atr_target"},
    "trailing_stop": {"trailing_exit", "profit_protection", "partial_ladder"},
    "volatility_control": {"volatility_band", "partial_ladder", "liquidity_exit"},
    "drawdown_guard": {"risk_reward", "trailing_exit", "liquidity_exit"},
    "signal_persistence": {"time_decay", "resistance", "atr_target"},
    "liquidity_monitor": {"liquidity_exit", "partial_ladder", "time_decay"},
    "time_stop": {"time_decay", "liquidity_exit", "risk_reward"},
    "profit_protection": {"partial_ladder", "trailing_exit", "resistance"},
}

def _softmax(values, temperature):
    x = np.asarray(values, float) / max(float(temperature), 0.15); x -= np.max(x)
    exp = np.exp(np.clip(x, -50, 50)); return exp / exp.sum()

def _nonlinear_pool(scores, weights):
    scores = np.clip(np.asarray(scores, float), -0.98, 0.98)
    latent = float(np.sum(weights * np.arctanh(scores)))
    leaders = np.argsort(weights)[-2:]
    synergy = 0.20 * math.sqrt(weights[leaders[0]] * weights[leaders[1]]) * scores[leaders[0]] * scores[leaders[1]]
    return float(np.tanh(latent + synergy))

def _stage(name, scores, regime, previous=None):
    methods = STAGE_METHODS[name]
    base = np.array([scores[m] for m in methods], float)
    temperature = 0.55 + 0.65 * regime["volatility"]
    inherited = None
    transition_boost = np.zeros(len(methods))
    if previous:
        inherited = previous["dominant_method"]
        compatible = TRANSITIONS.get(inherited, set())
        transition_boost = np.array([0.85 if method in compatible else -0.10 for method in methods])
    utilities = base + transition_boost + 0.25 * regime["trend"] * np.array([1 if "trend" in m or "momentum" in m or "breakout" in m else -0.2 for m in methods])
    weights = _softmax(utilities, temperature)
    dominant = methods[int(np.argmax(weights))]
    return {"stage": name, "dominant_method": dominant, "inherited_from": inherited, "temperature": temperature, "nonlinear_score": _nonlinear_pool(base, weights), "method_weights": {m: float(w) for m, w in zip(methods, weights)}, "method_scores": {m: float(scores[m]) for m in methods}, "transition_boost": {m: float(v) for m, v in zip(methods, transition_boost)}}

def four_stage_decision_chain(frame: pd.DataFrame, snap: dict, perf: dict, friction: float, alternative_signal=0.0, position=None):
    close = pd.to_numeric(frame["close"], errors="coerce").dropna()
    volume = pd.to_numeric(frame.get("volume", pd.Series(1.0, index=frame.index)), errors="coerce").fillna(0)
    returns = close.pct_change().dropna(); vol20 = float(returns.tail(20).std() * np.sqrt(252)) if len(returns) else 0.15
    drawdown = float(close.iloc[-1] / close.cummax().iloc[-1] - 1)
    volume_ratio = float(volume.tail(5).mean() / max(volume.tail(20).mean(), 1e-12))
    trend = float(np.tanh((snap["last_price"] / snap["ma60"] - 1) / max(snap["atr_pct"], 0.01)))
    regime = {"trend": trend, "volatility": float(np.clip(vol20 / 0.35, 0, 1)), "drawdown": drawdown, "liquidity": float(np.tanh(volume_ratio - 1)), "sentiment": float(np.clip(alternative_signal, -1, 1))}
    selection_scores = {
        "trend": trend, "momentum": float(np.tanh(snap["momentum_60d"] * 5)), "mean_reversion": float(np.tanh((snap["ma20"] / snap["last_price"] - 1) / max(snap["atr_pct"], .01))),
        "risk_adjusted": float(np.tanh(perf.get("sharpe", 0))), "drawdown_resilience": float(np.clip(1 + drawdown / .25, -1, 1)), "volatility_breakout": float(np.tanh(abs(snap["last_price"] / snap["ma20"] - 1) / max(snap["atr_pct"], .01) - 1)),
        "liquidity_confirmation": regime["liquidity"], "sentiment_confirmation": regime["sentiment"],
    }
    selection = _stage("selection", selection_scores, regime)
    entry_scores = {
        "atr_pullback": float(np.tanh((snap["ma20"] - snap["last_price"]) / max(snap["atr14"], .01))), "ma_retest": float(1 - min(abs(snap["last_price"] / snap["ma20"] - 1) / max(snap["atr_pct"], .01), 2)),
        "breakout_confirmation": trend * max(regime["liquidity"], 0), "volatility_ladder": 1 - regime["volatility"], "liquidity_aware": regime["liquidity"], "value_zone": float(np.tanh((snap["ma60"] - snap["last_price"]) / max(snap["atr14"], .01))),
        "momentum_continuation": float(np.tanh(snap["momentum_60d"] * 5)), "risk_budget": float(1 - regime["volatility"]),
    }
    entry = _stage("entry", entry_scores, regime, selection)
    position = position or {}; pnl = float(position.get("pnl_pct", 0)); holding_days = float(position.get("holding_days", 0))
    holding_scores = {
        "trend_follow": trend, "trailing_stop": float(np.clip(1 + drawdown / .12, -1, 1)), "volatility_control": 1 - 2 * regime["volatility"], "drawdown_guard": float(np.clip(-drawdown / .15, -1, 1)),
        "signal_persistence": selection["nonlinear_score"], "liquidity_monitor": regime["liquidity"], "time_stop": float(np.tanh((20 - holding_days) / 20)), "profit_protection": float(np.tanh(pnl * 8)),
    }
    holding = _stage("holding", holding_scores, regime, entry)
    exit_scores = {
        "atr_target": float(np.tanh((1.4 * snap["atr_pct"] - friction) * 10)), "volatility_band": regime["volatility"], "trailing_exit": float(np.clip(-drawdown / .10, -1, 1)), "risk_reward": float(np.tanh((pnl - friction) * 8)),
        "resistance": float(np.tanh((snap["last_price"] / max(close.tail(60).max(), .01) - .97) * 20)), "time_decay": float(np.tanh((holding_days - 20) / 20)), "liquidity_exit": -regime["liquidity"], "partial_ladder": float(np.tanh((pnl - .03) * 10)),
    }
    take_profit = _stage("take_profit", exit_scores, regime, holding)
    return {"market_regime": regime, "selection": selection, "entry": entry, "holding": holding, "take_profit": take_profit, "aggregation": "softmax regime gate + transition prior + nonlinear atanh pool with pair synergy"}
