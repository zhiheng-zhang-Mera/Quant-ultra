"""Decoupled four-sleeve capital-allocation layer (40/30/20/10).

The engine stays a single-book backtest. This module is a pure orchestrator
that composes independent engine books ("sleeves") and combines their daily
NAVs under a fixed capital-quota rule:

* Each sleeve is its own ``weekly_rotation_backtest`` run with its own
  archetype parameter overrides, costs, stops and holdings. Nothing inside the
  engine knows about sleeves.
* The orchestrator only combines daily returns. Constant-mix rebalancing
  restores the fixed quotas (e.g. 40% base / 30% balanced / 20% momentum /
  10% sprint) on a monthly grid, with a deviation threshold so small drift is
  not traded away.
* Rebalancing flow is charged a sleeve-level cost rate (fees + slippage on
  the traded notional between books), so the reported blend is net of the
  cost of keeping quotas fixed.
* After a profit the quotas stay unchanged by construction: the rebalance
  sells the winners back to their target weight (constant-mix, not
  buy-and-hold drift).

Evidence gate: like every layer in this repo, the sleeve portfolio must be
validated on the honest full-PIT pool (``run_sleeve_portfolio.py --pit``)
before it may replace the single-book production default. The production
default keeps ``run_weekly_rotation`` untouched.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from Main.strategy_selector import build_archetype_params
from Main.weekly_rotation import RotationParams, summarize, weekly_rotation_backtest


@dataclass(frozen=True)
class SleeveSpec:
    """One capital sleeve: a weight plus an engine archetype and overrides."""

    name: str
    weight: float
    archetype: str  # balanced | momentum | defensive | safe | sprint
    overrides: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SleevePortfolioConfig:
    """Combination rules for the sleeve portfolio."""

    rebalance_days: int = 21          # monthly grid in trading days
    threshold: float = 0.03           # rebalance only when max |weight - target| > this
    cost_rate: float = 0.0017         # per-unit flow cost between sleeves
    start_date: Optional[str] = None
    end_date: Optional[str] = None


# 40% 保底 / 30% 平衡 / 20% 先锋 / 10% 冲刺
DEFAULT_SLEEVES: Tuple[SleeveSpec, ...] = (
    SleeveSpec(name="base", weight=0.40, archetype="safe"),
    SleeveSpec(name="balanced", weight=0.30, archetype="balanced"),
    SleeveSpec(name="momentum", weight=0.20, archetype="momentum"),
    SleeveSpec(name="sprint", weight=0.10, archetype="sprint"),
)


def default_base_params() -> RotationParams:
    """Production base parameters (lazy import keeps the module import-light)."""
    from run_weekly_rotation import default_params

    return default_params()


def combine_sleeves(
    returns: pd.DataFrame,
    weights: Sequence[float],
    rebalance_days: int = 21,
    threshold: float = 0.03,
    cost_rate: float = 0.0017,
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Constant-mix combination of sleeve daily returns.

    ``returns`` is a (date x sleeve) DataFrame of daily returns. Returns a
    ``(combined, weights_history, meta)`` tuple where ``combined`` has
    equity/strategy_return/turnover/cost columns and ``meta`` carries the
    total rebalance cost and final weights.
    """
    R = returns.dropna()
    targets = np.asarray(weights, dtype=float)
    targets = targets / targets.sum()
    w = targets.copy()
    nav = 1.0
    equity: List[float] = []
    turnover_s: List[float] = []
    cost_s: List[float] = []
    w_hist: List[Dict[str, float]] = []
    n = len(R)
    rebalances = 0
    for i, (d, row) in enumerate(R.iterrows()):
        port_r = float(np.dot(w, row.values))
        nav *= 1.0 + port_r
        drift = w * (1.0 + row.values) / (1.0 + port_r)
        flow = 0.0
        cost = 0.0
        if i % rebalance_days == 0 and i < n - 1:
            if float(np.max(np.abs(drift - targets))) > threshold:
                flow = float(np.abs(drift - targets).sum())
                cost = flow * cost_rate
                nav *= 1.0 - cost
                w = targets.copy()
                rebalances += 1
            else:
                w = drift
        else:
            w = drift
        equity.append(nav)
        turnover_s.append(flow)
        cost_s.append(cost)
        w_hist.append({c: float(v) for c, v in zip(R.columns, w)})
    eq = pd.Series(equity, index=R.index)
    combined = pd.DataFrame(index=R.index)
    combined["equity"] = eq
    combined["strategy_return"] = eq.pct_change(fill_method=None).fillna(0.0)
    combined["turnover"] = turnover_s
    combined["cost"] = cost_s
    weights_history = pd.DataFrame(w_hist, index=R.index)
    meta = {
        "rebalances": int(rebalances),
        "rebalance_cost": float(np.sum(cost_s)),
        "final_weights": weights_history.iloc[-1].to_dict(),
        "rebalance_days": int(rebalance_days),
        "threshold": float(threshold),
        "cost_rate": float(cost_rate),
    }
    return combined, weights_history, meta


def run_sleeve_portfolio(
    frames: Dict[str, pd.DataFrame],
    sleeves: Sequence[SleeveSpec] = DEFAULT_SLEEVES,
    base_params: Optional[RotationParams] = None,
    config: Optional[SleevePortfolioConfig] = None,
    dividend_cash: Optional[pd.DataFrame] = None,
    alive_mask: Optional[pd.DataFrame] = None,
    benchmark_exclude: Optional[Tuple[str, ...]] = None,
    regime_detector_kwargs: Optional[dict] = None,
) -> dict:
    """Run each sleeve as an independent engine book, then combine net NAVs.

    Returns a dict with the combined returns/summary, per-sleeve summaries,
    the weights history and combination meta. The combined returns are
    summarized with the engine's own ``summarize`` so all metrics (Sharpe,
    Calmar, MDD, win rates, regime breakdown) are computed identically to the
    single-book baseline.
    """
    base_params = base_params or default_base_params()
    config = config or SleevePortfolioConfig()
    weights = [float(s.weight) for s in sleeves]
    sleeve_results: Dict[str, dict] = {}
    trade_frames: List[pd.DataFrame] = []
    for sleeve in sleeves:
        p = base_params
        if sleeve.archetype and sleeve.archetype != "balanced":
            p = build_archetype_params(p, sleeve.archetype)
        if sleeve.overrides:
            p = replace(p, **sleeve.overrides)
        if config.start_date:
            p = replace(p, start_date=config.start_date)
        if config.end_date:
            p = replace(p, end_date=config.end_date)
        p = replace(p, dividend_cash=dividend_cash, alive_mask=alive_mask)
        if benchmark_exclude is not None:
            p = replace(p, benchmark_exclude=benchmark_exclude)
        result = weekly_rotation_backtest(frames, p, regime_detector_kwargs=regime_detector_kwargs)
        sleeve_results[sleeve.name] = {
            "spec": sleeve,
            "summary": result["summary"],
            "returns": result["returns"],
            "regimes": result["regimes"],
            "closed_trades": result.get("closed_trades", pd.DataFrame()),
            "alternative_signal_governance": result.get("alternative_signal_governance"),
        }
        trades = result.get("closed_trades")
        if trades is not None and len(trades):
            tagged = trades.copy()
            tagged["sleeve"] = sleeve.name
            trade_frames.append(tagged)

    R = pd.DataFrame({name: res["returns"]["strategy_return"] for name, res in sleeve_results.items()}).dropna()
    bench = pd.DataFrame({name: res["returns"]["benchmark_return"] for name, res in sleeve_results.items()}).dropna()
    expo = pd.DataFrame({name: res["returns"]["gross_exposure"] for name, res in sleeve_results.items()}).dropna()
    combined, w_hist, meta = combine_sleeves(
        R, weights,
        rebalance_days=config.rebalance_days,
        threshold=config.threshold,
        cost_rate=config.cost_rate,
    )
    weights_sum = np.asarray(weights, dtype=float).sum()
    bench_w = np.asarray(weights, dtype=float) / weights_sum
    combined_frame = pd.DataFrame(index=combined.index)
    combined_frame["strategy_return"] = combined["strategy_return"]
    combined_frame["benchmark_return"] = (bench * bench_w).sum(axis=1)
    combined_frame["gross_exposure"] = (expo * bench_w).sum(axis=1)
    combined_frame["turnover"] = combined["turnover"]
    combined_frame["cost"] = combined["cost"]
    first_name = next(iter(sleeve_results))
    combined_frame["regime"] = sleeve_results[first_name]["returns"]["regime"].reindex(combined.index)

    closed_trades = pd.concat(trade_frames, ignore_index=True) if trade_frames else []
    if isinstance(closed_trades, pd.DataFrame) and len(closed_trades):
        closed_trades = closed_trades.to_dict("records")
    index_returns = None
    if "510300.SH" in frames:
        index_close = frames["510300.SH"]["close"].reindex(combined.index).ffill()
        index_returns = index_close.pct_change(fill_method=None)
    summary = summarize(combined_frame, sleeve_results[first_name]["regimes"], base_params, closed_trades, index_returns)
    summary["sleeve_rebalance_cost"] = meta["rebalance_cost"]
    summary["sleeve_rebalances"] = meta["rebalances"]
    return {
        "returns": combined_frame,
        "summary": summary,
        "meta": meta,
        "weights_history": w_hist,
        "sleeves": sleeve_results,
        "params": base_params,
    }
