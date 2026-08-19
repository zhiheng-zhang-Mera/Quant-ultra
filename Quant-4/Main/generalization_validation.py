"""Frozen-parameter cross-market and temporal-regime validation."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Mapping

import numpy as np
import pandas as pd

GENERALIZATION_SPEC = {
    "version": "generalization-validation/v1", "min_market_observations": 252,
    "min_window_observations": 20, "bull_60d_return": 0.10, "bear_60d_return": -0.10,
    "high_vol_annualized": 0.25, "low_vol_annualized": 0.10,
    "extreme_daily_return": 0.05, "liquidity_shock_change": -0.30,
    "transfer_sharpe_floor": 0.0,
}


@dataclass(frozen=True)
class FrozenResearchLogic:
    source_market: str
    frozen_at: str
    code_version: str
    parameter_hash: str
    data_cutoff: str
    factor_version: str
    risk_version: str
    execution_version: str

    def validate(self) -> None:
        if not all(asdict(self).values()):
            raise ValueError("all frozen research-logic fields are required")
        if pd.Timestamp(self.data_cutoff) > pd.Timestamp(self.frozen_at):
            raise ValueError("data cutoff cannot be after the freeze timestamp")


@dataclass(frozen=True)
class MarketValidationContext:
    market: str
    data_version: str
    parameter_hash: str
    validation_start: str
    validation_end: str
    calendar: str
    currency: str
    transaction_cost_model: str
    trading_constraints: str
    pit_enforced: bool
    target_search_trials: int = 0

    def validate(self, frozen: FrozenResearchLogic) -> None:
        if self.market == frozen.source_market:
            raise ValueError("external validation market must differ from the source market")
        if self.parameter_hash != frozen.parameter_hash:
            raise ValueError("target market parameters differ from the frozen source logic")
        if self.target_search_trials != 0:
            raise ValueError("target-market parameter search is prohibited")
        if not self.pit_enforced or not all((self.data_version, self.calendar, self.currency, self.transaction_cost_model, self.trading_constraints)):
            raise ValueError("external validation requires PIT data and complete market metadata")
        if not pd.Timestamp(self.validation_start) < pd.Timestamp(self.validation_end):
            raise ValueError("validation_start must precede validation_end")


def _metrics(values: pd.Series) -> dict:
    clean = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < 2:
        return {"observations": int(len(clean)), "annual_return": None, "sharpe": None, "max_drawdown": None}
    nav = (1.0 + clean).cumprod()
    std = float(clean.std(ddof=1))
    return {
        "observations": int(len(clean)),
        "annual_return": float(nav.iloc[-1] ** (252 / len(clean)) - 1.0) if nav.iloc[-1] > 0 else -1.0,
        "sharpe": float(clean.mean() / std * math.sqrt(252)) if std > 0 else 0.0,
        "max_drawdown": float((nav / nav.cummax() - 1.0).min()),
    }


def validate_external_market(returns: pd.DataFrame, context: MarketValidationContext, frozen: FrozenResearchLogic) -> dict:
    """Complete an external-market test without permitting target tuning."""
    frozen.validate()
    context.validate(frozen)
    required = {"strategy_return", "benchmark_return", "factor_return", "risk_return", "execution_cost"}
    missing = required - set(returns.columns)
    if missing:
        raise ValueError(f"external validation missing decomposition columns: {sorted(missing)}")
    if not isinstance(returns.index, pd.DatetimeIndex) or returns.index.has_duplicates or not returns.index.is_monotonic_increasing:
        raise ValueError("external returns require a sorted, unique DatetimeIndex")
    bounded = returns.loc[pd.Timestamp(context.validation_start):pd.Timestamp(context.validation_end)].apply(pd.to_numeric, errors="coerce")
    strategy, benchmark = _metrics(bounded["strategy_return"]), _metrics(bounded["benchmark_return"])
    complete = len(bounded.dropna(subset=list(required))) >= GENERALIZATION_SPEC["min_market_observations"]
    transfer_checks = {
        "positive_sharpe": complete and strategy["sharpe"] is not None and strategy["sharpe"] >= GENERALIZATION_SPEC["transfer_sharpe_floor"],
        "return_not_worse": complete and strategy["annual_return"] is not None and benchmark["annual_return"] is not None and strategy["annual_return"] >= benchmark["annual_return"],
        "sharpe_not_worse": complete and strategy["sharpe"] is not None and benchmark["sharpe"] is not None and strategy["sharpe"] >= benchmark["sharpe"],
        "drawdown_not_worse": complete and strategy["max_drawdown"] is not None and benchmark["max_drawdown"] is not None and strategy["max_drawdown"] >= benchmark["max_drawdown"],
    }
    if all(transfer_checks.values()):
        transfer_verdict = "TRANSFER_SUPPORTED"
    elif transfer_checks["positive_sharpe"] and any(list(transfer_checks.values())[1:]):
        transfer_verdict = "TRANSFER_MIXED"
    else:
        transfer_verdict = "TRANSFER_FAILED"
    payload = {
        "schema_version": GENERALIZATION_SPEC["version"], "frozen_logic": asdict(frozen),
        "market_context": asdict(context), "strategy": strategy, "benchmark": benchmark,
        "decomposition": {"factor_annual_mean": float(bounded["factor_return"].mean() * 252),
                          "risk_annual_mean": float(bounded["risk_return"].mean() * 252),
                          "execution_cost_annual": float(bounded["execution_cost"].sum() * 252 / max(len(bounded), 1))},
        "market_specialization": {"calendar": context.calendar, "currency": context.currency,
                                  "cost_model": context.transaction_cost_model,
                                  "constraints": context.trading_constraints},
        "validation_complete": complete, "transfer_checks": transfer_checks, "transfer_verdict": transfer_verdict,
    }
    payload["evidence_sha256"] = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()
    return {"status": "COMPLETE" if complete else "HOLD", "action": "EXTERNAL_EVIDENCE" if complete else "OBSERVATION_ONLY", **payload}


def classify_market_regimes(benchmark_returns: pd.Series, liquidity_change: pd.Series | None = None) -> pd.DataFrame:
    returns = pd.to_numeric(benchmark_returns, errors="coerce").sort_index()
    trailing = (1.0 + returns).rolling(60, min_periods=20).apply(np.prod, raw=True) - 1.0
    volatility = returns.rolling(20, min_periods=10).std(ddof=1) * math.sqrt(252)
    liquidity = pd.to_numeric(liquidity_change, errors="coerce").reindex(returns.index) if liquidity_change is not None else pd.Series(np.nan, index=returns.index)
    frame = pd.DataFrame(index=returns.index)
    frame["bull"] = trailing >= GENERALIZATION_SPEC["bull_60d_return"]
    frame["bear"] = trailing <= GENERALIZATION_SPEC["bear_60d_return"]
    frame["range"] = ~(frame["bull"] | frame["bear"])
    frame["high_vol"] = volatility >= GENERALIZATION_SPEC["high_vol_annualized"]
    frame["low_vol"] = volatility <= GENERALIZATION_SPEC["low_vol_annualized"]
    frame["liquidity_shock"] = liquidity <= GENERALIZATION_SPEC["liquidity_shock_change"]
    frame["extreme"] = returns.abs() >= GENERALIZATION_SPEC["extreme_daily_return"]
    frame["primary_regime"] = np.select(
        [frame["extreme"], frame["liquidity_shock"], frame["high_vol"], frame["bull"], frame["bear"]],
        ["EXTREME", "LIQUIDITY_SHOCK", "HIGH_VOL", "BULL", "BEAR"], default="RANGE",
    )
    frame["trailing_60d_return"] = trailing
    frame["annualized_volatility"] = volatility
    return frame


def evaluate_regime_robustness(
    strategy_returns: pd.Series, benchmark_returns: pd.Series, *,
    factor_returns: Mapping[str, pd.Series] | None = None,
    unprotected_returns: pd.Series | None = None, liquidity_change: pd.Series | None = None,
) -> dict:
    regimes = classify_market_regimes(benchmark_returns, liquidity_change)
    joined = pd.concat([strategy_returns.rename("strategy"), benchmark_returns.rename("benchmark"), regimes["primary_regime"]], axis=1).dropna()
    rows = {}
    for regime, group in joined.groupby("primary_regime"):
        rows[str(regime)] = {"strategy": _metrics(group["strategy"]), "benchmark": _metrics(group["benchmark"]),
                             "excess_annual_mean": float((group["strategy"] - group["benchmark"]).mean() * 252)}
    valid = {key: value for key, value in rows.items() if value["strategy"]["observations"] >= GENERALIZATION_SPEC["min_window_observations"]}
    best = max(valid, key=lambda key: valid[key]["excess_annual_mean"]) if valid else None
    worst = min(valid, key=lambda key: valid[key]["excess_annual_mean"]) if valid else None
    factor_by_regime = {}
    for name, series in (factor_returns or {}).items():
        data = pd.concat([series.rename("factor"), regimes["primary_regime"]], axis=1).dropna()
        factor_by_regime[name] = {str(key): _metrics(group["factor"]) for key, group in data.groupby("primary_regime")}
    risk_effect = {}
    if unprotected_returns is not None:
        data = pd.concat([strategy_returns.rename("protected"), unprotected_returns.rename("unprotected"), regimes["primary_regime"]], axis=1).dropna()
        for key, group in data.groupby("primary_regime"):
            protected, unprotected = _metrics(group["protected"]), _metrics(group["unprotected"])
            risk_effect[str(key)] = {"drawdown_improvement": (protected["max_drawdown"] - unprotected["max_drawdown"])
                                     if protected["max_drawdown"] is not None and unprotected["max_drawdown"] is not None else None}
    checks = {"multiple_regimes": len(valid) >= 2, "advantage_identified": best is not None,
              "weakness_identified": worst is not None, "factor_regimes": bool(factor_by_regime),
              "risk_mechanism": bool(risk_effect)}
    reasons = [name for name, passed in checks.items() if not passed]
    return {"status": "PASS" if not reasons else "HOLD", "action": "ROBUSTNESS_EVIDENCE" if not reasons else "OBSERVATION_ONLY",
            "checks": checks, "reasons": reasons, "regimes": rows, "best_regime": best, "worst_regime": worst,
            "factor_by_regime": factor_by_regime, "risk_effect_by_regime": risk_effect,
            "classification_spec": dict(GENERALIZATION_SPEC)}


def evaluate_temporal_windows(strategy_returns: pd.Series, windows: Mapping[str, tuple[str, str]]) -> dict:
    ordered = sorted((pd.Timestamp(start), pd.Timestamp(end), name) for name, (start, end) in windows.items())
    if any(start >= end for start, end, _ in ordered) or any(ordered[index][1] >= ordered[index + 1][0] for index in range(len(ordered) - 1)):
        raise ValueError("temporal validation windows must be ordered and non-overlapping")
    results = {name: _metrics(strategy_returns.loc[start:end]) for start, end, name in ordered}
    complete = bool(results) and all(item["observations"] >= GENERALIZATION_SPEC["min_window_observations"] for item in results.values())
    return {"status": "PASS" if complete else "HOLD", "windows": results,
            "best_window": max(results, key=lambda key: results[key]["sharpe"] if results[key]["sharpe"] is not None else -np.inf) if results else None,
            "worst_window": min(results, key=lambda key: results[key]["sharpe"] if results[key]["sharpe"] is not None else np.inf) if results else None}
