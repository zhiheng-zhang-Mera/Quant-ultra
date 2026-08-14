"""Pre-registered, non-tunable research evidence gate.

The thresholds are intentionally code-owned rather than CLI-tunable. Changing
them requires a reviewed commit and therefore cannot be done after seeing a
backtest result without leaving an auditable Git trace.
"""
from __future__ import annotations

import hashlib
import json
import math

import numpy as np
import pandas as pd


GATE_SPEC = {
    "version": "weekly-research-gate/v1",
    "registered_at": "2026-08-14",
    "oos_start": "2022-01-01",
    "min_oos_observations": 504,
    "min_oos_annual_return": 0.06,
    "min_oos_sharpe": 0.90,
    "max_drawdown_abs": 0.10,
    "min_calmar": 1.00,
    "max_avg_rebalance_turnover": 0.50,
    "max_total_cost_fraction": 0.15,
    "max_dsr_pvalue": 0.05,
    "min_num_trials": 1,
    "dsr_sharpe_threshold": 0.50,
}
GATE_SPEC_SHA256 = hashlib.sha256(
    json.dumps(GATE_SPEC, sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def _dsr_pvalue(returns: pd.Series, num_trials: int | None) -> tuple[float | None, str]:
    if not isinstance(num_trials, (int, np.integer)) or int(num_trials) < GATE_SPEC["min_num_trials"]:
        return None, "MISSING_NUM_TRIALS"
    clean = pd.to_numeric(returns, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < GATE_SPEC["min_oos_observations"]:
        return None, "INSUFFICIENT_OOS_SAMPLES"
    std = float(clean.std(ddof=1))
    if not np.isfinite(std) or std <= 0:
        return None, "INVALID_RETURN_VARIANCE"
    sharpe = float(clean.mean() / std * np.sqrt(252))
    skew = float(clean.skew())
    kurt = float(clean.kurtosis()) + 3.0  # Pearson kurtosis for the DSR denominator
    denom_sq = 1.0 - skew * sharpe + (kurt - 1.0) / 4.0 * sharpe ** 2
    if not np.isfinite(denom_sq) or denom_sq <= 0:
        return None, "INVALID_DSR_DENOMINATOR"
    t_stat = (sharpe - GATE_SPEC["dsr_sharpe_threshold"]) / math.sqrt(denom_sq / (len(clean) - 1))
    penalty = math.sqrt(2.0 * math.log(max(2, int(num_trials))))
    pvalue = float(_normal_cdf(-t_stat / penalty))
    return (pvalue, "CALCULATED") if np.isfinite(pvalue) else (None, "NONFINITE_DSR")


def evaluate_research_gate(
    returns: pd.DataFrame,
    summary: dict,
    num_trials: int | None,
) -> dict:
    """Evaluate OOS performance, risk, costs and multiple-testing evidence."""
    if not isinstance(returns.index, pd.DatetimeIndex):
        raise TypeError("research gate requires a DatetimeIndex")
    oos = returns.loc[returns.index >= pd.Timestamp(GATE_SPEC["oos_start"]), "strategy_return"]
    oos = pd.to_numeric(oos, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    n = int(len(oos))
    if n:
        growth = float((1.0 + oos).prod())
        oos_ann = float(growth ** (252.0 / n) - 1.0) if growth > 0 else -1.0
        std = float(oos.std(ddof=1)) if n > 1 else 0.0
        oos_sharpe = float(oos.mean() / std * np.sqrt(252)) if std > 0 else 0.0
    else:
        oos_ann = 0.0
        oos_sharpe = 0.0
    dsr_pvalue, dsr_status = _dsr_pvalue(oos, num_trials)
    metrics = {
        "oos_observations": n,
        "oos_annual_return": oos_ann,
        "oos_sharpe": oos_sharpe,
        "max_drawdown": float(summary.get("max_drawdown", 0.0)),
        "calmar": float(summary.get("calmar", 0.0)),
        "avg_rebalance_turnover": float(summary.get("avg_rebalance_turnover", float("inf"))),
        "total_cost_fraction": float(summary.get("total_cost_fraction", float("inf"))),
        "num_trials": int(num_trials) if isinstance(num_trials, (int, np.integer)) else None,
        "dsr_pvalue": dsr_pvalue,
        "dsr_evidence_status": dsr_status,
    }
    checks = {
        "oos_observations": n >= GATE_SPEC["min_oos_observations"],
        "oos_annual_return": oos_ann >= GATE_SPEC["min_oos_annual_return"],
        "oos_sharpe": oos_sharpe >= GATE_SPEC["min_oos_sharpe"],
        "max_drawdown": abs(metrics["max_drawdown"]) <= GATE_SPEC["max_drawdown_abs"],
        "calmar": metrics["calmar"] >= GATE_SPEC["min_calmar"],
        "turnover": metrics["avg_rebalance_turnover"] <= GATE_SPEC["max_avg_rebalance_turnover"],
        "cost": metrics["total_cost_fraction"] <= GATE_SPEC["max_total_cost_fraction"],
        "num_trials": metrics["num_trials"] is not None and metrics["num_trials"] >= GATE_SPEC["min_num_trials"],
        "dsr": dsr_pvalue is not None and dsr_pvalue <= GATE_SPEC["max_dsr_pvalue"],
    }
    reasons = [name for name, passed in checks.items() if not passed]
    passed = not reasons
    return {
        "passed": passed,
        "status": "PASS" if passed else "HOLD_FOR_REVIEW",
        "action": "RESEARCH_ELIGIBLE" if passed else "OBSERVATION_ONLY",
        "spec": dict(GATE_SPEC),
        "spec_sha256": GATE_SPEC_SHA256,
        "checks": checks,
        "reasons": reasons,
        "metrics": metrics,
    }
