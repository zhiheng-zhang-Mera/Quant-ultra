"""Run a parameter-grid experiment on the 100%-coverage PIT universe and save
the honest summary + OOS metrics. Each config is one backtest (~18 min at the
full 5,475-name pool); run several in parallel processes.

Usage:
    python tools/run_pit_experiments.py --config persist --output-dir reports/experiments
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from Main.pit_universe import ETF_UNIVERSE  # noqa: E402
from Main.weekly_rotation import weekly_rotation_backtest  # noqa: E402
from run_weekly_rotation import (  # noqa: E402
    DATA_CACHE,
    build_pit_universe,
    default_params,
    load_cached_dividends,
)

# Param overrides on top of the production default (balanced).
CONFIGS = {
    "base": {},
    "persist": {"hold_persistent": True, "persist_rank_floor": 8},
    "euphoria15": {"euphoria_threshold": 0.15},
    "liq150m": {"min_adv": 1.5e8},
    "breadth8": {"top_n": 8, "per_position_cap": 0.20},
    "persist_euphoria": {"hold_persistent": True, "persist_rank_floor": 8, "euphoria_threshold": 0.15},
    "persist_liq": {"hold_persistent": True, "persist_rank_floor": 8, "min_adv": 1.5e8},
    "persist_nomax": {"hold_persistent": True, "persist_rank_floor": 8, "max_holding_days": 0},
    "persist_nomax_euph": {"hold_persistent": True, "persist_rank_floor": 8, "max_holding_days": 0, "euphoria_threshold": 0.15},
    "slow42": {"rebalance_days": 42, "hold_persistent": True, "persist_rank_floor": 8, "max_holding_days": 0},
    "breadth20": {"top_n": 20, "per_position_cap": 0.06, "max_etf_positions": 4, "hold_persistent": True, "persist_rank_floor": 25, "max_holding_days": 0},
    "noharvest": {"enable_intraweek_stops": False},
    "noharvest_euph": {"enable_intraweek_stops": False, "euphoria_threshold": 0.15},
    "harvest_loose": {"take_profit_pct": 0.15, "stop_loss_pct": 0.10},
    "monthly": {"rebalance_weekday": None, "rebalance_days": 21},
    "monthly_persist": {"rebalance_weekday": None, "rebalance_days": 21, "hold_persistent": True, "persist_rank_floor": 8, "max_holding_days": 0},
    "monthly_noharvest": {"rebalance_weekday": None, "rebalance_days": 21, "enable_intraweek_stops": False},
    "best_guess": {
        "rebalance_weekday": None, "rebalance_days": 21,
        "enable_intraweek_stops": True, "take_profit_pct": 0.15, "stop_loss_pct": 0.10,
        "euphoria_threshold": 0.15, "hold_persistent": True, "persist_rank_floor": 8, "max_holding_days": 0,
    },
    "monthly_persist_euph": {
        "rebalance_weekday": None, "rebalance_days": 21,
        "hold_persistent": True, "persist_rank_floor": 8, "max_holding_days": 0,
        "euphoria_threshold": 0.15,
    },
    "best_guess_loose": {
        "rebalance_weekday": None, "rebalance_days": 21,
        "enable_intraweek_stops": True, "take_profit_pct": 0.20, "stop_loss_pct": 0.12,
        "euphoria_threshold": 0.15, "hold_persistent": True, "persist_rank_floor": 8, "max_holding_days": 0,
    },
    "mp_euph_tp10": {
        "rebalance_weekday": None, "rebalance_days": 21,
        "hold_persistent": True, "persist_rank_floor": 8, "max_holding_days": 0,
        "euphoria_threshold": 0.15, "take_profit_pct": 0.10, "stop_loss_pct": 0.09,
    },
    "mp_euph_tp12": {
        "rebalance_weekday": None, "rebalance_days": 21,
        "hold_persistent": True, "persist_rank_floor": 8, "max_holding_days": 0,
        "euphoria_threshold": 0.15, "take_profit_pct": 0.12, "stop_loss_pct": 0.09,
    },
    # ---- Calmar >= 1.0 push (2026-08-11): drawdown compression ----
    "ddg08": {
        "rebalance_weekday": None, "rebalance_days": 21,
        "hold_persistent": True, "persist_rank_floor": 8, "max_holding_days": 0,
        "euphoria_threshold": 0.15, "take_profit_pct": 0.12, "stop_loss_pct": 0.09,
        "drawdown_guard": 0.08, "drawdown_guard_max": 0.14, "dd_guard_exposure": 0.35,
    },
    "ddg06": {
        "rebalance_weekday": None, "rebalance_days": 21,
        "hold_persistent": True, "persist_rank_floor": 8, "max_holding_days": 0,
        "euphoria_threshold": 0.15, "take_profit_pct": 0.12, "stop_loss_pct": 0.09,
        "drawdown_guard": 0.06, "drawdown_guard_max": 0.12, "dd_guard_exposure": 0.30,
    },
    "safefrac75": {
        "rebalance_weekday": None, "rebalance_days": 21,
        "hold_persistent": True, "persist_rank_floor": 8, "max_holding_days": 0,
        "euphoria_threshold": 0.15, "take_profit_pct": 0.12, "stop_loss_pct": 0.09,
        "defensive_hold_safe_frac": 0.75,
    },
    "sl07": {
        "rebalance_weekday": None, "rebalance_days": 21,
        "hold_persistent": True, "persist_rank_floor": 8, "max_holding_days": 0,
        "euphoria_threshold": 0.15, "take_profit_pct": 0.12, "stop_loss_pct": 0.07,
    },
    "daily_mon": {
        "rebalance_weekday": None, "rebalance_days": 21,
        "hold_persistent": True, "persist_rank_floor": 8, "max_holding_days": 0,
        "euphoria_threshold": 0.15, "take_profit_pct": 0.12, "stop_loss_pct": 0.09,
        "daily_regime_monitoring": True,
    },
    "daily_mon_ddg": {
        "rebalance_weekday": None, "rebalance_days": 21,
        "hold_persistent": True, "persist_rank_floor": 8, "max_holding_days": 0,
        "euphoria_threshold": 0.15, "take_profit_pct": 0.12, "stop_loss_pct": 0.07,
        "daily_regime_monitoring": True, "drawdown_guard": 0.06, "dd_guard_exposure": 0.25,
    },
    "vol15": {
        "rebalance_weekday": None, "rebalance_days": 21,
        "hold_persistent": True, "persist_rank_floor": 8, "max_holding_days": 0,
        "euphoria_threshold": 0.15, "take_profit_pct": 0.12, "stop_loss_pct": 0.07,
        "vol_target": 0.15,
    },
    "safefrac75_sl07": {
        "rebalance_weekday": None, "rebalance_days": 21,
        "hold_persistent": True, "persist_rank_floor": 8, "max_holding_days": 0,
        "euphoria_threshold": 0.15, "take_profit_pct": 0.12, "stop_loss_pct": 0.07,
        "defensive_hold_safe_frac": 0.75,
    },
    "ddg_sanity": {
        "rebalance_weekday": None, "rebalance_days": 21,
        "hold_persistent": True, "persist_rank_floor": 8, "max_holding_days": 0,
        "euphoria_threshold": 0.15, "take_profit_pct": 0.12, "stop_loss_pct": 0.09,
        "drawdown_guard": 0.05, "drawdown_guard_max": 0.08, "dd_guard_exposure": 0.05,
    },
    "bg_sl07_sf75": {
        "rebalance_weekday": None, "rebalance_days": 21,
        "enable_intraweek_stops": True, "take_profit_pct": 0.15, "stop_loss_pct": 0.07,
        "euphoria_threshold": 0.15, "hold_persistent": True, "persist_rank_floor": 8, "max_holding_days": 0,
        "defensive_hold_safe_frac": 0.75,
    },
    "conc20": {
        "rebalance_weekday": None, "rebalance_days": 21,
        "hold_persistent": True, "persist_rank_floor": 8, "max_holding_days": 0,
        "euphoria_threshold": 0.15, "take_profit_pct": 0.12, "stop_loss_pct": 0.07,
        "per_position_cap": 0.20, "max_annual_vol": 0.30,
    },
    "monthly_momentum": {
        "rebalance_weekday": None, "rebalance_days": 21,
        "defensive_core": False, "defensive_filter": False, "defensive_core_bull_momentum": False,
        "require_relative_strength": True, "top_n": 8, "per_position_cap": 0.15,
        "hold_persistent": True, "persist_rank_floor": 12, "max_holding_days": 0,
        "stop_loss_pct": 0.08, "take_profit_pct": 0.15, "euphoria_threshold": 0.15,
    },
    "vtoff": {
        "rebalance_weekday": None, "rebalance_days": 21,
        "hold_persistent": True, "persist_rank_floor": 8, "max_holding_days": 0,
        "euphoria_threshold": 0.15, "take_profit_pct": 0.12, "stop_loss_pct": 0.07,
        "vol_target": 0.0,
    },
    "vtoff_sf75": {
        "rebalance_weekday": None, "rebalance_days": 21,
        "hold_persistent": True, "persist_rank_floor": 8, "max_holding_days": 0,
        "euphoria_threshold": 0.15, "take_profit_pct": 0.12, "stop_loss_pct": 0.07,
        "vol_target": 0.0, "defensive_hold_safe_frac": 0.75,
    },
    "vtoff_tp15": {
        "rebalance_weekday": None, "rebalance_days": 21,
        "hold_persistent": True, "persist_rank_floor": 8, "max_holding_days": 0,
        "euphoria_threshold": 0.15, "take_profit_pct": 0.15, "stop_loss_pct": 0.07,
        "vol_target": 0.0,
    },
    "bull_hold": {
        "rebalance_weekday": None, "rebalance_days": 21,
        "hold_persistent": True, "persist_rank_floor": 8, "max_holding_days": 0,
        "euphoria_threshold": 0.15, "take_profit_pct": 0.12, "stop_loss_pct": 0.07,
        "defensive_hold_safe_frac": 0.75, "bull_benchmark_hold": True,
    },
    "bull_hold_euph25": {
        "rebalance_weekday": None, "rebalance_days": 21,
        "hold_persistent": True, "persist_rank_floor": 8, "max_holding_days": 0,
        "euphoria_threshold": 0.25, "take_profit_pct": 0.12, "stop_loss_pct": 0.07,
        "defensive_hold_safe_frac": 0.75, "bull_benchmark_hold": True,
    },
    "bull_hold_shock": {
        "rebalance_weekday": None, "rebalance_days": 21,
        "hold_persistent": True, "persist_rank_floor": 8, "max_holding_days": 0,
        "euphoria_threshold": 0.15, "take_profit_pct": 0.12, "stop_loss_pct": 0.07,
        "defensive_hold_safe_frac": 0.75, "bull_benchmark_hold": True,
        "event_shock_threshold": 0.015, "event_shock_exposure": 0.30,
    },
    "bull_hold_sf85": {
        "rebalance_weekday": None, "rebalance_days": 21,
        "hold_persistent": True, "persist_rank_floor": 8, "max_holding_days": 0,
        "euphoria_threshold": 0.15, "take_profit_pct": 0.12, "stop_loss_pct": 0.07,
        "defensive_hold_safe_frac": 0.85, "bull_benchmark_hold": True,
    },
}


def _oos(rets: pd.DataFrame) -> dict:
    sub = rets[rets.index >= "2022-01-01"]
    n = len(sub)
    ann = float((1 + sub["strategy_return"]).prod() ** (252 / n) - 1) if n else 0.0
    sh = float(sub["strategy_return"].mean() / sub["strategy_return"].std(ddof=0) * np.sqrt(252)) if n > 1 else 0.0
    eq = (1 + sub["strategy_return"]).cumprod()
    mdd = float((eq / eq.cummax() - 1).min()) if n else 0.0
    return {"oos_ann": ann, "oos_sharpe": sh, "oos_mdd": mdd, "oos_days": int(n)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, choices=sorted(CONFIGS))
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "reports" / "experiments")
    args = parser.parse_args()

    pit = build_pit_universe(DATA_CACHE)
    params = default_params()
    for key, value in CONFIGS[args.config].items():
        setattr(params, key, value)
    params.dividend_cash = load_cached_dividends(sorted(pit["frames"]))
    params.alive_mask = pit["alive_mask"]
    params.benchmark_exclude = tuple(ETF_UNIVERSE)

    t0 = time.time()
    result = weekly_rotation_backtest(pit["frames"], params, regime_detector_kwargs={"bull_threshold": 0.55})
    s = result["summary"]
    oos = _oos(result["returns"])
    payload = {
        "config": args.config,
        "overrides": CONFIGS[args.config],
        "summary": {k: s[k] for k in [
            "observations", "annual_return", "benchmark_annual_return", "sharpe", "calmar",
            "max_drawdown", "max_drawdown_recovery_days", "max_drawdown_recovery_days_3y",
            "monthly_win_vs_index", "quarterly_win_vs_index", "quarterly_win_vs_benchmark",
            "avg_rebalance_turnover", "total_cost_fraction", "closed_trades_count",
            "average_exposure", "final_equity", "regime_breakdown",
        ]},
        "oos": oos,
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / f"{args.config}.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "config": args.config,
        "ann": s["annual_return"], "sharpe": s["sharpe"], "mdd": s["max_drawdown"],
        "cost": s["total_cost_fraction"], "trades": s["closed_trades_count"],
        "oos_ann": oos["oos_ann"], "oos_sharpe": oos["oos_sharpe"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
