"""Evidence-gated evaluation of the optional strategy-selection layer.

Runs the honest small-capital backtest under (1) the production "balanced"
archetype, (2) each fixed archetype, and (3) the hysteresis selector, then
decides whether the selector adds out-of-sample value after costs. The
production default keeps the selector disabled unless the gate passes.

Usage:
    python run_strategy_selector.py [--capital 100000]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from Main.strategy_selector import STRATEGY_ARCHETYPES
from Main.weekly_rotation import weekly_rotation_backtest
from run_weekly_rotation import (
    DATA_CACHE,
    PRODUCTION_UNIVERSE,
    build_pit_universe,
    default_params,
    load_cached_dividends,
    load_frames,
    load_pit_dividends,
)


def _oos(rets: pd.DataFrame) -> Dict[str, float]:
    sub = rets[rets.index >= "2022-01-01"]
    n = len(sub)
    ann = float((1 + sub["strategy_return"]).prod() ** (252 / n) - 1) if n else 0.0
    sh = float(sub["strategy_return"].mean() / sub["strategy_return"].std(ddof=0) * np.sqrt(252)) if n > 1 else 0.0
    eq = (1 + sub["strategy_return"]).cumprod()
    mdd = float((eq / eq.cummax() - 1).min()) if n else 0.0
    return {"oos_ann": ann, "oos_sharpe": sh, "oos_mdd": mdd, "oos_days": n}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capital", type=float, default=100_000.0)
    parser.add_argument("--pit", action="store_true", help="run on the survivorship-free PIT universe (current cache coverage)")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "reports" / "strategy_selector")
    args = parser.parse_args()

    alive_mask = None
    benchmark_exclude = None
    if args.pit:
        pit = build_pit_universe(DATA_CACHE)
        frames = pit["frames"]
        alive_mask = pit["alive_mask"]
        from Main.pit_universe import ETF_UNIVERSE
        benchmark_exclude = tuple(ETF_UNIVERSE)
    else:
        frames = load_frames(DATA_CACHE, sorted(set(PRODUCTION_UNIVERSE)))
    div_cash = load_cached_dividends(sorted(frames)) if args.pit else load_pit_dividends(sorted(set(PRODUCTION_UNIVERSE)))
    rows: list[dict] = []

    for name in ["balanced", *sorted(k for k in STRATEGY_ARCHETYPES if k != "balanced")]:
        p = default_params()
        p.dividend_cash = div_cash
        p.capital_base = args.capital
        p.alive_mask = alive_mask
        if benchmark_exclude:
            p.benchmark_exclude = benchmark_exclude
        if name != "balanced":
            from Main.strategy_selector import build_archetype_params
            p = build_archetype_params(p, name)
        result = weekly_rotation_backtest(frames, p, regime_detector_kwargs={"bull_threshold": 0.55})
        s, rets = result["summary"], result["returns"]
        oos = _oos(rets)
        row = {
            "archetype": name,
            "ann": s["annual_return"], "sharpe": s["sharpe"], "calmar": s["calmar"],
            "mdd": s["max_drawdown"], "recovery_3y": s.get("max_drawdown_recovery_days_3y"),
            "qwin_csi300": s["quarterly_win_vs_index"], "qwin_ew": s["quarterly_win_vs_benchmark"],
            "turnover": s["avg_rebalance_turnover"], "cost": s["total_cost_fraction"],
            "trades": s["closed_trades_count"], **oos,
        }
        rows.append(row)
        print(f"{name:>10s}: ann={row['ann']:.2%} sharpe={row['sharpe']:.2f} calmar={row['calmar']:.2f} "
              f"mdd={row['mdd']:.2%} | OOS ann={row['oos_ann']:.2%} sharpe={row['oos_sharpe']:.2f}")

    # selector run (hysteresis on top of the production base)
    p = default_params()
    p.dividend_cash = div_cash
    p.capital_base = args.capital
    p.alive_mask = alive_mask
    if benchmark_exclude:
        p.benchmark_exclude = benchmark_exclude
    p.strategy_selector = "hysteresis"
    result = weekly_rotation_backtest(frames, p, regime_detector_kwargs={"bull_threshold": 0.55})
    s, rets = result["summary"], result["returns"]
    oos = _oos(rets)
    sel = {
        "archetype": "selector",
        "ann": s["annual_return"], "sharpe": s["sharpe"], "calmar": s["calmar"],
        "mdd": s["max_drawdown"], "recovery_3y": s.get("max_drawdown_recovery_days_3y"),
        "qwin_csi300": s["quarterly_win_vs_index"], "qwin_ew": s["quarterly_win_vs_benchmark"],
        "turnover": s["avg_rebalance_turnover"], "cost": s["total_cost_fraction"],
        "trades": s["closed_trades_count"], **oos,
    }
    rows.append(sel)
    print(f"{'selector':>10s}: ann={sel['ann']:.2%} sharpe={sel['sharpe']:.2f} calmar={sel['calmar']:.2f} "
          f"mdd={sel['mdd']:.2%} | OOS ann={sel['oos_ann']:.2%} sharpe={sel['oos_sharpe']:.2f}")

    best_fixed = max((r for r in rows if r["archetype"] != "selector"), key=lambda r: r["oos_sharpe"])
    gate_pass = sel["oos_sharpe"] > best_fixed["oos_sharpe"] and sel["oos_ann"] > best_fixed["oos_ann"]
    verdict = {
        "gate_pass": gate_pass,
        "recommendation": "ENABLE_selector" if gate_pass else "KEEP_disabled",
        "best_fixed": best_fixed["archetype"],
        "reason": (
            f"selector OOS Sharpe {sel['oos_sharpe']:.3f} vs best fixed {best_fixed['archetype']} "
            f"{best_fixed['oos_sharpe']:.3f}; OOS ann {sel['oos_ann']:.2%} vs {best_fixed['oos_ann']:.2%}"
        ),
    }
    print("\nEvidence gate:", json.dumps(verdict, ensure_ascii=False, indent=2))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "strategy_selector_report.json").write_text(
        json.dumps({"capital": args.capital, "rows": rows, "verdict": verdict}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
