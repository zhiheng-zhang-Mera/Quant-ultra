"""CLI: run the decoupled 40/30/20/10 four-sleeve portfolio on top of the
weekly-rotation engine, with optional bottom-up ATR stop bands.

The engine itself is unchanged: each sleeve is an independent
``weekly_rotation_backtest`` book with its own archetype overrides, and this
CLI only combines their NAVs under constant-mix capital quotas
(``Main.sleeve_allocation``). This is the evidence harness for the sleeve
layer: it must beat the single-book baseline on the honest full-PIT pool
before the layer may become a production default.

Usage:
    python run_sleeve_portfolio.py [--pit] [--weights 40:30:20:10]
        [--dynamic-stops] [--rebalance-days 21] [--threshold 0.03]
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from Main.sleeve_allocation import (
    DEFAULT_SLEEVES,
    SleevePortfolioConfig,
    SleeveSpec,
    run_sleeve_portfolio,
)
from run_weekly_rotation import (
    DATA_CACHE,
    PRODUCTION_UNIVERSE,
    build_pit_universe,
    default_params,
    load_cached_dividends,
    load_frames,
    load_pit_dividends,
)

SLEEVE_NAME_TO_ARCHETYPE = {
    "base": "safe",          # 40% 保底 -> safe-asset rotation archetype
    "safe": "safe",
    "balanced": "balanced",
    "momentum": "momentum",
    "defensive": "defensive",
    "sprint": "sprint",      # 10% 冲刺 -> new sprint archetype
}


def parse_weights(text: str) -> Tuple[float, ...]:
    parts = [float(x) for x in text.replace("，", ",").split(",")]
    if len(parts) < 2:
        raise SystemExit("--weights must be comma-separated floats, e.g. 40,30,20,10")
    return tuple(parts)


def build_sleeves(names: List[str], weights: Tuple[float, ...]) -> List[SleeveSpec]:
    if len(names) != len(weights):
        raise SystemExit("--sleeves and --weights must have the same length")
    sleeves = []
    for name, w in zip(names, weights):
        if name not in SLEEVE_NAME_TO_ARCHETYPE:
            raise SystemExit(f"unknown sleeve {name!r}; choose from {sorted(SLEEVE_NAME_TO_ARCHETYPE)}")
        sleeves.append(SleeveSpec(name=name, weight=w, archetype=SLEEVE_NAME_TO_ARCHETYPE[name]))
    return sleeves


def main() -> int:
    parser = argparse.ArgumentParser(description="Four-sleeve 40/30/20/10 portfolio on the weekly-rotation engine")
    parser.add_argument("--pit", action="store_true", help="survivorship-free PIT universe (honest evidence run)")
    parser.add_argument("--universe", nargs="*", default=None, help="explicit symbol list (default: production universe)")
    parser.add_argument("--sleeves", nargs="*", default=["base", "balanced", "momentum", "sprint"])
    parser.add_argument("--weights", default="40,30,20,10", help="capital quotas, e.g. 40,30,20,10")
    parser.add_argument("--rebalance-days", type=int, default=21)
    parser.add_argument("--threshold", type=float, default=0.03, help="max weight deviation before rebalancing")
    parser.add_argument("--cost-rate", type=float, default=0.0017, help="sleeve-level flow cost per unit")
    parser.add_argument("--start", default=None)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "reports" / "sleeve_portfolio")
    parser.add_argument("--dynamic-stops", action="store_true", help="enable bottom-up ATR stop bands on every sleeve")
    parser.add_argument("--stops-atr-sl-mult", type=float, default=None)
    parser.add_argument("--stops-atr-tp-mult", type=float, default=None)
    parser.add_argument("--balanced-max-ret", type=float, default=0.0,
                        help="MAX-effect factor weight on the balanced sleeve (0 = off)")
    parser.add_argument("--balanced-roc20", type=float, default=0.10,
                        help="qlib Alpha158 ROC20 factor weight on the balanced "
                             "sleeve (0.10 = PIT-gate-validated default; 0 = off)")
    parser.add_argument("--balanced-sector-mom", type=float, default=0.15,
                        help="sector-momentum weight on the balanced sleeve "
                             "(0.15 = PIT-gate-validated default; 0 = off)")
    parser.add_argument("--momentum-sector-mom", type=float, default=0.15,
                        help="sector-momentum weight on the momentum sleeve "
                             "(0.15 = PIT-gate-validated default; 0 = off)")
    parser.add_argument("--sprint-sector-mom", type=float, default=0.25,
                        help="sector-momentum weight on the sprint sleeve "
                             "(0.25 = PIT-gate-validated default; 0 = off)")
    parser.add_argument("--sector-momentum-weight", type=float, default=0.0,
                        help="sector-momentum tilt weight on every sleeve (0 = off)")
    parser.add_argument("--fundamental-factors", default="gp_margin:0.10,yoy_ni:0.05,np_margin:0.05",
                        help="comma-separated NAME:WEIGHT pairs, e.g. "
                             "'gp_margin:0.10,yoy_ni:0.05' (PIT fundamentals, "
                             "uncovered names fall back to the median; "
                             "default = PIT-gate-validated; '' disables)")
    parser.add_argument("--fundamental-top-n", type=int, default=600,
                        help="fundamentals coverage: most-liquid N names "
                             "(600 = PIT-validated; wider coverage was "
                             "rejected by gates #20/#21)")
    parser.add_argument("--fundamental-source", default="annual",
                        choices=["auto", "annual", "quarterly"],
                        help="fundamentals cache to use (annual = PIT-gate-validated "
                             "default; auto prefers quarterly when complete)")
    parser.add_argument("--persist-rank-floor", type=int, default=12,
                        help="hold_persistent rank floor for every sleeve "
                             "(sleeve-layer default 12 = PIT-gate-validated; "
                             "pass 8 to match the single-book production base)")
    parser.add_argument("--sleeve-sprint-min-adv", type=float, default=None,
                        help="min_adv override for the sprint sleeve (e.g. 50000000)")
    parser.add_argument("--disable-short-sleeve", action="store_true",
                        help="turn off the sleeve-layer short overlay "
                             "(default on = PIT-gate-validated; engine "
                             "single-book default stays off)")
    parser.add_argument("--sleeve-factor", action="append", default=[],
                        metavar="SLEEVE=FACTOR=WEIGHT",
                        help="mount an open-source factor on a sleeve, repeatable, "
                             "e.g. --sleeve-factor balanced=roc20=0.10")
    parser.add_argument("--sleeve-fundamental", action="append", default=[],
                        metavar="SLEEVE=FACTOR=WEIGHT",
                        help="mount a fundamental factor on a sleeve, repeatable, "
                             "e.g. --sleeve-fundamental sprint=yoy_ni=0.10")
    parser.add_argument("--max-short-exposure", type=float, default=0.20)
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
        universe = args.universe or sorted(set(PRODUCTION_UNIVERSE))
        frames = load_frames(DATA_CACHE, universe)
    if len(frames) < 10:
        print(f"ERROR: only {len(frames)} symbols available in {DATA_CACHE}", file=sys.stderr)
        return 1

    params = default_params()
    params.start_date = args.start or params.start_date
    params.persist_rank_floor = args.persist_rank_floor
    if not args.disable_short_sleeve:
        params.enable_short_sleeve = True
        params.max_short_exposure = args.max_short_exposure
    if args.dynamic_stops:
        params.dynamic_stops = True
        if args.stops_atr_sl_mult is not None:
            params.stops_atr_sl_mult = args.stops_atr_sl_mult
        if args.stops_atr_tp_mult is not None:
            params.stops_atr_tp_mult = args.stops_atr_tp_mult
    div_cash = load_cached_dividends(sorted(frames)) if args.pit else load_pit_dividends(sorted(set(PRODUCTION_UNIVERSE)) if not args.universe else args.universe)

    sleeves = build_sleeves(args.sleeves, parse_weights(args.weights))
    if args.balanced_max_ret > 0:
        for i, sleeve in enumerate(sleeves):
            if sleeve.archetype == "balanced":
                overrides = dict(sleeve.overrides)
                overrides["max_ret_weight"] = args.balanced_max_ret
                sleeves[i] = SleeveSpec(name=sleeve.name, weight=sleeve.weight,
                                        archetype=sleeve.archetype, overrides=overrides)
                break
    if args.balanced_roc20 > 0:
        for i, sleeve in enumerate(sleeves):
            if sleeve.archetype == "balanced":
                overrides = dict(sleeve.overrides)
                extra = dict(overrides.get("extra_factor_weights", {}))
                extra["roc20"] = args.balanced_roc20
                overrides["extra_factor_weights"] = extra
                sleeves[i] = SleeveSpec(name=sleeve.name, weight=sleeve.weight,
                                        archetype=sleeve.archetype, overrides=overrides)
                break
    if args.balanced_sector_mom > 0:
        for i, sleeve in enumerate(sleeves):
            if sleeve.archetype == "balanced":
                overrides = dict(sleeve.overrides)
                overrides["sector_momentum_weight"] = args.balanced_sector_mom
                sleeves[i] = SleeveSpec(name=sleeve.name, weight=sleeve.weight,
                                        archetype=sleeve.archetype, overrides=overrides)
                break
    if args.momentum_sector_mom > 0:
        for i, sleeve in enumerate(sleeves):
            if sleeve.archetype == "momentum":
                overrides = dict(sleeve.overrides)
                overrides["sector_momentum_weight"] = args.momentum_sector_mom
                sleeves[i] = SleeveSpec(name=sleeve.name, weight=sleeve.weight,
                                        archetype=sleeve.archetype, overrides=overrides)
                break
    if args.sprint_sector_mom > 0:
        for i, sleeve in enumerate(sleeves):
            if sleeve.archetype == "sprint":
                overrides = dict(sleeve.overrides)
                overrides["sector_momentum_weight"] = args.sprint_sector_mom
                sleeves[i] = SleeveSpec(name=sleeve.name, weight=sleeve.weight,
                                        archetype=sleeve.archetype, overrides=overrides)
                break
    if args.sleeve_sprint_min_adv is not None:
        for i, sleeve in enumerate(sleeves):
            if sleeve.archetype == "sprint":
                overrides = dict(sleeve.overrides)
                overrides["min_adv"] = args.sleeve_sprint_min_adv
                sleeves[i] = SleeveSpec(name=sleeve.name, weight=sleeve.weight,
                                        archetype=sleeve.archetype, overrides=overrides)
                break
    for spec in args.sleeve_factor:
        parts = spec.split("=")
        if len(parts) != 3:
            raise SystemExit(f"--sleeve-factor must be SLEEVE=FACTOR=WEIGHT, got {spec!r}")
        sleeve_name, factor_name, weight_text = parts
        factor_weight = float(weight_text)
        matched = False
        for i, sleeve in enumerate(sleeves):
            if sleeve.name == sleeve_name:
                overrides = dict(sleeve.overrides)
                if factor_name == "sector_mom":
                    overrides["sector_momentum_weight"] = factor_weight
                else:
                    extra = dict(overrides.get("extra_factor_weights", {}))
                    extra[factor_name] = factor_weight
                    overrides["extra_factor_weights"] = extra
                sleeves[i] = SleeveSpec(name=sleeve.name, weight=sleeve.weight,
                                        archetype=sleeve.archetype, overrides=overrides)
                matched = True
                break
        if not matched:
            raise SystemExit(f"unknown sleeve {sleeve_name!r} in --sleeve-factor")
    for spec in args.sleeve_fundamental:
        parts = spec.split("=")
        if len(parts) != 3:
            raise SystemExit(f"--sleeve-fundamental must be SLEEVE=FACTOR=WEIGHT, got {spec!r}")
        sleeve_name, factor_name, weight_text = parts
        factor_weight = float(weight_text)
        matched = False
        for i, sleeve in enumerate(sleeves):
            if sleeve.name == sleeve_name:
                overrides = dict(sleeve.overrides)
                fund = dict(overrides.get("fundamental_factors", {}))
                fund[factor_name] = factor_weight
                overrides["fundamental_factors"] = fund
                sleeves[i] = SleeveSpec(name=sleeve.name, weight=sleeve.weight,
                                        archetype=sleeve.archetype, overrides=overrides)
                matched = True
                break
        if not matched:
            raise SystemExit(f"unknown sleeve {sleeve_name!r} in --sleeve-fundamental")
    if args.sector_momentum_weight > 0:
        params.sector_momentum_weight = args.sector_momentum_weight
    if args.fundamental_factors:
        fund: dict = {}
        for pair in args.fundamental_factors.split(","):
            name, _, weight = pair.partition(":")
            fund[name.strip()] = float(weight)
        params.fundamental_factors = fund
    params.fundamental_top_n = args.fundamental_top_n
    params.fundamental_source = args.fundamental_source
    config = SleevePortfolioConfig(
        rebalance_days=args.rebalance_days,
        threshold=args.threshold,
        cost_rate=args.cost_rate,
        start_date=args.start,
    )
    result = run_sleeve_portfolio(
        frames,
        sleeves=sleeves,
        base_params=params,
        config=config,
        dividend_cash=div_cash,
        alive_mask=alive_mask,
        benchmark_exclude=benchmark_exclude,
        regime_detector_kwargs={"bull_threshold": 0.55},
    )
    s = result["summary"]
    print("Sleeve portfolio summary:")
    print(json.dumps(
        {k: s[k] for k in [
            "observations", "start", "end", "annual_return", "monthly_avg_return",
            "annual_volatility", "sharpe", "calmar", "max_drawdown",
            "max_drawdown_recovery_days_3y", "monthly_win_rate", "quarterly_win_vs_index",
            "operation_win_rate", "position_win_rate", "avg_rebalance_turnover",
            "closed_trades_count", "final_equity", "average_exposure",
            "sleeve_rebalance_cost",
        ]},
        ensure_ascii=False, indent=2,
    ))
    print("\nPer-sleeve:")
    for name, res in result["sleeves"].items():
        ss = res["summary"]
        print(f"  {name:>9s}: ann={ss['annual_return']:.2%} sharpe={ss['sharpe']:.2f} "
              f"calmar={ss['calmar']:.2f} mdd={ss['max_drawdown']:.2%} cost={ss['total_cost_fraction']:.2%}")
    print(f"\nFinal sleeve weights: {result['meta']['final_weights']}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "sleeve_portfolio_summary.json").write_text(
        json.dumps({"summary": s, "meta": result["meta"], "sleeves": {
            name: {k: v for k, v in res["summary"].items() if k in (
                "annual_return", "sharpe", "calmar", "max_drawdown",
                "total_cost_fraction", "average_exposure", "regime_breakdown",
            )} for name, res in result["sleeves"].items()
        }}, ensure_ascii=False, indent=2), encoding="utf-8")
    result["returns"].to_csv(args.output_dir / "sleeve_portfolio_returns.csv", encoding="utf-8-sig")
    result["weights_history"].to_csv(args.output_dir / "sleeve_weights_history.csv", encoding="utf-8-sig")
    print(f"\n报告: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
