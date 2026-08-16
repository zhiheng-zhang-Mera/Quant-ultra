"""Round-19.8 step 4: sleeve A/B - ocf_np only (R19.7 adopted) vs the fully
activated fundamental set (gp_margin/yoy_ni/np_margin + ocf_np) once the
profit/balance caches exist, plus the headline production-default metrics
(annual return / sharpe / calmar / max drawdown / win rates).

Both runs: full PIT pool (5478 names), sleeve 40/30/20/10, ~14 min each.
Output: reports/_iter/sleeve_fundamentals_ab_20260816.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from Main.pit_universe import ETF_UNIVERSE  # noqa: E402
from Main.sleeve_allocation import (  # noqa: E402
    SleevePortfolioConfig, SleeveSpec, run_sleeve_portfolio,
)
from run_weekly_rotation import (  # noqa: E402
    DATA_CACHE, build_pit_universe, default_params, load_cached_dividends,
)

OUT_DIR = PROJECT_ROOT / "reports" / "_iter"


def run_one(frames, alive, div, fund: dict, tag: str) -> dict:
    params = default_params()
    params.research_num_trials = 30
    params.persist_rank_floor = 12
    params.enable_short_sleeve = True
    params.max_short_exposure = 0.20
    params.fundamental_factors = fund
    params.fundamental_top_n = 600
    params.fundamental_source = "annual"
    config = SleevePortfolioConfig(rebalance_days=21, threshold=0.03, cost_rate=0.0017)
    sleeves = [
        SleeveSpec(name="base", weight=0.40, archetype="safe", overrides={}),
        SleeveSpec(name="balanced", weight=0.30, archetype="balanced",
                   overrides={"extra_factor_weights": {"roc20": 0.10}, "sector_momentum_weight": 0.15}),
        SleeveSpec(name="momentum", weight=0.20, archetype="momentum",
                   overrides={"sector_momentum_weight": 0.15}),
        SleeveSpec(name="sprint", weight=0.10, archetype="sprint",
                   overrides={"sector_momentum_weight": 0.25}),
    ]
    t0 = time.time()
    r = run_sleeve_portfolio(frames, sleeves=sleeves, base_params=params, config=config,
                             dividend_cash=div, alive_mask=alive,
                             benchmark_exclude=tuple(ETF_UNIVERSE),
                             regime_detector_kwargs={"bull_threshold": 0.55})
    s = r["summary"]
    rets = r["returns"]
    oos = rets.loc[rets.index >= "2022-01-01"]
    eq = (1 + oos["strategy_return"]).cumprod()
    n = len(oos)
    oos_ann = float(eq.iloc[-1] ** (252 / n) - 1) if n and eq.iloc[-1] > 0 else 0.0
    oos_sh = float(oos["strategy_return"].mean() / oos["strategy_return"].std(ddof=1) * np.sqrt(252)) if n > 1 else 0.0
    rec = {
        "tag": tag, "fundamental_factors": fund,
        "annual_return": s["annual_return"], "sharpe": s["sharpe"],
        "calmar": s["calmar"], "max_drawdown": s["max_drawdown"],
        "recovery_3y": s.get("max_drawdown_recovery_days_3y"),
        "monthly_win_rate": s.get("monthly_win_rate"),
        "quarterly_win_vs_index": s.get("quarterly_win_vs_index"),
        "weekly_win_rate": s.get("weekly_win_rate"),
        "operation_win_rate": s.get("operation_win_rate"),
        "position_win_rate": s.get("position_win_rate"),
        "oos_ann": oos_ann, "oos_sharpe": oos_sh,
        "wall_seconds": round(time.time() - t0, 1),
        "per_sleeve": {name: {"ann": res["summary"]["annual_return"],
                              "sharpe": res["summary"]["sharpe"],
                              "calmar": res["summary"]["calmar"],
                              "mdd": res["summary"]["max_drawdown"]}
                       for name, res in r["sleeves"].items()},
    }
    print(f"{tag}: ann={s['annual_return']:.4f} sh={s['sharpe']:.3f} calmar={s['calmar']:.3f} "
          f"mdd={s['max_drawdown']:.4f} rec3y={rec['recovery_3y']} mwin={rec['monthly_win_rate']:.3f} "
          f"qwin={rec['quarterly_win_vs_index']:.3f} | OOS {oos_ann:.4f}/{oos_sh:.3f} "
          f"| {rec['wall_seconds']}s", flush=True)
    rets.to_csv(OUT_DIR / f"sleeve_fund_returns_{tag}.csv", encoding="utf-8-sig")
    return rec


def main() -> int:
    pit = build_pit_universe(DATA_CACHE)
    frames = pit["frames"]
    alive = pit["alive_mask"]
    div = load_cached_dividends(list(frames))
    print(f"PIT pool: {len(frames)} symbols", flush=True)
    # coverage of the fundamental caches
    from Main.fundamental_factors import load_all_fundamentals

    merged = load_all_fundamentals(top_n=None)
    covered = sum(1 for v in merged.values() if v)
    fields = set()
    for recs in merged.values():
        for rec in recs[:1]:
            fields.update(k for k in rec if k not in ("pub_date", "stat_date"))
        if len(fields) >= 8:
            break
    print(f"fundamentals cache: {covered} symbols, fields present: {sorted(fields)}", flush=True)

    results = {
        "ocf_only": run_one(frames, alive, div, {"ocf_np": 0.05}, "ocf_only"),
        "all_fund": run_one(frames, alive, div,
                            {"gp_margin": 0.10, "yoy_ni": 0.05, "np_margin": 0.05, "ocf_np": 0.05},
                            "all_fund"),
    }
    out = OUT_DIR / "sleeve_fundamentals_ab_20260816.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
