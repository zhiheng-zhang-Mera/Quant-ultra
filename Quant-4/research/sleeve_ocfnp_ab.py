"""Round-19.7: sleeve-level validation of the ocf_np cash-flow-quality factor.

Production default is the sleeve 40/30/20/10 (R19). The single-book full-pool
evidence passes decisively (ann 6.54->8.8-9.4%, calmar 0.64->1.16-1.35, OOS
8.35->10.5-10.9%, stable w=0.03-0.10). Adoption requires the sleeve-level A/B:
the sleeve books mount fundamental_factors (gp_margin/yoy_ni/np_margin, data-less
until now) - add ocf_np:0.05 to every book and compare the 40/30/20/10 blend
against the current production sleeve (same books, no ocf_np).

Both runs are full-pool PIT (5478 names), ~14 min each (4 books sequential).

Output: reports/_iter/sleeve_ocfnp_ab_fullpool_20260816.json
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
Q4 = PROJECT_ROOT

sys.path.insert(0, str(PROJECT_ROOT))

from Main.sleeve_allocation import (  # noqa: E402
    SleevePortfolioConfig, SleeveSpec, run_sleeve_portfolio,
)
from run_weekly_rotation import (  # noqa: E402
    DATA_CACHE, build_pit_universe, default_params, load_cached_dividends,
)

OUT_DIR = PROJECT_ROOT / "reports" / "_iter"


def run_one(frames, alive, div, extra_fund: dict, tag: str) -> dict:
    params = default_params()
    params.research_num_trials = 30
    params.persist_rank_floor = 12
    params.enable_short_sleeve = True
    params.max_short_exposure = 0.20
    params.fundamental_factors = {"gp_margin": 0.10, "yoy_ni": 0.05, "np_margin": 0.05, **extra_fund}
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
    from Main.pit_universe import ETF_UNIVERSE

    t0 = time.time()
    r = run_sleeve_portfolio(
        frames, sleeves=sleeves, base_params=params, config=config,
        dividend_cash=div, alive_mask=alive, benchmark_exclude=tuple(ETF_UNIVERSE),
        regime_detector_kwargs={"bull_threshold": 0.55},
    )
    s = r["summary"]
    rets = r["returns"]
    oos = rets.loc[rets.index >= "2022-01-01"]
    eq = (1 + oos["strategy_return"]).cumprod()
    n = len(oos)
    oos_ann = float(eq.iloc[-1] ** (252 / n) - 1) if n and eq.iloc[-1] > 0 else 0.0
    oos_sh = float(oos["strategy_return"].mean() / oos["strategy_return"].std(ddof=1) * np.sqrt(252)) if n > 1 else 0.0
    rec = {
        "tag": tag,
        "annual_return": s["annual_return"], "sharpe": s["sharpe"],
        "calmar": s["calmar"], "max_drawdown": s["max_drawdown"],
        "recovery_3y": s.get("max_drawdown_recovery_days_3y"),
        "monthly_win_rate": s.get("monthly_win_rate"),
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
          f"| OOS {oos_ann:.4f}/{oos_sh:.3f} | {rec['wall_seconds']}s", flush=True)
    rets.to_csv(OUT_DIR / f"sleeve_ocfnp_returns_{tag}.csv", encoding="utf-8-sig")
    return rec


def main() -> int:
    pit = build_pit_universe(DATA_CACHE)
    frames = pit["frames"]
    alive = pit["alive_mask"]
    div = load_cached_dividends(list(frames))
    print(f"PIT pool: {len(frames)} symbols", flush=True)
    results = {
        "base_sleeve": run_one(frames, alive, div, {}, "base_sleeve"),
        "sleeve_ocfnp": run_one(frames, alive, div, {"ocf_np": 0.05}, "sleeve_ocfnp"),
    }
    out = OUT_DIR / "sleeve_ocfnp_ab_fullpool_20260816.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
