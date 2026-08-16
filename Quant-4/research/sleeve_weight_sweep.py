"""Round-19.5 (direction 2): sleeve-weight rebalancing sweep, FULL PIT pool.

The 4 sleeve books are weight-independent: the 40/30/20/10 combination only
affects how their daily returns are blended (constant-mix). This script runs
the 4 production sleeve books ONCE on the full PIT pool (same parameter
construction as run_sleeve_portfolio.py --pit with the PIT-gate-validated
sleeve-layer defaults), then combines their cached returns under several
weight quotas and reports each blend's full-window / OOS / fold metrics.

R12 per-book full-pool evidence: base(safe) 7.86%/1.18/0.60, balanced
5.89%/0.92/0.53, momentum 5.78%/0.90/0.32, sprint 4.32%/0.57/0.20 - the
sprint is the weakest book, so rebalancing away from it is the hypothesis
to test (and toward it, from the non-PIT demo, as a falsification).

Output: reports/_iter/sleeve_weight_sweep_fullpool_20260816.json
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
    SleevePortfolioConfig, combine_sleeves, run_sleeve_portfolio,
)
from Main.weekly_rotation import summarize  # noqa: E402
from run_weekly_rotation import (  # noqa: E402
    DATA_CACHE, build_pit_universe, default_params, load_cached_dividends,
)

OUT_DIR = PROJECT_ROOT / "reports" / "_iter"
FOLDS = [
    ("2016-01-01", "2018-12-31"),
    ("2019-01-01", "2021-12-31"),
    ("2022-01-01", "2023-12-31"),
    ("2024-01-01", "2026-12-31"),
]
WEIGHT_VARIANTS = {
    "w_40_30_20_10": (0.40, 0.30, 0.20, 0.10),   # adopted production default (R12/R19)
    "w_40_35_25_00": (0.40, 0.35, 0.25, 0.00),   # drop the weakest full-pool book (sprint)
    "w_45_30_15_10": (0.45, 0.30, 0.15, 0.10),   # more of the strongest book (safe/base)
    "w_30_30_25_15": (0.30, 0.30, 0.25, 0.15),   # more sprint (falsification from non-PIT demo)
}


def fold_metrics(returns: pd.DataFrame, start: str, end: str) -> dict:
    sub = returns.loc[(returns.index >= start) & (returns.index <= end)]
    r = sub["strategy_return"]
    eq = (1 + r).cumprod()
    n = len(r)
    ann = float(eq.iloc[-1] ** (252 / n) - 1) if n and eq.iloc[-1] > 0 else 0.0
    mdd = float((eq / eq.cummax() - 1).min()) if n else 0.0
    sh = float(r.mean() / r.std(ddof=1) * np.sqrt(252)) if n > 1 and r.std(ddof=1) > 0 else 0.0
    calmar = float(ann / abs(mdd)) if mdd else 0.0
    return {"ann": ann, "sharpe": sh, "calmar": calmar, "mdd": mdd, "obs": n}


def oos_metrics(returns: pd.DataFrame) -> dict:
    oos = returns.loc[returns.index >= "2022-01-01"]
    r = oos["strategy_return"]
    eq = (1 + r).cumprod()
    n = len(r)
    ann = float(eq.iloc[-1] ** (252 / n) - 1) if n and eq.iloc[-1] > 0 else 0.0
    mdd = float((eq / eq.cummax() - 1).min()) if n else 0.0
    sh = float(r.mean() / r.std(ddof=1) * np.sqrt(252)) if n > 1 and r.std(ddof=1) > 0 else 0.0
    return {"oos_ann": ann, "oos_sharpe": sh, "oos_mdd": mdd, "oos_calmar": float(ann / abs(mdd)) if mdd else 0.0}


def main() -> int:
    pit = build_pit_universe(DATA_CACHE)
    frames = pit["frames"]
    alive = pit["alive_mask"]
    from Main.pit_universe import ETF_UNIVERSE

    div = load_cached_dividends(list(frames))

    # ---- production sleeve-layer base params (mirrors run_sleeve_portfolio.py defaults) ----
    params = default_params()
    params.research_num_trials = 88          # R12 sleeve run registered 88 trials
    params.persist_rank_floor = 12
    params.enable_short_sleeve = True
    params.max_short_exposure = 0.20
    params.fundamental_factors = {"gp_margin": 0.10, "yoy_ni": 0.05, "np_margin": 0.05}
    params.fundamental_top_n = 600
    params.fundamental_source = "annual"

    config = SleevePortfolioConfig(rebalance_days=21, threshold=0.03, cost_rate=0.0017)
    sleeves = [
        ("base", "safe", {}),
        ("balanced", "balanced", {"extra_factor_weights": {"roc20": 0.10}, "sector_momentum_weight": 0.15}),
        ("momentum", "momentum", {"sector_momentum_weight": 0.15}),
        ("sprint", "sprint", {"sector_momentum_weight": 0.25}),
    ]
    from Main.sleeve_allocation import SleeveSpec

    # ---- reuse cached book results if present (books are weight-independent) ----
    cached = {name: (OUT_DIR / f"sleeve_weight_book_{name}_returns.csv",
                     OUT_DIR / f"sleeve_weight_book_{name}_regimes.csv") for name, _, _ in sleeves}
    if all(r.exists() and g.exists() for r, g in cached.values()):
        books = {}
        for name, (rp, gp) in cached.items():
            rets = pd.read_csv(rp, parse_dates=["date"]).set_index("date").sort_index()
            rets = rets[~rets.index.duplicated(keep="last")]
            regs = pd.read_csv(gp, parse_dates=["date"]).set_index("date").sort_index()
            regs = regs[~regs.index.duplicated(keep="last")]
            books[name] = {"returns": rets, "regimes": regs, "summary": {}, "closed_trades": None}
        print("reused cached book results", flush=True)
    else:
        spec = [SleeveSpec(name=n, weight=0.25, archetype=a, overrides=ov) for n, a, ov in sleeves]
        t0 = time.time()
        r = run_sleeve_portfolio(
            frames, sleeves=spec, base_params=params, config=config,
            dividend_cash=div, alive_mask=alive, benchmark_exclude=tuple(ETF_UNIVERSE),
            regime_detector_kwargs={"bull_threshold": 0.55},
        )
        print(f"books run in {time.time() - t0:.0f}s", flush=True)
        books = r["sleeves"]
        for name, res in books.items():
            ss = res["summary"]
            print(f"  book {name}: ann={ss['annual_return']:.4f} sh={ss['sharpe']:.3f} "
                  f"calmar={ss['calmar']:.3f} mdd={ss['max_drawdown']:.4f}", flush=True)
            res["returns"].to_csv(OUT_DIR / f"sleeve_weight_book_{name}_returns.csv", encoding="utf-8-sig")
            res["regimes"].to_csv(OUT_DIR / f"sleeve_weight_book_{name}_regimes.csv", encoding="utf-8-sig")
        print("book returns cached to disk", flush=True)

    R = pd.DataFrame({name: res["returns"]["strategy_return"] for name, res in books.items()}).dropna()
    bench = pd.DataFrame({name: res["returns"]["benchmark_return"] for name, res in books.items()}).dropna()
    expo = pd.DataFrame({name: res["returns"]["gross_exposure"] for name, res in books.items()}).dropna()
    first_regimes = books["base"]["regimes"]
    trade_frames = [res.get("closed_trades") for res in books.values() if res.get("closed_trades") is not None and len(res.get("closed_trades"))]
    all_trades = pd.concat(trade_frames, ignore_index=True) if trade_frames else []
    if isinstance(all_trades, pd.DataFrame) and len(all_trades):
        all_trades = all_trades.to_dict("records")

    index_returns = None
    if "510300.SH" in frames:
        index_close = frames["510300.SH"]["close"].reindex(R.index).ffill()
        index_returns = index_close.pct_change(fill_method=None)

    results = {}
    for name, weights in WEIGHT_VARIANTS.items():
        combined, w_hist, meta = combine_sleeves(R, list(weights), 21, 0.03, 0.0017)
        ws = float(np.asarray(weights, dtype=float).sum())
        bw = np.asarray(weights, dtype=float) / ws
        cf = pd.DataFrame(index=combined.index)
        cf["strategy_return"] = combined["strategy_return"]
        cf["benchmark_return"] = (bench * bw).sum(axis=1)
        cf["gross_exposure"] = (expo * bw).sum(axis=1)
        cf["turnover"] = combined["turnover"]
        cf["cost"] = combined["cost"]
        cf["regime"] = books["base"]["returns"]["regime"].reindex(combined.index)
        s = summarize(cf, first_regimes, params, all_trades, index_returns)
        s["sleeve_rebalance_cost"] = meta["rebalance_cost"]
        rec = {
            "weights": weights,
            "annual_return": s["annual_return"], "sharpe": s["sharpe"],
            "calmar": s["calmar"], "max_drawdown": s["max_drawdown"],
            "recovery_3y": s.get("max_drawdown_recovery_days_3y"),
            "monthly_win_rate": s.get("monthly_win_rate"),
            "oos": oos_metrics(cf),
            "folds": {f"{a}~{b}": fold_metrics(cf, a, b) for a, b in FOLDS},
            "sleeve_rebalance_cost": meta["rebalance_cost"],
        }
        results[name] = rec
        print(f"{name}: ann={rec['annual_return']:.4f} sh={rec['sharpe']:.3f} "
              f"calmar={rec['calmar']:.3f} mdd={rec['max_drawdown']:.4f} rec3y={rec['recovery_3y']} "
              f"| OOS {rec['oos']['oos_ann']:.4f}/{rec['oos']['oos_sharpe']:.3f}", flush=True)

    out = OUT_DIR / "sleeve_weight_sweep_fullpool_20260816.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
