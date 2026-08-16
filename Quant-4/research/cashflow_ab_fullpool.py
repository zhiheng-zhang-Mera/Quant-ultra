"""Round-19.7: cash-flow-quality factor (ocf_np) A/B on the FULL PIT pool.

ocf_np = operating-cash-flow / net profit (accrual-quality proxy), PIT via the
fundamental framework (pub_date = announcement date, forward-filled). Coverage
= most-liquid ~600 names (the framework's PIT-validated coverage); uncovered
names fall back to the cross-sectional median inside the composite.

The 38-name pool gave misleading crowding results (R19.7), so this verdict
comes from the full PIT pool (5478 names) with warm-fold slicing.

Output: reports/_iter/cashflow_ab_fullpool_20260816.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
Q4 = PROJECT_ROOT

sys.path.insert(0, str(PROJECT_ROOT))

from Main.weekly_rotation import weekly_rotation_backtest  # noqa: E402
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


def fold_metrics(returns: pd.DataFrame, start: str, end: str) -> dict:
    sub = returns.loc[(returns.index >= start) & (returns.index <= end)]
    r = sub["strategy_return"]
    eq = (1 + r).cumprod()
    n = len(r)
    ann = float(eq.iloc[-1] ** (252 / n) - 1) if n and eq.iloc[-1] > 0 else 0.0
    mdd = float((eq / eq.cummax() - 1).min()) if n else 0.0
    sh = float(r.mean() / r.std(ddof=1) * np.sqrt(252)) if n > 1 and r.std(ddof=1) > 0 else 0.0
    return {"ann": ann, "sharpe": sh, "calmar": float(ann / abs(mdd)) if mdd else 0.0, "mdd": mdd}


def oos_metrics(returns: pd.DataFrame) -> dict:
    oos = returns.loc[returns.index >= "2022-01-01"]
    r = oos["strategy_return"]
    eq = (1 + r).cumprod()
    n = len(r)
    ann = float(eq.iloc[-1] ** (252 / n) - 1) if n and eq.iloc[-1] > 0 else 0.0
    mdd = float((eq / eq.cummax() - 1).min()) if n else 0.0
    sh = float(r.mean() / r.std(ddof=1) * np.sqrt(252)) if n > 1 and r.std(ddof=1) > 0 else 0.0
    return {"oos_ann": ann, "oos_sharpe": sh, "oos_mdd": mdd}


def main() -> int:
    pit = build_pit_universe(DATA_CACHE)
    frames = pit["frames"]
    alive = pit["alive_mask"]
    div = load_cached_dividends(list(frames))
    print(f"PIT pool: {len(frames)} symbols", flush=True)

    results = {}
    for w in (0.05, 0.10, 0.20):
        p = default_params()
        p.alive_mask = alive
        p.dividend_cash = div
        p.fundamental_factors = {"ocf_np": w}
        p.fundamental_top_n = 600
        p.fundamental_source = "annual"  # auto prefers quarterly when complete
        p.research_num_trials = 4
        t0 = time.time()
        r = weekly_rotation_backtest(frames, p, regime_detector_kwargs={"bull_threshold": 0.55})
        s = r["summary"]
        rets = r["returns"]
        name = f"ocf_np_w{w:g}"
        results[name] = {
            "weight": w,
            "annual_return": s["annual_return"], "sharpe": s["sharpe"],
            "calmar": s["calmar"], "max_drawdown": s["max_drawdown"],
            "recovery_3y": s.get("max_drawdown_recovery_days_3y"),
            "oos": oos_metrics(rets),
            "folds": {f"{a}~{b}": fold_metrics(rets, a, b) for a, b in FOLDS},
            "wall_seconds": round(time.time() - t0, 1),
        }
        rets.to_csv(OUT_DIR / f"cashflow_returns_{name}.csv", encoding="utf-8-sig")
        print(f"w={w}: ann={s['annual_return']:.4f} sh={s['sharpe']:.3f} "
              f"calmar={s['calmar']:.3f} mdd={s['max_drawdown']:.4f} "
              f"| OOS {results[name]['oos']['oos_ann']:.4f}/{results[name]['oos']['oos_sharpe']:.3f} "
              f"| {results[name]['wall_seconds']}s", flush=True)

    p = default_params()
    p.alive_mask = alive
    p.dividend_cash = div
    p.research_num_trials = 4
    r = weekly_rotation_backtest(frames, p, regime_detector_kwargs={"bull_threshold": 0.55})
    s = r["summary"]
    rets = r["returns"]
    results["base"] = {
        "annual_return": s["annual_return"], "sharpe": s["sharpe"],
        "calmar": s["calmar"], "max_drawdown": s["max_drawdown"],
        "recovery_3y": s.get("max_drawdown_recovery_days_3y"),
        "oos": oos_metrics(rets),
        "folds": {f"{a}~{b}": fold_metrics(rets, a, b) for a, b in FOLDS},
    }
    print(f"base: ann={s['annual_return']:.4f} sh={s['sharpe']:.3f} "
          f"calmar={s['calmar']:.3f} mdd={s['max_drawdown']:.4f} "
          f"| OOS {results['base']['oos']['oos_ann']:.4f}/{results['base']['oos']['oos_sharpe']:.3f}", flush=True)

    out = OUT_DIR / "cashflow_ab_fullpool_20260816.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
