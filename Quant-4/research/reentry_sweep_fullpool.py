"""Round-19.5 (direction 2): earlier re-entry sweep on the FULL PIT pool.

Evidence-gated test of ``reentry_skip_confirmation_periods`` (default off):
after N consecutive non-BULL rebalance periods, the BULL gate may skip the
long confirmation MA (200d) requirement when the rest of the bull condition
holds (close>MA40, MA10>MA40, mom20>0), restoring the bull book at the risk
layers' scaled exposure.

Harness mirrors the R10 recovery sweep: full PIT pool (5478 names),
adopted production defaults, explicit fees / board lots / 100k capital,
no leverage. Each config is one continuous run, then sliced into the 4
warm folds (same methodology as R5/R9/R13) - no cold-start noise.

Output: reports/_iter/reentry_sweep_fullpool_20260816.json + returns CSVs.
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
    expected = pit["expected"]
    print(f"PIT pool: {len(frames)} symbols loaded", flush=True)
    div = load_cached_dividends(list(frames))

    results = {}
    for periods in (0, 1, 2, 3):
        p = default_params()
        p.alive_mask = alive
        p.dividend_cash = div
        p.research_num_trials = 4
        if periods > 0:
            p = replace(p, reentry_skip_confirmation_periods=periods)
        t0 = time.time()
        r = weekly_rotation_backtest(frames, p, regime_detector_kwargs={"bull_threshold": 0.55})
        s = r["summary"]
        rets = r["returns"]
        name = f"periods_{periods}"
        rec = {
            "periods": periods,
            "annual_return": s["annual_return"], "sharpe": s["sharpe"],
            "calmar": s["calmar"], "max_drawdown": s["max_drawdown"],
            "recovery_3y": s.get("max_drawdown_recovery_days_3y"),
            "monthly_win_rate": s.get("monthly_win_rate"),
            "closed_trades": s.get("closed_trades_count"),
            "total_cost_fraction": s.get("total_cost_fraction"),
            "oos": oos_metrics(rets),
            "folds": {f"{a}~{b}": fold_metrics(rets, a, b) for a, b in FOLDS},
            "wall_seconds": round(time.time() - t0, 1),
        }
        win = rets.loc["2026-06-01":"2026-08-13", "strategy_return"]
        rec["win_2026_06_08"] = float((1 + win).prod() - 1) if len(win) else 0.0
        results[name] = rec
        rets.to_csv(OUT_DIR / f"reentry_returns_{name}.csv", encoding="utf-8-sig")
        print(f"periods={periods}: ann={s['annual_return']:.4f} sh={s['sharpe']:.3f} "
              f"calmar={s['calmar']:.3f} mdd={s['max_drawdown']:.4f} rec3y={rec['recovery_3y']} "
              f"| OOS {rec['oos']['oos_ann']:.4f}/{rec['oos']['oos_sharpe']:.3f} "
              f"| 2026-06~08 {rec['win_2026_06_08']:+.4f} | {rec['wall_seconds']}s", flush=True)

    out = OUT_DIR / "reentry_sweep_fullpool_20260816.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
