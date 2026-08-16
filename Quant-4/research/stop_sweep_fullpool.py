"""Round-19.5 (direction 2): stop-loss / take-profit recalibration sweep, FULL PIT pool.

The earlier re-entry rule was rejected on the full pool (reentry_sweep_fullpool_20260816.json:
all variants worse - deeper mdd, lower ann/sharpe/calmar). Next lever: the adopted
8% stop / 12% take-profit bands. R7 tested dynamic (ATR) stops and rejected them;
the fixed-band sensitivity on the honest full PIT pool has not been swept.

Same harness as the R10 recovery sweep: full PIT pool (5478), adopted defaults,
single continuous run per config (warm fold slicing below).

Output: reports/_iter/stop_sweep_fullpool_20260816.json
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
    div = load_cached_dividends(list(frames))
    print(f"PIT pool: {len(frames)} symbols", flush=True)

    variants = {
        "base_8_12": dict(stop_loss_pct=0.08, take_profit_pct=0.12),   # adopted
        "stop_6": dict(stop_loss_pct=0.06, take_profit_pct=0.12),
        "stop_10": dict(stop_loss_pct=0.10, take_profit_pct=0.12),
        "tp_15": dict(stop_loss_pct=0.08, take_profit_pct=0.15),
        "tp_10": dict(stop_loss_pct=0.08, take_profit_pct=0.10),
    }
    results = {}
    for name, ov in variants.items():
        p = default_params()
        p.alive_mask = alive
        p.dividend_cash = div
        p.research_num_trials = 4
        p = replace(p, **ov)
        t0 = time.time()
        r = weekly_rotation_backtest(frames, p, regime_detector_kwargs={"bull_threshold": 0.55})
        s = r["summary"]
        rets = r["returns"]
        rec = {
            "variant": name, "overrides": ov,
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
        rets.to_csv(OUT_DIR / f"stop_returns_{name}.csv", encoding="utf-8-sig")
        print(f"{name}: ann={s['annual_return']:.4f} sh={s['sharpe']:.3f} "
              f"calmar={s['calmar']:.3f} mdd={s['max_drawdown']:.4f} rec3y={rec['recovery_3y']} "
              f"| OOS {rec['oos']['oos_ann']:.4f}/{rec['oos']['oos_sharpe']:.3f} "
              f"| 2026-06~08 {rec['win_2026_06_08']:+.4f} | {rec['wall_seconds']}s", flush=True)

    out = OUT_DIR / "stop_sweep_fullpool_20260816.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
