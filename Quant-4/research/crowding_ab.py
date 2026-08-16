"""Round-19.7: crowding factor (amount_share) A/B + orthogonality check.

amount_share = this name's turnover share of the universe cross-section
(higher = more crowded/attention-heavy), z-scored into the composite via
extra_factor_weights. Both the raw direction (prefer crowded) and the
inverted direction (avoid crowded) are tested, plus the factor's correlation
with the base composite score (orthogonality).

38-name production pool, full 2016-2026 window, same harness as the other
factor A/Bs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
Q4 = PROJECT_ROOT

sys.path.insert(0, str(PROJECT_ROOT))

from Main.factor_library import compute_factors  # noqa: E402
from Main.weekly_rotation import (  # noqa: E402
    RegimeState, RotationParams, composite_factor_scores, precompute_panels,
    rank_candidates, weekly_rotation_backtest,
)
from run_weekly_rotation import DATA_CACHE, default_params, load_cached_dividends, load_frames  # noqa: E402

SAMPLE = ["600519.SH", "000858.SZ", "601318.SH", "600036.SH", "000001.SZ", "000333.SH",
          "600900.SH", "601899.SH", "002594.SZ", "300750.SZ", "600887.SH", "601012.SH",
          "600028.SH", "600309.SH", "601857.SH", "600941.SH", "601398.SH", "601628.SH",
          "600585.SH", "601088.SH", "600690.SH", "600048.SH", "600030.SH", "601166.SH",
          "600276.SH", "000651.SZ", "000725.SZ", "002415.SZ", "002714.SZ", "300059.SZ",
          "300760.SZ", "002475.SZ", "000002.SZ", "000568.SZ", "002304.SZ", "300124.SZ",
          "688981.SH", "688111.SH"]


def orthogonality(frames) -> dict:
    """Correlation between the amount_share z-score and the base composite score
    over a sample of rebalance dates (orthogonality check)."""
    params = default_params()
    params.research_num_trials = 4
    panel = precompute_panels(frames, params)
    rows = []
    # sample recent dates where the whole 38-name pool is listed
    recent = panel.common[panel.common >= pd.Timestamp("2018-01-01")]
    dates = recent[::84][:20]
    for d in dates:
        z = panel.amount.loc[d]
        z = (z - z.mean()) / (z.std(ddof=0) if z.std(ddof=0) > 1e-12 else 1.0)
        reg = RegimeState(date=d, regime="BULL", benchmark_close=1.0,
                          benchmark_ma_fast=1.0, benchmark_ma_slow=1.0,
                          exposure=1.0, advice_zh="", advice_en="")
        comp = composite_factor_scores(panel, d, SAMPLE, reg, params)
        comp_s = pd.Series(comp)
        m = pd.concat([z.rename("amount_share"), comp_s.rename("composite")], axis=1).dropna()
        if len(m) >= 5:
            rows.append({"date": d, "corr": m["amount_share"].corr(m["composite"]),
                         "n": len(m)})
    df = pd.DataFrame(rows)
    if len(df):
        df["date"] = df["date"].astype(str)
    return {"mean_corr": float(df["corr"].mean()) if len(df) else float("nan"),
            "n_dates": len(df),
            "corr_by_date": df.to_dict("records")}


def metrics(rets: pd.DataFrame, lo: str, hi: str) -> dict:
    sub = rets.loc[(rets.index >= lo) & (rets.index <= hi)]
    r = sub["strategy_return"]
    eq = (1 + r).cumprod()
    n = len(r)
    ann = float(eq.iloc[-1] ** (252 / n) - 1) if n and eq.iloc[-1] > 0 else 0.0
    mdd = float((eq / eq.cummax() - 1).min()) if n else 0.0
    sh = float(r.mean() / r.std(ddof=1) * np.sqrt(252)) if n > 1 and r.std(ddof=1) > 0 else 0.0
    return {"ann": ann, "sharpe": sh, "mdd": mdd, "calmar": float(ann / abs(mdd)) if mdd else 0.0}


def main() -> int:
    frames = load_frames(DATA_CACHE, SAMPLE)
    div = load_cached_dividends(SAMPLE)
    ortho = orthogonality(frames)
    print("orthogonality amount_share vs base composite:", json.dumps(
        {k: v for k, v in ortho.items() if k != "corr_by_date"}, ensure_ascii=False), flush=True)

    results = {"orthogonality": ortho}
    variants = {
        "crowd_pos_w0.05": {"amount_share": 0.05},
        "crowd_pos_w0.10": {"amount_share": 0.10},
        "crowd_pos_w0.15": {"amount_share": 0.15},
        "crowd_inv_w0.05": {"amount_share_inv": 0.05},
        "crowd_inv_w0.10": {"amount_share_inv": 0.10},
        "crowd_inv_w0.15": {"amount_share_inv": 0.15},
    }
    for name, extra in variants.items():
        p = default_params()
        p.dividend_cash = div
        p.extra_factor_weights = extra
        p.research_num_trials = 4
        r = weekly_rotation_backtest(frames, p, regime_detector_kwargs={"bull_threshold": 0.55})
        rets = r["returns"]
        results[name] = {
            "extra": extra,
            "full": metrics(rets, "2016-01-01", "2026-08-13"),
            "oos": metrics(rets, "2022-01-01", "2026-08-13"),
            "trades": r["summary"]["closed_trades_count"],
        }
        print(f"{name}: full {results[name]['full']['ann']:.4f}/{results[name]['full']['sharpe']:.3f} "
              f"mdd {results[name]['full']['mdd']:.4f} | OOS {results[name]['oos']['ann']:.4f}/"
              f"{results[name]['oos']['sharpe']:.3f}", flush=True)

    p = default_params()
    p.dividend_cash = div
    p.research_num_trials = 4
    r = weekly_rotation_backtest(frames, p, regime_detector_kwargs={"bull_threshold": 0.55})
    rets = r["returns"]
    results["base"] = {
        "full": metrics(rets, "2016-01-01", "2026-08-13"),
        "oos": metrics(rets, "2022-01-01", "2026-08-13"),
        "trades": r["summary"]["closed_trades_count"],
    }
    print(f"base: full {results['base']['full']['ann']:.4f}/{results['base']['full']['sharpe']:.3f} "
          f"mdd {results['base']['full']['mdd']:.4f} | OOS {results['base']['oos']['ann']:.4f}/"
          f"{results['base']['oos']['sharpe']:.3f}", flush=True)

    out = PROJECT_ROOT / "reports" / "_iter" / "crowding_ab_20260816.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
