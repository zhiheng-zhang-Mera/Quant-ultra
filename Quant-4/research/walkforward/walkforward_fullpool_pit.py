"""Regenerate the full-pool warm walk-forward with the CANONICAL PIT loader.

R9's walkforward_fullpool.py used the scratch loader (run_sweep.load_frames,
which globs ALL *_history.parquet = 5503 names, incl. 25 B-shares/extra
symbols not in the PIT master list), while every production number comes
from build_pit_universe (5478 names, master-list filtered). This script
re-runs the same 4-fold warm walk-forward on the canonical PIT universe so
the fold evidence is byte-comparable with the adopted production defaults.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, r"Quant-4")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from Main.pit_universe import ETF_UNIVERSE  # noqa: E402
from Main.weekly_rotation import weekly_rotation_backtest  # noqa: E402
from run_weekly_rotation import DATA_CACHE, build_pit_universe, default_params, load_cached_dividends  # noqa: E402

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


def main() -> int:
    pit = build_pit_universe(DATA_CACHE)
    frames = pit["frames"]
    alive = pit["alive_mask"]
    div = load_cached_dividends(sorted(frames))
    print(f"universe: {len(frames)} (canonical PIT master-list loader)")

    streams = {}
    for tag in ("production", "robust"):
        p = default_params()  # adopted 2026-08-15 defaults
        if tag == "production":
            # A/B baseline = pre-2026-08-15 defaults (legacy)
            p.event_shock_threshold = 0.025
            p.event_shock_zscore = 0.0
            p.stop_loss_pct = 0.07
        p.alive_mask = alive
        p.benchmark_exclude = tuple(ETF_UNIVERSE)
        p.dividend_cash = div
        res = weekly_rotation_backtest(frames, p, regime_detector_kwargs={"bull_threshold": 0.55})
        streams[tag] = res["returns"]
        s = res["summary"]
        print(f"{tag}: ann={s['annual_return']*100:.2f}% sharpe={s['sharpe']:.2f} calmar={s['calmar']:.2f} "
              f"mdd={s['max_drawdown']*100:.2f}%", flush=True)

    rows = []
    for start, end in FOLDS:
        base = fold_metrics(streams["production"], start, end)
        robust = fold_metrics(streams["robust"], start, end)
        rows.append({"fold": f"{start}~{end}", "base": base, "robust": robust})
        winner = "robust" if robust["ann"] > base["ann"] else "production"
        print(
            f"[{start}~{end}] base: ann={base['ann']*100:.2f}% sh={base['sharpe']:.2f} mdd={base['mdd']*100:.2f}% | "
            f"robust: ann={robust['ann']*100:.2f}% sh={robust['sharpe']:.2f} mdd={robust['mdd']*100:.2f}% | winner={winner}",
            flush=True,
        )

    w = {m: sum(1 for r in rows if r["robust"][m] > r["base"][m]) for m in ("ann", "sharpe", "calmar")}
    print(f"\nrobust wins {w['ann']}/4 by ann, {w['sharpe']}/4 by Sharpe, {w['calmar']}/4 by Calmar", flush=True)
    out = PROJECT_ROOT / "reports" / "_iter" / "walkforward_fullpool_pit_20260815.json"
    out.write_text(json.dumps({"rows": rows, "wins": w}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
