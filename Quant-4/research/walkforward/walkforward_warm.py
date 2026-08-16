"""Warm walk-forward fold evaluation (continuous state, no cold-start noise).

Each profile runs ONCE over 2016-2026 with fixed parameters; fold metrics are
measured on slices of the same continuous stream, exactly how the engine
operates in production. This is the honest robustness check for the profiles.
"""
from __future__ import annotations

import json
import sys

sys.path.insert(0, r"Quant-4")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pathlib import Path  # noqa: E402
from reports._iter.run_sweep import default_params, load_frames  # noqa: E402
from Main.weekly_rotation import weekly_rotation_backtest  # noqa: E402

frames = load_frames(PROJECT_ROOT / "Data_Cache")

FOLDS = [
    ("2016-01-01", "2018-12-31"),
    ("2019-01-01", "2021-12-31"),
    ("2022-01-01", "2023-12-31"),
    ("2024-01-01", "2026-12-31"),
]


def make(tag: str):
    p = default_params()
    if tag == "robust":
        p.event_shock_threshold = 0.03
        p.event_shock_zscore = 3.0
        p.stop_loss_pct = 0.08
    return p


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


streams = {}
for tag in ("production", "robust"):
    res = weekly_rotation_backtest(frames, make(tag), regime_detector_kwargs={"bull_threshold": 0.55})
    streams[tag] = res["returns"]

rows = []
for start, end in FOLDS:
    base = fold_metrics(streams["production"], start, end)
    robust = fold_metrics(streams["robust"], start, end)
    rows.append({"fold": f"{start}~{end}", "base": base, "robust": robust})
    winner = "robust" if robust["ann"] > base["ann"] else "production"
    print(
        f"[{start}~{end}] base: ann={base['ann']*100:.2f}% sharpe={base['sharpe']:.2f} "
        f"mdd={base['mdd']*100:.2f}% | robust: ann={robust['ann']*100:.2f}% sharpe={robust['sharpe']:.2f} "
        f"mdd={robust['mdd']*100:.2f}% | winner={winner}"
    )

w_ann = sum(1 for r in rows if r["robust"]["ann"] > r["base"]["ann"])
w_sh = sum(1 for r in rows if r["robust"]["sharpe"] > r["base"]["sharpe"])
w_mdd = sum(1 for r in rows if r["robust"]["mdd"] > r["base"]["mdd"])
w_calmar = sum(1 for r in rows if r["robust"]["calmar"] > r["base"]["calmar"])
print(f"\nrobust wins {w_ann}/4 by ann, {w_sh}/4 by Sharpe, {w_calmar}/4 by Calmar, {w_mdd}/4 by MDD")

PROJECT_ROOT / "reports" / "_iter".mkdir(parents=True, exist_ok=True)
PROJECT_ROOT / "reports" / "_iter" / "walkforward_warm_folds.json".write_text(
    json.dumps({"rows": rows, "summary": {"wins_ann": w_ann, "wins_sharpe": w_sh, "wins_calmar": w_calmar, "wins_mdd": w_mdd}}, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
print("wrote Quant-4/reports/_iter/walkforward_warm_folds.json")
