"""Full-pool warm walk-forward: production vs robust on 4 contiguous folds.

One full run per profile, then slice - continuous state, no cold-start noise.
"""
import json
import sys

sys.path.insert(0, r"Quant-4")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pathlib import Path  # noqa: E402
from Main.pit_universe import ETF_UNIVERSE, alive_matrix, ever_alive_stocks, load_master_list  # noqa: E402
from Main.weekly_rotation import weekly_rotation_backtest  # noqa: E402
from reports._iter.run_sweep import load_frames  # noqa: E402
from run_weekly_rotation import default_params  # noqa: E402

DATA_CACHE = PROJECT_ROOT / "Data_Cache"
frames = load_frames(DATA_CACHE)
master = load_master_list(DATA_CACHE)
end = (pd.Timestamp.now() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
stocks = ever_alive_stocks(master, "2016-01-01", end)
expected = stocks + list(ETF_UNIVERSE)
dates = pd.DatetimeIndex(np.unique(np.concatenate([f.index.values for f in frames.values()])))
alive = alive_matrix(master, dates, stocks).reindex(columns=expected)
for etf in ETF_UNIVERSE:
    alive[etf] = True

FOLDS = [
    ("2016-01-01", "2018-12-31"),
    ("2019-01-01", "2021-12-31"),
    ("2022-01-01", "2023-12-31"),
    ("2024-01-01", "2026-12-31"),
]


def make(tag: str):
    p = default_params()  # now the adopted (2026-08-15) defaults
    if tag == "production":
        # A/B baseline = the pre-2026-08-15 defaults
        p.event_shock_threshold = 0.025
        p.event_shock_zscore = 0.0
        p.stop_loss_pct = 0.07
    p.alive_mask = alive
    p.benchmark_exclude = tuple(ETF_UNIVERSE)
    from Main.pit_dividends import load_dividend_cash

    p.dividend_cash = load_dividend_cash(DATA_CACHE / "dividends", list(frames))
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
    s = res["summary"]
    print(f"{tag}: ann={s['annual_return']*100:.2f}% sharpe={s['sharpe']:.2f} calmar={s['calmar']:.2f} mdd={s['max_drawdown']*100:.2f}%", flush=True)

rows = []
for start, end in FOLDS:
    base = fold_metrics(streams["production"], start, end)
    robust = fold_metrics(streams["robust"], start, end)
    rows.append({"fold": f"{start}~{end}", "base": base, "robust": robust})
    winner = "robust" if robust["ann"] > base["ann"] else "production"
    print(
        f"[{start}~{end}] base: ann={base['ann']*100:.2f}% sharpe={base['sharpe']:.2f} mdd={base['mdd']*100:.2f}% | "
        f"robust: ann={robust['ann']*100:.2f}% sharpe={robust['sharpe']:.2f} mdd={robust['mdd']*100:.2f}% | winner={winner}",
        flush=True,
    )

w_ann = sum(1 for r in rows if r["robust"]["ann"] > r["base"]["ann"])
w_sh = sum(1 for r in rows if r["robust"]["sharpe"] > r["base"]["sharpe"])
w_calmar = sum(1 for r in rows if r["robust"]["calmar"] > r["base"]["calmar"])
print(f"\nrobust wins {w_ann}/4 by ann, {w_sh}/4 by Sharpe, {w_calmar}/4 by Calmar", flush=True)
PROJECT_ROOT / "reports" / "_iter".mkdir(parents=True, exist_ok=True)
PROJECT_ROOT / "reports" / "_iter" / "walkforward_fullpool.json".write_text(
    json.dumps({"rows": rows, "wins": {"ann": w_ann, "sharpe": w_sh, "calmar": w_calmar}}, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
