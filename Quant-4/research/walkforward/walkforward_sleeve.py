"""Sleeve portfolio warm walk-forward: single-book vs 40/30/20/10 sleeve.

Both full-pool runs (adopted defaults, 2016-01-04 ~ 2026-08-13, 2578 days)
were already executed in R12; this script only slices their daily returns
into the same 4 contiguous folds used by the R5/R9 methodology (one full
run per config, then slice - continuous state, no cold-start noise).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Q4
FOLDS = [
    ("2016-01-01", "2018-12-31"),
    ("2019-01-01", "2021-12-31"),
    ("2022-01-01", "2023-12-31"),
    ("2024-01-01", "2026-12-31"),
]


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


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
    single = load(ROOT / "reports" / "weekly_rotation_default" / "weekly_rotation_returns.csv")
    sleeve = load(ROOT / "reports" / "sleeve_portfolio_fullpool_20260815" / "sleeve_portfolio_returns.csv")
    assert single.index.equals(sleeve.index), "calendars must align"

    rows = []
    for start, end in FOLDS:
        s = fold_metrics(single, start, end)
        v = fold_metrics(sleeve, start, end)
        rows.append({"fold": f"{start}~{end}", "single": s, "sleeve": v})
        winner = "sleeve" if v["ann"] > s["ann"] else "single"
        print(
            f"[{start}~{end}] single: ann={s['ann']*100:.2f}% sh={s['sharpe']:.2f} calmar={s['calmar']:.2f} "
            f"mdd={s['mdd']*100:.2f}% | sleeve: ann={v['ann']*100:.2f}% sh={v['sharpe']:.2f} calmar={v['calmar']:.2f} "
            f"mdd={v['mdd']*100:.2f}% | winner(ann)={winner}",
            flush=True,
        )

    w = {m: sum(1 for r in rows if r["sleeve"][m] > r["single"][m]) for m in ("ann", "sharpe", "calmar")}
    w_mdd = sum(1 for r in rows if r["sleeve"]["mdd"] > r["single"]["mdd"])  # less negative = better
    print(f"\nsleeve wins {w['ann']}/4 by ann, {w['sharpe']}/4 by Sharpe, {w['calmar']}/4 by Calmar, {w_mdd}/4 by MDD")
    out = ROOT / "reports" / "_iter" / "walkforward_sleeve_fullpool_20260815.json"
    out.write_text(json.dumps({"rows": rows, "wins": {**w, "mdd": w_mdd}}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
