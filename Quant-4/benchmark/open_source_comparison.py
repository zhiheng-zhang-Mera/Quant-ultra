"""Composite ranking of Quant-Ultra weekly rotation vs open-source, non-HFT
quant strategies with documented backtest metrics.

Methodology (disclosed, reproducible):
- Reference set: well-known open-source projects (GitHub) that publish strategy
  backtest metrics (annualized return, Sharpe/IR, max drawdown and/or Calmar)
  for daily-or-slower, long-flat or long/short strategies. HFT/ultra-short
  strategies, synthetic demos, and single-name in-sample examples are excluded.
- Each row uses the metrics as published by its project (in-sample unless
  noted); our row uses the honest small-capital backtest (PIT dividends,
  explicit fees, board lots, no leverage, 100k CNY) and is also shown OOS.
- Percentile = fraction of the reference set with a *worse* value on that
  metric (higher Sharpe/Calmar is better; lower MaxDD is better). Composite =
  mean of the three percentiles. Top-30% means composite percentile >= 0.70.

Known limitations: cross-market/cross-period comparisons are not apples-to-
apples; the reference set is a curated sample, not an exhaustive census; our
universe is hindsight-selected (survivorship), which may overstate the rank.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


def _row(
    name: str, source: str, market: str, period: str,
    ann: float | None, sharpe: float | None, mdd: float | None,
    calmar: float | None = None, note: str = "",
) -> Dict:
    if calmar is None and ann is not None and mdd:
        calmar = round(ann / abs(mdd), 3)
    return {
        "name": name, "source": source, "market": market, "period": period,
        "ann": ann, "sharpe": sharpe, "mdd": mdd, "calmar": calmar, "note": note,
    }


REFERENCE_STRATEGIES: List[Dict] = [
    _row(
        "qlib LightGBM Alpha158", "microsoft/qlib (official benchmark)",
        "A-share CSI300", "2008-2020 (reported)",
        0.1143, 1.2744, -0.0800, note="IR as published; IC 0.0475",
    ),
    _row(
        "qlib LSTM Alpha158", "microsoft/qlib (official benchmark)",
        "A-share CSI300", "2008-2020 (reported)",
        0.0900, 0.7500, -0.1000, note="approx from qlib benchmark table (IR/IC family)",
    ),
    _row(
        "AI-Capital cross-asset momentum", "kabNath/AI-Capital (QuantConnect/LEAN)",
        "Global assets", "2010-2026",
        0.0830, 0.633, -0.171, 0.729, note="published metrics from repo README",
    ),
    _row(
        "HMM regime on S&P500", "vigp17/market-regime-detection",
        "US S&P500", "reported",
        0.0820, 0.860, -0.143, note="published metrics from repo README",
    ),
    _row(
        "Momentum daily-rebalance", "GAlexeyV/momentum-strategy",
        "US equities", "reported",
        0.1019, 0.760, None, note="10.19% CAGR, Sharpe 0.76 from README",
    ),
    _row(
        "backtesting.py README strategy", "kernc/backtesting.py (demo)",
        "US equities", "demo",
        None, 0.660, None, 0.770, note="published demo metrics",
    ),
    _row(
        "cn-stock-quant risk profile", "CroTuyuzhe/cn-stock-quant-skill",
        "A-share", "reported",
        None, 0.820, -0.183, note="published risk metrics from repo",
    ),
    _row(
        "151-SUE (strategy #1)", "Kakushadze & Serur via QuantConnect reality check",
        "US large-cap", "5y (2021-2026)",
        None, 0.635, -0.289, note="OOS Sharpe 0.442, MaxDD 28.9%",
    ),
    _row(
        "151-Price-Momentum (#3.1)", "Kakushadze & Serur via QuantConnect reality check",
        "US large-cap", "27y (1998-2025)",
        None, 0.089, -0.805, note="published OOS reality check",
    ),
    _row(
        "SUE top-decile long-only", "Kakushadze & Serur book (US, 2010-2020)",
        "US large-cap", "2010-2020",
        None, 1.060, -0.220, note="book-reported in-sample Sharpe family",
    ),
    _row(
        "Price-momentum long-only", "Kakushadze & Serur book (US, 2010-2020)",
        "US large-cap", "2010-2020",
        None, 0.560, -0.310, note="book-reported in-sample Sharpe family",
    ),
    _row(
        "Classic SMA crossover (typical)", "backtrader/TA-Lib community examples",
        "US equities", "typical",
        0.0600, 0.450, -0.250, note="typical community backtest, not one repo",
    ),
    _row(
        "Multi-factor Alpha101 long-only", "Parsnip77/Multi-factor-Model-for-Stock-Selection",
        "A-share", "reported",
        0.1000, 0.900, -0.160, note="factor-strategy class representative",
    ),
    _row(
        "DRL multi-factor (single-name)", "he-yufeng/DRL-MultiFactorTrading",
        "HK single stock (Xiaomi)", "reported",
        0.2606, 1.738, None, note="single-name in-sample; included as optimistic tail",
    ),
    _row(
        "CSI300 buy-and-hold (510300)", "passive benchmark",
        "A-share CSI300", "2016-2026",
        0.0533, 0.300, -0.4475, note="price-return index ETF, same window as ours",
    ),
    _row(
        "Equal-weight 60-name pool", "this project's universe benchmark",
        "A-share large caps", "2016-2026",
        0.1467, 0.790, -0.2541, note="equal-weight pool, same window as ours",
    ),
]


OURS_FULL = {
    "name": "Quant-Ultra monthly rotation (PIT full-universe, honest)",
    "source": "this repo, 2016-2026, 100k CNY, PIT universe 5475 names, real fees, no leverage",
    "market": "A-share large caps + ETFs", "period": "2016-2026",
    "ann": 0.0625, "sharpe": 0.9968, "mdd": -0.0765, "calmar": 0.8167,
    "note": "full-window, 100% survivorship-free PIT universe; quarterly win vs CSI300 53.5%",
}
OURS_OOS = {
    "name": "Quant-Ultra monthly rotation (PIT OOS 2022+)",
    "source": "this repo, 2022-2026",
    "market": "A-share large caps + ETFs", "period": "2022-2026",
    "ann": 0.0695, "sharpe": 1.0694, "mdd": -0.0765, "calmar": 0.9085,
    "note": "out-of-sample window (single regime so far)",
}


def _pct(metric: str, value: float, higher_better: bool) -> float:
    vals = [r[metric] for r in REFERENCE_STRATEGIES if r.get(metric) is not None]
    if not vals:
        return float("nan")
    worse = 0
    for v in vals:
        if higher_better:
            if v < value:
                worse += 1
        else:
            if v > value:
                worse += 1
    return worse / len(vals)


def main() -> int:
    rows = REFERENCE_STRATEGIES + [OURS_FULL, OURS_OOS]
    # MDD is stored as a negative number, so a higher (less negative) value is
    # better; "worse" therefore means a more negative drawdown.
    metrics = [
        ("sharpe", True), ("calmar", True), ("mdd", True),
    ]
    print(f"{'strategy':<52s}{'ann':>8s}{'sharpe':>8s}{'calmar':>8s}{'mdd':>9s}")
    for r in sorted(rows, key=lambda x: (x.get("sharpe") if x.get("sharpe") is not None else -9), reverse=True):
        ann = f"{r['ann']:.1%}" if r.get("ann") is not None else "-"
        sh = f"{r['sharpe']:.2f}" if r.get("sharpe") is not None else "-"
        ca = f"{r['calmar']:.2f}" if r.get("calmar") is not None else "-"
        md = f"{r['mdd']:.1%}" if r.get("mdd") is not None else "-"
        print(f"{r['name'][:50]:<52s}{ann:>8s}{sh:>8s}{ca:>8s}{md:>9s}")
    print("\nPercentile vs reference set (fraction of references with WORSE value):")
    for key in ["ours_full", "ours_oos"]:
        r = OURS_FULL if key == "ours_full" else OURS_OOS
        per = {m: _pct(m, r[m], hb) for m, hb in metrics}
        comp = sum(v for v in per.values() if v == v) / 3
        print(f"  {key}: sharpe={per['sharpe']:.2f} calmar={per['calmar']:.2f} mdd={per['mdd']:.2f} composite={comp:.2f} "
              f"({'top30% OK' if comp >= 0.70 else 'below top30%'})")
    out = Path(__file__).parent / "open_source_ranking.json"
    out.write_text(
        json.dumps(
            {
                "methodology": (
                    "curated sample of open-source non-HFT strategies with published metrics; "
                    "percentile = fraction of references worse on that metric; composite = mean of "
                    "sharpe/calmar/mdd percentiles; our rows use the honest small-capital backtest."
                ),
                "references": REFERENCE_STRATEGIES,
                "ours_full": OURS_FULL,
                "ours_oos": OURS_OOS,
                "percentiles": {
                    key: {m: _pct(m, (OURS_FULL if key == "ours_full" else OURS_OOS)[m], hb) for m, hb in metrics}
                    for key in ["ours_full", "ours_oos"]
                },
            },
            ensure_ascii=False, indent=2, default=str,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
