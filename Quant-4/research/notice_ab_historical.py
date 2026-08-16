"""Round-19.6: historical notice-sentiment A/B on the FULL 2016-2026 window.

R16/R18 established that REAL NEWS sentiment has no stable directional alpha,
but the free Sina source only covers ~5 months, so the signal could only be
validated on a 3-month window. The eastmoney notice source (data.eastmoney.com/
notices, full per-stock history back to IPO) removes that constraint: this
harness validates the notice (announcement) sentiment signal on the full
10-year backtest window, sampled at every engine rebalance date.

Notices differ from news: formal company disclosures (reports, board
resolutions, shareholder changes, contracts), ~3-5 per month per name, mostly
neutral titles. The dictionary sentiment and (optionally, later) LLM sentiment
are computed on the title text. Signal windows: 30d and 90d trailing mean
(the choice is disclosed, not tuned).

Same engine harness as R16/R18: 38-name production pool, adopted defaults,
research-relaxed governance, weights 0/0.05/0.10/0.20 + inverted 0.10.

Output: reports/_iter/notice_ab_historical_20260816.json
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

from Phase_3.alternative_data import score_text  # noqa: E402
from Phase_3.alternative_data_contract import SIGNAL_CONTRACT_VERSION, load_contract_text  # noqa: E402
from Main.alternative_signal_governance import evaluate_signal_panel  # noqa: E402
from Main.weekly_rotation import weekly_rotation_backtest  # noqa: E402
from run_weekly_rotation import DATA_CACHE, default_params, load_cached_dividends, load_frames  # noqa: E402

SAMPLE = ["600519.SH", "000858.SZ", "601318.SH", "600036.SH", "000001.SZ", "000333.SH",
          "600900.SH", "601899.SH", "002594.SZ", "300750.SZ", "600887.SH", "601012.SH",
          "600028.SH", "600309.SH", "601857.SH", "600941.SH", "601398.SH", "601628.SH",
          "600585.SH", "601088.SH", "600690.SH", "600048.SH", "600030.SH", "601166.SH",
          "600276.SH", "000651.SZ", "000725.SZ", "002415.SZ", "002714.SZ", "300059.SZ",
          "300760.SZ", "002475.SZ", "000002.SZ", "000568.SZ", "002304.SZ", "300124.SZ",
          "688981.SH", "688111.SH"]
NOTICE_PATH = PROJECT_ROOT / "Data_Cache" / "alternative_raw" / "eastmoney_notice_real_38.jsonl"
IMMUTABLE = PROJECT_ROOT / "Data_Cache" / "alternative_raw" / "_immutable"


def build_panel(frames: dict, roll_days: int, lo_window: str = "2016-01-01") -> pd.DataFrame:
    """Notice PIT panel: trailing-mean dictionary sentiment at engine rebalance
    dates over the FULL backtest window (2016-2026)."""
    # the corpus was ingested just now; as_of = now keeps the PIT filter honest
    as_of = pd.Timestamp.now(tz="UTC")
    frame, evidence = load_contract_text(NOTICE_PATH, SAMPLE, as_of, IMMUTABLE, require_real_source=True)
    frame["sentiment"] = frame["text"].map(score_text)
    frame["published_at"] = pd.to_datetime(frame["published_at"], utc=True)
    piv = frame.pivot_table(index=frame["published_at"].dt.normalize(), columns="symbol",
                            values="sentiment", aggfunc="mean").sort_index()
    roll = piv.rolling(roll_days, min_periods=1).mean()
    common = None
    for sym, df in frames.items():
        idx = df.index
        common = idx if common is None else common.union(idx)
    common = pd.DatetimeIndex(np.unique(common.values)).sort_values()
    common = common[common >= pd.Timestamp("2016-01-01")]
    rebal_dates = common[::21]
    grid = [d for d in rebal_dates if d >= pd.Timestamp(lo_window) and d <= pd.Timestamp("2026-08-13")]
    panel = pd.DataFrame(index=pd.DatetimeIndex(grid), columns=SAMPLE, dtype=float)
    for d in grid:
        hist = roll[roll.index <= pd.Timestamp(d, tz="UTC")]
        if len(hist):
            panel.loc[d] = hist.iloc[-1]
    panel = panel.clip(-1.0, 1.0)
    panel.attrs["provenance"] = {
        "contract_version": SIGNAL_CONTRACT_VERSION,
        "source_sha256": [evidence["sha256"]],
        "max_source_latency_hours": 24.0,
        "fallback_used": False,
    }
    panel.attrs["raw_coverage"] = float(panel.notna().mean().mean())
    panel.attrs["raw_rows_with_signal"] = int(panel.notna().any(axis=1).sum())
    panel = panel.fillna(0.0)
    return panel


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
    results = {}
    for roll_days in (30, 90):
        panel = build_panel(frames, roll_days)
        raw_cov = panel.attrs.get("raw_coverage", float("nan"))
        gov = evaluate_signal_panel(panel, SAMPLE, panel.attrs["provenance"])
        print(f"[roll{roll_days}] panel {panel.shape[0]} rebalance dates x {panel.shape[1]} symbols | "
              f"raw cov {raw_cov:.2f} | rows_with_signal {panel.attrs['raw_rows_with_signal']} | "
              f"gate {gov['status']}", flush=True)
        for weight in (0.05, 0.10, 0.20):
            p = default_params()
            p.dividend_cash = div
            p.alternative_signal_panel = panel
            p.alternative_signal_weight = weight
            p.research_num_trials = 4
            p.alternative_signal_min_coverage = 0.30
            p.alternative_signal_max_missing_rate = 0.70
            p.alternative_signal_max_latency_hours = 24.0 * 60
            r = weekly_rotation_backtest(frames, p, regime_detector_kwargs={"bull_threshold": 0.55})
            s = r["summary"]
            rets = r["returns"]
            key = f"roll{roll_days}_w{weight:g}"
            results[key] = {
                "roll_days": roll_days, "weight": weight,
                "full": metrics(rets, "2016-01-01", "2026-08-13"),
                "oos": metrics(rets, "2022-01-01", "2026-08-13"),
                "trades": s["closed_trades_count"],
                "gate": gov["status"], "raw_coverage": raw_cov,
            }
            print(f"  {key}: full {results[key]['full']['ann']:.4f}/{results[key]['full']['sharpe']:.3f} "
                  f"mdd {results[key]['full']['mdd']:.4f} | OOS {results[key]['oos']['ann']:.4f}/"
                  f"{results[key]['oos']['sharpe']:.3f}", flush=True)
        # inverted sign at w=0.10
        p = default_params()
        p.dividend_cash = div
        p.alternative_signal_panel = -panel
        p.alternative_signal_weight = 0.10
        p.research_num_trials = 4
        p.alternative_signal_min_coverage = 0.30
        p.alternative_signal_max_missing_rate = 0.70
        p.alternative_signal_max_latency_hours = 24.0 * 60
        r = weekly_rotation_backtest(frames, p, regime_detector_kwargs={"bull_threshold": 0.55})
        s = r["summary"]
        rets = r["returns"]
        key = f"roll{roll_days}_inv_w0.1"
        results[key] = {
            "roll_days": roll_days, "weight": -0.10,
            "full": metrics(rets, "2016-01-01", "2026-08-13"),
            "oos": metrics(rets, "2022-01-01", "2026-08-13"),
            "trades": s["closed_trades_count"],
            "gate": gov["status"], "raw_coverage": raw_cov,
        }
        print(f"  {key}: full {results[key]['full']['ann']:.4f}/{results[key]['full']['sharpe']:.3f} "
              f"mdd {results[key]['full']['mdd']:.4f} | OOS {results[key]['oos']['ann']:.4f}/"
              f"{results[key]['oos']['sharpe']:.3f}", flush=True)

    # base (no signal)
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
    print(f"  base: full {results['base']['full']['ann']:.4f}/{results['base']['full']['sharpe']:.3f} "
          f"mdd {results['base']['full']['mdd']:.4f} | OOS {results['base']['oos']['ann']:.4f}/"
          f"{results['base']['oos']['sharpe']:.3f}", flush=True)

    out = PROJECT_ROOT / "reports" / "_iter" / "notice_ab_historical_20260816.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
