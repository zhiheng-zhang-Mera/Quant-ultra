"""Round-19.7: shareholder-action (buyback/insider) event factor A/B, full 2016-2026.

Text sentiment had no historical alpha (R19.6), so this round pivots to
orthogonal FACTOR signals built from the eastmoney notice corpus (73,792
records, full history):
  - shareholder_actions: net = (回购非注销 + 增持) - 减持, 90d rolling sum,
    normalized to [-1,1] (clip /3), sampled at engine rebalance dates 2016-2026.
  - earnings_guidance: net = 预增 - 预减 (sparse for the 38-name blue-chip pool),
    same construction.

Both are PIT (published_at <= decision date), event-count based, and feed the
engine's alternative_signal_panel mechanism (z-scored into the composite) -
the same harness that rejected text sentiment (R11/R16-R19.6), so the verdict
is directly comparable.

Output: reports/_iter/event_signal_ab_20260816.json
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
Q4 = PROJECT_ROOT

sys.path.insert(0, str(PROJECT_ROOT))

from Main.alternative_signal_governance import evaluate_signal_panel  # noqa: E402
from Main.weekly_rotation import weekly_rotation_backtest  # noqa: E402
from Phase_3.alternative_data_contract import SIGNAL_CONTRACT_VERSION  # noqa: E402
from run_weekly_rotation import DATA_CACHE, default_params, load_cached_dividends, load_frames  # noqa: E402

SAMPLE = ["600519.SH", "000858.SZ", "601318.SH", "600036.SH", "000001.SZ", "000333.SH",
          "600900.SH", "601899.SH", "002594.SZ", "300750.SZ", "600887.SH", "601012.SH",
          "600028.SH", "600309.SH", "601857.SH", "600941.SH", "601398.SH", "601628.SH",
          "600585.SH", "601088.SH", "600690.SH", "600048.SH", "600030.SH", "601166.SH",
          "600276.SH", "000651.SZ", "000725.SZ", "002415.SZ", "002714.SZ", "300059.SZ",
          "300760.SZ", "002475.SZ", "000002.SZ", "000568.SZ", "002304.SZ", "300124.SZ",
          "688981.SH", "688111.SH"]
NOTICE_PATH = PROJECT_ROOT / "Data_Cache" / "alternative_raw" / "eastmoney_notice_real_38.jsonl"

# title keyword rules -> event score (PIT event count signal)
RULES = {
    "shareholder_actions": [
        (re.compile(r"回购(?!.*注销)"), 1.0),   # buyback (not cancel)
        (re.compile(r"增持"), 1.0),             # insider/big-shareholder increase
        (re.compile(r"减持"), -1.0),            # decrease
    ],
    "earnings_guidance": [
        (re.compile(r"预增|扭亏|续盈|略增|减亏"), 1.0),
        (re.compile(r"预减|首亏|续亏|略减|增亏"), -1.0),
    ],
}


def load_notices() -> pd.DataFrame:
    rows = [json.loads(line) for line in NOTICE_PATH.open(encoding="utf-8") if line.strip()]
    df = pd.DataFrame(rows)
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True)
    return df


def build_panel(frames: dict, variant: str, roll_days: int = 90) -> pd.DataFrame:
    df = load_notices()
    scores = []
    for _, r in df.iterrows():
        v = 0.0
        for pat, s in RULES[variant]:
            if pat.search(r["text"]):
                v = s
                break
        if v != 0.0:
            scores.append({"symbol": r["symbol"], "day": r["published_at"].normalize(), "score": v})
    ev = pd.DataFrame(scores)
    piv = ev.pivot_table(index="day", columns="symbol", values="score", aggfunc="sum").sort_index()
    roll = piv.rolling(roll_days, min_periods=1).sum().clip(-3.0, 3.0) / 3.0  # normalize to [-1,1]
    # engine rebalance grid (2016-2026, 21-day cadence)
    common = None
    for sym, dfr in frames.items():
        idx = dfr.index
        common = idx if common is None else common.union(idx)
    common = pd.DatetimeIndex(np.unique(common.values)).sort_values()
    common = common[common >= pd.Timestamp("2016-01-01")]
    grid = [d for d in common[::21] if d <= pd.Timestamp("2026-08-13")]
    panel = pd.DataFrame(index=pd.DatetimeIndex(grid), columns=SAMPLE, dtype=float)
    for d in grid:
        hist = roll[roll.index <= pd.Timestamp(d, tz="UTC")]
        if len(hist):
            panel.loc[d] = hist.iloc[-1]
    panel = panel.clip(-1.0, 1.0)
    panel.attrs["provenance"] = {
        "contract_version": SIGNAL_CONTRACT_VERSION,
        "source_sha256": [__import__("hashlib").sha256(NOTICE_PATH.read_bytes()).hexdigest()],
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
    for variant in ("shareholder_actions", "earnings_guidance"):
        panel = build_panel(frames, variant)
        raw_cov = panel.attrs.get("raw_coverage", float("nan"))
        gov = evaluate_signal_panel(panel, SAMPLE, panel.attrs["provenance"])
        print(f"[{variant}] panel {panel.shape[0]}x{panel.shape[1]} | raw cov {raw_cov:.2f} | "
              f"rows_with_signal {panel.attrs['raw_rows_with_signal']} | gate {gov['status']}", flush=True)
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
            key = f"{variant}_w{weight:g}"
            results[key] = {
                "variant": variant, "weight": weight,
                "full": metrics(rets, "2016-01-01", "2026-08-13"),
                "oos": metrics(rets, "2022-01-01", "2026-08-13"),
                "trades": s["closed_trades_count"], "gate": gov["status"], "raw_coverage": raw_cov,
            }
            print(f"  {key}: full {results[key]['full']['ann']:.4f}/{results[key]['full']['sharpe']:.3f} "
                  f"mdd {results[key]['full']['mdd']:.4f} | OOS {results[key]['oos']['ann']:.4f}/"
                  f"{results[key]['oos']['sharpe']:.3f}", flush=True)

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

    out = PROJECT_ROOT / "reports" / "_iter" / "event_signal_ab_20260816.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
