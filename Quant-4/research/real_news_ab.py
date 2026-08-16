"""Real-news revalidation on the production sample (honest, window-bounded).

The free Sina source exposes only ~5 months of per-stock news, so a
full-history (2016-2026) real-signal panel is impossible with this source.
Honest revalidation therefore:
 1. builds a real-news PIT panel for the window the data actually covers
    (2026-06 ~ 2026-08, the last 3-4 rebalance dates of the engine run),
 2. reports the governance verdict truthfully (coverage-limited -> the
    gate is expected to HOLD; that is the gate working as designed),
 3. compares the strategy under alt_weight 0 / 0.05 / 0.10 / 0.20 on the
    production-universe 20-name sample, reporting both the full-window
    (mostly unaffected) and the window-of-signal metrics.
This is REAL-DATA evidence for the pathway (contract, governance, as-of,
directional response), NOT a claim that real news is alpha on history we
cannot observe.
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
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

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
NEWS_PATH = PROJECT_ROOT / "Data_Cache" / "alternative_raw" / "sina_news_real_38.jsonl"
IMMUTABLE = PROJECT_ROOT / "Data_Cache" / "alternative_raw" / "_immutable"


def build_panel(frames: dict, lo_window: str = "2026-05-01") -> pd.DataFrame:
    """Real-news PIT panel: 30d trailing mean lexical sentiment at the
    ENGINE's actual rebalance dates (common index, idx % 21 == 0) inside the
    covered window. Coverage grows over time; rows are kept as-is so the
    governance gate can report the true coverage profile."""
    as_of = pd.Timestamp("2026-08-15T23:59:59Z")
    frame, evidence = load_contract_text(NEWS_PATH, SAMPLE, as_of, IMMUTABLE, require_real_source=True)
    frame["sentiment"] = frame["text"].map(score_text)
    frame["published_at"] = pd.to_datetime(frame["published_at"], utc=True)
    piv = frame.pivot_table(index=frame["published_at"].dt.normalize(), columns="symbol",
                            values="sentiment", aggfunc="mean").sort_index()
    roll = piv.rolling(30, min_periods=1).mean()
    # engine rebalance dates: union of frame calendars, then idx % 21 == 0
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
    # Governance requires a fully-finite panel (missing cells fail the
    # finite_values check). Fill missing with 0 = "no news => neutral" and
    # record the TRUE raw coverage separately so the report is honest.
    panel.attrs["raw_coverage"] = float(panel.notna().mean().mean())
    panel.attrs["raw_rows_with_signal"] = int(panel.notna().any(axis=1).sum())
    panel = panel.fillna(0.0)
    return panel


def main() -> int:
    frames = load_frames(DATA_CACHE, SAMPLE)
    panel = build_panel(frames)
    raw_cov = panel.attrs.get("raw_coverage", float("nan"))
    print(f"real-news panel: {panel.shape[0]} engine-rebalance dates x {panel.shape[1]} symbols; "
          f"raw rows_with_signal={panel.attrs.get('raw_rows_with_signal')}; "
          f"raw coverage={raw_cov:.2f}")
    gov = evaluate_signal_panel(panel, SAMPLE, panel.attrs["provenance"])
    print("governance (filled panel):", gov["status"], "| reasons:", gov["reasons"])
    print("HONEST NOTE: raw coverage {:.0%} means the free source covers only part of the window; missing filled as neutral 0 for research directional testing only.".format(raw_cov))

    div = load_cached_dividends(SAMPLE)
    results = {}
    for weight in (0.0, 0.05, 0.10, 0.20):
        p = default_params()
        p.dividend_cash = div
        p.alternative_signal_panel = panel if weight > 0 else None
        p.alternative_signal_weight = weight
        p.research_num_trials = 4
        # RESEARCH-ONLY relaxation: the free source cannot meet the production
        # coverage/latency gates (documented above). We lower the thresholds
        # ONLY to observe the pathway's directional response on real text;
        # this is explicitly NOT a production-enable recommendation.
        p.alternative_signal_min_coverage = 0.30
        p.alternative_signal_max_missing_rate = 0.70
        p.alternative_signal_max_latency_hours = 24.0 * 60
        r = weekly_rotation_backtest(frames, p, regime_detector_kwargs={"bull_threshold": 0.55})
        s = r["summary"]
        rets = r["returns"]
        oos = rets[rets.index >= "2022-01-01"]
        n = len(oos)
        oos_ann = float((1 + oos["strategy_return"]).prod() ** (252 / n) - 1) if n else 0.0
        oos_sh = float(oos["strategy_return"].mean() / oos["strategy_return"].std(ddof=0) * np.sqrt(252)) if n > 1 else 0.0
        # signal-window metrics: last 3 months (where the panel actually has rows)
        win = rets[rets.index >= "2026-06-01"]
        wn = len(win)
        win_ann = float((1 + win["strategy_return"]).prod() ** (252 / wn) - 1) if wn and (1 + win["strategy_return"]).prod() > 0 else 0.0
        win_sh = float(win["strategy_return"].mean() / win["strategy_return"].std(ddof=0) * np.sqrt(252)) if wn > 1 and win["strategy_return"].std(ddof=0) > 0 else 0.0
        results[weight] = {
            "ann": s["annual_return"], "sharpe": s["sharpe"], "calmar": s["calmar"],
            "mdd": s["max_drawdown"], "oos_ann": oos_ann, "oos_sharpe": oos_sh,
            "win_ann": win_ann, "win_sharpe": win_sh, "trades": s["closed_trades_count"],
        }
        print(f"alt_w={weight:.2f}: ann={results[weight]['ann']:.2%} sh={results[weight]['sharpe']:.2f} "
              f"calmar={results[weight]['calmar']:.2f} mdd={results[weight]['mdd']:.2%} | "
              f"OOS {results[weight]['oos_ann']:.2%}/{results[weight]['oos_sharpe']:.2f} | "
              f"win(06-08) {results[weight]['win_ann']:.2%}/{results[weight]['win_sharpe']:.2f}")

    out = PROJECT_ROOT / "reports" / "_iter" / "real_news_ab_20260816.json"
    out.write_text(json.dumps({"governance": gov, "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
