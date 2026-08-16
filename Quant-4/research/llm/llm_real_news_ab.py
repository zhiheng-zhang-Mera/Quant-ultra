"""Round-19 Priority 1: LLM-sentiment A/B on the real 38-name production pool.

Same harness and window as the R16/R18 real-news revalidation
(_real_news_ab.py): 30d trailing mean sentiment panel sampled at the
engine's actual rebalance dates, research-relaxed governance, weights
0/0.05/0.10/0.20 plus the inverted sign.

Difference: the panel is built from the DeepSeek LLM sentiment scores
(llm_scores_real_38.jsonl) instead of the dictionary score_text - this is
the 8-13 hypothesis test: does REAL text make LLM sentiment produce
differentiated value vs the lexical baseline?

Panel variants:
  lex  - dictionary sentiment (reproduces R18 numbers for consistency)
  llm  - LLM sentiment, lexical fallback where LLM missing
  llm0 - LLM sentiment, missing records treated as neutral 0

Usage: python reports/_iter/_llm_real_ab.py
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
NEWS_PATH = PROJECT_ROOT / "Data_Cache" / "alternative_raw" / "sina_news_real_38.jsonl"
IMMUTABLE = PROJECT_ROOT / "Data_Cache" / "alternative_raw" / "_immutable"
SCORES_PATH = PROJECT_ROOT / "reports" / "_iter" / "llm_deepseek" / "llm_scores_real_38.jsonl"


def load_scores() -> dict:
    if not SCORES_PATH.exists():
        return {}
    scores = {}
    for line in SCORES_PATH.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        scores[row["record_id"]] = row["llm"]
    return scores


def build_panel(frames: dict, variant: str, lo_window: str = "2026-05-01") -> pd.DataFrame:
    """Real-news PIT panel: 30d trailing mean sentiment at engine rebalance
    dates inside the covered window. variant selects the sentiment source."""
    as_of = pd.Timestamp("2026-08-15T23:59:59Z")
    frame, evidence = load_contract_text(NEWS_PATH, SAMPLE, as_of, IMMUTABLE, require_real_source=True)
    frame["sentiment"] = frame["text"].map(score_text)
    if variant.startswith("llm"):
        llm = load_scores()
        frame["llm"] = frame["record_id"].map(llm)
        have = frame["llm"].notna().sum()
        print(f"  [panel {variant}] LLM coverage {have}/{len(frame)} "
              f"({have / len(frame):.1%})", flush=True)
        if variant == "llm":
            frame["sentiment"] = frame["llm"].where(frame["llm"].notna(), frame["sentiment"])
        else:  # llm0: missing -> neutral
            frame["sentiment"] = frame["llm"].fillna(0.0)
    frame["published_at"] = pd.to_datetime(frame["published_at"], utc=True)
    piv = frame.pivot_table(index=frame["published_at"].dt.normalize(), columns="symbol",
                            values="sentiment", aggfunc="mean").sort_index()
    roll = piv.rolling(30, min_periods=1).mean()
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


def main() -> int:
    frames = load_frames(DATA_CACHE, SAMPLE)
    div = load_cached_dividends(SAMPLE)

    variants = [("lex", "lexical dictionary (R18 repro)"),
                ("llm", "DeepSeek LLM, lexical fallback"),
                ("llm0", "DeepSeek LLM, missing=neutral")]
    results = {}
    for variant, label in variants:
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
            oos = rets[rets.index >= "2022-01-01"]
            n = len(oos)
            oos_ann = float((1 + oos["strategy_return"]).prod() ** (252 / n) - 1) if n else 0.0
            oos_sh = float(oos["strategy_return"].mean() / oos["strategy_return"].std(ddof=0) * np.sqrt(252)) if n > 1 else 0.0
            win = rets[rets.index >= "2026-06-01"]
            wn = len(win)
            win_ann = float((1 + win["strategy_return"]).prod() ** (252 / wn) - 1) if wn and (1 + win["strategy_return"]).prod() > 0 else 0.0
            win_sh = float(win["strategy_return"].mean() / win["strategy_return"].std(ddof=0) * np.sqrt(252)) if wn > 1 and win["strategy_return"].std(ddof=0) > 0 else 0.0
            key = f"{variant}_w{weight:g}"
            results[key] = {
                "panel": variant, "weight": weight,
                "ann": s["annual_return"], "sharpe": s["sharpe"], "calmar": s["calmar"],
                "mdd": s["max_drawdown"], "oos_ann": oos_ann, "oos_sharpe": oos_sh,
                "win_ann": win_ann, "win_sharpe": win_sh, "trades": s["closed_trades_count"],
                "gate": gov["status"], "raw_coverage": raw_cov,
            }
            print(f"  {variant} w={weight:.2f}: ann={results[key]['ann']:.2%} sh={results[key]['sharpe']:.2f} "
                  f"calmar={results[key]['calmar']:.2f} mdd={results[key]['mdd']:.2%} | "
                  f"OOS {results[key]['oos_ann']:.2%}/{results[key]['oos_sharpe']:.2f} | "
                  f"win(06-08) {results[key]['win_ann']:.2%}/{results[key]['win_sharpe']:.2f}", flush=True)
        # inverted-sign run for the LLM panel at w=0.10
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
        oos = rets[rets.index >= "2022-01-01"]
        n = len(oos)
        oos_ann = float((1 + oos["strategy_return"]).prod() ** (252 / n) - 1) if n else 0.0
        oos_sh = float(oos["strategy_return"].mean() / oos["strategy_return"].std(ddof=0) * np.sqrt(252)) if n > 1 else 0.0
        win = rets[rets.index >= "2026-06-01"]
        wn = len(win)
        win_ann = float((1 + win["strategy_return"]).prod() ** (252 / wn) - 1) if wn and (1 + win["strategy_return"]).prod() > 0 else 0.0
        win_sh = float(win["strategy_return"].mean() / win["strategy_return"].std(ddof=0) * np.sqrt(252)) if wn > 1 and win["strategy_return"].std(ddof=0) > 0 else 0.0
        key = f"{variant}_inv_w0.1"
        results[key] = {
            "panel": variant, "weight": -0.10,
            "ann": s["annual_return"], "sharpe": s["sharpe"], "calmar": s["calmar"],
            "mdd": s["max_drawdown"], "oos_ann": oos_ann, "oos_sharpe": oos_sh,
            "win_ann": win_ann, "win_sharpe": win_sh, "trades": s["closed_trades_count"],
            "gate": gov["status"], "raw_coverage": raw_cov,
        }
        print(f"  {variant} inv w=0.10: ann={results[key]['ann']:.2%} sh={results[key]['sharpe']:.2f} "
              f"calmar={results[key]['calmar']:.2f} mdd={results[key]['mdd']:.2%} | "
              f"OOS {results[key]['oos_ann']:.2%}/{results[key]['oos_sharpe']:.2f} | "
              f"win(06-08) {results[key]['win_ann']:.2%}/{results[key]['win_sharpe']:.2f}", flush=True)

    out = PROJECT_ROOT / "reports" / "_iter" / "llm_real_ab_20260816.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
