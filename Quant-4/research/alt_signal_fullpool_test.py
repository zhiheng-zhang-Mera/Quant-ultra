"""Full-pool wiring verification for the alternative-signal pathway.

Exercises the PIT alternative-signal pathway end-to-end on the FULL PIT pool
(5478 names) under the adopted production defaults, using a *sentiment-shaped*
synthetic panel (event-spiked, mean-reverting, decaying — mimicking news/forum
sentiment dynamics) that is NOT a price-momentum proxy (R4 tested that shape on
the 421-name subset only).

Honesty framing: this is a WIRING TEST, not new alpha. The signal is price-
derived (the only PIT information available offline), so any lift/drag proves
the pathway (governance, exact as-of, directional response, weight sensitivity)
behaves correctly on the full pool — it does NOT validate real news/forum
sentiment, which still needs real data (blocked: no network).

Usage:
    python reports/_iter/_alt_signal_fullpool_test.py [--weights 0,0.05,0.10,0.20]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # Quant-4/
Q4 = PROJECT_ROOT

sys.path.insert(0, str(PROJECT_ROOT))

from run_weekly_rotation import DATA_CACHE, build_pit_universe, default_params, load_cached_dividends  # noqa: E402
from Main.weekly_rotation import weekly_rotation_backtest  # noqa: E402
from Main.pit_universe import ETF_UNIVERSE  # noqa: E402
from Phase_3.alternative_data_contract import SIGNAL_CONTRACT_VERSION  # noqa: E402


def build_sentiment_panel(frames: dict, rebalance_days: int = 21) -> pd.DataFrame:
    """Build a sentiment-shaped PIT panel at rebalance cadence.

    Sentiment_t = 0.7 * event_response_t + 0.3 * sentiment_{t-1}, where
    event_response = sign(1d ret) * clip(|1d ret| / (2 * 20d vol), 0, 1).
    Spikes on large moves, decays over time (mean-reverting, persistent) —
    qualitatively like news/forum sentiment after price events. Only uses
    information available at the as-of close (PIT). Rows only at rebalance
    dates; each row covers every symbol (governance: full coverage).
    """
    symbols = sorted(frames)
    closes = pd.DataFrame({s: frames[s]["close"] for s in symbols}).sort_index()
    ret1 = closes.pct_change(fill_method=None)
    vol20 = ret1.rolling(20, min_periods=10).std(ddof=0)
    denom = (2.0 * vol20).replace(0.0, np.nan)
    event = np.sign(ret1) * np.clip(ret1.abs() / denom, 0.0, 1.0)
    event = event.fillna(0.0)
    sent = event.copy() * np.nan
    prev = pd.Series(0.0, index=symbols)
    for i in range(len(event)):
        row = 0.7 * event.iloc[i].fillna(0.0) + 0.3 * prev
        sent.iloc[i] = row
        prev = row
    sent = sent.clip(-1.0, 1.0)
    # Align the as-of cadence with the engine's rebalance calendar: the engine
    # trims the common calendar to params.start_date (2016-01-01) and rebalances
    # at idx % rebalance_days == 0, so the panel rows must sit exactly on those
    # same positions or the signal would never be consumed at a rebalance.
    sent = sent.loc[sent.index >= "2016-01-01"]
    rebalance_dates = sent.index[::rebalance_days]
    rows = []
    digest = "a" * 64  # content hash placeholder for the synthetic derived panel
    for d in rebalance_dates:
        for s in symbols:
            v = float(sent.loc[d, s])
            if not np.isfinite(v):
                continue
            rows.append({
                "as_of": d,
                "symbol": s,
                "alternative_signal": v,
                "source_sha256": digest,
                "contract_version": SIGNAL_CONTRACT_VERSION,
                "source_latency_hours": 0.5,
                "fallback_used": False,
            })
    frame = pd.DataFrame(rows)
    frame["as_of"] = pd.to_datetime(frame["as_of"], utc=True).dt.tz_localize(None)
    panel = frame.pivot(index="as_of", columns="symbol", values="alternative_signal").sort_index()
    panel.attrs["provenance"] = {
        "contract_version": SIGNAL_CONTRACT_VERSION,
        "source_sha256": [digest],
        "max_source_latency_hours": 0.5,
        "fallback_used": False,
    }
    return panel


def run(frames, alive, div, weight, panel, name) -> dict:
    p = default_params()
    p.dividend_cash = div
    p.alive_mask = alive
    p.benchmark_exclude = tuple(ETF_UNIVERSE)
    p.alternative_signal_panel = panel
    p.alternative_signal_weight = weight
    p.research_num_trials = 5  # 5 registered variants in this experiment
    t0 = time.time()
    result = weekly_rotation_backtest(frames, p, regime_detector_kwargs={"bull_threshold": 0.55})
    s = result["summary"]
    returns = result["returns"]
    oos = returns[returns.index >= "2022-01-01"]
    n = len(oos)
    oos_ann = float((1 + oos["strategy_return"]).prod() ** (252 / n) - 1) if n else 0.0
    oos_sh = float(oos["strategy_return"].mean() / oos["strategy_return"].std(ddof=0) * np.sqrt(252)) if n > 1 else 0.0
    gov = result.get("alternative_signal_governance") or {}
    gov_status = gov.get("status", "n/a")
    row = {
        "name": name, "weight": weight,
        "ann": s["annual_return"], "sharpe": s["sharpe"], "calmar": s["calmar"],
        "mdd": s["max_drawdown"], "recovery_3y": s.get("max_drawdown_recovery_days_3y"),
        "oos_ann": oos_ann, "oos_sharpe": oos_sh,
        "cost": s["total_cost_fraction"], "trades": s["closed_trades_count"],
        "governance": gov_status,
    }
    print(f"{name:>10s} w={weight:.2f}: ann={row['ann']:.2%} sh={row['sharpe']:.2f} calmar={row['calmar']:.2f} "
          f"mdd={row['mdd']:.2%} | OOS {row['oos_ann']:.2%}/{row['oos_sharpe']:.2f} | gov={gov_status} | {int(time.time()-t0)}s")
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="0,0.05,0.10,0.20")
    args = parser.parse_args()
    weights = [float(x) for x in args.weights.split(",")]

    pit = build_pit_universe(DATA_CACHE)
    frames = pit["frames"]
    alive = pit["alive_mask"]
    div = load_cached_dividends(sorted(frames))
    print(f"universe: {len(frames)} names")

    panel = build_sentiment_panel(frames)
    print(f"sentiment panel: {panel.shape[0]} as-of rows x {panel.shape[1]} symbols, "
          f"range [{float(panel.min().min()):.2f}, {float(panel.max().max()):.2f}]")

    results = []
    for w in weights:
        results.append(run(frames, alive, div, w, panel, f"sent_w{w:.2f}"))
    # inverted-direction control: negative of the same signal at 0.10
    inv = panel.copy()
    inv.attrs = dict(panel.attrs)
    inv[:] = -panel
    results.append(run(frames, alive, div, 0.10, inv, "inverted_0.10"))

    out = Path(__file__).parent / "alt_signal_fullpool_20260815.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
