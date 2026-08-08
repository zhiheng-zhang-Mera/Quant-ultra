"""CLI: run the Quant-4 weekly-rotation backtest and write readable reports.

Strategy semantics:
- Signals are computed at the close of each rebalance day (weekly / every 5
  trading days, aligned to Fridays).
- Orders execute at the NEXT trading day's open (close[t] -> open[t+1]).
- Holdings are marked open-to-open between rebalances.
- A market-regime overlay (equal-weight benchmark vs 40/10-day MAs) scales
  exposure: bull = up to 1.5x financed, neutral = ~55%, bear = ~10%.
- Selection blends 5-day momentum (continuation) with 1-day reversal
  (A-share mean reversion) among liquid, trend-holding large caps.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from Main.weekly_rotation import RotationParams, build_reports, weekly_rotation_backtest

PROJECT_ROOT = Path(__file__).parent
DATA_CACHE = PROJECT_ROOT / "data_cache"

PRODUCTION_UNIVERSE = [
    "600036.SH", "600519.SH", "601318.SH", "600900.SH", "600887.SH",
    "601012.SH", "600028.SH", "600309.SH", "601857.SH", "600941.SH",
    "601398.SH", "601628.SH", "600585.SH", "601088.SH", "600690.SH",
    "600048.SH", "600030.SH", "601166.SH", "600276.SH", "601899.SH",
    "000001.SZ", "000858.SZ", "000333.SZ", "000651.SZ", "000725.SZ",
    "002594.SZ", "002415.SZ", "002714.SZ", "300750.SZ", "300059.SZ",
    "300760.SZ", "002475.SZ", "000002.SZ", "000568.SZ", "002304.SZ",
    "300124.SZ", "688981.SH", "688111.SH", "510300.SH", "510500.SH",
    "159915.SZ", "512880.SH",
    # curated liquid ETFs (broad, cross-border, and top sector themes)
    "510050.SH", "510300.SH", "510500.SH", "510880.SH", "159919.SZ",
    "159941.SZ", "513100.SH", "513500.SH", "513050.SH", "512000.SH",
    "512480.SH", "512660.SH", "512690.SH", "515030.SH", "515880.SH",
    "588000.SH", "159949.SZ",
]


def load_frames(cache_dir: Path, symbols: list[str]) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        path = cache_dir / f"{symbol}_history.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.tz_localize(None)
        df = df.dropna(subset=["date"]).sort_values("date").drop_duplicates("date", keep="last").set_index("date")
        frames[symbol] = df[["open", "high", "low", "close", "volume", "amount"]].astype(float)
    return frames


def default_params() -> RotationParams:
    return RotationParams(
        start_date="2016-01-01",
        momentum_windows=(5,),
        momentum_weights=(1.0,),
        reversal_1d_weight=3.0,
        reversal_window=1,
        top_n=2,
        max_etf_positions=1,
        regime_ma=40,
        regime_ma_fast=10,
        bull_exposure=1.0,
        bear_exposure=0.10,
        bull_leverage=1.5,
        max_gross_exposure=1.5,
        per_position_cap=0.75,
        leverage_annual_cost=0.06,
        hold_persistent=True,
        trend_filter_long=60,
        bull_only_trading=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Quant-4 weekly-rotation backtest")
    parser.add_argument("--start", default="2016-01-01")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "reports" / "weekly_rotation")
    parser.add_argument("--universe", nargs="*", default=None)
    args = parser.parse_args()

    universe = args.universe or PRODUCTION_UNIVERSE
    frames = load_frames(DATA_CACHE, universe)
    if len(frames) < 10:
        print(f"ERROR: only {len(frames)} symbols available in {DATA_CACHE}", file=sys.stderr)
        return 1
    params = default_params()
    params.start_date = args.start
    result = weekly_rotation_backtest(frames, params)
    summary = result["summary"]
    paths = build_reports(result, args.output_dir)
    print(json.dumps({k: summary[k] for k in ["observations", "start", "end", "annual_return", "monthly_avg_return", "monthly_win_rate", "operation_win_rate", "position_win_rate", "closed_trades_count", "annual_volatility", "sharpe", "max_drawdown", "final_equity", "average_exposure"]}, ensure_ascii=False, indent=2))
    print("\n牛熊市建议:")
    for regime in ("BULL", "NEUTRAL", "BEAR"):
        info = summary["regime_breakdown"].get(regime)
        if info:
            print(f"  {regime}: {info['days']} 天,累计 {info['cum_return']:.2%}")
    # ML addition gate: federated transfer learning + RL, evidence-based decision
    try:
        from Main.ml_gate import run_ml_gate
        close_panel = pd.DataFrame({s: frames[s]["close"] for s in frames}).sort_index()
        us_path = DATA_CACHE / "us_^GSPC_history.parquet"
        us_close = None
        if us_path.exists():
            us = pd.read_parquet(us_path)
            us["date"] = pd.to_datetime(us["date"], errors="coerce").dt.tz_localize(None)
            us = us.dropna(subset=["date"]).sort_values("date").set_index("date")
            us_close = us["close"].astype(float)
        gate = run_ml_gate(close_panel, us_close, len(frames), int(summary["observations"]), PROJECT_ROOT / "reports" / "ml_gate")
        print("\nML 新增模块门控评估:")
        print(f"  联邦迁移学习(美股): {gate['federated_transfer_decision']} — {gate['federated_transfer_rationale'][:80]}...")
        print(f"  强化学习: {gate['reinforcement_learning_decision']} — {gate['reinforcement_learning']['decision_rationale'][:80]}...")
        print(f"  硬件加速自检: {gate['hardware_acceleration']['self_check']}")
    except Exception as exc:
        print(f"\nML gate skipped: {exc}")
    print(f"\n报告: {paths['report']}")
    print(f"收益明细: {paths['returns']}")
    print(f"月度明细: {paths['monthly']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
