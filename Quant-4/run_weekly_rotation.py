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
from typing import Optional

import numpy as np
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
    # safe assets: government bonds, gold, money-market ETF (defensive hold)
    "511010.SH", "511260.SH", "518880.SH", "511880.SH",
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
        top_n=5,
        max_etf_positions=2,
        regime_ma=40,
        regime_ma_fast=10,
        regime_confirmation_ma=200,
        regime_model="logit",
        ml_bear_override=True,
        ml_bear_floor=0.30,
        ml_bear_low=0.40,
        ml_bull_high=0.60,
        bull_exposure=1.0,
        bear_exposure=1.0,
        neutral_exposure=0.0,
        bull_leverage=1.0,
        max_gross_exposure=1.0,
        per_position_cap=0.30,
        leverage_annual_cost=0.06,
        hold_persistent=False,
        max_holding_days=63,
        confirm_leverage=1.0,            # no margin/leverage for personal capital
        confirm_ml_prob=0.65,
        confirm_equity_proximity=0.97,
        enable_intraweek_stops=True,
        stop_loss_pct=0.08,
        take_profit_pct=0.06,
        trend_filter_long=60,
        bull_only_trading=False,
        require_relative_strength=False,
        signal_mode="composite",
        defensive_core=True,
        defensive_core_bull_momentum=True,
        defensive_filter=True,
        defensive_div_weight=0.0,
        max_annual_vol=0.40,
        rebalance_days=21,
        bear_no_loss=False,
        vol_target=0.20,
        vol_scale_floor=0.90,
        drawdown_guard=0.0,
        drawdown_guard_max=0.14,
        drawdown_floor=0.25,
        event_shock_threshold=0.025,
        event_shock_latch=True,
        event_shock_recovery_ma=5,
        event_shock_exposure=0.70,
        defensive_hold_assets=("511010.SH", "511260.SH", "518880.SH", "511880.SH"),
        defensive_hold_exposure=1.0,
        defensive_hold_safe_frac=0.65,
        defensive_hold_basket=(),
        safe_trend_gate=20,
        euphoria_threshold=0.10,
        benchmark_exclude=("511010.SH", "511260.SH", "518880.SH", "511880.SH"),
        capital_base=100_000.0,          # CNY, drives min-commission & board-lot model
        enable_board_lots=True,
        slippage_rate=0.0002,
    )


def load_pit_dividends(symbols: list[str]) -> Optional[pd.DataFrame]:
    """Load point-in-time dividend cash history (ex-date x symbol) from the
    local cache. Fetches from baostock on first use (network required). Never
    falls back to a static average map: no data means no dividend factor."""
    from Main.pit_dividends import fetch_dividend_history, load_dividend_cash

    cache_dir = DATA_CACHE / "dividends"
    cached = load_dividend_cash(cache_dir, symbols)
    if cached is not None and not cached.empty:
        return cached
    try:
        fetched = fetch_dividend_history(symbols, start_year=2014, cache_dir=cache_dir)
        return load_dividend_cash(cache_dir, symbols)
    except Exception as exc:
        print(f"WARNING: PIT dividend fetch failed ({exc}); running without dividend factor")
        return None


def load_cached_dividends(symbols: list[str]) -> Optional[pd.DataFrame]:
    """Cached-only PIT dividends (never triggers a network fetch)."""
    from Main.pit_dividends import load_dividend_cash

    return load_dividend_cash(DATA_CACHE / "dividends", symbols)


def build_pit_universe(cache_dir: Path) -> dict:
    """Load the survivorship-free PIT universe: ever-alive A-shares + ETFs."""
    from Main.pit_universe import (
        ETF_UNIVERSE,
        alive_matrix,
        ever_alive_stocks,
        load_master_list,
    )

    master = load_master_list(cache_dir)
    end = (pd.Timestamp.now() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    stocks = ever_alive_stocks(master, "2016-01-01", end)
    expected = stocks + list(ETF_UNIVERSE)
    frames = load_frames(cache_dir, expected)
    if not frames:
        raise RuntimeError("PIT universe: no cached frames available; run tools/download_universe.py first")
    dates = pd.DatetimeIndex(
        np.unique(np.concatenate([f.index.values for f in frames.values()]))
    )
    alive = alive_matrix(master, dates, stocks)
    alive = alive.reindex(columns=expected)
    for etf in ETF_UNIVERSE:
        alive[etf] = True
    return {"master": master, "frames": frames, "alive_mask": alive, "expected": expected}


def main() -> int:
    parser = argparse.ArgumentParser(description="Quant-4 weekly-rotation backtest")
    parser.add_argument("--start", default="2016-01-01")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "reports" / "weekly_rotation")
    parser.add_argument("--universe", nargs="*", default=None)
    parser.add_argument("--pit", action="store_true", help="survivorship-free PIT universe (ever-alive A-shares + ETFs)")
    parser.add_argument("--signal-mode", choices=["rule", "composite"], default="composite")
    args = parser.parse_args()

    universe = args.universe or PRODUCTION_UNIVERSE
    alive_mask = None
    benchmark_exclude = None
    pit_coverage = None
    if args.pit:
        pit = build_pit_universe(DATA_CACHE)
        universe = pit["expected"]
        frames = pit["frames"]
        alive_mask = pit["alive_mask"]
        # Pure-equity PIT benchmark: all ETFs (including safe assets) are
        # excluded from the equal-weight benchmark; regime/selection still see
        # them as tradeable assets.
        from Main.pit_universe import ETF_UNIVERSE as _ETFS
        benchmark_exclude = list(_ETFS)
    else:
        frames = load_frames(DATA_CACHE, universe)
    if len(frames) < 10:
        print(f"ERROR: only {len(frames)} symbols available in {DATA_CACHE}", file=sys.stderr)
        return 1
    params = default_params()
    params.start_date = args.start
    params.signal_mode = args.signal_mode
    params.alive_mask = alive_mask
    if benchmark_exclude:
        params.benchmark_exclude = tuple(benchmark_exclude)
    if args.pit:
        params.dividend_cash = load_cached_dividends(list(frames))
        if params.dividend_cash is None or params.dividend_cash.empty:
            print("WARNING: no cached PIT dividends for the PIT universe; running without dividend factor")
    else:
        params.dividend_cash = load_pit_dividends(universe)
    result = weekly_rotation_backtest(frames, params, regime_detector_kwargs={"bull_threshold": 0.55})
    summary = result["summary"]
    if args.pit:
        from Main.pit_universe import universe_coverage
        pit_coverage = universe_coverage(frames, pit["master"], "2016-01-01", str(summary.get("end")))
        result["universe_coverage"] = pit_coverage
    paths = build_reports(result, args.output_dir)
    if args.pit:
        (args.output_dir / "universe_coverage.json").write_text(
            json.dumps(pit_coverage, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print("\nPIT universe coverage:")
        print(json.dumps(pit_coverage, ensure_ascii=False, indent=2))
    print(json.dumps({k: summary[k] for k in ["observations", "start", "end", "annual_return", "monthly_avg_return", "annual_volatility", "sharpe", "calmar", "max_drawdown", "max_drawdown_recovery_days", "monthly_win_vs_index", "quarterly_win_vs_index", "monthly_win_vs_benchmark", "quarterly_win_vs_benchmark", "avg_rebalance_turnover", "operation_win_rate", "position_win_rate", "closed_trades_count", "final_equity", "average_exposure"]}, ensure_ascii=False, indent=2))
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
        gate = run_ml_gate(close_panel, us_close, len(frames), int(summary["observations"]), PROJECT_ROOT / "reports" / "ml_gate", params.rebalance_days)
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
