"""Standalone mount for real-universe backtesting of Quant-4 analysis advice."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pandas as pd

from Main.advice_portfolio_backtest import DEFAULT_EXCLUDED_PREFIXES, discover_real_universe, refresh_full_market_cache, run_advice_portfolio_backtest, write_portfolio_backtest_report


def main() -> int:
    root = Path(__file__).parent
    parser = argparse.ArgumentParser(description="真实股票池的引擎建议组合回测（不接受单标的参数）")
    parser.add_argument("--cache-dir", type=Path, default=root / "Data_Cache")
    parser.add_argument("--output-dir", type=Path, default=root / "reports" / "advice_portfolio_backtest")
    parser.add_argument("--years", type=int, default=8)
    parser.add_argument("--lookback", type=int, default=252)
    parser.add_argument("--rebalance-every", type=int, default=1, help="Recompute the as-of universe and rotation targets every N trading days")
    parser.add_argument("--fee-rate", type=float, default=0.001)
    parser.add_argument("--minimum-market-coverage", type=int, default=500, help="Fail closed unless this many real histories are available")
    parser.add_argument("--minimum-stock-coverage", type=int, default=1000, help="ETF histories cannot substitute for broad stock coverage")
    parser.add_argument("--cache-only", action="store_true", help="Diagnostic only: do not refresh the broad market cache")
    parser.add_argument("--download-workers", type=int, default=4)
    parser.add_argument("--source-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--max-positions", type=int, default=5)
    parser.add_argument("--min-exposure", type=float, default=0.15)
    parser.add_argument("--max-exposure", type=float, default=0.90)
    parser.add_argument("--max-holding-days", type=int, default=20)
    parser.add_argument("--harvest-cooldown-days", type=int, default=3)
    parser.add_argument("--min-trade-weight", type=float, default=0.02)
    parser.add_argument("--max-daily-turnover", type=float, default=0.25)
    parser.add_argument("--exclude-prefixes", default=",".join(DEFAULT_EXCLUDED_PREFIXES))
    args = parser.parse_args()
    prefixes = tuple(value.strip() for value in args.exclude_prefixes.split(",") if value.strip())
    refresh_audit = None
    if not args.cache_only:
        end = pd.Timestamp.now().normalize()
        start = end - pd.DateOffset(years=args.years + 2)
        refresh_audit = refresh_full_market_cache(args.cache_dir, str(start.date()), str(end.date()), args.download_workers, prefixes, args.minimum_market_coverage, args.minimum_stock_coverage, args.source_timeout_seconds)
    frames, audit = discover_real_universe(args.cache_dir, min_rows=max(args.lookback * 2, 504), excluded_prefixes=prefixes)
    stock_coverage = sum(symbol.split(".")[0].startswith(("0", "6")) and not symbol.split(".")[0].startswith(("15", "16")) for symbol in frames)
    if stock_coverage < args.minimum_stock_coverage:
        raise RuntimeError(f"full-market stock coverage requires {args.minimum_stock_coverage}; found {stock_coverage}")
    result = run_advice_portfolio_backtest(
        frames, audit, years=args.years, lookback=args.lookback,
        rebalance_every=args.rebalance_every, fee_rate=args.fee_rate,
        min_assets=max(2, args.minimum_market_coverage), max_positions=args.max_positions,
        min_exposure=args.min_exposure, max_exposure=args.max_exposure,
        max_holding_days=args.max_holding_days,
        harvest_cooldown_days=args.harvest_cooldown_days,
        min_trade_weight=args.min_trade_weight,
        max_daily_turnover=args.max_daily_turnover,
    )
    result["summary"]["full_market_refresh"] = refresh_audit
    paths = write_portfolio_backtest_report(result, args.output_dir)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    for name, path in paths.items(): print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
