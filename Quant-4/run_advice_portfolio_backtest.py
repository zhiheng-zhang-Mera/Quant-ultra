"""Standalone mount for real-universe backtesting of Quant-4 analysis advice."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from Main.advice_portfolio_backtest import DEFAULT_EXCLUDED_PREFIXES, discover_real_universe, run_advice_portfolio_backtest, write_portfolio_backtest_report


def main() -> int:
    root = Path(__file__).parent
    parser = argparse.ArgumentParser(description="真实股票池的引擎建议组合回测（不接受单标的参数）")
    parser.add_argument("--cache-dir", type=Path, default=root / "Data_Cache")
    parser.add_argument("--output-dir", type=Path, default=root / "reports" / "advice_portfolio_backtest")
    parser.add_argument("--years", type=int, default=8)
    parser.add_argument("--lookback", type=int, default=252)
    parser.add_argument("--rebalance-every", type=int, default=21)
    parser.add_argument("--fee-rate", type=float, default=0.001)
    parser.add_argument("--min-assets", type=int, default=2)
    parser.add_argument("--exclude-prefixes", default=",".join(DEFAULT_EXCLUDED_PREFIXES))
    args = parser.parse_args()
    prefixes = tuple(value.strip() for value in args.exclude_prefixes.split(",") if value.strip())
    frames, audit = discover_real_universe(args.cache_dir, min_rows=max(args.lookback * 2, 504), excluded_prefixes=prefixes)
    result = run_advice_portfolio_backtest(frames, audit, years=args.years, lookback=args.lookback, rebalance_every=args.rebalance_every, fee_rate=args.fee_rate, min_assets=max(2, args.min_assets))
    paths = write_portfolio_backtest_report(result, args.output_dir)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    for name, path in paths.items(): print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
