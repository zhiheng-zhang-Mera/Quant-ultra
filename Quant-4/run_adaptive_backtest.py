"""CLI for leakage-resistant adaptive backtesting of an A-share or ETF."""
from __future__ import annotations
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from Main.datasource_manager import FreeDataSourceManager
from Main.walk_forward_backtest import walk_forward_backtest, write_backtest_report
from analyze_cn_asset import normalize


DEFAULT_GRID = {
    "fast_window": [5, 10, 20], "slow_window": [40, 60, 120],
    "vol_window": [20, 40], "target_vol": [0.10, 0.15, 0.20],
}


def main():
    parser = argparse.ArgumentParser(description="无未来预知的走步自适应回测")
    parser.add_argument("code", help="6位股票或ETF代码")
    parser.add_argument("--kind", choices=["stock", "etf"], required=True)
    parser.add_argument("--years", type=int, default=8)
    parser.add_argument("--train-size", type=int, default=504)
    parser.add_argument("--test-size", type=int, default=63)
    parser.add_argument("--embargo", type=int, default=5)
    parser.add_argument("--fee-rate", type=float, default=0.001)
    parser.add_argument("--proxy")
    args = parser.parse_args()
    symbol = normalize(args.code, args.kind)
    end = datetime.now().date()
    start = end - timedelta(days=366 * args.years)
    manager = FreeDataSourceManager(proxy_url=args.proxy)
    frame = manager.fetch_historical(symbol, str(start), str(end))
    if frame is None:
        raise RuntimeError(f"无法取得 {symbol} 数据")
    result = walk_forward_backtest(frame, DEFAULT_GRID, args.train_size, args.test_size, args.embargo, args.fee_rate)
    paths = write_backtest_report(result, Path(__file__).parent / "reports" / "adaptive_backtests", symbol.replace(".", "_"))
    print(result["summary"])
    print(f"报告: {paths[0]}\n收益明细: {paths[1]}")


if __name__ == "__main__":
    main()
