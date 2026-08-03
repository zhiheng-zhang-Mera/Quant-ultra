"""Interactive A-share stock/ETF analyzer. Enter END to finish."""
from __future__ import annotations
import argparse, json, sys
from datetime import datetime, timedelta
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from Main.datasource_manager import FreeDataSourceManager
from Main.portfolio_analytics import trade_signal

def normalize(code: str, kind: str) -> str:
    c = code.strip().upper().split(".")[0]
    if len(c) != 6 or not c.isdigit(): raise ValueError("代码必须为6位数字")
    if kind == "stock": return f"{c}.SH" if c.startswith(("6", "9")) else f"{c}.SZ"
    if kind == "etf": return f"{c}.SH" if c.startswith(("5", "51", "56", "58")) else f"{c}.SZ"
    raise ValueError("类型必须是 stock 或 etf")

def analyze(manager, kind, code, weight, cost):
    symbol = normalize(code, kind); end = datetime.now().date(); start = end - timedelta(days=550)
    df = manager.fetch_historical(symbol, str(start), str(end))
    if df is None or len(df) < 60: raise RuntimeError(f"{symbol} 可验证数据不足60条")
    result = {"kind": kind, "symbol": symbol, **trade_signal(df["close"], weight, cost, float(df["close"].iloc[-1]))}
    return result

def main():
    p=argparse.ArgumentParser(description="指定A股股票/ETF交易分析"); p.add_argument("--proxy"); p.add_argument("--end-token", default="END"); args=p.parse_args()
    manager=FreeDataSourceManager(proxy_url=args.proxy)
    print(f"逐行输入: 类型(stock/etf) 代码 仓位(0~1) 成本；输入 {args.end_token} 结束。")
    while True:
        raw=input("asset> ").strip()
        if raw.upper()==args.end_token.upper(): break
        try:
            kind, code, weight, cost=raw.split(); print(json.dumps(analyze(manager, kind.lower(), code, float(weight), float(cost)), ensure_ascii=False, indent=2))
        except Exception as exc: print(f"输入或分析失败: {exc}")

if __name__ == "__main__": main()

