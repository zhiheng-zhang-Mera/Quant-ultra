"""Interactive A-share stock/ETF portfolio query. Enter END to finish."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from Main.datasource_manager import FreeDataSourceManager


def normalize(code: str, kind: str) -> str:
    c = code.strip().upper().split(".")[0]
    if len(c) != 6 or not c.isdigit():
        raise ValueError("代码必须为6位数字")
    if kind == "stock":
        return f"{c}.SH" if c.startswith(("6", "9")) else f"{c}.SZ"
    if kind == "etf":
        return f"{c}.SH" if c.startswith(("5", "51", "56", "58")) else f"{c}.SZ"
    raise ValueError("类型必须是 stock 或 etf")


def run_portfolio_query(manager=None, end_token: str = "END", report_dir: Path | None = None):
    from Main.investment_advisor import analyze_holding, save_holding_report
    manager = manager or FreeDataSourceManager()
    report_dir = report_dir or Path(__file__).parent / "reports" / "portfolio_queries"
    print("输入：总资金 股票/ETF代码 持仓数量 平均成本 [stock/etf]；输入 END 结束。")
    while True:
        raw = input("portfolio> ").strip()
        if raw.upper() == end_token.upper():
            break
        try:
            parts = raw.split()
            if len(parts) not in (4, 5):
                raise ValueError("需要4项必填参数，可选第5项类型 stock/etf")
            total, code, quantity, cost = parts[:4]
            kind = parts[4].lower() if len(parts) == 5 else None
            result = analyze_holding(manager, float(total), code, int(quantity), float(cost), kind)
            path = save_holding_report(result, report_dir)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            print(f"详细报告：{path}")
        except Exception as exc:
            print(f"输入或分析失败：{exc}")


def main():
    parser = argparse.ArgumentParser(description="A股股票/ETF持仓操作查询")
    parser.add_argument("--proxy")
    parser.add_argument("--end-token", default="END")
    args = parser.parse_args()
    run_portfolio_query(FreeDataSourceManager(proxy_url=args.proxy), args.end_token)


if __name__ == "__main__":
    main()
