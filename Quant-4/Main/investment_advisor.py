"""Post-pipeline candidate selection and portfolio query reporting."""
from __future__ import annotations
import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
from Main.portfolio_analytics import holding_advice, recommendation


def infer_kind(symbol: str) -> str:
    code = symbol.split(".")[0]
    return "ETF" if code.startswith(("15", "16", "50", "51", "56", "58")) else "股票"


def build_pipeline_recommendations(context: dict, top_n: int = 20) -> pd.DataFrame:
    weights = context.get("daily_weights")
    manager = context["data_manager"]
    if not isinstance(weights, pd.DataFrame) or weights.empty:
        raise ValueError("缺少 Phase 6 daily_weights，无法生成十阶段后的候选建议")
    latest_weights = weights.iloc[-1].dropna().sort_values(ascending=False)
    symbols = latest_weights[latest_weights > 0].head(max(top_n * 3, top_n)).index
    end = datetime.now().date()
    start = end - timedelta(days=550)
    rows = []
    alternative = context.get("alternative_signals")
    alternative_map = dict(zip(alternative["symbol"], alternative["alternative_signal"])) if isinstance(alternative, pd.DataFrame) and not alternative.empty else {}
    asset_names = manager.fetch_asset_names(list(symbols))
    context["asset_names"] = asset_names
    for symbol in symbols:
        try:
            frame = manager.fetch_historical(str(symbol), str(start), str(end))
            if frame is None or len(frame) < 60:
                continue
            asset_type = infer_kind(str(symbol))
            rec = recommendation(frame, model_weight=float(latest_weights[symbol]), asset_type=asset_type, cost_config=context.get("config"), alternative_signal=float(alternative_map.get(symbol, 0.0)))
            if rec["qualified"]:
                rows.append({"symbol": symbol, "asset_name": asset_names.get(str(symbol).upper(), "名称数据不可用 / Name data unavailable"), "asset_type": asset_type, **rec})
        except Exception:
            continue
    if not rows:
        return pd.DataFrame(columns=["symbol", "asset_type", "entry_price_low", "entry_price_high", "suggested_weight", "take_profit_pct"])
    return pd.DataFrame(rows).sort_values(["score", "sharpe"], ascending=False).head(top_n).reset_index(drop=True)


def write_candidate_report(frame: pd.DataFrame, report_dir: Path) -> tuple[Path, Path]:
    report_dir.mkdir(parents=True, exist_ok=True)
    csv_path = report_dir / "post_pipeline_candidates.csv"
    frame.to_csv(csv_path, index=False, encoding="utf-8-sig")
    digest = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    md_path = report_dir / "post_pipeline_candidates.md"
    columns = [c for c in ["symbol", "asset_type", "dominant_selection_method", "dominant_entry_method", "dominant_holding_method", "dominant_take_profit_method", "entry_price_low", "entry_price_high", "suggested_weight", "take_profit_pct", "score", "sharpe"] if c in frame]
    if frame.empty:
        table = "本次没有通过全部条件的候选标的。"
    else:
        table_lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
        for _, row in frame[columns].iterrows():
            table_lines.append("| " + " | ".join(str(row[c]) for c in columns) + " |")
        table = "\n".join(table_lines)
    layman_note = (
        "一句话结论：本次分析没有生成可观察候选，建议先查看治理门禁与审计报告。"
        if frame.empty
        else f"一句话结论：系统从 {len(frame)} 只标的中筛选出值得关注的投资观察候选，均为分析观察结果，不是下单指令。"
    )
    governance_note = ""
    if "advisory_mode" in frame.columns and frame["advisory_mode"].nunique() == 1:
        mode = frame["advisory_mode"].iloc[0]
        governance_note = f"\n\n> 治理状态：{mode}。OBSERVATION_ONLY 表示仅观察、禁止执行。"
    md_path.write_text(
        "# 十阶段后股票/ETF建议\n\n"
        + layman_note
        + governance_note
        + "\n\n## 候选清单 / Candidate List\n\n"
        + table
        + f"\n\n- CSV SHA-256：`{digest}`\n",
        encoding="utf-8",
    )
    return md_path, csv_path


def analyze_holding(manager, total_capital: float, code: str, quantity: int, average_cost: float, kind: str | None = None) -> dict:
    from analyze_cn_asset import normalize
    normalized_kind = kind or ("etf" if code.startswith(("15", "16", "50", "51", "56", "58")) else "stock")
    symbol = normalize(code, normalized_kind)
    end = datetime.now().date()
    frame = manager.fetch_historical(symbol, str(end - timedelta(days=550)), str(end))
    if frame is None or len(frame) < 60:
        raise RuntimeError(f"{symbol} 可验证行情不足60条")
    return {"symbol": symbol, "asset_type": normalized_kind, **holding_advice(frame, total_capital, quantity, average_cost, asset_type=normalized_kind)}


def save_holding_report(result: dict, report_dir: Path) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = report_dir / f"holding_{result['symbol'].replace('.', '_')}_{stamp}.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return path
