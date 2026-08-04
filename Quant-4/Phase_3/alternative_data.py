"""Point-in-time alternative data processing."""
from pathlib import Path
import hashlib
import re
import numpy as np
import pandas as pd

POSITIVE = {"利好", "增长", "增持", "突破", "回购", "盈利", "beat", "growth", "upgrade", "buyback", "bullish"}
NEGATIVE = {"利空", "下跌", "减持", "亏损", "处罚", "违约", "风险", "miss", "loss", "downgrade", "default", "bearish"}

def score_text(text):
    tokens = re.findall(r"[A-Za-z]+|[\u4e00-\u9fff]{2,}", str(text).lower())
    positive = sum(any(term in token for term in POSITIVE) for token in tokens)
    negative = sum(any(term in token for term in NEGATIVE) for token in tokens)
    return float((positive - negative) / max(positive + negative, 1))

def load_timed_text(path, assets, as_of):
    required = {"published_at", "symbol", "text"}
    if not path or not Path(path).exists():
        return pd.DataFrame(columns=[*required, "sentiment"]), {"status": "MISSING_OPTIONAL_SOURCE", "path": path}
    source = Path(path)
    frame = pd.read_json(source, lines=True) if source.suffix.lower() == ".jsonl" else pd.read_csv(source)
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Alternative-data file {source} missing columns: {sorted(missing)}")
    frame["published_at"] = pd.to_datetime(frame["published_at"], utc=True, errors="coerce")
    cutoff = pd.Timestamp(as_of)
    cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
    frame = frame[frame["published_at"].notna() & (frame["published_at"] <= cutoff) & frame["symbol"].isin(set(assets))].copy()
    frame["sentiment"] = frame["text"].map(score_text)
    return frame, {"status": "LOADED", "path": str(source), "records": len(frame), "sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "as_of": cutoff.isoformat()}

def capital_flow(asset_ohlcv):
    rows = []
    for symbol, raw in asset_ohlcv.items():
        frame = raw.copy(); frame.columns = [str(c).lower() for c in frame.columns]
        if not {"close", "volume"}.issubset(frame.columns) or len(frame) < 21: continue
        turnover = pd.to_numeric(frame["close"], errors="coerce") * pd.to_numeric(frame["volume"], errors="coerce")
        recent, baseline = turnover.tail(5).mean(), turnover.tail(20).mean()
        change = float(recent / baseline - 1.0) if baseline and np.isfinite(baseline) else np.nan
        rows.append({"symbol": symbol, "turnover_5d": float(recent), "turnover_20d": float(baseline), "capital_pool_change_5v20": change})
    return pd.DataFrame(rows, columns=["symbol", "turnover_5d", "turnover_20d", "capital_pool_change_5v20"])

def build_alternative_signals(context):
    config, assets = context.get("config", {}), context.get("assets", [])
    as_of = max(pd.Timestamp(x).tz_localize(None) for x in context.get("trading_days_dt", [pd.Timestamp.now("UTC")]))
    news, news_evidence = load_timed_text(config.get("news_input_path"), assets, as_of)
    forum, forum_evidence = load_timed_text(config.get("forum_input_path"), assets, as_of)
    result = pd.DataFrame({"symbol": assets})
    for name, frame in (("news", news), ("forum", forum)):
        agg = frame.groupby("symbol")["sentiment"].agg(["mean", "count"]).reset_index()
        agg.columns = ["symbol", f"{name}_sentiment", f"{name}_record_count"]
        result = result.merge(agg, on="symbol", how="left")
    result = result.merge(capital_flow(context.get("asset_ohlcv", {})), on="symbol", how="left")
    for col in ("news_sentiment", "forum_sentiment"): result[col] = result[col].fillna(0.0).clip(-1, 1)
    for col in ("news_record_count", "forum_record_count"): result[col] = result[col].fillna(0).astype(int)
    result["alternative_signal"] = (0.35 * result["news_sentiment"] + 0.25 * result["forum_sentiment"] + 0.40 * result["capital_pool_change_5v20"].fillna(0).clip(-1, 1)).clip(-1, 1)
    return {"alternative_signals": result, "alternative_data_evidence": {"news": news_evidence, "forum": forum_evidence, "method": "PIT lexical sentiment plus 5-day/20-day turnover pool change", "future_records_excluded": True}}
