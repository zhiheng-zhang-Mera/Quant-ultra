"""Point-in-time alternative data processing."""
from pathlib import Path
import hashlib
import json
import re
import urllib.request
import urllib.error
import numpy as np
import pandas as pd

POSITIVE = {"利好", "增长", "增持", "突破", "回购", "盈利", "beat", "growth", "upgrade", "buyback", "bullish"}
NEGATIVE = {"利空", "下跌", "减持", "亏损", "处罚", "违约", "风险", "miss", "loss", "downgrade", "default", "bearish"}

class OllamaClient:
    def __init__(self, base_url="http://127.0.0.1:11434", timeout=20.0):
        self.base_url, self.timeout = base_url.rstrip("/"), float(timeout)

    def _request(self, path, payload=None):
        data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(self.base_url + path, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def list_models(self):
        return [item.get("name", "") for item in self._request("/api/tags").get("models", [])]

    def generate(self, model, prompt):
        response = self._request("/api/generate", {"model": model, "prompt": prompt, "stream": False, "format": "json", "think": False, "keep_alive": "5m", "options": {"temperature": 0, "num_predict": 256}})
        return response.get("response", "")

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

def enhance_sentiment_with_local_llm(frames, config, client=None):
    evidence = {"status": "DISABLED", "analyzed_records": 0, "fallback_used": True}
    if not config.get("local_llm_sentiment_enabled", True): return frames, evidence
    model = str(config.get("local_llm_model", "qwen3-coder:30b"))
    timeout = float(config.get("local_llm_timeout_seconds", 20))
    client = client or OllamaClient(config.get("local_llm_base_url", "http://127.0.0.1:11434"), timeout)
    try:
        models = client.list_models()
        if model not in models:
            return frames, {**evidence, "status": "MODEL_NOT_FOUND", "model": model, "available_models": models}
        per_symbol = max(1, int(config.get("local_llm_max_records_per_symbol", 3)))
        total_limit = max(1, int(config.get("local_llm_max_records_total", 6)))
        max_chars = max(50, int(config.get("local_llm_max_chars_per_record", 300)))
        candidates = []
        for source_name, frame in frames.items():
            if frame.empty: continue
            selected = frame.sort_values("published_at", ascending=False).groupby("symbol", group_keys=False).head(per_symbol)
            for idx, row in selected.iterrows():
                candidates.append({"id": f"{source_name}:{idx}", "source": source_name, "index": idx, "symbol": row["symbol"], "text": str(row["text"])[:max_chars]})
        candidates = candidates[:total_limit]
        if not candidates:
            return frames, {**evidence, "status": "NO_TEXT_RECORDS", "model": model}
        prompt = "Analyze financial sentiment. Return JSON object with key results, an array of objects: id, score (-1 to 1), confidence (0 to 1). No prose.\n" + json.dumps([{"id": x["id"], "symbol": x["symbol"], "text": x["text"]} for x in candidates], ensure_ascii=False)
        raw = client.generate(model, prompt); parsed = json.loads(raw)
        scores = {item["id"]: float(np.clip(item["score"], -1, 1)) for item in parsed.get("results", []) if "id" in item and "score" in item}
        updated = {name: frame.copy() for name, frame in frames.items()}
        for item in candidates:
            if item["id"] in scores:
                updated[item["source"]].loc[item["index"], "llm_sentiment"] = scores[item["id"]]
        for frame in updated.values():
            if "llm_sentiment" not in frame.columns: frame["llm_sentiment"] = np.nan
            frame["effective_sentiment"] = frame["llm_sentiment"].where(frame["llm_sentiment"].notna(), frame["sentiment"])
        return updated, {"status": "ANALYZED", "model": model, "analyzed_records": len(scores), "requested_records": len(candidates), "fallback_used": len(scores) < len(candidates), "timeout_seconds": timeout}
    except (OSError, ValueError, KeyError, json.JSONDecodeError, urllib.error.URLError) as exc:
        return frames, {**evidence, "status": "FALLBACK_ON_ERROR", "model": model, "error": type(exc).__name__}

def build_alternative_signals(context, as_of=None):
    """Compile PIT alternative signals.

    ``as_of`` must be the current decision date. Explicit ``as_of`` (or
    ``context["as_of_dt"]``) is required for historical re-runs/backtests:
    falling back to ``max(trading_days_dt)`` is only correct for a live
    same-day run and is reported as ``as_of_source=LIVE_MAX_FALLBACK``.
    """
    config, assets = context.get("config", {}), context.get("assets", [])
    mode = str(config.get("alternative_data_mode", "live")).strip().lower()
    if mode not in {"live", "historical", "backtest", "replay"}:
        raise ValueError(f"unsupported alternative_data_mode: {mode}")
    if as_of is None:
        as_of = context.get("as_of_dt")
    if as_of is None and mode != "live":
        raise ValueError(
            "historical alternative-data runs require explicit as_of or context['as_of_dt']"
        )
    as_of_source = "EXPLICIT" if as_of is not None else "LIVE_MAX_FALLBACK"
    if as_of is None:
        as_of = max(pd.Timestamp(x).tz_localize(None) for x in context.get("trading_days_dt", [pd.Timestamp.now("UTC")]))
    news, news_evidence = load_timed_text(config.get("news_input_path"), assets, as_of)
    forum, forum_evidence = load_timed_text(config.get("forum_input_path"), assets, as_of)
    enhanced, llm_evidence = enhance_sentiment_with_local_llm({"news": news, "forum": forum}, config)
    news, forum = enhanced["news"], enhanced["forum"]
    result = pd.DataFrame({"symbol": assets})
    for name, frame in (("news", news), ("forum", forum)):
        score_column = "effective_sentiment" if "effective_sentiment" in frame.columns else "sentiment"
        agg = frame.groupby("symbol")[score_column].agg(["mean", "count"]).reset_index()
        agg.columns = ["symbol", f"{name}_sentiment", f"{name}_record_count"]
        result = result.merge(agg, on="symbol", how="left")
    result = result.merge(capital_flow(context.get("asset_ohlcv", {})), on="symbol", how="left")
    for col in ("news_sentiment", "forum_sentiment"): result[col] = result[col].fillna(0.0).clip(-1, 1)
    for col in ("news_record_count", "forum_record_count"): result[col] = result[col].fillna(0).astype(int)
    result["alternative_signal"] = (0.35 * result["news_sentiment"] + 0.25 * result["forum_sentiment"] + 0.40 * result["capital_pool_change_5v20"].fillna(0).clip(-1, 1)).clip(-1, 1)
    return {"alternative_signals": result, "alternative_data_evidence": {"news": news_evidence, "forum": forum_evidence, "local_llm": llm_evidence, "mode": mode, "as_of": str(pd.Timestamp(as_of).tz_localize(None)), "as_of_source": as_of_source, "pit_cutoff_explicit": as_of_source == "EXPLICIT", "method": "PIT lexical sentiment, optional bounded local-LLM enhancement, and 5-day/20-day turnover pool change", "future_records_excluded": True}}
