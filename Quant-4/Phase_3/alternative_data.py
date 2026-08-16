"""Point-in-time alternative data processing."""
from pathlib import Path
import hashlib
import json
import os
import re
import urllib.request
import urllib.error
import numpy as np
import pandas as pd

from Phase_3.alternative_data_contract import RAW_TEXT_CONTRACT_VERSION, load_contract_text
from Main.alternative_signal_governance import evaluate_phase3_signal

POSITIVE = {"利好", "增长", "增持", "突破", "回购", "盈利", "beat", "growth", "upgrade", "buyback", "bullish"}
NEGATIVE = {"利空", "下跌", "减持", "亏损", "处罚", "违约", "风险", "miss", "loss", "downgrade", "default", "bearish"}

LLM_PROMPT_VERSION = "financial-sentiment-json/v1"

# DeepSeek API configuration is read from the local environment. These names
# match the OpenAI-compatible DeepSeek endpoints:
#   DEEPSEEK_API_KEY    (required)  - e.g. "sk-..."
#   DEEPSEEK_BASE_URL   (optional)  - default https://api.deepseek.com
#   DEEPSEEK_MODEL      (optional)  - default deepseek-chat
ENV_DEEPSEEK_API_KEY = "DEEPSEEK_API_KEY"
ENV_DEEPSEEK_BASE_URL = "DEEPSEEK_BASE_URL"
ENV_DEEPSEEK_MODEL = "DEEPSEEK_MODEL"
DEEPSEEK_DEFAULT_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_DEFAULT_MODEL = "deepseek-chat"


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


class DeepSeekClient:
    """OpenAI-compatible DeepSeek chat-completions client.

    Used as a fallback when the local Ollama is unavailable. Credentials are
    read from the environment (DEEPSEEK_API_KEY / DEEPSEEK_BASE_URL /
    DEEPSEEK_MODEL); an explicit ``api_key`` may also be passed for tests.
    """

    def __init__(self, api_key=None, base_url=None, model=None, timeout=60.0):
        self.api_key = api_key if api_key is not None else os.environ.get(ENV_DEEPSEEK_API_KEY, "")
        self.base_url = (base_url or os.environ.get(ENV_DEEPSEEK_BASE_URL, DEEPSEEK_DEFAULT_BASE_URL)).rstrip("/")
        self.model = model or os.environ.get(ENV_DEEPSEEK_MODEL, DEEPSEEK_DEFAULT_MODEL)
        self.timeout = float(timeout)

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def generate(self, model, prompt):
        """One batched completion: the whole prompt (all candidates plus the
        packaged lexical scores) is analysed in a single API call."""
        payload = {
            "model": model or self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "stream": False,
            "response_format": {"type": "json_object"},
        }
        request = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
        return body["choices"][0]["message"]["content"]


def score_text(text):
    tokens = re.findall(r"[A-Za-z]+|[\u4e00-\u9fff]{2,}", str(text).lower())
    positive = sum(any(term in token for term in POSITIVE) for token in tokens)
    negative = sum(any(term in token for term in NEGATIVE) for token in tokens)
    return float((positive - negative) / max(positive + negative, 1))

def load_timed_text(path, assets, as_of, config=None):
    required = {"published_at", "symbol", "text"}
    if not path:
        return pd.DataFrame(columns=[*required, "sentiment"]), {
            "status": "MISSING_OPTIONAL_SOURCE", "path": path,
            "contract_version": RAW_TEXT_CONTRACT_VERSION,
        }
    config = config or {}
    cache_dir = config.get("alternative_data_cache_dir") or (
        Path(__file__).resolve().parents[1] / "Data_Cache" / "alternative_raw"
    )
    frame, evidence = load_contract_text(
        Path(path), assets, as_of, Path(cache_dir),
        require_real_source=bool(config.get("alternative_data_require_real_source", True)),
    )
    frame["sentiment"] = frame["text"].map(score_text)
    return frame, evidence

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

def _provider_chain(config, injected_client=None):
    """Build the ordered list of (client, provider, model) candidates.

    Order: injected client (tests) -> Ollama (when it lists the requested
    model) -> DeepSeek (when DEEPSEEK_API_KEY is present). Each later entry
    is a fallback if the earlier provider's generate call fails.
    """
    chain = []
    if injected_client is not None:
        chain.append((injected_client, "injected", str(config.get("local_llm_model", "qwen3-coder:30b"))))
        return chain
    model = str(config.get("local_llm_model", "qwen3-coder:30b"))
    timeout = float(config.get("local_llm_timeout_seconds", 20))
    provider_pref = str(config.get("local_llm_provider", "auto")).strip().lower()
    if provider_pref == "deepseek":
        ds = DeepSeekClient(timeout=timeout)
        if ds.available:
            chain.append((ds, "deepseek", ds.model))
        return chain
    ollama = OllamaClient(config.get("local_llm_base_url", "http://127.0.0.1:11434"), timeout)
    try:
        models = ollama.list_models()
    except Exception:
        models = []
    if model in models:
        chain.append((ollama, "ollama", model))
    if provider_pref == "ollama":
        return chain
    ds = DeepSeekClient(timeout=timeout)
    if ds.available:
        chain.append((ds, "deepseek", ds.model))
    return chain


def enhance_sentiment_with_local_llm(frames, config, client=None):
    evidence = {"status": "DISABLED", "analyzed_records": 0, "fallback_used": True,
                "prompt_version": LLM_PROMPT_VERSION}
    if not config.get("local_llm_sentiment_enabled", True): return frames, evidence
    timeout = float(config.get("local_llm_timeout_seconds", 20))
    model = str(config.get("local_llm_model", "qwen3-coder:30b"))
    # Provider chain: injected client (tests) -> Ollama -> DeepSeek. The
    # candidates + packaged lexical scores are built ONCE and the same prompt
    # is sent to whichever provider answers, so a failed Ollama call falls
    # through to DeepSeek without rebuilding or duplicating the prompt.
    chain = _provider_chain(config, injected_client=client)
    if not chain:
        return frames, {**evidence, "status": "PROVIDER_UNAVAILABLE", "model": model,
                        "provider": str(config.get("local_llm_provider", "auto"))}
    # Injected clients keep the legacy model-list gate (absent model -> no-op).
    if client is not None and hasattr(client, "list_models"):
        try:
            available = client.list_models()
            if model not in available:
                return frames, {**evidence, "status": "MODEL_NOT_FOUND", "model": model, "available_models": available}
        except Exception:
            pass
    try:
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
        # Package the lexical (dictionary) sentiment scores into the single
        # batched prompt so the model sees the dictionary baseline for every
        # record in one call (no per-record round trips -> token savings).
        for item in candidates:
            item["lexical_score"] = score_text(item["text"])
        prompt = "Analyze financial sentiment. Return JSON object with key results, an array of objects: id, score (-1 to 1), confidence (0 to 1). lexical_score is a dictionary baseline for reference; your score may agree or disagree. No prose.\n" + json.dumps([{"id": x["id"], "symbol": x["symbol"], "text": x["text"], "lexical_score": x["lexical_score"]} for x in candidates], ensure_ascii=False)
        prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        last_error = None
        for cand_client, provider, cand_model in chain:
            try:
                raw = cand_client.generate(cand_model, prompt)
                parsed = json.loads(raw)
                scores = {item["id"]: float(np.clip(item["score"], -1, 1)) for item in parsed.get("results", []) if "id" in item and "score" in item}
                updated = {name: frame.copy() for name, frame in frames.items()}
                for item in candidates:
                    if item["id"] in scores:
                        updated[item["source"]].loc[item["index"], "llm_sentiment"] = scores[item["id"]]
                for frame in updated.values():
                    if "llm_sentiment" not in frame.columns: frame["llm_sentiment"] = np.nan
                    frame["effective_sentiment"] = frame["llm_sentiment"].where(frame["llm_sentiment"].notna(), frame["sentiment"])
                return updated, {"status": "ANALYZED", "provider": provider, "model": cand_model,
                                 "prompt_version": LLM_PROMPT_VERSION,
                                 "prompt_sha256": prompt_sha256,
                                 "response_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
                                 "temperature": 0, "num_predict": 256, "response_format": "json",
                                 "analyzed_records": len(scores), "requested_records": len(candidates),
                                 "fallback_used": len(scores) < len(candidates), "timeout_seconds": timeout}
            except (OSError, ValueError, KeyError, json.JSONDecodeError, urllib.error.URLError, IndexError) as exc:
                last_error = type(exc).__name__
                continue  # try the next provider in the chain
        return frames, {**evidence, "status": "FALLBACK_ON_ERROR", "provider": "|".join(p for _, p, _ in chain),
                        "model": model, "error": last_error or "ALL_PROVIDERS_FAILED"}
    except (OSError, ValueError, KeyError, json.JSONDecodeError, urllib.error.URLError, IndexError) as exc:
        return frames, {**evidence, "status": "FALLBACK_ON_ERROR", "provider": "|".join(p for _, p, _ in chain),
                        "model": model, "error": type(exc).__name__}

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
    news, news_evidence = load_timed_text(config.get("news_input_path"), assets, as_of, config)
    forum, forum_evidence = load_timed_text(config.get("forum_input_path"), assets, as_of, config)
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
    governance = evaluate_phase3_signal(
        result, assets, [news_evidence, forum_evidence], llm_evidence, config
    )
    return {"alternative_signals": result, "alternative_data_evidence": {"news": news_evidence, "forum": forum_evidence, "local_llm": llm_evidence, "mode": mode, "as_of": str(pd.Timestamp(as_of).tz_localize(None)), "as_of_source": as_of_source, "pit_cutoff_explicit": as_of_source == "EXPLICIT", "method": "PIT lexical sentiment, optional bounded local-LLM enhancement, and 5-day/20-day turnover pool change", "future_records_excluded": True}, "alternative_signal_governance": governance}
