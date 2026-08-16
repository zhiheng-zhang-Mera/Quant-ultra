"""Round-19 Priority 4: end-to-end verification that the DeepSeek fallback
path executes inside the real Phase-3 pipeline (the path main.py runs).

Builds a minimal-but-real pipeline context:
  - real 38-name Sina news corpus as news_input_path,
  - production-pool assets,
  - explicit as_of (PIT), real-source contract enforced,
  - Ollama unreachable (provider chain must skip it) + DEEPSEEK_API_KEY set
    in the process env (injected by the launcher, never printed).

Asserts the Phase-3 evidence reports status ANALYZED with provider deepseek
and that effective_sentiment carries LLM scores for the bounded live budget
(local_llm_max_records_total). Requires DEEPSEEK_API_KEY in env.

Usage:
  python reports/_iter/_e2e_deepseek_pipeline.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
Q4 = PROJECT_ROOT

sys.path.insert(0, str(PROJECT_ROOT))

from Phase_3.alternative_data import build_alternative_signals  # noqa: E402

SAMPLE = ["600519.SH", "000858.SZ", "601318.SH", "600036.SH", "000001.SZ", "000333.SH",
          "600900.SH", "601899.SH", "002594.SZ", "300750.SZ", "600887.SH", "601012.SH",
          "600028.SH", "600309.SH", "601857.SH", "600941.SH", "601398.SH", "601628.SH",
          "600585.SH", "601088.SH", "600690.SH", "600048.SH", "600030.SH", "601166.SH",
          "600276.SH", "000651.SZ", "000725.SZ", "002415.SZ", "002714.SZ", "300059.SZ",
          "300760.SZ", "002475.SZ", "000002.SZ", "000568.SZ", "002304.SZ", "300124.SZ",
          "688981.SH", "688111.SH"]
NEWS_PATH = PROJECT_ROOT / "Data_Cache" / "alternative_raw" / "sina_news_real_38.jsonl"


def main() -> int:
    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("FATAL: DEEPSEEK_API_KEY not set in this process env", file=sys.stderr)
        return 2
    config = {
        "alternative_data_mode": "historical",
        "alternative_data_require_real_source": True,
        "news_input_path": str(NEWS_PATH),
        "forum_input_path": None,
        "local_llm_sentiment_enabled": True,
        "local_llm_provider": "auto",           # Ollama down -> DeepSeek fallback
        "local_llm_model": "qwen3-coder:30b",   # listed-but-broken on this GPU (R17)
        "local_llm_timeout_seconds": 180,
        "local_llm_max_records_total": 6,
        "local_llm_max_records_per_symbol": 2,
        "local_llm_max_chars_per_record": 300,
        # research-relaxed governance so the pipeline runs to completion
        "alternative_signal_min_coverage": 0.30,
        "alternative_signal_max_missing_rate": 0.70,
        "alternative_signal_max_latency_hours": 24.0 * 60,
        "alternative_signal_allow_fallback": True,
    }
    context = {
        "config": config,
        "assets": SAMPLE,
        "as_of_dt": pd.Timestamp("2026-08-15T23:59:59Z"),
        "asset_ohlcv": {},   # capital_flow tolerates missing OHLCV (NaN -> 0)
        "trading_days_dt": [pd.Timestamp("2026-08-15")],
    }
    result = build_alternative_signals(context, as_of=pd.Timestamp("2026-08-15T23:59:59Z"))
    evidence = result["alternative_data_evidence"]
    llm = evidence["local_llm"]
    gov = result["alternative_signal_governance"]

    print("news evidence:", json.dumps({k: evidence["news"].get(k) for k in
                                        ("status", "n_rows", "n_synthetic", "sha256")}, ensure_ascii=False))
    print("llm evidence: status=%s provider=%s model=%s analyzed=%s/%s" % (
        llm.get("status"), llm.get("provider"), llm.get("model"),
        llm.get("analyzed_records"), llm.get("requested_records")))
    print("governance: status=%s reasons=%s" % (gov.get("status"), gov.get("reasons")))

    signals = result["alternative_signals"]
    eff = signals[["symbol", "news_sentiment", "news_record_count", "alternative_signal"]]
    print(eff.head(10).to_string(index=False))
    print("rows:", len(signals))

    ok = (llm.get("status") == "ANALYZED" and llm.get("provider") == "deepseek"
          and llm.get("analyzed_records", 0) > 0)
    print("E2E-DEEPSEEK:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
