from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Main.alternative_data_research import evaluate_alternative_sources


def _frame(source, symbols):
    return pd.DataFrame([{"record_id": f"{source}-{index}", "symbol": symbol,
                          "published_at": "2026-08-01T09:00:00Z", "ingested_at": "2026-08-01T10:00:00Z",
                          "sentiment": .3} for index, symbol in enumerate(symbols)])


def _evidence(source):
    return {"status": "LOADED", "sha256": (source[0] * 64), "max_source_latency_hours": 1.0}


def test_news_announcement_forum_share_one_quality_and_conflict_view():
    symbols = ["A", "B", "C", "D"]
    frames = {name: _frame(name, symbols) for name in ("news", "announcement", "forum")}
    evidence = {name: _evidence(name) for name in frames}
    result = evaluate_alternative_sources(frames, evidence, symbols, "2026-08-02")
    assert result["status"] == "PASS", result["reasons"]
    assert set(result["sources"]) == {"news", "announcement", "forum"}
    assert all(item["state"] == "OBSERVED_EVENTS" for item in result["sources"].values())
    assert result["conflicts"]["conflict_rate"] == 0
    assert len(result["manifest_sha256"]) == 64


def test_missing_source_is_distinct_from_loaded_source_with_no_event():
    symbols = ["A", "B"]
    frames = {"news": _frame("news", symbols), "announcement": pd.DataFrame(), "forum": pd.DataFrame()}
    evidence = {"news": _evidence("news"), "announcement": _evidence("announcement"),
                "forum": {"status": "MISSING_OPTIONAL_SOURCE"}}
    result = evaluate_alternative_sources(frames, evidence, symbols, "2026-08-02")
    assert result["status"] == "HOLD"
    assert result["sources"]["announcement"]["state"] == "NO_ELIGIBLE_EVENT"
    assert result["sources"]["forum"]["state"] == "DATA_MISSING"
