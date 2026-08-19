from pathlib import Path
import json
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Main.financial_events import EVENT_SCHEMA_VERSION, evaluate_event_representation, parse_financial_event_response
from Phase_3.alternative_data import enhance_sentiment_with_local_llm


def _event(record_id="news:0"):
    return {"id": record_id, "event_type": "capital_action", "scope": "company", "direction": 1,
            "importance": .8, "novelty": .7, "uncertainty": .2, "horizon_days": 30,
            "reliability": .9, "score": .6, "confidence": .85}


def test_structured_event_parser_requires_full_auditable_schema():
    events = parse_financial_event_response(json.dumps({"results": [_event()]}), ["news:0"])
    assert events["news:0"]["event_schema_version"] == EVENT_SCHEMA_VERSION
    assert events["news:0"]["event_type"] == "capital_action"
    assert events["news:0"]["event_horizon_days"] == 30
    research = evaluate_event_representation(events, {"news:0": .2}, 1)
    assert research["status"] == "RESEARCH_ONLY"
    assert research["checks"]["complete_schema"] is True
    assert research["checks"]["independent_signal_validation"] is False


def test_sentiment_only_or_invalid_event_output_is_not_structured_evidence():
    with pytest.raises(ValueError, match="missing required fields"):
        parse_financial_event_response(json.dumps({"results": [{"id": "news:0", "score": .5, "confidence": .8}]}), ["news:0"])
    bad = _event()
    bad["horizon_days"] = 0
    with pytest.raises(ValueError, match="horizon_days"):
        parse_financial_event_response(json.dumps({"results": [bad]}), ["news:0"])


def test_llm_pipeline_writes_structured_event_fields_but_keeps_research_only():
    class EventClient:
        def list_models(self):
            return ["event-model"]

        def generate(self, model, prompt):
            assert "event_type" in prompt and "horizon_days" in prompt
            return json.dumps({"results": [_event()]})

    frame = pd.DataFrame([{"published_at": pd.Timestamp("2026-08-01", tz="UTC"),
                           "symbol": "600519.SH", "text": "buyback announced", "sentiment": .2}])
    updated, evidence = enhance_sentiment_with_local_llm(
        {"news": frame}, {"local_llm_model": "event-model", "local_llm_max_records_total": 1}, client=EventClient()
    )
    assert evidence["structured_records"] == 1
    assert evidence["event_schema_version"] == EVENT_SCHEMA_VERSION
    assert evidence["event_research"]["status"] == "RESEARCH_ONLY"
    assert updated["news"].loc[0, "event_type"] == "capital_action"
    assert updated["news"].loc[0, "event_horizon_days"] == 30
