"""Strict source, timestamp, license and immutable-cache evidence tests."""
import json
from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Phase_3.alternative_data_contract import (
    AlternativeDataContractError,
    RAW_TEXT_CONTRACT_VERSION,
    load_contract_text,
    sha256_file,
)


def _record(record_id, published_at, ingested_at=None, synthetic=False):
    return {
        "record_id": record_id,
        "source": "licensed-wire",
        "source_type": "news",
        "license": "research-entitlement-2026",
        "published_at": published_at,
        "ingested_at": ingested_at or published_at,
        "symbol": "600519.SH",
        "text": "growth and buyback",
        "is_synthetic": synthetic,
    }


def _write(path, records):
    path.write_text("\n".join(json.dumps(row) for row in records), encoding="utf-8")


def test_dual_timestamp_filter_and_content_addressed_cache(tmp_path):
    source = tmp_path / "news.jsonl"
    _write(source, [
        _record("known", "2025-06-01T09:00:00Z", "2025-06-01T09:01:00Z"),
        _record("late-ingest", "2025-06-01T09:00:00Z", "2025-07-01T09:00:00Z"),
        _record("future", "2025-07-01T09:00:00Z"),
    ])
    selected, evidence = load_contract_text(
        source, ["600519.SH"], pd.Timestamp("2025-06-30"), tmp_path / "cache"
    )
    assert selected["record_id"].tolist() == ["known"]
    assert evidence["contract_version"] == RAW_TEXT_CONTRACT_VERSION
    assert evidence["dual_timestamp_filter"] is True
    assert evidence["max_source_latency_hours"] == pytest.approx(24.0 * 30)
    cached = Path(evidence["immutable_cache_path"])
    assert cached.is_file() and cached.stem == evidence["sha256"]
    assert sha256_file(cached) == sha256_file(source)


def test_synthetic_source_is_rejected_by_production_contract(tmp_path):
    source = tmp_path / "synthetic.jsonl"
    _write(source, [_record("synthetic", "2025-06-01T09:00:00Z", synthetic=True)])
    with pytest.raises(AlternativeDataContractError, match="synthetic text is prohibited"):
        load_contract_text(source, ["600519.SH"], "2025-06-30", tmp_path / "cache")


def test_missing_license_and_duplicate_ids_fail_closed(tmp_path):
    missing = tmp_path / "missing.jsonl"
    row = _record("x", "2025-06-01T09:00:00Z")
    row.pop("license")
    _write(missing, [row])
    with pytest.raises(AlternativeDataContractError, match="missing columns"):
        load_contract_text(missing, ["600519.SH"], "2025-06-30", tmp_path / "cache")

    duplicate = tmp_path / "duplicate.jsonl"
    _write(duplicate, [_record("x", "2025-06-01T09:00:00Z"), _record("x", "2025-06-02T09:00:00Z")])
    with pytest.raises(AlternativeDataContractError, match="globally unique"):
        load_contract_text(duplicate, ["600519.SH"], "2025-06-30", tmp_path / "cache")
