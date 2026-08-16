"""Tests for the eastmoney full-history notice fetcher (Round 19.6)."""
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.fetch_history_notices_eastmoney import make_rows  # noqa: E402


def test_make_rows_contract_format():
    frame = pd.DataFrame([
        {"公告标题": "贵州茅台:关于回购股份实施结果暨股份变动的公告",
         "公告日期": "2026-08-15",
         "网址": "https://data.eastmoney.com/notices/detail/600519/AN202608151800000001.html"},
        {"公告标题": "贵州茅台:2026年半年度报告摘要",
         "公告日期": "2026-08-15",
         "网址": "https://data.eastmoney.com/notices/detail/600519/AN202608151800000002.html"},
        {"公告标题": "未来公告(不应入库)",
         "公告日期": "2026-08-20",
         "网址": "https://data.eastmoney.com/notices/detail/600519/AN202608201800000003.html"},
    ])
    ingested = datetime(2026, 8, 16, 4, 0, tzinfo=timezone.utc)
    rows = make_rows("600519.SH", frame, ingested)
    assert len(rows) == 2, "future-published rows must be dropped (PIT)"
    r = rows[0]
    assert r["record_id"] == "600519.SH:eastmoney_notice:202608151800000001"
    assert r["source"] == "eastmoney_notices"
    assert r["source_type"] == "notice"
    assert r["license"] == "public_display"
    assert r["symbol"] == "600519.SH"
    assert r["text"] == "贵州茅台:关于回购股份实施结果暨股份变动的公告"
    assert r["published_at"] == "2026-08-15T00:00:00+00:00"
    assert r["is_synthetic"] is False


def test_make_rows_dedupes_announcement_ids():
    frame = pd.DataFrame([
        {"公告标题": "A", "公告日期": "2026-08-15", "网址": "https://x/AN1.html"},
        {"公告标题": "A重复", "公告日期": "2026-08-15", "网址": "https://x/AN1.html"},
        {"公告标题": "B", "公告日期": "2026-08-15", "网址": "https://x/AN2.html"},
    ])
    ingested = datetime(2026, 8, 16, 4, 0, tzinfo=timezone.utc)
    rows = make_rows("000001.SZ", frame, ingested)
    assert len(rows) == 2
    ids = [r["record_id"] for r in rows]
    assert len(set(ids)) == 2
