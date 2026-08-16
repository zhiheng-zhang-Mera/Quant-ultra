"""Tests for the guba.eastmoney.com forum fetcher (Round-19 priority 3)."""
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from tools.fetch_forum_guba import parse_rows, to_contract_rows  # noqa: E402

HTML = """<table class="default_list"><thead class="listhead"><tr><td><div>阅读</div></td><td><div>评论</div></td><td><div>标题</div></td><td><div>作者</div></td><td><div>最后更新</div></td></tr></thead><tbody class="listbody">
<tr class="listitem"><td><div class="read">1384</div></td><td><div class="reply">9</div></td><td><div class="title"><a data-postid="1759532877" data-posttype="1" href="/news,600519,1759532877.html">茅台的“减速带”</a></div></td><td><div class="author"><a href="//i.eastmoney.com/7344113638256342">贵州茅台资讯</a></div></td><td><div class="update">08-15 07:57</div></td></tr>
<tr class="listitem"><td><div class="read">49</div></td><td><div class="reply">0</div></td><td><div class="title"><a data-postid="1759560001" data-posttype="2" href="/news,600519,1759560001.html">周一低开5％，低开低走</a></div></td><td><div class="author"><a href="//i.eastmoney.com/12345">老板娘</a></div></td><td><div class="update">08-16 08:00</div></td></tr>
</tbody></table>"""


def test_parse_rows_extracts_post_fields():
    rows = parse_rows(HTML)
    assert len(rows) == 2
    first = rows[0]
    assert first["postid"] == "1759532877"
    assert first["title"] == "茅台的“减速带”"
    assert first["author"] == "贵州茅台资讯"
    assert first["update"] == "08-15 07:57"
    assert first["read"] == "1384" and first["reply"] == "9"


def test_to_contract_rows_interprets_cst_and_guards_future():
    ingested = datetime(2026, 8, 16, 1, 0, tzinfo=timezone.utc)
    rows = parse_rows(HTML)
    contract = to_contract_rows("600519.SH", rows, ingested)
    # 08-16 08:00 CST = 08-16 00:00 UTC, which is before ingest -> kept
    assert len(contract) == 2
    r0 = contract[0]
    assert r0["record_id"] == "600519.SH:guba:1759532877"
    assert r0["source"] == "guba_eastmoney"
    assert r0["source_type"] == "forum"
    assert r0["license"] == "public_display"
    assert r0["symbol"] == "600519.SH"
    assert r0["is_synthetic"] is False
    # CST -> UTC conversion: 08-15 07:57 CST == 08-14 23:57 UTC
    assert contract[0]["published_at"] == "2026-08-14T23:57:00+00:00"
    assert contract[1]["published_at"] == "2026-08-16T00:00:00+00:00"


def test_to_contract_rows_drops_future_items_and_dupes():
    ingested = datetime(2026, 8, 15, 23, 30, tzinfo=timezone.utc)
    rows = parse_rows(HTML)
    contract = to_contract_rows("600519.SH", rows, ingested)
    # second row 08-16 08:00 CST = 08-16 00:00 UTC > ingest -> dropped (PIT)
    assert len(contract) == 1
    assert contract[0]["record_id"] == "600519.SH:guba:1759532877"
