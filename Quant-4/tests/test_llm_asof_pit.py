# -*- coding: utf-8 -*-
"""8-13: PIT / no-lookahead tests for the LLM alternative-data wiring."""
import sys
from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Phase_3.alternative_data import build_alternative_signals
from Main.advice_portfolio_backtest import UniverseDecision, run_advice_portfolio_backtest


def _write_records(path, records):
    enriched = []
    for index, record in enumerate(records):
        item = dict(record)
        item.setdefault("record_id", f"record-{index}")
        item.setdefault("source", "licensed-test-feed")
        item.setdefault("source_type", "news")
        item.setdefault("license", "test-research-license")
        item.setdefault("ingested_at", item["published_at"])
        item.setdefault("is_synthetic", False)
        enriched.append(item)
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in enriched), encoding="utf-8")


def _ohlcv_frame(n=170, start="2024-01-02", seed=7):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n)
    close = 10.0 * np.exp(np.cumsum(rng.normal(0.0004, 0.008, n)))
    open_ = close * (1 + rng.normal(0, 0.002, n))
    high = np.maximum(open_, close) * 1.01
    low = np.minimum(open_, close) * 0.99
    volume = rng.integers(1_000_000, 5_000_000, n).astype(float)
    return pd.DataFrame({
        "date": dates, "open": open_, "high": high, "low": low,
        "close": close, "volume": volume, "amount": close * volume,
    })


def test_explicit_asof_excludes_future_records(tmp_path):
    news = tmp_path / "news.jsonl"
    _write_records(news, [
        {"published_at": "2025-06-01T09:30:00Z", "symbol": "600519.SH", "text": "利好 增长 回购"},
        {"published_at": "2026-01-01T09:30:00Z", "symbol": "600519.SH", "text": "FUTURE_LEAK future message"},
    ])
    ctx = {
        "config": {"news_input_path": str(news), "forum_input_path": None, "local_llm_sentiment_enabled": False,
                   "alternative_data_cache_dir": str(tmp_path / "raw-cache")},
        "assets": ["600519.SH"],
        "trading_days_dt": [pd.Timestamp("2025-12-31")],
        "asset_ohlcv": {},
    }
    result = build_alternative_signals(ctx, as_of=pd.Timestamp("2025-12-31"))
    ev = result["alternative_data_evidence"]
    assert ev["news"]["records"] == 1, "future text record leaked into PIT signals"
    assert ev["as_of_source"] == "EXPLICIT"
    assert int(result["alternative_signals"].iloc[0]["news_record_count"]) == 1
    assert result["alternative_signals"].iloc[0]["news_sentiment"] > 0


def test_live_fallback_uses_max_and_is_labeled(tmp_path):
    news = tmp_path / "news.jsonl"
    _write_records(news, [
        {"published_at": "2025-06-01T09:30:00Z", "symbol": "600519.SH", "text": "利好"},
        {"published_at": "2025-12-01T09:30:00Z", "symbol": "600519.SH", "text": "利空"},
    ])
    ctx = {
        "config": {"news_input_path": str(news), "forum_input_path": None, "local_llm_sentiment_enabled": False,
                   "alternative_data_cache_dir": str(tmp_path / "raw-cache")},
        "assets": ["600519.SH"],
        "trading_days_dt": [pd.Timestamp("2025-06-15")],
        "asset_ohlcv": {},
    }
    result = build_alternative_signals(ctx)
    ev = result["alternative_data_evidence"]
    assert ev["as_of_source"] == "LIVE_MAX_FALLBACK"
    assert ev["news"]["records"] == 1
    assert ev["as_of"] == "2025-06-15 00:00:00"


@pytest.mark.parametrize("mode", ["historical", "backtest", "replay"])
def test_historical_modes_fail_closed_without_explicit_asof(mode):
    ctx = {
        "config": {"alternative_data_mode": mode, "local_llm_sentiment_enabled": False},
        "assets": ["600519.SH"],
        "trading_days_dt": [pd.Timestamp("2025-12-31")],
        "asset_ohlcv": {},
    }
    with pytest.raises(ValueError, match="require explicit as_of"):
        build_alternative_signals(ctx)


def test_historical_context_asof_is_accepted_and_audited():
    ctx = {
        "config": {"alternative_data_mode": "backtest", "local_llm_sentiment_enabled": False},
        "assets": ["600519.SH"],
        "as_of_dt": pd.Timestamp("2025-06-30"),
        "trading_days_dt": [pd.Timestamp("2025-12-31")],
        "asset_ohlcv": {},
    }
    evidence = build_alternative_signals(ctx)["alternative_data_evidence"]
    assert evidence["mode"] == "backtest"
    assert evidence["pit_cutoff_explicit"] is True
    assert evidence["as_of"] == "2025-06-30 00:00:00"


def test_advice_backtest_alternative_hook_recorded():
    symbols = ["000001.SZ", "000002.SZ", "000004.SZ"]
    frames = {s: _ohlcv_frame(seed=i).set_index("date") for i, s in enumerate(symbols)}
    decisions = [UniverseDecision(s, True, "TEST_REAL_HISTORY", None, len(frames[s])) for s in symbols]

    def signal_fn(date, symbol):
        return 1.0 if symbol == "000001.SZ" else -1.0

    result = run_advice_portfolio_backtest(
        frames, decisions, years=1, lookback=60, rebalance_every=5, fee_rate=0.0,
        min_assets=2, max_positions=3, min_exposure=0.5, max_exposure=0.9,
        max_holding_days=20, harvest_cooldown_days=1, min_trade_weight=0.0,
        max_daily_turnover=1.0, alternative_signal_fn=signal_fn,
        alternative_signal_rank_weight=0.5,
    )
    signals = result["signals"]
    assert "alternative_signal" in signals.columns
    assert set(signals["alternative_signal"].dropna().unique()) <= {1.0, -1.0, 0.0}
    assert result["returns"] is not None and len(result["returns"]) > 10
