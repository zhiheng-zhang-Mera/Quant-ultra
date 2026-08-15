"""Tests for PIT fundamental factors (Main.fundamental_factors)."""
from __future__ import annotations

import shutil
import sys
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from Main.fundamental_factors import (  # noqa: E402
    FUNDAMENTAL_FIELDS,
    build_fundamental_panels,
    load_all_fundamentals,
    load_fundamentals,
)
from Main.weekly_rotation import RotationParams, precompute_panels, weekly_rotation_backtest  # noqa: E402


def _workspace_tmpdir() -> Path:
    """Temporary directory under the test tree with default permissions.

    ``tempfile.TemporaryDirectory`` is avoided because some sandboxes map its
    POSIX 0o700 mode to a DACL that denies even the creator file access;
    a plain ``os.mkdir`` (default mode) is accessible everywhere.
    """
    base = Path(__file__).parent / ".tmp_fundamentals"
    base.mkdir(exist_ok=True)
    path = base / uuid.uuid4().hex
    path.mkdir()
    return path


def _cleanup_tmpdir(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)


def _synthetic_frames() -> dict:
    dates = pd.bdate_range("2023-01-02", periods=180)
    frames = {}
    for i, sym in enumerate(["600000.SH", "000001.SZ", "600036.SH", "601398.SH", "000002.SZ", "600519.SH", "000858.SZ"]):
        base = 10.0 * (1 + i)
        close = base * np.linspace(1.0, 1.6, len(dates))
        open_ = close * 0.999
        high = close * 1.01
        low = close * 0.99
        volume = np.full(len(dates), 2_000_000.0)
        amount = volume * close
        frames[sym] = pd.DataFrame(
            {
                "date": dates,
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
                "amount": amount,
            }
        ).set_index("date")
    return frames


def test_fundamentals_load_empty_when_missing():
    assert load_fundamentals(Path("does_not_exist.json")) == {}


def test_fundamental_panels_are_point_in_time():
    dates = pd.bdate_range("2024-01-01", periods=20)
    funds = {
        "600000.SH": [
            {"pub_date": "2024-01-10", "stat_date": "2023-12-31", "gpMargin": 0.50, "roeAvg": 0.10, "npMargin": 0.05, "yoyNI": 0.2, "yoyPNI": 0.3},
            {"pub_date": "2024-01-18", "stat_date": "2023-09-30", "gpMargin": 0.60, "roeAvg": 0.12, "npMargin": 0.06, "yoyNI": 0.3, "yoyPNI": 0.4},
        ],
    }
    panels = build_fundamental_panels(funds, dates, ["600000.SH"])
    gp = panels["gp_margin"]["600000.SH"]
    # before the first publication (2024-01-10, index 7): NaN
    assert np.isnan(gp.iloc[:7].max())
    # on/after the first publication -> value 0.50
    assert np.isclose(float(gp.iloc[7]), 0.50)
    # after the second publication (2024-01-18, index 13) -> 0.60
    assert np.isclose(float(gp.iloc[-1]), 0.60)


def test_load_all_fundamentals_merges_by_period():
    import json

    profit = {"600000.SH": [{"pub_date": "2024-04-03", "stat_date": "2023-12-31", "gpMargin": 0.50}]}
    balance = {"600000.SH": [{"pub_date": "2024-04-03", "stat_date": "2023-12-31", "debt_ratio": 0.20}]}
    tmp = _workspace_tmpdir()
    try:
        p = tmp / "profit.json"
        b = tmp / "balance.json"
        p.write_text(json.dumps(profit), encoding="utf-8")
        b.write_text(json.dumps(balance), encoding="utf-8")
        merged = load_all_fundamentals(p, b)
        rec = merged["600000.SH"][0]
        assert rec["gpMargin"] == 0.50
        assert rec["debt_ratio"] == 0.20
    finally:
        _cleanup_tmpdir(tmp)
    assert "debt_ratio" in FUNDAMENTAL_FIELDS


def test_load_fundamentals_top_n_filters_by_ranked_order():
    import json

    data = {"A": [1], "B": [2], "C": [3]}
    tmp = _workspace_tmpdir()
    try:
        p = tmp / "fund.json"
        p.write_text(json.dumps(data), encoding="utf-8")
        assert set(load_fundamentals(p, top_n=2)) == {"A", "B"}
        assert set(load_fundamentals(p, top_n=None)) == {"A", "B", "C"}
    finally:
        _cleanup_tmpdir(tmp)


def test_fundamental_factors_engine_integration(monkeypatch):
    frames = _synthetic_frames()
    funds = {
        "600000.SH": [{"pub_date": "2023-01-01", "stat_date": "2022-12-31", "gpMargin": 0.5, "roeAvg": 0.1, "npMargin": 0.05, "yoyNI": 0.2, "yoyPNI": 0.3}],
        "000001.SZ": [{"pub_date": "2023-01-01", "stat_date": "2022-12-31", "gpMargin": 0.3, "roeAvg": 0.08, "npMargin": 0.04, "yoyNI": 0.1, "yoyPNI": 0.2}],
        "600036.SH": [{"pub_date": "2023-01-01", "stat_date": "2022-12-31", "gpMargin": 0.4, "roeAvg": 0.09, "npMargin": 0.045, "yoyNI": 0.15, "yoyPNI": 0.25}],
    }
    import Main.fundamental_factors as ff
    monkeypatch.setattr(ff, "load_all_fundamentals", lambda *a, **k: funds)

    base = RotationParams(
        rebalance_weekday=None,
        rebalance_days=10,
        min_volume_days=20,
        defensive_hold_assets=(),
        enable_intraweek_stops=True,
    )
    with_fund = RotationParams(
        rebalance_weekday=None,
        rebalance_days=10,
        min_volume_days=20,
        defensive_hold_assets=(),
        enable_intraweek_stops=True,
        fundamental_factors={"gp_margin": 0.10, "yoy_ni": 0.05},
    )
    r0 = weekly_rotation_backtest(frames, base)
    r1 = weekly_rotation_backtest(frames, with_fund)
    assert r0["summary"]["observations"] == r1["summary"]["observations"]
    assert r1["summary"]["final_equity"] > 0
    # the fundamental panels were built
    panel = precompute_panels(frames, with_fund)
    assert panel.fundamental_panels is not None
    assert "gp_margin" in panel.fundamental_panels
