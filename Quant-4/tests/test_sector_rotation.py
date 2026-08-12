"""Tests for the sector-momentum factor (Main.sector_rotation)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from Main.sector_rotation import industry_momentum_series, load_sector_map  # noqa: E402
from Main.weekly_rotation import RotationParams, precompute_panels, weekly_rotation_backtest  # noqa: E402


def _synthetic_frames() -> dict:
    dates = pd.bdate_range("2023-01-02", periods=180)
    frames = {}
    symbols = [
        "600000.SH", "600036.SH", "601398.SH", "601288.SH", "601988.SH",  # 银行
        "000001.SZ", "000002.SZ", "600048.SH", "001979.SZ",               # 地产
        "600519.SH", "000858.SZ",                                          # 白酒
        "510300.SH",                                                        # 综合
    ]
    for i, sym in enumerate(symbols):
        base = 10.0 * (1 + i)
        # different per-symbol growth so industries have distinct momentum
        growth = 0.3 + 0.08 * i
        close = base * np.linspace(1.0, 1.0 + growth, len(dates))
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


def test_sector_map_loads_empty_when_missing():
    assert load_sector_map(Path("does_not_exist.json")) == {}


def test_industry_momentum_groups_and_z_scores():
    frames = _synthetic_frames()
    panel = precompute_panels(frames, RotationParams())
    sector_map = {
        "600000.SH": "银行", "600036.SH": "银行", "601398.SH": "银行",
        "601288.SH": "银行", "601988.SH": "银行",
        "000001.SZ": "地产", "000002.SZ": "地产", "600048.SH": "地产",
        "001979.SZ": "地产",
        "600519.SH": "白酒", "000858.SZ": "白酒",
    }
    row = industry_momentum_series(panel, panel.common[120], sector_map)
    assert row is not None
    # same-industry symbols share the same factor value
    assert np.isclose(row["600000.SH"], row["600036.SH"], atol=1e-9)
    assert np.isclose(row["000001.SZ"], row["000002.SZ"], atol=1e-9)
    # unmapped symbols fall back to the mean of the mapped industries
    mapped_mean = float(np.mean([row[s] for s in ("600000.SH", "000001.SZ", "600519.SH")]))
    assert abs(float(row["510300.SH"]) - mapped_mean) < 1e-9
    # PIT: corrupting future closes must not change the earlier factor row
    frames2 = {s: f.copy() for s, f in frames.items()}
    for f in frames2.values():
        f.loc[f.index[130]:, "close"] *= 3.0
    panel2 = precompute_panels(frames2, RotationParams())
    row2 = industry_momentum_series(panel2, panel2.common[120], sector_map)
    assert np.allclose(row.to_numpy(), row2.to_numpy(), equal_nan=True)


def test_sector_weight_default_off_and_engine_runs():
    p = RotationParams()
    assert p.sector_momentum_weight == 0.0
    frames = _synthetic_frames()
    base = RotationParams(
        rebalance_weekday=None,
        rebalance_days=10,
        min_volume_days=20,
        defensive_hold_assets=(),
    )
    with_sector = RotationParams(
        rebalance_weekday=None,
        rebalance_days=10,
        min_volume_days=20,
        defensive_hold_assets=(),
        sector_momentum_weight=0.15,
    )
    # the real sector map may not exist in test env; engine must still run
    r0 = weekly_rotation_backtest(frames, base)
    r1 = weekly_rotation_backtest(frames, with_sector)
    assert r0["summary"]["observations"] == r1["summary"]["observations"]
    assert r1["summary"]["final_equity"] > 0
