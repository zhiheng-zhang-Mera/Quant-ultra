"""Tests for the open-source factor library and its composite hookup."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from Main.factor_library import FACTOR_SPECS, compute_factor  # noqa: E402
from Main.weekly_rotation import RotationParams, precompute_panels, weekly_rotation_backtest  # noqa: E402


def _synthetic_frames() -> dict:
    dates = pd.bdate_range("2023-01-02", periods=180)
    frames = {}
    for i, sym in enumerate(["600000.SH", "000001.SZ", "510300.SH"]):
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


def test_registry_has_attributed_factors():
    assert "roc20" in FACTOR_SPECS and "rsi14" in FACTOR_SPECS
    for name, spec in FACTOR_SPECS.items():
        assert "source" in spec and "formula" in spec
        assert "qlib" in spec["source"].lower() or "wq" in spec["source"].lower() or "alpha" in spec["source"].lower()


def test_factor_default_off():
    p = RotationParams()
    assert p.extra_factor_weights == {}


def test_factor_is_pit_and_bounded():
    frames = _synthetic_frames()
    panel = precompute_panels(frames, RotationParams())
    rsi = compute_factor(panel, "rsi14", panel.common[120])
    assert rsi is not None and rsi.notna().any()
    assert float(rsi.dropna().min()) >= 0.0 and float(rsi.dropna().max()) <= 100.0
    # PIT: corrupt future closes, earlier factor rows must be unchanged
    frames2 = {s: f.copy() for s, f in frames.items()}
    for f in frames2.values():
        f.loc[f.index[120]:, "close"] *= 5.0
    panel2 = precompute_panels(frames2, RotationParams())
    for name in ("roc20", "rsi14", "std20"):
        a = compute_factor(panel, name, panel.common[90])
        b = compute_factor(panel2, name, panel2.common[90])
        assert np.allclose(a.to_numpy(), b.to_numpy(), equal_nan=True)


def test_extra_factor_weights_run_in_engine():
    frames = _synthetic_frames()
    base = RotationParams(
        rebalance_weekday=None,
        rebalance_days=10,
        min_volume_days=20,
        defensive_hold_assets=(),
        enable_intraweek_stops=True,
    )
    with_factors = RotationParams(
        rebalance_weekday=None,
        rebalance_days=10,
        min_volume_days=20,
        defensive_hold_assets=(),
        enable_intraweek_stops=True,
        extra_factor_weights={"roc20": 0.10, "rsi14": 0.05},
    )
    r0 = weekly_rotation_backtest(frames, base)
    r1 = weekly_rotation_backtest(frames, with_factors)
    assert r0["summary"]["observations"] == r1["summary"]["observations"]
    assert r1["summary"]["final_equity"] > 0
