"""Tests for the R19.7 orthogonal-factor additions (crowding proxies)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Main.factor_library import compute_factor, compute_factors  # noqa: E402


def _panel(n=120, symbols=6):
    dates = pd.bdate_range("2024-01-02", periods=n)
    amount = pd.DataFrame({f"00000{i + 1}.SZ": np.linspace(1e8, 1e8 + i * 1e7, n) for i in range(symbols)}, index=dates)
    close = pd.DataFrame({f"00000{i + 1}.SZ": 10.0 + i * 0.1 for i in range(symbols)}, index=dates)
    volume = pd.DataFrame({f"00000{i + 1}.SZ": 1e6 for i in range(symbols)}, index=dates)
    from Main.weekly_rotation import FeaturePanel

    return FeaturePanel(common=dates, symbols=list(close.columns), close=close, volume=volume,
                        amount=amount, momentum={}, volatility=close, trend=pd.DataFrame(True, index=dates, columns=close.columns),
                        volume_ratio=pd.DataFrame(1.0, index=dates, columns=close.columns), adv20=close,
                        div_yield=pd.DataFrame(0.0, index=dates, columns=close.columns),
                        ma20=None, ma60=None, bench_return_20d=pd.Series(0.01, index=dates),
                        bench_close=pd.Series(10.0, index=dates), bench_ma_fast=pd.Series(10.0, index=dates),
                        bench_ma_slow=pd.Series(10.0, index=dates), bench_ma_confirmation=pd.Series(np.nan, index=dates),
                        bench_ma_trend=pd.Series(10.0, index=dates), bench_shock_z=pd.Series(0.0, index=dates),
                        stop_band=close, take_band=close, breakout=None, max_ret=None, illiquidity=None,
                        fundamental_panels=None, alternative_signal=None, alternative_signal_governance=None)


def test_amount_share_is_cross_sectional_share():
    panel = _panel()
    row = compute_factor(panel, "amount_share", panel.common[10])
    assert row is not None
    total = row.sum()
    assert abs(total - 1.0) < 1e-9, "amount_share must sum to 1 across the cross-section"
    # symbol 0 has the smallest amount -> smallest share
    assert row.idxmin() == panel.symbols[0]
    assert row.idxmax() == panel.symbols[-1]


def test_amount_share_inv_is_negation():
    panel = _panel()
    pos = compute_factor(panel, "amount_share", panel.common[10])
    inv = compute_factor(panel, "amount_share_inv", panel.common[10])
    pd.testing.assert_series_equal(pos, -inv)
