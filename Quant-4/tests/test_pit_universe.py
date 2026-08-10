"""Tests for the survivorship-free PIT universe and its engine integration:
master-list membership, alive-mask filtering, and forced exits of terminal
(delisted/absorbed) holdings at the last available close."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from Main.pit_universe import (  # noqa: E402
    alive_matrix,
    ever_alive_stocks,
    to_symbol,
    universe_coverage,
)
from Main.weekly_rotation import RotationParams, weekly_rotation_backtest  # noqa: E402


def _master_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["600000.SH", "600001.SH", "000001.SZ", "830001.BJ"],
            "code_name": ["bank-a", "delisted-b", "bank-c", "bse-d"],
            "ipo_date": pd.to_datetime(["2010-01-01"] * 4),
            "out_date": pd.to_datetime([pd.NaT, "2018-06-30", pd.NaT, pd.NaT]),
            "type": ["1", "1", "1", "1"],
            "status": ["1", "0", "1", "1"],
        }
    )


def test_to_symbol_normalizes_baostock_codes():
    assert to_symbol("sh.600000") == "600000.SH"
    assert to_symbol("sz.000001") == "000001.SZ"


def test_ever_alive_stocks_include_delisted_and_exclude_bse():
    master = _master_df()
    syms = ever_alive_stocks(master, "2016-01-01", "2026-01-01")
    assert "600001.SH" in syms      # delisted mid-window is still a candidate
    assert "600000.SH" in syms
    assert "000001.SZ" in syms
    assert "830001.BJ" not in syms  # BSE prefix excluded
    later = ever_alive_stocks(master, "2020-01-01", "2026-01-01")
    assert "600001.SH" not in later  # already dead before the window


def test_alive_matrix_membership_boundaries():
    master = _master_df()
    dates = pd.to_datetime(["2016-01-01", "2018-06-29", "2018-06-30", "2018-07-01"])
    am = alive_matrix(master, dates, ["600000.SH", "600001.SH"])
    assert bool(am.at[dates[0], "600001.SH"])
    assert bool(am.at[dates[2], "600001.SH"])      # out date inclusive (last bar)
    assert not bool(am.at[dates[3], "600001.SH"])  # dead the next day
    assert bool(am.at[dates[3], "600000.SH"])      # survivor unaffected


def test_universe_coverage_reports_delisted_gap():
    master = _master_df()
    frames = {"600000.SH": pd.DataFrame({"date": pd.to_datetime(["2016-01-01"])})}
    cov = universe_coverage(frames, master, "2016-01-01", "2026-01-01")
    assert cov["expected_stocks"] == 3          # BSE excluded from expectation
    assert cov["delisted_expected"] == 1        # 600001.SH
    assert cov["delisted_covered"] == 0
    assert cov["covered_stocks"] == 1


def _frames_with_terminal_name() -> tuple[dict, pd.DataFrame]:
    """Two stocks plus an ETF; '600001.SH' dies at day 60 (data ends there)."""
    dates = pd.bdate_range("2023-01-02", periods=200)
    n = len(dates)
    frames: dict = {}
    for sym, start, end in [("600000.SH", 10.0, 13.0), ("000001.SZ", 10.0, 12.0)]:
        close = start * np.linspace(1.0, end / start, n)
        frames[sym] = pd.DataFrame(
            {
                "date": dates,
                "open": close * 0.999,
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "volume": np.full(n, 2_000_000.0),
                "amount": close * 2_000_000.0,
            }
        ).set_index("date")
    sub = dates[:60]
    ramp = np.linspace(10.0, 18.0, 60)  # strongest momentum, then it dies
    frames["600001.SH"] = pd.DataFrame(
        {
            "date": sub,
            "open": ramp * 0.999,
            "high": ramp * 1.01,
            "low": ramp * 0.99,
            "close": ramp,
            "volume": np.full(60, 2_000_000.0),
            "amount": ramp * 2_000_000.0,
        }
    ).set_index("date")
    frames["510300.SH"] = pd.DataFrame(
        {
            "date": dates,
            "open": np.full(n, 4.0),
            "high": np.full(n, 4.05),
            "low": np.full(n, 3.95),
            "close": np.full(n, 4.0),
            "volume": np.full(n, 20_000_000.0),
            "amount": np.full(n, 80_000_000.0),
        }
    ).set_index("date")
    master = pd.DataFrame(
        {
            "symbol": ["600000.SH", "000001.SZ", "600001.SH", "510300.SH"],
            "ipo_date": pd.to_datetime(["2010-01-01"] * 4),
            "out_date": pd.to_datetime([pd.NaT, pd.NaT, sub[-1], pd.NaT]),
            "type": ["1", "1", "1", "4"],
            "status": ["1", "1", "0", "1"],
        }
    )
    alive = alive_matrix(master, dates, ["600000.SH", "000001.SZ", "600001.SH"])
    alive["510300.SH"] = True
    return frames, alive


def _params() -> RotationParams:
    return RotationParams(
        start_date="2023-01-02",
        rebalance_weekday=4,
        rebalance_days=5,
        top_n=3,
        max_etf_positions=2,
        momentum_windows=(5,),
        momentum_weights=(1.0,),
        signal_mode="composite",
        defensive_core=False,
        defensive_filter=False,
        require_relative_strength=False,
        min_adv=1e6,
        hold_persistent=False,
        enable_intraweek_stops=False,
        capital_base=100_000.0,
    )


def test_terminal_holding_is_force_exited_at_last_close():
    """A held name whose PIT membership ends must be sold at its last close
    with a recorded closed trade (no zombie position)."""
    frames, alive = _frames_with_terminal_name()
    params = _params()
    params.alive_mask = alive
    result = weekly_rotation_backtest(frames, params)
    trades = result["closed_trades"]
    assert not trades.empty
    dead = trades[trades["symbol"] == "600001.SH"]
    assert not dead.empty
    last_close = float(frames["600001.SH"]["close"].iloc[-1])
    assert np.isclose(float(dead.iloc[-1]["exit"]), last_close, atol=1e-9)
    assert not trades[trades["symbol"] == "600001.SH"]["exit_date"].isna().any()


def test_without_alive_mask_no_force_exit_but_also_no_candidate():
    """Without a mask the dead name is not selected after data ends (NaN), but
    the engine keeps the stale weight (no recorded exit) - demonstrating why
    the PIT mask is required for honest forced exits."""
    frames, _ = _frames_with_terminal_name()
    result = weekly_rotation_backtest(frames, _params())
    trades = result["closed_trades"]
    dead = trades[trades["symbol"] == "600001.SH"] if not trades.empty else pd.DataFrame()
    assert dead.empty


def test_pit_mode_keeps_etfs_alive_in_mask():
    """ETFs (which are not in the stock master list) must be marked alive so
    the selection layer can still hold ETF seats."""
    frames, alive = _frames_with_terminal_name()
    assert bool(alive.at[alive.index[-1], "510300.SH"])
    assert not bool(alive.at[alive.index[-1], "600001.SH"])
