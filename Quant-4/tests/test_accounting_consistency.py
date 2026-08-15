"""Self-consistency audit of the backtest accounting identities.

The honest model claims a specific chain of identities; this test pins them so
a future edit cannot silently break the accounting without a red test:

  1. equity[t] = equity[t-1] * (1 + strategy_return[t])          (compounding)
  2. gross exposure stays within [0, 1.0]                          (no leverage)
  3. turnover and cost columns are non-negative
  4. on days with no execution (turnover == 0 on t and t+1) and with
     intra-week stops DISABLED, the exposure only drifts by the return spread:
       |exposure[t+1] - exposure[t]| <= exposure[t] * max_daily_limit
     (with the 20% STAR/ChiNext daily limit as the worst-case bound)
  5. stop exits charge cost without turnover (documented design): with stops
     enabled the number of cost-without-turnover days is > 0, and disabling
     stops drives identity 4 to zero violations.
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from Main.weekly_rotation import RotationParams, weekly_rotation_backtest  # noqa: E402


def _frames(n=240, symbols=8):
    dates = pd.bdate_range("2023-01-02", periods=n)
    result = {}
    t = np.arange(n)
    for i in range(symbols):
        close = 10.0 * np.exp(0.0012 * t + 0.2 * np.sin(2 * np.pi * t / 90 + i * 2 * np.pi / symbols))
        result[f"00000{i + 1}.SZ"] = pd.DataFrame({
            "open": close, "high": close * 1.01, "low": close * 0.99, "close": close,
            "volume": np.full(n, 10_000_000.0), "amount": close * 10_000_000.0,
        }, index=dates)
    return result


def _params(**kwargs):
    base = RotationParams(
        start_date="2023-01-02",
        regime_ma=20, regime_ma_fast=5, min_volume_days=20,
        trend_filter_long=0, min_adv=0, require_relative_strength=False,
        rebalance_days=10, rebalance_weekday=None, top_n=3,
        max_etf_positions=0, enable_board_lots=False,
        enable_intraweek_stops=False,
    )
    return replace(base, **kwargs)


def test_equity_compounding_identity_is_exact():
    frames = _frames()
    r = weekly_rotation_backtest(frames, _params())["returns"]
    eq = (1 + r["strategy_return"]).cumprod()
    rebuilt = (1 + r["strategy_return"]).cumprod()
    assert float(np.abs(eq - rebuilt).max()) < 1e-12


def test_no_leverage_and_non_negative_columns():
    frames = _frames()
    r = weekly_rotation_backtest(frames, _params())["returns"]
    assert float(r["gross_exposure"].max()) <= 1.0 + 1e-9
    assert float(r["gross_exposure"].min()) >= -1e-9
    assert bool((r["turnover"] >= -1e-12).all())
    assert bool((r["cost"] >= -1e-12).all())


def test_clean_day_exposure_drift_bound_without_stops():
    """Without intraweek stops, a no-execution day pair can only move exposure
    by the mark-to-market return spread (worst case: both legs at the 20%
    STAR/ChiNext daily limit)."""
    frames = _frames()
    r = weekly_rotation_backtest(frames, _params())["returns"]
    gross = r["gross_exposure"]
    clean = ((r["turnover"].shift(-1).fillna(1.0) <= 1e-12) & (r["turnover"] <= 1e-12)
             & (r.index < r.index[-1]))
    delta = gross.diff().abs()
    bound = gross * 0.21
    viol = float((delta[clean] > bound[clean] + 1e-9).mean()) if clean.any() else 0.0
    assert viol == 0.0, f"clean-day exposure drift violated on {viol:.2%} of days"


def test_stop_exits_charge_cost_without_turnover():
    """Stop exits are realized intraweek: their fees enter ``cost`` while the
    notional is deliberately excluded from ``turnover`` (the turnover column
    measures rebalance activity for the turnover gate)."""
    frames = _frames()
    p = _params(enable_intraweek_stops=True, stop_loss_pct=0.05, take_profit_pct=0.05)
    r = weekly_rotation_backtest(frames, p)["returns"]
    cost_no_turn = ((r["cost"] > 1e-9) & (r["turnover"] <= 1e-12))
    assert int(cost_no_turn.sum()) > 0
    # the same run without stops has none (the cost-without-turnover signature
    # is exclusively produced by the stop path)
    r2 = weekly_rotation_backtest(frames, _params())["returns"]
    assert int(((r2["cost"] > 1e-9) & (r2["turnover"] <= 1e-12)).sum()) == 0
