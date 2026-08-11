"""Tests for the decoupled four-sleeve layer (Main.sleeve_allocation):
constant-mix combination rules, threshold/cost behaviour, drift behaviour,
and end-to-end orchestration of independent engine books."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from Main.sleeve_allocation import (  # noqa: E402
    DEFAULT_SLEEVES,
    SleevePortfolioConfig,
    SleeveSpec,
    combine_sleeves,
    run_sleeve_portfolio,
)
from Main.strategy_selector import STRATEGY_ARCHETYPES, build_archetype_params  # noqa: E402
from Main.weekly_rotation import RotationParams  # noqa: E402


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


def test_combine_restores_targets_on_rebalance():
    """Constant-mix: after a rebalance date the weights return to targets and
    the rebalance flow is charged at the configured cost rate."""
    idx = pd.bdate_range("2024-01-01", periods=30)
    rng = np.random.default_rng(7)
    R = pd.DataFrame(
        {
            "a": rng.normal(0.001, 0.01, len(idx)),
            "b": rng.normal(0.001, 0.02, len(idx)),
        },
        index=idx,
    )
    combined, w_hist, meta = combine_sleeves(
        R, [0.4, 0.6], rebalance_days=10, threshold=0.0, cost_rate=0.0017
    )
    # rebalance happens at rows 0, 10, 20 (i % 10 == 0, i < n-1)
    assert meta["rebalances"] == 3
    assert combined["turnover"].iloc[10] > 0
    assert np.isclose(combined["cost"].iloc[10], combined["turnover"].iloc[10] * 0.0017)
    for i in (0, 10, 20):
        assert np.isclose(w_hist.iloc[i]["a"], 0.4, atol=1e-12)
        assert np.isclose(w_hist.iloc[i]["b"], 0.6, atol=1e-12)
    # every row's weights sum to one
    assert np.allclose(w_hist.sum(axis=1).to_numpy(), 1.0, atol=1e-9)


def test_combine_threshold_skips_small_drift():
    """With a threshold, tiny drift is left alone (no flow, no cost)."""
    idx = pd.bdate_range("2024-01-01", periods=25)
    R = pd.DataFrame({"a": 0.0001, "b": 0.0001}, index=idx)
    combined, w_hist, meta = combine_sleeves(
        R, [0.5, 0.5], rebalance_days=10, threshold=0.01, cost_rate=0.01
    )
    assert meta["rebalances"] == 0
    assert combined["turnover"].abs().sum() == 0.0
    assert combined["cost"].abs().sum() == 0.0


def test_drift_never_rebalances():
    """Without threshold triggers the weights simply drift (buy-and-hold)."""
    idx = pd.bdate_range("2024-01-01", periods=20)
    R = pd.DataFrame(
        {"a": np.full(20, 0.005), "b": np.full(20, -0.001)},
        index=idx,
    )
    combined, w_hist, meta = combine_sleeves(
        R, [0.5, 0.5], rebalance_days=5, threshold=99.0, cost_rate=0.0
    )
    assert meta["rebalances"] == 0
    # drifted weight of the winner grows monotonically
    assert w_hist["a"].iloc[-1] > w_hist["a"].iloc[0] > 0.5


def test_default_sleeves_are_40_30_20_10():
    weights = [s.weight for s in DEFAULT_SLEEVES]
    assert weights == [0.40, 0.30, 0.20, 0.10]
    assert [s.archetype for s in DEFAULT_SLEEVES] == ["safe", "balanced", "momentum", "sprint"]


def test_sprint_archetype_exists_and_builds():
    assert "sprint" in STRATEGY_ARCHETYPES
    p = build_archetype_params(RotationParams(), "sprint")
    assert p.top_n == 2
    assert p.take_profit_pct == 0.15
    assert p.stop_loss_pct == 0.08
    assert p.defensive_filter is False
    assert p.breakout_weight == 0.35


def test_run_sleeve_portfolio_end_to_end():
    frames = _synthetic_frames()
    base = RotationParams(
        rebalance_weekday=None,
        rebalance_days=10,
        min_volume_days=20,
        defensive_hold_assets=(),
        dividend_cash=None,
    )
    result = run_sleeve_portfolio(
        frames,
        sleeves=DEFAULT_SLEEVES,
        base_params=base,
        config=SleevePortfolioConfig(rebalance_days=10, threshold=0.02),
    )
    s = result["summary"]
    assert s["observations"] > 0
    assert len(result["sleeves"]) == 4
    for name, res in result["sleeves"].items():
        assert res["summary"]["observations"] == s["observations"]
    final = result["meta"]["final_weights"]
    assert abs(sum(final.values()) - 1.0) < 1e-9
    assert "sleeve_rebalance_cost" in s
    assert "sleeve_rebalances" in s
