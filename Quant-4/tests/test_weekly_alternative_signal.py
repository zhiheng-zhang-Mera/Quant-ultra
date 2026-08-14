"""Production weekly/sleeve wiring for point-in-time alternative signals."""
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Main.sleeve_allocation import SleevePortfolioConfig, SleeveSpec, run_sleeve_portfolio
from Main.weekly_rotation import RotationParams, detect_regime, precompute_panels, rank_candidates, weekly_rotation_backtest
from run_weekly_rotation import load_alternative_signal_panel
from Phase_3.alternative_data_contract import SIGNAL_CONTRACT_VERSION


def _frames(n=190, symbols=6):
    dates = pd.bdate_range("2024-01-02", periods=n)
    result = {}
    for i in range(symbols):
        close = 10.0 * np.exp(np.linspace(0, 0.05 + i * 0.01, n))
        result[f"00000{i + 1}.SZ"] = pd.DataFrame({
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.full(n, 10_000_000.0),
            "amount": close * 10_000_000.0,
        }, index=dates)
    return result


def _params(**kwargs):
    base = RotationParams(
        regime_ma=20, regime_ma_fast=5, min_volume_days=20,
        trend_filter_long=0, min_adv=0, require_relative_strength=False,
        rebalance_days=5, rebalance_weekday=None, top_n=2,
        max_etf_positions=0, enable_board_lots=False,
    )
    return replace(base, **kwargs)


def test_signal_loader_and_weekly_rank_use_exact_asof(tmp_path):
    frames = _frames()
    date = next(reversed(frames["000001.SZ"].index[:-1]))
    source = tmp_path / "signals.csv"
    pd.DataFrame([
        {"as_of": date, "symbol": "000001.sz", "alternative_signal": 1.0,
         "source_sha256": "a" * 64, "contract_version": SIGNAL_CONTRACT_VERSION},
        {"as_of": date, "symbol": "000006.SZ", "alternative_signal": -1.0,
         "source_sha256": "a" * 64, "contract_version": SIGNAL_CONTRACT_VERSION},
    ]).to_csv(source, index=False)
    signal = load_alternative_signal_panel(source)
    params = _params(alternative_signal_weight=1.0, alternative_signal_panel=signal)
    panel = precompute_panels(frames, params)
    regime = detect_regime(panel, date, params)
    ranked = rank_candidates(panel, date, params, regime=regime)
    assert ranked[0][0] == "000001.SZ"
    assert panel.alternative_signal.loc[date, "000006.SZ"] == -1.0
    assert signal.attrs["provenance"]["source_sha256"] == ["a" * 64]
    prior = panel.common[panel.common.get_loc(date) - 1]
    assert pd.isna(panel.alternative_signal.loc[prior, "000001.SZ"]), "engine must not backfill a future signal"


def test_zero_weight_is_byte_compatible_with_baseline():
    frames = _frames()
    dates = frames["000001.SZ"].index
    signal = pd.DataFrame(1.0, index=dates, columns=frames)
    base = _params()
    with_disabled_signal = replace(base, alternative_signal_panel=signal, alternative_signal_weight=0.0)
    r0 = weekly_rotation_backtest(frames, base)["returns"]
    r1 = weekly_rotation_backtest(frames, with_disabled_signal)["returns"]
    pd.testing.assert_frame_equal(r0, r1)


def test_future_signal_mutation_cannot_change_prior_engine_results():
    frames = _frames(n=230)
    dates = frames["000001.SZ"].index
    cutoff = dates[175]
    base_signal = pd.DataFrame(0.0, index=dates, columns=frames)
    base_signal.loc[:cutoff, "000001.SZ"] = 1.0
    mutated_signal = base_signal.copy()
    mutated_signal.loc[mutated_signal.index > cutoff, :] = -1.0
    mutated_signal.loc[mutated_signal.index > cutoff, "000006.SZ"] = 1.0
    params = _params(alternative_signal_panel=base_signal, alternative_signal_weight=0.3)
    mutated = replace(params, alternative_signal_panel=mutated_signal)
    first = weekly_rotation_backtest(frames, params)["returns"]
    second = weekly_rotation_backtest(frames, mutated)["returns"]
    pd.testing.assert_frame_equal(first.loc[:cutoff], second.loc[:cutoff])


def test_sleeve_orchestrator_preserves_signal_config(monkeypatch):
    frames = _frames()
    dates = frames["000001.SZ"].index
    signal = pd.DataFrame(0.2, index=dates, columns=frames)
    seen = []

    def fake_engine(_frames, params, regime_detector_kwargs=None):
        seen.append(params)
        idx = dates[-5:]
        returns = pd.DataFrame({
            "strategy_return": 0.0, "benchmark_return": 0.0,
            "gross_exposure": 0.0, "turnover": 0.0, "cost": 0.0,
            "regime": "NEUTRAL",
        }, index=idx)
        regimes = pd.DataFrame({"regime": "NEUTRAL"}, index=idx)
        return {"returns": returns, "regimes": regimes, "summary": {}, "closed_trades": pd.DataFrame()}

    monkeypatch.setattr("Main.sleeve_allocation.weekly_rotation_backtest", fake_engine)
    monkeypatch.setattr("Main.sleeve_allocation.summarize", lambda *args, **kwargs: {})
    run_sleeve_portfolio(
        frames,
        sleeves=(SleeveSpec("balanced", 1.0, "balanced"),),
        base_params=_params(alternative_signal_panel=signal, alternative_signal_weight=0.2),
        config=SleevePortfolioConfig(),
    )
    assert len(seen) == 1
    assert seen[0].alternative_signal_weight == pytest.approx(0.2)
    pd.testing.assert_frame_equal(seen[0].alternative_signal_panel, signal)
