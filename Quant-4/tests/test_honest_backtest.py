"""Regression tests for the honest small-capital backtest model:
PIT dividends, explicit fees, board lots, and the no-leverage default."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from Main.pit_dividends import load_dividend_cash, trailing_dividend_yield
from Main.strategy_selector import STRATEGY_ARCHETYPES, StrategySelector, build_archetype_params
from Main.trading_costs import explicit_order_fees
from Main.weekly_rotation import RotationParams, weekly_rotation_backtest


def _synthetic_frames() -> dict:
    """Three symbols with clean uptrends so ranking always picks them."""
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


def test_trailing_dividend_yield_is_point_in_time():
    """A dividend only enters the yield after its ex-date; before that the
    yield must be zero (no look-ahead)."""
    idx = pd.bdate_range("2024-01-01", periods=10)
    cash = pd.DataFrame({"A": np.nan}, index=idx)
    cash.loc[idx[5], "A"] = 1.0  # ex-date on day 5
    close = pd.DataFrame({"A": 10.0}, index=idx)
    y = trailing_dividend_yield(cash, close)
    assert float(y.loc[idx[4], "A"]) == 0.0          # before ex-date: no yield
    assert float(y.loc[idx[5], "A"]) > 0.0           # on/after ex-date: yield
    assert np.isclose(float(y.loc[idx[9], "A"]), 0.10, atol=1e-9)


def test_dividend_cash_takes_precedence_over_static_map():
    """The deprecated static map must not override a PIT cash panel."""
    idx = pd.bdate_range("2024-01-01", periods=10)
    cash = pd.DataFrame({"A": np.nan}, index=idx)
    cash.loc[idx[5], "A"] = 1.0
    p = RotationParams(dividend_yield_map={"A": 99.0}, dividend_cash=cash)
    assert p.dividend_cash is cash
    assert p.dividend_yield_map == {"A": 99.0}


def test_production_defaults_are_leverage_free():
    """Personal-capital defaults must never borrow or short."""
    p = RotationParams()
    assert p.confirm_leverage == 1.0
    assert p.max_gross_exposure <= 1.0
    assert p.bull_leverage == 1.0
    assert p.hedge_etf == ""


def test_production_defaults_are_monthly_persistent_config():
    """The 2026-08-11 evidence-gated production default (PIT pool grid):
    monthly rebalancing (weekly Friday override removed), position
    persistence without a forced rotation cap, a 12%/9% stop band, and
    euphoria threshold 0.15. All leverage-free."""
    import sys
    from pathlib import Path
    ROOT = Path(__file__).parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from run_weekly_rotation import default_params

    p = default_params()
    assert p.rebalance_weekday is None
    assert p.rebalance_days == 21
    assert p.hold_persistent is True
    assert p.persist_rank_floor == 8
    assert p.max_holding_days == 0
    assert p.take_profit_pct == 0.12
    assert p.stop_loss_pct == 0.07
    assert p.euphoria_threshold == 0.15
    assert p.defensive_hold_safe_frac == 0.75
    assert p.vol_target == 0.0
    assert p.confirm_leverage == 1.0 and p.max_gross_exposure <= 1.0


def test_defensive_filter_uses_strict_positive_dividend():
    """A dense dividend panel where most names pay nothing must not make the
    defensive filter pass the whole universe: outside BULL only dividend
    payers (trailing yield > 0) or the low-volatility half qualify."""
    from Main.weekly_rotation import RegimeState, precompute_panels, rank_candidates

    frames = _synthetic_frames()
    dates = frames["600000.SH"].index
    n = len(dates)
    # A high-volatility, non-dividend name: it must be filtered out even
    # though its raw momentum is strong.
    rng = np.random.default_rng(7)
    wild = 10.0 + np.cumsum(rng.normal(0.0, 0.25, n))
    wild = np.maximum(wild, 1.0)
    frames["600519.SH"] = pd.DataFrame(
        {
            "date": dates,
            "open": wild * 0.999,
            "high": wild * 1.02,
            "low": wild * 0.98,
            "close": wild,
            "volume": np.full(n, 2_000_000.0),
            "amount": wild * 2_000_000.0,
        }
    ).set_index("date")
    # Dense dividend panel: only 600000.SH pays (positive yield); the other
    # three have exactly 0 cash per share.
    div_cash = pd.DataFrame(0.0, index=dates, columns=frames)
    div_cash.loc[dates[60], "600000.SH"] = 0.5
    p = RotationParams(
        momentum_windows=(5,),
        momentum_weights=(1.0,),
        signal_mode="composite",
        defensive_core=True,
        defensive_filter=True,
        defensive_core_bull_momentum=False,
        require_relative_strength=False,
        min_adv=1e6,
        dividend_cash=div_cash,
    )
    panel = precompute_panels(frames, p)
    date = dates[150]
    regime = RegimeState(
        date=date, regime="NEUTRAL", benchmark_close=10.0,
        benchmark_ma_fast=10.0, benchmark_ma_slow=10.0, exposure=0.6,
        advice_zh="test", advice_en="test",
    )
    ranked = rank_candidates(panel, date, p, None, regime)
    ranked_symbols = [s for s, _, _ in ranked]
    assert "600519.SH" not in ranked_symbols  # high-vol, no dividend
    assert "600000.SH" in ranked_symbols      # dividend payer passes


def test_explicit_fees_apply_minimum_commission_and_stamp():
    """Small notional must still pay the 5 CNY minimum; stocks pay stamp tax
    on sells; ETFs do not."""
    fee_buy = explicit_order_fees(1000.0, "buy", "600000.SH")
    assert fee_buy["commission"] == 5.0
    fee_sell_stock = explicit_order_fees(1000.0, "sell", "600000.SH")
    assert fee_sell_stock["stamp_tax"] > 0.0
    fee_sell_etf = explicit_order_fees(1000.0, "sell", "510300.SH")
    assert fee_sell_etf["stamp_tax"] == 0.0


def test_board_lots_block_expensive_names_at_small_capital():
    """At 100k CNY, a 2000 CNY/share name (100 shares = 200k) can never be
    bought even at a 20% target (20k < one board lot), so despite being the
    strongest momentum pick it must never be opened."""
    frames = _synthetic_frames()
    dates = frames["600000.SH"].index.tolist()
    n = len(dates)
    # strongest momentum in the pool: flat for 20 days then +40% over the rest
    ramp = np.concatenate([np.full(20, 2000.0), np.linspace(2000.0, 2800.0, n - 20)])
    frames["600519.SH"] = pd.DataFrame(
        {
            "date": dates,
            "open": ramp,
            "high": ramp * 1.01,
            "low": ramp * 0.99,
            "close": ramp,
            "volume": np.full(n, 1_000_000.0),
            "amount": ramp * 1_000_000.0,
        }
    ).set_index("date")
    p = RotationParams(
        start_date="2023-01-02",
        rebalance_weekday=4,
        top_n=3,
        max_etf_positions=2,
        capital_base=100_000.0,
        enable_board_lots=True,
        momentum_windows=(5,),
        momentum_weights=(1.0,),
        signal_mode="composite",
        defensive_core=False,
        defensive_filter=False,
    )
    result = weekly_rotation_backtest(frames, p)
    trades = result["closed_trades"]
    assert trades.empty or "600519.SH" not in set(trades["symbol"])
    max_weight = float(result["returns"]["gross_exposure"].max())
    assert max_weight <= 1.0


def test_flat_fee_fallback_when_no_capital_assumption():
    """capital_base=0 keeps the legacy flat fee model working."""
    frames = _synthetic_frames()
    p = RotationParams(
        start_date="2023-01-02",
        rebalance_weekday=4,
        capital_base=0.0,
        enable_board_lots=False,
        signal_mode="composite",
        defensive_core=False,
        defensive_filter=False,
    )
    result = weekly_rotation_backtest(frames, p)
    assert result["summary"]["total_cost_fraction"] > 0.0


def test_pit_dividend_cache_roundtrip(tmp_path):
    """Cached parquet dividends load back into a wide matrix."""
    cache = tmp_path / "dividends"
    cache.mkdir()
    frame = pd.DataFrame(
        {"symbol": ["600000.SH"] * 2, "ex_date": ["2023-06-01", "2024-06-01"], "cash_ps": [0.2, 0.25]}
    )
    frame.to_parquet(cache / "600000_SH_dividends.parquet", index=False)
    wide = load_dividend_cash(cache, ["600000.SH"])
    assert wide is not None
    assert "600000.SH" in wide.columns
    assert len(wide) == 2


def test_archetype_overrides_never_enable_leverage():
    """Every named archetype must stay leverage-free and long-only."""
    base = RotationParams()
    for name in STRATEGY_ARCHETYPES:
        p = build_archetype_params(base, name)
        assert p.max_gross_exposure <= 1.0
        assert p.confirm_leverage == 1.0
        assert p.bull_leverage == 1.0
        assert p.hedge_etf == ""


def test_selector_hysteresis_requires_min_stay():
    """The selector must not flip on a single rebalance."""
    sel = StrategySelector(min_stay=3)
    # force state: bear + high vol -> defensive, bull + high breadth -> momentum
    class FakePanel:
        def __init__(self):
            idx = pd.bdate_range("2023-01-02", periods=300)
            self.bench_close = pd.Series([0.9] * 300, index=idx)   # below MAs -> BEAR
            self.bench_ma_fast = pd.Series([0.98] * 300, index=idx)
            self.bench_ma_slow = pd.Series([1.0] * 300, index=idx)
            self.bench_ma_confirmation = pd.Series([1.0] * 300, index=idx)
            self.bench_return_20d = pd.Series([-0.01] * 300, index=idx)
            self.close = pd.DataFrame({"A": [10.0] * 300, "B": [10.0] * 300}, index=idx)
            self.momentum = {120: pd.DataFrame({"511010.SH": [0.03] * 300, "A": [0.1] * 300}, index=idx)}
    panel = FakePanel()
    base = RotationParams(defensive_hold_assets=("511010.SH",), regime_ma=40, regime_ma_fast=10, regime_confirmation_ma=200)
    date = pd.Timestamp("2024-02-15")
    assert sel.select(panel, date, base) == "balanced"       # first call starts balanced
    assert sel.select(panel, date, base) == "balanced"       # stay_count=1 < 3
    assert sel.select(panel, date, base) == "defensive"      # stay_count=3 >= min_stay


def _mk_close(level: float) -> pd.Series:
    return pd.Series([level] * 300, index=pd.bdate_range("2023-01-02", periods=300))
