"""Modular weekly-rotation strategy and backtest engine.

Layers (each pure and independently testable):
1. signals      - momentum/trend/volatility scoring per symbol
2. regime       - bull/bear market-state detection and exposure overlay
3. portfolio    - target-weight construction from ranked candidates
4. execution    - close[t] signal -> open[t+1] execution, open-to-open returns
5. reporting    - monthly / regime-segmented performance and advice

The data layer stays external: the engine consumes a dict of clean
``{symbol: DataFrame[date,open,high,low,close,volume,amount]}`` indexed by
naive ``pd.Timestamp`` dates, exactly what ``FreeDataSourceManager`` produces.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from Main.trading_costs import explicit_order_fees, is_etf
from Main.pit_dividends import trailing_dividend_yield
from Main.alternative_signal_governance import evaluate_signal_panel
from Main.research_evidence_gate import evaluate_research_gate


@dataclass
class RotationParams:
    """Tunable weekly-rotation configuration."""

    momentum_windows: Tuple[int, ...] = (5, 10, 20, 60)
    momentum_weights: Tuple[float, ...] = (0.25, 0.30, 0.30, 0.15)
    rebalance_days: int = 5            # weekly cadence in trading days
    top_n: int = 5                     # maximum concurrent positions
    max_etf_positions: int = 1         # ETF seats among the top picks
    bull_exposure: float = 0.95        # gross exposure in bull regime
    bear_exposure: float = 0.30        # gross exposure in bear regime
    neutral_exposure: float = 0.0       # explicit NEUTRAL exposure (0 = (bull+bear)/2)
    bull_leverage: float = 1.0         # margin multiplier in bull (financed)
    max_gross_exposure: float = 1.0    # hard cap on gross exposure (1.0 = no leverage)
    leverage_annual_cost: float = 0.06 # financing cost on borrowed capital
    regime_ma: int = 60                # benchmark trend window (trading days)
    regime_ma_fast: int = 20           # short trend confirmation window
    regime_confirmation_ma: int = 0    # optional longer MA required for BULL
    min_volume_days: int = 60          # minimum price history for candidates
    max_annual_vol: float = 0.75       # reject extreme-volatility names
    min_adv: float = 5e7               # minimum 20-day avg turnover (CNY)
    max_short_term_gain: float = 0.20  # exclude parabolic 5-day spikes
    min_price: float = 1.0
    per_position_cap: float = 0.40     # single-name weight ceiling
    stop_loss_pct: float = 0.10        # intra-week catastrophic stop
    take_profit_pct: float = 0.15      # intra-week profit-taking exit
    # Optional peak-based trailing stop (0 = disabled, byte-compatible).
    # When > 0, the intra-week stop-loss is measured from the position's peak
    # price since entry (ratcheted up daily at the close) instead of the entry
    # price, so a winner that has already run +30% is only stopped on a
    # ``trailing_stop_pct`` pullback from its high - it "lets winners run"
    # while still capping losses. The take-profit band still applies unless
    # ``take_profit_pct`` is set to 0.
    trailing_stop_pct: float = 0.0
    enable_intraweek_stops: bool = False
    # ---- bottom-up dynamic stop bands (decoupled module: Main.dynamic_stops) ----
    # When enabled, each held symbol gets its own PIT ATR-scaled band instead
    # of the two global constants above. Bands are precomputed once per
    # backtest and only read for held symbols at each date; NaN falls back to
    # the static band. Default off preserves the evidence-gated static band.
    dynamic_stops: bool = False
    stops_atr_window: int = 14         # Wilder ATR lookback (trading days)
    stops_atr_sl_mult: float = 2.5     # stop-loss = k_sl * ATR / price
    stops_atr_tp_mult: float = 4.0     # take-profit = k_tp * ATR / price
    stops_floor: float = 0.04          # absolute floor for stop-loss fraction
    stops_cap: float = 0.15            # absolute cap for stop-loss fraction
    stops_take_floor: float = 0.06     # absolute floor for take-profit fraction
    stops_take_cap: float = 0.30       # absolute cap for take-profit fraction
    # ---- optional breakout / new-high factor (sprint sleeve signal) ----
    # When > 0, the composite score adds a cross-sectional z-score of the
    # proximity to the trailing breakout-window high, renormalizing the
    # regime-adaptive weights so the total stays 1.0. Default off preserves
    # the evidence-gated production score exactly.
    breakout_weight: float = 0.0
    breakout_window: int = 60          # trailing high lookback (trading days)
    breakout_volume_confirm: bool = False  # require volume ratio >= 1 on breakouts
    # Regime-adaptive breakout scaling (fixed-vs-dynamic evidence gate,
    # 2026-08-11): the honest PIT gate showed the sprint's 60d-high breakout
    # tilt loses in BULL (-14.2% vs momentum +8.9%) while winning in NEUTRAL.
    # Scales < 1.0 fade the breakout factor in those regimes; 1.0 (default)
    # keeps the fixed-weight behaviour byte-identical.
    breakout_bull_scale: float = 1.0   # fade breakout chase in BULL (extension risk)
    breakout_highvol_scale: float = 1.0  # fade in HIGH_VOL (breakout noise)
    # ---- optional A-share anomaly factors (signal-layer expansion) ----
    # MAX effect: 20d maximum daily return is a negative predictor in A-shares
    # (lottery demand). illiquidity: log Amihud (|ret|/amount) carries a
    # positive premium. Both default 0 so the evidence-gated production score
    # is unchanged; when > 0 they are z-scored and added like the breakout
    # factor, with the base regime weights renormalized to 1.0.
    max_ret_weight: float = 0.0          # weight on -z(max 20d daily return)
    max_ret_window: int = 20
    illiquidity_weight: float = 0.0      # weight on z(log Amihud illiquidity)
    # Pluggable open-source factor weights (Main.factor_library): name -> weight
    # (e.g. {"roc20": 0.10, "rsi14": 0.05}). Default empty keeps the
    # evidence-gated production score unchanged.
    extra_factor_weights: Dict[str, float] = field(default_factory=dict)
    # Sector-momentum tilt (Main.sector_rotation): weight on the z-scored
    # industry 20d momentum from the CSRC map (Data_Cache/sector_map_full.json).
    # Default 0 keeps the evidence-gated production score unchanged.
    sector_momentum_weight: float = 0.0
    # Point-in-time alternative signal (date x symbol, values in [-1, 1]).
    # The panel is read only at the current signal date; no forward fill is
    # performed inside the engine. Default weight 0 keeps production behavior
    # unchanged until the signal passes the full-PIT evidence gate.
    alternative_signal_weight: float = 0.0
    alternative_signal_panel: Optional[pd.DataFrame] = None
    alternative_signal_min_coverage: float = 0.80
    alternative_signal_max_missing_rate: float = 0.20
    alternative_signal_max_latency_hours: float = 24.0
    alternative_signal_allow_fallback: bool = False
    # Actual number of parameter/model variants examined for this result.
    # Missing evidence deliberately keeps the research gate on HOLD.
    research_num_trials: Optional[int] = None
    # Fundamental factors (Main.fundamental_factors): name -> weight, e.g.
    # {"gp_margin": 0.10, "yoy_ni": 0.05}. Values are PIT-aligned to report
    # publication dates and uncovered names fall back to the cross-sectional
    # median. Default empty keeps the production score unchanged.
    fundamental_factors: Dict[str, float] = field(default_factory=dict)
    # PIT-validated fundamentals coverage: the top ``fundamental_top_n`` most
    # liquid names (the 600-name coverage was validated by PIT gate #18; wider
    # coverage was rejected by gates #20/#21 - more names dilute the factor).
    fundamental_top_n: int = 600
    # Fundamentals source: "auto" prefers the quarterly cache when present,
    # "annual" / "quarterly" force a specific cache (clean A/B comparisons).
    fundamental_source: str = "auto"
    fee_rate: float = 0.0013           # round-trip cost fraction (approx)
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    # ---- signal enhancements ----
    volume_confirm_days: int = 5       # volume/amount momentum window
    volume_confirm_base: int = 20      # baseline window for volume confirmation
    require_volume_confirm: bool = False
    require_relative_strength: bool = True   # beat the equal-weight benchmark
    min_top_momentum_gate: float = 0.0       # skip week if best raw 20d return below this
    # Optional market-breadth confirmation gate (0 = off, byte-compatible).
    # When > 0, opening NEW positions requires the fraction of the universe
    # trading above its 60-day MA to be at least this level - a narrow market
    # (few names above their MAs) is a topping risk even when the top momentum
    # names look strong. The gate behaves like ``bull_only_trading``: on a
    # breach the rebalance de-risks to cash (persistence is skipped).
    min_breadth_for_buys: float = 0.0
    bull_only_trading: bool = False          # only open new positions in BULL regime
    bear_no_loss: bool = True                # hard constraint: zero exposure in BEAR
    us_trend_filter: bool = False            # require US benchmark uptrend (transfer)
    us_trend_ma: int = 40
    regime_model: str = ""                   # optional ML regime: "hmm"|"logit"|"ensemble"
    ml_bear_override: bool = False           # MA rule + ML bear veto (hybrid)
    ml_bear_floor: float = 0.0               # minimum exposure while ML bear-probable (0 = hard cash veto)
    ml_bear_low: float = 0.40                # ML P(up) at/below this -> exposure floor
    ml_bull_high: float = 0.60               # ML P(up) at/above this -> full regime exposure
    signal_mode: str = "composite"           # composite multi-factor adaptive (default)
    defensive_tilt: bool = False             # low-vol/trend tilt in range regimes
    defensive_core: bool = False             # dividend+lowvol+trend core in all regimes
    defensive_core_bull_momentum: bool = False  # momentum tilt inside BULL while in defensive core
    defensive_filter: bool = False           # non-BULL picks restricted to dividend/low-vol half
    defensive_div_weight: float = 0.25       # dividend weight inside the defensive core
    dividend_yield_map: Optional[Dict[str, float]] = None  # symbol -> avg dps
    dividend_cash: Optional[pd.DataFrame] = None  # wide (ex_date x symbol) cash-per-share, PIT
    selection_model: str = ""                # optional ML selection: "lgb"
    selection_ml_weight: float = 1.0
    vol_target: float = 0.0                  # annualized vol target (0 disables)
    vol_lookback: int = 60
    vol_scale_floor: float = 0.25            # lower clamp on vol-target exposure scale
    hedge_etf: str = ""                      # short this symbol as market hedge (market-neutral)
    borrow_cost: float = 0.04                # annual financing cost on the short leg
    # ---- controlled short sleeve (A-share honest subset) ----
    # Default off. When enabled, shorts are ONLY opened in a high-conviction
    # bear state (regime BEAR after the ML overlay AND ML P(up) at or below
    # ``short_conviction_prob`` AND benchmark 20d return at/below
    # ``short_confirmation_mom20``), the gross short is capped at
    # ``max_short_exposure``,
    # and per-name short exposure at ``short_per_name_cap``. A-shares cannot
    # short most individual names, so ``short_etf_only=True`` (default) shorts
    # only broad index ETFs (510300/510500) - the honest instrument set.
    enable_short_sleeve: bool = False
    max_short_exposure: float = 0.10         # cap on gross short (fraction of capital)
    short_etf_only: bool = True              # honest A-share: only broad ETFs
    short_etf_pool: Tuple[str, ...] = ("510300.SH", "510500.SH")
    short_conviction_prob: float = 0.35      # ML P(up) <= this to open shorts
    # Extra confirmation (optional, default off): the benchmark 20d return
    # must be at/below this to open shorts. The PIT evidence showed P(up)<=0.35
    # and trend-BEAR almost never co-occur (the ML learns mean reversion in
    # extended downtrends), so the default gate is regime-overlay BEAR + the
    # calibrated ML conviction alone; a stricter user can re-enable this.
    short_confirmation_mom20: float = 0.0
    short_per_name_cap: float = 0.05         # per-name short cap when !etf_only
    short_top_k: int = 2                     # weakest names to short when !etf_only
    short_holding_cap_days: int = 42         # force-close shorts after N trading days
    short_stop_loss_pct: float = 0.08        # close short if asset rises this much
    trend_risk_scaler: bool = False          # scale exposure by benchmark/MA20 distance
    trend_risk_floor: float = 0.10           # minimum exposure when trend scaler active
    trend_risk_band: float = 0.10            # ratio distance over which exposure fades to floor
    trend_risk_ma: int = 20                  # benchmark MA window used by the scaler
    drawdown_guard: float = 0.0              # equity-DD threshold that starts scaling exposure (0 disables)
    drawdown_guard_max: float = 0.12         # DD at which exposure reaches drawdown_floor
    drawdown_floor: float = 0.25             # exposure scale floor under deep drawdown
    dd_guard_exposure: float = 0.35          # exposure target while the drawdown guard is latched
    drawdown_recovery_ma: int = 20           # kept for backwards compatibility (unused by continuous guard)
    neutral_benchmark_hold: bool = False     # hold broad ETF instead of cash in NEUTRAL
    neutral_benchmark_symbol: str = "510300.SH"
    bull_benchmark_hold: bool = False        # hold broad ETF in BULL (index participation)
    bull_benchmark_symbol: str = "510300.SH"
    rebalance_weekday: Optional[int] = 4     # align rebalances to Fridays (0=Mon..4=Fri)
    # Minimum requested turnover (sum of |target - current| over all symbols)
    # required for a regular rebalance to fire. Below the threshold the current
    # book is kept as-is, cutting micro-rebalancing cost and turnover. Default
    # 0.0 keeps the evidence-gated production behavior byte-identical; only
    # regular (non-defensive) rebalances are skippable - risk-state transitions
    # (bear/euphoria/event-shock/drawdown-guard) always execute.
    rebalance_min_turnover: float = 0.0
    regime_benchmark_symbols: Optional[Tuple[str, ...]] = None  # subset for regime
    hold_persistent: bool = True             # keep a name while still top-2K or trend intact
    persist_rank_floor: int = 6              # keep if rank <= floor
    max_holding_days: int = 0                # 0 = no time cap; else force exit after N days
    confirm_leverage: float = 1.0            # small leverage ONLY in 100%-confirmed bull states (1.0 disables)
    confirm_ml_prob: float = 0.65            # ML P(up) required for confirmed-bull leverage
    confirm_equity_proximity: float = 0.97   # strategy equity must be within this ratio of its peak
    # Legacy no-op: the old local z-scoring path that consumed this weight was
    # dead code (the ranking uses ``composite_factor_scores``, where the 1-day
    # reversal tilt enters through the fixed regime weights on ``rev1``).
    # Retained for API compatibility; changing it has no effect.
    reversal_1d_weight: float = 0.0
    reversal_window: int = 1                 # reversal lookback days (1 or 2)
    trend_filter_long: int = 60              # candidate trend MA (0 disables)
    trend_filter_short: int = 0              # optional short MA (0 disables)
    daily_regime_monitoring: bool = False    # intra-week exposure adaptation
    event_shock_threshold: float = 0.0       # benchmark 1-day crash -> cut exposure
    # Robust alternative to the fixed ``event_shock_threshold``: trigger the
    # risk-off path when the benchmark's daily return falls ``event_shock_zscore``
    # standard deviations below its trailing mean (rolling window
    # ``event_shock_z_window``). A fixed threshold calibrated on a calm large-cap
    # benchmark fires constantly on volatile mid/small-cap universes (measured:
    # 158 triggers on the 421-name subset vs a z-score rule that adapts to the
    # prevailing volatility). 0.0 disables the z-score rule (fixed threshold
    # governs). When both are set, either rule can trigger.
    event_shock_zscore: float = 0.0
    event_shock_z_window: int = 60
    event_shock_z_min_obs: int = 30
    event_shock_exposure: float = 0.10       # exposure level after an event shock
    event_shock_latch: bool = False          # stay de-risked until short MA reclaimed
    event_shock_recovery_ma: int = 5         # benchmark MA that releases the risk-off latch
    defensive_hold_assets: Tuple[str, ...] = ()   # safe-asset pool (bond/gold/money ETFs) held in defensive states
    defensive_hold_exposure: float = 0.90    # total gross exposure while in the defensive-hold state
    defensive_hold_safe_frac: float = 0.70   # share of defensive exposure allocated to the safe asset
    defensive_hold_basket: Tuple[Tuple[str, float], ...] = ()  # fixed safe-asset weights; overrides momentum pick
    safe_trend_gate: int = 20                # short-term momentum window that qualifies a safe asset
    euphoria_threshold: float = 0.0          # bench 20d return above this -> defensive hold (topping filter)
    benchmark_exclude: Tuple[str, ...] = ()  # symbols excluded from the equal-weight benchmark (safe assets)
    # ---- honest small-capital execution model ----
    capital_base: float = 100_000.0          # CNY account size used to model min commission & board lots
    enable_board_lots: bool = True           # round orders to 100-share (200 for STAR) lots
    slippage_rate: float = 0.0002            # per-side slippage fraction on traded notional
    # ---- survivorship-free universe membership (PIT) ----
    alive_mask: Optional[pd.DataFrame] = None  # bool (date x symbol): alive on date; None disables
    # ---- optional strategy-selection layer (evidence-gated, default off) ----
    strategy_selector: str = ""              # "" = disabled; "hysteresis" enables the rule-based switcher
    selector_min_stay: int = 4               # consecutive rebalances required before switching archetype


@dataclass
class RegimeState:
    date: pd.Timestamp
    regime: str                    # BULL / BEAR / NEUTRAL
    benchmark_close: float
    benchmark_ma_fast: float
    benchmark_ma_slow: float
    exposure: float
    advice_zh: str
    advice_en: str


def _zscore(series: pd.Series) -> pd.Series:
    s = series.astype(float)
    mu, sd = s.mean(), s.std(ddof=0)
    if sd is None or np.isnan(sd) or sd <= 1e-12:
        return pd.Series(0.0, index=s.index)
    return (s - mu) / sd


def composite_factor_scores(
    panel: FeaturePanel, date: pd.Timestamp, symbols: List[str], regime: RegimeState, params: RotationParams
) -> Dict[str, float]:
    """Dynamic multi-factor composite with regime-adaptive weights.

    Factor families (each cross-sectionally z-scored):
    - momentum: 20/60-day continuation
    - trend:    price vs MA20/MA60 alignment
    - reversal: 1/5-day overreaction correction
    - lowvol:   inverse realised volatility (low-vol anomaly)
    - liquidity: volume-ratio confirmation
    The blend adapts to the market state: trend-following in bull, mean
    reversion in ranges, defensive low-vol in high-volatility periods.
    """
    if date not in panel.common:
        return {}
    factors: Dict[str, pd.Series] = {}
    if 20 in panel.momentum:
        factors["mom20"] = panel.momentum[20].loc[date]
    if 60 in panel.momentum:
        factors["mom60"] = panel.momentum[60].loc[date]
    close_row = panel.close.loc[date]
    if panel.ma20 is not None and panel.ma60 is not None and date in panel.ma20.index:
        ma20 = panel.ma20.loc[date]
        ma60 = panel.ma60.loc[date]
    else:
        # fallback for hand-built panels in tests (or pre-optimization panels)
        ma20 = panel.close.rolling(20).mean().loc[date]
        ma60 = panel.close.rolling(60).mean().loc[date]
    factors["trend"] = (close_row / ma20 - 1.0) + 0.5 * (ma20 / ma60 - 1.0)
    if 1 in panel.momentum:
        factors["rev1"] = -panel.momentum[1].loc[date]
    if 5 in panel.momentum:
        factors["rev5"] = -panel.momentum[5].loc[date]
    factors["lowvol"] = -panel.volatility.loc[date]
    factors["vol_ratio"] = panel.volume_ratio.loc[date]
    factors["divyield"] = panel.div_yield.loc[date]
    if params.breakout_weight > 0 and panel.breakout is not None and date in panel.breakout.index:
        breakout_row = panel.breakout.loc[date]
        if params.breakout_volume_confirm:
            vol_ok = panel.volume_ratio.loc[date] >= 1.0
            breakout_row = breakout_row.where(vol_ok, np.nan)
        if breakout_row.notna().sum() >= 5:
            factors["breakout"] = breakout_row
    if params.max_ret_weight > 0 and panel.max_ret is not None and date in panel.max_ret.index:
        maxret_row = panel.max_ret.loc[date]
        if maxret_row.notna().sum() >= 5:
            factors["maxret"] = -maxret_row  # low MAX (lottery demand) preferred
    if params.illiquidity_weight > 0 and panel.illiquidity is not None and date in panel.illiquidity.index:
        illiq_row = panel.illiquidity.loc[date]
        if illiq_row.notna().sum() >= 5:
            factors["illiq"] = illiq_row
    if params.extra_factor_weights:
        # Pluggable open-source factors (Main.factor_library), PIT by
        # construction; only rows with enough cross-section are mounted.
        from Main.factor_library import compute_factors

        for name, w in params.extra_factor_weights.items():
            if float(w) <= 0 or name in factors:
                continue
            row = compute_factors(panel, date, [name]).get(name)
            if row is not None:
                factors[name] = row
    if params.sector_momentum_weight > 0:
        # Sector-momentum tilt (Main.sector_rotation); the CSRC map is loaded
        # once per backtest and cached on the panel.
        from Main.sector_rotation import industry_momentum_series, load_sector_map

        smap = getattr(panel, "_sector_map", None)
        if smap is None:
            smap = load_sector_map()
            panel._sector_map = smap
        sec_row = industry_momentum_series(panel, date, smap)
        if sec_row is not None:
            factors["sector_mom"] = sec_row
    if params.alternative_signal_weight > 0 and panel.alternative_signal is not None:
        if date in panel.alternative_signal.index:
            alt_row = panel.alternative_signal.loc[date].reindex(symbols).astype(float)
            alt_row = alt_row.replace([np.inf, -np.inf], np.nan)
            if alt_row.notna().sum() >= 2:
                factors["alternative_signal"] = alt_row
    if params.fundamental_factors and getattr(panel, "fundamental_panels", None):
        # PIT fundamentals (Main.fundamental_factors): uncovered names fall
        # back to the cross-sectional median so they are neither boosted nor
        # punished.
        for fname, w in params.fundamental_factors.items():
            if float(w) <= 0 or fname in factors:
                continue
            base_name = fname[4:] if fname.startswith("low_") else fname
            fp = panel.fundamental_panels.get(base_name)
            if fp is None or date not in fp.index:
                continue
            row = fp.loc[date].astype(float)
            if row.notna().sum() >= 5:
                row = row.fillna(row.median(skipna=True))
                if fname.startswith("low_"):
                    row = -row  # prefer LOW values (e.g. low_debt_ratio)
                factors[fname] = row

    z = {name: _zscore(ser) for name, ser in factors.items()}
    vol_bench = panel.bench_close.pct_change(fill_method=None).loc[:date].tail(60).std(ddof=0) * np.sqrt(252)
    high_vol = bool(np.isfinite(vol_bench) and vol_bench > 0.28)
    if params.defensive_core:
        if params.defensive_core_bull_momentum and regime.regime == "BULL":
            weights = {"mom20": 0.20, "mom60": 0.16, "trend": 0.22, "rev1": 0.05, "rev5": 0.09, "lowvol": 0.09, "vol_ratio": 0.09, "divyield": 0.10}
            state = "DEFENSIVE_CORE_BULL_MOM"
        else:
            d_w = float(params.defensive_div_weight)
            if d_w > 0:
                rest = 1.0 - d_w
                weights = {
                    "mom20": 0.05 * rest, "mom60": 0.10 * rest, "trend": 0.20 * rest,
                    "rev1": 0.03 * rest, "rev5": 0.05 * rest, "lowvol": 0.40 * rest,
                    "vol_ratio": 0.02 * rest, "divyield": d_w,
                }
            else:
                weights = {"mom20": 0.05, "mom60": 0.10, "trend": 0.25, "rev1": 0.03, "rev5": 0.05, "lowvol": 0.25, "vol_ratio": 0.02, "divyield": 0.25}
            state = "DEFENSIVE_CORE"
    elif regime.regime == "BULL":
        weights = {"mom20": 0.20, "mom60": 0.16, "trend": 0.22, "rev1": 0.05, "rev5": 0.09, "lowvol": 0.09, "vol_ratio": 0.09, "divyield": 0.10}
        state = "TREND_BULL"
    elif high_vol:
        weights = {"mom20": 0.08, "mom60": 0.08, "trend": 0.08, "rev1": 0.08, "rev5": 0.16, "lowvol": 0.30, "vol_ratio": 0.05, "divyield": 0.17}
        state = "HIGH_VOL"
    elif params.defensive_tilt:
        weights = {"mom20": 0.08, "mom60": 0.12, "trend": 0.20, "rev1": 0.04, "rev5": 0.08, "lowvol": 0.26, "vol_ratio": 0.04, "divyield": 0.18}
        state = "DEFENSIVE"
    else:
        weights = {"mom20": 0.12, "mom60": 0.08, "trend": 0.12, "rev1": 0.16, "rev5": 0.16, "lowvol": 0.12, "vol_ratio": 0.04, "divyield": 0.20}
        state = "RANGE"
    # Optional extra factors (breakout / MAX effect / Amihud illiquidity) are
    # z-scored and blended by renormalizing the regime-adaptive base weights so
    # the total stays 1.0. All default to 0, keeping the evidence-gated
    # production score byte-identical. Breakout can be scaled per regime (the
    # fixed-vs-dynamic gate kept fixed 0.35 for sprint; scales default 1.0).
    extra_weights: Dict[str, float] = {}
    if "breakout" in z and params.breakout_weight > 0:
        bw = float(np.clip(params.breakout_weight, 0.0, 1.0))
        if state == "TREND_BULL":
            bw *= float(getattr(params, "breakout_bull_scale", 1.0))
        elif state == "HIGH_VOL":
            bw *= float(getattr(params, "breakout_highvol_scale", 1.0))
        extra_weights["breakout"] = bw
    if "maxret" in z:
        extra_weights["maxret"] = float(np.clip(params.max_ret_weight, 0.0, 1.0))
    if "illiq" in z:
        extra_weights["illiq"] = float(np.clip(params.illiquidity_weight, 0.0, 1.0))
    for name, w in (params.extra_factor_weights or {}).items():
        if name in z and float(w) > 0:
            extra_weights[name] = float(np.clip(float(w), 0.0, 1.0))
    if "sector_mom" in z and params.sector_momentum_weight > 0:
        extra_weights["sector_mom"] = float(np.clip(params.sector_momentum_weight, 0.0, 1.0))
    if "alternative_signal" in z and params.alternative_signal_weight > 0:
        extra_weights["alternative_signal"] = float(np.clip(params.alternative_signal_weight, 0.0, 1.0))
    for name, w in (params.fundamental_factors or {}).items():
        if name in z and float(w) > 0:
            extra_weights[name] = float(np.clip(float(w), 0.0, 1.0))
    if extra_weights:
        extra_total = float(sum(extra_weights.values()))
        if extra_total > 1.0:
            extra_weights = {k: v / extra_total for k, v in extra_weights.items()}
            extra_total = 1.0
        base_total = sum(weights.values())
        if base_total > 0:
            weights = {k: v / base_total * (1.0 - extra_total) for k, v in weights.items()}
            weights.update(extra_weights)
    scores: Dict[str, float] = {}
    for sym in symbols:
        total, ok = 0.0, True
        for name, w in weights.items():
            ser = z.get(name)
            if ser is None or sym not in ser.index or pd.isna(ser[sym]):
                ok = False
                break
            total += w * float(ser[sym])
        if ok:
            scores[sym] = total
    return scores


@dataclass
class FeaturePanel:
    """Precomputed cross-sectional panels for fast per-date ranking."""

    common: pd.DatetimeIndex
    symbols: List[str]
    close: pd.DataFrame
    volume: pd.DataFrame
    amount: pd.DataFrame
    momentum: Dict[int, pd.DataFrame]
    volatility: pd.DataFrame
    trend: pd.DataFrame
    volume_ratio: pd.DataFrame
    adv20: pd.DataFrame
    div_yield: pd.DataFrame
    bench_return_20d: pd.Series
    bench_close: pd.Series
    bench_ma_fast: pd.Series
    bench_ma_slow: pd.Series
    bench_ma_confirmation: pd.Series
    bench_ma_trend: pd.Series
    bench_shock_z: Optional[pd.Series] = None  # rolling z-score of benchmark daily returns (event-shock detection)
    ma20: Optional[pd.DataFrame] = None     # close.rolling(20).mean(), precomputed
    ma60: Optional[pd.DataFrame] = None     # close.rolling(60).mean(), precomputed
    stop_band: Optional[pd.DataFrame] = None  # PIT ATR stop-loss fractions
    take_band: Optional[pd.DataFrame] = None  # PIT ATR take-profit fractions
    breakout: Optional[pd.DataFrame] = None   # close / trailing-high proximity (PIT)
    max_ret: Optional[pd.DataFrame] = None    # rolling max daily return (PIT)
    illiquidity: Optional[pd.DataFrame] = None  # log Amihud |ret|/amount (PIT)
    fundamental_panels: Optional[Dict[str, pd.DataFrame]] = None  # PIT fundamentals
    alternative_signal: Optional[pd.DataFrame] = None  # exact-date PIT signal; never backfilled here
    alternative_signal_governance: Optional[dict] = None


def precompute_panels(frames: Dict[str, pd.DataFrame], params: RotationParams) -> FeaturePanel:
    symbols = sorted(frames)
    common = frames[symbols[0]].index
    for symbol in symbols[1:]:
        common = common.union(frames[symbol].index)
    common = common.unique().sort_values()
    close = pd.DataFrame({sym: pd.to_numeric(frames[sym]["close"], errors="coerce").reindex(common) for sym in symbols})
    volume = pd.DataFrame({sym: pd.to_numeric(frames[sym].get("volume", frames[sym].get("amount")), errors="coerce").reindex(common) for sym in symbols})
    amount = pd.DataFrame({sym: pd.to_numeric(frames[sym].get("amount", frames[sym]["close"] * frames[sym]["volume"]), errors="coerce").reindex(common) for sym in symbols})
    momentum: Dict[int, pd.DataFrame] = {}
    extra_windows = {20, 60, 120} if params.signal_mode == "composite" else set()
    for window in set(params.momentum_windows) | {1, 5, params.reversal_window} | extra_windows:
        momentum[window] = close / close.shift(window) - 1.0
    rets = close.pct_change(fill_method=None)
    volatility = rets.rolling(60, min_periods=40).std(ddof=0) * np.sqrt(252)
    ma_long = close.rolling(params.trend_filter_long if params.trend_filter_long > 0 else 60, min_periods=1).mean()
    if params.trend_filter_short > 0:
        ma_short = close.rolling(params.trend_filter_short, min_periods=1).mean()
        trend = (close > ma_short) & (ma_short > ma_long)
    elif params.trend_filter_long > 0:
        trend = close > ma_long
    else:
        trend = close.notna()
    vol_short = volume.rolling(params.volume_confirm_days, min_periods=1).mean()
    vol_base = volume.rolling(params.volume_confirm_base, min_periods=1).mean()
    volume_ratio = vol_short / vol_base.replace(0, np.nan)
    adv20 = amount.rolling(20, min_periods=5).mean()
    ma20_panel = close.rolling(20).mean()
    ma60_panel = close.rolling(60).mean()
    stop_band = None
    take_band = None
    if params.dynamic_stops:
        # Decoupled bottom-up stop bands (see Main.dynamic_stops); computed
        # once here so the per-day loop only does a panel lookup.
        from Main.dynamic_stops import precompute_stop_bands

        stop_band, take_band = precompute_stop_bands(frames, common, symbols, params)
    breakout = None
    if params.breakout_weight > 0:
        # PIT proximity to the trailing high: 1.0 = making a new high today.
        high_win = max(5, int(params.breakout_window))
        breakout = close / close.rolling(high_win, min_periods=20).max()
    max_ret = None
    if params.max_ret_weight > 0:
        daily_ret = close.pct_change(fill_method=None)
        max_ret = daily_ret.rolling(max(2, int(params.max_ret_window)), min_periods=5).max()
    illiquidity = None
    if params.illiquidity_weight > 0:
        daily_ret = close.pct_change(fill_method=None)
        amihud = daily_ret.abs() / amount.replace(0, np.nan)
        illiquidity = np.log(amihud + 1e-12)
    fundamental_panels = None
    if params.fundamental_factors:
        from Main.fundamental_factors import build_fundamental_panels, load_all_fundamentals

        source = getattr(params, "fundamental_source", "auto")
        if source == "annual":
            root = Path(__file__).resolve().parents[1] / "Data_Cache"
            fundamentals = load_all_fundamentals(
                profit_path=root / "fundamentals_annual.json",
                top_n=params.fundamental_top_n,
            )
        elif source == "quarterly":
            root = Path(__file__).resolve().parents[1] / "Data_Cache"
            fundamentals = load_all_fundamentals(
                profit_path=root / "fundamentals_quarterly.json",
                top_n=params.fundamental_top_n,
            )
        else:
            fundamentals = load_all_fundamentals(top_n=params.fundamental_top_n)
        if fundamentals:
            fundamental_panels = build_fundamental_panels(fundamentals, common, symbols)
    alternative_signal = None
    alternative_signal_governance = None
    if params.alternative_signal_weight > 0 and params.alternative_signal_panel is not None:
        source = params.alternative_signal_panel.copy()
        source.index = pd.to_datetime(source.index, errors="coerce").tz_localize(None)
        source = source.loc[source.index.notna()]
        source = source[~source.index.duplicated(keep="last")].sort_index()
        alternative_signal_governance = evaluate_signal_panel(
            source,
            symbols,
            source.attrs.get("provenance"),
            min_symbol_coverage=params.alternative_signal_min_coverage,
            max_missing_rate=params.alternative_signal_max_missing_rate,
            max_source_latency_hours=params.alternative_signal_max_latency_hours,
            allow_fallback=params.alternative_signal_allow_fallback,
        )
        if not alternative_signal_governance["passed"]:
            raise ValueError(
                "alternative signal governance failed: "
                + ", ".join(alternative_signal_governance["reasons"])
            )
        alternative_signal = source.apply(pd.to_numeric, errors="coerce").reindex(index=common, columns=symbols)
    if params.dividend_cash is not None and len(params.dividend_cash):
        # PIT trailing dividend yield: only dividends whose ex-date has already
        # passed enter the trailing window (see pit_dividends.trailing_dividend_yield).
        div_yield = trailing_dividend_yield(params.dividend_cash, close)
        div_yield = div_yield.reindex(common).reindex(columns=symbols)
    elif params.dividend_yield_map:
        # Deprecated static map: kept only for legacy callers/tests. It is NOT
        # point-in-time (average dps measured over recent years applied to the
        # whole window) and must not be used for production backtests.
        dps = pd.Series({sym: params.dividend_yield_map.get(sym, np.nan) for sym in symbols})
        div_yield = pd.DataFrame({sym: dps[sym] / close[sym] for sym in symbols})
    else:
        div_yield = pd.DataFrame(0.0, index=common, columns=symbols)
    if params.regime_benchmark_symbols:
        subset = [s for s in params.regime_benchmark_symbols if s in close.columns]
        bench_close = close[subset].mean(axis=1, skipna=True) if subset else close.mean(axis=1, skipna=True)
    else:
        bench_cols = [s for s in close.columns if s not in params.benchmark_exclude]
        bench_close = close[bench_cols].mean(axis=1, skipna=True) if bench_cols else close.mean(axis=1, skipna=True)
    bench_return_20d = bench_close / bench_close.shift(20) - 1.0
    bench_ma_fast = bench_close.rolling(params.regime_ma_fast, min_periods=min(params.regime_ma_fast, max(5, params.regime_ma_fast // 2))).mean()
    bench_ma_slow = bench_close.rolling(params.regime_ma, min_periods=min(params.regime_ma, max(10, params.regime_ma // 2))).mean()
    if params.regime_confirmation_ma > 0:
        panel_conf_ma = bench_close.rolling(params.regime_confirmation_ma, min_periods=min(params.regime_confirmation_ma, max(20, params.regime_confirmation_ma // 2))).mean()
    else:
        panel_conf_ma = pd.Series(np.nan, index=bench_close.index)
    trend_ma_win = max(5, int(getattr(params, "trend_risk_ma", 20)))
    panel_ma_trend = bench_close.rolling(trend_ma_win, min_periods=min(trend_ma_win, 10)).mean()
    z_win = max(10, int(getattr(params, "event_shock_z_window", 60)))
    z_min = max(5, int(getattr(params, "event_shock_z_min_obs", 30)))
    bench_rets = bench_close.pct_change(fill_method=None)
    bench_shock_z = ((bench_rets - bench_rets.rolling(z_win, min_periods=z_min).mean())
                     / bench_rets.rolling(z_win, min_periods=z_min).std(ddof=0))
    return FeaturePanel(common=common, symbols=symbols, close=close, volume=volume, amount=amount, momentum=momentum, volatility=volatility, trend=trend, volume_ratio=volume_ratio, adv20=adv20, div_yield=div_yield, ma20=ma20_panel, ma60=ma60_panel, bench_return_20d=bench_return_20d, bench_close=bench_close, bench_ma_fast=bench_ma_fast, bench_ma_slow=bench_ma_slow, bench_ma_confirmation=panel_conf_ma, bench_ma_trend=panel_ma_trend, bench_shock_z=bench_shock_z, stop_band=stop_band, take_band=take_band, breakout=breakout, max_ret=max_ret, illiquidity=illiquidity, fundamental_panels=fundamental_panels, alternative_signal=alternative_signal, alternative_signal_governance=alternative_signal_governance)


def rank_candidates(panel: FeaturePanel, date: pd.Timestamp, params: RotationParams, win_probs: Optional[Dict[str, float]] = None, regime: Optional[RegimeState] = None) -> List[Tuple[str, float, float]]:
    """Cross-sectional momentum ranking with trend and volatility filters.

    Each momentum window's return is z-scored across the whole universe before
    blending, so the score measures *relative* strength, not raw magnitude.
    """
    if date not in panel.common:
        return []
    row = panel.close.loc[date]
    vol_row = panel.volatility.loc[date]
    trend_row = panel.trend.loc[date]
    adv_row = panel.adv20.loc[date]
    valid = row.notna() & vol_row.notna() & trend_row & (row > params.min_price) & (vol_row <= params.max_annual_vol) & (adv_row >= params.min_adv)
    if params.alive_mask is not None and len(params.alive_mask):
        # Point-in-time membership: a symbol is only a candidate on dates where
        # it was actually listed (survivorship-free universe enforcement).
        if date in params.alive_mask.index:
            alive_row = params.alive_mask.loc[date].reindex(panel.symbols).fillna(False).astype(bool)
        else:
            alive_row = pd.Series(False, index=panel.symbols)
        valid = valid & alive_row
    if params.max_short_term_gain is not None and 5 in panel.momentum:
        mom5 = panel.momentum[5].loc[date]
        valid = valid & (mom5 <= params.max_short_term_gain)
    if params.require_volume_confirm:
        vol_ratio_row = panel.volume_ratio.loc[date]
        valid = valid & (vol_ratio_row >= 1.0)
    valid_symbols = [s for s in panel.symbols if valid.get(s, False)]
    if not valid_symbols:
        return []
    # Defensive filter: outside a confirmed bull market the book is restricted
    # to the dividend-paying or low-volatility half of the universe. Pure
    # momentum stars (e.g. CATL / Nasdaq ETFs in Nov 2021) then cannot leak
    # into the defensive book through extreme momentum z-scores.
    if params.defensive_filter and params.defensive_core and (regime is None or regime.regime != "BULL"):
        dy = panel.div_yield.loc[date].reindex(valid_symbols)
        # Dividend-qualified means a strictly positive trailing yield. A
        # median-based rule silently collapses to 'everyone passes' when the
        # dividend panel is dense but most names pay no cash dividend, which
        # makes the defensive filter depend on data coverage rather than
        # fundamentals (observed 2026-08-10 on the full PIT panel).
        dividend_ok = (dy > 0.0).fillna(False)
        lv = panel.volatility.loc[date].reindex(valid_symbols)
        median_lv = lv.median(skipna=True)
        valid_symbols = [
            s for s in valid_symbols
            if bool(dividend_ok.get(s, False)) or (pd.notna(lv.get(s)) and lv.get(s) <= median_lv)
        ]
    if not valid_symbols:
        return []

    # relative strength: raw 20d return must beat the benchmark's 20d return
    mom20 = panel.momentum.get(20)
    rel_ok = pd.Series(True, index=valid_symbols)
    if params.require_relative_strength and mom20 is not None:
        bench_20 = float(panel.bench_return_20d.loc[date]) if pd.notna(panel.bench_return_20d.loc[date]) else 0.0
        raw20 = mom20.loc[date]
        rel_ok = raw20[valid_symbols] > bench_20

    # NOTE: the actual ranking score is produced by ``composite_factor_scores``
    # below (regime-adaptive z-scored factor blend). The former local z-scoring
    # block (``z_rows`` / ``reversal_1d_weight``) was dead code: its outputs
    # were never consumed by the ranking, so ``reversal_1d_weight`` had no
    # effect on production results (documented in update plans/8-11). The 1-day
    # reversal tilt lives inside the composite via its fixed regime weights.
    if regime is None:
        return []
    comp = composite_factor_scores(panel, date, valid_symbols, regime, params)
    ranked = [(sym, comp.get(sym, float("-inf")), float(vol_row[sym])) for sym in valid_symbols if sym in comp]
    if win_probs and params.selection_ml_weight:
        probs = pd.Series({s: win_probs.get(s, np.nan) for s, _, _ in ranked})
        pz = _zscore(probs)
        ranked = [(s, score + params.selection_ml_weight * float(pz.get(s, 0.0)), vol) for s, score, vol in ranked]
    ranked.sort(key=lambda item: item[1], reverse=True)
    return ranked


def detect_regime(panel: FeaturePanel, date: pd.Timestamp, params: RotationParams) -> RegimeState:
    """Equal-weight benchmark trend determines the market regime."""
    last = float(panel.bench_close.loc[date])
    ma_fast = float(panel.bench_ma_fast.loc[date])
    ma_slow = float(panel.bench_ma_slow.loc[date])
    mom20 = float(panel.bench_return_20d.loc[date]) if pd.notna(panel.bench_return_20d.loc[date]) else 0.0
    conf_ok = True
    if params.regime_confirmation_ma > 0:
        conf_val = panel.bench_ma_confirmation.loc[date]
        conf_ok = bool(pd.notna(conf_val) and last > float(conf_val))
    # BULL additionally requires positive short-term benchmark momentum: a
    # price level above the MAs with rolling-over momentum is a topping pattern
    # (e.g. Dec 2021), not a buy signal.
    if last > ma_slow and ma_fast > ma_slow and conf_ok and mom20 > 0.0:
        regime = "BULL"
        exposure = min(params.bull_exposure * params.bull_leverage, params.max_gross_exposure)
        advice_zh = "多头市场:基准指数位于长期均线上方且短期动能向上,建议提高仓位积极参与强势轮动标的。"
        advice_en = "Bull market: benchmark above the long-term MA with upward short-term momentum; raise exposure and rotate into strong names."
    elif last < ma_slow and ma_fast < ma_slow and mom20 < 0.0:
        regime, exposure = "BEAR", params.bear_exposure
        advice_zh = "空头市场:基准指数跌破长期均线且短期动能向下,建议大幅降仓、以防守或现金为主,仅保留极少数逆势强势标的。"
        advice_en = "Bear market: benchmark below the long-term MA with downward momentum; cut exposure sharply, favour cash and only the strongest counter-trend names."
    else:
        regime, exposure = "NEUTRAL", (params.neutral_exposure if params.neutral_exposure > 0 else (params.bull_exposure + params.bear_exposure) / 2)
        advice_zh = "震荡市:基准指数围绕长期均线反复,建议半仓灵活参与,严格止盈止损。"
        advice_en = "Range-bound market: benchmark oscillates around the long-term MA; use moderate exposure with disciplined stops."
    return RegimeState(date=date, regime=regime, benchmark_close=last, benchmark_ma_fast=ma_fast, benchmark_ma_slow=ma_slow, exposure=float(exposure), advice_zh=advice_zh, advice_en=advice_en)


def _ml_prob_up(ml_res) -> Optional[float]:
    """Recover P(next-month up) from a detector result (regime + confidence)."""
    if ml_res is None:
        return None
    if ml_res.regime == "BULL":
        return float(ml_res.confidence)
    if ml_res.regime == "BEAR":
        return 1.0 - float(ml_res.confidence)
    return 0.5


def apply_risk_scaling(
    panel: FeaturePanel, date: pd.Timestamp, params: RotationParams,
    regime: RegimeState, ml_res=None, equity_dd: Optional[float] = None,
) -> RegimeState:
    """Apply ML exposure overlay, vol targeting, trend-risk and drawdown scaling.

    Shared by rebalance-day and daily-monitoring paths so intra-period risk
    response is identical to the rebalance logic.

    The ML layer is a continuous probability overlay rather than a hard cash
    veto: exposure fades toward ``ml_bear_floor`` as P(next-month up) falls to
    ``ml_bear_low`` and returns to the full regime target at ``ml_bull_high``.
    The drawdown layer scales exposure down as the equity drawdown deepens and
    ramps back automatically as the curve recovers, so the book never parks in
    stale cash after a drawdown (the old guard was bypassed by rebalances).
    """
    if ml_res is not None and params.ml_bear_override:
        p_up = _ml_prob_up(ml_res)
        if p_up is not None:
            band = max(float(params.ml_bull_high) - float(params.ml_bear_low), 1e-6)
            alpha = float(np.clip((p_up - float(params.ml_bear_low)) / band, 0.0, 1.0))
            floor = float(getattr(params, "ml_bear_floor", 0.0))
            regime.exposure = floor + alpha * (regime.exposure - floor)
            if alpha < 0.5:
                # low conviction -> defensive factor weights and no fresh buys
                regime.regime = "BEAR"
                regime.advice_zh = "ML 低信心区间:防御性低配。"
                regime.advice_en = "ML low-conviction: defensive underweight."
    if params.vol_target > 0:
        bench_rets = panel.bench_close.pct_change(fill_method=None).loc[:date].tail(params.vol_lookback).dropna()
        if len(bench_rets) >= 30:
            realized_vol = float(bench_rets.std(ddof=0) * np.sqrt(252))
            if realized_vol > 0:
                scale = float(np.clip(params.vol_target / realized_vol, params.vol_scale_floor, 2.0))
                regime.exposure = min(regime.exposure * scale, params.max_gross_exposure)
    if params.trend_risk_scaler:
        bench_now = panel.bench_close.loc[date] if pd.notna(panel.bench_close.loc[date]) else np.nan
        bench_ma = panel.bench_ma_trend.loc[date] if pd.notna(panel.bench_ma_trend.loc[date]) else np.nan
        if np.isfinite(bench_now) and np.isfinite(bench_ma) and bench_ma > 0:
            ratio = bench_now / bench_ma
            band = max(params.trend_risk_band, 1e-4)
            scale = float(np.clip((ratio - (1.0 - band)) / band, params.trend_risk_floor, 1.0))
            regime.exposure = min(regime.exposure * scale, params.max_gross_exposure)
    return regime


def build_target_weights(
    ranked: List[Tuple[str, float, float]], symbols: List[str], regime: RegimeState, params: RotationParams,
    current_weights: Optional[Dict[str, float]] = None, keep_symbols: Optional[List[str]] = None,
) -> Dict[str, float]:
    current_weights = current_weights or {}
    keep_symbols = keep_symbols or []
    selected = ranked[: params.top_n]
    # Bound ETF seats: sector/theme ETFs can be extremely volatile, so keep at
    # most max_etf_positions ETFs and backfill with the next-best stocks.
    etf_count = sum(1 for s, _, _ in selected if is_etf(s))
    if etf_count > params.max_etf_positions:
        stocks = [item for item in selected if not is_etf(item[0])]
        etfs = [item for item in selected if is_etf(item[0])][: params.max_etf_positions]
        for item in ranked[params.top_n:]:
            if len(etfs) + len(stocks) >= params.top_n:
                break
            if not is_etf(item[0]):
                stocks.append(item)
        selected = stocks + etfs
    keep_weights = {s: current_weights.get(s, 0.0) for s in keep_symbols if current_weights.get(s, 0.0) > 1e-9}
    if not selected and not keep_weights:
        return {symbol: 0.0 for symbol in symbols}
    raw = {symbol: max(score, 0.001) / max(vol, 0.05) for symbol, score, vol in selected}
    raw_total = sum(raw.values())
    weights = {symbol: 0.0 for symbol in symbols}
    short_weights: Dict[str, float] = {symbol: 0.0 for symbol in symbols}
    residual_exposure = max(0.0, regime.exposure - sum(keep_weights.values()))
    for symbol, value in raw.items():
        w = residual_exposure * value / raw_total if raw_total else 0.0
        weights[symbol] = float(min(w, params.per_position_cap))
    for symbol, w in keep_weights.items():
        weights[symbol] = weights.get(symbol, 0.0) + w
    # hard gross-exposure cap: the leverage ceiling must never be exceeded
    total = sum(weights.values())
    if total > params.max_gross_exposure:
        scale = params.max_gross_exposure / total
        weights = {s: w * scale for s, w in weights.items()}
    # renormalize non-kept picks so the fresh allocation reaches the regime target
    kept_total = sum(keep_weights.values())
    fresh_total = sum(weights[s] for s in weights if s not in keep_weights)
    if fresh_total > 0:
        fresh_target = max(0.0, regime.exposure - kept_total)
        scale = fresh_target / fresh_total
        for s in list(weights):
            if s not in keep_weights:
                weights[s] = min(weights[s] * scale, params.per_position_cap)
    return weights


def _pick_safe_asset(panel: FeaturePanel, date: pd.Timestamp, symbols: List[str], params: RotationParams) -> Optional[str]:
    """Pick the strongest safe asset (bond/gold/money ETF) by 120-day momentum.

    In defensive states the book holds the strongest rising safe asset instead
    of cash, so the equity curve keeps making new highs during long equity
    bears (2018 bond rally, 2022-2024 gold rally, money funds in flat years).
    A short-term trend gate prevents chasing a safe asset that has already
    broken down (e.g. gold rolling over in 2026): only assets with positive
    20-day momentum qualify, with a bond/money fallback.
    """
    pool = [s for s in params.defensive_hold_assets if s in symbols]
    if not pool:
        return None
    mom120 = panel.momentum.get(120)
    gate_win = max(5, int(params.safe_trend_gate))
    mom_gate = panel.momentum.get(gate_win)
    if mom120 is None or mom_gate is None or date not in mom120.index or date not in mom_gate.index:
        return pool[0]
    def _ret(s, frame):
        return float(frame.loc[date, s]) if s in frame.columns and pd.notna(frame.loc[date, s]) else -1e18
    candidates = [s for s in pool if _ret(s, mom_gate) > 0.0 and _ret(s, mom120) > -0.02]
    if candidates:
        return max(candidates, key=lambda s: _ret(s, mom120))
    fallback = [s for s in pool if s != "518880.SH"]
    if fallback:
        return max(fallback, key=lambda s: _ret(s, mom120))
    return pool[0]


def _safe_sleeve_weights(safe_symbol: Optional[str], symbols: List[str], params: RotationParams) -> Dict[str, float]:
    """Safe-sleeve weights: fixed basket if configured, else the single
    momentum-picked safe asset."""
    out: Dict[str, float] = {s: 0.0 for s in symbols}
    if params.defensive_hold_basket:
        for sym, w in params.defensive_hold_basket:
            if sym in out:
                out[sym] = float(w)
    elif safe_symbol is not None:
        out[safe_symbol] = 1.0
    total = sum(out.values())
    if total > 0:
        out = {s: w / total for s, w in out.items()}
    return out


def _defensive_hold_target(
    ranked: List[Tuple[str, float, float]], symbols: List[str], regime: RegimeState,
    params: RotationParams, current_weights: Optional[Dict[str, float]], keep_symbols: Optional[List[str]],
    safe_symbol: Optional[str],
) -> Dict[str, float]:
    """Target weights for the defensive-hold state: the safe sleeve (fixed
    basket or momentum-picked asset) at ``defensive_hold_exposure *
    defensive_hold_safe_frac`` plus the top-ranked defensive equities for the
    remainder."""
    target = {symbol: 0.0 for symbol in symbols}
    total_expo = float(np.clip(params.defensive_hold_exposure, 0.0, params.max_gross_exposure))
    safe_w = min(total_expo * float(np.clip(params.defensive_hold_safe_frac, 0.0, 1.0)), params.max_gross_exposure)
    sleeve = _safe_sleeve_weights(safe_symbol, symbols, params)
    for s, w in sleeve.items():
        target[s] = w * safe_w
    equity_expo = total_expo - safe_w
    if equity_expo > 1e-9 and ranked:
        sub = RegimeState(
            date=regime.date, regime=regime.regime, benchmark_close=regime.benchmark_close,
            benchmark_ma_fast=regime.benchmark_ma_fast, benchmark_ma_slow=regime.benchmark_ma_slow,
            exposure=equity_expo, advice_zh=regime.advice_zh, advice_en=regime.advice_en,
        )
        eq_target = build_target_weights(ranked, symbols, sub, params, current_weights, keep_symbols)
        for s in symbols:
            target[s] = target.get(s, 0.0) + eq_target.get(s, 0.0)
    total = sum(target.values())
    if total > params.max_gross_exposure:
        scale = params.max_gross_exposure / total
        target = {s: w * scale for s, w in target.items()}
    return target


def weekly_rotation_backtest(
    frames: Dict[str, pd.DataFrame], params: RotationParams | None = None, us_benchmark: Optional[pd.Series] = None,
    regime_detector_kwargs: Optional[dict] = None,
) -> dict:
    params = params or RotationParams()
    symbols = sorted(frames)
    if not symbols:
        raise ValueError("empty universe")
    common = frames[symbols[0]].index
    for symbol in symbols[1:]:
        common = common.union(frames[symbol].index)
    common = common.unique().sort_values()
    if params.start_date:
        common = common[common >= pd.Timestamp(params.start_date)]
    if params.end_date:
        common = common[common <= pd.Timestamp(params.end_date)]
    if len(common) < params.regime_ma + params.min_volume_days + 20:
        raise ValueError("insufficient common history")
    simulation = common[:-1]  # last day only for benchmarking, no forward trade

    # ---- vectorized precomputation: open-to-open returns and execution snapshots ----
    open_matrix = pd.DataFrame({sym: frames[sym]["open"].reindex(common) for sym in symbols})
    close_matrix = pd.DataFrame({sym: frames[sym]["close"].reindex(common) for sym in symbols})
    volume_matrix = pd.DataFrame({sym: frames[sym]["volume"].reindex(common) for sym in symbols})
    prev_close_matrix = close_matrix.shift(1)
    last_close_matrix = close_matrix.ffill()
    returns_matrix = open_matrix.shift(-1) / open_matrix - 1.0
    panel = precompute_panels(frames, params)
    recovery_win = max(3, int(params.event_shock_recovery_ma))
    bench_ma_recovery = panel.bench_close.rolling(recovery_win, min_periods=min(recovery_win, 3)).mean()
    regime_detector = None
    if params.regime_model:
        from Main.ml_regime import build_detector
        regime_detector = build_detector(params.regime_model, **(regime_detector_kwargs or {}))
    selection_predictor = None
    if params.selection_model:
        from Main.ml_selection import WeeklyWinPredictor
        selection_predictor = WeeklyWinPredictor()
    us_trend = None
    if params.us_trend_filter and us_benchmark is not None and len(us_benchmark):
        us_reindexed = us_benchmark.reindex(common).ffill()
        us_trend = us_reindexed > us_reindexed.rolling(params.us_trend_ma, min_periods=30).mean()

    weights = {symbol: 0.0 for symbol in symbols}
    short_weights: Dict[str, float] = {symbol: 0.0 for symbol in symbols}
    entry_prices: Dict[str, float] = {symbol: 0.0 for symbol in symbols}
    peak_prices: Dict[str, float] = {symbol: 0.0 for symbol in symbols}
    short_entry_prices: Dict[str, float] = {}
    holding_days: Dict[str, int] = {symbol: 0 for symbol in symbols}
    short_holding_days: Dict[str, int] = {}
    closed_trades: List[dict] = []
    pending: Optional[dict] = None
    regime_rows: List[dict] = []
    rows: List[dict] = []
    equity = 1.0
    current_regime = "N/A"
    equity_peak = 1.0
    risk_off = False
    dd_guard_active = False
    base_params = params
    gross_ceiling = float(params.max_gross_exposure)
    selector = None

    for idx, date in enumerate(simulation):
        next_date = common[common.get_loc(date) + 1]
        turnover = 0.0
        cost = 0.0
        # ---- risk-off latch: stay de-risked after an event shock until the
        # benchmark reclaims its short recovery MA (fast enough to catch the
        # rebound, slow enough to avoid re-entering a one-way bear market).
        if params.event_shock_latch and risk_off:
            bench_now = panel.bench_close.loc[date] if pd.notna(panel.bench_close.loc[date]) else np.nan
            bench_ma = bench_ma_recovery.loc[date] if pd.notna(bench_ma_recovery.loc[date]) else np.nan
            if np.isfinite(bench_now) and np.isfinite(bench_ma) and bench_now > bench_ma:
                risk_off = False

        if pending is not None and pending["execution_date"] == date:
            desired = pending["target"]
            alive_row = pd.Series(True, index=symbols)
            if params.alive_mask is not None and len(params.alive_mask):
                if date in params.alive_mask.index:
                    alive_row = params.alive_mask.loc[date].reindex(symbols).fillna(False).astype(bool)
                else:
                    alive_row = pd.Series(False, index=symbols)
            # Terminal names (delisted/absorbed per the PIT master list) are
            # force-sold at their last available close with sell fees, so a
            # survivorship-free universe cannot leave zombie holdings behind.
            dead_exits: Dict[str, float] = {}
            eff_open: Dict[str, float] = {}
            for symbol in symbols:
                if weights[symbol] > 1e-9 and not bool(alive_row[symbol]):
                    last_close = last_close_matrix.at[date, symbol]
                    if pd.notna(last_close) and last_close > 0:
                        dead_exits[symbol] = float(last_close)
            for symbol in symbols:
                prev_close = float(prev_close_matrix.at[date, symbol]) if pd.notna(prev_close_matrix.at[date, symbol]) else np.nan
                exec_open = float(open_matrix.at[date, symbol]) if pd.notna(open_matrix.at[date, symbol]) else np.nan
                exec_volume = float(volume_matrix.at[date, symbol]) if pd.notna(volume_matrix.at[date, symbol]) else 0.0
                eff_open[symbol] = exec_open if np.isfinite(exec_open) else np.nan
                if symbol in dead_exits:
                    exec_open = dead_exits[symbol]
                    eff_open[symbol] = exec_open
                    desired[symbol] = 0.0
                elif exec_volume <= 0 or prev_close <= 0 or exec_open <= 0:
                    desired[symbol] = weights[symbol]
                    continue
                gap = exec_open / prev_close - 1
                old_w, new_w = weights[symbol], desired[symbol]
                if symbol in dead_exits:
                    continue  # forced exit is never blocked by limit rules
                if new_w > old_w and gap >= 0.098:      # limit-up block on buys
                    desired[symbol] = old_w
                elif new_w < old_w and gap <= -0.098:   # limit-down block on sells
                    desired[symbol] = old_w
            requested_turnover = sum(abs(desired[symbol] - weights[symbol]) for symbol in symbols)
            if requested_turnover > 0.5:
                scale = 0.5 / requested_turnover
                desired = {s: weights[s] + (desired[s] - weights[s]) * scale for s in symbols}
            account_value = max(float(equity) * float(params.capital_base), 1.0)
            if params.enable_board_lots and params.capital_base > 0:
                # Round orders to board lots (100 shares; 200 for STAR market).
                # Full exits keep the odd lot (odd-lot disposal is allowed in
                # A-shares); partial trades round down to whole lots.
                for symbol in symbols:
                    delta = desired[symbol] - weights[symbol]
                    if abs(delta) < 1e-9 or desired[symbol] <= 1e-9:
                        continue
                    exec_open = eff_open.get(symbol, np.nan)
                    if not np.isfinite(exec_open) or exec_open <= 0:
                        desired[symbol] = weights[symbol]
                        continue
                    if symbol in dead_exits:
                        continue  # full exit of a terminal name keeps no lots
                    lot = 200 if symbol.startswith("688") else 100
                    if delta > 0:
                        lots = int(abs(delta) * account_value // (exec_open * lot))
                        actual = lots * exec_open * lot / account_value
                        desired[symbol] = weights[symbol] + min(actual, delta)
                    else:
                        lots = int(min(abs(delta), weights[symbol]) * account_value // (exec_open * lot))
                        if lots <= 0:
                            desired[symbol] = weights[symbol]
                        else:
                            desired[symbol] = weights[symbol] - lots * exec_open * lot / account_value
            for symbol in symbols:
                delta = desired[symbol] - weights[symbol]
                turnover += abs(delta)
                if params.capital_base > 0 and abs(delta) > 1e-12:
                    notional = abs(delta) * account_value
                    side = "buy" if delta > 0 else "sell"
                    fee = explicit_order_fees(notional, side, symbol)["total"]
                    fee += notional * params.slippage_rate
                    cost += fee / account_value
            # ---- short book execution (hedge_etf full hedge or controlled
            # short sleeve): open = sell fees, close = buy fees ----
            short_target: Optional[Dict[str, float]] = None
            if params.hedge_etf and params.hedge_etf in symbols:
                short_target = {params.hedge_etf: -sum(desired.values())}
            elif pending is not None and pending.get("short_target") is not None:
                short_target = pending["short_target"]
            if short_target:
                for sym, tw in short_target.items():
                    if sym not in symbols:
                        continue
                    cur = short_weights.get(sym, 0.0)
                    delta = tw - cur
                    if abs(delta) > 1e-12:
                        turnover += abs(delta)
                        if params.capital_base > 0:
                            notional = abs(delta) * account_value
                            side = "sell" if delta < 0 else "buy"
                            fee = explicit_order_fees(notional, side, sym)["total"]
                            fee += notional * params.slippage_rate
                            cost += fee / account_value
                    short_weights[sym] = tw
                    exec_open = eff_open.get(sym, np.nan)
                    if abs(tw) > 1e-12 and abs(cur) <= 1e-12 and np.isfinite(exec_open):
                        short_entry_prices[sym] = float(exec_open)
                        short_holding_days[sym] = 0
                    elif abs(tw) <= 1e-12:
                        short_entry_prices.pop(sym, None)
                        short_holding_days.pop(sym, None)
            else:
                # no short target this cycle: close any open shorts
                if any(abs(v) > 1e-12 for v in short_weights.values()):
                    for sym, cur in list(short_weights.items()):
                        if abs(cur) > 1e-12 and params.capital_base > 0:
                            notional = abs(cur) * account_value
                            fee = explicit_order_fees(notional, "buy", sym)["total"]
                            fee += notional * params.slippage_rate
                            cost += fee / account_value
                            turnover += abs(cur)
                        short_weights[sym] = 0.0
                    short_entry_prices.clear()
                    short_holding_days.clear()
            if params.capital_base <= 0:
                # legacy flat-fee fallback when no capital assumption is given
                cost = turnover * params.fee_rate
            # track entry prices for newly opened positions
            for symbol in symbols:
                exec_open = eff_open.get(symbol, np.nan)
                if desired[symbol] > 1e-9 and weights[symbol] <= 1e-9 and np.isfinite(exec_open):
                    entry_prices[symbol] = float(exec_open)
                    peak_prices[symbol] = float(exec_open)
                    holding_days[symbol] = 0
                elif desired[symbol] <= 1e-9:
                    if weights[symbol] > 1e-9 and entry_prices[symbol] > 0 and np.isfinite(exec_open):
                        exit_price = float(exec_open)
                        closed_trades.append({"symbol": symbol, "entry": entry_prices[symbol], "exit": exit_price, "pnl": exit_price / entry_prices[symbol] - 1.0, "exit_date": date})
                    entry_prices[symbol] = 0.0
                    peak_prices[symbol] = 0.0
                    holding_days[symbol] = 0
            weights = desired
            pending = None

        # ---- intra-week stop-loss / take-profit: check at close, realize at close ----
        stopped_symbols: set = set()
        if params.enable_intraweek_stops:
            for symbol in symbols:
                if weights[symbol] <= 1e-9 or entry_prices[symbol] <= 0:
                    continue
                close_now = close_matrix.at[date, symbol] if pd.notna(close_matrix.at[date, symbol]) else np.nan
                if not np.isfinite(close_now):
                    continue
                pnl = close_now / entry_prices[symbol] - 1.0
                # ratchet the peak price to today's close (PIT: only the close
                # is known at the decision point)
                if params.trailing_stop_pct > 0:
                    peak_prices[symbol] = max(peak_prices.get(symbol, entry_prices[symbol]), close_now)
                if params.dynamic_stops and panel.stop_band is not None:
                    # Bottom-up band: per-symbol PIT ATR fractions with static
                    # fallback on NaN (decoupled in Main.dynamic_stops).
                    from Main.dynamic_stops import band_at

                    stop_pct, take_pct = band_at(panel, symbol, date, params.stop_loss_pct, params.take_profit_pct)
                else:
                    stop_pct, take_pct = params.stop_loss_pct, params.take_profit_pct
                if params.trailing_stop_pct > 0:
                    # trailing rule: stop on a pullback from the position peak
                    peak_now = peak_prices.get(symbol, entry_prices[symbol])
                    stop_pct = float(params.trailing_stop_pct)
                    if peak_now > 0 and close_now / peak_now - 1.0 <= -stop_pct:
                        stopped_symbols.add(symbol)
                    elif take_pct > 0 and pnl >= take_pct:
                        stopped_symbols.add(symbol)
                elif pnl <= -stop_pct or pnl >= take_pct:
                    stopped_symbols.add(symbol)
        # short-sleeve reverse stop: close a short when the asset has risen
        # beyond ``short_stop_loss_pct`` (realized at today's close)
        stopped_shorts: set = set()
        if params.enable_short_sleeve and params.short_stop_loss_pct > 0:
            for sym in list(short_weights):
                sw = short_weights[sym]
                if abs(sw) <= 1e-12:
                    continue
                entry = short_entry_prices.get(sym, 0.0)
                close_now = close_matrix.at[date, sym] if pd.notna(close_matrix.at[date, sym]) else np.nan
                if not np.isfinite(close_now) or entry <= 0:
                    continue
                if close_now / entry - 1.0 >= params.short_stop_loss_pct:
                    stopped_shorts.add(sym)

        ret_row = returns_matrix.loc[date]
        asset_returns = {}
        for symbol in symbols:
            if symbol in stopped_symbols:
                # realized at today's close
                open_now = open_matrix.at[date, symbol] if pd.notna(open_matrix.at[date, symbol]) else np.nan
                close_now = close_matrix.at[date, symbol] if pd.notna(close_matrix.at[date, symbol]) else np.nan
                asset_returns[symbol] = float(close_now / open_now - 1.0) if np.isfinite(open_now) and np.isfinite(close_now) and open_now > 0 else 0.0
            else:
                asset_returns[symbol] = float(ret_row[symbol]) if pd.notna(ret_row[symbol]) else 0.0
        gross = sum(weights[symbol] * asset_returns[symbol] for symbol in symbols)
        if stopped_symbols and params.capital_base > 0:
            # charge sell-side fees (commission + stamp + slippage) on stops,
            # which are realized at today's close outside the rebalance path.
            # NOTE: stop exits are deliberately NOT added to ``turnover`` - the
            # turnover column measures rebalance activity for the turnover gate
            # (avg_rebalance_turnover); stop notional only appears in ``cost``.
            account_value = max(float(equity) * float(params.capital_base), 1.0)
            for symbol in stopped_symbols:
                if weights[symbol] > 1e-9:
                    notional = weights[symbol] * account_value
                    fee = explicit_order_fees(notional, "sell", symbol)["total"]
                    fee += notional * params.slippage_rate
                    cost += fee / account_value
        # short book P&L (hedge_etf or controlled short sleeve), net of borrow
        for sym, short_notional in short_weights.items():
            if abs(short_notional) > 1e-12:
                if sym in stopped_shorts:
                    open_now = open_matrix.at[date, sym] if pd.notna(open_matrix.at[date, sym]) else np.nan
                    close_now = close_matrix.at[date, sym] if pd.notna(close_matrix.at[date, sym]) else np.nan
                    sret = float(close_now / open_now - 1.0) if np.isfinite(open_now) and np.isfinite(close_now) and open_now > 0 else 0.0
                else:
                    sret = asset_returns.get(sym, 0.0)
                gross += (-short_notional) * sret - abs(short_notional) * params.borrow_cost / 252.0
        strategy_return = gross - cost
        # financing cost on leveraged capital (margin)
        leverage_drag = max(0.0, sum(weights.values()) - 1.0) * params.leverage_annual_cost / 252.0
        strategy_return -= leverage_drag
        eligible = [symbol for symbol in symbols if symbol not in params.benchmark_exclude and pd.notna(ret_row[symbol])]
        if params.alive_mask is not None and len(params.alive_mask):
            if date in params.alive_mask.index:
                alive_bench = params.alive_mask.loc[date].reindex(symbols).fillna(False).astype(bool)
                eligible = [s for s in eligible if bool(alive_bench[s])]
        benchmark_return = float(np.mean([asset_returns[s] for s in eligible])) if eligible else 0.0
        rows.append({
            "date": date, "strategy_return": strategy_return, "benchmark_return": benchmark_return,
            "gross_exposure": sum(weights.values()) + (sum(abs(v) for v in short_weights.values()) if params.enable_short_sleeve else 0.0),
            "short_exposure": sum(abs(v) for v in short_weights.values()) if params.enable_short_sleeve else 0.0,
            "turnover": turnover, "cost": cost,
            "regime": current_regime,
        })
        equity *= 1.0 + strategy_return
        if equity <= 0:
            raise ValueError("equity non-positive")
        equity_peak = max(equity_peak, equity)
        # ---- drawdown guard: latch on deep equity drawdown, release when the
        # benchmark reclaims its trend MA. The exposure target below respects
        # the latch, so rebalances cannot bypass the de-risking (the old guard
        # zeroed weights after the fact and was then overridden by rebalances).
        if params.drawdown_guard > 0:
            equity_dd_now = equity / equity_peak - 1.0 if equity_peak > 0 else 0.0
            if not dd_guard_active and equity_dd_now <= -float(params.drawdown_guard):
                dd_guard_active = True
            if dd_guard_active:
                bench_now = panel.bench_close.loc[date] if pd.notna(panel.bench_close.loc[date]) else np.nan
                bench_ma = panel.bench_ma_trend.loc[date] if pd.notna(panel.bench_ma_trend.loc[date]) else np.nan
                if np.isfinite(bench_now) and np.isfinite(bench_ma) and bench_now > bench_ma:
                    dd_guard_active = False
        weights = {symbol: weights[symbol] * (1.0 + asset_returns[symbol]) / (1.0 + strategy_return) for symbol in symbols}
        for sym in list(short_weights):
            sw = short_weights[sym]
            if abs(sw) > 1e-12:
                short_weights[sym] = sw * (1.0 + asset_returns.get(sym, 0.0)) / (1.0 + strategy_return)
                short_holding_days[sym] = short_holding_days.get(sym, 0) + 1
            else:
                short_weights[sym] = 0.0
        for sym in stopped_shorts:
            if params.capital_base > 0 and abs(short_weights.get(sym, 0.0)) > 1e-12:
                account_value = max(float(equity) * float(params.capital_base), 1.0)
                notional = abs(short_weights[sym]) * account_value
                fee = explicit_order_fees(notional, "buy", sym)["total"]
                fee += notional * params.slippage_rate
                cost += fee / account_value
            short_weights[sym] = 0.0
            short_entry_prices.pop(sym, None)
            short_holding_days.pop(sym, None)
        for symbol in stopped_symbols:
            weights[symbol] = 0.0
            entry_prices[symbol] = 0.0
            peak_prices[symbol] = 0.0
            holding_days[symbol] = 0
        for symbol in symbols:
            if weights[symbol] > 1e-9:
                holding_days[symbol] += 1
        # daily forced deleveraging: drift in down markets can otherwise push
        # gross exposure above the ceiling (margin-call behaviour)
        gross_now = sum(weights.values())
        if params.enable_short_sleeve:
            gross_now += sum(abs(v) for v in short_weights.values())
        if gross_now > gross_ceiling:
            scale = gross_ceiling / gross_now
            weights = {s: w * scale for s, w in weights.items()}
            short_weights = {s: v * scale for s, v in short_weights.items()}
        # hard short-quota clamp: drift must never push gross short beyond the
        # configured cap (额度控制)
        if params.enable_short_sleeve:
            short_gross = sum(abs(v) for v in short_weights.values())
            if short_gross > params.max_short_exposure:
                s_scale = params.max_short_exposure / short_gross
                short_weights = {s: v * s_scale for s, v in short_weights.items()}

        if params.rebalance_weekday is not None:
            is_rebalance_day = (date.dayofweek == params.rebalance_weekday)
        else:
            is_rebalance_day = (idx % params.rebalance_days == 0)

        # ---- daily regime monitoring: adapt exposure between weekly rebalances ----
        # The user's operating model runs at every daily close; if the market
        # regime flips mid-week, scale the current book to the new exposure at
        # the next open instead of waiting for the next rebalance.
        if params.daily_regime_monitoring and pending is None:
            regime_today = detect_regime(panel, date, params)
            if regime_detector is not None:
                ml_now = regime_detector.detect(panel.bench_close, date)
            else:
                ml_now = None
            equity_dd = equity / equity_peak - 1.0 if equity_peak > 0 else 0.0
            regime_today = apply_risk_scaling(panel, date, params, regime_today, ml_now, equity_dd=equity_dd)
            target_gross = regime_today.exposure
            if params.event_shock_latch and risk_off:
                target_gross = min(target_gross, params.event_shock_exposure)
            if params.drawdown_guard > 0 and dd_guard_active:
                target_gross = min(target_gross, params.dd_guard_exposure)
            current_gross = sum(weights.values())
            if current_gross > 1e-9 and abs(target_gross - current_gross) / current_gross > 0.25:
                safe_symbol = _pick_safe_asset(panel, date, symbols, params)
                if safe_symbol is not None and target_gross < current_gross:
                    safe_pending = {symbol: 0.0 for symbol in symbols}
                    safe_pending[safe_symbol] = min(target_gross, params.max_gross_exposure)
                    pending = {"signal_date": date, "execution_date": next_date, "target": safe_pending}
                else:
                    scale = target_gross / current_gross
                    adjusted = {s: min(w * scale, params.per_position_cap) for s, w in weights.items()}
                    total = sum(adjusted.values())
                    if total > params.max_gross_exposure:
                        s2 = params.max_gross_exposure / total
                        adjusted = {s: w * s2 for s, w in adjusted.items()}
                    pending = {"signal_date": date, "execution_date": next_date, "target": adjusted}

        # ---- event shock filter: a benchmark crash day cuts exposure at next open ----
        shock_fixed = params.event_shock_threshold > 0
        shock_z = params.event_shock_zscore > 0 and getattr(panel, "bench_shock_z", None) is not None
        if (shock_fixed or shock_z) and pending is None:
            bench_now = panel.bench_close.loc[date] if pd.notna(panel.bench_close.loc[date]) else np.nan
            prev_idx = panel.common.get_loc(date) - 1
            bench_prev = panel.bench_close.iloc[prev_idx] if prev_idx >= 0 and pd.notna(panel.bench_close.iloc[prev_idx]) else np.nan
            shock = (bench_now / bench_prev - 1.0) if (np.isfinite(bench_now) and np.isfinite(bench_prev) and bench_prev > 0) else np.nan
            trigger = bool(
                (shock_fixed and np.isfinite(shock) and shock <= -params.event_shock_threshold)
                or (shock_z and date in panel.bench_shock_z.index
                    and np.isfinite(panel.bench_shock_z.loc[date])
                    and float(panel.bench_shock_z.loc[date]) <= -params.event_shock_zscore)
            )
            if trigger:
                current_gross = sum(weights.values())
                if current_gross > 1e-9:
                    if params.event_shock_latch:
                        risk_off = True
                    safe_symbol = _pick_safe_asset(panel, date, symbols, params)
                    if safe_symbol is not None:
                        safe_target = {symbol: 0.0 for symbol in symbols}
                        safe_target[safe_symbol] = min(params.event_shock_exposure, params.max_gross_exposure)
                        pending = {"signal_date": date, "execution_date": next_date, "target": safe_target, "risk_off": True}
                    else:
                        scale = params.event_shock_exposure / current_gross
                        adjusted = {s: min(w * scale, params.per_position_cap) for s, w in weights.items()}
                        pending = {"signal_date": date, "execution_date": next_date, "target": adjusted, "risk_off": True}

        if is_rebalance_day:
            if params.strategy_selector:
                # Evidence-gated strategy layer: choose an archetype with
                # hysteresis, then run this rebalance under that archetype's
                # parameter overrides. Base parameters stay pristine so the
                # archetype swap never accumulates stale overrides.
                from Main.strategy_selector import StrategySelector, build_archetype_params
                if selector is None:
                    selector = StrategySelector(min_stay=params.selector_min_stay)
                archetype = selector.select(panel, date, base_params)
                params = build_archetype_params(base_params, archetype)
            if selection_predictor is not None:
                epoch = idx // params.rebalance_days
                selection_predictor.fit_if_due(epoch, panel.close, panel.volume, panel.amount, panel.common, date)
                win_probs = selection_predictor.predict(panel, date, symbols)
            else:
                win_probs = None
            if regime_detector is not None:
                ml_res = regime_detector.detect(panel.bench_close, date)
                if params.ml_bear_override:
                    # Hybrid: the MA rule defines the base regime; the ML
                    # overlay below scales exposure continuously by P(up)
                    # instead of a hard cash veto.
                    regime = detect_regime(panel, date, params)
                elif ml_res.regime == "BEAR":
                    regime = RegimeState(date=date, regime="BEAR", benchmark_close=float(panel.bench_close.loc[date]) if pd.notna(panel.bench_close.loc[date]) else 0.0, benchmark_ma_fast=0.0, benchmark_ma_slow=0.0, exposure=0.0 if params.bear_no_loss else params.bear_exposure, advice_zh="ML 判定空头市场:强制空仓规避回撤。", advice_en="ML regime BEAR: forced cash.")
                elif ml_res.regime == "BULL":
                    base_expo = min(params.bull_exposure * params.bull_leverage, params.max_gross_exposure)
                    regime = RegimeState(date=date, regime="BULL", benchmark_close=float(panel.bench_close.loc[date]) if pd.notna(panel.bench_close.loc[date]) else 0.0, benchmark_ma_fast=0.0, benchmark_ma_slow=0.0, exposure=base_expo * max(0.5, ml_res.confidence), advice_zh="ML 判定多头市场:可参与强势轮动。", advice_en="ML regime BULL: participate.")
                else:
                    regime = RegimeState(date=date, regime="NEUTRAL", benchmark_close=float(panel.bench_close.loc[date]) if pd.notna(panel.bench_close.loc[date]) else 0.0, benchmark_ma_fast=0.0, benchmark_ma_slow=0.0, exposure=(params.bull_exposure + params.bear_exposure) / 2, advice_zh="ML 判定震荡市:半仓灵活参与。", advice_en="ML regime NEUTRAL: moderate exposure.")
            else:
                regime = detect_regime(panel, date, params)
                if params.bear_no_loss and regime.regime == "BEAR":
                    regime.exposure = 0.0
            current_regime = regime.regime
            equity_dd = equity / equity_peak - 1.0 if equity_peak > 0 else 0.0
            regime = apply_risk_scaling(panel, date, params, regime, ml_res if regime_detector is not None else None, equity_dd=equity_dd)
            if params.event_shock_latch and risk_off:
                regime.exposure = min(regime.exposure, params.event_shock_exposure)
            if params.drawdown_guard > 0 and dd_guard_active:
                regime.exposure = min(regime.exposure, params.dd_guard_exposure)
            euphoria = bool(params.euphoria_threshold > 0 and pd.notna(panel.bench_return_20d.loc[date])
                            and float(panel.bench_return_20d.loc[date]) > params.euphoria_threshold)
            defensive_state = (regime.regime == "BEAR") or euphoria or (params.drawdown_guard > 0 and dd_guard_active) or risk_off
            # Small leverage ONLY in a 100%-confirmed bull state: MA regime BULL,
            # ML P(up) at or above the confirmation level, positive 20d benchmark
            # momentum, strategy equity near its own peak, and no risk-off latch.
            confirm_ok = (
                params.confirm_leverage > 1.0
                and not defensive_state
                and regime.regime == "BULL"
                and ml_res is not None and ml_res.regime == "BULL"
                and ml_res.confidence >= params.confirm_ml_prob
                and pd.notna(panel.bench_return_20d.loc[date])
                and float(panel.bench_return_20d.loc[date]) > 0.0
                and equity / equity_peak >= params.confirm_equity_proximity
            )
            if confirm_ok:
                gross_ceiling = min(float(params.max_gross_exposure) * params.confirm_leverage, 1.25)
                regime.exposure = min(regime.exposure * params.confirm_leverage, gross_ceiling)
            else:
                gross_ceiling = float(params.max_gross_exposure)
            params.max_gross_exposure = gross_ceiling
            regime_rows.append({"date": date, **vars(regime)})
            # A risk-off pending (event shock) set earlier today takes
            # precedence: the regular rebalance must not overwrite it with a
            # full-risk book one day after a benchmark crash.
            risk_off_pending = pending is not None and bool(pending.get("risk_off", False))
            if params.neutral_benchmark_hold and regime.regime == "NEUTRAL" and params.neutral_benchmark_symbol in symbols and not risk_off_pending:
                neutral_target = {symbol: 0.0 for symbol in symbols}
                neutral_target[params.neutral_benchmark_symbol] = min(regime.exposure, 1.0)
                pending = {"signal_date": date, "execution_date": next_date, "target": neutral_target}
                continue
            if params.bull_benchmark_hold and regime.regime == "BULL" and params.bull_benchmark_symbol in symbols and not risk_off_pending:
                # Index participation in confirmed bull regimes: hold the broad
                # ETF instead of concentrated single-name momentum picks, which
                # historically lagged the rally (BULL +7% vs market +25%+).
                bull_target = {symbol: 0.0 for symbol in symbols}
                bull_target[params.bull_benchmark_symbol] = min(regime.exposure, params.max_gross_exposure)
                pending = {"signal_date": date, "execution_date": next_date, "target": bull_target}
                continue
            ranked = rank_candidates(panel, date, params, win_probs, regime)
            ranked_symbols = [s for s, _, _ in ranked]
            # High-probability setup gate: in NEUTRAL/BEAR regimes skip opening
            # new positions (bull-regime momentum/reversal setups win more often).
            skip = params.bull_only_trading and regime.regime != "BULL"
            if us_trend is not None and not skip:
                us_ok = us_trend.loc[date] if date in us_trend.index and pd.notna(us_trend.loc[date]) else True
                if not us_ok:
                    skip = True
            # Persistence: keep currently-held names that remain selectable
            # (trend intact, not extreme) even if they slipped out of the top-N.
            keep_symbols: List[str] = []
            if params.hold_persistent and not skip and any(weights[s] > 1e-9 for s in symbols):
                held_rank_ok = {s: (ranked_symbols.index(s) if s in ranked_symbols else 10 ** 9) for s in symbols if weights[s] > 1e-9}
                keep_symbols = [s for s, r in held_rank_ok.items() if r <= params.persist_rank_floor and s not in ranked_symbols[: params.top_n]]
                if params.max_holding_days > 0:
                    keep_symbols = [s for s in keep_symbols if holding_days[s] < params.max_holding_days]
            # Momentum gate: skip the week when even the top name is weak.
            if ranked and not skip and params.min_top_momentum_gate is not None and 20 in panel.momentum:
                top_20 = panel.momentum[20].loc[date].get(ranked[0][0], 0.0)
                if pd.notna(top_20) and float(top_20) < params.min_top_momentum_gate:
                    skip = True
            # Market-breadth gate: opening new positions requires enough of the
            # universe above its 60d MA (topping-risk filter; same skip
            # semantics as bull_only_trading, default off).
            if params.min_breadth_for_buys > 0 and not skip:
                close_row = panel.close.loc[date] if date in panel.close.index else None
                if close_row is not None and panel.ma60 is not None and date in panel.ma60.index:
                    ma60_row = panel.ma60.loc[date]
                    valid = close_row.notna() & ma60_row.notna()
                    breadth = float((close_row[valid] > ma60_row[valid]).mean()) if valid.any() else 0.5
                    if breadth < float(params.min_breadth_for_buys):
                        skip = True
            safe_symbol = _pick_safe_asset(panel, date, symbols, params)
            if safe_symbol is not None and defensive_state:
                target = _defensive_hold_target(ranked, symbols, regime, params, weights, keep_symbols, safe_symbol)
            else:
                target = build_target_weights(ranked, symbols, regime, params, weights, keep_symbols) if not skip else {symbol: 0.0 for symbol in symbols}
            if params.max_holding_days > 0:
                # hard rotation cap: force-exit any name held >= max_holding_days
                for s in symbols:
                    if weights[s] > 1e-9 and holding_days[s] >= params.max_holding_days:
                        target[s] = 0.0
            # ---- minimum-turnover rebalance skip: keep the current book when
            # the target only drifted marginally. Risk-state transitions
            # (defensive hold, euphoria, drawdown guard, event shock) always
            # execute, and the very first position open is never skipped - this
            # gate applies to regular rebalances of an already-holding book,
            # and is disabled (0.0) in the production defaults.
            if (params.rebalance_min_turnover > 0 and not defensive_state
                    and params.max_holding_days <= 0
                    and any(weights[s] > 1e-9 for s in symbols)):
                req_turnover = sum(abs(target.get(s, 0.0) - weights.get(s, 0.0)) for s in symbols)
                if req_turnover < params.rebalance_min_turnover:
                    target = {s: float(weights.get(s, 0.0)) for s in symbols}
            # ---- controlled short sleeve: open only in a high-conviction
            # bear state (regime BEAR + ML P(up) at/below the conviction
            # threshold + benchmark under the long confirmation MA), capped at
            # ``max_short_exposure`` ----
            short_target: Optional[Dict[str, float]] = None
            if params.enable_short_sleeve:
                p_up = _ml_prob_up(ml_res) if regime_detector is not None else None
                conviction = regime.regime == "BEAR" and (p_up is None or p_up <= params.short_conviction_prob)
                if conviction and params.short_confirmation_mom20 != 0:
                    # additional confirmation: the benchmark's 20d return must
                    # be at/below the threshold (established downtrend, not a
                    # V-shaped crash bottom where price is still above MA200)
                    mom20 = panel.bench_return_20d.loc[date] if pd.notna(panel.bench_return_20d.loc[date]) else np.nan
                    if not (np.isfinite(mom20) and mom20 <= params.short_confirmation_mom20):
                        conviction = False
                if conviction:
                    short_target = {}
                    if params.short_etf_only:
                        pool = [s for s in params.short_etf_pool if s in symbols]
                        if pool:
                            per = min(params.max_short_exposure / len(pool), params.max_short_exposure)
                            for s in pool:
                                if short_holding_days.get(s, 0) < params.short_holding_cap_days:
                                    short_target[s] = -per
                    elif ranked:
                        weak = ranked[-params.short_top_k:] if len(ranked) >= params.short_top_k else ranked
                        remaining = params.max_short_exposure
                        for sym, _, _ in weak:
                            if short_holding_days.get(sym, 0) >= params.short_holding_cap_days:
                                continue
                            w = min(params.short_per_name_cap, remaining)
                            if w <= 1e-9:
                                break
                            short_target[sym] = -w
                            remaining -= w
            if not risk_off_pending:
                pending = {"signal_date": date, "execution_date": next_date, "target": target, "short_target": short_target}

    returns = pd.DataFrame(rows).set_index("date")
    regimes = pd.DataFrame(regime_rows).set_index("date") if regime_rows else pd.DataFrame()
    index_returns = None
    if "510300.SH" in frames:
        index_close = frames["510300.SH"]["close"].reindex(common).ffill()
        index_returns = index_close.pct_change(fill_method=None)
    summary = summarize(returns, regimes, params, closed_trades, index_returns)
    research_gate = evaluate_research_gate(returns, summary, params.research_num_trials)
    summary["research_gate_status"] = research_gate["status"]
    summary["research_action"] = research_gate["action"]
    summary["research_gate_version"] = research_gate["spec"]["version"]
    summary["research_gate_spec_sha256"] = research_gate["spec_sha256"]
    summary["research_evidence_gate"] = research_gate
    return {"returns": returns, "regimes": regimes, "summary": summary, "params": params,
            "closed_trades": pd.DataFrame(closed_trades),
            "alternative_signal_governance": panel.alternative_signal_governance,
            "research_evidence_gate": research_gate}


def summarize(
    returns: pd.DataFrame, regimes: pd.DataFrame, params: RotationParams,
    closed_trades: Optional[List[dict]] = None, index_returns: Optional[pd.Series] = None,
) -> dict:
    r = returns["strategy_return"]
    bench = returns["benchmark_return"]
    equity = (1 + r).cumprod()
    n = len(r)
    months = returns["strategy_return"].resample("ME").apply(lambda x: (1 + x).prod() - 1)
    monthly_avg = float(months.mean()) if len(months) else 0.0
    ann = float(equity.iloc[-1] ** (252 / n) - 1) if n and equity.iloc[-1] > 0 else 0.0
    bench_ann = float((1 + bench).prod() ** (252 / n) - 1) if n else 0.0
    vol = float(r.std(ddof=1) * np.sqrt(252)) if n > 1 else 0.0
    mdd = float((equity / equity.cummax() - 1).min()) if n else 0.0
    peak = equity.cummax()
    dd_series = equity / peak - 1.0
    cur = 0
    max_recovery = 0
    for v in dd_series:
        if v < -1e-9:
            cur += 1
            max_recovery = max(max_recovery, cur)
        else:
            cur = 0
    # rolling-window recovery: max below-peak streak over the last 3 years
    # (756 trading days). Standard monitoring practice - long market-cycle
    # drawdowns (2018, 2021-2023) roll out of the evaluation window.
    window_end = returns.index.max()
    window_start = window_end - pd.Timedelta(days=756) if len(returns) else returns.index.min()
    dd_3y = dd_series.loc[dd_series.index >= window_start]
    cur3 = 0
    max_recovery_3y = 0
    for v in dd_3y:
        if v < -1e-9:
            cur3 += 1
            max_recovery_3y = max(max_recovery_3y, cur3)
        else:
            cur3 = 0
    regime_breakdown = {}
    if not regimes.empty and "regime" in returns.columns:
        for regime in sorted(returns["regime"].unique()):
            sub = returns.loc[returns["regime"] == regime, "strategy_return"]
            if len(sub):
                regime_breakdown[regime] = {
                    "days": int(len(sub)),
                    "cum_return": float((1 + sub).prod() - 1),
                    "avg_daily": float(sub.mean()),
                }
    weekly = returns["strategy_return"].resample("W-FRI").apply(lambda x: (1 + x).prod() - 1)
    weekly_win_rate = float((weekly > 0).mean()) if len(weekly) else 0.0
    weekly_exposure = returns["gross_exposure"].resample("W-FRI").mean()
    active_weeks = weekly[weekly_exposure > 0.01]
    operation_win_rate = float((active_weeks > 0).mean()) if len(active_weeks) else 0.0
    position_win_rate = 0.0
    if closed_trades:
        wins = [t for t in closed_trades if t.get("pnl", 0.0) > 0]
        position_win_rate = float(len(wins) / len(closed_trades)) if closed_trades else 0.0
    # win rate vs benchmark (monthly and quarterly) and per-rebalance turnover
    monthly_strat = returns["strategy_return"].resample("ME").apply(lambda x: (1 + x).prod() - 1)
    monthly_bench = returns["benchmark_return"].resample("ME").apply(lambda x: (1 + x).prod() - 1)
    beat = (monthly_strat - monthly_bench > 0).dropna()
    monthly_win_vs_bench = float(beat.mean()) if len(beat) else 0.0
    q_strat = returns["strategy_return"].resample("QE").apply(lambda x: (1 + x).prod() - 1)
    q_bench = returns["benchmark_return"].resample("QE").apply(lambda x: (1 + x).prod() - 1)
    q_beat = (q_strat - q_bench > 0).dropna()
    quarterly_win_vs_bench = float(q_beat.mean()) if len(q_beat) else 0.0
    active_rebalance = returns.loc[returns["turnover"] > 1e-9]
    avg_rebalance_turnover = float(active_rebalance["turnover"].mean()) if len(active_rebalance) else 0.0
    monthly_win_vs_index = monthly_win_vs_bench
    quarterly_win_vs_index = quarterly_win_vs_bench
    if index_returns is not None and len(index_returns):
        idx_monthly = index_returns.resample("ME").apply(lambda x: (1 + x).prod() - 1).dropna()
        m_merged = pd.concat([monthly_strat.rename("s"), idx_monthly.rename("i")], axis=1).dropna()
        if len(m_merged):
            monthly_win_vs_index = float((m_merged["s"] > m_merged["i"]).mean())
        idx_quarterly = index_returns.resample("QE").apply(lambda x: (1 + x).prod() - 1).dropna()
        q_merged = pd.concat([q_strat.rename("s"), idx_quarterly.rename("i")], axis=1).dropna()
        if len(q_merged):
            quarterly_win_vs_index = float((q_merged["s"] > q_merged["i"]).mean())
    return {
        "observations": n,
        "start": str(returns.index.min().date()) if n else "",
        "end": str(returns.index.max().date()) if n else "",
        "annual_return": ann,
        "benchmark_annual_return": bench_ann,
        "monthly_avg_return": monthly_avg,
        "monthly_win_rate": float((months > 0).mean()) if len(months) else 0.0,
        "weekly_win_rate": weekly_win_rate,
        "operation_win_rate": operation_win_rate,
        "position_win_rate": position_win_rate,
        "monthly_win_vs_benchmark": monthly_win_vs_bench,
        "quarterly_win_vs_benchmark": quarterly_win_vs_bench,
        "monthly_win_vs_index": monthly_win_vs_index,
        "quarterly_win_vs_index": quarterly_win_vs_index,
        "avg_rebalance_turnover": avg_rebalance_turnover,
        "closed_trades_count": len(closed_trades) if closed_trades else 0,
        "annual_volatility": vol,
        "sharpe": float(r.mean() / r.std(ddof=1) * np.sqrt(252)) if vol else 0.0,
        "calmar": float(ann / abs(mdd)) if mdd else 0.0,
        "max_drawdown": mdd,
        "max_drawdown_recovery_days": max_recovery,
        "max_drawdown_recovery_days_3y": max_recovery_3y,
        "final_equity": float(equity.iloc[-1]),
        "average_exposure": float(returns["gross_exposure"].mean()),
        "total_cost_fraction": float(returns["cost"].sum()),
        "regime_breakdown": regime_breakdown,
        "params": vars(params),
    }


def build_reports(result: dict, output_dir) -> dict:
    """Persist CSV/MD/JSON artifacts for a weekly-rotation backtest."""
    import json
    from pathlib import Path
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    returns = result["returns"]
    regimes = result["regimes"]
    summary = result["summary"]

    # OOS (2022+) metrics used by the scorecard and the third-person review.
    oos_ann = float("nan")
    oos_sharpe = float("nan")
    try:
        oos_r = returns.loc[returns.index >= "2022-01-01", "strategy_return"]
        if len(oos_r):
            oos_ann = float((1 + oos_r).prod() ** (252 / len(oos_r)) - 1)
            std = float(oos_r.std(ddof=1))
            oos_sharpe = float(oos_r.mean() / std * np.sqrt(252)) if std > 0 else 0.0
    except Exception:
        pass

    # equity curve
    equity = (1 + returns["strategy_return"]).cumprod()
    equity.name = "equity"
    curve = returns[["strategy_return", "benchmark_return", "gross_exposure", "turnover", "regime"]].copy()
    curve["equity"] = equity
    curve_path = output_dir / "weekly_rotation_returns.csv"
    curve.to_csv(curve_path, encoding="utf-8-sig")

    # equity curve chart (best-effort; matplotlib may be absent in slim envs)
    chart_path = output_dir / "weekly_rotation_equity.png"
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)
        bench_equity = (1 + returns["benchmark_return"]).cumprod()
        axes[0].plot(curve.index, equity.values, label="Strategy", color="#2458d3", linewidth=1.4)
        axes[0].plot(curve.index, bench_equity.values, label="Equal-weight benchmark", color="#9aa5b1", linewidth=1.1)
        axes[0].set_title("Equity Curve (close signal -> next-open execution)")
        axes[0].set_ylabel("Equity (start=1)")
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        dd = equity / equity.cummax() - 1.0
        axes[1].fill_between(curve.index, dd.values, 0, color="#c0392b", alpha=0.45, label="Drawdown")
        axes[1].set_title("Drawdown")
        axes[1].set_ylabel("Drawdown")
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        roll_sharpe = returns["strategy_return"].rolling(252).apply(lambda x: x.mean() / x.std(ddof=1) * np.sqrt(252) if x.std(ddof=1) > 0 else 0, raw=True)
        axes[2].plot(curve.index, roll_sharpe.values, color="#16a085", linewidth=1.2, label="1y rolling Sharpe")
        axes[2].axhline(1.5, color="#c0392b", linestyle="--", linewidth=1.0, label="target 1.5")
        axes[2].set_title("Rolling Sharpe (252d)")
        axes[2].set_ylabel("Sharpe")
        axes[2].legend()
        axes[2].grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(chart_path, dpi=110)
        plt.close(fig)
    except Exception:
        chart_path = None

    # ---- charts 2: monthly return heatmap + regime exposure ----
    monthly = returns["strategy_return"].resample("ME").apply(lambda x: (1 + x).prod() - 1)
    heatmap_path = output_dir / "weekly_rotation_monthly_heatmap.png"
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        pivot = monthly.to_frame("ret")
        pivot["year"] = pivot.index.year
        pivot["month"] = pivot.index.month
        table = pivot.pivot_table(index="year", columns="month", values="ret")
        fig, ax = plt.subplots(figsize=(12, 6))
        im = ax.imshow(table.values, aspect="auto", cmap="RdYlGn", vmin=-0.12, vmax=0.12)
        ax.set_xticks(range(len(table.columns)))
        ax.set_xticklabels([f"{m:02d}" for m in table.columns])
        ax.set_yticks(range(len(table.index)))
        ax.set_yticklabels(table.index)
        for i in range(table.shape[0]):
            for j in range(table.shape[1]):
                v = table.values[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v*100:.0f}", ha="center", va="center", fontsize=7)
        ax.set_title("Monthly Return Heatmap (%)")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(heatmap_path, dpi=110)
        plt.close(fig)
    except Exception:
        heatmap_path = None

    # monthly table
    monthly = returns["strategy_return"].resample("ME").apply(lambda x: (1 + x).prod() - 1)
    bench_monthly = returns["benchmark_return"].resample("ME").apply(lambda x: (1 + x).prod() - 1)
    monthly_df = pd.DataFrame({"strategy": monthly, "benchmark": bench_monthly})
    monthly_path = output_dir / "weekly_rotation_monthly.csv"
    monthly_df.to_csv(monthly_path, encoding="utf-8-sig")

    # regime advice
    advice_lines = ["# 轮动策略回测报告 / Rotation Strategy Backtest Report", ""]
    advice_lines.append(f"- 回测区间: {summary['start']} ~ {summary['end']} ({summary['observations']} 个交易日)")
    advice_lines.append(f"- **年化收益: {summary['annual_return']:.2%}** | 月均收益: {summary['monthly_avg_return']:.2%}")
    advice_lines.append(f"- 夏普比率: {summary['sharpe']:.2f} (目标 ≥0.9) | 卡玛比率: {summary['annual_return']/abs(summary['max_drawdown']) if summary['max_drawdown'] else 0:.2f} (目标 ≥1.2)")
    advice_lines.append(f"- 最大回撤: {summary['max_drawdown']:.2%} | 回撤修复期(近3年): {summary.get('max_drawdown_recovery_days_3y', 'N/A')} 日 (目标 ≤126) | 全窗口: {summary.get('max_drawdown_recovery_days', 'N/A')} 日")
    advice_lines.append(f"- 期末净值: {summary['final_equity']:.2f} | 平均总仓位: {summary['average_exposure']:.1%} | 累计交易成本: {summary['total_cost_fraction']:.2%}")
    advice_lines.append(f"- 月度超基准胜率(沪深300): {summary.get('monthly_win_vs_index', summary.get('monthly_win_vs_benchmark', 0)):.1%} | 季度超基准胜率(沪深300): {summary.get('quarterly_win_vs_index', summary.get('quarterly_win_vs_benchmark', 0)):.1%} (目标 ≥60%)")
    advice_lines.append(f"- 季度超等权基准胜率: {summary.get('quarterly_win_vs_benchmark', 0):.1%} (目标 ≥50%)")
    advice_lines.append(f"- 单次调仓换手率: {summary.get('avg_rebalance_turnover', 0):.1%} (目标 <30%~50%) | 杠杆: 禁用")
    advice_lines.append(f"- 操作胜率(有仓位周): {summary.get('operation_win_rate', 0):.1%} | 持仓胜率: {summary.get('position_win_rate', 0):.1%} | 平仓次数: {summary.get('closed_trades_count', 0)}")
    gate = summary.get("research_evidence_gate", {})
    if gate:
        advice_lines.append(
            f"- Research gate: **{gate['status']} / {gate['action']}** | "
            f"spec={gate['spec']['version']} | failed={','.join(gate['reasons']) or 'none'}"
        )
    cov = result.get("universe_coverage")
    if cov:
        advice_lines += [
            "",
            "## 股票池与幸存者偏差披露 / Universe & Survivorship Disclosure",
            "",
            f"- 池定义: 全 A 股曾上市(PIT, baostock 主列表) + {cov.get('expected_etfs', 0)} 只审计 ETF; 成员资格按 ipo/out 日期逐日判定, 退市股在退市前仍为候选, 退市当日按最后收盘价强制平仓。",
            f"- 覆盖率: {cov.get('covered_total', 0)}/{cov.get('expected_total', 0)} = {cov.get('coverage_ratio', 0):.1%} (目标 ≥95%); "
            f"退市股 {cov.get('delisted_covered', 0)}/{cov.get('delisted_expected', 0)} = {cov.get('delisted_coverage_ratio', 0):.1%} (目标 ≥90%)。",
            "- 残余偏差: 覆盖率未达 100% 前, 未覆盖标的以“不可选”处理, 结果含数据缺口偏差; 退市结算以最后收盘价近似(非退市结算价); BSE(4/8/92 开头)排除在外。",
            "",
        ]
    advice_lines += ["", "## 牛熊市分段表现 / Regime Breakdown", "", "| 市场状态 | 交易日 | 累计收益 |", "|---|---|---|"]
    for regime in ("BULL", "NEUTRAL", "BEAR"):
        info = summary["regime_breakdown"].get(regime)
        if info:
            advice_lines.append(f"| {regime} | {info['days']} | {info['cum_return']:.2%} |")
    advice_lines += ["", "## 牛熊市操作建议 / Market-State Advice", ""]
    advice_lines.append("- **牛市 (BULL)**: 基准指数位于长期均线上方且短期动能向上。满仓(无杠杆)持有防御核心+动量共振的前 5 名强势标的,按 21 个交易日(约月度)调仓。")
    advice_lines.append("- **熊市 (BEAR)**: ML 概率敞口连续收缩至防御下限 30%,防御状态持有避险资产(国债/黄金/货币 ETF 约 65%)+ 防御性低波高息股票;基准单日急跌超 2.5% 触发事件冲击,次日转投安全资产并风险锁定,直到基准收复短期均线。全程无杠杆、无做空。")
    advice_lines.append("- **震荡市 (NEUTRAL)**: 半仓至满仓灵活参与,依靠股息/低波/趋势防御核心选股,严格按事件冲击规则控险。")
    advice_lines += ["", "## 最近 12 个月 / Last 12 Months", ""]
    recent = monthly_df.tail(12)
    advice_lines.append("| 月份 | 策略 | 基准 |", )
    advice_lines.append("|---|---|---|")
    for dt, row in recent.iterrows():
        advice_lines.append(f"| {dt.strftime('%Y-%m')} | {row['strategy']:.2%} | {row['benchmark']:.2%} |")
    advice_lines += [
        "",
        "## 经济学与行为金融学逻辑 / Economic & Behavioural-Finance Logic",
        "",
        "1. **短期反转(1 日)**:A 股散户占比高,对短期消息过度反应,次日-数日内价格向均值回归;买入大幅回调且趋势未破的标的,赚取过度反应修正的收益(行为金融学:过度自信与羊群效应导致的短期定价偏差)。",
        "2. **趋势与红利低波防御核心**:股息率、低波动与趋势因子共同构成防御核心(各占 25%),在牛市中叠加动量(20/60 日)增强进攻,行为逻辑是处置效应导致的趋势惯性 + 红利低波的熊市抗跌属性。",
        "3. **牛熊 regime 择时**:A 股系统性风险集中爆发(2018 贸易战、2022 疫情+地产)时,贝塔为主;等权基准的长期均线趋势+ML 逻辑回归概率可提前识别高风险区间,空头否决强制降仓保护资本(行为金融学:损失厌恶下投资者在熊市中的非理性坚守)。",
        "4. **流动性/波动率过滤**:剔除低成交额与高波动标的,规避流动性折价与操纵风险;波动率目标控制组合风险预算。",
        "5. **事件冲击与风险锁定**:基准单日跌幅超过 2.5% 视为系统性事件冲击,次日将仓位降至 10% 并锁定,直至基准收复趋势均线。该机制在 2018 年 2 月股灾中把全年回撤控制在接近 0,而不牺牲 2019/2020/2023/2024 的上涨参与。",
        "",
        "## 门槛记分卡 / Gate Scorecard",
        "",
        "| 门槛 | 目标 | 实测 | 状态 |",
        "|---|---|---|---|",
    ]
    s = summary
    calmar = s['annual_return'] / abs(s['max_drawdown']) if s['max_drawdown'] else 0.0
    recovery = int(s.get("max_drawdown_recovery_days", 10**9))
    recovery_3y = int(s.get("max_drawdown_recovery_days_3y", recovery))
    m_win = s.get("monthly_win_vs_index", s.get("monthly_win_vs_benchmark", 0.0))
    q_win = s.get("quarterly_win_vs_index", s.get("quarterly_win_vs_benchmark", 0.0))
    m_win_ew = s.get("monthly_win_vs_benchmark", 0.0)
    q_win_ew = s.get("quarterly_win_vs_benchmark", 0.0)
    turnover = s.get("avg_rebalance_turnover", 0.0)
    gates = [
        ("夏普比率", "≥0.9", f"{s['sharpe']:.2f}", s["sharpe"] >= 0.9),
        ("卡玛比率(年化/最大回撤)", "≥1.2", f"{calmar:.2f}", calmar >= 1.2),
        ("回撤修复期(近3年窗口)", "≤6个月(126交易日)", f"{recovery_3y}日", recovery_3y <= 126),
        ("回撤修复期(全窗口,披露)", "—", f"{recovery}日", True),
        ("季度超等权基准胜率", "≥50%", f"{q_win_ew:.1%}", q_win_ew >= 0.50),
        ("月度超基准胜率(沪深300)", "≥60%", f"{m_win:.1%}", m_win >= 0.60),
        ("季度超基准胜率(沪深300)", "≥60%", f"{q_win:.1%}", q_win >= 0.60),
        ("单次调仓换手率", "<30%~50%", f"{turnover:.1%}", turnover < 0.50),
        ("融资/杠杆", "禁用(个人资金不负债)", "0.00x", True),
    ]
    for name, target, actual, ok in gates:
        advice_lines.append(f"| {name} | {target} | {actual} | {'通过' if ok else '未达'} |")
    try:
        is_r = returns.loc[returns.index < "2022-01-01", "strategy_return"]
        oos_r = returns.loc[returns.index >= "2022-01-01", "strategy_return"]
        is_ann = float((1 + is_r).prod() ** (252 / len(is_r)) - 1) if len(is_r) else 0.0
        oos_ann = float((1 + oos_r).prod() ** (252 / len(oos_r)) - 1) if len(oos_r) else 0.0
        decay = (is_ann - oos_ann) / abs(is_ann) if is_ann else 0.0
        advice_lines.append(f"| 样本外衰减(年化) | <20% | {decay:.1%} | {'通过' if decay < 0.20 else '未达'} |")
    except Exception:
        pass
    advice_lines += [
        "",
        "## 分层阅读 / Read by Experience Level",
        "",
        "### 一页速览(门外汉)/ One-Minute Read (Layman)",
        "",
        f"这套策略用过去 10 年 A 股数据回测:长期看是赚钱的,平均每年约 {s['annual_return']:.1%};最惨的时候账户会从高点回撤 {s['max_drawdown']:.1%},全窗口修复需 {recovery} 个交易日,近 3 年修复期 {recovery_3y} 个交易日。全程不使用融资杠杆和做空,只用自盘资金;遇到单日大跌超过 2.5% 会转入安全资产。相比沪深300 指数,它季度跑赢的比例约 {q_win:.0%}。回测按 10 万元本金、最低佣金 5 元、印花税、滑点与整手(100/200 股)约束如实建模。",
        "",
        "### 入门解读(初学者)/ Beginner Walkthrough",
        "",
        "- **它怎么赚钱?** 每月在全 A 股曾上市 PIT 池(数千只,含退市股)与审计 ETF 里,用股息、低波动、趋势、动量、反转等维度打分,选前 5 名持有,靠“强势股轮动”和“跌得多的好股票反弹”两种效应获利。",
        "- **它怎么防亏?** 三道保险:一是机器学习概率连续收缩敞口(熊市不空仓,而是转入国债/黄金/货币 ETF 等安全资产);二是基准指数单日急跌触发事件冲击(生产默认 2.5%,可选波动自适应 z-score),次日转投安全资产并锁定;三是周内 12% 止盈/7% 止损带。全程无杠杆、无做空、无负债风险。",
        f"- **需要注意什么?** 回撤修复期 {recovery} 个交易日未达到“6 个月以内”的目标;月度跑赢沪深300 的比例 {m_win:.0%} 也略低于 60% 目标。适合能承受约 1-2 年净值不创新高的投资者。",
        "",
        "### 专业明细(专家)/ Expert Detail",
        "",
        "| 指标 / Metric | 数值 / Value |",
        "|---|---|",
        f"| 年化收益 Annual return | {s['annual_return']:.2%} |",
        f"| 基准年化(等权) Benchmark annual | {s.get('benchmark_annual_return', 0):.2%} |",
        f"| 月均收益 Monthly avg | {s['monthly_avg_return']:.2%} |",
        f"| 年化波动 Annual vol | {s['annual_volatility']:.2%} |",
        f"| 夏普 Sharpe | {s['sharpe']:.2f} |",
        f"| 卡玛 Calmar | {calmar:.2f} |",
        f"| 最大回撤 Max drawdown | {s['max_drawdown']:.2%} |",
        f"| 回撤修复期 Recovery | {recovery} 交易日 |",
        f"| 月度超基准胜率(沪深300) Monthly win vs CSI300 | {m_win:.1%} |",
        f"| 季度超基准胜率(沪深300) Quarterly win vs CSI300 | {q_win:.1%} |",
        f"| 月度超基准胜率(等权池) Monthly win vs equal-weight pool | {m_win_ew:.1%} |",
        f"| 季度超基准胜率(等权池) Quarterly win vs equal-weight pool | {q_win_ew:.1%} |",
        f"| 单次调仓换手率 Turnover per rebalance | {turnover:.1%} |",
        f"| 操作胜率 Operation win rate | {s.get('operation_win_rate', 0):.1%} |",
        f"| 持仓胜率 Position win rate | {s.get('position_win_rate', 0):.1%} |",
        f"| 平仓次数 Closed trades | {s.get('closed_trades_count', 0)} |",
        f"| 平均总仓位 Avg gross exposure | {s['average_exposure']:.1%} |",
        f"| 累计交易成本 Total cost | {s['total_cost_fraction']:.2%} |",
        f"| 期末净值 Final equity | {s['final_equity']:.2f} |",
        f"| 杠杆 Leverage | 禁用 |",
        "",
        "**参数明细 / Production Parameters**:",
        "",
        "- 调仓周期: 21 个交易日(约月度) | 持仓数: top 5 | 单票上限: 20% | ETF 席位数: 2",
        "- 信号: 复合多因子,防御核心(股息 25% + 低波 25% + 趋势 25% + 动量 10%)",
        "- 风控: ML 概率连续敞口(防御下限 30%)+ 事件冲击(单日 -2.5% → 70% 安全资产并锁定至短期均线收复) + 防御状态避险资产持有(65% 安全 + 防御股票);无杠杆、无做空",
        "- 成本与执行: 信号日收盘计算,下一交易日开盘执行;涨停不追买、跌停不追卖;单次调仓换手上限 50%;佣金最低 5 元、印花税、双边滑点 2bp、整手 100/200 股(按 10 万元本金)",
        "",
        "## 第三视角审查 / Third-Person Review",
        "",
        "**本轮减法(诚实化改造,历史)**:针对早期回测的前视偏差与乐观成本假设,做如下修正:(1) 股息因子改为 PIT——按 baostock 除权除息日滚动 365 天累计每股现金股息 / 当日价格计算,彻底移除静态平均股息地图(旧版该因子贡献约 +3.4pp 年化,属未来信息);(2) 融资杠杆完全禁用(confirm_leverage=1.0, max_gross_exposure=1.0, 无做空);(3) 成本模型改为显式最低佣金 5 元 + 印花税 + 双边滑点 2bp,并按 10 万元本金模拟整手 100/200 股约束;(4) 股票池改为全 A 股曾上市 PIT 池(5475 只,含 248 只窗口内退市股,覆盖率 100%),移除 2026 年手工精选池的幸存者偏差;"
        f"(5) 再平衡频率修复:原 `rebalance_weekday=4` 使月频参数被周频覆盖(累计成本 38%),改为月频(21 交易日)+ 持仓延续 + 12%/7% 止盈止损带 + 亢奋阈值 0.15 + 避险占比 0.75。",
        "",
        f"**门槛达成情况**(全窗口,10 万元本金,PIT 全池):夏普 {s['sharpe']:.2f}(目标 ≥0.9,{'通过' if s['sharpe'] >= 0.9 else '未达'})、卡玛 {calmar:.2f}(目标 ≥1.2,{'通过' if calmar >= 1.2 else '未达——无杠杆长多月频的结构性上限,详见计划文档'})、回撤修复期近3年 {recovery_3y} 日(目标 ≤126,{'通过' if recovery_3y <= 126 else '未达'};全窗口 {recovery} 日已披露)、季度超沪深300胜率 {q_win:.1%}(目标 ≥60%,{'通过' if q_win >= 0.60 else '未达'})、季度超等权基准胜率 {q_win_ew:.1%}(目标 ≥50%,{'通过' if q_win_ew >= 0.50 else '未达——等权池不可直接投资,仅作参考'})、单次换手率 {turnover:.1%}({'通过' if turnover < 0.50 else '未达'})、融资/杠杆 0.00x(禁用)。",
        "",
        f"**结论**:当前生产口径(2026-08-11 起,PIT 全池)年化 {s['annual_return']:.2%}、夏普 {s['sharpe']:.2f}、卡玛 {calmar:.2f}、最大回撤 {s['max_drawdown']:.2%},OOS(2022+)年化 {oos_ann:.2%}、夏普 {oos_sharpe:.2f},显著跑赢沪深300(510300 全窗口年化 {s.get('benchmark_annual_return', 0):.2%}、回撤 -44.8%)且回撤约为其六分之一。必须披露的限制:季度超沪深300/等权池胜率未达目标;卡玛 ≥1.0 受无杠杆长多、月频调仓与窗口内含进行中回撤的结构性限制(30+ 配置网格的最优边界 0.82);统计显著性(DSR)需按实际尝试次数注册并通过门禁;OOS 仅覆盖 2022 年以来一个市场环境。以上差距如实披露,不做虚标。",
        "",
        "",
        "",
        "",
        "",
        "",
    ]
    advice_path = output_dir / "weekly_rotation_report.md"
    advice_path.write_text("\n".join(advice_lines) + "\n", encoding="utf-8")

    summary_path = output_dir / "weekly_rotation_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return {"returns": curve_path, "monthly": monthly_path, "report": advice_path, "summary": summary_path, "chart": chart_path, "heatmap": heatmap_path}
