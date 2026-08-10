"""Small, rule-based strategy-selection layer for the weekly rotation.

Design goal: absolute adaptivity with the smallest possible footprint, so future
versions only add named archetypes instead of new engine code.

The layer defines a handful of *archetypes* (parameter overrides of the existing
``RotationParams``) and a conservative, point-in-time selector that switches
between them with hysteresis. Everything here is pure and testable; the engine
consumes it only through a tiny optional hook.

Evidence gate: a selector that does not beat the best single archetype
out-of-sample must stay disabled (see ``run_strategy_selector.py``). The
production default keeps ``strategy_selector=""``.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Dict, Optional

import numpy as np
import pandas as pd

from Main.weekly_rotation import RotationParams


STRATEGY_ARCHETYPES: Dict[str, Dict] = {
    # Production default: defensive multi-factor core with regime adaptivity.
    "balanced": {},
    # Momentum/trend participation: pure factor-regime weights, no dividend
    # defensive filter, full bull exposure.
    "momentum": {
        "defensive_core": False,
        "defensive_core_bull_momentum": False,
        "defensive_filter": False,
        "require_relative_strength": True,
        "bull_exposure": 1.0,
        "bear_exposure": 0.0,
        "neutral_exposure": 0.0,
        "min_top_momentum_gate": 0.02,
        "max_short_term_gain": 0.30,
        "top_n": 4,
    },
    # Defensive low-vol/high-dividend core with tighter stops.
    "defensive": {
        "defensive_core": True,
        "defensive_core_bull_momentum": False,
        "defensive_filter": True,
        "max_annual_vol": 0.30,
        "per_position_cap": 0.25,
        "top_n": 6,
        "stop_loss_pct": 0.06,
        "take_profit_pct": 0.05,
        "vol_target": 0.15,
    },
    # Safe-asset rotation: mostly bonds/gold/money ETFs in defensive states.
    "safe": {
        "defensive_hold_exposure": 1.0,
        "defensive_hold_safe_frac": 0.85,
        "safe_trend_gate": 20,
        "defensive_core": True,
        "defensive_filter": True,
        "max_annual_vol": 0.25,
        "per_position_cap": 0.20,
        "top_n": 3,
    },
}


def build_archetype_params(base: RotationParams, name: str) -> RotationParams:
    """Return a copy of ``base`` with the named archetype's overrides applied."""
    if name not in STRATEGY_ARCHETYPES:
        raise ValueError(f"unknown archetype {name!r}")
    return replace(base, **STRATEGY_ARCHETYPES[name])


class StrategySelector:
    """Hysteresis-protected archetype switcher.

    Uses only point-in-time (backward-looking) state at the decision date:
    benchmark trend (reusing the engine's regime rule), realized volatility,
    cross-sectional breadth and safe-asset momentum. A switch requires the new
    archetype to be indicated for ``min_stay`` consecutive rebalances.
    """

    def __init__(self, min_stay: int = 4):
        self.min_stay = max(1, int(min_stay))
        self.last_archetype = "balanced"
        self.stay_count = 0

    def _state(self, panel, date: pd.Timestamp, params: RotationParams) -> Dict:
        try:
            regime = _regime_for(panel, date, params)
        except Exception:
            regime = "NEUTRAL"
        # realized 60d benchmark vol (annualized)
        bench = panel.bench_close.loc[:date].dropna()
        vol = float(bench.pct_change(fill_method=None).tail(60).std(ddof=0) * np.sqrt(252)) if len(bench) >= 30 else 0.5
        # breadth: fraction of the (ex-benchmark) universe above its 60d MA
        close = panel.close.loc[date]
        ma60_panel = getattr(panel, "ma60", None)
        if ma60_panel is not None and date in ma60_panel.index:
            ma60 = ma60_panel.loc[date]
        else:
            ma60 = panel.close.rolling(60, min_periods=20).mean().loc[date]
        valid = close.notna() & ma60.notna()
        breadth = float((close[valid] > ma60[valid]).mean()) if valid.any() else 0.5
        # safe-asset momentum: best 120d momentum among defensive assets
        safe_mom = None
        if 120 in panel.momentum and params.defensive_hold_assets:
            mom = panel.momentum[120].loc[date]
            vals = [float(mom[s]) for s in params.defensive_hold_assets if s in mom.index and np.isfinite(mom[s])]
            if vals:
                safe_mom = max(vals)
        return {"regime": regime, "vol": vol, "breadth": breadth, "safe_mom": safe_mom}

    def select(
        self, panel, date: pd.Timestamp, params: RotationParams
    ) -> str:
        """Return the archetype to use at ``date`` (PIT), updating hysteresis
        state. ``panel`` must be a ``FeaturePanel`` built only from data
        available up to ``date`` (the engine's panels are backward-looking)."""
        st = self._state(panel, date, params)
        if st["regime"] == "BEAR" or st["vol"] >= 0.28:
            want = "safe" if (st["safe_mom"] is not None and st["safe_mom"] > 0.05) else "defensive"
        elif st["regime"] == "BULL" and st["breadth"] >= 0.55 and st["vol"] < 0.22:
            want = "momentum"
        else:
            want = "balanced"
        if want == self.last_archetype:
            self.stay_count = 0
        else:
            self.stay_count += 1
            if self.stay_count >= self.min_stay:
                self.last_archetype = want
                self.stay_count = 0
        return self.last_archetype


def _regime_for(panel, date: pd.Timestamp, params: RotationParams) -> str:
    """Reuse the engine's trend rule to label the regime at ``date``."""
    from Main.weekly_rotation import detect_regime

    return detect_regime(panel, date, params).regime
