"""Dev harness: run the weekly-rotation engine on the locally cached universe.

This is a scratch tool (not part of the shipped product): the cached universe
is the ~420-name PIT-download subset available offline, so absolute numbers are
NOT comparable to the documented full-pool runs. It is used to (a) smoke-test
the engine end-to-end, (b) compare engine logic variants on identical data and
accounting, and (c) verify the research evidence gate responds as expected.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # Quant-4/
sys.path.insert(0, str(PROJECT_ROOT))

from Main.weekly_rotation import RotationParams, weekly_rotation_backtest  # noqa: E402
from Main.research_evidence_gate import evaluate_research_gate  # noqa: E402

DATA_CACHE = PROJECT_ROOT / "Data_Cache"


def load_frames(cache_dir: Path, symbols=None) -> dict:
    frames: dict = {}
    for path in sorted(cache_dir.glob("*_history.parquet")):
        symbol = path.name.replace("_history.parquet", "")
        if symbols is not None and symbol not in symbols:
            continue
        try:
            df = pd.read_parquet(path)
        except Exception:
            continue
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.tz_localize(None)
        df = df.dropna(subset=["date"]).sort_values("date").drop_duplicates("date", keep="last").set_index("date")
        cols = [c for c in ("open", "high", "low", "close", "volume", "amount") if c in df.columns]
        frames[symbol] = df[cols].astype(float)
    return frames


def default_params() -> RotationParams:
    """Adopted production defaults (2026-08-15 full-pool PIT revalidated:
    fixed 3% + z-score 3.0 crash detection, 8% stop band).

    Delegate to ``run_weekly_rotation.default_params()`` so the harness
    baseline always matches the production profile. (Prior to 2026-08-15
    this function baked in the legacy defaults - stop 7%, fixed 2.5% shock,
    no z-score, empty safe-asset holdings - which silently made "baseline"
    sweeps compare against the pre-adoption profile; see R13 loader-fix
    note. Historical sweep JSONs were computed with the legacy bake and are
    preserved as-is.)
    """
    from run_weekly_rotation import default_params as _prod_defaults

    return _prod_defaults()


def oos_metrics(result: dict) -> dict:
    returns = result["returns"]
    oos = returns.loc[returns.index >= "2022-01-01", "strategy_return"]
    n = len(oos)
    growth = float((1.0 + oos).prod())
    ann = float(growth ** (252.0 / n) - 1.0) if n and growth > 0 else -1.0
    std = float(oos.std(ddof=1)) if n > 1 else 0.0
    sharpe = float(oos.mean() / std * np.sqrt(252)) if std > 0 else 0.0
    eq = (1 + oos).cumprod()
    mdd = float((eq / eq.cummax() - 1.0).min()) if n else 0.0
    return {"oos_obs": n, "oos_ann": ann, "oos_sharpe": sharpe, "oos_mdd": mdd,
            "oos_calmar": float(ann / abs(mdd)) if mdd else 0.0}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2016-01-01")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "sweep_results.json")
    parser.add_argument("--trials", type=int, default=None, help="num_trials registered for the DSR gate")
    parser.add_argument("--configs", default="baseline", help="comma list of config keys (see CONFIGS)")
    args = parser.parse_args()

    frames = load_frames(DATA_CACHE)
    print(f"loaded {len(frames)} symbols")
    if len(frames) < 30:
        print("too few symbols", file=sys.stderr)
        return 1

    CONFIGS: dict[str, dict] = {
        "baseline": {},
        "tp10_sl6": {"take_profit_pct": 0.10, "stop_loss_pct": 0.06},
        "tp15_sl9": {"take_profit_pct": 0.15, "stop_loss_pct": 0.09},
        "tp20_sl10": {"take_profit_pct": 0.20, "stop_loss_pct": 0.10},
        "topn7": {"top_n": 7, "per_position_cap": 0.25},
        "topn10": {"top_n": 10, "per_position_cap": 0.15},
        "persist12": {"persist_rank_floor": 12},
        "persist4": {"persist_rank_floor": 4},
        "euph20": {"euphoria_threshold": 0.20},
        "euph10": {"euphoria_threshold": 0.10},
        "sfr0": {"defensive_hold_safe_frac": 0.0},
        "mllogit_off": {"regime_model": "", "ml_bear_override": False},
        "rebal15": {"rebalance_days": 15},
        "rebal30": {"rebalance_days": 30},
        "shock2": {"event_shock_threshold": 0.02},
        "shock3": {"event_shock_threshold": 0.03},
        "no_intraweek": {"enable_intraweek_stops": False},
        "defcore_off": {"defensive_core": False, "defensive_core_bull_momentum": False, "defensive_filter": False},
        "minadv1e8": {"min_adv": 1e8},
        "rev2": {"reversal_1d_weight": 2.0},
        "rev4": {"reversal_1d_weight": 4.0},
        "shock035": {"event_shock_threshold": 0.035},
        "shock04": {"event_shock_threshold": 0.04},
        "drift05": {"rebalance_min_turnover": 0.05},
        "drift10": {"rebalance_min_turnover": 0.10},
        "drift20": {"rebalance_min_turnover": 0.20},
        "shock3_drift10": {"event_shock_threshold": 0.03, "rebalance_min_turnover": 0.10},
        "shock3_drift20": {"event_shock_threshold": 0.03, "rebalance_min_turnover": 0.20},
        "z25": {"event_shock_threshold": 0.0, "event_shock_zscore": 2.5},
        "z30": {"event_shock_threshold": 0.0, "event_shock_zscore": 3.0},
        "z35": {"event_shock_threshold": 0.0, "event_shock_zscore": 3.5},
        "z30_shock3": {"event_shock_threshold": 0.03, "event_shock_zscore": 3.0},
        "trail10": {"trailing_stop_pct": 0.10, "take_profit_pct": 0.0},
        "trail12": {"trailing_stop_pct": 0.12, "take_profit_pct": 0.0},
        "trail15": {"trailing_stop_pct": 0.15, "take_profit_pct": 0.0},
        "trail10_tp12": {"trailing_stop_pct": 0.10, "take_profit_pct": 0.12},
        "trail12_z30": {"trailing_stop_pct": 0.12, "take_profit_pct": 0.0, "event_shock_zscore": 3.0, "event_shock_threshold": 0.03},
        "z3s3_tp10": {"event_shock_threshold": 0.03, "event_shock_zscore": 3.0, "take_profit_pct": 0.10},
        "z3s3_tp15": {"event_shock_threshold": 0.03, "event_shock_zscore": 3.0, "take_profit_pct": 0.15},
        "z3s3_sl6": {"event_shock_threshold": 0.03, "event_shock_zscore": 3.0, "stop_loss_pct": 0.06},
        "z3s3_sl9": {"event_shock_threshold": 0.03, "event_shock_zscore": 3.0, "stop_loss_pct": 0.09},
        "z3s3_expo50": {"event_shock_threshold": 0.03, "event_shock_zscore": 3.0, "event_shock_exposure": 0.50},
        "z3s3_expo30": {"event_shock_threshold": 0.03, "event_shock_zscore": 3.0, "event_shock_exposure": 0.30},
        "z3s3_euph20": {"event_shock_threshold": 0.03, "event_shock_zscore": 3.0, "euphoria_threshold": 0.20},
        "z3s3_euph25": {"event_shock_threshold": 0.03, "event_shock_zscore": 3.0, "euphoria_threshold": 0.25},
        "z3s3_topn4": {"event_shock_threshold": 0.03, "event_shock_zscore": 3.0, "top_n": 4},
        "z3s3_topn6": {"event_shock_threshold": 0.03, "event_shock_zscore": 3.0, "top_n": 6},
        "z3s3_nolatch": {"event_shock_threshold": 0.03, "event_shock_zscore": 3.0, "event_shock_latch": False},
        "z3s9": {"event_shock_threshold": 0.03, "event_shock_zscore": 3.0, "stop_loss_pct": 0.09},
        "z3s8": {"event_shock_threshold": 0.03, "event_shock_zscore": 3.0, "stop_loss_pct": 0.08},
        "z3s10": {"event_shock_threshold": 0.03, "event_shock_zscore": 3.0, "stop_loss_pct": 0.10},
        "z3s9_tp10": {"event_shock_threshold": 0.03, "event_shock_zscore": 3.0, "stop_loss_pct": 0.09, "take_profit_pct": 0.10},
        "z3s9_tp15": {"event_shock_threshold": 0.03, "event_shock_zscore": 3.0, "stop_loss_pct": 0.09, "take_profit_pct": 0.15},
        "z3s9_euph20": {"event_shock_threshold": 0.03, "event_shock_zscore": 3.0, "stop_loss_pct": 0.09, "euphoria_threshold": 0.20},
        "z3s9_nolatch": {"event_shock_threshold": 0.03, "event_shock_zscore": 3.0, "stop_loss_pct": 0.09, "event_shock_latch": False},
        "z3s9_expo50": {"event_shock_threshold": 0.03, "event_shock_zscore": 3.0, "stop_loss_pct": 0.09, "event_shock_exposure": 0.50},
        "z3s9_rev": {"event_shock_threshold": 0.03, "event_shock_zscore": 3.0, "stop_loss_pct": 0.09, "reversal_window": 2},
        "s9_only": {"stop_loss_pct": 0.09},
        "z3s8": {"event_shock_threshold": 0.03, "event_shock_zscore": 3.0, "stop_loss_pct": 0.08},
        "z3s8_gate1": {"event_shock_threshold": 0.03, "event_shock_zscore": 3.0, "stop_loss_pct": 0.08, "min_top_momentum_gate": 0.01},
        "z3s8_gate2": {"event_shock_threshold": 0.03, "event_shock_zscore": 3.0, "stop_loss_pct": 0.08, "min_top_momentum_gate": 0.02},
        "z3s8_persist6": {"event_shock_threshold": 0.03, "event_shock_zscore": 3.0, "stop_loss_pct": 0.08, "persist_rank_floor": 6},
        "z3s8_persist10": {"event_shock_threshold": 0.03, "event_shock_zscore": 3.0, "stop_loss_pct": 0.08, "persist_rank_floor": 10},
        "z3s8_vol35": {"event_shock_threshold": 0.03, "event_shock_zscore": 3.0, "stop_loss_pct": 0.08, "max_annual_vol": 0.35},
        "z3s8_vol50": {"event_shock_threshold": 0.03, "event_shock_zscore": 3.0, "stop_loss_pct": 0.08, "max_annual_vol": 0.50},
        "z3s8_recm3": {"event_shock_threshold": 0.03, "event_shock_zscore": 3.0, "stop_loss_pct": 0.08, "event_shock_recovery_ma": 3},
        "z3s8_recm8": {"event_shock_threshold": 0.03, "event_shock_zscore": 3.0, "stop_loss_pct": 0.08, "event_shock_recovery_ma": 8},
        "z3s8_floor20": {"event_shock_threshold": 0.03, "event_shock_zscore": 3.0, "stop_loss_pct": 0.08, "ml_bear_floor": 0.20},
        "z3s8_floor40": {"event_shock_threshold": 0.03, "event_shock_zscore": 3.0, "stop_loss_pct": 0.08, "ml_bear_floor": 0.40},
        "z3s8_conf120": {"event_shock_threshold": 0.03, "event_shock_zscore": 3.0, "stop_loss_pct": 0.08, "regime_confirmation_ma": 120},
        "z3s8_conf0": {"event_shock_threshold": 0.03, "event_shock_zscore": 3.0, "stop_loss_pct": 0.08, "regime_confirmation_ma": 0},
        "z3s8_voltarget15": {"event_shock_threshold": 0.03, "event_shock_zscore": 3.0, "stop_loss_pct": 0.08, "vol_target": 0.15},
        "z3s8_adv3e7": {"event_shock_threshold": 0.03, "event_shock_zscore": 3.0, "stop_loss_pct": 0.08, "min_adv": 3e7},
        "z3s8_rsi10": {"event_shock_threshold": 0.03, "event_shock_zscore": 3.0, "stop_loss_pct": 0.08, "extra_factor_weights": {"rsi14": 0.10}},
        "z3s8_idll10": {"event_shock_threshold": 0.03, "event_shock_zscore": 3.0, "stop_loss_pct": 0.08, "extra_factor_weights": {"idll20": 0.10}},
        "z3s8_rsi_idll": {"event_shock_threshold": 0.03, "event_shock_zscore": 3.0, "stop_loss_pct": 0.08, "extra_factor_weights": {"rsi14": 0.10, "idll20": 0.10}},
        "z3s8_maxret10": {"event_shock_threshold": 0.03, "event_shock_zscore": 3.0, "stop_loss_pct": 0.08, "max_ret_weight": 0.10},
        "z3s8_maxret20": {"event_shock_threshold": 0.03, "event_shock_zscore": 3.0, "stop_loss_pct": 0.08, "max_ret_weight": 0.20},
        "z3s8_illiq10": {"event_shock_threshold": 0.03, "event_shock_zscore": 3.0, "stop_loss_pct": 0.08, "illiquidity_weight": 0.10},
        "z3s8_breadth40": {"event_shock_threshold": 0.03, "event_shock_zscore": 3.0, "stop_loss_pct": 0.08, "min_breadth_for_buys": 0.40},
        "z3s8_breadth50": {"event_shock_threshold": 0.03, "event_shock_zscore": 3.0, "stop_loss_pct": 0.08, "min_breadth_for_buys": 0.50},
        "z3s8_breadth60": {"event_shock_threshold": 0.03, "event_shock_zscore": 3.0, "stop_loss_pct": 0.08, "min_breadth_for_buys": 0.60},
        "nb_thr25": {"event_shock_threshold": 0.025, "event_shock_zscore": 3.0, "stop_loss_pct": 0.08},
        "nb_thr35": {"event_shock_threshold": 0.035, "event_shock_zscore": 3.0, "stop_loss_pct": 0.08},
        "nb_z25": {"event_shock_threshold": 0.03, "event_shock_zscore": 2.5, "stop_loss_pct": 0.08},
        "nb_z35": {"event_shock_threshold": 0.03, "event_shock_zscore": 3.5, "stop_loss_pct": 0.08},
        "nb_sl7": {"event_shock_threshold": 0.03, "event_shock_zscore": 3.0, "stop_loss_pct": 0.07},
        "nb_sl9": {"event_shock_threshold": 0.03, "event_shock_zscore": 3.0, "stop_loss_pct": 0.09},
        "robust_cap30": {"event_shock_threshold": 0.03, "event_shock_zscore": 3.0, "stop_loss_pct": 0.08, "capital_base": 300_000.0},
        "robust_cap50": {"event_shock_threshold": 0.03, "event_shock_zscore": 3.0, "stop_loss_pct": 0.08, "capital_base": 500_000.0},
        "robust_dynstops": {"event_shock_threshold": 0.03, "event_shock_zscore": 3.0, "stop_loss_pct": 0.08, "dynamic_stops": True},
        "robust_dftilt": {"event_shock_threshold": 0.03, "event_shock_zscore": 3.0, "stop_loss_pct": 0.08, "defensive_tilt": True},
        "robust_ma60": {"event_shock_threshold": 0.03, "event_shock_zscore": 3.0, "stop_loss_pct": 0.08, "regime_ma": 60},
        "robust_ma100": {"event_shock_threshold": 0.03, "event_shock_zscore": 3.0, "stop_loss_pct": 0.08, "regime_ma": 100},
        "robust_tf120": {"event_shock_threshold": 0.03, "event_shock_zscore": 3.0, "stop_loss_pct": 0.08, "trend_filter_long": 120},
        "robust_bearnoloss": {"event_shock_threshold": 0.03, "event_shock_zscore": 3.0, "stop_loss_pct": 0.08, "bear_no_loss": True},
    }
    keys = [k.strip() for k in args.configs.split(",") if k.strip()]
    results: dict = {}
    for key in keys:
        if key not in CONFIGS:
            print(f"unknown config {key}", file=sys.stderr)
            continue
        params = default_params()
        params.start_date = args.start
        params.research_num_trials = args.trials
        for k, v in CONFIGS[key].items():
            setattr(params, k, v)
        res = weekly_rotation_backtest(frames, params, regime_detector_kwargs={"bull_threshold": 0.55})
        s = res["summary"]
        gate = res["research_evidence_gate"]
        results[key] = {
            "annual_return": s["annual_return"],
            "sharpe": s["sharpe"],
            "calmar": s["calmar"],
            "max_drawdown": s["max_drawdown"],
            "recovery_3y": s.get("max_drawdown_recovery_days_3y"),
            "recovery_full": s.get("max_drawdown_recovery_days"),
            "quarterly_win_index": s.get("quarterly_win_vs_index"),
            "quarterly_win_ew": s.get("quarterly_win_vs_benchmark"),
            "monthly_win_index": s.get("monthly_win_vs_index"),
            "turnover": s.get("avg_rebalance_turnover"),
            "total_cost": s.get("total_cost_fraction"),
            "avg_exposure": s.get("average_exposure"),
            "final_equity": s["final_equity"],
            "observations": s["observations"],
            "oos": oos_metrics(res),
            "gate_status": gate["status"],
            "gate_reasons": gate["reasons"],
            "gate_checks": gate["checks"],
        }
        print(f"[{key}] ann={results[key]['annual_return']:.2%} sharpe={results[key]['sharpe']:.2f} "
              f"calmar={results[key]['calmar']:.2f} mdd={results[key]['max_drawdown']:.2%} "
              f"oos_ann={results[key]['oos']['oos_ann']:.2%} oos_sharpe={results[key]['oos']['oos_sharpe']:.2f} "
              f"qwin_idx={results[key]['quarterly_win_index']:.1%} qwin_ew={results[key]['quarterly_win_ew']:.1%} "
              f"cost={results[key]['total_cost']:.2%} turnover={results[key]['turnover']:.1%} "
              f"gate={results[key]['gate_status']}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
