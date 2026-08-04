"""Sequential portfolio backtest of Quant-4 analysis advice on real OHLCV files."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from Main.portfolio_analytics import recommendation
from Main.data_quality import validate_ohlcv

DEFAULT_EXCLUDED_PREFIXES = ("200", "300", "301", "4", "8", "92", "688", "689", "900")
FORBIDDEN_PROVIDERS = {"synthetic", "simulated", "mock", "offline_debug"}


@dataclass(frozen=True)
class UniverseDecision:
    symbol: str
    included: bool
    reason: str
    provider: str | None = None
    rows: int = 0


def symbol_purchase_eligibility(symbol: str, excluded_prefixes=DEFAULT_EXCLUDED_PREFIXES) -> tuple[bool, str]:
    code, dot, exchange = str(symbol).upper().partition(".")
    if not dot or exchange not in {"SH", "SZ"} or len(code) != 6 or not code.isdigit():
        return False, "NOT_STANDARD_A_SHARE"
    if any(code.startswith(prefix) for prefix in excluded_prefixes):
        return False, "EXCLUDED_ACCOUNT_PERMISSION_PREFIX"
    if not code.startswith(("0", "6")):
        return False, "UNSUPPORTED_PURCHASE_CODE"
    return True, "ELIGIBLE_CODE"


def _load_evidence(cache_dir: Path, symbol: str) -> dict | None:
    candidates = sorted((cache_dir / "evidence").glob(f"{symbol}_*.json"))
    valid = []
    for path in candidates:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            provider = str(payload.get("provider", "")).lower()
            if payload.get("valid") is True and provider and provider not in FORBIDDEN_PROVIDERS:
                valid.append(payload)
        except (OSError, json.JSONDecodeError):
            continue
    return max(valid, key=lambda item: int(item.get("rows", 0))) if valid else None


def discover_real_universe(cache_dir: Path, min_rows: int = 504, excluded_prefixes=DEFAULT_EXCLUDED_PREFIXES) -> tuple[dict[str, pd.DataFrame], list[UniverseDecision]]:
    cache_dir = Path(cache_dir)
    frames: dict[str, pd.DataFrame] = {}
    decisions: list[UniverseDecision] = []
    for path in sorted(cache_dir.glob("*_history.parquet")):
        symbol = path.name.removesuffix("_history.parquet").upper()
        eligible, reason = symbol_purchase_eligibility(symbol, excluded_prefixes)
        if not eligible:
            decisions.append(UniverseDecision(symbol, False, reason))
            continue
        evidence = _load_evidence(cache_dir, symbol)
        if evidence is None:
            decisions.append(UniverseDecision(symbol, False, "MISSING_VALID_REAL_SOURCE_EVIDENCE"))
            continue
        try:
            frame = pd.read_parquet(path)
        except Exception:
            decisions.append(UniverseDecision(symbol, False, "UNREADABLE_PARQUET", str(evidence.get("provider"))))
            continue
        validation = validate_ohlcv(frame, symbol)
        if not validation["valid"]:
            decisions.append(UniverseDecision(symbol, False, "DATA_QUALITY_FAILED", str(evidence.get("provider")), len(frame)))
            continue
        if int(evidence.get("rows", -1)) != len(frame) or evidence.get("sha256") != validation["sha256"]:
            decisions.append(UniverseDecision(symbol, False, "EVIDENCE_HASH_MISMATCH", str(evidence.get("provider")), len(frame)))
            continue
        required = {"date", "open", "high", "low", "close", "volume"}
        if not required.issubset(frame.columns):
            decisions.append(UniverseDecision(symbol, False, "MISSING_OHLCV_COLUMNS", str(evidence.get("provider")), len(frame)))
            continue
        frame = frame.copy()
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.tz_localize(None)
        frame = frame.dropna(subset=["date"]).sort_values("date").drop_duplicates("date", keep="last").set_index("date")
        numeric = ["open", "high", "low", "close", "volume"]
        frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce")
        frame = frame.dropna(subset=numeric)
        valid_prices = (frame[["open", "high", "low", "close"]] > 0).all(axis=1)
        frame = frame.loc[valid_prices]
        if len(frame) < min_rows:
            decisions.append(UniverseDecision(symbol, False, "INSUFFICIENT_REAL_HISTORY", str(evidence.get("provider")), len(frame)))
            continue
        frames[symbol] = frame
        decisions.append(UniverseDecision(symbol, True, "INCLUDED_REAL_HISTORY", str(evidence.get("provider")), len(frame)))
    return frames, decisions


def _summary(returns: pd.DataFrame, fee_rate: float, signals: pd.DataFrame, decisions: list[UniverseDecision]) -> dict:
    r = returns["strategy_return"]
    b = returns["benchmark_return"]
    equity = (1 + r).cumprod(); benchmark = (1 + b).cumprod(); n = len(r)
    ann = float(equity.iloc[-1] ** (252 / n) - 1)
    ann_b = float(benchmark.iloc[-1] ** (252 / n) - 1)
    vol = float(r.std(ddof=1) * np.sqrt(252))
    return {
        "analysis_only": True,
        "observations": n,
        "start": str(returns.index.min().date()), "end": str(returns.index.max().date()),
        "eligible_assets": int(sum(d.included for d in decisions)), "discovered_assets": len(decisions),
        "annual_return": ann, "benchmark_annual_return": ann_b,
        "annualized_excess_return": float((equity.iloc[-1] / benchmark.iloc[-1]) ** (252 / n) - 1),
        "annual_volatility": vol, "sharpe": float(r.mean() / r.std(ddof=1) * np.sqrt(252)) if vol else 0.0,
        "max_drawdown": float((equity / equity.cummax() - 1).min()),
        "final_equity": float(equity.iloc[-1]), "benchmark_final_equity": float(benchmark.iloc[-1]),
        "average_exposure": float(returns["gross_exposure"].mean()),
        "annualized_turnover": float(returns["turnover"].sum() * 252 / n),
        "total_cost_fraction": float(returns["cost"].sum()), "fee_rate": float(fee_rate),
        "rebalance_signals": int(signals["signal_date"].nunique()) if not signals.empty else 0,
        "qualified_recommendations": int(signals["qualified"].sum()) if not signals.empty else 0,
        "rejected_limit_or_halt_orders": int((signals["execution_status"] != "EXECUTED").sum()) if not signals.empty else 0,
        "execution_rule": "close[t] advice -> open[t+1] execution; open-to-open portfolio returns",
        "data_policy": "real source evidence required; no synthetic/mock/offline-debug provider accepted",
    }


def run_advice_portfolio_backtest(frames: dict[str, pd.DataFrame], decisions: list[UniverseDecision], years: int = 8, lookback: int = 252, rebalance_every: int = 21, fee_rate: float = 0.001, min_assets: int = 2) -> dict:
    if len(frames) < min_assets:
        raise ValueError(f"at least {min_assets} eligible real assets are required; found {len(frames)}")
    if years < 1 or lookback < 60 or rebalance_every < 2 or fee_rate < 0:
        raise ValueError("invalid portfolio backtest controls")
    symbols = sorted(frames)
    common = frames[symbols[0]].index
    for symbol in symbols[1:]: common = common.intersection(frames[symbol].index)
    common = common.sort_values()
    end = common.max(); requested_start = end - pd.DateOffset(years=years)
    eligible_dates = common[common >= requested_start]
    if len(eligible_dates) < lookback + 2:
        raise ValueError("insufficient common real history for requested portfolio backtest")
    start_pos = common.get_loc(eligible_dates[0])
    signal_positions = list(range(max(lookback, start_pos), len(common) - 1, rebalance_every))
    target_by_execution: dict[pd.Timestamp, dict[str, float]] = {}
    signal_rows = []
    for pos in signal_positions:
        signal_date, execution_date = common[pos], common[pos + 1]
        target = {}
        for symbol in symbols:
            history = frames[symbol].loc[:signal_date].tail(max(lookback, 252)).reset_index()
            rec = recommendation(history, asset_type="stock")
            weight = float(rec["suggested_weight"]) if rec["qualified"] else 0.0
            previous_close = float(frames[symbol].at[signal_date, "close"])
            execution_open = float(frames[symbol].at[execution_date, "open"])
            execution_volume = float(frames[symbol].at[execution_date, "volume"])
            gap = execution_open / previous_close - 1
            status = "EXECUTED"
            if execution_volume <= 0:
                weight, status = 0.0, "REJECTED_HALTED_OR_ZERO_VOLUME"
            elif weight > 0 and gap >= 0.098:
                weight, status = 0.0, "REJECTED_LIMIT_UP_BUY"
            target[symbol] = weight
            signal_rows.append({"signal_date": signal_date, "execution_date": execution_date, "symbol": symbol, "qualified": bool(rec["qualified"]), "suggested_weight": float(rec["suggested_weight"]), "executed_target_weight": weight, "execution_gap": gap, "execution_status": status, "dominant_selection_method": rec["dominant_selection_method"]})
        if sum(target.values()) > 1.0:
            scale = 1.0 / sum(target.values()); target = {k: v * scale for k, v in target.items()}
        target_by_execution[execution_date] = target

    simulation_dates = common[(common >= eligible_dates[0]) & (common < common[-1])]
    weights = {symbol: 0.0 for symbol in symbols}; rows = []
    for date in simulation_dates:
        next_date = common[common.get_loc(date) + 1]
        turnover = 0.0
        if date in target_by_execution:
            desired = target_by_execution[date].copy()
            for symbol in symbols:
                previous_close = float(frames[symbol].at[common[common.get_loc(date) - 1], "close"])
                execution_open = float(frames[symbol].at[date, "open"])
                execution_volume = float(frames[symbol].at[date, "volume"])
                gap = execution_open / previous_close - 1
                if desired[symbol] < weights[symbol] and (execution_volume <= 0 or gap <= -0.098):
                    desired[symbol] = weights[symbol]
                turnover += abs(desired[symbol] - weights[symbol])
            weights = desired
        asset_returns = {symbol: float(frames[symbol].at[next_date, "open"] / frames[symbol].at[date, "open"] - 1) for symbol in symbols}
        cost = turnover * fee_rate
        gross_return = sum(weights[symbol] * asset_returns[symbol] for symbol in symbols)
        strategy_return = gross_return - cost
        benchmark_return = float(np.mean(list(asset_returns.values())))
        rows.append({"date": date, "strategy_return": strategy_return, "benchmark_return": benchmark_return, "gross_exposure": sum(weights.values()), "cash_weight": 1.0 - sum(weights.values()), "turnover": turnover, "cost": cost, **{f"weight_{symbol}": weights[symbol] for symbol in symbols}})
        # Holdings drift with their open-to-open returns until the next rebalance;
        # this avoids silently assuming cost-free daily rebalancing.
        ending_equity = 1.0 + strategy_return
        if ending_equity <= 0:
            raise ValueError("portfolio equity became non-positive")
        weights = {symbol: weights[symbol] * (1.0 + asset_returns[symbol]) / ending_equity for symbol in symbols}
    returns = pd.DataFrame(rows).set_index("date")
    signals = pd.DataFrame(signal_rows)
    return {"summary": _summary(returns, fee_rate, signals, decisions), "returns": returns, "signals": signals, "universe_audit": pd.DataFrame([d.__dict__ for d in decisions])}


def write_portfolio_backtest_report(result: dict, output_dir: Path) -> dict[str, Path]:
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    paths = {"returns": output_dir / "advice_portfolio_returns.csv", "signals": output_dir / "advice_portfolio_signals.csv", "universe": output_dir / "advice_portfolio_universe.csv", "summary": output_dir / "advice_portfolio_summary.json"}
    result["returns"].to_csv(paths["returns"], encoding="utf-8-sig")
    result["signals"].to_csv(paths["signals"], index=False, encoding="utf-8-sig")
    result["universe_audit"].to_csv(paths["universe"], index=False, encoding="utf-8-sig")
    hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items() if name != "summary"}
    payload = {**result["summary"], "artifact_sha256": hashes}
    body = json.dumps(payload, ensure_ascii=False, indent=2)
    payload["summary_sha256"] = hashlib.sha256(body.encode("utf-8")).hexdigest()
    paths["summary"].write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return paths
