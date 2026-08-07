"""Sequential portfolio backtest of Quant-4 analysis advice on real OHLCV files."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from Main.portfolio_analytics import recommendation
from Main.data_quality import validate_ohlcv
from Main.datasource_manager import FreeDataSourceManager
from Main.universe_rules import DEFAULT_EXCLUDED_PREFIXES, symbol_purchase_eligibility, asset_type_for_symbol

FORBIDDEN_PROVIDERS = {"synthetic", "simulated", "mock", "offline_debug"}
_THREAD_LOCAL = threading.local()


@dataclass(frozen=True)
class UniverseDecision:
    symbol: str
    included: bool
    reason: str
    provider: str | None = None
    rows: int = 0


def dynamic_risk_exposure(
    qualified_ratio: float,
    trend_ratio: float,
    annual_volatility: float,
    portfolio_drawdown: float,
    minimum: float = 0.15,
    maximum: float = 0.90,
) -> float:
    """Bound portfolio exposure using only information observable at the signal close."""
    if not 0 <= minimum <= maximum <= 1:
        raise ValueError("exposure bounds must satisfy 0 <= minimum <= maximum <= 1")
    breadth = float(np.clip(qualified_ratio, 0, 1))
    trend = float(np.clip(trend_ratio, 0, 1))
    vol_penalty = float(np.clip((annual_volatility - 0.15) / 0.35, 0, 1))
    drawdown_penalty = float(np.clip(abs(min(portfolio_drawdown, 0.0)) / 0.15, 0, 1))
    raw = minimum + 0.45 * breadth + 0.25 * trend - 0.20 * vol_penalty - 0.30 * drawdown_penalty
    return float(np.clip(raw, minimum, maximum))


def eligible_as_of(frames: dict[str, pd.DataFrame], signal_date: pd.Timestamp, lookback: int) -> list[str]:
    """Reconstruct the purchasable research universe without reading future rows."""
    eligible = []
    for symbol, frame in frames.items():
        history = frame.loc[:signal_date]
        if signal_date not in frame.index or len(history) < lookback:
            continue
        recent = history.tail(20)
        if float(history.at[signal_date, "volume"]) <= 0 or float(recent["volume"].median()) <= 0:
            continue
        eligible.append(symbol)
    return sorted(eligible)


def _execution_snapshot(frame: pd.DataFrame, date: pd.Timestamp) -> tuple[float, float, float]:
    prior = frame.loc[frame.index < date]
    if prior.empty:
        return np.nan, np.nan, 0.0
    previous_close = float(prior["close"].iloc[-1])
    if date not in frame.index:
        return previous_close, previous_close, 0.0
    return previous_close, float(frame.at[date, "open"]), float(frame.at[date, "volume"])


def _open_to_open_return(frame: pd.DataFrame, date: pd.Timestamp, next_date: pd.Timestamp) -> float:
    if date not in frame.index or next_date not in frame.index:
        return 0.0
    return float(frame.at[next_date, "open"] / frame.at[date, "open"] - 1)


def refresh_full_market_cache(
    cache_dir: Path, start_date: str, end_date: str, workers: int = 4,
    excluded_prefixes=DEFAULT_EXCLUDED_PREFIXES,
    minimum_market_coverage: int = 500, minimum_stock_coverage: int = 1000,
    source_timeout_seconds: float = 30.0, source_cooldown_seconds: float = 60.0,
) -> dict:
    """Query the broad real A-share universe and materialize evidence-backed daily histories."""
    cache_dir = Path(cache_dir)
    seed = FreeDataSourceManager(cache_dir=cache_dir, offline_debug=False, source_timeout_seconds=source_timeout_seconds, source_cooldown_seconds=source_cooldown_seconds)
    universe = seed.fetch_full_market_list(include_delisted=True)
    eligible = [s for s in universe if symbol_purchase_eligibility(s, excluded_prefixes)[0]]
    stock_count = sum(asset_type_for_symbol(s) == "stock" for s in eligible)
    if len(eligible) < minimum_market_coverage or stock_count < minimum_stock_coverage:
        raise RuntimeError(f"full-market source coverage is insufficient before history download: securities={len(eligible)}/{minimum_market_coverage}, stocks={stock_count}/{minimum_stock_coverage}")

    def fetch(symbol: str) -> tuple[str, bool]:
        manager = getattr(_THREAD_LOCAL, "manager", None)
        if manager is None:
            manager = FreeDataSourceManager(cache_dir=cache_dir, offline_debug=False, source_timeout_seconds=source_timeout_seconds, source_cooldown_seconds=source_cooldown_seconds)
            _THREAD_LOCAL.manager = manager
        frame = manager.fetch_historical(symbol, start_date, end_date)
        return symbol, frame is not None and not frame.empty

    succeeded = 0
    with ThreadPoolExecutor(max_workers=max(1, min(int(workers), 16))) as pool:
        futures = [pool.submit(fetch, symbol) for symbol in eligible]
        for future in as_completed(futures):
            try:
                _, ok = future.result()
                succeeded += int(ok)
            except Exception:
                continue
    return {"queried_symbols": len(universe), "permission_eligible_symbols": len(eligible), "eligible_stocks": sum(asset_type_for_symbol(s) == "stock" for s in eligible), "eligible_etfs": sum(asset_type_for_symbol(s) == "ETF" for s in eligible), "histories_available": succeeded, "start_date": start_date, "end_date": end_date}


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
        "take_profit_exits": int((signals["exit_reason"] == "TAKE_PROFIT").sum()) if not signals.empty else 0,
        "stop_loss_exits": int((signals["exit_reason"] == "STOP_LOSS").sum()) if not signals.empty else 0,
        "time_harvest_exits": int((signals["exit_reason"] == "TIME_HARVEST").sum()) if not signals.empty else 0,
        "average_historical_universe": float(returns["historically_eligible_assets"].mean()),
        "minimum_historical_universe": int(returns["historically_eligible_assets"].min()),
        "full_market_ready": int(sum(d.included for d in decisions)) >= 500 and int(sum(d.included and asset_type_for_symbol(d.symbol) == "stock" for d in decisions)) >= 1000,
        "eligible_stocks": int(sum(d.included and asset_type_for_symbol(d.symbol) == "stock" for d in decisions)),
        "eligible_etfs": int(sum(d.included and asset_type_for_symbol(d.symbol) == "ETF" for d in decisions)),
        "universe_scope": "broad market cache with daily as-of eligibility; not a fixed ticker list",
        "execution_rule": "close[t] advice -> open[t+1] execution; open-to-open portfolio returns",
        "data_policy": "real source evidence required; no synthetic/mock/offline-debug provider accepted",
    }


def run_advice_portfolio_backtest(
    frames: dict[str, pd.DataFrame], decisions: list[UniverseDecision], years: int = 8,
    lookback: int = 252, rebalance_every: int = 1, fee_rate: float = 0.001,
    min_assets: int = 2, max_positions: int = 5, min_exposure: float = 0.15,
    max_exposure: float = 0.90, max_holding_days: int = 20, harvest_cooldown_days: int = 3,
    min_trade_weight: float = 0.02, max_daily_turnover: float = 0.25,
) -> dict:
    if len(frames) < min_assets:
        raise ValueError(f"at least {min_assets} eligible real assets are required; found {len(frames)}")
    if years < 1 or lookback < 60 or rebalance_every < 1 or fee_rate < 0 or max_positions < 1 or max_holding_days < 2:
        raise ValueError("invalid portfolio backtest controls")
    if not 0 <= min_exposure <= max_exposure <= 1 or not 0 <= min_trade_weight <= 1 or not 0 < max_daily_turnover <= 1:
        raise ValueError("invalid exposure bounds")
    symbols = sorted(frames)
    common = frames[symbols[0]].index
    for symbol in symbols[1:]: common = common.union(frames[symbol].index)
    common = common.unique().sort_values()
    end = common.max(); requested_start = end - pd.DateOffset(years=years)
    eligible_dates = common[common >= requested_start]
    if len(eligible_dates) < lookback + 2:
        raise ValueError("insufficient common real history for requested portfolio backtest")
    start_pos = common.get_loc(eligible_dates[0])
    simulation_dates = common[(common >= eligible_dates[0]) & (common < common[-1])]
    weights = {symbol: 0.0 for symbol in symbols}
    entry_prices = {symbol: 0.0 for symbol in symbols}
    holding_days = {symbol: 0 for symbol in symbols}
    cooldown = {symbol: 0 for symbol in symbols}
    pending: dict | None = None
    rows, signal_rows, universe_rows, equity_history = [], [], [], [1.0]
    for day_number, date in enumerate(simulation_dates):
        next_date = common[common.get_loc(date) + 1]
        turnover = 0.0
        execution_cost = 0.0
        if pending is not None and pending["execution_date"] == date:
            desired = pending["target"].copy()
            for symbol in symbols:
                meta = pending["metadata"].get(symbol, {})
                if not meta.get("exit_reason") and abs(desired[symbol] - weights[symbol]) < min_trade_weight:
                    desired[symbol] = weights[symbol]
            requested_turnover = sum(abs(desired[symbol] - weights[symbol]) for symbol in symbols)
            if requested_turnover > max_daily_turnover:
                scale = max_daily_turnover / requested_turnover
                desired = {symbol: weights[symbol] + (desired[symbol] - weights[symbol]) * scale for symbol in symbols}
            for symbol in symbols:
                previous_close, execution_open, execution_volume = _execution_snapshot(frames[symbol], date)
                gap = execution_open / previous_close - 1
                old_weight, requested = weights[symbol], desired[symbol]
                status = "EXECUTED"
                if execution_volume <= 0:
                    desired[symbol], status = old_weight, "REJECTED_HALTED_OR_ZERO_VOLUME"
                elif requested > old_weight and gap >= 0.098:
                    desired[symbol], status = old_weight, "REJECTED_LIMIT_UP_BUY"
                elif requested < old_weight and gap <= -0.098:
                    desired[symbol] = weights[symbol]
                    status = "REJECTED_LIMIT_DOWN_SELL"
                delta = desired[symbol] - old_weight
                turnover += abs(delta)
                meta = pending["metadata"].get(symbol, {})
                if delta > 1e-12:
                    entry_prices[symbol] = execution_open if old_weight <= 1e-12 else (entry_prices[symbol] * old_weight + execution_open * delta) / (old_weight + delta)
                    holding_days[symbol] = 0
                elif desired[symbol] <= 1e-12 and old_weight > 1e-12:
                    if meta.get("exit_reason") in {"TAKE_PROFIT", "STOP_LOSS", "TIME_HARVEST"}:
                        cooldown[symbol] = harvest_cooldown_days
                    entry_prices[symbol], holding_days[symbol] = 0.0, 0
                signal_rows.append({"signal_date": pending["signal_date"], "execution_date": date, "symbol": symbol, "historically_eligible": bool(meta.get("eligible", False)), "qualified": bool(meta.get("qualified", False)), "score": meta.get("score"), "dynamic_exposure": pending["dynamic_exposure"], "suggested_weight": meta.get("suggested_weight", 0.0), "executed_target_weight": desired[symbol], "execution_gap": gap, "execution_status": status, "exit_reason": meta.get("exit_reason", ""), "dominant_selection_method": meta.get("dominant_selection_method", "")})
            weights = desired
            pending = None
        as_of_universe = eligible_as_of(frames, date, lookback)
        universe_rows.extend({"signal_date": date, "symbol": symbol, "eligible_as_of_close": symbol in as_of_universe} for symbol in symbols)
        asset_returns = {symbol: _open_to_open_return(frames[symbol], date, next_date) for symbol in symbols}
        cost = turnover * fee_rate
        execution_cost += cost
        gross_return = sum(weights[symbol] * asset_returns[symbol] for symbol in symbols)
        strategy_return = gross_return - execution_cost
        benchmark_return = float(np.mean([asset_returns[symbol] for symbol in as_of_universe])) if as_of_universe else 0.0
        rows.append({"date": date, "strategy_return": strategy_return, "benchmark_return": benchmark_return, "gross_exposure": sum(weights.values()), "cash_weight": 1.0 - sum(weights.values()), "historically_eligible_assets": len(as_of_universe), "turnover": turnover, "cost": execution_cost, **{f"weight_{symbol}": weights[symbol] for symbol in symbols}})
        # Holdings drift with their open-to-open returns until the next rebalance;
        # this avoids silently assuming cost-free daily rebalancing.
        ending_equity = 1.0 + strategy_return
        if ending_equity <= 0:
            raise ValueError("portfolio equity became non-positive")
        weights = {symbol: weights[symbol] * (1.0 + asset_returns[symbol]) / ending_equity for symbol in symbols}
        equity_history.append(equity_history[-1] * ending_equity)
        for symbol in symbols:
            if weights[symbol] > 1e-12:
                holding_days[symbol] += 1
            cooldown[symbol] = max(0, cooldown[symbol] - 1)

        if day_number % rebalance_every == 0:
            candidates, metadata, engine_qualified_count = [], {}, 0
            for symbol in as_of_universe:
                history = frames[symbol].loc[:date].tail(max(lookback, 252)).reset_index()
                rec = recommendation(history, asset_type=asset_type_for_symbol(symbol))
                close = float(frames[symbol].at[date, "close"])
                pnl = close / entry_prices[symbol] - 1 if entry_prices[symbol] > 0 else 0.0
                exit_reason = ""
                if weights[symbol] > 1e-12:
                    if pnl >= float(rec["take_profit_pct"]): exit_reason = "TAKE_PROFIT"
                    elif pnl <= -float(rec["stop_loss_pct"]): exit_reason = "STOP_LOSS"
                    elif holding_days[symbol] >= max_holding_days: exit_reason = "TIME_HARVEST"
                # Once opened, keep a name through ordinary signal noise until a
                # harvest/risk exit or a stronger name rotates it out.
                held = weights[symbol] > 1e-12
                tradable_candidate = bool((rec["qualified"] or held) and cooldown[symbol] == 0 and not exit_reason)
                engine_qualified_count += int(bool(rec["qualified"]))
                annual_vol = max(float(rec.get("annual_volatility", 0.0)), 0.05)
                rank_score = float(rec["score"] + rec["decision_chain"]["selection"]["nonlinear_score"] - 0.25 * annual_vol)
                metadata[symbol] = {"eligible": True, "qualified": bool(rec["qualified"]), "selectable": tradable_candidate, "score": rank_score, "suggested_weight": float(rec["suggested_weight"]), "exit_reason": exit_reason, "annual_volatility": annual_vol, "dominant_selection_method": rec["dominant_selection_method"]}
                if tradable_candidate:
                    candidates.append((symbol, rank_score, annual_vol, close > float(rec["ma60"])))
            selected = sorted(candidates, key=lambda item: item[1], reverse=True)[:max_positions]
            qualified_ratio = engine_qualified_count / max(len(as_of_universe), 1)
            trend_ratio = sum(item[3] for item in selected) / max(len(selected), 1)
            median_vol = float(np.median([item[2] for item in selected])) if selected else 1.0
            peak = max(equity_history); drawdown = equity_history[-1] / peak - 1
            exposure = dynamic_risk_exposure(qualified_ratio, trend_ratio, median_vol, drawdown, min_exposure, max_exposure) if selected else 0.0
            raw = {symbol: max(score, 0.05) / vol for symbol, score, vol, _ in selected}
            raw_total = sum(raw.values())
            target = {symbol: exposure * raw.get(symbol, 0.0) / raw_total if raw_total else 0.0 for symbol in symbols}
            pending = {"signal_date": date, "execution_date": next_date, "target": target, "metadata": metadata, "dynamic_exposure": exposure}
    returns = pd.DataFrame(rows).set_index("date")
    signals = pd.DataFrame(signal_rows)
    return {"summary": _summary(returns, fee_rate, signals, decisions), "returns": returns, "signals": signals, "universe_audit": pd.DataFrame([d.__dict__ for d in decisions]), "daily_universe": pd.DataFrame(universe_rows)}


def write_portfolio_backtest_report(result: dict, output_dir: Path) -> dict[str, Path]:
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    paths = {"returns": output_dir / "advice_portfolio_returns.csv", "signals": output_dir / "advice_portfolio_signals.csv", "universe": output_dir / "advice_portfolio_universe.csv", "daily_universe": output_dir / "advice_portfolio_daily_universe.csv", "summary": output_dir / "advice_portfolio_summary.json"}
    result["returns"].to_csv(paths["returns"], encoding="utf-8-sig")
    result["signals"].to_csv(paths["signals"], index=False, encoding="utf-8-sig")
    result["universe_audit"].to_csv(paths["universe"], index=False, encoding="utf-8-sig")
    result["daily_universe"].to_csv(paths["daily_universe"], index=False, encoding="utf-8-sig")
    hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items() if name != "summary"}
    payload = {**result["summary"], "artifact_sha256": hashes}
    body = json.dumps(payload, ensure_ascii=False, indent=2)
    payload["summary_sha256"] = hashlib.sha256(body.encode("utf-8")).hexdigest()
    paths["summary"].write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return paths
