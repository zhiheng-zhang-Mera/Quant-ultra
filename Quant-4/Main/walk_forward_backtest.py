"""Leakage-resistant walk-forward backtest with fold-local parameter adaptation."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from itertools import product
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BacktestParams:
    fast_window: int
    slow_window: int
    vol_window: int
    target_vol: float


def _prepare(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "open", "close"}
    if not required.issubset(frame.columns):
        raise ValueError(f"回测数据缺少列: {sorted(required-set(frame.columns))}")
    data = frame.copy()
    data["date"] = pd.to_datetime(data["date"], errors="raise")
    data = data.sort_values("date").drop_duplicates("date", keep="last").set_index("date")
    data[["open", "close"]] = data[["open", "close"]].apply(pd.to_numeric, errors="raise")
    if not data.index.is_monotonic_increasing or (data[["open", "close"]] <= 0).any().any():
        raise ValueError("回测日期或价格无效")
    return data


def _signals(data: pd.DataFrame, params: BacktestParams) -> pd.DataFrame:
    """Signals use close[t]; execution and realized return start at open[t+1]."""
    close = data["close"]
    fast = close.rolling(params.fast_window, min_periods=params.fast_window).mean()
    slow = close.rolling(params.slow_window, min_periods=params.slow_window).mean()
    historical_vol = close.pct_change().rolling(params.vol_window, min_periods=params.vol_window).std(ddof=1) * np.sqrt(252)
    raw = (fast > slow).astype(float)
    leverage = (params.target_vol / historical_vol.replace(0, np.nan)).clip(0, 1).fillna(0)
    decision_weight = raw * leverage
    result = pd.DataFrame(index=data.index)
    result["signal_time"] = data.index
    result["execution_time"] = pd.Series(data.index, index=data.index).shift(-1)
    result["weight"] = decision_weight
    # Position decided at close[t], entered at open[t+1], exited/rebalanced at open[t+2].
    result["asset_return"] = data["open"].shift(-2) / data["open"].shift(-1) - 1
    return result


def _score_train(data: pd.DataFrame, params: BacktestParams, fee_rate: float) -> float:
    stream = _signals(data, params).dropna(subset=["execution_time", "asset_return"])
    turnover = stream["weight"].diff().abs().fillna(stream["weight"].abs())
    returns = stream["weight"] * stream["asset_return"] - turnover * fee_rate
    if len(returns) < max(params.slow_window, 20) or returns.std(ddof=1) <= 0:
        return -np.inf
    sharpe = returns.mean() / returns.std(ddof=1) * np.sqrt(252)
    drawdown = (1 + returns).cumprod() / (1 + returns).cumprod().cummax() - 1
    return float(sharpe + 0.25 * drawdown.min())


def _grid(param_grid: dict) -> list[BacktestParams]:
    keys = ("fast_window", "slow_window", "vol_window", "target_vol")
    missing = [k for k in keys if k not in param_grid]
    if missing:
        raise ValueError(f"参数网格缺少: {missing}")
    candidates = [BacktestParams(*values) for values in product(*(param_grid[k] for k in keys))]
    valid = [p for p in candidates if 1 < p.fast_window < p.slow_window and p.vol_window > 1 and 0 < p.target_vol <= 1]
    if not valid:
        raise ValueError("参数网格没有有效组合")
    return valid


def walk_forward_backtest(
    frame: pd.DataFrame,
    param_grid: dict,
    train_size: int = 252,
    test_size: int = 63,
    embargo: int = 5,
    fee_rate: float = 0.001,
) -> dict:
    """Expanding-window selection; parameters are frozen in each out-of-sample fold."""
    data = _prepare(frame)
    candidates = _grid(param_grid)
    if train_size < max(p.slow_window for p in candidates) * 2 or test_size < 2 or embargo < 1:
        raise ValueError("训练窗、测试窗或 embargo 设置不满足无泄漏要求")
    fold_rows, return_parts, audit_rows = [], [], []
    test_start = train_size + embargo
    fold = 0
    while test_start < len(data) - 2:
        test_end = min(test_start + test_size, len(data) - 2)
        train_end = test_start - embargo
        train = data.iloc[:train_end].copy()
        scores = [(p, _score_train(train, p, fee_rate)) for p in candidates]
        best, best_score = max(scores, key=lambda item: item[1])
        if not np.isfinite(best_score):
            raise RuntimeError(f"折 {fold} 训练样本不足以自适应参数")
        full_stream = _signals(data.iloc[:test_end + 2], best)
        test_index = data.index[test_start:test_end]
        stream = full_stream.loc[test_index].dropna(subset=["execution_time", "asset_return"]).copy()
        turnover = stream["weight"].diff().abs().fillna(stream["weight"].abs())
        stream["strategy_return"] = stream["weight"] * stream["asset_return"] - turnover * fee_rate
        stream["fold"] = fold
        violation = (pd.to_datetime(stream["signal_time"]) >= pd.to_datetime(stream["execution_time"])).any()
        if violation or train.index.max() >= data.index[test_start]:
            raise RuntimeError("检测到未来数据泄漏或训练/测试时间重叠")
        return_parts.append(stream[["fold", "signal_time", "execution_time", "weight", "asset_return", "strategy_return"]])
        fold_rows.append({
            "fold": fold, "train_start": str(train.index.min()), "train_end": str(train.index.max()),
            "embargo_days": embargo, "test_start": str(data.index[test_start]),
            "test_end": str(data.index[test_end-1]), "train_score": best_score, **asdict(best),
        })
        audit_rows.append({"fold": fold, "train_before_test": True, "signal_before_execution": True, "parameters_frozen": True})
        fold += 1
        test_start = test_end
    if not return_parts:
        raise ValueError("数据长度不足以形成走步回测折")
    returns = pd.concat(return_parts).sort_index()
    folds = pd.DataFrame(fold_rows)
    audit = pd.DataFrame(audit_rows)
    equity = (1 + returns["strategy_return"]).cumprod()
    r = returns["strategy_return"]
    annual_return = float(equity.iloc[-1] ** (252 / len(r)) - 1)
    annual_vol = float(r.std(ddof=1) * np.sqrt(252))
    max_drawdown = float((equity / equity.cummax() - 1).min())
    summary = {
        "observations": len(r), "folds": fold, "annual_return": annual_return,
        "annual_volatility": annual_vol, "sharpe": (annual_return / annual_vol if annual_vol else 0.0),
        "max_drawdown": max_drawdown, "final_equity": float(equity.iloc[-1]),
        "lookahead_audit_passed": bool(audit[["train_before_test", "signal_before_execution", "parameters_frozen"]].all(axis=None)),
        "execution_rule": "close[t] signal -> open[t+1] execution -> open[t+2] return",
    }
    return {"summary": summary, "folds": folds, "returns": returns, "audit": audit}


def write_backtest_report(result: dict, report_dir: Path, name: str) -> tuple[Path, Path]:
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / f"walk_forward_{name}.json"
    payload = {"summary": result["summary"], "folds": result["folds"].to_dict("records"), "audit": result["audit"].to_dict("records")}
    body = json.dumps(payload, ensure_ascii=False, indent=2)
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    payload["sha256"] = digest
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    csv_path = report_dir / f"walk_forward_{name}_returns.csv"
    result["returns"].to_csv(csv_path, encoding="utf-8-sig")
    return json_path, csv_path
