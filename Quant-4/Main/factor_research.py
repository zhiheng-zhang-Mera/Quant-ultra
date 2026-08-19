"""Independent, point-in-time factor diagnostics and production admission gate."""
from __future__ import annotations

import hashlib
import json
from typing import Mapping

import numpy as np
import pandas as pd

FACTOR_RESEARCH_SPEC = {
    "version": "factor-research/v1", "min_dates": 20, "min_abs_rank_ic": 0.02,
    "min_oriented_positive_month_ratio": 0.55, "min_quantile_monotonicity": 0.60,
    "max_turnover": 0.80, "max_abs_size_exposure": 0.50,
    "max_abs_sector_exposure": 2.50, "max_redundancy_correlation": 0.90,
    "min_regime_observations": 5, "quantiles": 5, "participation_rate": 0.01,
}
FACTOR_RESEARCH_SPEC_SHA256 = hashlib.sha256(
    json.dumps(FACTOR_RESEARCH_SPEC, sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()


def _aligned_rows(factor: pd.DataFrame, returns: pd.DataFrame):
    dates = factor.index.intersection(returns.index)
    columns = factor.columns.intersection(returns.columns)
    for date in dates:
        pair = pd.concat([factor.loc[date, columns], returns.loc[date, columns]], axis=1).dropna()
        if len(pair) >= 5:
            yield date, pair.iloc[:, 0].astype(float), pair.iloc[:, 1].astype(float)


def _daily_ic(factor: pd.DataFrame, returns: pd.DataFrame, method: str) -> pd.Series:
    result = {}
    for date, values, target in _aligned_rows(factor, returns):
        if values.nunique() < 2 or target.nunique() < 2:
            continue
        result[date] = values.corr(target, method=method)
    return pd.Series(result, dtype=float).dropna()


def _quantile_returns(factor: pd.DataFrame, returns: pd.DataFrame, quantiles: int) -> dict:
    buckets = {index: [] for index in range(1, quantiles + 1)}
    for _, values, target in _aligned_rows(factor, returns):
        labels = pd.qcut(values.rank(method="first"), quantiles, labels=False, duplicates="drop")
        for label in sorted(labels.dropna().unique()):
            buckets[int(label) + 1].append(float(target.loc[labels.index[labels == label]].mean()))
    return {str(key): float(np.mean(items)) if items else None for key, items in buckets.items()}


def _top_turnover(factor: pd.DataFrame, quantile: float = 0.20) -> float | None:
    prior, changes = None, []
    for _, row in factor.iterrows():
        clean = pd.to_numeric(row, errors="coerce").dropna()
        count = max(1, int(np.ceil(len(clean) * quantile)))
        current = set(clean.nlargest(count).index)
        if prior:
            changes.append(1.0 - len(current & prior) / max(len(current | prior), 1))
        prior = current
    return float(np.mean(changes)) if changes else None


def _mean_cross_section_correlation(left: pd.DataFrame, right: pd.DataFrame, method="spearman") -> float | None:
    values = []
    for _, first, second in _aligned_rows(left, right):
        if first.nunique() < 2 or second.nunique() < 2:
            continue
        corr = first.corr(second, method=method)
        if pd.notna(corr):
            values.append(float(corr))
    return float(np.mean(values)) if values else None


def evaluate_factor(
    factor: pd.DataFrame, forward_returns: Mapping[int, pd.DataFrame], *,
    sectors: Mapping[str, str] | None = None, market_caps: pd.DataFrame | None = None,
    regimes: pd.Series | None = None, adv: pd.DataFrame | None = None,
    transaction_cost_bps: float = 10.0,
) -> dict:
    """Evaluate prediction, stability, decay, cost, capacity and exposures."""
    if 1 not in forward_returns:
        raise ValueError("forward_returns must include horizon 1")
    factor = factor.sort_index().apply(pd.to_numeric, errors="coerce")
    one_day = forward_returns[1].sort_index().apply(pd.to_numeric, errors="coerce")
    pearson, rank_ic = _daily_ic(factor, one_day, "pearson"), _daily_ic(factor, one_day, "spearman")
    mean_rank_ic = float(rank_ic.mean()) if len(rank_ic) else None
    orientation = 1.0 if mean_rank_ic is not None and mean_rank_ic >= 0 else -1.0
    monthly = (rank_ic * orientation).groupby(rank_ic.index.to_period("M")).mean() if isinstance(rank_ic.index, pd.DatetimeIndex) else rank_ic
    stability = float((monthly > 0).mean()) if len(monthly) else None
    icir = float(rank_ic.mean() / rank_ic.std(ddof=1)) if len(rank_ic) > 1 and rank_ic.std(ddof=1) > 0 else None
    quantile_returns = _quantile_returns(factor, one_day, FACTOR_RESEARCH_SPEC["quantiles"])
    ordered = pd.Series(quantile_returns, dtype=float).dropna()
    monotonicity = float(ordered.reset_index(drop=True).corr(pd.Series(range(len(ordered))), method="spearman") * orientation) if len(ordered) >= 3 else None
    decay = {str(horizon): (float(series.mean()) if len(series := _daily_ic(factor, target, "spearman")) else None)
             for horizon, target in sorted(forward_returns.items())}
    turnover = _top_turnover(factor)
    spread = (quantile_returns.get(str(FACTOR_RESEARCH_SPEC["quantiles"])) or 0.0) - (quantile_returns.get("1") or 0.0)
    net_spread = spread * orientation - (turnover or 1.0) * 2.0 * float(transaction_cost_bps) / 10000.0
    size_exposure = _mean_cross_section_correlation(factor, np.log(market_caps.where(market_caps > 0))) if market_caps is not None else None
    sector_exposures = []
    if sectors:
        for _, row in factor.iterrows():
            clean = pd.to_numeric(row, errors="coerce").dropna()
            if len(clean) < 5 or clean.std(ddof=0) == 0:
                continue
            normalized = (clean - clean.mean()) / clean.std(ddof=0)
            grouped = normalized.groupby(pd.Series(sectors).reindex(normalized.index)).mean().dropna()
            if len(grouped):
                sector_exposures.append(float(grouped.abs().max()))
    sector_exposure = float(np.mean(sector_exposures)) if sector_exposures else None
    capacity = None
    if adv is not None:
        capacities = []
        for date, values, _ in _aligned_rows(factor, adv):
            top = values.nlargest(max(1, int(np.ceil(len(values) * 0.20)))).index
            capacities.append(float(pd.to_numeric(adv.loc[date, top], errors="coerce").clip(lower=0).sum()) * FACTOR_RESEARCH_SPEC["participation_rate"])
        capacity = float(np.median(capacities)) if capacities else None
    regime_ic = {}
    if regimes is not None:
        joined = rank_ic.to_frame("rank_ic").join(regimes.rename("regime"), how="inner").dropna()
        regime_ic = {str(key): {"observations": int(len(group)), "rank_ic": float(group["rank_ic"].mean())}
                     for key, group in joined.groupby("regime")}
    checks = {
        "sample_size": len(rank_ic) >= FACTOR_RESEARCH_SPEC["min_dates"],
        "rank_ic": mean_rank_ic is not None and abs(mean_rank_ic) >= FACTOR_RESEARCH_SPEC["min_abs_rank_ic"],
        "time_stability": stability is not None and stability >= FACTOR_RESEARCH_SPEC["min_oriented_positive_month_ratio"],
        "monotonicity": monotonicity is not None and monotonicity >= FACTOR_RESEARCH_SPEC["min_quantile_monotonicity"],
        "decay": len(decay) >= 2 and all(value is not None for value in decay.values()),
        "cost_adjusted_value": net_spread > 0,
        "turnover": turnover is not None and turnover <= FACTOR_RESEARCH_SPEC["max_turnover"],
        "capacity": capacity is not None and capacity > 0,
        "sector_exposure": sector_exposure is not None and sector_exposure <= FACTOR_RESEARCH_SPEC["max_abs_sector_exposure"],
        "size_exposure": size_exposure is not None and abs(size_exposure) <= FACTOR_RESEARCH_SPEC["max_abs_size_exposure"],
        "regime_coverage": len(regime_ic) >= 2 and all(item["observations"] >= FACTOR_RESEARCH_SPEC["min_regime_observations"] for item in regime_ic.values()),
    }
    reasons = [name for name, passed in checks.items() if not passed]
    return {
        "status": "PASS" if not reasons else "HOLD", "action": "PRODUCTION_ELIGIBLE" if not reasons else "RESEARCH_ONLY",
        "checks": checks, "reasons": reasons, "spec": dict(FACTOR_RESEARCH_SPEC),
        "spec_sha256": FACTOR_RESEARCH_SPEC_SHA256,
        "metrics": {"pearson_ic": float(pearson.mean()) if len(pearson) else None, "rank_ic": mean_rank_ic,
                    "icir": icir, "time_stability": stability, "quantile_returns": quantile_returns,
                    "monotonicity": monotonicity, "decay": decay, "turnover": turnover,
                    "gross_spread": float(spread * orientation), "net_spread": float(net_spread),
                    "estimated_capacity": capacity, "sector_exposure": sector_exposure,
                    "size_exposure": size_exposure, "regime_ic": regime_ic},
    }


def analyze_redundancy(factors: Mapping[str, pd.DataFrame], forward_returns: pd.DataFrame) -> dict:
    names, correlations, incremental = sorted(factors), {}, {}
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            value = _mean_cross_section_correlation(factors[left], factors[right])
            correlations[f"{left}|{right}"] = value
        peers = [factors[name] for name in names if name != left]
        residual = factors[left]
        if peers:
            common = sum(frame.rank(axis=1, pct=True) for frame in peers) / len(peers)
            residual = factors[left].rank(axis=1, pct=True) - common
        series = _daily_ic(residual, forward_returns, "spearman")
        incremental[left] = float(series.mean()) if len(series) else None
    redundant = sorted(key for key, value in correlations.items() if value is not None and abs(value) >= FACTOR_RESEARCH_SPEC["max_redundancy_correlation"])
    return {"pairwise_correlation": correlations, "redundant_pairs": redundant, "incremental_rank_ic": incremental}
