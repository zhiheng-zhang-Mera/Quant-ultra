from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Main.factor_research import analyze_redundancy, evaluate_factor


def _panels():
    rng = np.random.default_rng(819)
    dates = pd.bdate_range("2023-01-02", periods=80)
    symbols = [f"S{i:02d}" for i in range(20)]
    base = np.linspace(-1, 1, len(symbols))
    factor = pd.DataFrame([base + rng.normal(0, .03, len(symbols)) for _ in dates], index=dates, columns=symbols)
    one = factor * .01 + pd.DataFrame(rng.normal(0, .001, factor.shape), index=dates, columns=symbols)
    five = factor * .006 + pd.DataFrame(rng.normal(0, .002, factor.shape), index=dates, columns=symbols)
    caps = pd.DataFrame(rng.lognormal(20, 1, factor.shape), index=dates, columns=symbols)
    adv = pd.DataFrame(rng.lognormal(16, .2, factor.shape), index=dates, columns=symbols)
    sectors = {symbol: f"sector-{index % 4}" for index, symbol in enumerate(symbols)}
    regimes = pd.Series(np.where(np.arange(len(dates)) % 2, "BULL", "BEAR"), index=dates)
    return factor, {1: one, 5: five}, caps, adv, sectors, regimes


def test_complete_factor_report_covers_prediction_cost_exposure_and_regimes():
    factor, returns, caps, adv, sectors, regimes = _panels()
    report = evaluate_factor(factor, returns, sectors=sectors, market_caps=caps, regimes=regimes, adv=adv)
    assert report["status"] == "PASS", report["reasons"]
    assert report["metrics"]["rank_ic"] > 0
    assert set(report["metrics"]["decay"]) == {"1", "5"}
    assert len(report["metrics"]["quantile_returns"]) == 5
    assert set(report["metrics"]["regime_ic"]) == {"BULL", "BEAR"}
    assert report["metrics"]["estimated_capacity"] > 0


def test_missing_capacity_exposure_and_regime_evidence_holds_factor():
    factor, returns, *_ = _panels()
    report = evaluate_factor(factor, returns)
    assert report["status"] == "HOLD"
    assert report["action"] == "RESEARCH_ONLY"
    assert {"capacity", "sector_exposure", "size_exposure", "regime_coverage"} <= set(report["reasons"])


def test_redundant_factor_and_incremental_information_are_reported():
    factor, returns, *_ = _panels()
    result = analyze_redundancy({"alpha": factor, "clone": factor * 2, "noise": factor.sample(frac=1, axis=1)}, returns[1])
    assert "alpha|clone" in result["redundant_pairs"]
    assert set(result["incremental_rank_ic"]) == {"alpha", "clone", "noise"}
