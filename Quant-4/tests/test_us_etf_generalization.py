from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.run_us_etf_generalization import FROZEN_PARAMETERS, PARAMETER_HASH, US_ETF_UNIVERSE, frozen_momentum_backtest


def _frames():
    dates = pd.bdate_range("2019-01-01", periods=420)
    frames = {}
    for index, ticker in enumerate(US_ETF_UNIVERSE):
        close = 100 * np.exp(np.linspace(0, .1 + index * .02, len(dates)))
        frames[ticker] = pd.DataFrame({"open": close, "close": close}, index=dates)
    return frames


def test_frozen_us_etf_entry_has_no_target_search_and_no_leverage():
    assert FROZEN_PARAMETERS["max_gross_exposure"] == 1.0
    assert len(PARAMETER_HASH) == 64
    result = frozen_momentum_backtest(_frames())
    assert len(result) >= 252
    assert {"strategy_return", "benchmark_return", "factor_return", "risk_return", "execution_cost"} <= set(result)
    assert result["execution_cost"].sum() > 0


def test_future_price_mutation_cannot_change_prior_external_results():
    frames = _frames()
    cutoff = next(iter(frames.values())).index[300]
    first = frozen_momentum_backtest(frames)
    mutated = {ticker: frame.copy() for ticker, frame in frames.items()}
    for frame in mutated.values():
        frame.loc[frame.index > cutoff, ["open", "close"]] *= 10
    second = frozen_momentum_backtest(mutated)
    pd.testing.assert_frame_equal(first.loc[:cutoff], second.loc[:cutoff])
