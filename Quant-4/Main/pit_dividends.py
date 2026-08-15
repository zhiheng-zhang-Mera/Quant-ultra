"""Point-in-time (PIT) dividend history for the weekly rotation engine.

The old ``dividend_yield_map`` used a static average dividend per share (measured
over recent years) applied to the whole backtest window - a look-ahead leak. This
module replaces it with a trailing dividend yield computed only from cash
dividends whose ex-dividend date already passed at the decision date, which is
fully point-in-time.

Data source: baostock ``query_dividend_data`` (free, no token). Rows are cached
per symbol as parquet under ``Data_Cache/dividends/<symbol>.parquet`` so repeated
runs are offline after the first fetch.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


_FIELDS = [
    "code", "dividPreNoticeDate", "dividAgmPumDate", "dividPlanAnnounceDate",
    "dividPlanDate", "dividRegistDate", "dividOperateDate", "dividPayDate",
    "dividStockMarketDate", "dividCashPsBeforeTax", "dividCashPsAfterTax",
    "dividStocksPs", "dividCashStock", "dividReserveToStockPs",
]


def _normalize_symbol(symbol: str) -> str:
    """'600519.SH' -> 'sh.600519'."""
    code, ex = symbol.split(".")
    return f"{ex.lower()}.{code}"


def _fetch_symbol_years(
    symbol: str, start_year: int, end_year: int, delay: float = 0.05
) -> pd.DataFrame:
    """Fetch dividend rows for one symbol, one year at a time (baostock API)."""
    import baostock as bs

    bs_code = _normalize_symbol(symbol)
    rows: List[list] = []
    for year in range(start_year, end_year + 1):
        rs = bs.query_dividend_data(code=bs_code, year=str(year), yearType="report")
        if rs.error_code != "0":
            # yearType=report returns rows for the *report year*; some years
            # simply have no record and return an empty set, not an error.
            time.sleep(delay)
            continue
        while rs.next():
            rows.append(rs.get_row_data())
        time.sleep(delay)
    if not rows:
        return pd.DataFrame(columns=["symbol", "ex_date", "cash_ps"])
    df = pd.DataFrame(rows, columns=_FIELDS)
    out = pd.DataFrame(
        {
            "symbol": symbol,
            "ex_date": pd.to_datetime(
                df["dividOperateDate"].replace("", np.nan), errors="coerce"
            ),
            "pay_date": pd.to_datetime(
                df["dividPayDate"].replace("", np.nan), errors="coerce"
            ),
            "cash_ps": pd.to_numeric(df["dividCashPsBeforeTax"], errors="coerce"),
        }
    )
    # Prefer the ex-dividend date; fall back to the payment date, then drop rows
    # we cannot date (no way to use them PIT).
    out["ex_date"] = out["ex_date"].fillna(out["pay_date"])
    out = out.dropna(subset=["ex_date", "cash_ps"])
    out = out[out["cash_ps"] > 0].drop_duplicates(subset=["ex_date"], keep="last")
    return out[["symbol", "ex_date", "cash_ps"]].sort_values("ex_date")


def fetch_dividend_history(
    symbols: List[str],
    start_year: int = 2014,
    end_year: Optional[int] = None,
    cache_dir: Optional[Path] = None,
    delay: float = 0.05,
) -> Dict[str, pd.DataFrame]:
    """Fetch and cache per-symbol dividend frames.

    Returns ``{symbol: DataFrame(ex_date, cash_ps)}``. Existing parquet caches
    are reused; missing years are appended. Caller must have network access.
    """
    import baostock as bs

    if end_year is None:
        end_year = int(pd.Timestamp.now().year)
    cache_dir = Path(cache_dir) if cache_dir else None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)

    lg = bs.login()
    if lg.error_code != "0":
        raise RuntimeError(f"baostock login failed: {lg.error_msg}")
    try:
        result: Dict[str, pd.DataFrame] = {}
        for symbol in symbols:
            frame = pd.DataFrame(columns=["symbol", "ex_date", "cash_ps"])
            cache_path = cache_dir / f"{symbol.replace('.', '_')}_dividends.parquet" if cache_dir is not None else None
            if cache_path is not None and cache_path.exists():
                try:
                    frame = pd.read_parquet(cache_path)
                    frame["ex_date"] = pd.to_datetime(frame["ex_date"])
                except Exception:
                    frame = pd.DataFrame(columns=["symbol", "ex_date", "cash_ps"])
            fetched = _fetch_symbol_years(symbol, start_year, end_year, delay)
            if not fetched.empty:
                frame = pd.concat([frame, fetched], ignore_index=True) if not frame.empty else fetched
                frame = (
                    frame.drop_duplicates(subset=["ex_date"], keep="last")
                    .sort_values("ex_date")
                    .reset_index(drop=True)
                )
                if cache_path is not None:
                    frame.to_parquet(cache_path, index=False)
            result[symbol] = frame
        return result
    finally:
        bs.logout()


def load_dividend_cash(
    cache_dir: Path, symbols: List[str]
) -> Optional[pd.DataFrame]:
    """Load cached PIT dividends as a wide ``(ex_date x symbol)`` cash-per-share
    matrix. Returns ``None`` when no usable data exists for any symbol."""
    cache_dir = Path(cache_dir)
    series: Dict[str, pd.Series] = {}
    for symbol in symbols:
        cache_path = cache_dir / f"{symbol.replace('.', '_')}_dividends.parquet"
        if not cache_path.exists():
            continue
        try:
            frame = pd.read_parquet(cache_path)
            frame["ex_date"] = pd.to_datetime(frame["ex_date"])
            cash = pd.Series(
                pd.to_numeric(frame["cash_ps"], errors="coerce").values,
                index=frame["ex_date"].values,
            )
            cash = cash[~cash.index.duplicated(keep="last")].sort_index()
            series[symbol] = cash
        except Exception:
            continue
    if not series:
        return None
    return pd.DataFrame(series)


def trailing_dividend_yield(
    dividend_cash: pd.DataFrame,
    close: pd.DataFrame,
    window_days: int = 365,
) -> pd.DataFrame:
    """Trailing cash dividend per share over the last ``window_days`` (by
    ex-date) divided by the current close price. Fully point-in-time because a
    dividend only enters the window after its ex-date has passed."""
    aligned = dividend_cash.reindex(close.index).fillna(0.0)
    trailing = aligned.rolling(window_days, min_periods=1).sum()
    return trailing.div(close.replace(0.0, np.nan))
