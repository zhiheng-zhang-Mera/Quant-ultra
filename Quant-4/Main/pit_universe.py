"""Point-in-time (PIT), survivorship-free universe construction.

Survivorship bias is removed at the universe level: membership comes from
baostock's full A-share master list (every stock ever listed, including the
~250 names that delisted during the backtest window), not from a 2026
hand-picked list. A symbol is a candidate on date ``t`` only if it was alive
on ``t`` (``ipo_date <= t < out_date``); a name that later delists is still a
candidate while it was listed, which is exactly the point-in-time property a
fair backtest needs.

ETFs are a small, auditable fixed universe (broad / theme / cross-border /
safe assets). Index products persist through market cycles and are not subject
to the same selection bias, but the list is kept explicit so it can be audited.

All functions are pure pandas and run offline once the master list is cached.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

MASTER_REL = Path("universe") / "ashare_master.parquet"

# Broad / theme / cross-border ETF seats plus safe assets for defensive holds.
ETF_UNIVERSE: Tuple[str, ...] = (
    # broad market
    "510050.SH", "510300.SH", "510500.SH", "510880.SH", "159915.SZ", "159919.SZ",
    "588000.SH", "159949.SZ",
    # cross-border / sector themes
    "159941.SZ", "513100.SH", "513500.SH", "513050.SH", "512000.SH",
    "512480.SH", "512660.SH", "512690.SH", "515030.SH", "515880.SH",
    # safe assets (defensive hold)
    "511010.SH", "511260.SH", "518880.SH", "511880.SH",
)

# Beijing Stock Exchange prefixes are excluded: they are not part of the
# baostock type-1 master set we consume and their trading rules/liquidity are
# incompatible with the rotation's execution model.
EXCLUDED_PREFIXES: Tuple[str, ...] = ("4", "8", "92")

_MASTER_COLUMNS = ["symbol", "code_name", "ipo_date", "out_date", "type", "status"]


def to_symbol(code: str) -> str:
    """'sh.600000' -> '600000.SH' (baostock code -> engine symbol)."""
    exchange, number = str(code).lower().split(".")
    return f"{number}.{exchange.upper()}"


def fetch_master_list(cache_dir: Path, force: bool = False) -> pd.DataFrame:
    """Fetch the ever-listed A-share master list from baostock and cache it.

    Columns: symbol, code_name, ipo_date, out_date, type, status.
    ``type == '1'`` are stocks; a non-empty ``out_date`` or ``status == '0'``
    means the name is no longer listed (delisted/absorbed).
    """
    import baostock as bs

    cache_path = cache_dir / MASTER_REL
    if cache_path.exists() and not force:
        cached = load_master_list(cache_dir)
        if len(cached):
            logger.info("Reusing cached A-share master list: %d rows", len(cached))
            return cached

    login = bs.login()
    if login.error_code != "0":
        raise RuntimeError(f"baostock login failed: {login.error_msg}")
    try:
        rs = bs.query_stock_basic()
        rows: List[list] = []
        while (rs.error_code == "0") and rs.next():
            rows.append(rs.get_row_data())
        if not rows:
            raise RuntimeError("baostock query_stock_basic returned no rows")
        raw = pd.DataFrame(rows, columns=rs.fields)
    finally:
        bs.logout()

    df = pd.DataFrame(
        {
            "symbol": raw["code"].map(to_symbol),
            "code_name": raw["code_name"],
            "ipo_date": pd.to_datetime(raw["ipoDate"].replace("", pd.NaT), errors="coerce"),
            "out_date": pd.to_datetime(raw["outDate"].replace("", pd.NaT), errors="coerce"),
            "type": raw["type"],
            "status": raw["status"],
        }
    )
    df = df[df["symbol"].str.match(r"^\d{6}\.(SH|SZ)$", na=False)].drop_duplicates("symbol")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    logger.info("A-share master list fetched and cached: %d rows", len(df))
    return df


def load_master_list(cache_dir: Path) -> pd.DataFrame:
    """Load the cached master list (empty frame when not yet fetched)."""
    cache_path = cache_dir / MASTER_REL
    if not cache_path.exists():
        return pd.DataFrame(columns=_MASTER_COLUMNS)
    df = pd.read_parquet(cache_path)
    for col in ("ipo_date", "out_date"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def ever_alive_stocks(
    master: pd.DataFrame,
    start: str,
    end: str,
    stock_type: str = "1",
    exclude_prefixes: Sequence[str] = EXCLUDED_PREFIXES,
) -> List[str]:
    """Every A-share stock alive at some point inside ``[start, end]``.

    This is the survivorship-free universe: a name that delisted inside the
    window is included so the backtest can see its decline and death.
    """
    if master.empty:
        return []
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    sub = master[(master["type"] == stock_type) & (master["ipo_date"] <= end_ts)]
    sub = sub[sub["out_date"].isna() | (sub["out_date"] >= start_ts)]
    out: List[str] = []
    for symbol in sub["symbol"]:
        code = symbol.split(".")[0]
        if code.startswith(exclude_prefixes):
            continue
        out.append(symbol)
    return sorted(out)


def alive_matrix(
    master: pd.DataFrame,
    dates: Sequence,
    symbols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Boolean ``(date x symbol)`` matrix: symbol alive on date (PIT membership)."""
    timeline = pd.to_datetime(pd.Index(dates))
    if symbols is None:
        symbols = list(master["symbol"])
    if master.empty or not symbols:
        return pd.DataFrame(False, index=timeline, columns=list(symbols))
    sub = master[master["symbol"].isin(symbols)].set_index("symbol").reindex(list(symbols))
    born = pd.to_datetime(sub["ipo_date"], errors="coerce")
    death = pd.to_datetime(sub["out_date"], errors="coerce").fillna(pd.Timestamp("2099-12-31"))
    mask = (born.values[None, :] <= timeline.values[:, None]) & (
        timeline.values[:, None] <= death.values[None, :]
    )
    return pd.DataFrame(mask, index=timeline, columns=list(symbols))


def stock_out_date(master: pd.DataFrame) -> Dict[str, Optional[pd.Timestamp]]:
    """symbol -> out_date (None when still listed)."""
    if master.empty:
        return {}
    sub = master[master["type"] == "1"]
    return {
        row.symbol: (row.out_date if pd.notna(row.out_date) else None)
        for row in sub.itertuples()
    }


def universe_coverage(
    frames: Dict[str, pd.DataFrame],
    master: pd.DataFrame,
    start: str,
    end: str,
) -> dict:
    """Audit dict describing how much of the PIT universe the cache covers."""
    expected_stocks = ever_alive_stocks(master, start, end)
    expected_etfs = [s for s in ETF_UNIVERSE]
    expected = expected_stocks + expected_etfs
    covered = sorted(set(frames) & set(expected))
    out_dates = stock_out_date(master)
    covered_delisted = [s for s in covered if out_dates.get(s) is not None]
    delisted_expected = [s for s in expected_stocks if out_dates.get(s) is not None]
    return {
        "start": start,
        "end": end,
        "expected_stocks": len(expected_stocks),
        "expected_etfs": len(expected_etfs),
        "expected_total": len(expected),
        "covered_total": len(covered),
        "covered_stocks": len([s for s in covered if s in expected_stocks]),
        "coverage_ratio": round(len(covered) / len(expected), 4) if expected else 0.0,
        "delisted_expected": len(delisted_expected),
        "delisted_covered": len(covered_delisted),
        "delisted_coverage_ratio": (
            round(len(covered_delisted) / len(delisted_expected), 4) if delisted_expected else 0.0
        ),
        "missing_sample": sorted(set(expected) - set(covered))[:50],
    }
