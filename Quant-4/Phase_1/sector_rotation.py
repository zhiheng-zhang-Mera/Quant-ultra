"""Sector-first universe selection for Phase 1."""
from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("Orchestrator.Phase1.SectorRotation")


class SectorSelectionUnavailableError(RuntimeError):
    """Raised when sector-first selection cannot run and no cached selection exists."""


def _column(frame: pd.DataFrame, *candidates: str) -> str:
    normalized = {str(col).strip().lower(): col for col in frame.columns}
    for candidate in candidates:
        if candidate.lower() in normalized:
            return normalized[candidate.lower()]
    raise KeyError(f"none of {candidates!r} found in {list(frame.columns)!r}")


def _safe_key(code: str, name: str) -> str:
    value = re.sub(r"[^0-9A-Za-z_-]+", "_", str(code or name)).strip("_")
    return value or "unknown_sector"


def normalize_sector_history(raw: pd.DataFrame) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()
    mapping = {
        _column(raw, "日期", "date"): "date",
        _column(raw, "开盘", "open"): "open",
        _column(raw, "最高", "high"): "high",
        _column(raw, "最低", "low"): "low",
        _column(raw, "收盘", "close"): "close",
        _column(raw, "成交额", "amount", "turnover"): "amount",
    }
    frame = raw.rename(columns=mapping)[list(mapping.values())].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    for name in ("open", "high", "low", "close", "amount"):
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
    return frame.dropna(subset=["date", "close", "amount"]).sort_values("date").drop_duplicates("date", keep="last")


def score_sector_histories(histories: dict[str, pd.DataFrame], as_of: str | pd.Timestamp) -> pd.DataFrame:
    """Score sectors and enforce the 40% volatility / 1% liquidity gates."""
    cutoff = pd.Timestamp(as_of)
    rows = []
    for name, history in histories.items():
        frame = history[pd.to_datetime(history["date"]) <= cutoff].sort_values("date")
        if len(frame) < 61:
            continue
        returns = frame["close"].pct_change()
        adv5 = frame["amount"].tail(5).mean()
        rows.append({
            "sector": name,
            "momentum_20d": frame["close"].iloc[-1] / frame["close"].iloc[-21] - 1.0,
            "adv5": adv5,
            "adv60": frame["amount"].tail(60).mean(),
            "annualized_volatility": returns.tail(60).std(ddof=0) * np.sqrt(252),
        })
    scored = pd.DataFrame(rows)
    if scored.empty:
        return scored
    total_adv5 = scored["adv5"].sum()
    scored["amount_share_5d"] = scored["adv5"] / total_adv5 if total_adv5 > 0 else 0.0
    scored["adv5_adv60"] = scored["adv5"] / scored["adv60"].replace(0, np.nan)
    for source, rank in (("momentum_20d", "momentum_rank"), ("amount_share_5d", "amount_share_rank"), ("adv5_adv60", "adv_ratio_rank")):
        scored[rank] = scored[source].rank(pct=True, method="average")
    scored["score"] = 0.4 * scored["momentum_rank"] + 0.3 * scored["amount_share_rank"] + 0.3 * scored["adv_ratio_rank"]
    scored["eligible"] = (scored["annualized_volatility"] <= 0.40) & (scored["amount_share_5d"] >= 0.01)
    return scored.sort_values(["eligible", "score", "sector"], ascending=[False, False, True]).reset_index(drop=True)


def _merge_parquet(path: Path, fresh: pd.DataFrame) -> pd.DataFrame:
    existing = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    merged = pd.concat([existing, fresh], ignore_index=True) if not existing.empty else fresh.copy()
    merged = merged.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    temp = path.with_suffix(".parquet.tmp")
    merged.to_parquet(temp, index=False)
    temp.replace(path)
    return merged


def _symbol(raw) -> str | None:
    code = str(raw).strip().split(".")[0].zfill(6)
    if len(code) != 6 or not code.isdigit():
        return None
    suffix = "BJ" if code.startswith(("4", "8", "9")) else ("SH" if code.startswith(("5", "6")) else "SZ")
    return f"{code}.{suffix}"


def select_sector_universe(data_manager, as_of: str, top_n: int = 3, lookback_days: int = 140) -> tuple[list[str], pd.DataFrame]:
    """Refresh market-wide sector indexes, then load members of only the top sectors."""
    if not hasattr(data_manager, "_ak"):
        raise RuntimeError("sector-first selection requires the AkShare industry-board provider")
    root = data_manager.cache_dir / "sector_rotation"
    index_dir, selection_dir = root / "indexes", root / "selections"
    index_dir.mkdir(parents=True, exist_ok=True)
    selection_dir.mkdir(parents=True, exist_ok=True)

    try:
        return _refresh_sector_selection(data_manager, index_dir, selection_dir, as_of, top_n, lookback_days)
    except SectorSelectionUnavailableError:
        raise
    except Exception as exc:
        logger.warning("[SECTOR] Network refresh failed (%s); attempting cached selection fallback", exc)
        cached = _load_latest_cached_selection(selection_dir)
        if cached is None:
            raise SectorSelectionUnavailableError(
                "sector selection unavailable and no cached selection exists; caller should fall back to full-universe screening"
            ) from exc
        symbols, selected, selection_date = cached
        logger.warning("[SECTOR] Using cached sector selection from %s (degraded mode); sectors=%s constituents=%s", selection_date, selected["sector"].tolist(), len(symbols))
        selected.attrs["degraded"] = True
        return symbols, selected


def _load_latest_cached_selection(selection_dir: Path):
    """Return (symbols, ranking, selection_date) from the most recent cached selection."""
    manifests = sorted(selection_dir.glob("*.json"))
    if not manifests:
        return None
    latest = manifests[-1]
    try:
        payload = json.loads(latest.read_text(encoding="utf-8"))
        symbols = payload.get("symbols") or []
        selection_date = payload.get("selection_date", latest.stem)
        if not symbols:
            return None
        ranking = pd.DataFrame(payload.get("sectors", []))
        if ranking.empty and payload.get("sectors"):
            ranking = pd.DataFrame({"sector": list(payload["sectors"])})
        ranking.attrs["degraded"] = True
        ranking.attrs["selection_date"] = selection_date
        return symbols, ranking, selection_date
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        logger.warning("[SECTOR] Cached selection %s is unreadable: %s", latest, exc)
        return None


def _refresh_sector_selection(data_manager, index_dir: Path, selection_dir: Path, as_of: str, top_n: int, lookback_days: int):
    """Network path: refresh board indexes, score, and select top sectors."""
    board_frame = data_manager._bounded_source_call("industry_board_list", data_manager._ak.stock_board_industry_name_em)
    name_col = _column(board_frame, "板块名称", "行业名称", "name")
    try:
        code_col = _column(board_frame, "板块代码", "行业代码", "code")
        boards = board_frame[[code_col, name_col]].dropna().drop_duplicates().rename(columns={code_col: "code", name_col: "name"})
    except KeyError:
        boards = board_frame[[name_col]].dropna().drop_duplicates().rename(columns={name_col: "name"})
        boards["code"] = boards["name"]
        boards = boards[["code", "name"]]
    histories: dict[str, pd.DataFrame] = {}
    end = pd.Timestamp(as_of)
    default_start = (end - pd.Timedelta(days=lookback_days)).strftime("%Y%m%d")

    def refresh_board(row):
        path = index_dir / f"{_safe_key(row.code, row.name)}.parquet"
        start = default_start
        if path.exists():
            cached = pd.read_parquet(path, columns=["date"])
            if not cached.empty:
                start = (pd.to_datetime(cached["date"]).max() - pd.Timedelta(days=5)).strftime("%Y%m%d")
        raw = data_manager._bounded_source_call(
            f"industry_board_history:{row.name}", data_manager._ak.stock_board_industry_hist_em,
            symbol=row.name, period="日k", start_date=start, end_date=end.strftime("%Y%m%d"), adjust="",
        )
        fresh = normalize_sector_history(raw)
        if not fresh.empty:
            return row.name, _merge_parquet(path, fresh)
        if path.exists():
            return row.name, pd.read_parquet(path)
        return row.name, pd.DataFrame()

    rows = list(boards.itertuples(index=False))
    workers = min(len(rows), max(1, getattr(getattr(data_manager, "download_plan", None), "workers", 4)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(refresh_board, row): row.name for row in rows}
        for future in as_completed(futures):
            name, history = future.result()
            if not history.empty:
                histories[name] = history

    ranking = score_sector_histories(histories, as_of)
    selected = ranking.loc[ranking["eligible"]].head(top_n).copy()
    if len(selected) < top_n:
        raise RuntimeError(f"only {len(selected)} sectors passed volatility/liquidity gates; require {top_n}")

    symbols: list[str] = []
    selected_names = selected["sector"].tolist()
    for sector in selected_names:
        constituents = data_manager._bounded_source_call(
            f"industry_board_constituents:{sector}", data_manager._ak.stock_board_industry_cons_em, symbol=sector,
        )
        code = _column(constituents, "代码", "股票代码", "symbol", "code")
        symbols.extend(filter(None, (_symbol(value) for value in constituents[code])))
    symbols = sorted(set(symbols))
    if not symbols:
        raise RuntimeError("selected sectors returned no valid constituents")

    selected["selection_date"] = end.strftime("%Y-%m-%d")
    selected.to_parquet(selection_dir / f"{end.strftime('%Y-%m-%d')}.parquet", index=False)
    manifest = {"selection_date": end.strftime("%Y-%m-%d"), "sectors": selected_names, "symbols": symbols, "generated_at": datetime.now().astimezone().isoformat()}
    (selection_dir / f"{end.strftime('%Y-%m-%d')}.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("[SECTOR] Selected top %s sectors: %s; constituents=%s", top_n, selected_names, len(symbols))
    return symbols, selected
