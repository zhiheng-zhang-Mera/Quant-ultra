"""Deterministic OHLCV validation and evidence generation."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

REQUIRED = ("date", "open", "high", "low", "close", "volume")

def validate_ohlcv(df: pd.DataFrame, symbol: str = "unknown") -> dict:
    errors = []
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing: errors.append(f"missing columns: {missing}")
    if df.empty: errors.append("empty dataset")
    if not missing and not df.empty:
        d = pd.to_datetime(df["date"], errors="coerce")
        if d.isna().any(): errors.append("invalid dates")
        if d.duplicated().any(): errors.append("duplicate dates")
        if not d.is_monotonic_increasing: errors.append("dates not ascending")
        numeric = df[["open", "high", "low", "close", "volume"]].apply(pd.to_numeric, errors="coerce")
        if not np.isfinite(numeric.to_numpy()).all(): errors.append("non-finite OHLCV")
        if (numeric[["open", "high", "low", "close"]] <= 0).any().any(): errors.append("non-positive price")
        if (numeric["volume"] < 0).any(): errors.append("negative volume")
        if (numeric["high"] < numeric[["open", "low", "close"]].max(axis=1)).any(): errors.append("high below OHLC")
        if (numeric["low"] > numeric[["open", "high", "close"]].min(axis=1)).any(): errors.append("low above OHLC")
    canonical = df.to_csv(index=False, date_format="%Y-%m-%d").encode("utf-8")
    return {"symbol": symbol, "valid": not errors, "errors": errors, "rows": len(df), "sha256": hashlib.sha256(canonical).hexdigest(), "first_date": str(df["date"].min()) if "date" in df and len(df) else None, "last_date": str(df["date"].max()) if "date" in df and len(df) else None}

def write_manifest(path: Path, evidence: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")

