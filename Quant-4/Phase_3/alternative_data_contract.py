"""Strict provenance contract for production alternative-text inputs.

The contract distinguishes when information was published from when the
research system actually ingested it. Historical decisions may only consume a
record when both timestamps are at or before the decision cutoff.
"""
from __future__ import annotations

import hashlib
import re
import shutil
from pathlib import Path

import pandas as pd


RAW_TEXT_CONTRACT_VERSION = "alternative-text/v1"
SIGNAL_CONTRACT_VERSION = "alternative-signal/v1"
RAW_REQUIRED_COLUMNS = {
    "record_id", "source", "source_type", "license", "published_at",
    "ingested_at", "symbol", "text", "is_synthetic",
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class AlternativeDataContractError(ValueError):
    """Raised when an alternative-data source is unsafe for production use."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cache_immutable_source(path: Path, cache_dir: Path) -> Path:
    """Copy a validated source into a content-addressed, append-only cache."""
    digest = sha256_file(path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    destination = cache_dir / f"{digest}{path.suffix.lower()}"
    if destination.exists():
        if sha256_file(destination) != digest:
            raise AlternativeDataContractError("immutable cache hash mismatch")
        return destination
    shutil.copyfile(path, destination)
    if sha256_file(destination) != digest:
        destination.unlink(missing_ok=True)
        raise AlternativeDataContractError("immutable cache verification failed")
    return destination


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        return pd.read_json(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _as_bool(series: pd.Series) -> pd.Series:
    mapped = series.map({True: True, False: False, 1: True, 0: False,
                         "true": True, "false": False, "True": True, "False": False})
    if mapped.isna().any():
        raise AlternativeDataContractError("is_synthetic must be an explicit boolean")
    return mapped.astype(bool)


def load_contract_text(
    path: Path,
    assets,
    as_of,
    cache_dir: Path,
    require_real_source: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Validate, cache and PIT-filter a raw text source."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"alternative-data source not found: {path}")
    frame = _read_table(path)
    missing = RAW_REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise AlternativeDataContractError(f"alternative-data source missing columns: {sorted(missing)}")
    frame = frame.loc[:, sorted(RAW_REQUIRED_COLUMNS)].copy()
    for column in ("record_id", "source", "source_type", "license", "symbol", "text"):
        frame[column] = frame[column].astype(str).str.strip()
        if frame[column].eq("").any():
            raise AlternativeDataContractError(f"{column} must be non-empty")
    if frame["record_id"].duplicated().any():
        raise AlternativeDataContractError("record_id must be globally unique within a source file")
    frame["symbol"] = frame["symbol"].str.upper()
    frame["is_synthetic"] = _as_bool(frame["is_synthetic"])
    if require_real_source and frame["is_synthetic"].any():
        raise AlternativeDataContractError("synthetic text is prohibited by the production contract")
    for column in ("published_at", "ingested_at"):
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce")
        if frame[column].isna().any():
            raise AlternativeDataContractError(f"{column} contains invalid timestamps")
    if (frame["ingested_at"] < frame["published_at"]).any():
        raise AlternativeDataContractError("ingested_at cannot precede published_at")
    cutoff = pd.Timestamp(as_of)
    cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
    digest = sha256_file(path)
    cached = cache_immutable_source(path, Path(cache_dir))
    eligible = (
        (frame["published_at"] <= cutoff)
        & (frame["ingested_at"] <= cutoff)
        & frame["symbol"].isin({str(asset).upper() for asset in assets})
    )
    selected = frame.loc[eligible].copy()
    evidence = {
        "status": "LOADED",
        "contract_version": RAW_TEXT_CONTRACT_VERSION,
        "path": str(path),
        "immutable_cache_path": str(cached),
        "sha256": digest,
        "records_total": int(len(frame)),
        "records": int(len(selected)),
        "sources": sorted(frame["source"].unique().tolist()),
        "source_types": sorted(frame["source_type"].unique().tolist()),
        "licenses": sorted(frame["license"].unique().tolist()),
        "synthetic_records": int(frame["is_synthetic"].sum()),
        "published_at_max": frame["published_at"].max().isoformat() if len(frame) else None,
        "ingested_at_max": frame["ingested_at"].max().isoformat() if len(frame) else None,
        "as_of": cutoff.isoformat(),
        "dual_timestamp_filter": True,
    }
    return selected, evidence


def validate_signal_provenance(frame: pd.DataFrame) -> dict:
    """Validate provenance columns on a derived PIT signal table."""
    required = {"source_sha256", "contract_version"}
    missing = required - set(frame.columns)
    if missing:
        raise AlternativeDataContractError(f"signal file missing provenance columns: {sorted(missing)}")
    hashes = sorted({str(value).lower() for value in frame["source_sha256"]})
    if not hashes or any(not _SHA256.fullmatch(value) for value in hashes):
        raise AlternativeDataContractError("source_sha256 must contain valid SHA-256 digests")
    versions = sorted({str(value) for value in frame["contract_version"]})
    if versions != [SIGNAL_CONTRACT_VERSION]:
        raise AlternativeDataContractError(
            f"contract_version must be {SIGNAL_CONTRACT_VERSION!r}; found {versions}"
        )
    return {"contract_version": SIGNAL_CONTRACT_VERSION, "source_sha256": hashes}
