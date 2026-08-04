"""Audited, data-gated parameter iteration for analysis-only backtests."""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pandas as pd

SCHEMA_VERSION = 1
GRID_KEYS = ("fast_window", "slow_window", "vol_window", "target_vol")


def _canonical(value: dict) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _digest(value: dict) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _normalize_grid(grid: dict) -> dict:
    missing = [key for key in GRID_KEYS if key not in grid]
    if missing:
        raise ValueError(f"adaptive grid missing keys: {missing}")
    normalized = {
        "fast_window": sorted({int(v) for v in grid["fast_window"] if 1 < int(v) <= 60}),
        "slow_window": sorted({int(v) for v in grid["slow_window"] if 10 <= int(v) <= 252}),
        "vol_window": sorted({int(v) for v in grid["vol_window"] if 1 < int(v) <= 120}),
        "target_vol": sorted({round(float(v), 6) for v in grid["target_vol"] if 0.05 <= float(v) <= 0.50}),
    }
    if any(not values for values in normalized.values()):
        raise ValueError("adaptive grid contains an empty bounded axis")
    return normalized


def _state_path(state_dir: Path, symbol: str) -> Path:
    return Path(state_dir) / f"{symbol.replace('.', '_')}.json"


def _load(path: Path, symbol: str, base_hash: str) -> dict | None:
    if not path.exists():
        return None
    state = json.loads(path.read_text(encoding="utf-8"))
    checksum = state.pop("checksum", None)
    if checksum != _digest(state):
        raise ValueError(f"adaptive parameter state checksum mismatch: {path}")
    if state.get("schema_version") != SCHEMA_VERSION or state.get("symbol") != symbol:
        raise ValueError("adaptive parameter state identity mismatch")
    if state.get("base_grid_hash") != base_hash:
        raise ValueError("adaptive parameter state belongs to a different base grid")
    return state


def prepare_iteration(symbol: str, data_end, base_grid: dict, state_dir: Path) -> dict:
    """Choose a run grid without advancing state until the run succeeds."""
    base = _normalize_grid(base_grid)
    base_hash = _digest(base)
    end = pd.Timestamp(data_end).date().isoformat()
    path = _state_path(state_dir, symbol)
    state = _load(path, symbol, base_hash)
    if state is None:
        return {"grid": base, "generation": 1, "advance": True, "status": "COLD_START", "state_path": path, "data_end": end, "base_grid_hash": base_hash}
    trained_through = state["trained_through"]
    if end < trained_through:
        raise ValueError(f"data end {end} predates adaptive state {trained_through}")
    if end == trained_through:
        return {"grid": _normalize_grid(state["last_run_grid"]), "generation": state["generation"], "advance": False, "status": "REPLAY_NO_NEW_DATA", "state_path": path, "data_end": end, "base_grid_hash": base_hash}
    return {"grid": _normalize_grid(state["next_grid"]), "generation": state["generation"] + 1, "advance": True, "status": "NEW_DATA_ITERATION", "state_path": path, "data_end": end, "base_grid_hash": base_hash}


def _bounded_neighbors(key: str, value: float) -> list:
    if key == "fast_window": return [max(2, int(value) - 5), min(60, int(value) + 5)]
    if key == "slow_window": return [max(10, int(value) - 20), min(252, int(value) + 20)]
    if key == "vol_window": return [max(2, int(value) - 10), min(120, int(value) + 10)]
    return [max(0.05, round(float(value) - 0.025, 6)), min(0.50, round(float(value) + 0.025, 6))]


def _refine_grid(active_grid: dict, folds: pd.DataFrame) -> dict:
    refined = deepcopy(_normalize_grid(active_grid))
    caps = {"fast_window": 5, "slow_window": 5, "vol_window": 4, "target_vol": 5}
    if folds.empty:
        return refined
    for key in GRID_KEYS:
        mode = folds[key].mode(dropna=True)
        if mode.empty:
            continue
        center = mode.iloc[0]
        candidates = sorted(set(refined[key]) | set(_bounded_neighbors(key, center)))
        cap = caps[key]
        if len(candidates) > cap:
            mandatory = {candidates[0], candidates[-1], center}
            ranked = sorted(candidates, key=lambda value: (abs(float(value) - float(center)), float(value)))
            selected = list(mandatory)
            for value in ranked:
                if value not in selected and len(selected) < cap:
                    selected.append(value)
            candidates = sorted(selected)
        refined[key] = candidates
    return _normalize_grid(refined)


def finalize_iteration(symbol: str, prepared: dict, folds: pd.DataFrame) -> dict:
    """Persist the next proposal only after a successful leakage-audited run."""
    evidence = {"generation": int(prepared["generation"]), "status": prepared["status"], "advanced": bool(prepared["advance"]), "trained_through": prepared["data_end"], "state_path": str(prepared["state_path"])}
    if not prepared["advance"]:
        return evidence
    active = _normalize_grid(prepared["grid"])
    payload = {
        "schema_version": SCHEMA_VERSION,
        "symbol": symbol,
        "generation": int(prepared["generation"]),
        "trained_through": prepared["data_end"],
        "base_grid_hash": prepared["base_grid_hash"],
        "last_run_grid": active,
        "next_grid": _refine_grid(active, folds),
        "fold_parameter_frequency": folds.groupby(list(GRID_KEYS)).size().sort_values(ascending=False).head(10).reset_index(name="folds").to_dict("records"),
    }
    payload["checksum"] = _digest(payload)
    path = Path(prepared["state_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)
    evidence["next_grid"] = payload["next_grid"]
    evidence["checksum"] = payload["checksum"]
    return evidence
