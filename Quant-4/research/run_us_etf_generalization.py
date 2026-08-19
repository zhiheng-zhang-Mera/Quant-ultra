"""Frozen-parameter US ETF external validation (research evidence only).

The target market is never searched: the universe and parameters below are
code-owned. Runtime price caches and reports are ignored by Git; only this
reproducible source entry is published.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

QUANT4_ROOT = Path(__file__).resolve().parents[1]
if str(QUANT4_ROOT) not in sys.path:
    sys.path.insert(0, str(QUANT4_ROOT))

from Main.generalization_validation import FrozenResearchLogic, MarketValidationContext, validate_external_market

US_ETF_UNIVERSE = ("SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "GLD", "VNQ", "DBC", "SHY")
FROZEN_PARAMETERS = {"lookback": 126, "rebalance_days": 21, "top_n": 3, "transaction_cost_rate": 0.0005, "max_gross_exposure": 1.0}
PARAMETER_HASH = hashlib.sha256(json.dumps(FROZEN_PARAMETERS, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _git_commit(root: Path) -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "UNKNOWN"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_or_load(cache_dir: Path, *, start: str, end: str, download: bool) -> tuple[dict[str, pd.DataFrame], str]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = {ticker: cache_dir / f"us_etf_{ticker}_history.parquet" for ticker in US_ETF_UNIVERSE}
    raw_paths = {ticker: cache_dir / f"us_etf_{ticker}_raw.csv" for ticker in US_ETF_UNIVERSE}
    missing = [ticker for ticker in US_ETF_UNIVERSE if not paths[ticker].exists() or not raw_paths[ticker].exists()]
    if missing and not download:
        raise FileNotFoundError(f"missing US ETF caches: {missing}; rerun with --download")
    if missing:
        import yfinance as yf
        # Keep yfinance's timezone/cookie SQLite stores inside the project cache;
        # this also makes the one-command run work in isolated CI/workspaces.
        yf.set_tz_cache_location(str(cache_dir / "yfinance_cache"))
        raw = yf.download(list(US_ETF_UNIVERSE), start=start, end=end, auto_adjust=True, actions=False, group_by="ticker", threads=True, progress=False)
        for ticker, path in paths.items():
            frame = raw[ticker].reset_index() if isinstance(raw.columns, pd.MultiIndex) else raw.reset_index()
            frame.to_csv(raw_paths[ticker], index=False)
            frame.columns = [str(column).lower().replace(" ", "_") for column in frame.columns]
            frame = frame.rename(columns={"datetime": "date"})
            required = ["date", "open", "high", "low", "close", "volume"]
            if not set(required).issubset(frame) or frame["close"].dropna().empty:
                raise RuntimeError(f"download returned no valid history for {ticker}")
            frame = frame.loc[:, required].dropna(subset=["date", "open", "close"])
            frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce").dt.tz_localize(None)
            frame["amount"] = pd.to_numeric(frame["close"]) * pd.to_numeric(frame["volume"])
            frame.to_parquet(path, index=False)
    frames = {}
    for ticker, path in paths.items():
        frame = pd.read_parquet(path)
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frames[ticker] = frame.sort_values("date").drop_duplicates("date").set_index("date")
    hashes = {ticker: {"raw": _sha256_file(raw_paths[ticker]), "cleaned": _sha256_file(paths[ticker])}
              for ticker in US_ETF_UNIVERSE}
    data_version = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode("utf-8")).hexdigest()
    return frames, data_version


def frozen_momentum_backtest(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    common = sorted(set.intersection(*(set(frame.index) for frame in frames.values())))
    index = pd.DatetimeIndex(common)
    opens = pd.DataFrame({ticker: frame.reindex(index)["open"] for ticker, frame in frames.items()}).astype(float)
    closes = pd.DataFrame({ticker: frame.reindex(index)["close"] for ticker, frame in frames.items()}).astype(float)
    weights = pd.DataFrame(0.0, index=index, columns=closes.columns)
    current = pd.Series(0.0, index=closes.columns)
    costs = pd.Series(0.0, index=index)
    lookback, cadence, top_n = (FROZEN_PARAMETERS[key] for key in ("lookback", "rebalance_days", "top_n"))
    for position in range(int(lookback), len(index) - 1):
        if (position - int(lookback)) % int(cadence) == 0:
            signal = closes.iloc[position] / closes.iloc[position - int(lookback)] - 1.0
            selected = signal.replace([np.inf, -np.inf], np.nan).dropna().nlargest(int(top_n)).index
            target = pd.Series(0.0, index=closes.columns)
            target.loc[selected] = 1.0 / len(selected)
            execution_position = position + 1
            costs.iloc[execution_position] = float((target - current).abs().sum()) * FROZEN_PARAMETERS["transaction_cost_rate"]
            current = target
        weights.iloc[position + 1] = current
    weights = weights.replace(0.0, np.nan).ffill().fillna(0.0)
    asset_returns = opens.pct_change(fill_method=None)
    gross = (weights.shift(1) * asset_returns).sum(axis=1).fillna(0.0)
    strategy = gross - costs
    benchmark = asset_returns["SPY"].fillna(0.0)
    result = pd.DataFrame({"strategy_return": strategy, "benchmark_return": benchmark,
                           "factor_return": gross, "risk_return": 0.0, "execution_cost": costs}, index=index)
    # Include the first execution date so initial portfolio formation and its
    # transaction cost cannot disappear from the validation ledger.
    return result.iloc[int(lookback) + 1:]


def build_validation(frames: dict[str, pd.DataFrame], data_version: str, project_root: Path) -> dict:
    returns = frozen_momentum_backtest(frames)
    commit = _git_commit(project_root)
    frozen = FrozenResearchLogic("CN_A", "2026-08-19", commit, PARAMETER_HASH, "2026-08-18", "cross-sectional-momentum/v1", "no-overlay/v1", "next-open-us-etf/v1")
    context = MarketValidationContext("US_ETF", f"sha256:{data_version}", PARAMETER_HASH,
                                      str(returns.index.min().date()), str(returns.index.max().date()),
                                      "XNYS", "USD", "fixed-5bps/v1", "T+0;lot=1;gross<=1", True, 0)
    return validate_external_market(returns, context, frozen)


def main() -> int:
    parser = argparse.ArgumentParser(description="Frozen US ETF generalization validation")
    parser.add_argument("--cache-dir", type=Path, default=Path(__file__).parents[1] / "Data_Cache" / "external_validation")
    parser.add_argument("--output", type=Path, default=Path(__file__).parents[1] / "reports" / "us_etf_generalization.json")
    parser.add_argument("--start", default="2016-01-01")
    parser.add_argument("--end", default=(pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    project_root = Path(__file__).parents[2]
    frames, data_version = fetch_or_load(args.cache_dir, start=args.start, end=args.end, download=args.download)
    validation = build_validation(frames, data_version, project_root)
    payload = {"validation": validation, "parameters": FROZEN_PARAMETERS,
               "universe": list(US_ETF_UNIVERSE),
               "limitations": ["Static liquid ETF universe; not a historical constituent study.",
                               "Vendor-adjusted history can be revised after the fact.",
                               "Engineering research evidence only; not investment approval."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": validation["status"], "transfer_verdict": validation["transfer_verdict"],
                      "observations": validation["strategy"]["observations"], "output": str(args.output)}, ensure_ascii=False))
    return 0 if validation["status"] == "COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
