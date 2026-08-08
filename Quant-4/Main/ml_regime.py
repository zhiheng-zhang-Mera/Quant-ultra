"""Optional machine-learning market-regime detectors for the weekly rotation.

Kept fully decoupled from the core strategy: the strategy consumes a detector
through a tiny ``detect(date, bench_close) -> dict`` interface. Every detector
is walk-forward (fit only on data available at the decision date), so no
look-ahead leaks into the backtest. Fits use a bounded rolling window so the
cost per decision stays roughly constant.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class MLRegimeResult:
    regime: str          # BULL / BEAR / NEUTRAL
    confidence: float    # 0..1 probability-like score
    model: str


def _precompute_features(bench_close: pd.Series) -> pd.DataFrame:
    """Vectorized feature panel for the logistic regime detector."""
    close = bench_close.astype(float)
    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()
    ma120 = close.rolling(120).mean()
    fwd21 = close.shift(-21) / close - 1.0
    ret5 = close / close.shift(5) - 1.0
    ret20 = close / close.shift(20) - 1.0
    ret60 = close / close.shift(60) - 1.0
    vol20 = close.pct_change(fill_method=None).rolling(60).std(ddof=0)
    return pd.DataFrame({
        "close_ma20": close / ma20 - 1.0,
        "close_ma60": close / ma60 - 1.0,
        "close_ma120": close / ma120 - 1.0,
        "ma20_ma60": ma20 / ma60 - 1.0,
        "ret5": ret5,
        "ret20": ret20,
        "ret60": ret60,
        "vol20": vol20,
        "_fwd21": fwd21,
    })


class HMMRegimeDetector:
    """Gaussian HMM over benchmark log-returns; the high-mean state is BULL."""

    def __init__(self, n_states: int = 3, min_train: int = 200, fit_window: int = 750):
        self.n_states = n_states
        self.min_train = min_train
        self.fit_window = fit_window

    def detect(self, bench_close: pd.Series, date: pd.Timestamp) -> MLRegimeResult:
        hist = bench_close.loc[:date]
        rets_full = np.log(hist / hist.shift(1)).dropna()
        rets = rets_full.tail(self.fit_window).values.reshape(-1, 1)
        if len(rets) < self.min_train:
            return MLRegimeResult("NEUTRAL", 0.5, "hmm")
        try:
            from hmmlearn import hmm
            model = hmm.GaussianHMM(n_components=self.n_states, covariance_type="diag", n_iter=30, random_state=7)
            model.fit(rets)
            state = model.predict(rets)[-1]
            means = model.means_.flatten()
            if state == int(np.argmax(means)):
                return MLRegimeResult("BULL", 0.75, "hmm")
            if state == int(np.argmin(means)):
                return MLRegimeResult("BEAR", 0.75, "hmm")
            return MLRegimeResult("NEUTRAL", 0.5, "hmm")
        except Exception:
            return MLRegimeResult("NEUTRAL", 0.5, "hmm")


class LogisticRegimeDetector:
    """Rolling-window logistic regression on trend features -> P(next month up).

    The target uses the forward monthly return only to build training labels;
    predictions use only features available at the decision date.
    """

    def __init__(self, min_train: int = 300, bull_threshold: float = 0.48, fit_window: int = 750):
        self.min_train = min_train
        self.bull_threshold = bull_threshold
        self.fit_window = fit_window
        self._feature_cache: Optional[pd.DataFrame] = None

    def detect(self, bench_close: pd.Series, date: pd.Timestamp) -> MLRegimeResult:
        if self._feature_cache is None:
            self._feature_cache = _precompute_features(bench_close)
        feats = self._feature_cache
        pos = feats.index.get_indexer([date], method="ffill")
        if len(pos) == 0 or pos[0] < 0:
            return MLRegimeResult("NEUTRAL", 0.5, "logit")
        end = pos[0]
        start = max(0, end - self.fit_window)
        window = feats.iloc[start:end + 1]
        if len(window) < self.min_train:
            return MLRegimeResult("NEUTRAL", 0.5, "logit")
        cols = [c for c in sorted(feats.columns) if c != "_fwd21"]
        # Strict walk-forward: labels use the 21-day forward return, so training
        # rows must end at least 21 days before the decision date.
        train = window.iloc[:-21]
        if len(train) < self.min_train:
            return MLRegimeResult("NEUTRAL", 0.5, "logit")
        X = train[cols].values.astype(float)
        fwd = train["_fwd21"].values.astype(float)
        mask = np.isfinite(X).all(axis=1) & np.isfinite(fwd)
        if int(mask.sum()) < self.min_train:
            return MLRegimeResult("NEUTRAL", 0.5, "logit")
        X, y = X[mask], (fwd[mask] > 0).astype(float)
        try:
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression(max_iter=500, C=0.5)
            clf.fit(X, y)
            x_now = window[cols].values.astype(float)[-1]
            prob = float(clf.predict_proba(x_now.reshape(1, -1))[0][1])
            if prob >= self.bull_threshold:
                return MLRegimeResult("BULL", prob, "logit")
            if prob <= 1.0 - self.bull_threshold:
                return MLRegimeResult("BEAR", 1.0 - prob, "logit")
            return MLRegimeResult("NEUTRAL", prob, "logit")
        except Exception:
            return MLRegimeResult("NEUTRAL", 0.5, "logit")


class EnsembleRegimeDetector:
    """Conservative vote: any two detectors agreeing on BEAR forces cash."""

    def __init__(self, bull_threshold: float = 0.50, **kwargs):
        self.hmm = HMMRegimeDetector(**kwargs)
        self.logit = LogisticRegimeDetector(bull_threshold=bull_threshold)

    def detect(self, bench_close: pd.Series, date: pd.Timestamp) -> MLRegimeResult:
        r1 = self.hmm.detect(bench_close, date)
        r2 = self.logit.detect(bench_close, date)
        votes = [r1.regime, r2.regime]
        if votes.count("BEAR") >= 2:
            return MLRegimeResult("BEAR", max(r1.confidence, r2.confidence), "ensemble")
        if votes.count("BULL") >= 2:
            return MLRegimeResult("BULL", max(r1.confidence, r2.confidence), "ensemble")
        return MLRegimeResult("NEUTRAL", 0.5, "ensemble")


def build_detector(model: str, **kwargs):
    """Factory: 'hmm' | 'logit' | 'ensemble'."""
    if model == "hmm":
        return HMMRegimeDetector(**kwargs)
    if model == "logit":
        return LogisticRegimeDetector(**kwargs)
    if model == "ensemble":
        return EnsembleRegimeDetector(**kwargs)
    raise ValueError(f"unknown regime model {model!r}")
