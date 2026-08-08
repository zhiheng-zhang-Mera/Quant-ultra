"""Optional ML stock-selection layer for the weekly rotation.

A walk-forward LightGBM classifier learns ``P(next-week return > 0)`` from
cross-sectional price/volume features. The core strategy remains rule-based;
this layer only re-ranks candidates when mounted (``selection_model``).
Fits are refreshed on a schedule (e.g. every 25 rebalances) using only data
available at the decision date.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class WeeklyWinPredictor:
    def __init__(self, min_samples: int = 3000, refresh_every: int = 25, prob_threshold: float = 0.0):
        self.min_samples = min_samples
        self.refresh_every = refresh_every
        self.prob_threshold = prob_threshold
        self._model = None
        self._feature_cols: List[str] = []
        self._last_fit_epoch = -10**9

    def _build_samples(self, close: pd.DataFrame, volume: pd.DataFrame, amount: pd.DataFrame, common: pd.DatetimeIndex, upto: pd.Timestamp) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        c = close.loc[:upto]
        cols = ["mom5", "mom10", "mom20", "mom60", "rev1", "close_ma20", "close_ma60", "ma20_ma60", "vol20", "vol_ratio"]
        frames = []
        v = volume.reindex(c.index)
        vol_ratio = v.rolling(5).mean() / v.rolling(20).mean().replace(0, np.nan)
        rets = c.pct_change(fill_method=None)
        vol20 = rets.rolling(20).std(ddof=0) * np.sqrt(252)
        ma20, ma60 = c.rolling(20).mean(), c.rolling(60).mean()
        for sym in c.columns:
            s = c[sym]
            fwd = (s.shift(-5) / s - 1.0) > 0
            df = pd.DataFrame({
                "mom5": s / s.shift(5) - 1.0,
                "mom10": s / s.shift(10) - 1.0,
                "mom20": s / s.shift(20) - 1.0,
                "mom60": s / s.shift(60) - 1.0,
                "rev1": -(s / s.shift(1) - 1.0),
                "close_ma20": s / ma20[sym] - 1.0,
                "close_ma60": s / ma60[sym] - 1.0,
                "ma20_ma60": ma20[sym] / ma60[sym] - 1.0,
                "vol20": vol20[sym],
                "vol_ratio": vol_ratio[sym],
                "_y": fwd,
            })
            frames.append(df)
        train = pd.concat(frames).dropna()
        if len(train) < self.min_samples:
            return np.empty((0, len(cols))), np.empty(0), cols
        return train[cols].values.astype(float), train["_y"].values.astype(int), cols

    def fit_if_due(self, epoch: int, close: pd.DataFrame, volume: pd.DataFrame, amount: pd.DataFrame, common: pd.DatetimeIndex, upto: pd.Timestamp) -> None:
        if self._model is not None and epoch - self._last_fit_epoch < self.refresh_every:
            return
        X, y, cols = self._build_samples(close, volume, amount, common, upto)
        if len(X) < self.min_samples:
            return
        try:
            import lightgbm as lgb
            model = lgb.LGBMClassifier(
                n_estimators=120, num_leaves=15, learning_rate=0.05,
                min_child_samples=80, subsample=0.8, colsample_bytree=0.8,
                random_state=7, verbosity=-1, n_jobs=4,
            )
            model.fit(X, y)
            self._model = model
            self._feature_cols = cols
            self._last_fit_epoch = epoch
        except Exception:
            self._model = None

    def predict(self, panel, date: pd.Timestamp, symbols: List[str]) -> Dict[str, float]:
        """Return P(win) per symbol (empty dict when the model is unavailable)."""
        if self._model is None:
            return {}
        cols = self._feature_cols
        c = panel.close.loc[date]
        ma20 = panel.close.rolling(20).mean().loc[date]
        ma60 = panel.close.rolling(60).mean().loc[date]
        vol20 = panel.volatility.loc[date]
        vol_ratio = panel.volume_ratio.loc[date]
        rows = []
        for sym in symbols:
            if sym not in c.index or not np.isfinite(c[sym]) or sym not in vol_ratio.index or not np.isfinite(vol_ratio[sym]):
                rows.append([np.nan] * len(cols))
                continue
            feat = {}
            for window in (5, 10, 20, 60):
                feat[f"mom{window}"] = panel.momentum[window].loc[date, sym] if window in panel.momentum else np.nan
            feat["rev1"] = -(panel.momentum[1].loc[date, sym]) if 1 in panel.momentum else np.nan
            feat["close_ma20"] = c[sym] / ma20[sym] - 1.0
            feat["close_ma60"] = c[sym] / ma60[sym] - 1.0
            feat["ma20_ma60"] = ma20[sym] / ma60[sym] - 1.0
            feat["vol20"] = vol20[sym]
            feat["vol_ratio"] = vol_ratio[sym]
            rows.append([feat.get(col, np.nan) for col in cols])
        X = np.asarray(rows, dtype=float)
        try:
            probs = self._model.predict_proba(X)
            return {sym: float(probs[i][1]) for i, sym in enumerate(symbols)}
        except Exception:
            return {}
