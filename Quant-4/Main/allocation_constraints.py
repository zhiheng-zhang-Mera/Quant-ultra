"""Pure allocation constraints shared by optimizers and lightweight tests."""
from __future__ import annotations

import numpy as np


def apply_allocation_cap(upper_bounds: np.ndarray, context: dict, config: dict) -> np.ndarray:
    """Apply the strictest valid per-asset cap before compiling constraints."""
    raw_cap = context.get("enforce_crowded_allocation_cap", config.get("enforce_crowded_allocation_cap", 1.0))
    cap = float(np.clip(raw_cap, 0.0, 1.0))
    return np.minimum(np.maximum(np.asarray(upper_bounds, dtype=float), 0.0), cap)
