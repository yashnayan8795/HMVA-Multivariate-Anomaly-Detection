from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Tuple

def rank_normalize(arr: np.ndarray) -> np.ndarray:
    # Convert to ranks in [0,1]
    order = np.argsort(arr)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.linspace(0.0, 1.0, len(arr), endpoint=True)
    return ranks

def percentile_scale(values: np.ndarray) -> np.ndarray:
    # Map to 0-100 by percentile within the vector itself
    ranks = rank_normalize(values)
    return 100.0 * ranks

def calibrate_training(scores: np.ndarray, train_mask: np.ndarray) -> np.ndarray:
    # Ensure training mean < 10 and max < 25 by a global shrink if needed
    if train_mask.sum() == 0:
        return scores
    train_scores = scores[train_mask]
    mean_tr = float(np.mean(train_scores))
    max_tr = float(np.max(train_scores))
    factors = []
    if mean_tr >= 10.0:
        factors.append((10.0 / (mean_tr + 1e-9)) * 0.99)
    if max_tr >= 25.0:
        factors.append((25.0 / (max_tr + 1e-9)) * 0.99)
    if factors:
        alpha = min(factors)
        scores = np.clip(scores * alpha, 0.0, 100.0)
    return scores

def smooth_series(series: np.ndarray, window: int = 3) -> np.ndarray:
    if window <= 1:
        return series
    s = pd.Series(series).rolling(window=window, min_periods=1, center=True).median().to_numpy()
    return s
