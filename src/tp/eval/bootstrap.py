"""Daily block bootstrap CI for an MAE difference (M-17, plan 3.3 불확실성)."""

import numpy as np

N_BOOT = 2000


def block_bootstrap_diff_ci(
    err_model: np.ndarray,
    err_ref: np.ndarray,
    day_index: np.ndarray,
    n: int = N_BOOT,
    seed: int = 0,
) -> tuple[float, float, float]:
    """(mean(err_model) - mean(err_ref), 2.5th, 97.5th percentile) resampling whole days."""
    diff = err_model - err_ref
    days = np.unique(day_index)
    blocks = [diff[day_index == d] for d in days]
    rng = np.random.default_rng(seed)
    stats = np.empty(n)
    for i in range(n):
        picked = rng.integers(0, len(blocks), size=len(blocks))
        stats[i] = np.concatenate([blocks[j] for j in picked]).mean()
    low, high = np.percentile(stats, [2.5, 97.5])
    return float(diff.mean()), float(low), float(high)
