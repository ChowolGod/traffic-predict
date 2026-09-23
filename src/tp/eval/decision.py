"""Decision-level metrics for congestion alarms (M-18, F-13, plan 3.3 결정 지표)."""

import numpy as np

from tp.config import DecisionConfig

__all__ = ["DecisionConfig", "decision_metrics", "episodes", "threshold"]

SLOT_MIN = 10


def threshold(train_y: np.ndarray, cfg: DecisionConfig) -> float:
    return float(cfg.threshold_ratio * np.quantile(train_y, cfg.peak_quantile))


def episodes(congested: np.ndarray, merge_gap: int) -> list[tuple[int, int]]:
    """Runs of True as (start, end) inclusive; runs split by <= merge_gap False slots merge."""
    runs: list[tuple[int, int]] = []
    for i in np.flatnonzero(congested):
        if runs and i - runs[-1][1] - 1 <= merge_gap:
            runs[-1] = (runs[-1][0], int(i))
        else:
            runs.append((int(i), int(i)))
    return runs


def decision_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    imputed: np.ndarray,
    thr: float,
    horizon: int,
    merge_gap: int,
) -> dict:
    """Arrays are consecutive 10-minute target slots. An alarm for slot t is issued at t - h."""
    congested = (y_true >= thr) & ~imputed
    alarm = (y_pred >= thr) & ~imputed
    spans = episodes(congested, merge_gap)
    in_episode = np.zeros(len(y_true), dtype=bool)
    leads = []
    for start, end in spans:
        in_episode[start : end + 1] = True
        early = np.flatnonzero(alarm[start : min(end, start + horizon) + 1])
        if len(early):  # earliest target in [s, s + h] → earliest issue time <= s
            leads.append((horizon - early[0]) * SLOT_MIN)
    return {
        "episodes": len(spans),
        "missed": len(spans) - len(leads),
        "false_alarm_min": int((alarm & ~in_episode).sum()) * SLOT_MIN,
        "lead_min": float(np.mean(leads)) if leads else None,
    }
