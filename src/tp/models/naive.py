"""Seasonal naive / persistence baseline: y_hat[t] = y[t - lag] (M-07, F-05)."""

import pandas as pd

from tp.errors import TPError


def predict_naive(series: pd.DataFrame, lag: int, targets: pd.DatetimeIndex) -> pd.Series:
    """series is a D-04 frame on a regular 10-minute grid."""
    shifted = series.set_index("time_utc")["y"].shift(lag)
    missing = targets.difference(shifted.dropna().index)
    if len(missing):
        raise TPError("E-4004", f"평가 시각 불일치: {len(missing)} (lag {lag} 기록 부족)")
    return shifted.loc[targets]
