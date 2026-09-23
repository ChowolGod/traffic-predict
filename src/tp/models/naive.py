"""Naive baselines: y_hat[t] = y[t - max(lag, horizon)] (M-07, F-05, F-12)."""

import pandas as pd

from tp.errors import TPError


def predict_naive(
    series: pd.DataFrame, lag: int, targets: pd.DatetimeIndex, horizon: int = 1
) -> pd.Series:
    """series is a D-04 frame on a regular 10-minute grid. With lag < horizon the forecast is
    the last value observed at the origin t - horizon (lag 1 = persistence)."""
    effective = max(lag, horizon)
    shifted = series.set_index("time_utc")["y"].shift(effective)
    missing = targets.difference(shifted.dropna().index)
    if len(missing):
        raise TPError("E-4004", f"평가 시각 불일치: {len(missing)} (lag {effective} 기록 부족)")
    return shifted.loc[targets]
