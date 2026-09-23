"""Segment labels by target time and cutting a series at a segment end (M-05, plan 3.3)."""

import numpy as np
import pandas as pd

from tp import config
from tp.config import PhaseConfig


def segment_of(times: pd.Series | pd.DatetimeIndex, phase: PhaseConfig) -> np.ndarray:
    """Label each UTC time by its Europe/Rome calendar day; times outside the phase get None."""
    local_day = pd.DatetimeIndex(times).tz_convert(config.TZ).date
    labels = np.full(len(local_day), None, dtype=object)
    for name, (start, end) in phase.segments().items():
        labels[(local_day >= start) & (local_day <= end)] = name
    return labels


def cut_until(df: pd.DataFrame, segment: str) -> pd.DataFrame:
    """Rows up to and including the last row of `segment` (e.g. drop test before `run`)."""
    last = df.loc[df["segment"] == segment, "time_utc"].max()
    return df[df["time_utc"] <= last].reset_index(drop=True)
