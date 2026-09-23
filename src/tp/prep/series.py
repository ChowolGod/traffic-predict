"""Per-zone regular 10-minute series with causal short-gap filling (M-04, D-04)."""

import json
import logging
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from tp import config
from tp.config import PhaseConfig
from tp.data import cache
from tp.errors import TPError
from tp.prep.split import segment_of

log = logging.getLogger(__name__)

MAX_FILL = 3  # consecutive missing slots that may be forward-filled


def series_path(phase_name: str, square_id: int) -> Path:
    return config.PROCESSED_DIR / phase_name / f"series_{square_id}.parquet"


def meta_path(phase_name: str, square_id: int) -> Path:
    return config.PROCESSED_DIR / phase_name / f"series_{square_id}_meta.json"


def local_grid(phase: PhaseConfig) -> pd.DatetimeIndex:
    start = pd.Timestamp(phase.start, tz=config.TZ)
    end = pd.Timestamp(phase.end + timedelta(days=1), tz=config.TZ)
    grid = pd.date_range(start, end, freq="10min", inclusive="left")
    return grid.tz_convert("UTC").as_unit("ns")


def _expected_meta(phase: PhaseConfig, square_id: int) -> dict:
    return {
        "square_id": square_id,
        "phase": phase.name,
        "phase_ranges": {
            s: [a.isoformat(), b.isoformat()] for s, (a, b) in phase.segments().items()
        },
        "agg_version": config.AGG_VERSION,
        "impute_version": config.IMPUTE_VERSION,
    }


def build_series(square_id: int, phase: PhaseConfig) -> pd.DataFrame:
    grid = local_grid(phase)
    present: set[int] = set()
    frames = []
    day = phase.start - timedelta(days=1)
    while day <= phase.end + timedelta(days=1):
        if cache.is_valid(day):
            present.update(cache.read_meta(day)["present_times"])
            frames.append(
                pd.read_parquet(cache.cache_path(day), filters=[("square_id", "==", square_id)])
            )
        day += timedelta(days=1)

    rows = pd.concat(frames, ignore_index=True)
    observed = rows.groupby("time_utc")["internet"].sum().reindex(grid)
    is_present = np.isin(grid.asi8 // 1_000_000, np.fromiter(present, dtype="int64"))
    y = pd.Series(np.where(is_present, observed.fillna(0.0), np.nan), index=grid)

    first = int(np.argmax(is_present)) if is_present.any() else len(y)
    if first > 0:
        log.info(
            "구역 %d: 계열 앞 %d칸이 결측이라 시작을 %s로 미룸",
            square_id,
            first,
            grid[min(first, len(grid) - 1)],
        )
        y = y.iloc[first:]

    missing = y.isna()
    run_id = (missing != missing.shift()).cumsum()
    runs = missing.groupby(run_id).agg(["all", "size"])
    long_runs = runs[runs["all"] & (runs["size"] > MAX_FILL)]
    if not long_runs.empty:
        times = y.index[run_id == long_runs.index[0]].tz_convert(config.TZ)
        raise TPError(
            "E-2002",
            f"긴 결측: {square_id} {times[0]:%Y-%m-%d %H:%M}~{times[-1]:%Y-%m-%d %H:%M} "
            f"({len(times)}칸)",
        )

    out = pd.DataFrame(
        {
            "time_utc": y.index,
            "y": y.ffill().to_numpy(),
            "is_imputed": missing.to_numpy(),
            "segment": segment_of(y.index, phase),
        }
    )
    out["segment"] = out["segment"].astype("category")
    log.info(
        "구역 %d: 채운 칸 %d, 값이 0인 칸 %d",
        square_id,
        out["is_imputed"].sum(),
        (out["y"] == 0).sum(),
    )
    return out


def ensure_series(phase: PhaseConfig, square_id: int) -> Path:
    path, meta = series_path(phase.name, square_id), meta_path(phase.name, square_id)
    expected = _expected_meta(phase, square_id)
    if path.is_file() and meta.is_file():
        if json.loads(meta.read_text(encoding="utf-8")) == expected:
            return path
    df = build_series(square_id, phase)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    meta.write_text(json.dumps(expected), encoding="utf-8")
    return path


def load_series(phase: PhaseConfig, square_id: int) -> pd.DataFrame:
    path, meta = series_path(phase.name, square_id), meta_path(phase.name, square_id)
    if not (path.is_file() and meta.is_file()) or (
        json.loads(meta.read_text(encoding="utf-8")) != _expected_meta(phase, square_id)
    ):
        raise TPError("E-2003", f"prepare 먼저 실행: --phase {phase.name}")
    return pd.read_parquet(path)
