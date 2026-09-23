"""MAE/RMSE on original units, imputed targets excluded, shared eval-time check (M-10, D-08)."""

import numpy as np
import pandas as pd

from tp.errors import TPError

LAG_REF = 144  # official seasonal-naive baseline for rel_mae


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def eval_mask(pred: pd.DataFrame, segment: str) -> pd.Series:
    return (pred["segment"] == segment) & ~pred["is_imputed"].astype(bool)


def eval_times(pred: pd.DataFrame, segment: str) -> pd.DatetimeIndex:
    """Scored target times of the first seed (every seed must share them, see evaluate)."""
    scored = pred[eval_mask(pred, segment)]
    first_seed = scored["seed"].iloc[0]
    return pd.DatetimeIndex(scored.loc[scored["seed"] == first_seed, "time_utc"]).sort_values()


def _check_same(times: pd.DatetimeIndex, reference: pd.DatetimeIndex, what: str) -> None:
    if not times.equals(reference):
        diff = len(times.symmetric_difference(reference))
        raise TPError("E-4004", f"평가 시각 불일치: {diff} ({what})")


def lag144_mae(series: pd.DataFrame, times: pd.DatetimeIndex) -> float:
    y = series.set_index("time_utc")["y"]
    ref = y.shift(LAG_REF).reindex(times)
    return mae(y.reindex(times).to_numpy(), ref.to_numpy())


def evaluate(
    pred: pd.DataFrame,
    series: pd.DataFrame,
    segment: str,
    reference_times: pd.DatetimeIndex | None = None,
) -> dict:
    """D-08 metrics for one experiment's predictions (D-07) on `segment`."""
    scored = pred[eval_mask(pred, segment)]
    times = eval_times(pred, segment)
    if reference_times is not None:
        _check_same(times, pd.DatetimeIndex(reference_times).sort_values(), "기준 실험")

    per_seed = {}
    for seed, group in scored.groupby("seed", sort=True):
        _check_same(pd.DatetimeIndex(group["time_utc"]).sort_values(), times, f"seed {seed}")
        y_true, y_pred = group["y_true"].to_numpy(), group["y_pred"].to_numpy()
        per_seed[int(seed)] = {"mae": mae(y_true, y_pred), "rmse": rmse(y_true, y_pred)}

    ref_mae = lag144_mae(series, times)
    if list(per_seed) == [-1]:
        main, std, seeds = per_seed[-1], None, None
    else:
        maes = [m["mae"] for m in per_seed.values()]
        rmses = [m["rmse"] for m in per_seed.values()]
        main = {"mae": float(np.mean(maes)), "rmse": float(np.mean(rmses))}
        std = {"mae": float(np.std(maes, ddof=1)), "rmse": float(np.std(rmses, ddof=1))}
        seeds = {str(s): m for s, m in per_seed.items()}
    return {
        segment: {**main, "rel_mae": main["mae"] / ref_mae, "n": len(times)},
        f"{segment}_std": std,
        "seeds": seeds,
    }
