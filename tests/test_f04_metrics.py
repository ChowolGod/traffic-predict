import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tp.errors import TPError
from tp.eval import metrics

GOLDEN = json.loads((Path(__file__).parent / "fixtures" / "golden.json").read_text("utf-8"))
T0 = pd.Timestamp("2013-11-03 23:00", tz="UTC").as_unit("ns")


def make_series(n_days: int = 3, y=None) -> pd.DataFrame:
    n = 144 * n_days
    times = pd.date_range(T0, periods=n, freq="10min")
    values = np.arange(n, dtype=float) if y is None else y
    seg = np.where(np.arange(n) < 144 * (n_days - 1), "train", "val")
    return pd.DataFrame({"time_utc": times, "y": values, "is_imputed": False, "segment": seg})


def make_pred(series: pd.DataFrame, offset: float = 1.0, seeds=(-1,)) -> pd.DataFrame:
    val = series[series["segment"] == "val"]
    frames = [
        pd.DataFrame(
            {
                "time_utc": val["time_utc"].to_numpy(),
                "segment": "val",
                "y_true": val["y"].to_numpy(),
                "y_pred": val["y"].to_numpy() + offset * (1 + i),
                "seed": seed,
                "is_imputed": val["is_imputed"].to_numpy(),
            }
        )
        for i, seed in enumerate(seeds)
    ]
    return pd.concat(frames, ignore_index=True)


# --- 완료 조건 (1) 손계산 예제와 1e-9 이내 -------------------------------------------
def test_golden_mae_rmse():
    g = GOLDEN["metrics"]
    assert metrics.mae(np.array(g["y_true"]), np.array(g["y_pred"])) == pytest.approx(
        g["mae"], abs=1e-9
    )
    assert metrics.rmse(np.array(g["y_true"]), np.array(g["y_pred"])) == pytest.approx(
        g["rmse"], abs=1e-9
    )


def test_evaluate_single_model():
    s = make_series()
    out = metrics.evaluate(make_pred(s, offset=2.0), s, "val")
    assert out["val"]["mae"] == pytest.approx(2.0, abs=1e-9)
    assert out["val"]["rmse"] == pytest.approx(2.0, abs=1e-9)
    assert out["val"]["n"] == 144
    assert out["val_std"] is None and out["seeds"] is None


# --- 완료 조건 (2) 채운 목표 시각은 지표에서 빠짐 --------------------------------------
def test_imputed_targets_excluded():
    s = make_series()
    s.loc[s.index[-5:], "is_imputed"] = True
    pred = make_pred(s, offset=2.0)
    pred.loc[pred["is_imputed"], "y_pred"] += 1e6
    out = metrics.evaluate(pred, s, "val")
    assert out["val"]["mae"] == pytest.approx(2.0, abs=1e-9)
    assert out["val"]["n"] == 139


def test_only_requested_segment_is_scored():
    s = make_series()
    pred = make_pred(s, offset=2.0)
    extra = pred.iloc[:3].copy()
    extra["segment"] = "train"
    extra["y_pred"] += 1e6
    out = metrics.evaluate(pd.concat([pred, extra]), s, "val")
    assert out["val"]["mae"] == pytest.approx(2.0, abs=1e-9)


# --- 시드 평균·표준편차(ddof=1) -------------------------------------------------------
def test_multi_seed_mean_and_std():
    s = make_series()
    out = metrics.evaluate(make_pred(s, offset=1.0, seeds=(0, 1, 2)), s, "val")
    maes = [1.0, 2.0, 3.0]
    assert out["seeds"]["0"]["mae"] == pytest.approx(1.0)
    assert out["val"]["mae"] == pytest.approx(np.mean(maes))
    assert out["val_std"]["mae"] == pytest.approx(np.std(maes, ddof=1))
    assert out["val"]["n"] == 144


# --- rel_mae = 모델 MAE / 같은 시각의 lag-144 MAE ----------------------------------------
def test_rel_mae_against_lag144():
    s = make_series()  # y = t, so y[t] - y[t-144] = 144 everywhere
    out = metrics.evaluate(make_pred(s, offset=36.0), s, "val")
    assert out["val"]["rel_mae"] == pytest.approx(36.0 / 144.0, abs=1e-12)
    assert metrics.lag144_mae(s, metrics.eval_times(make_pred(s), "val")) == pytest.approx(144.0)


# --- 완료 조건 (3) 평가 시각 집합이 다르면 E-4004 ---------------------------------------
def test_e4004_reference_mismatch():
    s = make_series()
    pred = make_pred(s)
    ref = metrics.eval_times(pred, "val")[1:]
    with pytest.raises(TPError) as exc:
        metrics.evaluate(pred, s, "val", reference_times=ref)
    assert exc.value.code == "E-4004"
    assert "1" in exc.value.message


def test_e4004_seed_sets_differ():
    s = make_series()
    pred = make_pred(s, seeds=(0, 1, 2))
    pred = pred.drop(pred[(pred["seed"] == 1)].index[:2])
    with pytest.raises(TPError) as exc:
        metrics.evaluate(pred, s, "val")
    assert exc.value.code == "E-4004"


def test_reference_match_passes():
    s = make_series()
    pred = make_pred(s)
    ref = metrics.eval_times(pred, "val")
    assert metrics.evaluate(pred, s, "val", reference_times=ref)["val"]["n"] == 144
    assert math.isfinite(metrics.evaluate(pred, s, "val")["val"]["rel_mae"])
