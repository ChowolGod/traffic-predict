import json

import numpy as np
import pandas as pd
import pytest

from tp import cli
from tp.errors import TPError
from tp.models import arima

T0 = pd.Timestamp("2013-11-03 23:00", tz="UTC").as_unit("ns")  # 2013-11-04 00:00 Rome


def synthetic(days: int = 6, seed: int = 0) -> pd.DataFrame:
    """Daily sine + AR(1) noise; last day is val."""
    rng = np.random.default_rng(seed)
    n = 144 * days
    noise = np.zeros(n)
    for t in range(1, n):
        noise[t] = 0.6 * noise[t - 1] + rng.normal(0, 2)
    y = 100 + 30 * np.sin(2 * np.pi * np.arange(n) / 144) + noise
    seg = np.where(np.arange(n) < 144 * (days - 1), "train", "val")
    return pd.DataFrame(
        {
            "time_utc": pd.date_range(T0, periods=n, freq="10min"),
            "y": y,
            "is_imputed": False,
            "segment": pd.Categorical(seg),
        }
    )


def cfg(seasonal: str = "none", order=(2, 1, 2), k: int = 3) -> dict:
    out = {"arima.order": list(order), "arima.seasonal": seasonal}
    if seasonal == "fourier":
        out["arima.fourier_k"] = k
    return out


def fit_predict(s: pd.DataFrame, c: dict):
    params = arima.fit_arima(s[s["segment"] == "train"], c)
    return params, arima.predict_arima(params, s)


# --- 완료 조건 (2) 같은 설정이면 결과가 같음(1e-9) ---------------------------------------
@pytest.mark.parametrize("seasonal", ["none", "diff144", "fourier"])
def test_reproducible(seasonal):
    s = synthetic()
    p1, y1 = fit_predict(s, cfg(seasonal))
    p2, y2 = fit_predict(s, cfg(seasonal))
    assert np.allclose(p1["params"], p2["params"], rtol=0, atol=1e-9)
    pd.testing.assert_series_equal(y1, y2, check_exact=False, atol=1e-9, rtol=0)


# --- 평가 구간에서 재적합하지 않음 --------------------------------------------------------
def test_predict_does_not_refit(monkeypatch):
    s = synthetic()
    params = arima.fit_arima(s[s["segment"] == "train"], cfg())

    def no_fit(*a, **k):
        raise AssertionError("predict에서 재적합함")

    monkeypatch.setattr(arima.ARIMA, "fit", no_fit)
    assert len(arima.predict_arima(params, s)) > 0


# --- 1스텝 예측: t의 예측은 y[t] 이후 값에 의존하지 않음 -----------------------------------
@pytest.mark.parametrize("seasonal", ["none", "diff144", "fourier"])
def test_one_step_is_causal(seasonal):
    # h=1 only; h>1 causality is tested in test_f12_horizon
    s = synthetic()
    params, base = fit_predict(s, cfg(seasonal))
    k = s.index[s["segment"] == "val"][50]
    changed = s.copy()
    changed.loc[k, "y"] += 1000.0
    after = arima.predict_arima(params, changed)
    assert np.allclose(after.loc[:k].to_numpy(), base.loc[:k].to_numpy(), atol=1e-9)
    assert abs(after.loc[k + 1] - base.loc[k + 1]) > 1.0


def test_predictions_cover_val_and_are_indexed_by_row():
    s = synthetic()
    _, pred = fit_predict(s, cfg("diff144"))
    val_rows = s.index[s["segment"] == "val"]
    assert set(val_rows) <= set(pred.index)
    assert pred.index.min() == 144  # diff144은 앞 144칸 이후부터 예측
    assert np.isfinite(pred.to_numpy()).all()


# --- diff144: ŷ_t = y_{t-144} + ẑ_t ---------------------------------------------------------
def test_diff144_reconstruction():
    s = synthetic()
    params, pred = fit_predict(s, cfg("diff144", order=(0, 0, 0)))
    y = s["y"].to_numpy()
    const = params["params"][0]  # ARIMA(0,0,0) with constant: ẑ_t = c
    t = np.arange(144, len(s))
    assert np.allclose(pred.loc[t].to_numpy(), y[t - 144] + const, atol=1e-6)


# --- fourier: Europe/Rome 기준 하루 슬롯으로 위상 결정 ------------------------------------
def test_fourier_phase_by_local_slot():
    s = synthetic(days=2)
    x = arima.fourier_terms(s["time_utc"], k=2)
    assert list(x.columns) == ["sin1", "cos1", "sin2", "cos2"]
    assert x.iloc[0]["sin1"] == pytest.approx(0.0) and x.iloc[0]["cos1"] == pytest.approx(1.0)
    assert x.iloc[36]["sin1"] == pytest.approx(1.0)  # 06:00 local → quarter day
    assert np.allclose(x.iloc[:144].to_numpy(), x.iloc[144:].to_numpy())


# --- D-11 파라미터 저장·로드 --------------------------------------------------------------
def test_params_roundtrip(tmp_path):
    s = synthetic()
    params, pred = fit_predict(s, cfg("fourier"))
    path = tmp_path / "model" / "arima_params.json"
    arima.save_params(path, params)
    loaded = arima.load_params(path)
    assert set(json.loads(path.read_text(encoding="utf-8"))) >= {
        "order", "seasonal", "fourier_k", "param_names", "params",
    }  # fmt: skip
    pd.testing.assert_series_equal(arima.predict_arima(loaded, s), pred)


# --- E-3001 수렴 실패 ---------------------------------------------------------------------
class _NotConverged:
    def __init__(self, *a, **k):
        pass

    def fit(self, *a, **k):
        class R:
            mle_retvals = {"converged": False}
            params = np.array([0.0])
            param_names = ["const"]

        return R()


def test_e3001_not_converged(monkeypatch):
    monkeypatch.setattr(arima, "ARIMA", _NotConverged)
    s = synthetic()
    with pytest.raises(TPError) as exc:
        arima.fit_arima(s[s["segment"] == "train"], cfg())
    assert exc.value.code == "E-3001"


def test_e3001_on_convergence_warning(monkeypatch):
    from statsmodels.tools.sm_exceptions import ConvergenceWarning

    class Warns(_NotConverged):
        def fit(self, *a, **k):
            import warnings

            warnings.warn("did not converge", ConvergenceWarning, stacklevel=1)

            class R:
                mle_retvals = {"converged": True}
                params = np.array([0.0])
                param_names = ["const"]

            return R()

    monkeypatch.setattr(arima, "ARIMA", Warns)
    s = synthetic()
    with pytest.raises(TPError) as exc:
        arima.fit_arima(s[s["segment"] == "train"], cfg())
    assert exc.value.code == "E-3001"


# --- 레지스트리 연동(완료 조건 1) -----------------------------------------------------------
def test_registry_runs_arima(workspace):
    from tests.test_f08_registry import ROOT_CFG, child, folder, meta, run

    assert cli.main(["prepare", "--phase", "dev"]) == 0
    assert run(ROOT_CFG) == 0
    c = child(changed="model.type", model={"type": "arima"},
              arima={"order": [2, 1, 2], "seasonal": "none"})  # fmt: skip
    del c["naive"]
    assert run(c) == 0
    assert meta("EXP-002")["status"] == "completed"
    assert (folder("EXP-002") / "model" / "arima_params.json").is_file()
    m = json.loads((folder("EXP-002") / "metrics.json").read_text(encoding="utf-8"))
    assert np.isfinite(m["val"]["mae"]) and m["val"]["n"] == 144
