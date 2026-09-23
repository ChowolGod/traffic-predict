import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.arima.model import ARIMA

from tests.test_f06_arima import cfg as arima_cfg
from tests.test_f06_arima import synthetic
from tests.test_f07_lstm import SMALL
from tp import cli, config
from tp.errors import TPError
from tp.exp import registry
from tp.models import arima, lstm, naive
from tp.seed import set_seed


# --- (2) naive: ŷ_t = y[t − max(lag, h)] ----------------------------------------------------
@pytest.mark.parametrize("lag,h", [(1, 1), (1, 3), (1, 6), (144, 6), (1008, 3)])
def test_naive_uses_last_observation_or_seasonal_lag(lag, h):
    s = synthetic(days=9)
    eff = max(lag, h)
    targets = pd.DatetimeIndex(s["time_utc"].iloc[eff:])
    pred = naive.predict_naive(s, lag, targets, horizon=h)
    assert (pred.to_numpy() == s["y"].to_numpy()[:-eff]).all()


# --- (3) ARIMA h스텝 = 원점별 기준 방법(필터 후 forecast), 1e-6 ----------------------------------
def reference_forecast(params: dict, s: pd.DataFrame, origin_row: int, h: int) -> float:
    """Filter the data up to origin_row with fixed params, forecast h steps, undo diff144."""
    endog, exog, start = arima._design(s, params)
    o = origin_row - start  # endog index of the origin
    ex_now = None if exog is None else exog[start : start + o + 1]
    ex_future = None if exog is None else exog[start + o + 1 : start + o + 1 + h]
    model = ARIMA(endog[: o + 1], exog=ex_now, order=tuple(params["order"]))
    z = model.filter(np.asarray(params["params"])).forecast(h, exog=ex_future)[-1]
    target_row = origin_row + h
    return z + s["y"].to_numpy()[target_row - 144] if start else z


@pytest.mark.parametrize("seasonal", ["none", "diff144", "fourier"])
@pytest.mark.parametrize("h", [3, 6])
def test_arima_h_step_matches_reference(seasonal, h):
    s = synthetic()
    params = arima.fit_arima(s[s["segment"] == "train"], arima_cfg(seasonal))
    pred = arima.predict_arima(params, s, horizon=h)
    for origin_row in (300, 500, 700, len(s) - h - 1):
        expected = reference_forecast(params, s, origin_row, h)
        assert pred.loc[origin_row + h] == pytest.approx(expected, abs=1e-6)


def test_arima_h1_unchanged():
    s = synthetic()
    params = arima.fit_arima(s[s["segment"] == "train"], arima_cfg("fourier"))
    pd.testing.assert_series_equal(
        arima.predict_arima(params, s), arima.predict_arima(params, s, horizon=1)
    )


# --- (4) 인과성: y[k]를 바꾸면 대상 시각 k + h부터만 달라짐 ------------------------------------
@pytest.mark.parametrize("h", [3, 6])
def test_arima_h_step_is_causal(h):
    s = synthetic()
    params = arima.fit_arima(s[s["segment"] == "train"], arima_cfg())
    base = arima.predict_arima(params, s, horizon=h)
    k = s.index[s["segment"] == "val"][50]
    changed = s.copy()
    changed.loc[k, "y"] += 1000.0
    after = arima.predict_arima(params, changed, horizon=h)
    assert np.allclose(after.loc[: k + h - 1], base.loc[: k + h - 1], atol=1e-9)
    assert abs(after.loc[k + h] - base.loc[k + h]) > 1.0


def train_lstm(s, h, seed=0):
    set_seed(seed, 2)
    return lstm.train_lstm(s, {**SMALL, "horizon": h}, seed, lambda: None)


def test_lstm_h_step_is_causal():
    h = 3
    s = synthetic()
    model = train_lstm(s, h)
    targets = s.index[SMALL["lstm.window"] + h - 1 :]
    base = lstm.predict_lstm(model, s, targets)
    k = s.index[s["segment"] == "val"][50]
    changed = s.copy()
    changed.loc[k, "y"] += 1000.0
    after = lstm.predict_lstm(model, changed, targets)
    assert np.allclose(after.loc[: k + h - 1], base.loc[: k + h - 1], atol=1e-9)
    assert abs(after.loc[k + h] - base.loc[k + h]) > 1e-3


def test_lstm_rejects_targets_without_history():
    s = synthetic()
    model = train_lstm(s, 6)
    with pytest.raises(TPError) as exc:
        lstm.predict_lstm(model, s, s.index[SMALL["lstm.window"] : SMALL["lstm.window"] + 3])
    assert exc.value.code == "E-4004"


def test_lstm_train_samples_shrink_with_horizon():
    s = synthetic()
    w = SMALL["lstm.window"]
    assert lstm.count_train_samples(s, w, 6) == lstm.count_train_samples(s, w, 1) - 5


def test_lstm_saved_model_keeps_horizon(tmp_path):
    s = synthetic()
    model = train_lstm(s, 3)
    lstm.save_model(model, tmp_path, seed=0)
    loaded = lstm.load_model(tmp_path, seed=0, cfg={**SMALL, "horizon": 3})
    targets = s.index[SMALL["lstm.window"] + 2 :]
    pd.testing.assert_series_equal(
        lstm.predict_lstm(loaded, s, targets), lstm.predict_lstm(model, s, targets)
    )


# --- (1) 레지스트리: 기존 실험은 horizon=1, 한 요소 규칙, 루트 --------------------------------
@pytest.fixture
def prepared(workspace):
    assert cli.main(["prepare", "--phase", "dev"]) == 0
    return workspace


def test_resolved_config_defaults_horizon_1():
    from tests.test_f08_registry import ROOT_CFG

    assert registry.resolve_config(ROOT_CFG)["horizon"] == 1


def test_old_experiment_without_horizon_is_read_as_1(prepared):
    from tests.test_f08_registry import ROOT_CFG, folder, run

    assert run(ROOT_CFG) == 0
    path = folder("EXP-001") / "config.yaml"
    text = path.read_text(encoding="utf-8").replace("horizon: 1\n", "")
    path.write_text(text, encoding="utf-8")  # simulate a v1 experiment folder
    assert "horizon" not in text
    assert registry.scan()["EXP-001"].config["horizon"] == 1
    # results reads the same folder as horizon 1 and still computes decision metrics
    from tp.exp import results

    rows = results.load_rows(config.load_decision())
    row = next(r for r in rows if r["exp_id"] == "EXP-001")
    assert row["status"] == "completed" and row["horizon"] == 1
    assert row["dec_episodes"] is not None


def test_old_lstm_config_without_horizon_loads_as_1(tmp_path):
    s = synthetic()
    model = train_lstm(s, 1)
    lstm.save_model(model, tmp_path, seed=0)
    loaded = lstm.load_model(tmp_path, seed=0, cfg=dict(SMALL))  # v1 config: no horizon key
    assert "horizon" not in SMALL and loaded.horizon == 1
    targets = s.index[SMALL["lstm.window"] :]
    pd.testing.assert_series_equal(
        lstm.predict_lstm(loaded, s, targets), lstm.predict_lstm(model, s, targets)
    )


def test_changed_horizon_child_runs_with_exact_naive_values(prepared):
    from tests.test_f08_registry import ROOT_CFG, child, folder, run

    assert run(ROOT_CFG) == 0
    assert run(child(changed="naive.lag", naive={"lag": 1})) == 0
    grand = {**child(changed="horizon", naive={"lag": 1}), "id": "EXP-003", "name": "h3",
             "parent": "EXP-002", "horizon": 3}  # fmt: skip
    assert run(grand) == 0
    pred = pd.read_parquet(folder("EXP-003") / "predictions.parquet").set_index("time_utc")
    y = pd.read_parquet(config.PROCESSED_DIR / "dev" / "series_5000.parquet").set_index("time_utc")
    t = pred.index[pred["segment"] == "val"]
    assert (pred.loc[t, "y_pred"].to_numpy() == y["y"].shift(3).loc[t].to_numpy()).all()
    assert registry.scan()["EXP-003"].config["horizon"] == 3


@pytest.mark.parametrize(
    "value,expected",
    [pytest.param(3, "루트", id="root-not-h1"), pytest.param(2, "horizon 값", id="not-allowed")],
)
def test_e4001_horizon_rules(prepared, value, expected):
    from tests.test_f08_registry import ROOT_CFG, write_cfg

    with pytest.raises(TPError) as exc:
        registry.run_from_file(write_cfg({**ROOT_CFG, "horizon": value}))
    assert exc.value.code == "E-4001" and expected in exc.value.message


def test_lag_1008_is_allowed():
    from tests.test_f08_registry import ROOT_CFG

    assert registry.resolve_config({**ROOT_CFG, "naive": {"lag": 1008}})["naive.lag"] == 1008
