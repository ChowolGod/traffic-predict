"""F-14 켜기/끄기 시뮬레이션 (plan 3.3 켜기/끄기 시뮬레이션, M-19, D-16)."""

import numpy as np
import pandas as pd
import pytest
import yaml

from tests.test_f09_testrun import full_ws  # noqa: F401  (fixture)
from tp import cli, config
from tp.errors import TPError
from tp.eval import switching

THR = 10.0
#                 0  1   2   3  4  5  6   7
Y_TRUE = np.array([0, 12, 12, 0, 0, 0, 12, 0], dtype=float)
PRED = np.array([0, 11, 0, 0, 0, 0, 11, 0], dtype=float)


def on(y_pred=PRED, horizon=1, hold_min=0, on_ratio=1.0, delay_min=0, imputed=None):
    imputed = np.zeros(len(y_pred), dtype=bool) if imputed is None else imputed
    return switching.simulate(y_pred, imputed, THR, horizon, hold_min, on_ratio, delay_min)


def metrics(state, imputed=None, merge_gap=1):
    imputed = np.zeros(len(Y_TRUE), dtype=bool) if imputed is None else imputed
    return switching.switching_metrics(state, Y_TRUE, imputed, THR, merge_gap)


# --- (1) 손계산 예제 ------------------------------------------------------------------------
def test_follows_the_forecast_without_hold():
    state = on()
    assert state.tolist() == [False, True, False, False, False, False, True, False]
    # congested 1, 2, 6 -> off at 2; episodes [1, 2] and [6] both start while on
    assert metrics(state) == {"episodes": 2, "missed_min": 10, "missed_episodes": 0,
                              "on_min": 20, "switches": 2}  # fmt: skip


def test_minimum_hold_keeps_the_cell_on():
    state = on(hold_min=30)  # on at 1 -> held 1..3; on at 6 -> held 6..7 (series ends)
    assert state.tolist() == [False, True, True, True, False, False, True, True]
    assert metrics(state) == {"episodes": 2, "missed_min": 0, "missed_episodes": 0,
                              "on_min": 50, "switches": 2}  # fmt: skip


def test_hold_ends_only_when_the_forecast_drops():
    pred = np.array([0, 11, 11, 11, 11, 0, 0, 0], dtype=float)
    assert on(pred, hold_min=20).tolist() == [False, True, True, True, True, False, False, False]


def test_on_ratio_lowers_the_switch_on_level():
    pred = np.array([0, 9.5, 9.5, 0, 0, 0, 0, 0])
    assert not on(pred, on_ratio=1.0).any()
    assert on(pred, on_ratio=0.9).tolist()[:3] == [False, True, True]


def test_delay_uses_the_newest_forecast_that_is_still_in_time():
    # h=1, d=10 min: L = max(0, 1 + 1 - 1) = 1 -> slot t uses ŷ[t-1]; slot 0 has none
    state = on(horizon=1, delay_min=10)
    assert state.tolist() == [False, False, True, False, False, False, False, True]
    assert metrics(state) == {"episodes": 2, "missed_min": 20, "missed_episodes": 2,
                              "on_min": 20, "switches": 2}  # fmt: skip
    # h=3 has 20 minutes to spare: L = max(0, 1 + 1 - 3) = 0 -> same as no delay
    assert (on(horizon=3, delay_min=10) == on(horizon=3)).all()


def test_imputed_slots_are_neither_wanted_nor_congestion():
    imputed = np.zeros(8, dtype=bool)
    imputed[1] = True
    state = on(imputed=imputed)
    assert not state[1]
    assert metrics(state, imputed)["missed_min"] == 10  # slot 2 only; slot 1 is not scored


def test_baselines():
    imputed = np.zeros(8, dtype=bool)
    always = metrics(switching.always_on(8))
    assert always == {"episodes": 2, "missed_min": 0, "missed_episodes": 0, "on_min": 80,
                      "switches": 1}  # fmt: skip
    oracle = metrics(switching.oracle(Y_TRUE, imputed, THR))
    assert oracle == {"episodes": 2, "missed_min": 0, "missed_episodes": 0, "on_min": 30,
                      "switches": 2}  # fmt: skip


# --- (2) H=0, 1.0θ, 지연 0이면 3단계 경보와 같음 ------------------------------------------------
@pytest.mark.parametrize("horizon", [1, 3, 6])
def test_no_hold_no_delay_equals_step3_alarm(horizon):
    rng = np.random.default_rng(horizon)
    pred = rng.uniform(0, 20, 300)
    imputed = rng.random(300) < 0.05
    expected = (pred >= THR) & ~imputed  # plan 3.3 결정 지표 경보
    assert (on(pred, horizon=horizon, imputed=imputed) == expected).all()


# --- (3) r1 선택 ---------------------------------------------------------------------------
def combo(hold, ratio, missed, on_min, switches):
    return [{"zone_rank": z, "hold_min": hold, "on_ratio": ratio, "missed_min": m,
             "on_min": o, "switches": s} for z, (m, o, s) in enumerate(zip(missed, on_min,
             switches, strict=True), start=1)]  # fmt: skip


def test_r1_no_worse_in_every_zone_then_least_on_time_then_switches():
    rows = (
        combo(0, 1.0, missed=[10, 0], on_min=[100, 100], switches=[5, 5])  # baseline
        + combo(0, 0.9, missed=[0, 10], on_min=[20, 30], switches=[1, 1])  # zone 2 worse
        + combo(30, 1.0, missed=[10, 0], on_min=[90, 100], switches=[2, 2])  # total 190, 4
        + combo(60, 0.8, missed=[0, 0], on_min=[100, 90], switches=[1, 1])  # total 190, 2
    )
    assert switching.choose_r1(rows) == (60, 0.8)


def test_r1_ties_go_to_smaller_hold_then_larger_ratio():
    rows = (
        combo(0, 1.0, missed=[0], on_min=[50], switches=[1])
        + combo(30, 0.9, missed=[0], on_min=[40], switches=[1])
        + combo(30, 1.0, missed=[0], on_min=[40], switches=[1])
        + combo(60, 1.0, missed=[0], on_min=[40], switches=[1])
    )
    assert switching.choose_r1(rows) == (30, 1.0)


def test_r1_keeps_the_baseline_when_nothing_is_better():
    rows = combo(0, 1.0, missed=[0], on_min=[50], switches=[3]) + combo(
        30, 0.9, missed=[10], on_min=[10], switches=[1]
    )
    assert switching.choose_r1(rows) == (0, 1.0)


# --- (6) D-14 switching 검증 (E-2001) ------------------------------------------------------
BASE = {"threshold_ratio": 0.7, "peak_quantile": 0.99, "merge_gap": 1, "report_horizon": 6,
        "switching": {"hold_min": [0, 30, 60, 120], "on_ratio": [1.0, 0.9, 0.8],
                      "delay_min": [0, 10]}}  # fmt: skip


def test_repo_switching_matches_plan():
    cfg = config.load_decision(config.ROOT / "configs" / "decision.yaml")
    assert (cfg.hold_min, cfg.on_ratio, cfg.delay_min) == ((0, 30, 60, 120), (1.0, 0.9, 0.8),
                                                           (0, 10))  # fmt: skip


@pytest.mark.parametrize(
    "switch",
    [
        None,  # key missing
        {"on_ratio": [1.0], "delay_min": [0]},  # hold_min missing
        {"hold_min": [30, 60], "on_ratio": [1.0], "delay_min": [0]},  # no 0
        {"hold_min": [0, 25], "on_ratio": [1.0], "delay_min": [0]},  # not a multiple of 10
        {"hold_min": [0, 250], "on_ratio": [1.0], "delay_min": [0]},
        {"hold_min": [0, 0], "on_ratio": [1.0], "delay_min": [0]},  # duplicate
        {"hold_min": [0], "on_ratio": [0.9], "delay_min": [0]},  # no 1.0
        {"hold_min": [0], "on_ratio": [1.0, 1.2], "delay_min": [0]},
        {"hold_min": [0], "on_ratio": [1.0, 0], "delay_min": [0]},
        {"hold_min": [0], "on_ratio": [1.0], "delay_min": [70]},
        {"hold_min": [0], "on_ratio": [1.0], "delay_min": []},
        {"hold_min": [0], "on_ratio": [1.0], "delay_min": [True]},
    ],
)
def test_e2001_bad_switching(tmp_path, switch):
    data = {k: v for k, v in BASE.items() if k != "switching"}
    if switch is not None:
        data["switching"] = switch
    path = tmp_path / "decision.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(TPError) as exc:
        config.load_decision(path)
    assert exc.value.code == "E-2001" and "결정 설정" in exc.value.message


# --- (4)(5) 결과 표: 재학습 없음, LSTM 시드 평균, (계열, 거리, 지연)마다 한 조합 ----
def test_results_writes_switching_tables_without_retraining(full_ws, monkeypatch):  # noqa: F811
    from tp.exp import registry
    from tp.models import arima, lstm, naive

    for module, name in ((registry, "_execute"), (naive, "predict_naive"),
                         (arima, "fit_arima"), (arima, "predict_arima"),
                         (lstm, "train_lstm"), (lstm, "predict_lstm")):  # fmt: skip
        monkeypatch.setattr(module, name, lambda *a, **k: pytest.fail("재학습·재예측함"))
    assert cli.main(["results"]) == 0
    table = pd.read_csv(config.RESULTS_DIR / "switching.csv")
    models = table[~table["family"].isin(["always_on", "oracle"])]
    assert set(models["family"]) == {"lag144", "lag1", "arima", "lstm"}
    assert len(models) == 4 * 1 * 2 * 12  # families x zones x delays x combos (h=1 only)
    chosen = models[models["chosen"]]
    assert len(chosen) == 4 * 2  # one per (family, horizon, delay), same for every zone
    assert chosen.groupby(["family", "horizon", "delay_min"]).size().eq(1).all()
    lstm_rows = models[models["family"] == "lstm"]
    assert lstm_rows["on_min_std"].notna().all()
    assert models.loc[models["family"] != "lstm", "on_min_std"].isna().all()
    base = table[table["family"].isin(["always_on", "oracle"])]
    assert set(base["family"]) == {"always_on", "oracle"} and (base["missed_min"] == 0).all()
    md = (config.RESULTS_DIR / "switching.md").read_text(encoding="utf-8")
    for text in ("지연 0분", "지연 10분", "항상 켬", "미래를 아는 경우", "t − h + 1"):
        assert text in md
