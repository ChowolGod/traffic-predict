import numpy as np
import pandas as pd
import pytest
import yaml

from tests.test_f09_testrun import full_ws  # noqa: F401  (fixture)
from tp import cli, config
from tp.errors import TPError
from tp.eval import decision

Y_TRUE = np.array([0, 12, 12, 0, 12, 0, 0, 15, 15, 15, 0, 0], dtype=float)


def pred_with_alarms(alarms: list[int], n: int = len(Y_TRUE)) -> np.ndarray:
    y = np.zeros(n)
    y[alarms] = 11.0
    return y


def metrics(y_true, y_pred, horizon=3, merge_gap=1, imputed=None):
    imputed = np.zeros(len(y_true), dtype=bool) if imputed is None else imputed
    return decision.decision_metrics(y_true, y_pred, imputed, 10.0, horizon, merge_gap)


# --- (1) 손계산 예제 ------------------------------------------------------------------------
def test_hand_example():
    out = metrics(Y_TRUE, pred_with_alarms([0, 3, 10, 11]))
    assert out == {"episodes": 2, "missed": 1, "false_alarm_min": 30, "lead_min": 10.0}


def test_merge_gap_changes_episode_count():
    assert metrics(Y_TRUE, pred_with_alarms([]), merge_gap=0)["episodes"] == 3
    assert metrics(Y_TRUE, pred_with_alarms([]), merge_gap=1)["episodes"] == 2


def test_only_alarms_issued_before_start_count():
    # episode [7, 9], h=1: an alarm for τ=9 is issued at 8 > 7 → too late → missed
    late = metrics(Y_TRUE, pred_with_alarms([9]), horizon=1)
    assert late["missed"] == 2 and late["lead_min"] is None
    # h=3: τ=8 is issued at 5 ≤ 7 → detected with lead (7 - 5) = 2 slots
    early = metrics(Y_TRUE, pred_with_alarms([8]), horizon=3)
    assert early["missed"] == 1 and early["lead_min"] == 20.0


def test_earliest_alarm_gives_the_lead():
    out = metrics(Y_TRUE, pred_with_alarms([7, 8, 9]), horizon=3)  # τ=7 issued at 4
    assert out["lead_min"] == 30.0


def test_imputed_slots_are_neither_congestion_nor_alarm():
    imputed = np.zeros(len(Y_TRUE), dtype=bool)
    imputed[[7, 8, 9, 11]] = True
    out = metrics(Y_TRUE, pred_with_alarms([11]), imputed=imputed)
    assert out["episodes"] == 1 and out["false_alarm_min"] == 0


def test_no_episodes():
    out = metrics(np.zeros(5), pred_with_alarms([2], n=5))
    assert out == {"episodes": 0, "missed": 0, "false_alarm_min": 10, "lead_min": None}


def test_threshold_is_ratio_of_train_quantile():
    train = np.arange(1, 101, dtype=float)
    cfg = decision.DecisionConfig(threshold_ratio=0.7, peak_quantile=0.99, merge_gap=1,
                                  report_horizon=6)  # fmt: skip
    assert decision.threshold(train, cfg) == pytest.approx(0.7 * np.quantile(train, 0.99))


# --- D-14 설정 검증 (E-2001) ---------------------------------------------------------------
def test_repo_decision_yaml_matches_plan():
    cfg = config.load_decision(config.ROOT / "configs" / "decision.yaml")
    assert (cfg.threshold_ratio, cfg.peak_quantile, cfg.merge_gap, cfg.report_horizon) == (
        0.7, 0.99, 1, 6,
    )  # fmt: skip


@pytest.mark.parametrize(
    "change",
    [
        {"threshold_ratio": 1.5},
        {"threshold_ratio": 0},
        {"peak_quantile": 0.3},
        {"merge_gap": 7},
        {"merge_gap": 1.5},
        {"report_horizon": 2},
        {"threshold_ratio": None},
    ],
)
def test_e2001_bad_decision_config(tmp_path, change):
    data = {"threshold_ratio": 0.7, "peak_quantile": 0.99, "merge_gap": 1, "report_horizon": 6}
    data.update(change)
    data = {k: v for k, v in data.items() if v is not None}
    path = tmp_path / "decision.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(TPError) as exc:
        config.load_decision(path)
    assert exc.value.code == "E-2001" and "결정 설정" in exc.value.message


# --- (2)(3)(4) 결과 표 연동: 재학습 없이 설정 변경 반영, LSTM은 시드 평균 ---------------------
def test_results_table_has_decision_columns_and_follows_config(workspace, monkeypatch):
    from tests.test_f08_registry import ROOT_CFG, child, run

    assert cli.main(["prepare", "--phase", "dev"]) == 0
    assert run(ROOT_CFG) == 0
    assert run(child(changed="naive.lag", naive={"lag": 1})) == 0
    table = pd.read_csv(config.RESULTS_DIR / "results.csv").set_index("exp_id")
    for col in ("horizon", "dec_threshold", "dec_episodes", "dec_missed", "dec_false_alarm_min",
                "dec_lead_min"):  # fmt: skip
        assert col in table.columns
    assert (table["horizon"] == 1).all()
    before = table.loc["EXP-002", "dec_threshold"]

    path = config.CONFIGS_DIR / "decision.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    path.write_text(yaml.safe_dump({**data, "threshold_ratio": 0.35}), encoding="utf-8")
    from tp.exp import registry

    monkeypatch.setattr(registry, "_execute", lambda *a, **k: pytest.fail("재학습함"))
    assert cli.main(["results"]) == 0
    after = pd.read_csv(config.RESULTS_DIR / "results.csv").set_index("exp_id")
    assert after.loc["EXP-002", "dec_threshold"] == pytest.approx(before / 2)


def test_lstm_decision_uses_seed_mean():
    from tp.exp import results

    t = pd.date_range("2013-11-14", periods=4, freq="10min", tz="UTC")
    pred = pd.DataFrame({
        "time_utc": np.tile(t, 2), "segment": "val", "y_true": [0.0, 12, 0, 0] * 2,
        "y_pred": [0.0, 14, 0, 0, 0, 7, 0, 0], "seed": [0] * 4 + [1] * 4, "is_imputed": False,
    })  # fmt: skip
    mean = results.seed_mean(pred)
    assert list(mean["y_pred"]) == [0.0, 10.5, 0.0, 0.0]  # seed mean 10.5 ≥ 10 → alarm


# --- D-15 거리별 표·그래프 (F-12 완료 조건 6) -------------------------------------------------
def test_horizon_table_and_figure(full_ws):  # noqa: F811  (fixture from test_f09)
    from tests.test_f09_testrun import exp, write_exp

    h3 = exp("EXP-005", "lag1-h3", "EXP-002", "horizon", naive={"lag": 1}, horizon=3)
    h6 = exp("EXP-006", "lag1-h6", "EXP-002", "horizon", naive={"lag": 1}, horizon=6)
    for cfg in (h3, h6):
        assert cli.main(["run", "--config", write_exp(cfg)]) == 0
    assert cli.main(["results"]) == 0
    md = (config.RESULTS_DIR / "horizon.md").read_text(encoding="utf-8")
    assert "EXP-002" in md and "EXP-005" in md and "EXP-006" in md
    for header in ("10분", "30분", "60분", "놓친 혼잡", "불필요 경보", "리드타임"):
        assert header in md
    png = config.RESULTS_DIR / "figures" / "horizon_mae.png"
    assert png.is_file() and png.stat().st_size > 5000
    table = pd.read_csv(config.RESULTS_DIR / "results.csv").set_index("exp_id")
    assert table.loc["EXP-006", "horizon"] == 6
    assert table.loc["EXP-006", "dec_episodes"] >= 0
