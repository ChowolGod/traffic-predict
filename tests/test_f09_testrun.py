import json

import numpy as np
import pandas as pd
import pytest
import yaml

from tp import cli, config
from tp.errors import TPError
from tp.eval.bootstrap import block_bootstrap_diff_ci
from tp.exp import registry, testrun
from tp.models import lstm

SMALL_LSTM = {
    "lstm.window": 12, "lstm.hidden": 8, "lstm.layers": 1, "lstm.dropout": 0.0, "lstm.lr": 1e-2,
    "lstm.batch": 64, "lstm.max_epochs": 3, "lstm.patience": 2, "lstm.scaler": "standard",
}  # fmt: skip
BASE = {"phase": "full", "reason": "r", "hypothesis": "h", "model": {"type": "naive"},
        "naive": {"lag": 144}}  # fmt: skip


# --- M-17 부트스트랩 (완료 조건 4) ------------------------------------------------------------
def test_bootstrap_identical_errors_give_zero():
    err = np.abs(np.random.default_rng(1).normal(size=144 * 7))
    days = np.repeat(np.arange(7), 144)
    assert block_bootstrap_diff_ci(err, err, days) == (0.0, 0.0, 0.0)


def test_bootstrap_same_seed_same_interval_and_contains_estimate():
    rng = np.random.default_rng(2)
    a = np.abs(rng.normal(size=144 * 7))
    b = np.abs(rng.normal(size=144 * 7)) + 0.3
    days = np.repeat(np.arange(7), 144)
    first = block_bootstrap_diff_ci(a, b, days, n=2000, seed=0)
    assert first == block_bootstrap_diff_ci(a, b, days, n=2000, seed=0)
    assert first != block_bootstrap_diff_ci(a, b, days, n=2000, seed=1)
    diff, low, high = first
    assert diff == pytest.approx(a.mean() - b.mean())
    assert low <= diff <= high < 0


# --- 준비: full 단계(train 11-04 / val 11-05 / test 11-06) + 계열별 실험 ----------------------
def fill_gap_on_1106(workspace):
    path = workspace / "raw" / "sms-call-internet-mi-2013-11-06.txt"
    rows = [line.split("\t") for line in path.read_text(encoding="ascii").splitlines()]
    base_ms = 1383692400000 + 79 * 600000  # 2013-11-06 slot 79 (Rome)
    extra = []
    for r in rows:
        if int(r[1]) == base_ms:
            for k in range(1, 5):
                extra.append("\t".join([r[0], str(base_ms + k * 600000), *r[2:]]))
    path.write_text("\n".join(["\t".join(r) for r in rows] + extra) + "\n", encoding="ascii")


def write_exp(cfg: dict) -> str:
    path = config.CONFIGS_DIR / "experiments" / f"{cfg['id']}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    return str(path)


def exp(exp_id, name, parent, changed, **fields):
    cfg = {**json.loads(json.dumps(BASE)), "id": exp_id, "name": name, "parent": parent,
           "changed": changed, **fields}  # fmt: skip
    if fields.get("model", {}).get("type") in ("arima", "lstm"):
        cfg.pop("naive")
    return cfg


def final_yaml(mapping: dict) -> str:
    path = config.CONFIGS_DIR / "final.yaml"
    path.write_text(yaml.safe_dump({"zones": mapping}), encoding="utf-8")
    return str(path)


FINAL = {1: {"naive_144": "EXP-001", "naive_1": "EXP-002", "arima": "EXP-003", "lstm": "EXP-004"}}


@pytest.fixture
def full_ws(workspace, monkeypatch):
    fill_gap_on_1106(workspace)
    phases = config.CONFIGS_DIR / "phases.yaml"
    data = yaml.safe_load(phases.read_text(encoding="utf-8"))
    data["full"] = {"train": ["2013-11-04", "2013-11-04"], "val": ["2013-11-05", "2013-11-05"],
                    "test": ["2013-11-06", "2013-11-06"], "k": 1}  # fmt: skip
    phases.write_text(yaml.safe_dump(data), encoding="utf-8")
    monkeypatch.setitem(registry.INITIAL, "lstm", SMALL_LSTM)
    assert cli.main(["prepare", "--phase", "full"]) == 0
    lstm_cfg = {k.split(".")[1]: v for k, v in SMALL_LSTM.items()}
    configs = [
        exp("EXP-001", "lag144", None, None),
        exp("EXP-002", "lag1", "EXP-001", "naive.lag", naive={"lag": 1}),
        exp("EXP-003", "arima", "EXP-001", "model.type", model={"type": "arima"},
            arima={"order": [2, 1, 2], "seasonal": "none"}),
        exp("EXP-004", "lstm", "EXP-001", "model.type", model={"type": "lstm"}, lstm=lstm_cfg),
    ]  # fmt: skip
    for cfg in configs:
        assert cli.main(["run", "--config", write_exp(cfg)]) == 0
    return workspace


def out_dir():
    return config.RESULTS_DIR / "test"


# --- 완료 조건 (3) 결과 표 ---------------------------------------------------------------
def test_happy_path_writes_results_and_lock(full_ws):
    assert cli.main(["test", "--final", final_yaml(FINAL), "--confirm"]) == 0
    table = pd.read_csv(out_dir() / "metrics.csv")
    assert list(table["family"]) == ["naive_144", "naive_1", "arima", "lstm"]
    for col in ("square_id", "mae", "mae_std", "rmse", "rmse_std", "rel_mae",
                "mae_diff_vs_lag144", "ci_low", "ci_high"):  # fmt: skip
        assert col in table.columns
    lag = table.set_index("family").loc["naive_144"]
    assert lag["rel_mae"] == pytest.approx(1.0) and lag["mae_diff_vs_lag144"] == 0.0
    assert (table["ci_low"] <= table["mae_diff_vs_lag144"] + 1e-12).all()
    assert (table["mae_diff_vs_lag144"] <= table["ci_high"] + 1e-12).all()
    assert not np.isnan(table.set_index("family").loc["lstm", "mae_std"])
    pred = pd.read_parquet(out_dir() / "predictions.parquet")
    assert list(pred.columns) == ["family", "zone_rank", "square_id", "time_utc", "y_true",
                                  "y_pred", "seed", "is_imputed"]  # fmt: skip
    assert pred["time_utc"].dt.tz_convert(config.TZ).dt.date.astype(str).unique().tolist() == [
        "2013-11-06"
    ]
    lock = json.loads((out_dir() / "LOCK").read_text(encoding="utf-8"))
    assert set(lock) == {"evaluated_at", "final", "git_commit"}
    assert "선택용" in (out_dir() / "metrics.md").read_text(encoding="utf-8")


# --- 완료 조건 (1) 두 번째 실행은 거부 ---------------------------------------------------------
def test_second_run_e4005(full_ws):
    assert cli.main(["test", "--final", final_yaml(FINAL), "--confirm"]) == 0
    with pytest.raises(TPError) as exc:
        testrun.run_test(final_yaml(FINAL), confirm=True)
    assert exc.value.code == "E-4005"


def test_experiments_after_lock_are_post_test(full_ws):
    assert cli.main(["test", "--final", final_yaml(FINAL), "--confirm"]) == 0
    cfg = exp("EXP-005", "arima-111", "EXP-003", "arima.order", model={"type": "arima"},
              arima={"order": [1, 1, 1], "seasonal": "none"})  # fmt: skip
    assert cli.main(["run", "--config", write_exp(cfg)]) == 0
    folder = next(config.EXPERIMENTS_DIR.glob("EXP-005_*"))
    assert json.loads((folder / "meta.json").read_text(encoding="utf-8"))["post_test"] is True


# --- 완료 조건 (2) --confirm 없음·dirty 거부, 아무것도 쓰지 않음 ---------------------------
def test_requires_confirm(full_ws):
    assert cli.main(["test", "--final", final_yaml(FINAL)]) == 1
    assert not out_dir().exists()


def test_dirty_rejected(full_ws, monkeypatch):
    monkeypatch.setattr(registry, "git_state", lambda: ("abc1234", True))
    with pytest.raises(TPError) as exc:
        testrun.run_test(final_yaml(FINAL), confirm=True)
    assert exc.value.code == "E-4006"
    assert not out_dir().exists()


# --- E-4006 최종 설정 검증 --------------------------------------------------------------------
@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda f: f[1].pop("lstm"), id="missing-family"),
        pytest.param(lambda f: f[1].update(arima="EXP-002"), id="family-type-mismatch"),
        pytest.param(lambda f: f[1].update(naive_1="EXP-001"), id="lag-mismatch"),
        pytest.param(lambda f: f[1].update(lstm="EXP-099"), id="unknown-experiment"),
        pytest.param(lambda f: f.update({2: dict(f[1])}), id="zone-count-not-k"),
    ],
)
def test_e4006_final_config(full_ws, mutate):
    final = json.loads(json.dumps(FINAL))
    final = {int(k): v for k, v in final.items()}
    mutate(final)
    with pytest.raises(TPError) as exc:
        testrun.run_test(final_yaml(final), confirm=True)
    assert exc.value.code == "E-4006"
    assert not out_dir().exists()


def test_e4006_not_selection_rule_best(full_ws):
    cfg = exp("EXP-005", "arima-111", "EXP-003", "arima.order", model={"type": "arima"},
              arima={"order": [1, 1, 1], "seasonal": "none"})  # fmt: skip
    assert cli.main(["run", "--config", write_exp(cfg)]) == 0

    def val_mae(exp_id):
        folder = next(config.EXPERIMENTS_DIR.glob(f"{exp_id}_*"))
        return json.loads((folder / "metrics.json").read_text(encoding="utf-8"))["val"]["mae"]

    worse = "EXP-003" if val_mae("EXP-003") > val_mae("EXP-005") else "EXP-005"
    better = "EXP-005" if worse == "EXP-003" else "EXP-003"
    with pytest.raises(TPError) as exc:
        testrun.run_test(final_yaml({1: {**FINAL[1], "arima": worse}}), confirm=True)
    assert exc.value.code == "E-4006" and better in exc.value.message
    assert cli.main(["test", "--final", final_yaml({1: {**FINAL[1], "arima": better}}),
                     "--confirm"]) == 0  # fmt: skip


def test_e4006_missing_model_file(full_ws):
    folder = next(config.EXPERIMENTS_DIR.glob("EXP-003_*"))
    (folder / "model" / "arima_params.json").unlink()
    with pytest.raises(TPError) as exc:
        testrun.run_test(final_yaml(FINAL), confirm=True)
    assert exc.value.code == "E-4006"


def test_e4006_dev_experiment(full_ws):
    root = {**json.loads(json.dumps(BASE)), "id": "EXP-005", "name": "dev-root", "phase": "dev",
            "parent": None, "changed": None}  # fmt: skip
    assert cli.main(["prepare", "--phase", "dev"]) == 0
    assert cli.main(["run", "--config", write_exp(root)]) == 0
    with pytest.raises(TPError) as exc:
        testrun.run_test(final_yaml({1: {**FINAL[1], "naive_144": "EXP-005"}}), confirm=True)
    assert exc.value.code == "E-4006"


# --- 원자성: 계산 중 실패하면 아무것도 쓰지 않음 ---------------------------------------------
def test_failure_during_prediction_writes_nothing(full_ws, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(lstm, "predict_lstm", boom)
    with pytest.raises(RuntimeError):
        testrun.run_test(final_yaml(FINAL), confirm=True)
    assert not out_dir().exists()
    assert not (config.RESULTS_DIR / ".test_tmp").exists()


# --- 재학습 없이 저장된 모델을 사용 ------------------------------------------------------------
def test_no_training_during_test(full_ws, monkeypatch):
    from tp.models import arima

    def forbidden(*a, **k):
        raise AssertionError("test 중 학습함")

    monkeypatch.setattr(lstm, "train_lstm", forbidden)
    monkeypatch.setattr(arima, "fit_arima", forbidden)
    assert cli.main(["test", "--final", final_yaml(FINAL), "--confirm"]) == 0
