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


H1 = {"naive_144": "EXP-001", "naive_1": "EXP-002", "arima": "EXP-003", "lstm": "EXP-004"}
H3 = {"naive_144": "EXP-005", "naive_1": "EXP-006", "arima": "EXP-007", "lstm": "EXP-008"}
FINAL = {1: {1: H1, 3: H3}}
FAMILY_ORDER = ["naive_144", "naive_1", "arima", "lstm"]


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


def _config_of(exp_id: str) -> dict:
    path = config.CONFIGS_DIR / "experiments" / f"{exp_id}.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


@pytest.fixture
def v2_ws(full_ws, monkeypatch):
    """full_ws plus horizon-3 children. The 3-day fixture cannot hold lag 1008 or horizon 6,
    so test uses horizons (1, 3) and four families here; the real constants are checked below."""
    for family, parent in H1.items():
        child = _config_of(parent)
        child.update(id=H3[family], name=f"{family.replace('_', '')}-h3", parent=parent,
                     changed="horizon", horizon=3, reason="r", hypothesis="h")  # fmt: skip
        assert cli.main(["run", "--config", write_exp(child)]) == 0
    monkeypatch.setattr(testrun, "HORIZONS", (1, 3))
    four = {k: v for k, v in testrun.FAMILIES.items() if k != "naive_1008"}
    monkeypatch.setattr(testrun, "FAMILIES", four)
    return full_ws


def out_dir():
    return config.RESULTS_DIR / "test"


def run_ok(final=FINAL) -> pd.DataFrame:
    assert cli.main(["test", "--final", final_yaml(final), "--confirm"]) == 0
    return pd.read_csv(out_dir() / "metrics.csv")


# --- 계획 상수 (v2.12: 거리 10·30·60분, 계열 5개) ------------------------------------------------
def test_repo_constants_match_plan():
    assert testrun.HORIZONS == (1, 3, 6)
    assert list(testrun.FAMILIES) == ["naive_144", "naive_1", "naive_1008", "arima", "lstm"]


# --- 완료 조건 (3)(7) 결과 표 -------------------------------------------------------------------
def test_happy_path_writes_results_and_lock(v2_ws):
    table = run_ok()
    assert list(table.columns) == testrun.COLUMNS
    assert list(zip(table["horizon"], table["family"], strict=True)) == [
        (h, f) for h in (1, 3) for f in FAMILY_ORDER
    ]
    by = table.set_index(["horizon", "family"])
    for h in (1, 3):
        assert by.loc[(h, "naive_144"), "rel_mae"] == pytest.approx(1.0)
        assert by.loc[(h, "naive_144"), "diff_vs_lag144"] == 0.0
        assert by.loc[(h, "naive_1"), "diff_vs_lag1"] == 0.0
        assert not np.isnan(by.loc[(h, "lstm"), "mae_std"])
        assert not np.isnan(by.loc[(h, "lstm"), "dec_missed_std"])
    for ref in ("lag144", "lag1"):
        assert (table[f"ci_{ref}_low"] <= table[f"diff_vs_{ref}"] + 1e-12).all()
        assert (table[f"diff_vs_{ref}"] <= table[f"ci_{ref}_high"] + 1e-12).all()
    assert (table["dec_episodes"] >= 0).all() and table["dec_threshold"].nunique() == 1
    pred = pd.read_parquet(out_dir() / "predictions.parquet")
    assert list(pred.columns) == ["family", "zone_rank", "square_id", "horizon", "time_utc",
                                  "y_true", "y_pred", "seed", "is_imputed"]  # fmt: skip
    assert pred["time_utc"].dt.tz_convert(config.TZ).dt.date.astype(str).unique().tolist() == [
        "2013-11-06"
    ]
    lock = json.loads((out_dir() / "LOCK").read_text(encoding="utf-8"))
    assert set(lock) == {"evaluated_at", "final", "git_commit"}
    assert lock["final"] == {"1": {"1": H1, "3": H3}}
    md = (out_dir() / "metrics.md").read_text(encoding="utf-8")
    for text in ("선택용", "어제 같은 시각", "마지막 관측값", "지연 0분", "미래를 아는 경우"):
        assert text in md


# --- 완료 조건 (6) LSTM 구간 = 시드별 절대오차의 칸별 평균 ---------------------------------------
def test_lstm_interval_uses_mean_of_per_seed_absolute_errors(v2_ws):
    table = run_ok().set_index(["horizon", "family"])
    pred = pd.read_parquet(out_dir() / "predictions.parquet")
    pred = pred[(pred["horizon"] == 1) & ~pred["is_imputed"]]

    def err(family):
        p = pred[pred["family"] == family].assign(e=lambda d: (d["y_true"] - d["y_pred"]).abs())
        return p.groupby("time_utc")["e"].mean()

    lstm_err, ref = err("lstm"), err("naive_144")
    days = np.array(ref.index.tz_convert(config.TZ).date, dtype=object)
    expected = block_bootstrap_diff_ci(lstm_err.loc[ref.index].to_numpy(), ref.to_numpy(), days)
    row = table.loc[(1, "lstm")]
    got = (row["diff_vs_lag144"], row["ci_lag144_low"], row["ci_lag144_high"])
    assert got == pytest.approx(expected)
    assert row["mae"] == pytest.approx(lstm_err.mean())  # same quantity as the reported MAE


# --- 완료 조건 (8) 켜기/끄기: val에서 고른 조합을 그대로 적용 -----------------------------
def test_switching_applies_the_val_choice_without_reselecting(v2_ws):
    assert cli.main(["results"]) == 0
    val = pd.read_csv(config.RESULTS_DIR / "switching.csv")
    chosen = val[val["chosen"]].groupby(["family", "horizon", "delay_min"])
    val_choice = chosen[["hold_min", "on_ratio"]].first()
    run_ok()
    test = pd.read_csv(out_dir() / "switching.csv")
    models = test[~test["family"].isin(["always_on", "oracle"])]
    names = {v: k for k, v in testrun.FAMILIES.items()}
    assert len(models) == len(val_choice)  # one zone: one row per (family, horizon, delay)
    for (family, h, delay), r in val_choice.iterrows():
        row = models[(models["family"] == names[family]) & (models["horizon"] == h)
                     & (models["delay_min"] == delay)]  # fmt: skip
        assert (row["hold_min"].iloc[0], row["on_ratio"].iloc[0]) == (r["hold_min"], r["on_ratio"])
    assert {"always_on", "oracle"} <= set(test["family"])


# --- 완료 조건 (1) 두 번째 실행은 거부 ---------------------------------------------------------
def test_second_run_e4005(v2_ws):
    run_ok()
    with pytest.raises(TPError) as exc:
        testrun.run_test(final_yaml(FINAL), confirm=True)
    assert exc.value.code == "E-4005"


def test_experiments_after_lock_are_post_test(v2_ws):
    run_ok()
    cfg = exp("EXP-009", "arima-111", "EXP-003", "arima.order", model={"type": "arima"},
              arima={"order": [1, 1, 1], "seasonal": "none"})  # fmt: skip
    assert cli.main(["run", "--config", write_exp(cfg)]) == 0
    folder = next(config.EXPERIMENTS_DIR.glob("EXP-009_*"))
    assert json.loads((folder / "meta.json").read_text(encoding="utf-8"))["post_test"] is True


# --- 완료 조건 (2) --confirm 없음·dirty 거부, 아무것도 쓰지 않음 ---------------------------
def test_requires_confirm(v2_ws):
    assert cli.main(["test", "--final", final_yaml(FINAL)]) == 1
    assert not out_dir().exists()


def test_dirty_rejected(v2_ws, monkeypatch):
    monkeypatch.setattr(registry, "git_state", lambda: ("abc1234", True))
    with pytest.raises(TPError) as exc:
        testrun.run_test(final_yaml(FINAL), confirm=True)
    assert exc.value.code == "E-4006"
    assert not out_dir().exists()


# --- E-4006 최종 설정 검증 (완료 조건 5 포함) ----------------------------------------------------
@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda f: f[1][1].pop("lstm"), id="missing-family"),
        pytest.param(lambda f: f[1].pop(3), id="missing-horizon"),
        pytest.param(lambda f: f[1][1].update(arima="EXP-002"), id="family-type-mismatch"),
        pytest.param(lambda f: f[1][1].update(naive_1="EXP-001"), id="lag-mismatch"),
        pytest.param(lambda f: f[1][1].update(lstm="EXP-099"), id="unknown-experiment"),
        pytest.param(lambda f: f[1][3].update(arima="EXP-003"), id="horizon-mismatch"),
        pytest.param(lambda f: f.update({2: f[1]}), id="zone-count-not-k"),
    ],
)
def test_e4006_final_config(v2_ws, mutate):
    final = {1: {1: dict(H1), 3: dict(H3)}}
    mutate(final)
    with pytest.raises(TPError) as exc:
        testrun.run_test(final_yaml(final), confirm=True)
    assert exc.value.code == "E-4006"
    assert not out_dir().exists()


def test_e4006_not_selection_rule_best(v2_ws):
    cfg = exp("EXP-009", "arima-111", "EXP-003", "arima.order", model={"type": "arima"},
              arima={"order": [1, 1, 1], "seasonal": "none"})  # fmt: skip
    assert cli.main(["run", "--config", write_exp(cfg)]) == 0

    def val_mae(exp_id):
        folder = next(config.EXPERIMENTS_DIR.glob(f"{exp_id}_*"))
        return json.loads((folder / "metrics.json").read_text(encoding="utf-8"))["val"]["mae"]

    worse = "EXP-003" if val_mae("EXP-003") > val_mae("EXP-009") else "EXP-009"
    better = "EXP-009" if worse == "EXP-003" else "EXP-003"
    with pytest.raises(TPError) as exc:
        testrun.run_test(final_yaml({1: {1: {**H1, "arima": worse}, 3: H3}}), confirm=True)
    assert exc.value.code == "E-4006" and better in exc.value.message
    run_ok({1: {1: {**H1, "arima": better}, 3: H3}})


def test_e4006_missing_model_file(v2_ws):
    folder = next(config.EXPERIMENTS_DIR.glob("EXP-007_*"))
    (folder / "model" / "arima_params.json").unlink()
    with pytest.raises(TPError) as exc:
        testrun.run_test(final_yaml(FINAL), confirm=True)
    assert exc.value.code == "E-4006"


def test_e4006_dev_experiment(v2_ws):
    root = {**json.loads(json.dumps(BASE)), "id": "EXP-009", "name": "dev-root", "phase": "dev",
            "parent": None, "changed": None}  # fmt: skip
    assert cli.main(["prepare", "--phase", "dev"]) == 0
    assert cli.main(["run", "--config", write_exp(root)]) == 0
    final = {1: {1: {**H1, "naive_144": "EXP-009"}, 3: H3}}
    with pytest.raises(TPError) as exc:
        testrun.run_test(final_yaml(final), confirm=True)
    assert exc.value.code == "E-4006"


def test_e2001_bad_decision_config_writes_nothing(v2_ws):
    (config.CONFIGS_DIR / "decision.yaml").write_text("threshold_ratio: 2\n", encoding="utf-8")
    with pytest.raises(TPError) as exc:
        testrun.run_test(final_yaml(FINAL), confirm=True)
    assert exc.value.code == "E-2001"
    assert not out_dir().exists()


# --- 원자성: 계산 중 실패하면 아무것도 쓰지 않음 ---------------------------------------------
def test_failure_during_prediction_writes_nothing(v2_ws, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(lstm, "predict_lstm", boom)
    with pytest.raises(RuntimeError):
        testrun.run_test(final_yaml(FINAL), confirm=True)
    assert not out_dir().exists()
    assert not (config.RESULTS_DIR / ".test_tmp").exists()


# --- 재학습 없이 저장된 모델을 사용 ------------------------------------------------------------
def test_no_training_during_test(v2_ws, monkeypatch):
    from tp.models import arima

    def forbidden(*a, **k):
        raise AssertionError("test 중 학습함")

    monkeypatch.setattr(lstm, "train_lstm", forbidden)
    monkeypatch.setattr(arima, "fit_arima", forbidden)
    run_ok()


# --- F-11: K=3이면 구역별 행과 (계열, 거리)마다 평균 행 ------------------------------------------
def test_f11_three_zones_rows_and_mean(v2_ws):
    phases = config.CONFIGS_DIR / "phases.yaml"
    data = yaml.safe_load(phases.read_text(encoding="utf-8"))
    data["full"]["k"] = 3
    phases.write_text(yaml.safe_dump(data), encoding="utf-8")
    assert cli.main(["prepare", "--phase", "full"]) == 0
    next_id = 9
    final = {1: {1: dict(H1), 3: dict(H3)}, 2: {1: {}, 3: {}}, 3: {1: {}, 3: {}}}
    for h, mapping in ((1, H1), (3, H3)):
        for family, parent in mapping.items():
            start = f"EXP-{next_id:03d}"
            name = f"z-{family.replace('_', '')}-h{h}"
            registry.sweep(parent, "zone_rank", "[2, 3]", start, name, "r", "h")
            final[2][h][family], final[3][h][family] = start, f"EXP-{next_id + 1:03d}"
            next_id += 2
    table = run_ok(final)
    zone_rows = table[table["zone_rank"] != "mean"]
    assert sorted(zone_rows["zone_rank"].astype(int).unique()) == [1, 2, 3]
    assert zone_rows["square_id"].nunique() == 3 and len(zone_rows) == 3 * 2 * 4
    means = table[table["zone_rank"] == "mean"].set_index(["horizon", "family"])["rel_mae"]
    expected = zone_rows.groupby(["horizon", "family"])["rel_mae"].mean()
    assert len(means) == 2 * 4
    for key in means.index:
        assert means[key] == pytest.approx(expected[key])
    assert means[(1, "naive_144")] == pytest.approx(1.0)
    switch = pd.read_csv(out_dir() / "switching.csv")
    assert sorted(switch["zone_rank"].unique()) == [1, 2, 3]
