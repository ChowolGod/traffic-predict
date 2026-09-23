import json
import shutil
import subprocess
import sys

import pandas as pd
import pytest
import yaml

from tp import cli, config
from tp.errors import TPError
from tp.exp import registry, results
from tp.models import naive
from tp.prep import series

ROOT_CFG = {
    "id": "EXP-001",
    "name": "lag144",
    "phase": "dev",
    "parent": None,
    "changed": None,
    "reason": "공식 기준선",
    "hypothesis": "하루 주기가 강하면 전날 같은 시각이 좋은 예측",
    "model": {"type": "naive"},
    "naive": {"lag": 144},
}


@pytest.fixture
def prepared(workspace):
    assert cli.main(["prepare", "--phase", "dev"]) == 0
    return workspace


def write_cfg(cfg: dict) -> str:
    path = config.CONFIGS_DIR / "experiments" / f"{cfg['id']}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    return str(path)


def child(**changes) -> dict:
    cfg = json.loads(json.dumps(ROOT_CFG))
    cfg.update(id="EXP-002", name="child", parent="EXP-001", reason="r", hypothesis="h")
    for key, value in changes.items():
        cfg[key] = value
    return cfg


def run(cfg: dict, retry: bool = False) -> int:
    args = ["run", "--config", write_cfg(cfg)] + (["--retry"] if retry else [])
    return cli.main(args)


def run_err(cfg: dict, retry: bool = False) -> str:
    with pytest.raises(TPError) as exc:
        registry.run_from_file(write_cfg(cfg), retry=retry)
    return exc.value.code


def folder(exp_id: str):
    return next(config.EXPERIMENTS_DIR.glob(f"{exp_id}_*"))


def meta(exp_id: str) -> dict:
    return json.loads((folder(exp_id) / "meta.json").read_text(encoding="utf-8"))


def top_series():
    return series.load_series(config.load_phase("dev"), 5000)


# --- F-05 기준선 ------------------------------------------------------------------------
@pytest.mark.parametrize("lag", [1, 144])
def test_naive_prediction_is_exact_lag(prepared, lag):
    s = top_series()
    targets = pd.DatetimeIndex(s["time_utc"].iloc[lag:])
    pred = naive.predict_naive(s, lag, targets)
    assert (pred.to_numpy() == s["y"].to_numpy()[:-lag]).all()


def test_naive_rejects_target_without_history(prepared):
    s = top_series()
    with pytest.raises(TPError) as exc:
        naive.predict_naive(s, 144, pd.DatetimeIndex(s["time_utc"].iloc[:5]))
    assert exc.value.code == "E-4004"


# --- F-08 완료 조건 (1) 실험 폴더 산출물 --------------------------------------------------
def test_root_run_writes_all_outputs(prepared):
    assert run(ROOT_CFG) == 0
    f = folder("EXP-001")
    assert f.name == "EXP-001_lag144"
    for name in ("config.yaml", "meta.json", "metrics.json", "predictions.parquet", "run.log"):
        assert (f / name).is_file(), name
    m = meta("EXP-001")
    assert m["status"] == "completed" and m["error_code"] is None
    assert m["git_commit"] == "abc1234" and m["git_dirty"] is False
    assert m["square_id"] == 5000 and len(m["compare_group"]) == 8
    cfg = yaml.safe_load((f / "config.yaml").read_text(encoding="utf-8"))
    assert cfg["runtime"] == {"time_limit_min": 30, "threads": 4}  # 기본값을 풀어 씀
    assert cfg["zone_rank"] == 1
    metrics = json.loads((f / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["val"]["n"] == 144 and metrics["val"]["rel_mae"] == pytest.approx(1.0)
    pred = pd.read_parquet(f / "predictions.parquet")
    assert set(pred["segment"]) <= {"train", "val"} and (pred["seed"] == -1).all()


def test_metrics_reproducible(prepared):
    assert run(ROOT_CFG) == 0
    first = (folder("EXP-001") / "metrics.json").read_text(encoding="utf-8")
    shutil.rmtree(folder("EXP-001"))
    assert run(ROOT_CFG) == 0
    assert (folder("EXP-001") / "metrics.json").read_text(encoding="utf-8") == first


# --- F-08 완료 조건 (2) 한 요소 규칙 -----------------------------------------------------
def test_one_change_child_runs(prepared):
    assert run(ROOT_CFG) == 0
    assert run(child(changed="naive.lag", naive={"lag": 1})) == 0
    assert meta("EXP-002")["changed"] == "naive.lag"


@pytest.mark.parametrize(
    "cfg",
    [
        pytest.param(child(changed="naive.lag"), id="zero-changes"),
        pytest.param(
            child(changed="naive.lag", naive={"lag": 1}, runtime={"threads": 2}), id="two-changes"
        ),
        pytest.param(child(changed="runtime.threads", naive={"lag": 1}), id="changed-mismatch"),
    ],
)
def test_e4001_change_count(prepared, cfg):
    assert run(ROOT_CFG) == 0
    assert run_err(cfg) == "E-4001"
    assert not list(config.EXPERIMENTS_DIR.glob("EXP-002_*"))  # 폴더를 만들지 않음


def test_time_limit_change_does_not_count(prepared):
    assert run(ROOT_CFG) == 0
    cfg = child(changed="naive.lag", naive={"lag": 1}, runtime={"time_limit_min": 10})
    assert run(cfg) == 0


def test_family_switch_requires_initial_values(prepared):
    assert run(ROOT_CFG) == 0
    bad = child(changed="model.type", model={"type": "arima"},
                arima={"order": [1, 1, 1], "seasonal": "none"})  # fmt: skip
    del bad["naive"]
    assert run_err(bad) == "E-4001"


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda c: c.update(naive={"lag": 1}), id="root-not-lag144"),
        pytest.param(lambda c: c.update(zone_rank=2), id="zone-rank-over-k"),
        pytest.param(lambda c: c.update(id="EXP-002"), id="id-not-max-plus-1"),
        pytest.param(lambda c: c.update(name="Bad Name"), id="name"),
        pytest.param(lambda c: c.update(reason="  "), id="empty-reason"),
        pytest.param(lambda c: c.update(naive={"lag": 7}), id="lag-range"),
        pytest.param(lambda c: c.update(typo_key=1), id="unknown-key"),
        pytest.param(lambda c: c.update(runtime={"threads": 9}), id="threads-range"),
    ],
)
def test_e4001_root_and_schema(prepared, mutate):
    cfg = json.loads(json.dumps(ROOT_CFG))
    mutate(cfg)
    assert run_err(cfg) == "E-4001"


def test_e4001_second_root(prepared):
    assert run(ROOT_CFG) == 0
    assert run_err({**ROOT_CFG, "id": "EXP-002", "name": "again"}) == "E-4001"


def test_e4001_full_phase_dirty(prepared, monkeypatch):
    phases = config.CONFIGS_DIR / "phases.yaml"
    data = yaml.safe_load(phases.read_text(encoding="utf-8"))
    data["full"] = {"train": ["2013-11-04", "2013-11-04"], "val": ["2013-11-05", "2013-11-05"],
                    "test": ["2013-11-06", "2013-11-06"], "k": 1}  # fmt: skip
    phases.write_text(yaml.safe_dump(data), encoding="utf-8")
    monkeypatch.setattr(registry, "git_state", lambda: ("abc1234", True))
    assert run_err({**ROOT_CFG, "phase": "full"}) == "E-4001"


def test_dev_dirty_is_recorded(prepared, monkeypatch):
    monkeypatch.setattr(registry, "git_state", lambda: ("abc1234", True))
    assert run(ROOT_CFG) == 0
    assert meta("EXP-001")["git_dirty"] is True


# --- E-4002 부모 ------------------------------------------------------------------------
def test_e4002_missing_parent(prepared):
    assert run(ROOT_CFG) == 0
    assert run_err(child(parent="EXP-009", changed="naive.lag", naive={"lag": 1})) == "E-4002"


def test_e4002_failed_parent(prepared, monkeypatch):
    assert run(ROOT_CFG) == 0
    m = meta("EXP-001")
    m["status"] = "failed"
    (folder("EXP-001") / "meta.json").write_text(json.dumps(m), encoding="utf-8")
    assert run_err(child(changed="naive.lag", naive={"lag": 1})) == "E-4002"


# --- E-4003 재실행 ----------------------------------------------------------------------
def test_e4003_existing_id_without_retry(prepared):
    assert run(ROOT_CFG) == 0
    assert run_err(ROOT_CFG) == "E-4003"
    assert run_err(ROOT_CFG, retry=True) == "E-4003"  # 완료된 실험


def test_retry_failed_same_config_and_reject_changed_config(prepared, monkeypatch):
    def boom(*a, **k):
        raise TPError("E-3003", "시간 초과: test")

    original = naive.predict_naive
    monkeypatch.setattr(naive, "predict_naive", boom)
    assert run(ROOT_CFG) == 1
    assert meta("EXP-001")["status"] == "failed" and meta("EXP-001")["error_code"] == "E-3003"
    monkeypatch.setattr(naive, "predict_naive", original)
    assert run_err({**ROOT_CFG, "reason": "바꿈"}, retry=True) == "E-4003"
    assert run(ROOT_CFG, retry=True) == 0
    assert meta("EXP-001")["status"] == "completed"


def test_retry_interrupted_running(prepared):
    assert run(ROOT_CFG) == 0
    m = meta("EXP-001")
    m["status"] = "running"
    (folder("EXP-001") / "meta.json").write_text(json.dumps(m), encoding="utf-8")
    assert run(ROOT_CFG, retry=True) == 0


# --- E-3003 시간 초과 --------------------------------------------------------------------
def test_e3003_time_limit(prepared, monkeypatch):
    ticks = iter([0.0] + [10_000.0] * 50)
    monkeypatch.setattr(registry, "clock", lambda: next(ticks))
    assert run({**ROOT_CFG, "runtime": {"time_limit_min": 1}}) == 1
    assert meta("EXP-001")["error_code"] == "E-3003"


# --- E-2003 / E-4004 / E-4007 ------------------------------------------------------------
def test_e2003_not_prepared(workspace):
    assert run_err(ROOT_CFG) == "E-2003"
    assert not config.EXPERIMENTS_DIR.exists() or not list(config.EXPERIMENTS_DIR.glob("EXP-*"))


def test_e4004_reference_eval_times(prepared):
    assert run(ROOT_CFG) == 0
    path = folder("EXP-001") / "predictions.parquet"
    pred = pd.read_parquet(path)
    pred.drop(pred[pred["segment"] == "val"].index[:1]).to_parquet(path, index=False)
    assert run(child(changed="naive.lag", naive={"lag": 1})) == 1
    assert meta("EXP-002")["error_code"] == "E-4004"


def test_e4007_lock_held(prepared):
    config.EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    lock = config.EXPERIMENTS_DIR / ".lock"
    holder = subprocess.Popen(
        [sys.executable, "-c",
         f"import filelock,sys; l=filelock.FileLock(r'{lock}'); l.acquire(); "
         "print('ok', flush=True); sys.stdin.read()"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
    )  # fmt: skip
    try:
        assert holder.stdout.readline().strip() == "ok"
        assert run_err(ROOT_CFG) == "E-4007"
    finally:
        holder.stdin.close()
        holder.wait(timeout=10)


# --- post_test 표시 ---------------------------------------------------------------------
def test_post_test_flag_when_lock_exists(prepared):
    (config.RESULTS_DIR / "test").mkdir(parents=True)
    (config.RESULTS_DIR / "test" / "LOCK").write_text("{}", encoding="utf-8")
    assert run(ROOT_CFG) == 0
    assert meta("EXP-001")["post_test"] is True


# --- dev → full 옮기기(changed: phase) ----------------------------------------------------
def test_changed_phase_rule():
    parent = registry.resolve_config(
        {
            **ROOT_CFG,
            "id": "EXP-002",
            "parent": "EXP-001",
            "changed": "naive.lag",
            "naive": {"lag": 1},
        }
    )
    moved = registry.resolve_config(
        {
            **ROOT_CFG,
            "id": "EXP-003",
            "parent": "EXP-002",
            "changed": "phase",
            "phase": "full",
            "naive": {"lag": 1},
        }
    )
    registry.check_one_change(parent, moved)
    bad = {**moved, "naive.lag": 144}
    with pytest.raises(TPError):
        registry.check_one_change(parent, bad)


# --- 완료 조건 (3) 결과 표 ----------------------------------------------------------------
def test_results_table_lists_all_experiments(prepared):
    assert run(ROOT_CFG) == 0
    assert run(child(changed="naive.lag", naive={"lag": 1})) == 0
    table = pd.read_csv(config.RESULTS_DIR / "results.csv")
    assert list(table["exp_id"]) == ["EXP-001", "EXP-002"]
    expected_cols = [  # D-09 (v2: horizon, dec_*; v2.3: dec_*_std)
        "exp_id", "name", "phase", "parent", "changed", "model_type", "horizon", "zone_rank",
        "square_id", "val_mae", "val_mae_std", "val_rmse", "val_rmse_std", "val_rel_mae", "n",
        "compare_group", "dec_threshold", "dec_episodes", "dec_missed", "dec_false_alarm_min",
        "dec_lead_min", "dec_missed_std", "dec_false_alarm_min_std", "dec_lead_min_std",
        "status", "post_test", "duration_s",
    ]  # fmt: skip
    assert list(table.columns) == expected_cols
    assert table["compare_group"].nunique() == 1
    md = (config.RESULTS_DIR / "results.md").read_text(encoding="utf-8")
    assert "EXP-002" in md and "선택용" in md


def test_results_marks_corrupt_folder(prepared):
    assert run(ROOT_CFG) == 0
    (config.EXPERIMENTS_DIR / "EXP-002_broken").mkdir()
    assert cli.main(["results"]) == 0
    table = pd.read_csv(config.RESULTS_DIR / "results.csv")
    assert table.set_index("exp_id").loc["EXP-002", "status"] == "corrupt"


@pytest.mark.parametrize(
    ("filename", "content"),
    [
        ("config.yaml", ""),  # empty YAML -> None
        ("config.yaml", "- a list\n"),
        ("config.yaml", "name: x\n"),  # required keys missing
        ("meta.json", "{}"),  # no status
        ("meta.json", "[]"),
        ("metrics.json", "{not json"),
        ("metrics.json", "{}"),
    ],
)
def test_results_marks_unreadable_folder_corrupt_and_keeps_the_rest(prepared, filename, content):
    # I-04: a folder that cannot be read is marked corrupt and skipped; the command still succeeds
    assert run(ROOT_CFG) == 0
    root = next(config.EXPERIMENTS_DIR.glob("EXP-001_*"))
    shutil.copytree(root, config.EXPERIMENTS_DIR / "EXP-002_x")
    (config.EXPERIMENTS_DIR / "EXP-002_x" / filename).write_text(content, encoding="utf-8")
    assert cli.main(["results"]) == 0
    table = pd.read_csv(config.RESULTS_DIR / "results.csv").set_index("exp_id")
    assert table.loc["EXP-002", "status"] == "corrupt"
    assert table.loc["EXP-001", "status"] == "completed"


@pytest.mark.parametrize(
    ("filename", "content"),
    [
        ("meta.json", "{}"),  # no status
        ("meta.json", "[]"),
        ("meta.json", '{"status": "completed"}'),  # other required fields missing
        (
            "meta.json",
            '{"status": "weird", "post_test": false, "config_hash": "x", '
            '"compare_group": "g", "square_id": 1}',
        ),  # fmt: skip
        ("config.yaml", "name: x\n"),  # required keys missing
    ],
)
def test_scan_marks_unreadable_folder_corrupt(prepared, filename, content):
    # A broken folder must not crash later runs; it counts as corrupt (not a usable parent)
    assert run(ROOT_CFG) == 0
    shutil.copytree(folder("EXP-001"), config.EXPERIMENTS_DIR / "EXP-002_x")
    (config.EXPERIMENTS_DIR / "EXP-002_x" / filename).write_text(content, encoding="utf-8")
    assert registry.scan()["EXP-002"].status == "corrupt"
    assert run(child(id="EXP-003", changed="naive.lag", naive={"lag": 1})) == 0
    assert run_err(child(id="EXP-002", changed="naive.lag", naive={"lag": 1}), retry=True) == (
        "E-4003"
    )
    assert run_err(child(id="EXP-004", parent="EXP-002", changed="naive.lag",
                         naive={"lag": 1})) == "E-4002"  # fmt: skip


# --- I-03 sweep --------------------------------------------------------------------------
def sweep_args(values: str) -> list[str]:
    return [
        "sweep",
        "--parent",
        "EXP-001",
        "--key",
        "naive.lag",
        "--values",
        values,
        "--start-id",
        "EXP-002",
        "--name",
        "lag",
        "--reason",
        "r",
        "--hypothesis",
        "h",
    ]


def test_sweep_creates_and_runs_children(prepared):
    assert run(ROOT_CFG) == 0
    assert cli.main(sweep_args("[1]")) == 0
    assert (config.CONFIGS_DIR / "experiments" / "EXP-002.yaml").is_file()
    assert folder("EXP-002").name == "EXP-002_lag-1"
    assert meta("EXP-002")["status"] == "completed"


def test_sweep_validates_all_before_creating_any(prepared):
    assert run(ROOT_CFG) == 0
    assert cli.main(sweep_args("[1, 7]")) == 1
    assert not (config.CONFIGS_DIR / "experiments" / "EXP-002.yaml").exists()
    assert not list(config.EXPERIMENTS_DIR.glob("EXP-002_*"))


def test_sweep_continues_after_run_failure(prepared, monkeypatch):
    assert run(ROOT_CFG) == 0
    original = naive.predict_naive
    calls = []

    def flaky(s, lag, targets, horizon=1):
        calls.append(lag)
        if len(calls) == 1:
            raise TPError("E-3003", "시간 초과: test")
        return original(s, lag, targets, horizon)

    monkeypatch.setattr(naive, "predict_naive", flaky)
    registry.sweep("EXP-001", "runtime.threads", "[2, 3]", "EXP-002", "threads", "r", "h")
    assert meta("EXP-002")["status"] == "failed"
    assert meta("EXP-003")["status"] == "completed"
    assert folder("EXP-003").name == "EXP-003_threads-3"


def test_results_command_exit_0_without_experiments(workspace):
    assert cli.main(["results"]) == 0
    assert results.load_rows() == []


# --- 비활성 키는 비교 대상이 아님 (3.3, 레드팀 #16) ---------------------------------------
def _arima_cfg(exp_id: str, parent: str | None, changed: str | None, **arima_fields) -> dict:
    cfg = {**json.loads(json.dumps(ROOT_CFG)), "id": exp_id, "parent": parent,
           "changed": changed, "model": {"type": "arima"},
           "arima": {"order": [2, 1, 2], "seasonal": "none", **arima_fields}}  # fmt: skip
    del cfg["naive"]
    return registry.resolve_config(cfg)


def test_fourier_k_default_does_not_count_as_second_change():
    parent = _arima_cfg("EXP-003", "EXP-001", "model.type")
    child = _arima_cfg("EXP-004", "EXP-003", "arima.seasonal", seasonal="fourier")
    assert child["arima.fourier_k"] == 3
    registry.check_one_change(parent, child)
    registry.check_one_change(child, parent | {"id": "EXP-005", "changed": "arima.seasonal"})


def test_active_fourier_k_change_still_counts():
    parent = _arima_cfg("EXP-004", "EXP-003", "arima.seasonal", seasonal="fourier")
    both = _arima_cfg("EXP-005", "EXP-004", "arima.fourier_k", seasonal="fourier", fourier_k=5)
    registry.check_one_change(parent, both)
    two = both | {"arima.order": [1, 1, 1]}
    with pytest.raises(TPError):
        registry.check_one_change(parent, two)
