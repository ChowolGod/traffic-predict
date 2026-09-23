"""Experiment registry: config resolution, rules, run, sweep (M-11, plan 3.3 / D-05 / D-13)."""

import hashlib
import json
import logging
import os
import re
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import filelock
import pandas as pd
import yaml

from tp import config
from tp.config import PHASES, PhaseConfig
from tp.data import zones
from tp.errors import TPError
from tp.eval import metrics
from tp.exp import results
from tp.models import arima, lstm, naive
from tp.prep import series as series_mod
from tp.prep.split import cut_until
from tp.seed import set_seed

log = logging.getLogger(__name__)

ID_RE = re.compile(r"^EXP-(\d{3})$")
NAME_RE = re.compile(r"^[a-z0-9-]{1,40}$")
META_KEYS = ("id", "name", "phase", "parent", "changed", "reason", "hypothesis")
NOT_COMPARED = {"id", "name", "parent", "changed", "reason", "hypothesis", "runtime.time_limit_min"}
FAMILIES = ("naive", "arima", "lstm")
DEFAULTS = {"zone_rank": 1, "runtime.time_limit_min": 30, "runtime.threads": 4}
INITIAL = {  # plan 4.2 모델 초기 설정
    "naive": {"naive.lag": 144},
    "arima": {"arima.order": [2, 1, 2], "arima.seasonal": "none"},
    "lstm": {
        "lstm.window": 144,
        "lstm.hidden": 64,
        "lstm.layers": 1,
        "lstm.dropout": 0.0,
        "lstm.lr": 1e-3,
        "lstm.batch": 256,
        "lstm.max_epochs": 50,
        "lstm.patience": 5,
        "lstm.scaler": "standard",
    },
}
CODE_PATHS = ("src", "pyproject.toml", "uv.lock", "configs/phases.yaml")

clock = time.monotonic


def _int(lo: int, hi: int, allowed: tuple[int, ...] = ()) -> Callable[[object], bool]:
    def check(v: object) -> bool:
        ok = isinstance(v, int) and not isinstance(v, bool) and lo <= v <= hi
        return ok and (not allowed or v in allowed)

    return check


def _float(lo: float, hi: float) -> Callable[[object], bool]:
    return lambda v: isinstance(v, int | float) and not isinstance(v, bool) and lo <= v <= hi


def _choice(*options: str) -> Callable[[object], bool]:
    return lambda v: isinstance(v, str) and v in options


def _order(v: object) -> bool:
    return (
        isinstance(v, list)
        and len(v) == 3
        and _int(0, 5)(v[0])
        and _int(0, 2)(v[1])
        and _int(0, 5)(v[2])
    )


SCHEMA: dict[str, Callable[[object], bool]] = {  # D-05
    "zone_rank": _int(1, 3),
    "model.type": _choice(*FAMILIES),
    "naive.lag": _int(1, 144, allowed=(1, 144)),
    "arima.order": _order,
    "arima.seasonal": _choice("none", "diff144", "fourier"),
    "arima.fourier_k": _int(1, 10),
    "lstm.window": _int(6, 1008),
    "lstm.hidden": _int(8, 256),
    "lstm.layers": _int(1, 3),
    "lstm.dropout": _float(0.0, 0.5),
    "lstm.lr": _float(1e-5, 1e-1),
    "lstm.batch": _int(32, 1024),
    "lstm.max_epochs": _int(1, 200),
    "lstm.patience": _int(1, 20),
    "lstm.scaler": _choice("standard", "minmax"),
    "runtime.time_limit_min": _int(1, 30),
    "runtime.threads": _int(1, 8),
}


def _rule(message: str) -> TPError:
    return TPError("E-4001", f"실험 규칙 위반: {message}")


def flatten(raw: dict, prefix: str = "") -> dict:
    out = {}
    for key, value in raw.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(flatten(value, f"{name}."))
        else:
            out[name] = value
    return out


def unflatten(flat: dict) -> dict:
    out: dict = {}
    for key, value in flat.items():
        node = out
        *parents, last = key.split(".")
        for part in parents:
            node = node.setdefault(part, {})
        node[last] = value
    return out


def applicable_keys(model_type: str, seasonal: str | None) -> list[str]:
    keys = [k for k in SCHEMA if k.startswith(f"{model_type}.")]
    if model_type == "arima" and seasonal != "fourier":
        keys.remove("arima.fourier_k")
    return keys


def resolve_config(raw: dict) -> dict:
    """Validate D-05 and return the flat, fully-resolved config (defaults written out)."""
    flat = flatten(raw)
    unknown = sorted(set(flat) - set(META_KEYS) - set(SCHEMA))
    if unknown:
        raise _rule(f"알 수 없는 키 {unknown}")
    for key in META_KEYS:
        if key not in flat:
            raise _rule(f"{key}가 없음")
    if not isinstance(flat["id"], str) or not ID_RE.match(flat["id"]):
        raise _rule(f"id 형식: {flat['id']!r}")
    if not isinstance(flat["name"], str) or not NAME_RE.match(flat["name"]):
        raise _rule(f"name 형식: {flat['name']!r}")
    if flat["phase"] not in PHASES:
        raise _rule(f"phase: {flat['phase']!r}")
    for key in ("reason", "hypothesis"):
        if not isinstance(flat[key], str) or not flat[key].strip():
            raise _rule(f"{key}가 비어 있음")
    if flat["parent"] is not None and not (
        isinstance(flat["parent"], str) and ID_RE.match(flat["parent"])
    ):
        raise _rule(f"parent 형식: {flat['parent']!r}")
    if flat["changed"] is not None and not isinstance(flat["changed"], str):
        raise _rule(f"changed 형식: {flat['changed']!r}")

    model_type = flat.get("model.type")
    if not SCHEMA["model.type"](model_type):
        raise _rule(f"model.type: {model_type!r}")
    seasonal = flat.get("arima.seasonal")
    keys = applicable_keys(model_type, seasonal)
    resolved = {k: flat[k] for k in META_KEYS}
    resolved["zone_rank"] = flat.get("zone_rank", DEFAULTS["zone_rank"])
    resolved["model.type"] = model_type
    for key in keys:
        if key in flat:
            resolved[key] = flat[key]
        elif key == "arima.fourier_k":
            resolved[key] = 3
        else:
            raise _rule(f"{key}가 없음 (model.type={model_type})")
    for key in ("runtime.time_limit_min", "runtime.threads"):
        resolved[key] = flat.get(key, DEFAULTS[key])
    ignored = sorted(set(flat) - set(resolved))
    if ignored:
        log.warning("현재 model.type에 해당하지 않아 무시하는 키: %s", ignored)

    for key, value in resolved.items():
        if key in SCHEMA and not SCHEMA[key](value):
            raise _rule(f"{key} 값 {value!r}")
    if model_type == "lstm" and resolved["lstm.layers"] == 1 and resolved["lstm.dropout"] != 0:
        raise _rule("lstm.layers=1이면 lstm.dropout은 0")
    return resolved


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()


def check_one_change(parent: dict, child: dict) -> None:
    keys = (set(parent) | set(child)) - NOT_COMPARED
    missing = object()
    diff = sorted(k for k in keys if parent.get(k, missing) != child.get(k, missing))
    changed = child["changed"]
    if parent["model.type"] != child["model.type"]:
        rest = [k for k in diff if k.split(".")[0] not in FAMILIES]
        if rest != ["model.type"] or changed != "model.type":
            raise _rule(f"계열 전환은 model.type 하나만 바꿔야 함: 바뀐 키 {diff}")
        new = child["model.type"]
        off = [
            k
            for k in applicable_keys(new, child.get("arima.seasonal"))
            if child.get(k) != INITIAL[new].get(k)
        ]
        if off:
            raise _rule(f"계열 전환 시 {new} 설정은 초기값이어야 함: {off}")
        return
    # Resolved configs hold only active keys, so a key on one side only is inactive on the
    # other (e.g. arima.fourier_k when seasonal changes) and is not compared (plan 3.3).
    diff = [k for k in diff if k in parent and k in child]
    if diff != [changed]:
        raise _rule(f"부모 대비 바뀐 키는 정확히 1개여야 함: 바뀐 키 {diff}, changed={changed!r}")


@dataclass
class Experiment:
    id: str
    folder: Path
    meta: dict | None
    config: dict | None  # flat resolved

    @property
    def status(self) -> str:
        return self.meta["status"] if self.meta else "corrupt"


def scan() -> dict[str, Experiment]:
    found: dict[str, Experiment] = {}
    if not config.EXPERIMENTS_DIR.is_dir():
        return found
    for folder in sorted(config.EXPERIMENTS_DIR.glob("EXP-*_*")):
        match = ID_RE.match(folder.name[:7])
        if not folder.is_dir() or not match:
            continue
        try:
            meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
            cfg = flatten(yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8")))
        except (OSError, ValueError, yaml.YAMLError, AttributeError):
            meta, cfg = None, None
        found[folder.name[:7]] = Experiment(folder.name[:7], folder, meta, cfg)
    return found


def git_state() -> tuple[str | None, bool]:
    def git(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", "-C", str(config.ROOT), *args], capture_output=True, text=True
        )

    head = git("rev-parse", "HEAD")
    if head.returncode != 0:
        return None, True
    status = git("status", "--porcelain", "--", *CODE_PATHS)
    return head.stdout.strip(), bool(status.stdout.strip())


def check_rules(
    cfg: dict, index: dict[str, Experiment], phase: PhaseConfig, retry: bool, dirty: bool
) -> None:
    exp_id, number = cfg["id"], int(cfg["id"][4:])
    if cfg["zone_rank"] > phase.k:
        raise _rule(f"zone_rank {cfg['zone_rank']} > K {phase.k}")
    existing = index.get(exp_id)
    if existing:
        if not retry:
            raise TPError("E-4003", f"재실행 불가: {exp_id} {existing.status} (--retry 필요)")
        if existing.status not in ("failed", "running"):
            raise TPError("E-4003", f"재실행 불가: {exp_id} {existing.status}")
        if existing.meta["config_hash"] != config_hash(cfg):
            raise TPError("E-4003", f"재실행 불가: {exp_id} 설정이 처음 실행과 다름")
    else:
        if retry:
            raise TPError("E-4003", f"재실행 불가: {exp_id} 없음")
        expected = max((int(i[4:]) for i in index), default=0) + 1
        if number != expected:
            raise _rule(f"id는 EXP-{expected:03d}이어야 함: {exp_id}")

    if cfg["parent"] is None:
        if cfg["changed"] is not None:
            raise _rule("루트 실험의 changed는 null")
        root_ok = (
            cfg["model.type"] == "naive" and cfg.get("naive.lag") == 144 and cfg["zone_rank"] == 1
        )
        if not root_ok:
            raise _rule("루트 실험은 naive, lag 144, zone_rank 1")
        others = [
            e.id
            for e in index.values()
            if e.id != exp_id
            and e.config
            and e.config["phase"] == cfg["phase"]
            and e.config["parent"] is None
        ]
        if others:
            raise _rule(f"{cfg['phase']} 단계의 루트가 이미 있음: {others}")
    else:
        parent = index.get(cfg["parent"])
        if parent is None or parent.status != "completed":
            raise TPError("E-4002", f"부모 오류: {cfg['parent']} (없거나 완료되지 않음)")
        moving = cfg["changed"] == "phase" and parent.config["phase"] == "dev"
        if parent.config["phase"] != cfg["phase"] and not (moving and cfg["phase"] == "full"):
            raise TPError("E-4002", f"부모 오류: {cfg['parent']} (phase가 다름)")
        check_one_change(parent.config, cfg)

    if cfg["phase"] == "full" and dirty:
        raise _rule("full 단계는 커밋되지 않은 코드 변경이 있으면 실행할 수 없음")


def acquire_lock() -> filelock.FileLock:
    config.EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    lock = filelock.FileLock(str(config.EXPERIMENTS_DIR / ".lock"))
    try:
        lock.acquire(timeout=0)
    except filelock.Timeout as exc:
        raise TPError("E-4007", "다른 실행이 진행 중") from exc
    return lock


def zone_square(phase: PhaseConfig, zone_rank: int) -> int:
    path = zones.zones_path(phase.name)
    if not path.is_file():
        raise TPError("E-2003", f"prepare 먼저 실행: --phase {phase.name}")
    listed = json.loads(path.read_text(encoding="utf-8"))
    if listed.get("k") != phase.k or zone_rank > len(listed["zones"]):
        raise TPError("E-2003", f"prepare 먼저 실행: --phase {phase.name}")
    return listed["zones"][zone_rank - 1]["square_id"]


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _pred_frame(rows: pd.DataFrame, y_pred, seed: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time_utc": rows["time_utc"].to_numpy(),
            "segment": rows["segment"].astype(str).to_numpy(),
            "y_true": rows["y"].to_numpy(),
            "y_pred": y_pred,
            "seed": seed,
            "is_imputed": rows["is_imputed"].to_numpy(),
        }
    )


def _run_naive(data: pd.DataFrame, cfg: dict, folder: Path, check_time) -> pd.DataFrame:
    lag = cfg["naive.lag"]
    rows = data.iloc[lag:]
    y_pred = naive.predict_naive(data, lag, pd.DatetimeIndex(rows["time_utc"]))
    return _pred_frame(rows, y_pred.to_numpy(), seed=-1)


def _run_arima(data: pd.DataFrame, cfg: dict, folder: Path, check_time) -> pd.DataFrame:
    params = arima.fit_arima(data[data["segment"] == "train"], cfg)
    check_time()
    arima.save_params(folder / "model" / "arima_params.json", params)
    y_pred = arima.predict_arima(params, data)
    return _pred_frame(data.loc[y_pred.index], y_pred.to_numpy(), seed=-1)


def _run_lstm(data: pd.DataFrame, cfg: dict, folder: Path, check_time) -> pd.DataFrame:
    targets = data.index[cfg["lstm.window"] :]
    frames = []
    for seed in config.LSTM_SEEDS:
        set_seed(seed, cfg["runtime.threads"])
        model = lstm.train_lstm(data, cfg, seed, check_time)
        lstm.save_model(model, folder / "model", seed)
        lstm.save_curve(model.history, model.best_epoch, folder / "curves", seed)
        y_pred = lstm.predict_lstm(model, data, targets)
        frames.append(_pred_frame(data.loc[targets], y_pred.to_numpy(), seed=seed))
        log.info("seed %d: best epoch %d / %d", seed, model.best_epoch, len(model.history))
    return pd.concat(frames, ignore_index=True)


MODELS = {"naive": _run_naive, "arima": _run_arima, "lstm": _run_lstm}


def _reference_times(index: dict[str, Experiment], cfg: dict) -> pd.DatetimeIndex | None:
    same = [
        e
        for e in index.values()
        if e.id != cfg["id"]
        and e.status == "completed"
        and e.config["phase"] == cfg["phase"]
        and e.config["zone_rank"] == cfg["zone_rank"]
    ]
    if not same:
        return None
    first = min(same, key=lambda e: e.id)
    return metrics.eval_times(pd.read_parquet(first.folder / "predictions.parquet"), "val")


def _compare_group(phase: str, square_id: int, times: pd.DatetimeIndex) -> str:
    key = [phase, square_id, [t.isoformat() for t in times], config.AGG_VERSION,
           config.IMPUTE_VERSION]  # fmt: skip
    return hashlib.sha256(json.dumps(key).encode()).hexdigest()[:8]


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _execute(cfg: dict, index: dict[str, Experiment], phase: PhaseConfig, retry: bool,
             commit: str | None, dirty: bool) -> Path:  # fmt: skip
    square_id = zone_square(phase, cfg["zone_rank"])
    series = series_mod.load_series(phase, square_id)

    folder = config.EXPERIMENTS_DIR / f"{cfg['id']}_{cfg['name']}"
    folder.mkdir(parents=True, exist_ok=retry)
    (folder / "config.yaml").write_text(
        yaml.safe_dump(unflatten(cfg), allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    meta = {
        **{k: cfg[k] for k in ("id", "parent", "changed", "reason", "hypothesis")},
        "status": "running",
        "error_code": None,
        "post_test": (config.RESULTS_DIR / "test" / "LOCK").exists(),
        "config_hash": config_hash(cfg),
        "git_commit": commit,
        "git_dirty": dirty,
        "started_at": _now(),
        "ended_at": None,
        "duration_s": None,
        "pid": os.getpid(),
        "square_id": square_id,
        "agg_version": config.AGG_VERSION,
        "impute_version": config.IMPUTE_VERSION,
        "compare_group": None,
    }
    _write_json(folder / "meta.json", meta)

    handler = logging.FileHandler(folder / "run.log", mode="w", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    logging.getLogger("tp").addHandler(handler)
    start = clock()
    limit_s = cfg["runtime.time_limit_min"] * 60

    def check_time() -> None:
        elapsed = clock() - start
        if elapsed > limit_s:
            raise TPError("E-3003", f"시간 초과: {elapsed:.0f}s > {limit_s}s")

    try:
        log.info("실험 시작 %s (구역 %d, %s)", cfg["id"], square_id, cfg["model.type"])
        set_seed(0, cfg["runtime.threads"])
        data = cut_until(series, "val")
        pred = MODELS[cfg["model.type"]](data, cfg, folder, check_time)
        check_time()
        result = metrics.evaluate(pred, data, "val", _reference_times(index, cfg))
        pred.to_parquet(folder / "predictions.parquet", index=False)
        _write_json(folder / "metrics.json", result)
        meta["compare_group"] = _compare_group(
            cfg["phase"], square_id, metrics.eval_times(pred, "val")
        )
        meta["status"] = "completed"
        log.info("실험 완료 %s: val MAE %.4f", cfg["id"], result["val"]["mae"])
    except TPError as err:
        meta["status"], meta["error_code"] = "failed", err.code
        log.error("실험 실패 %s: %s", cfg["id"], err)
        raise
    except Exception:
        meta["status"] = "failed"
        log.exception("실험 실패 %s", cfg["id"])
        raise
    finally:
        meta["ended_at"] = _now()
        meta["duration_s"] = round(clock() - start, 3)
        _write_json(folder / "meta.json", meta)
        logging.getLogger("tp").removeHandler(handler)
        handler.close()
        results.rebuild_results()
    return folder


def _prepare_run(cfg: dict, retry: bool) -> tuple[dict, PhaseConfig, str | None, bool]:
    index = scan()
    phase = config.load_phase(cfg["phase"])
    commit, dirty = git_state()
    check_rules(cfg, index, phase, retry, dirty)
    check_data(cfg, phase)
    return index, phase, commit, dirty


def check_data(cfg: dict, phase: PhaseConfig) -> None:
    """Data-dependent checks that must pass before any folder is created (E-2003, E-4001)."""
    series = series_mod.load_series(phase, zone_square(phase, cfg["zone_rank"]))
    if cfg["model.type"] == "lstm":
        n = lstm.count_train_samples(cut_until(series, "val"), cfg["lstm.window"])
        if n < cfg["lstm.batch"]:
            raise _rule(f"LSTM 학습 샘플 수 {n} < lstm.batch {cfg['lstm.batch']}")


def run_from_file(path: str | Path, retry: bool = False) -> Path:
    cfg = resolve_config(yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})
    lock = acquire_lock()
    try:
        index, phase, commit, dirty = _prepare_run(cfg, retry)
        return _execute(cfg, index, phase, retry, commit, dirty)
    finally:
        lock.release()


def _slug(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "-", json.dumps(value).lower()).strip("-")


def sweep(parent_id: str, key: str, values_json: str, start_id: str, name: str,
          reason: str, hypothesis: str) -> None:  # fmt: skip
    try:
        values = json.loads(values_json)
    except json.JSONDecodeError as exc:
        raise _rule(f"--values는 JSON 리스트여야 함: {values_json}") from exc
    if not isinstance(values, list) or not values:
        raise _rule(f"--values는 비어 있지 않은 JSON 리스트여야 함: {values_json}")
    if not ID_RE.match(start_id):
        raise _rule(f"--start-id 형식: {start_id}")

    lock = acquire_lock()
    try:
        index = scan()
        parent = index.get(parent_id)
        if parent is None or parent.status != "completed":
            raise TPError("E-4002", f"부모 오류: {parent_id} (없거나 완료되지 않음)")
        start = int(start_id[4:])
        _, dirty = git_state()
        children = []
        for offset, value in enumerate(values):
            flat = dict(parent.config)
            if key == "model.type":
                flat = {k: v for k, v in flat.items() if k.split(".")[0] not in FAMILIES}
                flat.update(INITIAL.get(value, {}))
            flat[key] = value
            flat.update(
                id=f"EXP-{start + offset:03d}", name=f"{name}-{_slug(value)}"[:40],
                parent=parent_id, changed=key, reason=reason, hypothesis=hypothesis,
            )  # fmt: skip
            cfg = resolve_config(unflatten(flat))
            simulated = {**index, **{c["id"]: Experiment(c["id"], Path(), None, None)
                                     for c in children}}  # fmt: skip
            phase = config.load_phase(cfg["phase"])
            check_rules(cfg, simulated, phase, retry=False, dirty=dirty)
            check_data(cfg, phase)
            children.append(cfg)

        outcomes = []
        for cfg in children:
            path = config.CONFIGS_DIR / "experiments" / f"{cfg['id']}.yaml"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                yaml.safe_dump(unflatten(cfg), allow_unicode=True, sort_keys=False),
                encoding="utf-8",
            )
            try:
                index, phase, commit, dirty = _prepare_run(cfg, retry=False)
                _execute(cfg, index, phase, False, commit, dirty)
                outcomes.append((cfg["id"], "completed"))
            except TPError as err:
                outcomes.append((cfg["id"], f"failed {err.code}"))
        for exp_id, outcome in outcomes:
            log.info("sweep %s: %s", exp_id, outcome)
    finally:
        lock.release()
