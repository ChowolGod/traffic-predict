"""Paths, constants and phase config loading (M-15, plan 3.3 / D-06a / D-12)."""

import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import yaml

from tp.errors import TPError

ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = ROOT / "configs"
INTERIM_DIR = ROOT / "data" / "interim"
PROCESSED_DIR = ROOT / "data" / "processed"
EXPERIMENTS_DIR = ROOT / "experiments"
RESULTS_DIR = ROOT / "results"

TZ = "Europe/Rome"
AGG_VERSION = 1
IMPUTE_VERSION = 1
LSTM_SEEDS = (0, 1, 2)

DATA_START = date(2013, 11, 1)
DATA_END = date(2013, 12, 22)
PHASES = ("dev", "full")
SEGMENTS = ("train", "val", "test")

DateRange = tuple[date, date]


def raw_dir() -> Path:
    env = os.environ.get("TP_RAW_DIR")
    return Path(env) if env else ROOT / "data" / "raw"


@dataclass(frozen=True)
class PhaseConfig:
    name: str
    train: DateRange
    val: DateRange
    test: DateRange | None
    k: int

    @property
    def start(self) -> date:
        return self.train[0]

    @property
    def end(self) -> date:
        return (self.test or self.val)[1]

    def segments(self) -> dict[str, DateRange]:
        out = {"train": self.train, "val": self.val}
        if self.test:
            out["test"] = self.test
        return out


@dataclass(frozen=True)
class DecisionConfig:  # D-14
    threshold_ratio: float
    peak_quantile: float
    merge_gap: int
    report_horizon: int


def load_decision(path: Path | None = None) -> DecisionConfig:
    path = path or CONFIGS_DIR / "decision.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) if path.is_file() else None
    if not isinstance(raw, dict):
        raise TPError("E-2001", f"결정 설정 오류: {path.name}이 없거나 비어 있음")
    checks = {
        "threshold_ratio": lambda v: isinstance(v, int | float) and 0 < v <= 1,
        "peak_quantile": lambda v: isinstance(v, int | float) and 0.5 <= v <= 1,
        "merge_gap": lambda v: isinstance(v, int) and not isinstance(v, bool) and 0 <= v <= 6,
        "report_horizon": lambda v: v in (1, 3, 6) and not isinstance(v, bool),
    }
    for key, ok in checks.items():
        if key not in raw:
            raise TPError("E-2001", f"결정 설정 오류: {key}가 없음")
        if not ok(raw[key]):
            raise TPError("E-2001", f"결정 설정 오류: {key} 값 {raw[key]!r}")
    return DecisionConfig(**{key: raw[key] for key in checks})


def _bad(message: str) -> TPError:
    return TPError("E-2001", f"분할 설정 오류: {message}")


def _parse_range(name: str, segment: str, value) -> DateRange:
    if not isinstance(value, list) or len(value) != 2:
        raise _bad(f"{name}.{segment}는 [시작, 끝] 형식이어야 함: {value!r}")
    try:
        start, end = (v if isinstance(v, date) else date.fromisoformat(str(v)) for v in value)
    except ValueError as exc:
        raise _bad(f"{name}.{segment} 날짜 형식 오류: {value!r}") from exc
    if start > end:
        raise _bad(f"{name}.{segment} 시작이 끝보다 늦음: {start} > {end}")
    if start < DATA_START or end > DATA_END:
        raise _bad(f"{name}.{segment}가 허용 범위 {DATA_START}~{DATA_END} 밖: {start}~{end}")
    return start, end


def load_phase(name: str, path: Path | None = None) -> PhaseConfig:
    if name not in PHASES:
        raise _bad(f"알 수 없는 phase: {name}")
    path = path or CONFIGS_DIR / "phases.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    cfg = raw.get(name)
    if not isinstance(cfg, dict):
        raise _bad(f"{path.name}에 {name} 항목이 없음")
    for key in ("train", "val", "k"):
        if key not in cfg:
            raise _bad(f"{name}.{key}가 없음")
    if name == "dev" and "test" in cfg:
        raise _bad("dev에는 test를 둘 수 없음")

    ranges = [_parse_range(name, s, cfg[s]) for s in SEGMENTS if s in cfg]
    for (_, prev_end), (next_start, _) in zip(ranges, ranges[1:], strict=False):
        if next_start <= prev_end:
            raise _bad(f"{name} 구간이 겹치거나 순서가 틀림: {prev_end} >= {next_start}")

    k = cfg["k"]
    if not isinstance(k, int) or isinstance(k, bool) or not 1 <= k <= 3:
        raise _bad(f"{name}.k는 1~3이어야 함: {k!r}")

    return PhaseConfig(
        name=name,
        train=ranges[0],
        val=ranges[1],
        test=ranges[2] if len(ranges) == 3 else None,
        k=k,
    )
