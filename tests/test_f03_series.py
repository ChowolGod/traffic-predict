import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from tp import cli, config
from tp.config import PhaseConfig
from tp.data import cache
from tp.errors import TPError
from tp.prep import series, split

GOLDEN = json.loads((Path(__file__).parent / "fixtures" / "golden.json").read_text("utf-8"))
DAYS = [date(2013, 11, 4), date(2013, 11, 5), date(2013, 11, 6)]
TOP = 5000


def dev_phase() -> PhaseConfig:
    return config.load_phase("dev")  # train 11-04, val 11-05


def local(ts: str) -> pd.Timestamp:
    return pd.Timestamp(ts, tz=config.TZ).tz_convert("UTC").as_unit("ns")


def build(square_id: int = TOP, phase: PhaseConfig | None = None) -> pd.DataFrame:
    cache.ensure_daily_caches(DAYS)
    return series.build_series(square_id, phase or dev_phase())


# --- D-04 스키마와 정규 인덱스 ------------------------------------------------------
def test_schema_and_regular_grid(workspace):
    df = build()
    assert list(df.columns) == ["time_utc", "y", "is_imputed", "segment"]
    assert str(df["time_utc"].dtype) == "datetime64[ns, UTC]"
    assert df["y"].dtype == "float64" and df["is_imputed"].dtype == bool
    assert df["time_utc"].iloc[0] == local("2013-11-04 00:00")
    assert df["time_utc"].iloc[-1] == local("2013-11-05 23:50")
    assert (df["time_utc"].diff().dropna() == pd.Timedelta("10min")).all()
    assert not df["y"].isna().any() and (df["y"] >= 0).all()


# --- 완료 조건 (1) 구간 경계가 설정 날짜와 정확히 같고 겹침 0 -------------------------
def test_segment_boundaries_are_exact(workspace):
    df = build().set_index("time_utc")["segment"]
    assert df[local("2013-11-04 23:50")] == "train"
    assert df[local("2013-11-05 00:00")] == "val"
    assert (df == "train").sum() == 144 and (df == "val").sum() == 144


def test_segment_of_labels_each_time_once():
    phase = config.load_phase("full")
    times = pd.Series([local("2013-12-08 23:50"), local("2013-12-09 00:00"),
                       local("2013-12-15 23:50"), local("2013-12-16 00:00")])  # fmt: skip
    assert list(split.segment_of(times, phase)) == ["train", "val", "val", "test"]


# --- 완료 조건 (2) run에 넘기는 계열에 test 구간이 없음 --------------------------------
def test_cut_until_val_drops_test():
    phase = config.load_phase("full")
    grid = series.local_grid(phase)
    df = pd.DataFrame({"time_utc": grid, "y": 1.0, "is_imputed": False})
    df["segment"] = split.segment_of(df["time_utc"], phase)
    cut = split.cut_until(df, "val")
    assert set(cut["segment"]) == {"train", "val"}
    assert cut["time_utc"].iloc[-1] == local("2013-12-15 23:50")


# --- 완료 조건 (3) 3칸 이하 결측은 앞값 채우기 + is_imputed -----------------------------
def test_short_gap_forward_filled_and_marked(workspace):
    df = build().set_index("time_utc")
    gap = pd.Timestamp(GOLDEN["missing_gaps"][0]["start_utc"])  # 1칸, 11-04
    before = gap - pd.Timedelta("10min")
    assert df.loc[gap, "is_imputed"]
    assert df.loc[gap, "y"] == df.loc[before, "y"]
    assert df["is_imputed"].sum() == 1


def test_cell_without_rows_at_present_time_is_zero(workspace):
    df = build(square_id=10000).set_index("time_utc")
    for case in GOLDEN["zero_cells"]:
        t = pd.Timestamp(case["time_utc"])
        assert df.loc[t, "y"] == 0.0 and not df.loc[t, "is_imputed"]


def test_values_equal_cache(workspace):
    df = build().set_index("time_utc")
    case = GOLDEN["cache_values"][0]
    assert df.loc[pd.Timestamp(case["time_utc"]), "y"] == pytest.approx(case["internet"], abs=1e-9)


# --- 완료 조건 (4) 연속 4칸 이상 결측은 E-2002 -----------------------------------------
def test_e2002_long_gap(workspace):
    phase = PhaseConfig("dev", (DAYS[0], DAYS[1]), (DAYS[2], DAYS[2]), None, 1)
    with pytest.raises(TPError) as exc:
        build(phase=phase)
    assert exc.value.code == "E-2002"
    assert str(TOP) in exc.value.message


def test_leading_gap_moves_series_start(workspace):
    cache.ensure_daily_caches(DAYS)
    first = local("2013-11-04 00:00")
    drop = [first + pd.Timedelta(minutes=10 * i) for i in range(5)]
    meta = cache.read_meta(DAYS[0])
    drop_ms = {int(t.value // 1_000_000) for t in drop}
    meta["present_times"] = [t for t in meta["present_times"] if t not in drop_ms]
    cache.meta_path(DAYS[0]).write_text(json.dumps(meta), encoding="utf-8")
    df = pd.read_parquet(cache.cache_path(DAYS[0]))
    df[~df["time_utc"].isin(drop)].to_parquet(cache.cache_path(DAYS[0]), index=False)
    out = series.build_series(TOP, dev_phase())
    assert out["time_utc"].iloc[0] == drop[-1] + pd.Timedelta("10min")
    assert not out["is_imputed"].iloc[:3].any()


# --- 완료 조건 (5) 사용 구간의 모든 날이 144칸 -------------------------------------------
@pytest.mark.parametrize("name", ["dev", "full"])
def test_every_day_has_144_slots(name, monkeypatch):
    monkeypatch.setattr(config, "CONFIGS_DIR", config.ROOT / "configs")
    phase = config.load_phase(name)
    grid = series.local_grid(phase)
    per_day = pd.Series(1, index=grid.tz_convert(config.TZ).date).groupby(level=0).sum()
    assert (per_day == 144).all()
    assert len(per_day) == (phase.end - phase.start).days + 1


# --- 산출물 유효성과 E-2003 -------------------------------------------------------------
def test_ensure_series_reuse_and_rebuild_on_version(workspace, monkeypatch):
    cache.ensure_daily_caches(DAYS)
    phase = dev_phase()
    path = series.ensure_series(phase, TOP)
    calls = []
    original = series.build_series
    monkeypatch.setattr(series, "build_series", lambda s, p: calls.append(s) or original(s, p))
    series.ensure_series(phase, TOP)
    series.ensure_series(PhaseConfig(**{**phase.__dict__, "k": 3}), TOP)  # k는 판정에 안 씀
    assert calls == []
    monkeypatch.setattr(config, "IMPUTE_VERSION", config.IMPUTE_VERSION + 1)
    assert series.ensure_series(phase, TOP) == path
    assert calls == [TOP]


def test_load_series_e2003_when_not_prepared(workspace):
    with pytest.raises(TPError) as exc:
        series.load_series(dev_phase(), TOP)
    assert exc.value.code == "E-2003"


def test_load_series_e2003_when_meta_stale(workspace, monkeypatch):
    cache.ensure_daily_caches(DAYS)
    series.ensure_series(dev_phase(), TOP)
    monkeypatch.setattr(config, "AGG_VERSION", config.AGG_VERSION + 1)
    with pytest.raises(TPError) as exc:
        series.load_series(dev_phase(), TOP)
    assert exc.value.code == "E-2003"


def test_prepare_builds_series_for_zones(workspace):
    assert cli.main(["prepare", "--phase", "dev"]) == 0
    df = series.load_series(dev_phase(), TOP)
    assert len(df) == 288
    assert list(df["segment"].unique()) == ["train", "val"]
