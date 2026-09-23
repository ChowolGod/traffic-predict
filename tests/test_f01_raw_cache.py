import gzip
import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from tp import cli, config
from tp.data import cache, raw
from tp.errors import TPError

GOLDEN = json.loads((Path(__file__).parent / "fixtures" / "golden.json").read_text("utf-8"))
DAY = date(2013, 11, 4)
DAYS = [date(2013, 11, 4), date(2013, 11, 5), date(2013, 11, 6)]


def raw_path(workspace: Path, day: date) -> Path:
    return workspace / "raw" / f"sms-call-internet-mi-{day.isoformat()}.txt"


def write_raw(workspace: Path, day: date, lines: list[str]) -> None:
    raw_path(workspace, day).write_text("\n".join(lines) + "\n", encoding="ascii")


GOOD = "1\t1383519600000\t39\t\t\t\t\t1.5"  # 2013-11-04 00:00 CET


# --- 완료 조건 (1) 캐시 스키마 ---------------------------------------------------
def test_cache_schema(workspace):
    df = pd.read_parquet(cache.build_daily_cache(DAY))
    assert list(df.columns) == ["square_id", "time_utc", "internet"]
    assert df["square_id"].dtype == "int16"
    assert str(df["time_utc"].dtype) == "datetime64[ns, UTC]"
    assert df["internet"].dtype == "float64"
    assert not df.duplicated(["square_id", "time_utc"]).any()
    assert (df["internet"] >= 0).all()
    assert (df["time_utc"].dt.minute % 10 == 0).all()


# --- 완료 조건 (2) 원본에서 직접 합산한 값과 일치(국가 코드 합, 빈 값 = 0) --------------
def test_cache_values_match_golden(workspace):
    frames = pd.concat(pd.read_parquet(cache.build_daily_cache(d)) for d in DAYS)
    for case in GOLDEN["cache_values"]:
        row = frames[
            (frames.square_id == case["square_id"])
            & (frames.time_utc == pd.Timestamp(case["time_utc"]))
        ]
        assert len(row) == 1
        assert row["internet"].iloc[0] == pytest.approx(case["internet"], abs=1e-9)


def test_missing_cell_rows_are_absent_not_zero_filled_in_cache(workspace):
    frames = pd.concat(pd.read_parquet(cache.build_daily_cache(d)) for d in DAYS)
    for case in GOLDEN["zero_cells"]:
        hit = frames[
            (frames.square_id == case["square_id"])
            & (frames.time_utc == pd.Timestamp(case["time_utc"]))
        ]
        assert hit.empty


# --- 완료 조건 (3) 캐시가 있으면 원본을 다시 읽지 않음 ------------------------------------
def test_valid_cache_is_reused_without_reading_raw(workspace, monkeypatch):
    first = cache.build_daily_cache(DAY)
    raw_path(workspace, DAY).unlink()

    def forbidden(day):
        raise AssertionError("원본을 다시 읽음")

    monkeypatch.setattr(raw, "read_raw_day", forbidden)
    assert cache.build_daily_cache(DAY) == first


def test_force_and_agg_version_change_rebuild(workspace, monkeypatch):
    cache.build_daily_cache(DAY)
    calls = []
    original = raw.read_raw_day

    def spy(day):
        calls.append(day)
        return original(day)

    monkeypatch.setattr(raw, "read_raw_day", spy)
    cache.build_daily_cache(DAY, force=True)
    monkeypatch.setattr(config, "AGG_VERSION", config.AGG_VERSION + 1)
    cache.build_daily_cache(DAY)
    assert calls == [DAY, DAY]


def test_meta_records_present_times_and_version(workspace):
    cache.build_daily_cache(DAY)
    meta = cache.read_meta(DAY)
    assert meta["agg_version"] == config.AGG_VERSION
    assert meta["raw_file"] == raw_path(workspace, DAY).name
    assert len(meta["present_times"]) == GOLDEN["present_times_per_day"][DAY.isoformat()]
    assert meta["rows_outside_local_day"] == 0


# --- .txt.gz 지원 (D-12) --------------------------------------------------------------
def test_reads_gzip_when_txt_missing(workspace):
    txt = raw_path(workspace, DAY)
    expected = pd.read_parquet(cache.build_daily_cache(DAY))
    gz = txt.with_name(txt.name + ".gz")
    gz.write_bytes(gzip.compress(txt.read_bytes()))
    txt.unlink()
    actual = pd.read_parquet(cache.build_daily_cache(DAY, force=True))
    pd.testing.assert_frame_equal(expected, actual)
    assert cache.read_meta(DAY)["raw_file"] == gz.name


# --- 오류 코드 --------------------------------------------------------------------------
def test_e1001_lists_every_missing_day(workspace):
    raw_path(workspace, DAYS[0]).unlink()
    raw_path(workspace, DAYS[2]).unlink()
    with pytest.raises(TPError) as exc:
        cache.ensure_daily_caches(DAYS)
    assert exc.value.code == "E-1001"
    assert "2013-11-04" in exc.value.message and "2013-11-06" in exc.value.message


def test_e1001_not_raised_when_cache_exists_and_raw_dir_gone(workspace, monkeypatch):
    cache.ensure_daily_caches(DAYS)
    monkeypatch.setenv("TP_RAW_DIR", str(workspace / "nowhere"))
    cache.ensure_daily_caches(DAYS)


@pytest.mark.parametrize(
    "line",
    [
        pytest.param("1\t1383519600000\t39\t\t\t\t", id="7-columns"),
        pytest.param("1\t1383519600000\t39\t\t\t\t\t1.5\t9", id="9-columns"),
        pytest.param("x\t1383519600000\t39\t\t\t\t\t1.5", id="non-numeric"),
        pytest.param("0\t1383519600000\t39\t\t\t\t\t1.5", id="square-low"),
        pytest.param("10001\t1383519600000\t39\t\t\t\t\t1.5", id="square-high"),
        pytest.param("1\t1383519600000\t\t\t\t\t\t1.5", id="cc-missing"),
        pytest.param("1\t1383519600000\t39\t\t\t\t\t-0.1", id="negative"),
    ],
)
def test_e1002_bad_format(workspace, line):
    write_raw(workspace, DAY, [GOOD, line])
    with pytest.raises(TPError) as exc:
        raw.read_raw_day(DAY)
    assert exc.value.code == "E-1002"
    assert ":2" in exc.value.message  # 줄 번호


def test_e1003_time_not_on_10_minute_grid(workspace):
    write_raw(workspace, DAY, [GOOD, "1\t1383519660000\t39\t\t\t\t\t1.5"])
    with pytest.raises(TPError) as exc:
        raw.read_raw_day(DAY)
    assert exc.value.code == "E-1003"


def test_rows_outside_file_day_are_counted_not_rejected(workspace):
    write_raw(workspace, DAY, [GOOD, "1\t1383519000000\t39\t\t\t\t\t2.0"])  # 11-03 23:50 CET
    day = raw.read_raw_day(DAY)
    assert day.rows_outside_local_day == 1
    assert day.rows_outside_utc_day == 2  # 00:00 CET = 전날 23:00 UTC
    assert len(day.frame) == 2


# --- 진단 보고: 파일 전체에서 빠진 시각(결측 구간) -----------------------------------------
def test_inspect_reports_global_gaps(workspace):
    cache.ensure_daily_caches(DAYS)
    report = cache.write_inspect(DAYS)
    gaps = [(g["start_utc"], g["length"]) for g in report["gaps"]]
    expected = [
        (pd.Timestamp(g["start_utc"]).isoformat(), g["length"]) for g in GOLDEN["missing_gaps"]
    ]
    assert gaps == expected
    assert (config.INTERIM_DIR / "inspect.json").is_file()
    assert report["files"]["2013-11-05"]["rows_outside_local_day"] == 0


# --- 인터페이스: prepare (캐시 부분) ---------------------------------------------------
def test_prepare_builds_caches_and_inspect(workspace):
    assert cli.main(["prepare", "--phase", "dev"]) == 0
    for d in DAYS:
        assert cache.cache_path(d).is_file()
    assert (config.INTERIM_DIR / "inspect.json").is_file()


def test_prepare_exit_1_on_missing_raw(workspace, capsys):
    raw_path(workspace, DAYS[1]).unlink()
    assert cli.main(["prepare", "--phase", "dev"]) == 1
    assert "[E-1001]" in capsys.readouterr().err
