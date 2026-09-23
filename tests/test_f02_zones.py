import json
import logging
from datetime import date

import pandas as pd
import pytest

from tp import cli, config
from tp.data import cache, zones
from tp.errors import TPError

DAYS = [date(2013, 11, 4), date(2013, 11, 5), date(2013, 11, 6)]


def dev_phase():
    return config.load_phase("dev")  # train 11-04..11-05, val 11-06 (conftest)


def train_totals() -> pd.Series:
    frames = pd.concat(pd.read_parquet(cache.cache_path(d)) for d in DAYS)
    local_day = frames["time_utc"].dt.tz_convert(config.TZ).dt.date
    train = frames[(local_day >= DAYS[0]) & (local_day <= DAYS[1])]
    return train.groupby("square_id")["internet"].sum()


def overwrite_cache(day: date, frame: pd.DataFrame) -> None:
    frame.to_parquet(cache.cache_path(day), index=False)


def row(square_id: int, local: str, value: float) -> dict:
    ts = pd.Timestamp(local, tz=config.TZ).tz_convert("UTC").as_unit("ns")
    return {"square_id": square_id, "time_utc": ts, "internet": value}


def frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    return df.astype({"square_id": "int16", "internet": "float64"})


def test_top_k_by_train_total(workspace):
    cache.ensure_daily_caches(DAYS)
    result = zones.select_top_zones(dev_phase(), k=3)
    expected = train_totals().sort_values(ascending=False).head(3)
    assert [z["square_id"] for z in result["zones"]] == list(expected.index)
    assert [z["rank"] for z in result["zones"]] == [1, 2, 3]
    for z in result["zones"]:
        assert z["train_total"] == pytest.approx(expected[z["square_id"]], rel=1e-12)
    assert result["zones"][0]["square_id"] == 5000


def test_val_data_is_not_used(workspace):
    cache.ensure_daily_caches(DAYS)
    before = zones.select_top_zones(dev_phase(), k=3)
    # val 날짜(11-06) 캐시와, train 날짜 파일에 섞인 11-06 00:00 행에
    # 큰 값을 넣어도 결과가 같아야 함
    overwrite_cache(DAYS[2], frame([row(10000, "2013-11-06 12:00", 1e12)]))
    tail = pd.read_parquet(cache.cache_path(DAYS[1]))
    extra = frame([row(10000, "2013-11-06 00:00", 1e12)])
    overwrite_cache(DAYS[1], pd.concat([tail, extra], ignore_index=True))
    after = zones.select_top_zones(dev_phase(), k=3)
    assert after["zones"] == before["zones"]


def test_tie_breaks_on_smaller_square_id(workspace):
    cache.ensure_daily_caches(DAYS)
    rows = [row(7, "2013-11-04 01:00", 5.0), row(3, "2013-11-04 01:00", 5.0)]
    overwrite_cache(DAYS[0], frame(rows))
    overwrite_cache(DAYS[1], frame([row(9, "2013-11-05 01:00", 1.0)]))
    result = zones.select_top_zones(dev_phase(), k=3)
    assert [z["square_id"] for z in result["zones"]] == [3, 7, 9]


def test_deterministic(workspace):
    cache.ensure_daily_caches(DAYS)
    assert zones.select_top_zones(dev_phase(), 3) == zones.select_top_zones(dev_phase(), 3)


def test_e1001_when_train_cache_missing(workspace):
    cache.ensure_daily_caches(DAYS)
    cache.cache_path(DAYS[1]).unlink()
    with pytest.raises(TPError) as exc:
        zones.select_top_zones(dev_phase(), k=1)
    assert exc.value.code == "E-1001"
    assert "2013-11-05" in exc.value.message


def test_ensure_zones_writes_d03_and_reuses_until_k_changes(workspace, monkeypatch):
    cache.ensure_daily_caches(DAYS)
    phase = dev_phase()
    path = zones.ensure_zones(phase)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["phase"] == "dev" and data["k"] == 1
    assert data["train_range"] == ["2013-11-04", "2013-11-05"]
    assert set(data["zones"][0]) == {"rank", "square_id", "train_total"}

    calls = []
    original = zones.select_top_zones
    monkeypatch.setattr(zones, "select_top_zones", lambda p, k: calls.append(k) or original(p, k))
    zones.ensure_zones(phase)
    assert calls == []
    zones.ensure_zones(phase.__class__(**{**phase.__dict__, "k": 2}))
    assert calls == [2]


def test_warns_when_dev_and_full_top_zone_differ(workspace, caplog):
    cache.ensure_daily_caches(DAYS)
    other = config.PROCESSED_DIR / "full"
    other.mkdir(parents=True)
    (other / "zones.json").write_text(
        json.dumps({"phase": "full", "zones": [{"rank": 1, "square_id": 1, "train_total": 1}]}),
        encoding="utf-8",
    )
    with caplog.at_level(logging.WARNING):
        zones.ensure_zones(dev_phase())
    assert "상위 구역" in caplog.text


def test_prepare_writes_zones(workspace):
    assert cli.main(["prepare", "--phase", "dev"]) == 0
    data = json.loads((config.PROCESSED_DIR / "dev" / "zones.json").read_text(encoding="utf-8"))
    assert data["zones"][0]["square_id"] == 5000
