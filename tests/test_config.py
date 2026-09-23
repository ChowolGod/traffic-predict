from datetime import date
from pathlib import Path

import pytest
import yaml

from tp import config
from tp.errors import TPError

VALID = {
    "dev": {
        "train": ["2013-11-04", "2013-11-13"],
        "val": ["2013-11-14", "2013-11-17"],
        "k": 1,
    },
    "full": {
        "train": ["2013-11-01", "2013-12-08"],
        "val": ["2013-12-09", "2013-12-15"],
        "test": ["2013-12-16", "2013-12-22"],
        "k": 1,
    },
}


def write_phases(tmp_path: Path, data: dict) -> Path:
    path = tmp_path / "phases.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_repo_phases_yaml_matches_plan():
    dev = config.load_phase("dev")
    full = config.load_phase("full")
    assert dev.train == (date(2013, 11, 4), date(2013, 11, 13))
    assert dev.val == (date(2013, 11, 14), date(2013, 11, 17))
    assert dev.test is None
    assert full.train == (date(2013, 11, 1), date(2013, 12, 8))
    assert full.val == (date(2013, 12, 9), date(2013, 12, 15))
    assert full.test == (date(2013, 12, 16), date(2013, 12, 22))
    assert dev.k == 1 and full.k == 3  # full k: 1 → 3 by F-11 (plan 2장 Should)


def test_phase_span_covers_all_segments():
    full = config.load_phase("full")
    assert full.start == date(2013, 11, 1)
    assert full.end == date(2013, 12, 22)


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda d: d["dev"].update(train=["2013-11-13", "2013-11-04"]), id="reversed"),
        pytest.param(lambda d: d["dev"].update(val=["2013-11-13", "2013-11-17"]), id="overlap"),
        pytest.param(lambda d: d["dev"].update(test=["2013-11-18", "2013-11-19"]), id="dev-test"),
        pytest.param(lambda d: d["full"].update(test=["2013-12-16", "2013-12-23"]), id="range"),
        pytest.param(lambda d: d["full"].update(train=["2013-10-31", "2013-12-08"]), id="early"),
        pytest.param(lambda d: d["dev"].update(k=4), id="k-high"),
        pytest.param(lambda d: d["dev"].update(k=0), id="k-low"),
        pytest.param(lambda d: d["dev"].pop("val"), id="missing-key"),
        pytest.param(lambda d: d["dev"].update(train=["2013-11-04"]), id="bad-range"),
        pytest.param(lambda d: d["dev"].update(train=["2013-11-04", "nope"]), id="bad-date"),
    ],
)
def test_invalid_phase_raises_e2001(tmp_path, mutate):
    data = {name: dict(cfg) for name, cfg in VALID.items()}
    mutate(data)
    path = write_phases(tmp_path, data)
    with pytest.raises(TPError) as exc:
        config.load_phase("dev", path)
        config.load_phase("full", path)
    assert exc.value.code == "E-2001"


def test_unknown_phase_raises_e2001():
    with pytest.raises(TPError) as exc:
        config.load_phase("prod")
    assert exc.value.code == "E-2001"


def test_raw_dir_default_and_env(monkeypatch, tmp_path):
    monkeypatch.delenv("TP_RAW_DIR", raising=False)
    assert config.raw_dir() == config.ROOT / "data" / "raw"
    monkeypatch.setenv("TP_RAW_DIR", str(tmp_path))
    assert config.raw_dir() == tmp_path


def test_constants():
    assert config.LSTM_SEEDS == (0, 1, 2)
    assert config.TZ == "Europe/Rome"
    assert isinstance(config.AGG_VERSION, int)
    assert isinstance(config.IMPUTE_VERSION, int)
