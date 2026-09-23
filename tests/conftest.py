import shutil
from pathlib import Path

import pytest
import yaml

from tp import config

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def isolate_outputs(tmp_path, monkeypatch):
    """No test may write into the real repo's data/experiments/results/configs dirs."""
    guard = tmp_path / "_guard"
    monkeypatch.setattr(config, "INTERIM_DIR", guard / "interim")
    monkeypatch.setattr(config, "PROCESSED_DIR", guard / "processed")
    monkeypatch.setattr(config, "EXPERIMENTS_DIR", guard / "experiments")
    monkeypatch.setattr(config, "RESULTS_DIR", guard / "results")
    monkeypatch.setenv("TP_RAW_DIR", str(guard / "raw"))


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """Isolated raw/interim/processed/configs dirs with the fixture raw files copied in."""
    raw = tmp_path / "raw"
    shutil.copytree(FIXTURES / "raw", raw)
    monkeypatch.setenv("TP_RAW_DIR", str(raw))
    monkeypatch.setattr(config, "INTERIM_DIR", tmp_path / "interim")
    monkeypatch.setattr(config, "PROCESSED_DIR", tmp_path / "processed")
    configs = tmp_path / "configs"
    configs.mkdir()
    phases = {
        "dev": {"train": ["2013-11-04", "2013-11-04"], "val": ["2013-11-05", "2013-11-05"], "k": 1}
    }
    (configs / "phases.yaml").write_text(yaml.safe_dump(phases), encoding="utf-8")
    monkeypatch.setattr(config, "CONFIGS_DIR", configs)
    monkeypatch.setattr(config, "EXPERIMENTS_DIR", tmp_path / "experiments")
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path / "results")
    from tp.exp import registry

    monkeypatch.setattr(registry, "git_state", lambda: ("abc1234", False))
    return tmp_path
