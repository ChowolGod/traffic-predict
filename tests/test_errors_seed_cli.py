import os
import subprocess
import sys

import numpy as np
import pytest
import torch

from tp import cli
from tp.errors import TPError
from tp.seed import set_seed


def test_tperror_format():
    err = TPError("E-2001", "분할 설정 오류: 겹침")
    assert err.code == "E-2001"
    assert str(err) == "[E-2001] 분할 설정 오류: 겹침"


def test_tperror_rejects_unknown_code():
    with pytest.raises(ValueError):
        TPError("E-9999", "없는 번호")


def test_set_seed_reproduces_numpy_and_torch():
    set_seed(0, threads=2)
    a = (np.random.rand(3), torch.randn(3))
    set_seed(0, threads=2)
    b = (np.random.rand(3), torch.randn(3))
    assert np.array_equal(a[0], b[0])
    assert torch.equal(a[1], b[1])
    assert torch.get_num_threads() == 2
    assert torch.are_deterministic_algorithms_enabled()


def test_main_maps_tperror_to_exit_1(monkeypatch, capsys):
    def boom(args):
        raise TPError("E-2001", "분할 설정 오류: 테스트")

    monkeypatch.setitem(cli.HANDLERS, "results", boom)
    assert cli.main(["results"]) == 1
    assert "[E-2001] 분할 설정 오류: 테스트" in capsys.readouterr().err


def test_main_maps_unexpected_error_to_exit_2(monkeypatch):
    def boom(args):
        raise RuntimeError("unexpected")

    monkeypatch.setitem(cli.HANDLERS, "results", boom)
    assert cli.main(["results"]) == 2


def test_parser_accepts_planned_interface():
    parser = cli.build_parser()
    assert parser.parse_args(["prepare", "--phase", "dev", "--force"]).force
    assert parser.parse_args(["run", "--config", "x.yaml", "--retry"]).retry
    args = parser.parse_args(
        [
            "sweep",
            "--parent",
            "EXP-001",
            "--key",
            "naive.lag",
            "--values",
            "[1]",
            "--start-id",
            "EXP-002",
            "--name",
            "lag",
            "--reason",
            "r",
            "--hypothesis",
            "h",
        ]
    )
    assert args.key == "naive.lag"
    assert parser.parse_args(["test", "--final", "f.yaml", "--confirm"]).confirm
    with pytest.raises(SystemExit):
        parser.parse_args(["prepare", "--phase", "prod"])


def test_module_entry_sets_thread_env_from_config(tmp_path):
    cfg = tmp_path / "exp.yaml"
    cfg.write_text("runtime:\n  threads: 3\n", encoding="utf-8")
    code = (
        "import sys, os; from tp.__main__ import configure_threads; "
        f"configure_threads(['run', '--config', r'{cfg}']); "
        "print(os.environ['OMP_NUM_THREADS'], os.environ['MKL_NUM_THREADS'])"
    )
    env = {k: v for k, v in os.environ.items() if k not in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert out.stdout.split() == ["3", "3"], out.stderr


def test_module_help_exits_0():
    out = subprocess.run([sys.executable, "-m", "tp", "--help"], capture_output=True, text=True)
    assert out.returncode == 0
    assert "prepare" in out.stdout
