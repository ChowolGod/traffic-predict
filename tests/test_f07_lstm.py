import json

import numpy as np
import pandas as pd
import pytest

from tests.test_f06_arima import synthetic
from tp import cli
from tp.errors import TPError
from tp.exp import registry
from tp.models import lstm
from tp.prep.scale import Scaler
from tp.seed import set_seed

SMALL = {
    "lstm.window": 24,
    "lstm.hidden": 8,
    "lstm.layers": 1,
    "lstm.dropout": 0.0,
    "lstm.lr": 1e-2,
    "lstm.batch": 64,
    "lstm.max_epochs": 4,
    "lstm.patience": 2,
    "lstm.scaler": "standard",
    "runtime.threads": 2,
}


def never_late() -> None:
    pass


def train(s: pd.DataFrame, seed: int = 0, cfg: dict | None = None):
    set_seed(seed, 2)
    return lstm.train_lstm(s, cfg or SMALL, seed, never_late)


# --- M-06 스케일러: train 통계만 사용 ------------------------------------------------------
@pytest.mark.parametrize("kind", ["standard", "minmax"])
def test_scaler_fits_train_only_and_inverts(kind):
    s = synthetic()
    scaler = Scaler(kind).fit(s.loc[s["segment"] == "train", "y"].to_numpy())
    stats = scaler.to_dict()
    s2 = s.copy()
    s2.loc[s2["segment"] == "val", "y"] *= 1000
    assert Scaler(kind).fit(s2.loc[s2["segment"] == "train", "y"].to_numpy()).to_dict() == stats
    x = s["y"].to_numpy()
    assert np.allclose(scaler.inverse(scaler.transform(x)), x)
    assert Scaler.from_dict(stats).to_dict() == stats


def test_lstm_scaler_uses_train_only():
    s = synthetic()
    model = train(s)
    train_y = s.loc[s["segment"] == "train", "y"].to_numpy()
    assert model.scaler.to_dict()["mean"] == pytest.approx(train_y.mean())


# --- 창 데이터셋: 목표가 채운 칸이면 제외 ---------------------------------------------------
def test_train_windows_skip_imputed_targets():
    s = synthetic()
    base = lstm.count_train_samples(s, 24)
    s.loc[s.index[100:103], "is_imputed"] = True
    assert lstm.count_train_samples(s, 24) == base - 3
    assert base == (s["segment"] == "train").sum() - 24


# --- 완료 조건 (3) 같은 시드면 1e-6 이내 동일, 다른 시드면 다름 -----------------------------
def test_same_seed_reproducible_different_seed_differs():
    s = synthetic()
    targets = s.index[24:]
    p1 = lstm.predict_lstm(train(s, 0), s, targets)
    p2 = lstm.predict_lstm(train(s, 0), s, targets)
    p3 = lstm.predict_lstm(train(s, 1), s, targets)
    assert np.allclose(p1.to_numpy(), p2.to_numpy(), atol=1e-6, rtol=0)
    assert not np.allclose(p1.to_numpy(), p3.to_numpy(), atol=1e-6, rtol=0)


# --- 완료 조건 (1) 학습 곡선, 조기 종료, 최저 val 가중치 복원 --------------------------------
def test_history_and_best_weights_restored():
    s = synthetic()
    model = train(s)
    h = model.history
    assert list(h.columns) == ["epoch", "train_loss", "val_loss"]
    assert 1 <= len(h) <= SMALL["lstm.max_epochs"]
    assert model.best_epoch == int(h.loc[h["val_loss"].idxmin(), "epoch"])
    assert lstm.val_loss(model, s) == pytest.approx(h["val_loss"].min(), rel=1e-5)


def test_early_stopping_stops_after_patience(monkeypatch):
    s = synthetic()
    losses = iter([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    monkeypatch.setattr(lstm, "val_loss", lambda model, series: next(losses))
    model = train(s, cfg={**SMALL, "lstm.max_epochs": 6, "lstm.patience": 2})
    assert len(model.history) == 3 and model.best_epoch == 1


def test_one_step_is_causal():
    # h=1 only; h>1 causality is tested in test_f12_horizon
    s = synthetic()
    model = train(s)
    targets = s.index[24:]
    base = lstm.predict_lstm(model, s, targets)
    k = s.index[s["segment"] == "val"][50]
    changed = s.copy()
    changed.loc[k, "y"] += 1000.0
    after = lstm.predict_lstm(model, changed, targets)
    assert np.allclose(after.loc[:k].to_numpy(), base.loc[:k].to_numpy(), atol=1e-9)
    assert abs(after.loc[k + 1] - base.loc[k + 1]) > 1e-3


# --- D-11 저장·로드 -----------------------------------------------------------------------
def test_save_load_roundtrip(tmp_path):
    s = synthetic()
    model = train(s)
    lstm.save_model(model, tmp_path / "model", seed=0)
    loaded = lstm.load_model(tmp_path / "model", seed=0, cfg=SMALL)
    targets = s.index[24:]
    pd.testing.assert_series_equal(
        lstm.predict_lstm(loaded, s, targets), lstm.predict_lstm(model, s, targets)
    )
    assert (tmp_path / "model" / "lstm_seed0.pt").is_file()
    assert json.loads((tmp_path / "model" / "scaler.json").read_text("utf-8"))["kind"] == "standard"


def test_curve_files(tmp_path):
    s = synthetic()
    model = train(s)
    lstm.save_curve(model.history, model.best_epoch, tmp_path / "curves", seed=0)
    csv = pd.read_csv(tmp_path / "curves" / "seed0.csv")
    assert list(csv.columns) == ["epoch", "train_loss", "val_loss"]
    assert (tmp_path / "curves" / "seed0.png").stat().st_size > 1000


# --- 오류 코드 ----------------------------------------------------------------------------
def test_e3002_nan_loss(monkeypatch):
    s = synthetic()
    original = lstm.LSTMNet.forward
    monkeypatch.setattr(lstm.LSTMNet, "forward", lambda self, x: original(self, x) * float("nan"))
    with pytest.raises(TPError) as exc:
        train(s)
    assert exc.value.code == "E-3002"
    assert "seed=0" in exc.value.message


def test_e3003_checked_every_epoch():
    s = synthetic()
    calls = []

    def late():
        calls.append(1)
        if len(calls) == 2:
            raise TPError("E-3003", "시간 초과: test")

    set_seed(0, 2)
    with pytest.raises(TPError) as exc:
        lstm.train_lstm(s, SMALL, 0, late)
    assert exc.value.code == "E-3003" and len(calls) == 2


# --- 레지스트리 연동 ----------------------------------------------------------------------
@pytest.fixture
def lstm_workspace(workspace, monkeypatch):
    small = {k: v for k, v in SMALL.items() if k.startswith("lstm.")} | {"lstm.window": 12}
    monkeypatch.setitem(registry.INITIAL, "lstm", small)
    assert cli.main(["prepare", "--phase", "dev"]) == 0
    return workspace


def lstm_child(**overrides):
    from tests.test_f08_registry import child

    small = {k.split(".")[1]: v for k, v in registry.INITIAL["lstm"].items()}
    c = child(changed="model.type", model={"type": "lstm"}, lstm={**small, **overrides})
    del c["naive"]
    return c


def test_registry_runs_lstm_three_seeds(lstm_workspace):
    from tests.test_f08_registry import ROOT_CFG, folder, meta, run

    assert run(ROOT_CFG) == 0
    assert run(lstm_child()) == 0
    f = folder("EXP-002")
    assert meta("EXP-002")["status"] == "completed"
    for seed in (0, 1, 2):
        assert (f / "curves" / f"seed{seed}.csv").is_file()
        assert (f / "curves" / f"seed{seed}.png").is_file()
        assert (f / "model" / f"lstm_seed{seed}.pt").is_file()
    assert (f / "model" / "scaler.json").is_file()
    m = json.loads((f / "metrics.json").read_text(encoding="utf-8"))
    assert set(m["seeds"]) == {"0", "1", "2"} and m["val_std"] is not None
    pred = pd.read_parquet(f / "predictions.parquet")
    assert sorted(pred["seed"].unique()) == [0, 1, 2]


def test_e4001_too_few_training_samples(lstm_workspace, monkeypatch):
    from tests.test_f08_registry import ROOT_CFG, run, run_err

    assert run(ROOT_CFG) == 0
    monkeypatch.setitem(registry.INITIAL["lstm"], "lstm.batch", 1024)
    assert run_err(lstm_child(batch=1024)) == "E-4001"
