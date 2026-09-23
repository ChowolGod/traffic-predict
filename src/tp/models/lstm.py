"""Window LSTM for 1-step prediction, early stopping on val, per-seed curves (M-09, plan 4.2)."""

import copy
import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

from tp.errors import TPError  # noqa: E402
from tp.prep.scale import Scaler  # noqa: E402

PREDICT_CHUNK = 4096
# Curve colours: validated default palette slots 1-2 and chart chrome (dataviz skill).
TRAIN_COLOR, VAL_COLOR = "#2a78d6", "#eb6834"
SURFACE, INK, INK_2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e8e7e3"


class LSTMNet(nn.Module):
    def __init__(self, hidden: int, layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, num_layers=layers, dropout=dropout, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


@dataclass
class TrainedLSTM:
    net: LSTMNet
    scaler: Scaler
    window: int
    horizon: int = 1
    history: pd.DataFrame | None = None
    best_epoch: int | None = None


def _offset(window: int, horizon: int) -> int:
    """Position of the first target: its input window ends horizon steps earlier."""
    return window + horizon - 1


def _windows(model: TrainedLSTM, series: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Inputs ending at t - horizon for every target position t >= window + horizon - 1."""
    z = model.scaler.transform(series["y"].to_numpy(dtype=float)).astype(np.float32)
    offset = _offset(model.window, model.horizon)
    x = np.lib.stride_tricks.sliding_window_view(z, model.window)[: len(z) - offset]
    return x, z[offset:]


def _mask(series: pd.DataFrame, window: int, segment: str, horizon: int = 1) -> np.ndarray:
    offset = _offset(window, horizon)
    seg = series["segment"].astype(str).to_numpy()[offset:]
    imputed = series["is_imputed"].to_numpy(dtype=bool)[offset:]
    return (seg == segment) & ~imputed


def count_train_samples(series: pd.DataFrame, window: int, horizon: int = 1) -> int:
    return int(_mask(series, window, "train", horizon).sum())


def _mse(model: TrainedLSTM, x: np.ndarray, y: np.ndarray) -> float:
    model.net.eval()
    total = 0.0
    with torch.no_grad():
        for i in range(0, len(x), PREDICT_CHUNK):
            xb = torch.from_numpy(np.ascontiguousarray(x[i : i + PREDICT_CHUNK])).unsqueeze(-1)
            pred = model.net(xb)
            total += float(((pred - torch.from_numpy(y[i : i + PREDICT_CHUNK])) ** 2).sum())
    return total / len(x)


def val_loss(model: TrainedLSTM, series: pd.DataFrame) -> float:
    x, y = _windows(model, series)
    sel = _mask(series, model.window, "val", model.horizon)
    return _mse(model, x[sel], y[sel])


def train_lstm(
    series: pd.DataFrame, cfg: dict, seed: int, check_time: Callable[[], None]
) -> TrainedLSTM:
    """Caller fixes the seed first (set_seed); shuffling uses its own seeded generator."""
    window = cfg["lstm.window"]
    train_y = series.loc[series["segment"].astype(str) == "train", "y"].to_numpy(dtype=float)
    model = TrainedLSTM(
        net=LSTMNet(cfg["lstm.hidden"], cfg["lstm.layers"], cfg["lstm.dropout"]),
        scaler=Scaler(cfg["lstm.scaler"]).fit(train_y),
        window=window,
        horizon=cfg.get("horizon", 1),
    )
    x, y = _windows(model, series)
    sel = _mask(series, window, "train", model.horizon)
    x_train = torch.from_numpy(np.ascontiguousarray(x[sel])).unsqueeze(-1)
    y_train = torch.from_numpy(y[sel])
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.Adam(model.net.parameters(), lr=cfg["lstm.lr"])
    loss_fn = nn.MSELoss(reduction="sum")

    rows, best, best_state, wait = [], math.inf, None, 0
    for epoch in range(1, cfg["lstm.max_epochs"] + 1):
        model.net.train()
        total = 0.0
        for batch in torch.randperm(len(x_train), generator=generator).split(cfg["lstm.batch"]):
            optimizer.zero_grad()
            loss = loss_fn(model.net(x_train[batch]), y_train[batch])
            loss.backward()
            optimizer.step()
            total += loss.item()
        train_loss = total / len(x_train)
        current = val_loss(model, series)
        if not (math.isfinite(train_loss) and math.isfinite(current)):
            raise TPError("E-3002", f"학습 발산: seed={seed} epoch={epoch}")
        rows.append({"epoch": epoch, "train_loss": train_loss, "val_loss": current})
        if current < best:
            best, best_state, model.best_epoch, wait = (
                current, copy.deepcopy(model.net.state_dict()), epoch, 0,
            )  # fmt: skip
        else:
            wait += 1
        check_time()
        if wait >= cfg["lstm.patience"]:
            break
    model.net.load_state_dict(best_state)
    model.history = pd.DataFrame(rows, columns=["epoch", "train_loss", "val_loss"])
    return model


def predict_lstm(model: TrainedLSTM, series: pd.DataFrame, targets: pd.Index) -> pd.Series:
    """Original-scale predictions for target row labels (each needs window + horizon - 1 rows)."""
    x, _ = _windows(model, series)
    positions = series.index.get_indexer(targets) - _offset(model.window, model.horizon)
    if (positions < 0).any():
        raise TPError("E-4004", f"평가 시각 불일치: {(positions < 0).sum()} (window 기록 부족)")
    out = []
    model.net.eval()
    with torch.no_grad():
        for i in range(0, len(positions), PREDICT_CHUNK):
            xb = np.ascontiguousarray(x[positions[i : i + PREDICT_CHUNK]])
            out.append(model.net(torch.from_numpy(xb).unsqueeze(-1)).numpy())
    return pd.Series(model.scaler.inverse(np.concatenate(out).astype(float)), index=targets)


def save_model(model: TrainedLSTM, folder: Path, seed: int) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    torch.save(model.net.state_dict(), folder / f"lstm_seed{seed}.pt")
    (folder / "scaler.json").write_text(json.dumps(model.scaler.to_dict()), encoding="utf-8")


def load_model(folder: Path, seed: int, cfg: dict) -> TrainedLSTM:
    net = LSTMNet(cfg["lstm.hidden"], cfg["lstm.layers"], cfg["lstm.dropout"])
    net.load_state_dict(torch.load(folder / f"lstm_seed{seed}.pt", weights_only=True))
    scaler = Scaler.from_dict(json.loads((folder / "scaler.json").read_text(encoding="utf-8")))
    return TrainedLSTM(
        net=net, scaler=scaler, window=cfg["lstm.window"], horizon=cfg.get("horizon", 1)
    )


def save_curve(history: pd.DataFrame, best_epoch: int, folder: Path, seed: int) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    history.to_csv(folder / f"seed{seed}.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    last = history.iloc[-1]
    upper = "train_loss" if last["train_loss"] >= last["val_loss"] else "val_loss"
    for column, color, label in (
        ("train_loss", TRAIN_COLOR, "train"),
        ("val_loss", VAL_COLOR, "val"),
    ):
        ax.plot(history["epoch"], history[column], color=color, linewidth=2, label=label)
        ax.annotate(label, (last["epoch"], last[column]),
                    xytext=(6, 7 if column == upper else -7), textcoords="offset points",
                    va="center", color=INK_2, fontsize=9)  # fmt: skip
    ax.set_yscale("log")
    best = history.loc[history["epoch"] == best_epoch, "val_loss"].iloc[0]
    ax.plot([best_epoch], [best], "o", markersize=8, color=VAL_COLOR,
            markeredgecolor=SURFACE, markeredgewidth=2)  # fmt: skip
    ax.annotate(f"best epoch {best_epoch}", (best_epoch, best), xytext=(0, -14),
                textcoords="offset points", ha="center", color=INK_2, fontsize=9)  # fmt: skip
    ax.set_title(f"Learning curve (seed {seed}, MSE on scaled y)", color=INK, fontsize=11,
                 loc="left")  # fmt: skip
    ax.set_xlabel("epoch", color=MUTED, fontsize=9)
    ax.set_ylabel("loss", color=MUTED, fontsize=9)
    ax.tick_params(which="both", colors=MUTED, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2)
    fig.tight_layout()
    fig.savefig(folder / f"seed{seed}.png", facecolor=SURFACE)
    plt.close(fig)
