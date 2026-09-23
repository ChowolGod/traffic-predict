"""LSTM input scaling fitted on train only (M-06). ARIMA is not scaled."""

import numpy as np


class Scaler:
    def __init__(self, kind: str):
        if kind not in ("standard", "minmax"):
            raise ValueError(f"unknown scaler: {kind}")
        self.kind = kind
        self.a = 0.0  # mean or min
        self.b = 1.0  # std or (max - min)

    def fit(self, train: np.ndarray) -> "Scaler":
        if self.kind == "standard":
            self.a, self.b = float(np.mean(train)), float(np.std(train))
        else:
            self.a, self.b = float(np.min(train)), float(np.max(train) - np.min(train))
        if self.b == 0.0:
            self.b = 1.0
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.a) / self.b

    def inverse(self, z: np.ndarray) -> np.ndarray:
        return z * self.b + self.a

    def to_dict(self) -> dict:
        if self.kind == "standard":
            return {"kind": self.kind, "mean": self.a, "std": self.b}
        return {"kind": self.kind, "min": self.a, "range": self.b}

    @classmethod
    def from_dict(cls, data: dict) -> "Scaler":
        scaler = cls(data["kind"])
        if scaler.kind == "standard":
            scaler.a, scaler.b = data["mean"], data["std"]
        else:
            scaler.a, scaler.b = data["min"], data["range"]
        return scaler
