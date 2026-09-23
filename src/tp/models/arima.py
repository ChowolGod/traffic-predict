"""ARIMA fitted on train, applied without refitting for 1-step predictions (M-08, plan 4.2)."""

import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.arima.model import ARIMA

from tp import config
from tp.errors import TPError

log = logging.getLogger(__name__)

SEASON = 144
MAXITER = 200


def fourier_terms(times: pd.Series, k: int) -> pd.DataFrame:
    """Daily sin/cos terms by the Europe/Rome slot of the day (0..143)."""
    local = pd.DatetimeIndex(times).tz_convert(config.TZ)
    slot = (local.hour * 60 + local.minute).to_numpy() // 10
    cols = {}
    for i in range(1, k + 1):
        angle = 2 * np.pi * i * slot / SEASON
        cols[f"sin{i}"] = np.sin(angle)
        cols[f"cos{i}"] = np.cos(angle)
    return pd.DataFrame(cols)


def _design(df: pd.DataFrame, spec: dict) -> tuple[np.ndarray, np.ndarray | None, int]:
    y = df["y"].to_numpy(dtype=float)
    start = SEASON if spec["seasonal"] == "diff144" else 0
    endog = y[SEASON:] - y[:-SEASON] if start else y
    exog = None
    if spec["seasonal"] == "fourier":
        exog = fourier_terms(df["time_utc"], spec["fourier_k"]).to_numpy()
    return endog, exog, start


def _spec(cfg: dict) -> dict:
    return {
        "order": list(cfg["arima.order"]),
        "seasonal": cfg["arima.seasonal"],
        "fourier_k": cfg.get("arima.fourier_k"),
    }


def fit_arima(train: pd.DataFrame, cfg: dict) -> dict:
    spec = _spec(cfg)
    endog, exog, _ = _design(train, spec)
    model = ARIMA(endog, exog=exog, order=tuple(spec["order"]))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = model.fit(method_kwargs={"maxiter": MAXITER})
    converged = (getattr(res, "mle_retvals", None) or {}).get("converged", True)
    if not converged or any(issubclass(w.category, ConvergenceWarning) for w in caught):
        raise TPError("E-3001", f"ARIMA 수렴 실패: {spec['order']} {spec['seasonal']}")
    for w in caught:
        log.info("statsmodels 경고: %s", w.message)
    return {**spec, "param_names": list(res.param_names), "params": [float(p) for p in res.params]}


def _at(matrix: np.ndarray, t: np.ndarray | int) -> np.ndarray:
    """Pick time t from a statsmodels system matrix (last axis is 1 when time-invariant)."""
    return matrix[..., 0] if matrix.shape[-1] == 1 else matrix[..., t]


def _h_step(res, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """h-step-ahead forecasts from every origin o: Z (T^(h-1) a_{o+1|o} + sum T^k c) + d.

    Returns (endog indices of the targets, forecasts). Exact for the Kalman filter used by
    statsmodels; checked against filter-then-forecast per origin in tests."""
    fr = res.filter_results
    transition, intercept = _at(fr.transition, 0), _at(fr.state_intercept, 0)
    if fr.transition.shape[-1] != 1 or fr.state_intercept.shape[-1] != 1:
        raise NotImplementedError("time-varying state equation is not supported")
    n = fr.nobs
    states = fr.predicted_state[:, 1 : n - horizon + 2]  # a_{o+1|o} for o = 0 .. n - horizon
    for _ in range(horizon - 1):
        states = transition @ states + intercept[:, None]
    targets = np.arange(horizon, n + 1)[: states.shape[1]]
    targets = targets[targets < n]
    states = states[:, : len(targets)]
    design = fr.design
    if design.shape[-1] == 1:
        forecast = design[0, :, 0] @ states
    else:
        forecast = np.einsum("kt,kt->t", design[0][:, targets], states)
    return targets, forecast + np.atleast_2d(_at(fr.obs_intercept, targets))[0]


def predict_arima(params: dict, df: pd.DataFrame, horizon: int = 1) -> pd.Series:
    """Rolling predictions of y[t] from origin t - horizon with fixed params (no refit)."""
    endog, exog, start = _design(df, params)
    model = ARIMA(endog, exog=exog, order=tuple(params["order"]))
    res = model.filter(np.asarray(params["params"]))
    y = df["y"].to_numpy(dtype=float)
    if horizon == 1:
        z_hat = np.asarray(res.predict())
        y_hat = z_hat + y[:-SEASON] if start else z_hat
        return pd.Series(y_hat, index=df.index[start:])
    targets, z_hat = _h_step(res, horizon)
    y_hat = z_hat + y[targets] if start else z_hat  # diff144: y[t - 144] is known at t - h
    return pd.Series(y_hat, index=df.index[start + targets])


def save_params(path: Path, params: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(params, indent=2), encoding="utf-8")


def load_params(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
