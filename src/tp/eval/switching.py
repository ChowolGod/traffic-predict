"""Rule-based capacity-cell on/off simulation (M-19, F-14, plan 3.3 켜기/끄기 시뮬레이션)."""

import numpy as np

from tp.eval.decision import SLOT_MIN, episodes


def lag_slots(horizon: int, delay_min: int) -> int:
    """ŷ_t is available at t - h + 1 and acts d slots later, so slot t can only use ŷ_{t - L}."""
    return max(0, delay_min // SLOT_MIN + 1 - horizon)


def simulate(
    y_pred: np.ndarray,
    imputed: np.ndarray,
    thr: float,
    horizon: int,
    hold_min: int,
    on_ratio: float,
    delay_min: int,
) -> np.ndarray:
    """Consecutive 10-minute target slots -> cell state per slot (True = on); starts off."""
    lag = lag_slots(horizon, delay_min)
    wanted = np.zeros(len(y_pred), dtype=bool)
    wanted[lag:] = y_pred[: len(y_pred) - lag] >= on_ratio * thr
    wanted &= ~imputed
    hold = hold_min // SLOT_MIN
    state = np.zeros(len(y_pred), dtype=bool)
    on, since = False, 0
    for t, want in enumerate(wanted):
        if not on and want:
            on, since = True, t
        elif on and t - since >= hold and not want:
            on = False
        state[t] = on
    return state


def always_on(n: int) -> np.ndarray:
    return np.ones(n, dtype=bool)


def oracle(y_true: np.ndarray, imputed: np.ndarray, thr: float) -> np.ndarray:
    """On exactly in congested slots (perfect foresight, no delay)."""
    return (y_true >= thr) & ~imputed


def switching_metrics(
    state: np.ndarray, y_true: np.ndarray, imputed: np.ndarray, thr: float, merge_gap: int
) -> dict:
    congested = (y_true >= thr) & ~imputed
    spans = episodes(congested, merge_gap)
    return {
        "episodes": len(spans),
        "missed_min": int((congested & ~state).sum()) * SLOT_MIN,
        "missed_episodes": sum(1 for start, _ in spans if not state[start]),
        "on_min": int(state.sum()) * SLOT_MIN,
        "switches": int(state[0]) + int((state[1:] & ~state[:-1]).sum()),
    }


def choose_r1(rows: list[dict]) -> tuple[int, float]:
    """Rule r1 for one (family, horizon, delay). Rows: one per (zone, hold_min, on_ratio) with
    missed_min, on_min, switches (LSTM: seed means). A combination qualifies when every zone
    misses no more than the baseline (hold 0, ratio 1.0); then least total on time, fewest
    total switches, smaller hold, larger ratio."""
    combos: dict[tuple[int, float], dict[int, dict]] = {}
    for r in rows:
        combos.setdefault((int(r["hold_min"]), float(r["on_ratio"])), {})[int(r["zone_rank"])] = r
    baseline = combos[(0, 1.0)]

    def qualifies(zones: dict[int, dict]) -> bool:
        return all(zones[z]["missed_min"] <= baseline[z]["missed_min"] + 1e-9 for z in baseline)

    def order(item):
        (hold, ratio), zones = item
        return (sum(r["on_min"] for r in zones.values()),
                sum(r["switches"] for r in zones.values()), hold, -ratio)  # fmt: skip

    return min((item for item in combos.items() if qualifies(item[1])), key=order)[0]
