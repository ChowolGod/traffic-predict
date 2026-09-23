"""Final-setting selection rule (plan 3.3), the one place shared by `results` and `test`."""

from collections.abc import Iterable

Key = tuple[int, int, str, str]  # (zone_rank, horizon, family, compare_group)


def family(model_type: str, lag: float | None) -> str:
    """lag may arrive as float from a DataFrame column mixed with NaN (non-naive rows)."""
    return f"lag{int(lag)}" if model_type == "naive" else model_type


def best_ids(records: Iterable[dict]) -> dict[Key, str]:
    """Among full, completed, post_test=false experiments, the smallest val MAE (LSTM: seed
    mean) per (zone_rank, horizon, family, compare_group); ties go to the smaller ID.

    Each record needs exp_id, phase, status, post_test, zone_rank, horizon, family,
    compare_group and val_mae.
    """
    best: dict[Key, tuple[float, str]] = {}
    for r in records:
        if r["phase"] != "full" or r["status"] != "completed" or r["post_test"]:
            continue
        key = (int(r["zone_rank"]), int(r["horizon"]), r["family"], r["compare_group"])
        candidate = (float(r["val_mae"]), r["exp_id"])
        if key not in best or candidate < best[key]:
            best[key] = candidate
    return {key: exp_id for key, (_, exp_id) in best.items()}
