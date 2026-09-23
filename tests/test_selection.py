"""Final-setting selection rule (plan 3.3), shared by `results` (D-15) and `test` (M-14)."""

import pandas as pd

from tp.exp import results, selection


def record(exp_id, val_mae, family="lstm", group="A", **kw):
    base = {"exp_id": exp_id, "phase": "full", "status": "completed", "post_test": False,
            "zone_rank": 1, "horizon": 1, "family": family, "compare_group": group,
            "val_mae": val_mae}  # fmt: skip
    return base | kw


def test_min_val_mae_ties_by_smaller_id():
    best = selection.best_ids([record("EXP-003", 90.0), record("EXP-002", 90.0),
                               record("EXP-004", 95.0)])  # fmt: skip
    assert best == {(1, 1, "lstm", "A"): "EXP-002"}


def test_only_full_completed_pre_test_experiments_are_candidates():
    best = selection.best_ids([
        record("EXP-002", 99.0),
        record("EXP-003", 10.0, phase="dev"),
        record("EXP-004", None, status="failed"),
        record("EXP-005", 10.0, post_test=True),
    ])  # fmt: skip
    assert best == {(1, 1, "lstm", "A"): "EXP-002"}


def test_candidates_are_split_by_zone_horizon_family_and_compare_group():
    best = selection.best_ids([
        record("EXP-002", 50.0), record("EXP-003", 40.0, group="B"),
        record("EXP-004", 30.0, horizon=6), record("EXP-005", 20.0, zone_rank=2),
        record("EXP-006", 10.0, family="arima"),
    ])  # fmt: skip
    assert best == {
        (1, 1, "lstm", "A"): "EXP-002", (1, 1, "lstm", "B"): "EXP-003",
        (1, 6, "lstm", "A"): "EXP-004", (2, 1, "lstm", "A"): "EXP-005",
        (1, 1, "arima", "A"): "EXP-006",
    }  # fmt: skip


def test_family_names():
    assert selection.family("naive", 1.0) == "lag1"  # lag may arrive as float from a frame
    assert selection.family("naive", 1008) == "lag1008"
    assert selection.family("arima", None) == "arima"


def row(exp_id, model_type, val_mae, group, lag=None):
    return {"exp_id": exp_id, "phase": "full", "status": "completed", "post_test": False,
            "zone_rank": 1, "horizon": 1, "model_type": model_type, "_lag": lag,
            "compare_group": group, "val_mae": val_mae}  # fmt: skip


def test_horizon_table_keeps_the_reference_compare_group():
    # The zone's reference is its first completed full experiment (E-4004), group A.
    # A better LSTM in another group must not be reported as "best" (plan 3.3 모델 비교 조건).
    frame = pd.DataFrame([
        row("EXP-001", "naive", 500.0, "A", lag=144),
        row("EXP-002", "lstm", 100.0, "A"),
        row("EXP-003", "lstm", 50.0, "B"),
    ])  # fmt: skip
    best = results._best(frame)
    assert best.set_index("family")["exp_id"].to_dict() == {"lag144": "EXP-001", "lstm": "EXP-002"}
