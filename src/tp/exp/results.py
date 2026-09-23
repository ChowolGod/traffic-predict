"""Rebuild the results table from experiment folders (M-12, D-09)."""

import json

import pandas as pd
import yaml

from tp import config

COLUMNS = [
    "exp_id", "name", "phase", "parent", "changed", "model_type", "zone_rank", "square_id",
    "val_mae", "val_mae_std", "val_rmse", "val_rmse_std", "val_rel_mae", "n",
    "compare_group", "status", "post_test", "duration_s",
]  # fmt: skip

NOTE = (
    "val 지표는 설정 **선택용**이라 낙관적으로 편향되어 있습니다(조기 종료까지 val로 하는 "
    "LSTM은 더 편향됨). 계열 간 공정한 비교는 test 결과(`results/test/`)만 해당합니다. "
    "`compare_group`이 같은 행끼리만 비교할 수 있습니다."
)


def load_rows() -> list[dict]:
    rows = []
    if not config.EXPERIMENTS_DIR.is_dir():
        return rows
    for folder in sorted(config.EXPERIMENTS_DIR.glob("EXP-*_*")):
        if not folder.is_dir():
            continue
        row = dict.fromkeys(COLUMNS)
        row["exp_id"] = folder.name[:7]
        try:
            meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
            cfg = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
        except (OSError, ValueError, yaml.YAMLError):
            row["status"] = "corrupt"
            rows.append(row)
            continue
        row.update(
            name=cfg["name"], phase=cfg["phase"], parent=cfg["parent"], changed=cfg["changed"],
            model_type=cfg["model"]["type"], zone_rank=cfg["zone_rank"],
            square_id=meta.get("square_id"), compare_group=meta.get("compare_group"),
            status=meta["status"], post_test=meta["post_test"], duration_s=meta["duration_s"],
        )  # fmt: skip
        metrics_path = folder / "metrics.json"
        if meta["status"] == "completed" and metrics_path.is_file():
            m = json.loads(metrics_path.read_text(encoding="utf-8"))
            std = m.get("val_std") or {}
            row.update(
                val_mae=m["val"]["mae"], val_rmse=m["val"]["rmse"],
                val_rel_mae=m["val"]["rel_mae"], n=m["val"]["n"],
                val_mae_std=std.get("mae"), val_rmse_std=std.get("rmse"),
            )  # fmt: skip
        rows.append(row)
    return rows


def _cell(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def rebuild_results() -> None:
    rows = load_rows()
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=COLUMNS).to_csv(config.RESULTS_DIR / "results.csv", index=False)
    lines = ["# 실험 결과 (val)", "", NOTE, "", "| " + " | ".join(COLUMNS) + " |",
             "|" + "---|" * len(COLUMNS)]  # fmt: skip
    lines += ["| " + " | ".join(_cell(r[c]) for c in COLUMNS) + " |" for r in rows]
    (config.RESULTS_DIR / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
