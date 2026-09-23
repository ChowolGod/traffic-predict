"""One-shot test evaluation with LOCK (M-14, F-09, plan 4.1 test 순서)."""

import json
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from tp import config
from tp.errors import TPError
from tp.eval import metrics
from tp.eval.bootstrap import block_bootstrap_diff_ci
from tp.exp import registry
from tp.models import arima, lstm, naive
from tp.prep import series as series_mod

FAMILIES = {  # final.yaml key -> (model.type, naive.lag)
    "naive_144": ("naive", 144),
    "naive_1": ("naive", 1),
    "arima": ("arima", None),
    "lstm": ("lstm", None),
}
COLUMNS = ["family", "zone_rank", "square_id", "mae", "mae_std", "rmse", "rmse_std", "rel_mae",
           "mae_diff_vs_lag144", "ci_low", "ci_high"]  # fmt: skip
NOTE = (
    "test는 한 번만 평가합니다(`LOCK`). 모든 모델은 train으로만 학습했고, 설정은 val MAE로 "
    "골랐습니다(val은 선택용이라 편향됨). `ci_low`~`ci_high`는 lag-144 대비 MAE 차이의 95% "
    "일별 블록 부트스트랩 구간이며, test가 7일이라 블록이 7개뿐이어서 거칩니다. LSTM의 "
    "mae·rmse는 시드별 지표의 평균±표준편차이고, 차이와 구간은 시드 평균 예측으로 계산합니다."
)


def _bad(message: str) -> TPError:
    return TPError("E-4006", f"최종 설정 오류: {message}")


def _family_of(cfg: dict) -> str | None:
    for family, (model_type, lag) in FAMILIES.items():
        if cfg["model.type"] == model_type and (lag is None or cfg.get("naive.lag") == lag):
            return family
    return None


def _val_mae(e: registry.Experiment) -> float:
    return json.loads((e.folder / "metrics.json").read_text(encoding="utf-8"))["val"]["mae"]


def _validate(final: dict, index: dict, k: int) -> dict[int, dict[str, registry.Experiment]]:
    zones = final.get("zones") if isinstance(final, dict) else None
    if not isinstance(zones, dict) or sorted(zones) != list(range(1, k + 1)):
        raise _bad(f"zones 키는 1~{k}이어야 함: {sorted(zones or {})}")
    chosen: dict[int, dict[str, registry.Experiment]] = {}
    for rank, mapping in zones.items():
        if not isinstance(mapping, dict) or set(mapping) != set(FAMILIES):
            raise _bad(f"구역 {rank}의 계열 키는 {list(FAMILIES)}이어야 함")
        chosen[rank] = {}
        for family in FAMILIES:  # fixed order: naive_144 first (bootstrap reference)
            exp_id = mapping[family]
            e = index.get(exp_id)
            if e is None or e.status != "completed":
                raise _bad(f"{exp_id}가 없거나 완료되지 않음")
            if e.config["phase"] != "full" or e.meta["post_test"]:
                raise _bad(f"{exp_id}는 full 단계의 post_test=false 실험이어야 함")
            if e.config["zone_rank"] != rank or _family_of(e.config) != family:
                raise _bad(f"{exp_id}는 구역 {rank}의 {family} 실험이 아님")
            candidates = [
                c for c in index.values()
                if c.status == "completed" and c.config["phase"] == "full"
                and not c.meta["post_test"] and c.config["zone_rank"] == rank
                and _family_of(c.config) == family
                and c.meta["compare_group"] == e.meta["compare_group"]
            ]  # fmt: skip
            best = min(candidates, key=lambda c: (_val_mae(c), c.id))
            if best.id != exp_id:
                raise _bad(f"{family} 선택 규칙과 다름: {exp_id} 대신 {best.id}(val MAE 최소)")
            chosen[rank][family] = e
        groups = {e.meta["compare_group"] for e in chosen[rank].values()}
        if len(groups) != 1:
            raise _bad(f"구역 {rank}의 계열들이 같은 compare_group이 아님: {sorted(groups)}")
    return chosen


def _load_models(chosen: dict) -> dict:
    """Load every D-11 artifact before any prediction (E-4006 on missing/broken files)."""
    models = {}
    for rank, families in chosen.items():
        for family, e in families.items():
            try:
                if family == "arima":
                    models[rank, family] = arima.load_params(
                        e.folder / "model" / "arima_params.json"
                    )
                elif family == "lstm":
                    models[rank, family] = [
                        lstm.load_model(e.folder / "model", seed, e.config)
                        for seed in config.LSTM_SEEDS
                    ]
            except (OSError, ValueError, RuntimeError, KeyError) as exc:
                raise _bad(f"{e.id} 모델 파일을 읽을 수 없음: {exc}") from exc
    return models


def _predict(family: str, e, model, series: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    frames = []
    if family.startswith("naive"):
        y = naive.predict_naive(series, e.config["naive.lag"], pd.DatetimeIndex(rows["time_utc"]))
        frames.append(registry._pred_frame(rows, y.to_numpy(), seed=-1))
    elif family == "arima":
        y = arima.predict_arima(model, series).loc[rows.index]
        frames.append(registry._pred_frame(rows, y.to_numpy(), seed=-1))
    else:
        for seed, m in zip(config.LSTM_SEEDS, model, strict=True):
            y = lstm.predict_lstm(m, series, rows.index)
            frames.append(registry._pred_frame(rows, y.to_numpy(), seed=seed))
    return pd.concat(frames, ignore_index=True)


def _evaluate_zone(rank: int, families: dict, models: dict, phase) -> tuple[list, list]:
    square_id = next(iter(families.values())).meta["square_id"]
    series = series_mod.load_series(phase, square_id)
    rows = series[series["segment"].astype(str) == "test"]
    reference, table, preds = None, [], []
    lag_err = None
    for family, e in families.items():
        pred = _predict(family, e, models.get((rank, family)), series, rows)
        result = metrics.evaluate(pred, series, "test", reference)
        scored = pred[metrics.eval_mask(pred, "test")]
        reference = metrics.eval_times(pred, "test")
        mean_pred = scored.groupby("time_utc")[["y_true", "y_pred"]].mean().loc[reference]
        err = np.abs(mean_pred["y_true"] - mean_pred["y_pred"]).to_numpy()
        if family == "naive_144":  # first by FAMILIES order
            lag_err = err
        days = reference.tz_convert(config.TZ).date
        diff, low, high = block_bootstrap_diff_ci(err, lag_err, np.array(days, dtype=object))
        std = result["test_std"] or {}
        table.append({
            "family": family, "zone_rank": rank, "square_id": square_id,
            "mae": result["test"]["mae"], "mae_std": std.get("mae"),
            "rmse": result["test"]["rmse"], "rmse_std": std.get("rmse"),
            "rel_mae": result["test"]["rel_mae"],
            "mae_diff_vs_lag144": diff, "ci_low": low, "ci_high": high,
        })  # fmt: skip
        preds.append(pred.assign(family=family, zone_rank=rank, square_id=square_id))
    return table, preds


def _write(out_tmp: Path, table: pd.DataFrame, preds: pd.DataFrame, lock: dict) -> None:
    out_tmp.mkdir(parents=True)
    table.to_csv(out_tmp / "metrics.csv", index=False)
    lines = ["# test 결과", "", NOTE, "", "| " + " | ".join(COLUMNS) + " |",
             "|" + "---|" * len(COLUMNS)]  # fmt: skip
    for row in table.to_dict("records"):
        cells = [
            "" if v is None or (isinstance(v, float) and np.isnan(v))
            else f"{v:.4f}" if isinstance(v, float) else str(v)
            for v in (row[c] for c in COLUMNS)
        ]  # fmt: skip
        lines.append("| " + " | ".join(cells) + " |")
    (out_tmp / "metrics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    cols = ["family", "zone_rank", "square_id", "time_utc", "y_true", "y_pred", "seed",
            "is_imputed"]  # fmt: skip
    preds[cols].to_parquet(out_tmp / "predictions.parquet", index=False)
    (out_tmp / "LOCK").write_text(json.dumps(lock, indent=2), encoding="utf-8")


def run_test(final_path: str | Path, confirm: bool) -> Path:
    if not confirm:
        raise _bad("--confirm 없음 (test는 한 번만 평가할 수 있음)")
    commit, dirty = registry.git_state()
    if dirty:
        raise _bad("커밋되지 않은 코드 변경이 있음")
    lock = registry.acquire_lock()
    try:
        out = config.RESULTS_DIR / "test"
        if (out / "LOCK").exists():
            evaluated = json.loads((out / "LOCK").read_text(encoding="utf-8"))["evaluated_at"]
            raise TPError("E-4005", f"test는 이미 평가됨: {evaluated}")
        phase = config.load_phase("full")
        if phase.test is None:
            raise _bad("full 단계에 test 구간이 없음")
        final = yaml.safe_load(Path(final_path).read_text(encoding="utf-8")) or {}
        chosen = _validate(final, registry.scan(), phase.k)
        models = _load_models(chosen)

        tables, preds = [], []
        for rank, families in chosen.items():
            table, pred = _evaluate_zone(rank, families, models, phase)
            tables += table
            preds += pred
        table = pd.DataFrame(tables, columns=COLUMNS)
        if phase.k > 1:
            means = table.groupby("family", sort=False)["rel_mae"].mean().reset_index()
            table = pd.concat([table, means.assign(zone_rank="mean")], ignore_index=True)

        record = {
            "evaluated_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "final": {str(r): {f: e.id for f, e in fams.items()} for r, fams in chosen.items()},
            "git_commit": commit,
        }
        tmp = config.RESULTS_DIR / ".test_tmp"
        if tmp.exists():
            shutil.rmtree(tmp)
        try:
            _write(tmp, table, pd.concat(preds, ignore_index=True), record)
            if out.exists():
                shutil.rmtree(out)  # no LOCK inside (checked above): stale partial output
            os.replace(tmp, out)
        finally:
            if tmp.exists():
                shutil.rmtree(tmp)
        return out
    finally:
        lock.release()
