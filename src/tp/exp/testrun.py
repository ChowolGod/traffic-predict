"""One-shot test evaluation with LOCK (M-14, F-09 v2.12, plan 4.1 test 순서)."""

import json
import os
import shutil
from datetime import UTC, datetime
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from tp import config
from tp.errors import TPError
from tp.eval import decision, metrics, switching
from tp.eval.bootstrap import block_bootstrap_diff_ci
from tp.exp import registry, results, selection
from tp.models import arima, lstm, naive
from tp.prep import series as series_mod

HORIZONS = (1, 3, 6)  # final.yaml horizon keys (steps of 10 minutes)
FAMILIES = {  # final.yaml key -> selection family; the two CI references come first
    "naive_144": "lag144",
    "naive_1": "lag1",
    "naive_1008": "lag1008",
    "arima": "arima",
    "lstm": "lstm",
}
REFERENCES = {"lag144": "naive_144", "lag1": "naive_1"}  # plan 3.3 불확실성(test)
COLUMNS = [
    "family", "zone_rank", "square_id", "horizon", "mae", "mae_std", "rmse", "rmse_std",
    "rel_mae", "diff_vs_lag144", "ci_lag144_low", "ci_lag144_high", "diff_vs_lag1",
    "ci_lag1_low", "ci_lag1_high", "dec_threshold", "dec_episodes", "dec_missed",
    "dec_missed_std", "dec_false_alarm_min", "dec_false_alarm_min_std", "dec_lead_min",
    "dec_lead_min_std",
]  # fmt: skip
SWITCH_COLUMNS = [
    "family", "zone_rank", "square_id", "horizon", "delay_min", "hold_min", "on_ratio",
    "episodes", "missed_min", "missed_min_std", "missed_episodes", "missed_episodes_std",
    "on_min", "on_min_std", "switches", "switches_std",
]  # fmt: skip
PRED_COLUMNS = ["family", "zone_rank", "square_id", "horizon", "time_utc", "y_true", "y_pred",
                "seed", "is_imputed"]  # fmt: skip
NOTE = (
    "test는 한 번만 평가합니다(`LOCK`). 모든 모델은 train으로만 학습했고, 실험은 (구역, 거리, "
    "계열)마다 val MAE로 골랐습니다(val은 선택용이라 편향됨). 구간은 어제 같은 시각과 마지막 "
    "관측값 각각 대비 MAE 차이의 95% 일별 블록 부트스트랩 구간이며, test가 7일이라 블록이 "
    "7개뿐이어서 거칩니다. LSTM은 시드별 지표의 평균±표준편차이고, 구간은 칸마다 시드별 "
    "절대오차를 평균해 계산합니다(보고 MAE와 같은 양). 결정 지표 임계치는 train에서 정했고, "
    "켜기/끄기 조합은 val에서 s1로 고른 것을 그대로 씁니다. 켜기/끄기는 예측이 t − h + 1에 "
    "나온다고 봅니다(3단계 리드타임은 t − h 기준)."
)


def _bad(message: str) -> TPError:
    return TPError("E-4006", f"최종 설정 오류: {message}")


def _family_of(cfg: dict) -> str | None:
    """final.yaml key of an experiment, or None for families test does not evaluate."""
    name = selection.family(cfg["model.type"], cfg.get("naive.lag"))
    return next((key for key, fam in FAMILIES.items() if fam == name), None)


def _val_mae(e: registry.Experiment) -> float:
    return json.loads((e.folder / "metrics.json").read_text(encoding="utf-8"))["val"]["mae"]


def _best_ids(index: dict) -> dict[selection.Key, str]:
    records = [
        {"exp_id": e.id, "phase": e.config["phase"], "status": e.status,
         "post_test": e.meta["post_test"], "zone_rank": e.config["zone_rank"],
         "horizon": e.config["horizon"], "compare_group": e.meta["compare_group"],
         "family": selection.family(e.config["model.type"], e.config.get("naive.lag")),
         "val_mae": _val_mae(e) if e.status == "completed" else None}
        for e in index.values() if e.status != "corrupt"
    ]  # fmt: skip
    return selection.best_ids(records)


Chosen = dict[int, dict[int, dict[str, registry.Experiment]]]  # zone -> horizon -> family


def _validate(final: dict, index: dict, k: int) -> Chosen:
    zones = final.get("zones") if isinstance(final, dict) else None
    if not isinstance(zones, dict) or sorted(zones) != list(range(1, k + 1)):
        raise _bad(f"zones 키는 1~{k}이어야 함: {sorted(zones or {})}")
    chosen: Chosen = {}
    best_ids = None  # read metrics only after the cheap checks pass
    for rank, by_horizon in zones.items():
        if not isinstance(by_horizon, dict) or sorted(by_horizon) != sorted(HORIZONS):
            raise _bad(f"구역 {rank}의 거리 키는 {list(HORIZONS)}이어야 함")
        chosen[rank] = {}
        for h in HORIZONS:
            mapping = by_horizon[h]
            if not isinstance(mapping, dict) or set(mapping) != set(FAMILIES):
                raise _bad(f"구역 {rank} 거리 {h}의 계열 키는 {list(FAMILIES)}이어야 함")
            chosen[rank][h] = {}
            for family in FAMILIES:  # fixed order: CI references first
                exp_id = mapping[family]
                e = index.get(exp_id)
                if e is None or e.status != "completed":
                    raise _bad(f"{exp_id}가 없거나 완료되지 않음")
                if e.config["phase"] != "full" or e.meta["post_test"]:
                    raise _bad(f"{exp_id}는 full 단계의 post_test=false 실험이어야 함")
                if e.config["zone_rank"] != rank or _family_of(e.config) != family:
                    raise _bad(f"{exp_id}는 구역 {rank}의 {family} 실험이 아님")
                if e.config["horizon"] != h:
                    raise _bad(f"{exp_id}는 horizon {e.config['horizon']}인데 거리 {h}에 있음")
                best_ids = best_ids if best_ids is not None else _best_ids(index)
                key = (rank, h, FAMILIES[family], e.meta["compare_group"])
                if best_ids[key] != exp_id:
                    raise _bad(
                        f"{family} 거리 {h} 선택 규칙과 다름: {exp_id} 대신 {best_ids[key]}"
                        "(val MAE 최소)"
                    )
                chosen[rank][h][family] = e
        groups = {e.meta["compare_group"] for fams in chosen[rank].values() for e in fams.values()}
        if len(groups) != 1:
            raise _bad(f"구역 {rank}의 실험들이 같은 compare_group이 아님: {sorted(groups)}")
    return chosen


def _load_models(chosen: Chosen) -> dict:
    """Load every D-11 artifact before any prediction (E-4006 on missing/broken files)."""
    models = {}
    for rank, by_horizon in chosen.items():
        for h, families in by_horizon.items():
            for family, e in families.items():
                try:
                    if family == "arima":
                        path = e.folder / "model" / "arima_params.json"
                        models[rank, h, family] = arima.load_params(path)
                    elif family == "lstm":
                        models[rank, h, family] = [
                            lstm.load_model(e.folder / "model", seed, e.config)
                            for seed in config.LSTM_SEEDS
                        ]
                except (OSError, ValueError, RuntimeError, KeyError) as exc:
                    raise _bad(f"{e.id} 모델 파일을 읽을 수 없음: {exc}") from exc
    return models


def _predict(family: str, e, model, series: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    h = e.config["horizon"]
    frames = []
    if family.startswith("naive"):
        targets = pd.DatetimeIndex(rows["time_utc"])
        y = naive.predict_naive(series, e.config["naive.lag"], targets, h)
        frames.append(registry._pred_frame(rows, y.to_numpy(), seed=-1))
    elif family == "arima":
        y = arima.predict_arima(model, series, h).loc[rows.index]
        frames.append(registry._pred_frame(rows, y.to_numpy(), seed=-1))
    else:
        for seed, m in zip(config.LSTM_SEEDS, model, strict=True):
            y = lstm.predict_lstm(m, series, rows.index)
            frames.append(registry._pred_frame(rows, y.to_numpy(), seed=seed))
    return pd.concat(frames, ignore_index=True)


def _abs_errors(pred: pd.DataFrame, times: pd.DatetimeIndex) -> np.ndarray:
    """Per scored slot, the mean over seeds of |y - ŷ| (plan 3.3: same quantity as the MAE)."""
    scored = pred[metrics.eval_mask(pred, "test")]
    err = (scored["y_true"] - scored["y_pred"]).abs().groupby(scored["time_utc"]).mean()
    return err.loc[times].to_numpy()


def _evaluate_zone(rank: int, by_horizon: dict, models: dict, phase, dcfg) -> dict:
    first = next(iter(next(iter(by_horizon.values())).values()))
    square_id = first.meta["square_id"]
    series = series_mod.load_series(phase, square_id)
    rows = series[series["segment"].astype(str) == "test"]
    train_y = series.loc[series["segment"].astype(str) == "train", "y"].to_numpy()
    thr = decision.threshold(train_y, dcfg)
    reference, table, preds = None, [], {}
    for h, families in by_horizon.items():
        results_, errors = {}, {}
        for family, e in families.items():
            pred = _predict(family, e, models.get((rank, h, family)), series, rows)
            results_[family] = metrics.evaluate(pred, series, "test", reference)
            reference = metrics.eval_times(pred, "test")
            errors[family] = _abs_errors(pred, reference)
            preds[h, family] = pred
        days = np.array(reference.tz_convert(config.TZ).date, dtype=object)
        for family, result in results_.items():
            std = result["test_std"] or {}
            row = {"family": family, "zone_rank": rank, "square_id": square_id, "horizon": h,
                   "mae": result["test"]["mae"], "mae_std": std.get("mae"),
                   "rmse": result["test"]["rmse"], "rmse_std": std.get("rmse"),
                   "rel_mae": result["test"]["rel_mae"]}  # fmt: skip
            for ref, ref_family in REFERENCES.items():
                diff, low, high = block_bootstrap_diff_ci(errors[family], errors[ref_family], days)
                row |= {f"diff_vs_{ref}": diff, f"ci_{ref}_low": low, f"ci_{ref}_high": high}
            row |= results.decision_by_seed(preds[h, family], thr, h, dcfg.merge_gap)
            table.append(row)
    return {"square_id": square_id, "thr": thr, "table": table, "preds": preds}


def _switching(chosen: Chosen, zones: dict, dcfg) -> list[dict]:
    """Per (family, horizon, delay) pick the combination on val with s1 (plan 3.3), then apply
    it unchanged to the test predictions of every zone; add the two baselines per zone."""
    rows = []
    for h in HORIZONS:
        for family in FAMILIES:
            folders = {rank: chosen[rank][h][family].folder for rank in chosen}
            val = {r: results.seed_arrays(results.val_predictions(f)) for r, f in folders.items()}
            test = {r: results.seed_arrays(zones[r]["preds"][h, family]) for r in chosen}
            for delay in dcfg.delay_min:
                candidates = [
                    {"zone_rank": rank, "hold_min": hold, "on_ratio": ratio,
                     **results.switch_metrics(val[rank], zones[rank]["thr"], h, hold, ratio,
                                              delay, dcfg.merge_gap)}
                    for rank in chosen for hold, ratio in product(dcfg.hold_min, dcfg.on_ratio)
                ]  # fmt: skip
                hold, ratio = switching.choose_s1(candidates)
                for rank in chosen:
                    m = results.switch_metrics(test[rank], zones[rank]["thr"], h, hold, ratio,
                                               delay, dcfg.merge_gap)  # fmt: skip
                    rows.append({"family": family, "zone_rank": rank,
                                 "square_id": zones[rank]["square_id"], "horizon": h,
                                 "delay_min": delay, "hold_min": hold, "on_ratio": ratio,
                                 **m})  # fmt: skip
    for rank in chosen:
        seeds = results.seed_arrays(next(iter(zones[rank]["preds"].values())))
        for family, m in results.baseline_metrics(
            seeds, zones[rank]["thr"], dcfg.merge_gap
        ).items():
            rows.append({"family": family, "zone_rank": rank,
                         "square_id": zones[rank]["square_id"], **m})  # fmt: skip
    return rows


def _fmt(value, std=None, digits=1) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = f"{value:.{digits}f}"
    if std is not None and not (isinstance(std, float) and np.isnan(std)):
        text += f"±{std:.{digits}f}"
    return text


def _markdown(table: pd.DataFrame, switch: pd.DataFrame, delays: tuple[int, ...]) -> str:
    labels = {key: label for key, _, label, _, _ in results.FAMILIES}
    name = {key: labels[fam] for key, fam in FAMILIES.items()} | {
        "always_on": "항상 켬", "oracle": "미래를 아는 경우"}  # fmt: skip
    lines = ["# test 결과", "", NOTE, "", "## 예측 오차와 결정 지표", "",
             "| 구역 | 거리 | 계열 | MAE | 상대 MAE | 어제 같은 시각 대비 [95% 구간] | "
             "마지막 관측값 대비 [95% 구간] | 놓친 혼잡 | 불필요 경보(분) | 리드타임(분) |",
             "|" + "---|" * 10]  # fmt: skip
    for r in table.to_dict("records"):
        zone = "평균" if r["zone_rank"] == "mean" else r["square_id"]
        if r["zone_rank"] == "mean":
            lines.append(f"| {zone} | {r['horizon'] * 10}분 | {name[r['family']]} | | "
                         f"{_fmt(r['rel_mae'], digits=3)} | | | | | |")  # fmt: skip
            continue
        refs = [f"{_fmt(r[f'diff_vs_{ref}'])} [{_fmt(r[f'ci_{ref}_low'])}, "
                f"{_fmt(r[f'ci_{ref}_high'])}]" for ref in REFERENCES]  # fmt: skip
        missed = _fmt(r["dec_missed"], r["dec_missed_std"])
        lines.append(
            f"| {zone} | {r['horizon'] * 10}분 | {name[r['family']]} | "
            f"{_fmt(r['mae'], r['mae_std'])} | {_fmt(r['rel_mae'], digits=3)} | {refs[0]} | "
            f"{refs[1]} | {missed}/{r['dec_episodes']} | "
            f"{_fmt(r['dec_false_alarm_min'], r['dec_false_alarm_min_std'])} | "
            f"{_fmt(r['dec_lead_min'], r['dec_lead_min_std'])} |"
        )
    for delay in delays:
        lines += ["", f"## 켜기/끄기 (지연 {delay}분)", "",
                  "| 구역 | 거리 | 계열 | 조합(val에서 선택) | 놓친 혼잡 시간 | 놓친 혼잡 | "
                  "켜 둔 시간 | 켜고 끈 횟수 |", "|" + "---|" * 8]  # fmt: skip
        sub = switch[(switch["delay_min"] == delay) | switch["delay_min"].isna()]
        for r in sub.to_dict("records"):
            base = r["family"] in ("always_on", "oracle")
            combo = "기준선" if base else f"H {int(r['hold_min'])}분·{r['on_ratio']:.1f}θ"
            horizon = "" if base else f"{int(r['horizon']) * 10}분"
            lines.append(
                f"| {r['square_id']} | {horizon} | {name[r['family']]} | {combo} | "
                f"{_fmt(r['missed_min'], r.get('missed_min_std'), 0)} | "
                f"{_fmt(r['missed_episodes'], r.get('missed_episodes_std'), 0)}/{r['episodes']} | "
                f"{_fmt(r['on_min'], r.get('on_min_std'), 0)} | "
                f"{_fmt(r['switches'], r.get('switches_std'), 0)} |"
            )
    return "\n".join(lines) + "\n"


def _write(out_tmp: Path, table: pd.DataFrame, switch: pd.DataFrame, preds: pd.DataFrame,
           lock: dict, delays: tuple[int, ...]) -> None:  # fmt: skip
    out_tmp.mkdir(parents=True)
    table.to_csv(out_tmp / "metrics.csv", index=False)
    switch.to_csv(out_tmp / "switching.csv", index=False)
    (out_tmp / "metrics.md").write_text(_markdown(table, switch, delays), encoding="utf-8")
    preds[PRED_COLUMNS].to_parquet(out_tmp / "predictions.parquet", index=False)
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
        dcfg = config.load_decision()
        models = _load_models(chosen)

        zones = {rank: _evaluate_zone(rank, by_h, models, phase, dcfg)
                 for rank, by_h in chosen.items()}  # fmt: skip
        table = pd.DataFrame([r for z in zones.values() for r in z["table"]], columns=COLUMNS)
        if phase.k > 1:
            means = table.groupby(["family", "horizon"], sort=False)["rel_mae"].mean()
            table = pd.concat([table, means.reset_index().assign(zone_rank="mean")],
                              ignore_index=True)  # fmt: skip
        switch = pd.DataFrame(_switching(chosen, zones, dcfg), columns=SWITCH_COLUMNS)
        preds = pd.concat(
            [p.assign(family=f, zone_rank=rank, square_id=z["square_id"], horizon=h)
             for rank, z in zones.items() for (h, f), p in z["preds"].items()],
            ignore_index=True,
        )  # fmt: skip

        record = {
            "evaluated_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "final": {str(r): {str(h): {f: e.id for f, e in fams.items()}
                               for h, fams in by_h.items()} for r, by_h in chosen.items()},
            "git_commit": commit,
        }  # fmt: skip
        tmp = config.RESULTS_DIR / ".test_tmp"
        if tmp.exists():
            shutil.rmtree(tmp)
        try:
            _write(tmp, table, switch, preds, record, dcfg.delay_min)
            if out.exists():
                shutil.rmtree(out)  # no LOCK inside (checked above): stale partial output
            os.replace(tmp, out)
        finally:
            if tmp.exists():
                shutil.rmtree(tmp)
        return out
    finally:
        lock.release()
