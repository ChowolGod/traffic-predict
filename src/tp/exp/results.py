"""Rebuild the results table, decision metrics and the horizon comparison (M-12, D-09, D-15)."""

import json
import logging
from itertools import product
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from tp import config  # noqa: E402
from tp.errors import TPError  # noqa: E402
from tp.eval import decision, switching  # noqa: E402
from tp.exp import selection  # noqa: E402
from tp.prep import series as series_mod  # noqa: E402

log = logging.getLogger(__name__)

COLUMNS = [
    "exp_id", "name", "phase", "parent", "changed", "model_type", "horizon", "zone_rank",
    "square_id", "val_mae", "val_mae_std", "val_rmse", "val_rmse_std", "val_rel_mae", "n",
    "compare_group", "dec_threshold", "dec_episodes", "dec_missed", "dec_false_alarm_min",
    "dec_lead_min", "dec_missed_std", "dec_false_alarm_min_std", "dec_lead_min_std",
    "status", "post_test", "duration_s",
]  # fmt: skip

NOTE = (
    "val 지표는 설정 **선택용**이라 낙관적으로 편향되어 있습니다(조기 종료까지 val로 하는 "
    "LSTM은 더 편향됨). 계열 간 공정한 비교는 test 결과(`results/test/`)만 해당합니다. "
    "`compare_group`과 `horizon`이 같은 행끼리만 비교할 수 있습니다. `dec_*`는 각 실험의 "
    "거리 기준 결정 지표입니다(`configs/decision.yaml`). LSTM의 `dec_*`는 시드별로 계산한 "
    "평균이고 `dec_*_std`는 시드 간 표준편차입니다."
)

# Family order is fixed so colour follows the entity (dataviz: palette slots 1-5, never cycled).
# (key, chart label, table label, colour, marker)
FAMILIES = [
    ("lag1", "last observed value (= 10 min ago at 10-min horizon)",
     "마지막 관측값(10분 뒤 예측에서는 10분 전 값)", "#2a78d6", "o"),
    ("lag144", "same time yesterday (lag 144)", "어제 같은 시각(lag-144)", "#eb6834", "s"),
    ("lag1008", "same time last week (lag 1008)", "지난주 같은 시각(lag-1008)", "#1baf7a", "^"),
    ("arima", "ARIMA", "ARIMA", "#eda100", "D"),
    ("lstm", "LSTM", "LSTM", "#e87ba4", "v"),
]  # fmt: skip
HORIZONS = (1, 3, 6)
SURFACE, INK, INK_2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e8e7e3"


def _mean_std(values: list) -> tuple[float | None, float | None]:
    present = [v for v in values if v is not None]
    if not present:
        return None, None
    std = float(np.std(present, ddof=1)) if len(values) > 1 and len(present) > 1 else None
    return float(np.mean(present)), std


def decision_by_seed(pred: pd.DataFrame, thr: float, horizon: int, merge_gap: int) -> dict:
    """Decision metrics per seed, then mean (and ddof=1 std across LSTM seeds; plan v2.3)."""
    per_seed = []
    for _, group in pred.sort_values("time_utc").groupby("seed", sort=True):
        per_seed.append(decision.decision_metrics(
            group["y_true"].to_numpy(), group["y_pred"].to_numpy(),
            group["is_imputed"].to_numpy(bool), thr, horizon, merge_gap,
        ))  # fmt: skip
    out = {"dec_threshold": thr, "dec_episodes": per_seed[0]["episodes"]}
    for key in ("missed", "false_alarm_min", "lead_min"):
        mean, std = _mean_std([m[key] for m in per_seed])
        out[f"dec_{key}"] = mean if len(per_seed) > 1 else per_seed[0][key]
        out[f"dec_{key}_std"] = std if len(per_seed) > 1 else None
    return out


class _Series:
    """Cache of zone series and thresholds for decision metrics (None when not prepared)."""

    def __init__(self, dcfg: decision.DecisionConfig | None):
        self.dcfg = dcfg
        self.cache: dict[tuple[str, int], float | None] = {}

    def threshold(self, phase: str, square_id: int) -> float | None:
        key = (phase, square_id)
        if key not in self.cache:
            try:
                s = series_mod.load_series(config.load_phase(phase), square_id)
                train = s.loc[s["segment"].astype(str) == "train", "y"].to_numpy()
                self.cache[key] = decision.threshold(train, self.dcfg)
            except TPError as err:
                log.warning("결정 지표 생략 (%s, %s): %s", phase, square_id, err)
                self.cache[key] = None
        return self.cache[key]


def _decision(folder: Path, cfg: dict, meta: dict, series: _Series) -> dict:
    if series.dcfg is None or meta.get("square_id") is None:
        return {}
    thr = series.threshold(cfg["phase"], meta["square_id"])
    if thr is None:
        return {}
    pred = pd.read_parquet(folder / "predictions.parquet")
    val = pred[pred["segment"] == "val"]
    return decision_by_seed(val, thr, cfg.get("horizon", 1), series.dcfg.merge_gap)


# A folder whose files are missing, malformed or lack required fields (I-04: status=corrupt).
UNREADABLE = (OSError, ValueError, KeyError, TypeError, AttributeError, yaml.YAMLError)


def _row(folder: Path, series: "_Series") -> dict:
    row = dict.fromkeys(COLUMNS)
    row["exp_id"] = folder.name[:7]
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    cfg = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
    cfg.setdefault("horizon", 1)  # v1 experiments were all 1-step
    row.update(
        name=cfg["name"], phase=cfg["phase"], parent=cfg["parent"], changed=cfg["changed"],
        model_type=cfg["model"]["type"], horizon=cfg["horizon"], zone_rank=cfg["zone_rank"],
        square_id=meta.get("square_id"), compare_group=meta.get("compare_group"),
        status=meta["status"], post_test=meta["post_test"], duration_s=meta["duration_s"],
    )  # fmt: skip
    row["_lag"] = (cfg.get("naive") or {}).get("lag")
    metrics_path = folder / "metrics.json"
    if meta["status"] == "completed" and metrics_path.is_file():
        m = json.loads(metrics_path.read_text(encoding="utf-8"))
        std = m.get("val_std") or {}
        row.update(
            val_mae=m["val"]["mae"], val_rmse=m["val"]["rmse"],
            val_rel_mae=m["val"]["rel_mae"], n=m["val"]["n"],
            val_mae_std=std.get("mae"), val_rmse_std=std.get("rmse"),
        )  # fmt: skip
        row.update(_decision(folder, cfg, meta, series))
    return row


def load_rows(dcfg: decision.DecisionConfig | None = None) -> list[dict]:
    rows = []
    if not config.EXPERIMENTS_DIR.is_dir():
        return rows
    series = _Series(dcfg)
    for folder in sorted(config.EXPERIMENTS_DIR.glob("EXP-*_*")):
        if not folder.is_dir():
            continue
        try:
            rows.append(_row(folder, series))
        except UNREADABLE as exc:
            log.warning("읽을 수 없는 실험 폴더(corrupt): %s (%s: %s)",
                        folder.name, type(exc).__name__, exc)  # fmt: skip
            rows.append(dict.fromkeys(COLUMNS) | {"exp_id": folder.name[:7], "status": "corrupt"})
    return rows


def _cell(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _best(rows: pd.DataFrame) -> pd.DataFrame:
    """Best experiment per (zone, horizon, family) by the shared selection rule (3.3), within
    each zone's reference compare_group: that of its first completed full experiment (E-4004).
    """
    full = rows[(rows["phase"] == "full") & (rows["status"] == "completed")]
    pairs = zip(full["model_type"], full["_lag"], strict=True)
    full = full.assign(family=[selection.family(t, lag) for t, lag in pairs])
    reference = full.sort_values("exp_id").groupby("zone_rank")["compare_group"].first()
    chosen = set(selection.best_ids(full.to_dict("records")).values())
    best = full[full["exp_id"].isin(chosen)]
    others = best["compare_group"] != best["zone_rank"].map(reference)
    if others.any():
        log.warning("거리별 표에서 기준 compare_group이 아닌 실험 제외: %s",
                    list(best.loc[others, "exp_id"]))  # fmt: skip
    return best[~others].sort_values(["zone_rank", "horizon", "family"]).reset_index(drop=True)


def _num(value, std) -> str:
    """Integer-like values as ints, LSTM seed means with one decimal and ±std."""
    if pd.isna(value):
        return ""
    if pd.notna(std):
        return f"{value:.1f}±{std:.1f}"
    return f"{value:.0f}"


def _horizon_table(best: pd.DataFrame, report_horizon: int) -> str:
    minutes = {h: f"{h * 10}분" for h in HORIZONS}
    lines = [
        "# 예측 거리별 비교 (val, full 단계)", "",
        "(구역, 거리, 계열)마다 선택 규칙(val MAE 최소)상 최선 실험입니다. 괄호는 실험 ID, "
        "LSTM은 시드 3개 평균±표준편차입니다. 30·60분 실험은 10분 뒤에서 고른 하이퍼파라미터를 "
        "그대로 씁니다. val은 선택용이라 낙관적입니다.", "",
        f"결정 지표는 {report_horizon * 10}분 뒤 예측 기준입니다(`configs/decision.yaml`). "
        "놓친 혼잡은 '놓침/전체 구간', 불필요 경보와 리드타임은 분 단위입니다. 혼잡 시작 직전 "
        "거리만큼(60분 뒤 예측이면 60분)의 경보는 미리 켠 것으로 보아 불필요 경보에서 뺍니다. "
        "LSTM은 시드별 결과의 평균±표준편차입니다.", "",
        "| 구역 | 계열 | " + " | ".join(f"MAE {minutes[h]}" for h in HORIZONS)
        + " | 놓친 혼잡 | 불필요 경보 | 리드타임 |",
        "|---|---|" + "---|" * (len(HORIZONS) + 3),
    ]  # fmt: skip
    for zone in sorted(best["zone_rank"].unique()):
        zb = best[best["zone_rank"] == zone]
        square = int(zb["square_id"].iloc[0])
        for family, _, label, _, _ in FAMILIES:
            fb = zb[zb["family"] == family].set_index("horizon")
            if fb.empty:
                continue
            cells = []
            for h in HORIZONS:
                if h not in fb.index:
                    cells.append("")
                    continue
                r = fb.loc[h]
                std = f"±{r['val_mae_std']:.1f}" if pd.notna(r["val_mae_std"]) else ""
                cells.append(f"{r['val_mae']:.1f}{std} ({r['exp_id']})")
            if report_horizon in fb.index and pd.notna(fb.loc[report_horizon, "dec_episodes"]):
                r = fb.loc[report_horizon]
                dec = [
                    f"{_num(r['dec_missed'], r['dec_missed_std'])}/{int(r['dec_episodes'])}",
                    _num(r["dec_false_alarm_min"], r["dec_false_alarm_min_std"]),
                    _num(r["dec_lead_min"], r["dec_lead_min_std"]),
                ]
            else:
                dec = ["", "", ""]
            lines.append(f"| {square} | {label} | " + " | ".join(cells + dec) + " |")
    return "\n".join(lines) + "\n"


def _horizon_figure(best: pd.DataFrame, path: Path) -> None:
    zones = sorted(best["zone_rank"].unique())
    fig, axes = plt.subplots(1, len(zones), figsize=(4.2 * len(zones), 3.8), dpi=150,
                             squeeze=False)  # fmt: skip
    fig.patch.set_facecolor(SURFACE)
    for ax, zone in zip(axes[0], zones, strict=True):
        zb = best[best["zone_rank"] == zone]
        ax.set_facecolor(SURFACE)
        for family, label, _, color, marker in FAMILIES:
            fb = zb[zb["family"] == family].sort_values("horizon")
            if fb.empty:
                continue
            ax.plot(
                fb["horizon"] * 10, fb["val_mae"], color=color, linewidth=2, marker=marker,
                markersize=8, markeredgecolor=SURFACE, markeredgewidth=1.5, label=label,
            )  # fmt: skip
        ax.set_title(f"Zone {int(zb['square_id'].iloc[0])} (rank {zone})", color=INK, fontsize=10,
                     loc="left")  # fmt: skip
        ax.set_xticks([h * 10 for h in HORIZONS])
        ax.set_xlabel("forecast horizon (minutes)", color=MUTED, fontsize=9)
        ax.tick_params(which="both", colors=MUTED, labelsize=8)
        ax.grid(True, color=GRID, linewidth=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
    axes[0][0].set_ylabel("val MAE (activity units)", color=MUTED, fontsize=9)
    handles, labels = axes[0][0].get_legend_handles_labels()
    for ax in axes[0][1:]:
        for h, lab in zip(*ax.get_legend_handles_labels(), strict=True):
            if lab not in labels:
                handles.append(h)
                labels.append(lab)
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False, fontsize=8,
               labelcolor=INK_2)  # fmt: skip
    fig.suptitle("Validation MAE by forecast horizon (lower is better)", color=INK, fontsize=11,
                 x=0.01, ha="left")  # fmt: skip
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


SWITCH_COLUMNS = [
    "family", "zone_rank", "square_id", "exp_id", "horizon", "delay_min", "hold_min", "on_ratio",
    "episodes", "missed_min", "missed_min_std", "missed_episodes", "missed_episodes_std",
    "on_min", "on_min_std", "switches", "switches_std", "chosen",
]  # fmt: skip
SWITCH_METRICS = ("missed_min", "missed_episodes", "on_min", "switches")
BASELINES = [("always_on", "항상 켬"), ("oracle", "미래를 아는 경우")]


def val_predictions(folder: Path) -> pd.DataFrame:
    pred = pd.read_parquet(folder / "predictions.parquet")
    return pred[pred["segment"] == "val"]


def seed_arrays(pred: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """(y_true, y_pred, is_imputed) per seed, each in time order."""
    return [
        (g["y_true"].to_numpy(), g["y_pred"].to_numpy(), g["is_imputed"].to_numpy(bool))
        for _, g in pred.sort_values(["seed", "time_utc"]).groupby("seed", sort=True)
    ]


def switch_metrics(seeds: list, thr: float, horizon: int, hold: int, ratio: float, delay: int,
                   merge_gap: int) -> dict:  # fmt: skip
    """D-16 metrics of one on/off candidate; LSTM seeds -> mean and ddof=1 std (M-19)."""
    per_seed = [
        switching.switching_metrics(
            switching.simulate(p, imp, thr, horizon, hold, ratio, delay), y, imp, thr, merge_gap
        )
        for y, p, imp in seeds
    ]
    row = {"episodes": per_seed[0]["episodes"]}
    for key in SWITCH_METRICS:
        mean, std = _mean_std([m[key] for m in per_seed])
        row[key] = mean if len(per_seed) > 1 else per_seed[0][key]
        row[f"{key}_std"] = std if len(per_seed) > 1 else None
    return row


def baseline_metrics(seeds: list, thr: float, merge_gap: int) -> dict[str, dict]:
    """Always-on and oracle baselines from the observed series (any seed has the same y)."""
    y, _, imp = seeds[0]
    states = {"always_on": switching.always_on(len(y)), "oracle": switching.oracle(y, imp, thr)}
    return {f: switching.switching_metrics(s, y, imp, thr, merge_gap) for f, s in states.items()}


def _switching_rows(best: pd.DataFrame, dcfg: decision.DecisionConfig) -> list[dict]:
    """D-16: every candidate per (family, zone, horizon, delay) plus the two baselines per zone,
    computed from saved val predictions (no retraining); `chosen` marks rule s1 (plan v2.11)."""
    series, rows = _Series(dcfg), []
    for zone in sorted(best["zone_rank"].unique()):
        zb = best[best["zone_rank"] == zone]
        square = int(zb["square_id"].iloc[0])
        thr = series.threshold("full", square)
        if thr is None:
            continue
        for _, b in zb.iterrows():
            seeds = seed_arrays(
                val_predictions(next(config.EXPERIMENTS_DIR.glob(f"{b['exp_id']}_*")))
            )
            h = int(b["horizon"])
            for delay, hold, ratio in product(dcfg.delay_min, dcfg.hold_min, dcfg.on_ratio):
                rows.append({"family": b["family"], "zone_rank": zone, "square_id": square,
                             "exp_id": b["exp_id"], "horizon": h, "delay_min": delay,
                             "hold_min": hold, "on_ratio": ratio, "chosen": False,
                             **switch_metrics(seeds, thr, h, hold, ratio, delay,
                                              dcfg.merge_gap)})  # fmt: skip
        for family, m in baseline_metrics(seeds, thr, dcfg.merge_gap).items():
            rows.append({"family": family, "zone_rank": zone, "square_id": square,
                         "chosen": False, **m})  # fmt: skip
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        if r["family"] not in dict(BASELINES):
            groups.setdefault((r["family"], r["horizon"], r["delay_min"]), []).append(r)
    for group in groups.values():
        hold, ratio = switching.choose_s1(group)
        for r in group:
            r["chosen"] = r["hold_min"] == hold and r["on_ratio"] == ratio
    return rows


SWITCH_NOTE = (
    "용량 셀은 예측이 켜기 기준 이상이면 켜고, 켠 뒤 최소 H분 유지한 다음 예측이 기준 아래면 "
    "끕니다(목적 단계 4). 예측은 입력 마지막 칸이 끝난 시각 t − h + 1에 나온다고 보고, 켜는 데 "
    "걸리는 시간만큼 늦게 반영합니다. 3단계 리드타임은 t − h 기준이라 10분 차이가 납니다.\n\n"
    "고른 조합(s1): 계열마다, 세 구역 **각각** 켜 둔 시간이 기준(H 0분·1.0θ = 3단계 경보)의 "
    "1.2배 이하인 조합 중 세 구역 놓친 혼잡 시간 합이 가장 적은 것, 같으면 켜 둔 시간, 그다음 "
    "켜고 끈 횟수가 적은 것. 후보는 `configs/decision.yaml`의 `switching`. LSTM은 시드별 결과의 "
    "평균±표준편차이고 선택은 평균으로 합니다. 전체 조합은 `results/switching.csv`에 있습니다. "
    "val은 선택용이라 낙관적입니다."
)


def _switch_line(square: int, label: str, name: str, r) -> str:
    cells = [
        _num(r["missed_min"], r.get("missed_min_std")),
        f"{_num(r['missed_episodes'], r.get('missed_episodes_std'))}/{int(r['episodes'])}",
        _num(r["on_min"], r.get("on_min_std")),
        _num(r["switches"], r.get("switches_std")),
    ]
    return f"| {square} | {label} | {name} | " + " | ".join(cells) + " |"


def _switching_table(rows: list[dict], report_horizon: int, delays: tuple[int, ...]) -> str:
    frame = pd.DataFrame(rows, columns=SWITCH_COLUMNS)
    labels = {f: label for f, _, label, _, _ in FAMILIES}
    lines = ["# 켜기/끄기 시뮬레이션 (val, full 단계)", "", SWITCH_NOTE, "",
             f"아래는 {report_horizon * 10}분 뒤 예측 기준입니다. 놓친 혼잡은 "
             "'놓친 횟수/전체 구간', 시간은 분입니다.", ""]  # fmt: skip
    for delay in delays:
        lines += [f"## 지연 {delay}분", "",
                  "| 구역 | 계열 | 조합 | 놓친 혼잡 시간 | 놓친 혼잡 | 켜 둔 시간 | 켜고 끈 횟수 |",
                  "|---|---|---|---|---|---|---|"]  # fmt: skip
        sub = frame[(frame["horizon"] == report_horizon) & (frame["delay_min"] == delay)]
        if sub.empty:
            lines.append(f"| | ({report_horizon * 10}분 뒤 실험 없음) | | | | | |")
        for zone in sorted(frame["zone_rank"].unique()):
            zf = sub[sub["zone_rank"] == zone]
            square = int(frame.loc[frame["zone_rank"] == zone, "square_id"].iloc[0])
            for family, _, _, _, _ in FAMILIES:
                ff = zf[zf["family"] == family]
                if ff.empty:
                    continue
                base = ff[(ff["hold_min"] == 0) & (ff["on_ratio"] == 1.0)].iloc[0]
                pick = ff[ff["chosen"].astype(bool)].iloc[0]
                name = f"선택 H {int(pick['hold_min'])}분·{pick['on_ratio']:.1f}θ"
                lines.append(_switch_line(square, labels[family], "기준 H 0분·1.0θ", base))
                lines.append(_switch_line(square, labels[family], name, pick))
            for family, label in BASELINES:
                r = frame[(frame["zone_rank"] == zone) & (frame["family"] == family)].iloc[0]
                lines.append(_switch_line(square, label, "기준선", r))
        lines.append("")
    return "\n".join(lines)


def rebuild_results(strict: bool = True) -> None:
    """strict=False (after `run`) keeps going without decision metrics if D-14 is invalid."""
    try:
        dcfg = config.load_decision()
    except TPError:
        if strict:
            raise
        log.warning("configs/decision.yaml을 읽을 수 없어 결정 지표를 생략")
        dcfg = None
    rows = load_rows(dcfg)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=COLUMNS).to_csv(config.RESULTS_DIR / "results.csv", index=False)
    lines = ["# 실험 결과 (val)", "", NOTE, "", "| " + " | ".join(COLUMNS) + " |",
             "|" + "---|" * len(COLUMNS)]  # fmt: skip
    lines += ["| " + " | ".join(_cell(r[c]) for c in COLUMNS) + " |" for r in rows]
    (config.RESULTS_DIR / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    frame = pd.DataFrame(rows)
    if frame.empty or "phase" not in frame:
        return
    if not ((frame["phase"] == "full") & (frame["status"] == "completed")).any():
        return
    frame["post_test"] = frame["post_test"].fillna(False).astype(bool)
    best = _best(frame)
    report = dcfg.report_horizon if dcfg else 6
    (config.RESULTS_DIR / "horizon.md").write_text(_horizon_table(best, report), encoding="utf-8")
    _horizon_figure(best, config.RESULTS_DIR / "figures" / "horizon_mae.png")
    if dcfg is None:
        return
    switch_rows = _switching_rows(best, dcfg)
    if switch_rows:
        pd.DataFrame(switch_rows, columns=SWITCH_COLUMNS).to_csv(
            config.RESULTS_DIR / "switching.csv", index=False
        )
        (config.RESULTS_DIR / "switching.md").write_text(
            _switching_table(switch_rows, dcfg.report_horizon, dcfg.delay_min), encoding="utf-8"
        )
