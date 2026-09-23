# ruff: noqa: E501  (README text template lines are long by nature)
"""Generate README.md from committed results (F-10). Numbers are read, never typed by hand.

Run after `tp results` or a new test: `uv run python scripts/make_readme.py`.
`tests/test_f10_readme.py` checks that the committed README matches this output.
"""

from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
FAM = {
    "naive_144": "어제 같은 시각",
    "naive_1": "마지막 관측값",
    "naive_1008": "지난주 같은 시각",
    "arima": "ARIMA",
    "lstm": "LSTM",
}
SQUARE = {1: 5161, 2: 5059, 3: 5259}

test = pd.read_csv(ROOT / "results/test/metrics.csv")
zones = test[test["zone_rank"] != "mean"].copy()
zones["zone_rank"] = zones["zone_rank"].astype(int)
means = test[test["zone_rank"] == "mean"]
switch = pd.read_csv(ROOT / "results/test/switching.csv")
results = pd.read_csv(ROOT / "results/results.csv")


def f(x, d=1):
    return "" if pd.isna(x) else f"{x:.{d}f}"


# --- test summary: mean relative MAE by horizon ---------------------------------------------
rel = means.pivot(index="family", columns="horizon", values="rel_mae").loc[list(FAM)]
rel_lines = ["| 계열 | 10분 뒤 | 30분 뒤 | 60분 뒤 |", "|---|---|---|---|"]
for fam, r in rel.iterrows():
    cells = [f(r[h], 3) for h in (1, 3, 6)]
    if fam == "lstm":
        cells = [f"**{c}**" for c in cells]
    rel_lines.append(f"| {FAM[fam]} | " + " | ".join(cells) + " |")

# --- vs last observed value, 95% CI per zone -----------------------------------------------
ci_lines = [
    "| 거리 | 계열 | " + " | ".join(f"구역 {SQUARE[z]}" for z in (1, 2, 3)) + " |",
    "|---|---|---|---|---|",
]
for h in (1, 3, 6):
    for fam in ("arima", "lstm"):
        cells = []
        for z in (1, 2, 3):
            r = zones[
                (zones["horizon"] == h) & (zones["family"] == fam) & (zones["zone_rank"] == z)
            ].iloc[0]
            mark = " ✔" if r["ci_lag1_high"] < 0 else ""
            cells.append(
                f"{f(r['diff_vs_lag1'])} [{f(r['ci_lag1_low'])}, {f(r['ci_lag1_high'])}]{mark}"
            )
        ci_lines.append(f"| {h * 10}분 | {FAM[fam]} | " + " | ".join(cells) + " |")

# --- on/off at 60 minutes, delay 0, three zones summed --------------------------------------
m = switch[(switch["horizon"] == 6) & (switch["delay_min"] == 0)]
sw_lines = [
    "| 계열 | 조합(val에서 선택) | 놓친 혼잡 시간 | 놓친 혼잡 | 켜 둔 시간 | 켜고 끈 횟수 |",
    "|---|---|---|---|---|---|",
]
for fam in FAM:
    g = m[m["family"] == fam]
    combo = f"H {int(g['hold_min'].iloc[0])}분·{g['on_ratio'].iloc[0]:.1f}θ"
    sw_lines.append(
        f"| {FAM[fam]} | {combo} | {g['missed_min'].sum():.0f}분 | "
        f"{g['missed_episodes'].sum():.0f}/{g['episodes'].sum():.0f} | "
        f"{g['on_min'].sum():,.0f}분 | {g['switches'].sum():.0f}회 |"
    )
for fam, label in (("oracle", "(참고) 미래를 아는 경우"), ("always_on", "(참고) 항상 켬")):
    g = switch[switch["family"] == fam]
    sw_lines.append(
        f"| {label} | 기준선 | {g['missed_min'].sum():.0f}분 | "
        f"{g['missed_episodes'].sum():.0f}/{g['episodes'].sum():.0f} | "
        f"{g['on_min'].sum():,.0f}분 | {g['switches'].sum():.0f}회 |"
    )

# --- experiment log with links ----------------------------------------------------------------
exp_lines = [
    "| 실험 | 단계 | 바꾼 요소 | 모델 | 거리 | 구역 | val MAE | 상태 | 바꾼 이유 |",
    "|---|---|---|---|---|---|---|---|---|",
]
for _, r in results.iterrows():
    folder = next((ROOT / "experiments").glob(f"{r['exp_id']}_*"))
    cfg = yaml.safe_load(
        (ROOT / "configs/experiments" / f"{r['exp_id']}.yaml").read_text(encoding="utf-8")
    )
    reason = " ".join(str(cfg["reason"]).split()).replace("|", "/")
    changed = "(루트)" if pd.isna(r["changed"]) else f"`{r['changed']}`"
    status = "완료" if r["status"] == "completed" else "실패"
    exp_lines.append(
        f"| [{r['exp_id']}](experiments/{folder.name}/) | {r['phase']} | {changed} | {r['model_type']} | "
        f"{int(r['horizon']) * 10}분 | {SQUARE[int(r['zone_rank'])] if r['phase'] == 'full' else '5259'} | "
        f"{f(r['val_mae'])} | {status} | {reason} |"
    )

# --- post-test sensitivity: which family has the least w x missed + on time ----------------
NAMES = {
    "naive_144": "lag144",
    "naive_1": "lag1",
    "naive_1008": "lag1008",
    "arima": "arima",
    "lstm": "lstm",
}
LABEL = {NAMES[k]: v for k, v in FAM.items()}
val_sw = pd.read_csv(ROOT / "results/switching.csv")
val_tot = (
    val_sw[(val_sw["horizon"] == 6) & (val_sw["delay_min"] == 0) & val_sw["chosen"]]
    .groupby("family")[["missed_min", "on_min"]]
    .sum()
)
test_tot = m.assign(family=m["family"].map(NAMES)).groupby("family")[["missed_min", "on_min"]].sum()


def winners(tot):
    """[(w_from, w_to, family)] over w in (0, 50], boundaries at the exact break-even ratios."""
    grid = [i / 100 for i in range(1, 5001)]
    best = [(w, (w * tot["missed_min"] + tot["on_min"]).idxmin()) for w in grid]
    spans, start = [], best[0]
    for prev, cur in zip(best, best[1:], strict=False):
        if cur[1] != prev[1]:
            a, b = tot.loc[prev[1]], tot.loc[cur[1]]
            edge = (b["on_min"] - a["on_min"]) / (a["missed_min"] - b["missed_min"])
            spans.append((start[0], edge, prev[1]))
            start = (edge, cur[1])
    spans.append((start[0], None, start[1]))
    return spans


def span_text(spans):
    parts = []
    for lo, hi, fam in spans:
        rng = (
            f"w < {hi:.1f}"
            if lo == 0.01
            else (f"w > {lo:.1f}" if hi is None else f"{lo:.1f} ~ {hi:.1f}")
        )
        parts.append(f"{rng}: {LABEL[fam]}")
    return "<br>".join(parts)


val_spans, test_spans = winners(val_tot), winners(test_tot)
both = [
    (max(a, c), min(b or 99, d or 99))
    for a, b, fa in val_spans
    for c, d, fb in test_spans
    if fa == fb == "lstm" and max(a, c) < min(b or 99, d or 99)
]
assert len(both) == 1
sens_lines = [
    "| | val (선택용) | test (한 번 평가) |",
    "|---|---|---|",
    f"| 비용이 가장 낮은 계열 | {span_text(val_spans)} | {span_text(test_spans)} |",
]
lag1008_val, lag1008_test = (
    val_tot.loc["lag1008", "missed_min"],
    test_tot.loc["lag1008", "missed_min"],
)

short = zones[(zones["horizon"] == 1) & zones["family"].isin(["arima", "lstm"])]["diff_vs_lag1"]
short_lo, short_hi = short.min(), short.max()
assert (zones[(zones["horizon"] == 6) & (zones["family"] == "lstm")]["ci_lag1_high"] < 0).all()
sw = {fam: m[m["family"] == fam][["missed_min", "on_min", "switches"]].sum() for fam in FAM}
oracle_on = switch.loc[switch["family"] == "oracle", "on_min"].sum()
exp1 = results.set_index("exp_id").loc["EXP-001", "val_mae"]
exp13 = results.set_index("exp_id").loc["EXP-013", "val_mae"]
lstm60 = rel.loc["lstm", 6]
last60 = rel.loc["naive_1", 6]

readme = f"""# 밀라노 모바일 트래픽 예측 → 용량 셀 켜기/끄기

밀라노 인터넷 트래픽이 가장 많은 격자 칸 3곳에서 **10·30·60분 뒤 트래픽**을 예측하고, 그 예측으로
**혼잡이 오기 전에 용량 셀을 켜는 운용 규칙**이 얼마나 잘 동작하는지 평가한 개인 실험입니다.
단순 기준선 → ARIMA → LSTM을 같은 데이터·같은 전처리·같은 test 구간에서 비교했고,
test(2013-12-16 ~ 12-22)는 모든 설정을 val로 확정한 뒤 **한 번만** 평가했습니다(`results/test/LOCK`).

## 결과 요약 (test, 한 번 평가)

**예측 오차** — 상대 MAE(어제 같은 시각의 MAE = 1, 세 구역 평균, 낮을수록 좋음)

{chr(10).join(rel_lines)}

- 60분 뒤 LSTM의 오차는 어제 같은 시각의 {lstm60:.0%}, 마지막 관측값({last60:.0%})보다 크게 낮습니다. 거리가 멀수록 모델의 이점이 커집니다.
- 10분 뒤에는 모델과 "마지막 관측값"의 MAE 차이가 {short_lo:.1f}~{short_hi:.1f}로 작고, 구역에 따라 95% 구간이 0을 포함합니다(아래 표). 60분 뒤 LSTM은 세 구역 모두 구간이 0 미만입니다.

**마지막 관측값 대비 MAE 차이와 95% 구간**(일별 블록 부트스트랩, ✔ = 구간 전체가 0 미만)

{chr(10).join(ci_lines)}

**60분 뒤 예측으로 용량 셀 켜기/끄기** — 세 구역 합계, 켜는 데 걸리는 시간 0분

{chr(10).join(sw_lines)}

- LSTM은 켜고 끄는 횟수가 가장 적고({sw["lstm"]["switches"]:.0f}회) 켜 둔 시간이 미래를 아는 경우보다 {sw["lstm"]["on_min"] / oracle_on - 1:.0%} 길 뿐이지만, 놓친 혼잡 시간({sw["lstm"]["missed_min"]:.0f}분)은 ARIMA({sw["arima"]["missed_min"]:.0f}분)보다 깁니다.
- 지난주 같은 시각은 혼잡을 거의 놓치지 않지만({sw["naive_1008"]["missed_min"]:.0f}분) 셀을 미래를 아는 경우보다 {sw["naive_1008"]["on_min"] / oracle_on - 1:.0%} 더 오래 켜 둡니다. **놓침 1분의 비용을 켜 둔 시간 몇 분으로 볼지에 따라 순위가 바뀝니다(아래 표).**

**놓침 비용 비율에 따른 순위 (사후 민감도 분석)** — 비용 = w × 놓친 혼잡 시간 + 켜 둔 시간, w = 놓침 1분이 켜 둔 몇 분만큼 비싼가. 60분 뒤, 지연 0분, 세 구역 합계.

{chr(10).join(sens_lines)}

- 이 표는 test 결과를 본 뒤 추가한 분석입니다. 셀 용량의 절대 비용을 몰라 w를 하나로 정하지 않고, w 범위별 결과를 보고합니다.
- **LSTM은 w가 약 {both[0][0]:.1f}~{both[0][1]:.1f}일 때 val과 test 모두에서 비용이 가장 낮습니다.** 놓침을 그보다 더 무겁게 보면 test에서는 지난주 같은 시각이 앞섭니다.
- 다만 지난주 같은 시각의 놓친 혼잡 시간은 val {lag1008_val:.0f}분, test {lag1008_test:.0f}분으로 주마다 크게 흔들렸습니다. test 주의 혼잡 시간대가 바로 전 주와 비슷했던 영향으로 보이며, 한 주의 결과로 일반화하기 어렵습니다.
- 켜고 끄는 횟수의 비용은 넣지 않았습니다.

- 전체 표: [test 결과](results/test/metrics.md), [켜기/끄기(test)](results/test/switching.csv), [거리별 비교(val)](results/horizon.md), [켜기/끄기 후보 전체(val)](results/switching.md)

![거리별 val MAE](results/figures/horizon_mae.png)

## 데이터
- Telecom Italia, "Telecommunications - SMS, Call, Internet - MI", Harvard Dataverse, DOI [10.7910/DVN/EGZHFV](https://doi.org/10.7910/DVN/EGZHFV). 10분 간격, 10,000개 격자.
- `CellID`는 무선 셀이 아니라 약 235m 격자 칸입니다. 이 실험은 **지역별 수요 예측**입니다.
- 값은 절대량이 아니라 스케일된 활동량이라 단위 없이 비교만 합니다.
- 구역: train 기간 인터넷 총량 상위 3곳(5161, 5059, 5259). 국가 코드별 행을 합산했습니다.

## 방법
- **분할(시간순, Europe/Rome):** train 11/01~12/08, val 12/09~12/15, test 12/16~12/22. 개발 단계(dev)는 11/04~11/17만 사용했습니다.
- **결측:** 연속 30분 이하는 직전 값으로 채우고(`is_imputed`), 채운 시각은 채점과 학습 손실에서 뺍니다. 이번 데이터에는 결측이 없었습니다.
- **예측 거리 h(10·30·60분):** 대상 시각 t를 t − h까지의 관측만으로 예측합니다. 기준선은 y[t − max(lag, h)], ARIMA는 train에서 적합한 파라미터로 h스텝 앞 예측(재적합 없음), LSTM은 거리마다 따로 학습합니다.
- **한 실험에 한 요소:** 모든 실험은 부모 실험에서 설정 키를 정확히 하나만 바꾸고, 바꾼 이유와 가설을 적습니다. 이 규칙은 코드가 강제합니다.
- **선택:** (구역, 거리, 계열)마다 val MAE가 가장 낮은 실험. test에 쓴 목록은 [`configs/final.yaml`](configs/final.yaml).
- **결정 지표:** 혼잡 임계치 θ = 0.7 × 구역별 train 99번째 백분위수. 예측 ≥ θ이면 경보, 혼잡이 시작되기 **전에** 나온 경보만 탐지로 인정합니다.
- **켜기/끄기 규칙:** 예측이 켜기 기준 이상이면 켜고, 최소 H분 유지한 뒤 기준 아래면 끕니다. 후보(H 0·30·60·120분 × 기준 1.0·0.9·0.8θ)는 미리 고정했고, val에서 "켜 둔 시간이 3단계 경보의 1.2배 이하인 조합 중 놓침 최소"로 골라 test에 그대로 적용했습니다.
- 모든 결정과 그 이유, 중간에 바뀐 설계(예: 선택 규칙 r1 → s1)는 [기획서](docs/traffic-predict_기획서.md)의 "설계 결정 기록"에 있습니다.

## 한계
- **val은 선택용이라 편향됨:** val은 설정을 고르는 데 썼으므로 val 수치는 낙관적입니다. 조기 종료까지 val로 한 LSTM은 더 편향됩니다. 공정한 비교는 test뿐입니다.
- **모든 모델은 train으로만 학습:** test 전에 train+val로 다시 학습하지 않았습니다. 계열 간 조건은 같지만 최신 1주를 학습에 쓰지 않았습니다.
- **공휴일:** train에 공휴일(11/01, 12/07, 12/08)이 섞여 있습니다.
- **test는 성탄 직전 쇼핑 주간:** 평소 주와 다를 수 있습니다.
- **부트스트랩 블록이 7개뿐:** test가 7일이라 신뢰구간이 거칩니다. 결정 지표와 켜기/끄기는 혼잡 구간 19개에 기반해 구간 없이 값만 보고합니다.
- 30·60분 실험은 10분 뒤에서 고른 하이퍼파라미터를 그대로 썼습니다. 셀 용량은 절대 단위를 몰라 train 피크 대비 비율로 가정했습니다.

## 재현
```bash
uv sync
# 원본 일별 파일(sms-call-internet-mi-2013-11-01.txt ~ 12-22)을 data/raw/ 또는 TP_RAW_DIR에 둡니다(.txt.gz도 가능)
uv run python -m tp prepare --phase full
uv run python -m tp run --config configs/experiments/EXP-013.yaml   # experiments/를 비운 작업 사본에서, EXP-001부터 순서대로
uv run python -m tp results
uv run pytest -q
```
- 시드: LSTM은 0·1·2(실험마다 다시 고정), naive·ARIMA는 시드를 쓰지 않습니다. CPU 전용, 결정적 연산.
- 기대 값: EXP-013(full, 어제 같은 시각) val MAE {exp13:.4f}, EXP-001(dev, 같은 모델) val MAE {exp1:.4f}. naive·ARIMA는 1e-9, LSTM은 같은 시드에서 1e-6 이내로 같게 나와야 합니다(같은 머신 기준, 다른 머신은 상대 1e-3).
- test는 `results/test/LOCK` 때문에 다시 실행되지 않습니다.

## 실험 기록
각 실험 폴더에는 설정(`config.yaml`), 이유·가설·시드·커밋(`meta.json`), 지표(`metrics.json`), 예측, 학습 곡선이 있습니다.
EXP-001~012는 dev 단계(구역 5259, 10분 뒤), EXP-013 이후는 full 단계입니다. EXP-015는 ARIMA(5,1,5) 수렴 실패로 EXP-017(2,1,2)로 대체했습니다.

<details>
<summary>전체 실험 {len(results)}개</summary>

{chr(10).join(exp_lines)}

</details>

## 구조
```
configs/      phases.yaml(분할), decision.yaml(임계치·켜기/끄기 후보), final.yaml(test 목록), experiments/
src/tp/       data → prep → models → eval → exp → cli
experiments/  실험별 폴더
results/      results.md(전체 실험), horizon.md, switching.md, figures/, test/
docs/         기획서(단일 원천)
```
"""
if __name__ == "__main__":
    (ROOT / "README.md").write_text(readme, encoding="utf-8", newline="\n")
