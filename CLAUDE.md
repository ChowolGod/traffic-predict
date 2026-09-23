# traffic-predict

밀라노 모바일 트래픽 예측 실험입니다. 인터넷 트래픽 총량 상위 구역에서 10분 뒤 값을 예측하고, seasonal naive, ARIMA, LSTM을 비교합니다.

- 기획서(단일 원천): `docs/traffic-predict_기획서.md` (동결 v1). 기획서와 다르게 바꿔야 하면 코드를 고치기 전에 변경 통제 표로 영향을 보고합니다.
- 프로필: 데이터·ML 실험
- 스택: Python 3.12(uv), pandas, numpy, pyarrow, statsmodels, torch(CPU), matplotlib, pyyaml, filelock, pytest, ruff

## 프로젝트 목적과 운영 규칙 (2026-09-23 확정)

### 최종 목적
트래픽 예측을 무선망 운용 결정에 연결한다.
- 현재 대상: 트래픽 최상위 3개 구역(5161, 5059, 5259) → "혼잡 예방" 시나리오
- 가정: 구역마다 항상 켜진 커버리지 셀 + 켜고 끌 수 있는 용량 셀.
  예측 트래픽이 임계치를 넘을 것 같으면 용량 셀을 미리 켠다.
- 과소예측(혼잡 놓침)이 과대예측(괜히 켬)보다 비용이 크다.
- 켜고 끄기가 잦지 않도록 최소 유지 시간 규칙을 둔다.
- 셀 용량은 절대 단위를 모르므로 구역별 train 피크 대비 비율로 정의한다.
- 향후(현재 범위 아님): 한산한 지역 1~2곳 추가 → "절전" 시나리오 비교.

### 데이터 해석 주의
- CellID는 무선 셀이 아니라 약 235m 격자 칸. 예측은 "지역별 수요 예측"이다.
- 값은 절대량이 아니라 스케일된 값. 절대량으로 해석하지 않는다.

### 진행 순서
1) 목적 확정 — 완료
2) 예측 거리 비교 (10/30/60분, 베이스라인: 10분 전 값·어제 같은 시각·지난주 같은 시각)
3) 결정 수준 평가 (놓친 혼잡, 불필요 경보 시간, 리드타임)
4) 규칙 기반 켜기/끄기 시뮬레이션
5) 모든 설정 동결 후 test 1회 실행

### 작업 규칙
- 한 번에 한 단계만 진행하고, 결과를 보고한 뒤 멈춘다. 다음 단계는 사용자 지시가 있을 때만.
- 임계치, 최소 유지 시간, 예측 거리 같은 결정 값은 스스로 확정하지 않고 후보와 근거를 제시한다.
- test(12/16~22)는 사용자가 "설정 동결, test 실행"이라고 지시하기 전까지 어떤 용도로도 쓰지 않는다.

## 명령
```bash
uv sync
uv run python -m tp prepare --phase dev
uv run python -m tp run --config configs/experiments/EXP-001.yaml
uv run python -m tp results
uv run pytest -q
uv run ruff check src tests
uv run ruff format --check src tests
```
원본 경로는 환경 변수 `TP_RAW_DIR`로 정합니다. 없으면 `data/raw`를 씁니다. `.txt`와 `.txt.gz`를 모두 읽습니다.

## 핵심 원칙
- test 구간(12/16~12/22)의 y는 `test` 명령만 읽습니다. `run`은 계열을 val 끝에서 잘라 모델에 넘깁니다.
- 한 실험에서는 설정 키를 정확히 하나만 바꿉니다. 이 규칙은 사람이 아니라 코드(M-11)가 강제합니다.
- 채운(`is_imputed`) 시각은 채점과 학습 손실에서 뺍니다. 결측은 앞값으로만 채웁니다(인과적).
- 모든 무작위성은 `set_seed`를 거칩니다. 같은 머신에서 같은 설정으로 돌리면 같은 지표가 나와야 합니다.
- 오류는 `TPError("E-xxxx", 메시지)`로만 냅니다. 번호는 기획서 4.3에 있는 것만 씁니다.
- (v2) 최종 목적은 무선망 운용 결정입니다. 예측 거리 h(1·3·6스텝)마다 대상 시각 t는 t − h까지의 관측만으로 예측하고, 결정 지표는 저장된 예측에서 계산합니다. test는 3·4단계 확정과 F-09 개정 전에는 실행하지 않습니다.

## 경계
- **항상:** 기능 하나를 끝낼 때 pytest와 ruff를 통과시키고, 기획서의 구현 진행표를 갱신합니다. 새 산출물에는 `_meta.json`/`meta.json`을 남깁니다.
- **물어보고:** 기획서에 없는 설정 키·오류 코드·의존성을 추가하는 일, 분할 날짜나 결측 규칙을 바꾸는 일, full 단계 실험을 실행하는 일은 먼저 묻습니다.
- **절대:** 사용자 승인 없이 `test`를 실행하지 않습니다. `results/test/LOCK`을 지우지 않습니다. 완료된 실험 폴더를 고치지 않습니다. `raw_dir`의 원본을 수정하거나 지우지 않습니다. 테스트를 건너뛰거나 기대값을 낮춰서 통과시키지 않습니다.

## 코드 스타일 예시
```python
def predict_naive(series: pd.DataFrame, lag: int, targets: pd.DatetimeIndex) -> pd.Series:
    """y[t - lag]를 그대로 예측값으로 쓴다. series는 10분 정규 인덱스(D-04)."""
    shifted = series["y"].shift(lag)
    missing = targets.difference(shifted.dropna().index)
    if len(missing):
        raise TPError("E-4004", f"평가 시각 불일치: {len(missing)}")
    return shifted.loc[targets]
```
