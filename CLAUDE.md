# traffic-predict

밀라노 모바일 트래픽 예측 실험입니다. 인터넷 트래픽 총량 상위 구역에서 10분 뒤 값을 예측하고, seasonal naive, ARIMA, LSTM을 비교합니다.

- 기획서(단일 원천): `docs/traffic-predict_기획서.md` (동결 v1). 기획서와 다르게 바꿔야 하면 코드를 고치기 전에 변경 통제 표로 영향을 보고합니다.
- 프로필: 데이터·ML 실험
- 스택: Python 3.12(uv), pandas, numpy, pyarrow, statsmodels, torch(CPU), matplotlib, pyyaml, filelock, pytest, ruff

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
