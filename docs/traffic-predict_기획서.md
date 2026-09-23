# 밀라노 모바일 트래픽 예측 기획서

상태: 동결 (2026-09-23, v1)
프로필: 데이터·ML 실험

## 다음 세션에게
- 현재 단계: 구현 중. 1(골격) 완료, 2(F-01)·3(F-02)·4(F-03)·5(F-04) 완료, 6(F-08+F-05) 코드 완료·실데이터 실행 남음, 7(F-06)·8(F-07) 완료, 다음은 9(F-09)
- 확정된 결정:
  - 프로필: 데이터·ML 실험 / 개인 프로젝트, 1인, 직접 구현
  - 결과 소비자: GitHub 포트폴리오(README). 정해진 마감 없음
  - 기준선: lag-144(전날 같은 시각)가 공식 기준선, lag-1(직전 값)은 참고 기준선으로 함께 보고
  - 1단계(1~2주치)는 train/val만 사용. test는 전체 기간으로 확장한 뒤 1회만 평가
  - 공식 test 구간: 2013-12-16 00:00 ~ 2013-12-22 23:50 (CET). 12/23 이후 데이터는 사용하지 않음
  - 비교 순서: seasonal naive → ARIMA → LSTM. 한 실험에서는 한 요소만 바꾸고 이유·가설을 기록
  - 지표: MAE, RMSE (원래 단위). 실험마다 시드 재고정, 결과 표·학습 곡선을 실험별로 저장
  - 구역: Must는 상위 1개, Should는 상위 3개(구역별로 따로 학습)
  - LSTM은 실험마다 시드 3개(0, 1, 2)로 돌려 평균±표준편차를 보고
  - 결측: 3칸(30분) 이하는 직전 관측값으로 채워(앞값 채우기, 레드팀 #1로 선형 보간에서 변경) 입력으로만 사용하고, 채운 시각은 모든 모델의 채점·학습 손실에서 제외. 더 긴 결측은 오류로 중단
  - 실행 시간 상한: 실험당 30분(CPU, 전체 기간, 구역 1개, 시드 모두 포함)
  - 평가 방식: 1스텝 롤링. 평가 구간에서는 재적합하지 않고 실제 관측만 반영
  - test는 잠금 시점의 K로 1회만. 잠금 후 실험은 허용하되 `post_test` 표시
  - 실험 기록은 평문 파일(YAML/JSON/CSV). 원본은 수동 다운로드
  - 원본 보관: 사용자가 디스크 공간을 마련 중. 코드는 `TP_RAW_DIR` + `.txt`/`.txt.gz`를 모두 읽어 어떤 보관 방식이든 대응
- 남은 일: 구현 진행표 순서대로 구현. 원본을 받으면 F-01의 실제 데이터 확인(진행표 2 비고)을 먼저 끝낸다. 디스크 여유 52GB라 공간 문제는 해소
- 제약: 마감 없음 · 채점 기준 없음 · 제출물 = GitHub 저장소 README · 구현함
- 최근 변경: 없음 (v1)
- 구현 환경(2026-09-23 확인): 커밋은 기능 완료마다 자동(`feat(F-xx): …`, push 안 함). 원본은 기본 경로 `data/raw`(사용자가 받는 중). CPU만 사용. 디스크 여유 52GB

## 1. 개요
- **문제:** 셀 단위 모바일 인터넷 트래픽의 단기(10분 뒤) 수요 예측
- **대상 사용자:** 본인(학습)과 포트폴리오를 보는 사람(채용 담당자·엔지니어). 결과 표·그래프·실험 기록으로 방법론의 엄밀함을 보여 주는 것이 목적
- **한 문장 요약:** 밀라노 격자 중 인터넷 트래픽 총량 상위 구역에서 10분 뒤(1스텝) 인터넷 활동량을 예측하고, seasonal naive → ARIMA → LSTM을 같은 test 구간·같은 전처리 조건에서 MAE·RMSE로 비교한다.

### 검토한 방향
사용자가 방향을 정해서 왔으므로 방향 비교는 생략함.

### 전제 (구현 첫 단계 F-01에서 실제 파일로 확인)
| # | 전제 |
|---|---|
| A1 | 데이터는 Harvard Dataverse DOI 10.7910/DVN/EGZHFV의 일별 파일(2013-11-01 ~ 2014-01-01). 탭 구분, 헤더 없음. 열: `square_id, time_ms, country_code, smsin, smsout, callin, callout, internet` |
| A2 | 같은 (구역, 시각)의 행이 국가 코드별로 나뉘어 있다 → 국가 코드를 합쳐 (구역, 시각)당 한 값으로 만든다. 빈 값은 0으로 본다 |
| A3 | `internet`은 바이트가 아니라 정규화된 활동량이다. 지표는 이 단위 그대로 보고한다 |
| A4 | `time_ms`는 UTC epoch(ms)이고, 분할·요일 판단은 Europe/Rome 기준이다(사용 구간은 모두 UTC+1) |
| A5 | CPU로만 학습한다(결정적 재현을 위해) |

### 핵심 가정 (틀리면 계획이 무너지는 것)
- 상위 구역의 트래픽이 하루 주기로 충분히 규칙적이어서 1스텝 예측의 모델 간 차이가 의미 있게 드러난다.
- 원본의 10분 구간 결측이 적어 하나의 보간 규칙으로 처리할 수 있다.
- 1~2주치로 만든 파이프라인이 전체 기간에서도 설정 변경만으로 돌아간다(일별 사전 집계 캐시로 메모리를 제한).

### 위험과 대응
| 위험 | 영향 | 대응 |
|---|---|---|
| 기준선이 약하면 모델 우위가 부풀려진다 | 결론이 틀림 | lag-1 참고 기준선을 함께 보고 |
| 연말 연휴의 분포 변화 | test가 평소 예측력을 반영하지 못함 | test를 12/16~12/22로 고정하고 12/23 이후는 제외 |
| test를 여러 번 보게 됨 | 누수·과대평가 | test는 1회만 평가, 잠금 파일로 강제 |
| SARIMA를 주기 144로 적합하면 매우 느림 | 실험이 멈춤 | 4단계에서 계절성 처리 방식을 결정(계절 차분 또는 푸리에 항) |
| LSTM 결과가 시드에 따라 흔들림 | 한 요소 비교의 신뢰도가 떨어짐 | 시드 3개의 평균±표준편차로 보고 |
| 원본 용량: 61개 파일 20.5GB(하루 약 280~376MB). 사용 구간 11/01~12/22만 약 17GB인데, 디스크 여유는 8GB(2026-09-23 확인) | 전체 기간 원본을 한 번에 둘 수 없음. 메모리(16GB)도 파일 하나씩 읽어야 함 | 파일 단위로 읽어 일별 집계 캐시(parquet)에 저장. 원본은 `TP_RAW_DIR` + `.gz` 지원으로 어떤 보관 방식이든 대응(D-12) |
| train에 공휴일(11/01, 12/07, 12/08)이 있고, test는 성탄 직전 쇼핑 주간 | 평소 패턴과 다른 날이 섞임 | README 한계 절에 명시(F-10) |
| Dataverse 목록은 61개인데 기간은 62일 | 하루치가 빠져 있을 수 있음 | F-01에서 사용 구간의 파일이 모두 있는지 확인. 빠진 날이 사용 구간 안이면 변경 통제 |

## 2. 범위
- **핵심 흐름:** 원본 일별 파일 → 구역·시각별 집계 캐시 → 상위 구역 선택(train 구간) → 전처리·분할 → 모델 학습(val로 선택) → 평가(MAE·RMSE) → 결과 표·학습 곡선 → (전체 기간에서) test 1회 평가

### 분할
| 단계 | train | val | test |
|---|---|---|---|
| 1단계 (개발, 2주) | 11/04 ~ 11/13 (월~수, 10일) | 11/14 ~ 11/17 (목~일, 4일) | 없음 |
| 전체 기간 | 11/01 ~ 12/08 | 12/09 ~ 12/15 (월~일) | 12/16 ~ 12/22 (월~일) |

### Must
| ID | 이름 | 설명 | 완료 조건 | 예외 |
|---|---|---|---|---|
| F-01 | 원본 로드·집계 | 일별 원본을 읽어 국가 코드를 합치고, (square_id, time) → internet 값을 일별 parquet 캐시로 저장 | (1) 캐시 스키마가 정의와 일치 (2) 무작위 3개 (구역, 시각)의 값이 원본에서 직접 합산한 값과 일치 (3) 캐시가 있으면 원본을 다시 읽지 않음 (4) 전제 A1~A4가 실제 파일과 맞는지 확인 결과를 로그로 남김 | E-1001, E-1002, E-1003 |
| F-02 | 상위 구역 선택 | train 구간의 internet 총량으로 구역을 정렬해 상위 K개를 저장(Must에서는 K=1) | 같은 입력이면 같은 목록이 나옴(동률이면 square_id가 작은 쪽). val·test 구간 데이터는 계산에 쓰이지 않음(테스트로 검증) | E-1001(train 구간 원본 없음) |
| F-03 | 전처리·분할 | Europe/Rome 기준 날짜 판정, 10분 정규 인덱스, 결측 처리, 고정 날짜로 분할 | (1) 구간 경계가 설정 날짜와 정확히 같고 겹침 0 (2) `run`에 넘기는 계열에 test 구간이 없음 (3) 연속 3칸 이하의 결측은 직전 관측값으로 채우고(앞값 채우기) `is_imputed=True`로 표시 (4) 연속 4칸 이상의 결측은 오류로 중단 (5) 사용 구간의 모든 날이 144칸 | E-2001, E-2002 |
| F-04 | 평가 | 원래 단위로 MAE·RMSE를 계산. 모든 모델을 같은 평가 시각 집합(보간 시각 제외)에서 비교 | (1) 손으로 계산한 예제와 1e-9 이내로 일치 (2) 보간된 목표 시각은 지표에서 빠짐 (3) 모델 간 예측 시각 집합이 다르면 오류 | E-4004 |
| F-05 | 기준선 | lag-144(공식), lag-1(참고) 예측 | 예측값이 각각 y[t-144], y[t-1]과 정확히 같음 | 없음 (예측 대상은 val·test뿐이라 항상 144칸 이상의 과거 기록이 있음) |
| F-06 | ARIMA | train에서 적합, val로 차수 선택. 평가 구간에서는 재적합 없이 실제 관측을 반영하며 1스텝 롤링 예측 | (1) 지표·예측 파일 저장 (2) 같은 설정으로 다시 실행하면 지표가 동일 (3) 전체 기간·구역 1개 기준으로 30분 이내 | E-3001, E-3003 |
| F-07 | LSTM | 입력 창 → 1스텝 예측. val 손실로 조기 종료, 시드 3개(0, 1, 2) 각각 학습 | (1) 시드마다 학습 곡선(train/val 손실) png·csv 저장. 스케일러 통계가 train에서만 계산됨 (2) 평균±표준편차 지표 저장 (3) 같은 시드로 다시 실행하면 지표가 1e-6 이내로 동일 (4) 시드 3개 합쳐 30분 이내 | E-3002, E-3003 |
| F-08 | 실험 기록·재현 | 실험마다 설정·부모 실험·바꾼 요소·이유·가설·시드·결과를 저장하고, 전체 결과 표를 자동 갱신 | (1) 실험 폴더에 설정·지표·예측·곡선이 있음 (2) 부모 대비 바뀐 설정 키가 정확히 1개가 아니면 실행 거부 (3) 결과 표 한 곳에 모든 실험이 나열됨 | E-4001, E-4002, E-4003 |
| F-09 | test 1회 평가 | 전체 기간에서 계열별 최종 설정(lag-144, lag-1, ARIMA 최선, LSTM 최선)을 한 번에 test로 평가하고 잠금 | (1) 두 번째 실행은 오류로 거부됨 (2) `--confirm`이 없거나 dirty면 거부됨 (3) test 결과 표에 MAE·RMSE·상대 MAE·lag-144 대비 차이의 95% 신뢰구간이 있음 (4) 부트스트랩이 같은 시드에서 같은 구간을 냄 | E-4005, E-4006 |

### Should
| ID | 이름 | 설명 | 완료 조건 |
|---|---|---|---|
| F-10 | README 결과 정리 | 결과 표·대표 그래프·실험 흐름·한계를 README에 정리 | (1) README에서 test 표와 각 실험 폴더로 가는 링크가 동작 (2) 한계 절에 다음이 있음: val은 선택용이라 편향됨(LSTM은 더), 모든 모델은 train으로만 학습, 공휴일(11/01, 12/07, 12/08)이 train에 있음, test는 성탄 직전 쇼핑 주간, 부트스트랩 블록이 7개뿐 |
| F-11 | 상위 3개 구역 확장 | K=3으로 같은 최종 설정을 구역별로 학습·평가하고, 구역별 지표와 평균을 보고 | `phases.yaml`의 full `k`를 3으로 바꾸고 `prepare`한 뒤, 구역 2·3은 구역 1의 최종 실험을 부모로 한 자식 실험(`changed: zone_rank`)으로 실행. 결과 표에 구역별 행이 나오고, test 표에 구역별 행과 평균 행이 나옴. test 잠금 전에 끝나야 F-09에 포함됨(3.2) |

### 제외·향후 확장
| 항목 | 제외 이유 |
|---|---|
| 12/23 ~ 01/01 데이터 | 연휴의 분포 변화. 평소 예측력을 재는 목표와 맞지 않음 |
| SMS·통화를 외생 변수로 사용 | 목표는 인터넷 단일 변수 예측. 요소가 늘어나면 비교가 흐려짐 |
| 이웃 구역·공간 모델(GNN 등) | 범위 밖. 구역별 단일 시계열로 한정 |
| 여러 스텝 앞(30분·1시간) 예측 | 목표 수평선은 10분 하나 |
| Transformer 등 다른 신경망 | 비교 순서는 naive → ARIMA → LSTM으로 확정 |
| 자동 하이퍼파라미터 탐색 | "한 실험에 한 요소" 원칙과 충돌 |
| GPU 학습 | 결정적 재현을 우선함 |
| 대시보드·웹 시각화 | 결과 소비처는 README |
| 원본 자동 다운로드 스크립트 | 수동 다운로드로 충분. 받는 방법은 README에 적음 |
| 조기 종료용 holdout 분리 | 편향을 명시하는 것으로 처리. 분할 규칙이 복잡해짐(레드팀 #3) |
| lag-1008(지난주 같은 시각) 참고 기준선 | 기준선은 lag-144·lag-1로 확정. 요일 효과는 한계로 적음 |
| 결과 표의 모델별 입력 정보 열 | `config.yaml`에서 확인할 수 있음 |
| sweep 그룹 크기 열 | `changed`와 부모로 추적할 수 있음 |
| `MKL_CBWR=COMPATIBLE` 고정 | 같은 머신 재현이 목표. 다른 머신은 상대 1e-3 허용 |

## 3. 뼈대

### 3.1 상호작용 주체와 권한
| 주체 | 하는 일 | 할 수 있는 것 | 할 수 없는 것 |
|---|---|---|---|
| 실행자(본인) | CLI로 파이프라인·실험 실행 | 모든 단계 실행, 실험 생성, test 실행 1회 | 잠금 파일을 지워 test 재평가, 완료된 실험 폴더 수정 |
| 구현 에이전트(Claude) | 코드 작성·테스트, dev 단계 실험 실행 | 코드·테스트 수정, dev·full 단계의 val 실험 실행 | 사용자 승인 없이 test 실행, 원본 데이터 수정, 잠금 해제 |
| 데이터 출처(Harvard Dataverse) | 원본 일별 파일 제공 | — | — (원본은 `raw_dir`(D-12)에 두고 읽기 전용) |
| 결과 소비자(README 독자) | 결과 표·그래프·실험 기록 열람 | 저장소의 `results/`, `experiments/` 열람 | — |

### 3.2 상태와 전이
**파이프라인 산출물**
| 단계 | 산출물 | 만드는 기능 | 재실행 조건 |
|---|---|---|---|
| S0 원본 | `<raw_dir>/sms-call-internet-mi-YYYY-MM-DD.txt` 또는 `.txt.gz` (D-12) | 수동 배치 | 없음(불변) |
| S1 일별 캐시 | `data/interim/daily/YYYY-MM-DD.parquet` | F-01 | 파일이 없을 때, 집계 코드 버전(`AGG_VERSION`)이 바뀌었을 때, `--force` |
| S2 구역 목록 | `data/processed/<phase>/zones.json` | F-02 | 분할 설정 또는 K가 바뀌었을 때 |
| S3 구역 시계열 | `data/processed/<phase>/series_<square_id>.parquet` | F-03 | S1·S2가 바뀌었거나 결측 규칙 버전이 바뀌었을 때 |
| S4 실험 | `experiments/EXP-###_<name>/` | F-05~F-08 | 아래 실험 상태를 따름 |
| S5 test 결과 | `results/test/` + `results/test/LOCK` | F-09 | 재실행 불가 |

각 산출물에는 입력 파일 목록과 코드 버전을 담은 메타(`_meta.json`)를 함께 저장한다. 메타가 현재 설정과 다르면 다시 만든다.

**단계(phase)**: `dev`(11/04~11/17, test 없음) → `full`(11/01~12/22). dev 실험 결과는 full과 비교하지 않고, 최종 설정은 full 단계의 val에서 고른다(3.3 선택 규칙). dev에서 찾은 설정은 `changed: phase`로 full에 옮긴다.

**실험 상태**
| 상태 | 전이 | 주체 | 불허 전이 |
|---|---|---|---|
| 생성 → 실행 중 | `run` 명령 | 실행자·에이전트 | 설정 검증(E-4001, E-4002) 실패 시 생성 거부 |
| 실행 중 → 완료 | 지표·산출물 저장 성공 | 코드 | — |
| 실행 중 → 실패 | 예외·시간 초과 | 코드 | — |
| 실패 → 실행 중 | `run --retry` (같은 ID) | 실행자·에이전트 | 완료 → 실행 중 (완료된 실험은 불변. 바꾸려면 새 ID) |
| 실행 중(중단됨) → 실행 중 | `run --retry`. 프로세스가 죽어 `meta.json`의 상태가 `running`으로 남은 경우(락을 잡을 수 있으면 중단된 것으로 봄, 3.3 동시 실행) | 실행자·에이전트 | 재시도 시 설정 해시가 다르면 불가(E-4003) |

**test 상태**: 미평가 → 평가됨(LOCK 생성). 평가됨 → 미평가는 불가.
- test는 잠금 시점의 K(구역 수)로 한 번만 평가한다. 그때까지 F-11이 끝나지 않았으면 test는 구역 1개로 진행하고 F-11은 제외 목록으로 옮긴다.
- 잠금 뒤에도 full 단계의 val 실험은 허용한다. 대신 `post_test=true`로 기록해 결과 표에 표시하고, test에는 쓰지 않는다.

### 3.3 공통 정책
| 항목 | 결정 | 이유 |
|---|---|---|
| 난수 시드 | 실험 시작 때와 시드별 학습 시작 때 `set_seed(seed, threads)`로 `random`·`numpy`·`torch`를 다시 고정. `torch.use_deterministic_algorithms(True)`, `torch.set_num_threads(threads)`, DataLoader 셔플용 generator에도 시드 지정. `OMP_NUM_THREADS`·`MKL_NUM_THREADS`는 CLI 프로세스 시작 시 numpy import 전에 `runtime.threads`로 설정(M-16) | 요구: 실험마다 시드 재고정. CPU에서 결정적 재현 |
| 시드 값 | LSTM은 상수 `LSTM_SEEDS=(0, 1, 2)`(설정으로 바꿀 수 없음). naive·ARIMA는 시드를 쓰지 않으며 기록 값은 −1 | 시드 3개로 평균±표준편차. 시드 수가 다른 실험끼리 비교되는 일을 막음 |
| 분할 규칙 | 날짜는 Europe/Rome 기준 하루 단위, 양끝 포함. 구간은 **예측 대상 시각**으로 판정. 입력(과거 값)은 앞 구간 데이터를 써도 됨. 사용 구간(11/01~12/22)은 서머타임 종료(10/27) 뒤라 모든 날이 144칸(테스트로 확인) | 1스텝 롤링에서 val 첫 시각도 예측할 수 있게. 대상 시각이 겹치지 않으므로 누수 없음 |
| 미래 데이터 차단 | `run`(I-02)은 D-04 계열을 **val 마지막 시각에서 잘라** 모델에 넘기고, 예측 파일에는 train·val 행만 저장한다. test 구간의 y는 `test`(I-05)만 읽는다 | test를 여러 번 보는 경로 차단 |
| 결측 채우기 | 연속 3칸 이하 결측은 **직전 관측값으로 채움**(앞값 채우기, 인과적)하고 `is_imputed=True`. 선형 보간은 쓰지 않음 | 선형 보간은 결측 뒤의 실제 값을 입력에 섞어 누수를 만듦 |
| 평가 지표 | MAE, RMSE. 원래 단위, 대상 시각 중 `is_imputed=False`만. 소수 4자리로 보고. LSTM은 시드별 지표를 계산한 뒤 평균과 표준편차(ddof=1)를 보고. 예측값은 음수여도 자르지 않음 | 요구 사항 + 결측 결정 |
| 학습 손실 | LSTM의 학습 손실과 조기 종료용 val 손실도 `is_imputed=False`인 목표만 사용 | 채운 값으로 채점하지 않는다는 원칙을 학습에도 적용 |
| val 수치의 성격 | val은 **선택용**이라 모든 계열이 낙관적으로 편향되고, 조기 종료까지 val로 하는 LSTM은 더 편향됨. 공정한 비교는 test뿐임을 결과 표 머리말과 README에 적음 | 레드팀 #3. 분할을 복잡하게 만들지 않고 명시로 처리 |
| 학습 데이터 범위 | 모든 모델은 train으로만 학습(ARIMA 적합, LSTM 가중치·스케일러). val·test 구간에서는 실제 관측을 입력으로만 반영하고, train+val로 다시 학습하지 않음 | 계열 간 조건을 같게 함. README에 명시 |
| 상대 MAE | `rel_mae` = 모델 MAE ÷ 같은 계열·같은 평가 시각에서 계산한 lag-144 MAE. lag-144 MAE는 실험마다 M-10이 내부적으로 계산(따로 실험을 만들지 않음) | 구역 간 평균과 README 가독성 |
| 불확실성(test) | 계열마다 "모델 MAE − lag-144 MAE"의 95% 신뢰구간을 **일별 블록 부트스트랩**으로 계산(블록 = Europe/Rome 하루 144칸, 반복 2000, 시드 0, 백분위수 방식). test가 7일이라 블록이 7개뿐이어서 구간이 거칠다는 점을 한계로 적음 | test 1주 단일 구간에서 우연한 차이인지 판단 |
| 코드 상태 | full 단계의 `run`·`sweep`과 `test`는 git 작업 트리가 dirty면 거부(E-4001, E-4006). dev는 경고만 하고 `git_dirty=true`로 기록 | 결과가 어느 커밋에서 나왔는지 보장 |
| 모델 비교 조건 | 결과 표의 `compare_group`(D-09)이 같은 행끼리만 비교. 같은 (phase, zone_rank) 안에서 평가 시각 집합이 기준 실험과 다르면 E-4004 | 공정한 비교 |
| 최종 설정 선택 규칙 | 계열마다 같은 `compare_group`에서 full·완료·`post_test=false`인 실험 중 **val MAE(LSTM은 시드 평균)가 가장 작은 것**. 동률이면 ID가 작은 것. M-14가 `final.yaml`이 이 규칙과 맞는지 검증(E-4006) | test 전 선택을 사후 판단에 맡기지 않음 |
| 재현 허용 오차 | 같은 머신·같은 `uv.lock`·같은 `runtime.threads`에서: naive·ARIMA는 1e-9, LSTM은 같은 시드에서 1e-6. 다른 머신에서는 상대 오차 1e-3을 재현으로 인정 | BLAS·CPU에 따라 부동소수 결과가 달라짐 |
| 실험 이름·번호 | `EXP-###`(세 자리). 설정에 직접 적고, 기존 최대 번호+1이어야 함(실패한 실험 포함). 폴더 `experiments/EXP-###_<name>/`(`name`은 D-05) | 한 요소 변경 규칙을 추적 |
| 루트 실험 | phase마다 루트(부모 없음)는 정확히 1개이고, `model.type=naive, naive.lag=144, zone_rank=1`로 고정 | 부모 없는 실험을 여러 개 만들어 한 요소 규칙을 우회하는 일을 막음 |
| "한 요소" 판정 | 부모와 **최종 확정 설정**을 비교해 바뀐 키가 정확히 1개. 최종 확정 설정에는 현재 `model.type`(ARIMA는 `seasonal`까지)에 해당하는 키만 남고, 해당하지 않는 키는 비교 대상이 아님. 비교에서 빼는 키: `id, name, parent, changed, reason, hypothesis, runtime.time_limit_min`. `model.type`을 바꾸면 새 계열의 하위 키는 모두 4.2 초기값이어야 함. dev 실험을 full로 옮길 때는 dev 실험을 부모로 두고 `changed: phase`로 만듦(이 경우 다른 키는 그대로) | 코드로 강제하고 우회를 막음 |
| sweep | 키 하나의 값 목록을 받아 값마다 자식 실험을 만듦(키 하나에 대한 격자). 모든 값을 먼저 검증하고 하나라도 실패하면 아무것도 만들지 않음. 실행 중 한 값이 실패해도 다음 값으로 진행하고, 끝에 요약을 출력 | val로 차수·크기를 고르되 한 요소 규칙 유지 |
| 산출물 경로 | 실험별: `config.yaml`(최종 확정본), `meta.json`(D-13), `metrics.json`, `predictions.parquet`, `model/`(D-11), `curves/seed{n}.csv·png`, `run.log`. 전체: `results/results.csv·md` | 요구: 결과 표·학습 곡선을 실험별로 저장 |
| 하이퍼파라미터 기록 | YAML 설정 파일 하나가 원천. 코드 안의 기본값은 실행 시 풀어서 `config.yaml`에 모두 기록 | 숨은 기본값 방지 |
| 재현 절차 | `uv sync`(`uv.lock`) → 원본을 `raw_dir`(D-12)에 배치 → `prepare` → 같은 `config.yaml`로 `run` 재실행 → 지표가 허용 오차 이내 | 제3자 재현 |
| 실행 시간 | 실험당 30분 상한. ARIMA 적합이 끝난 뒤, LSTM은 에폭마다 경과 시간을 확인해 넘었으면 E-3003으로 실패 처리(적합 도중에는 끊지 않음. 한 번의 적합은 `maxiter`로 제한) | Windows에는 `signal.alarm`이 없음. 단순하게 구현 |
| 동시 실행 | `run`·`sweep`·`test`는 `experiments/.lock`을 OS 파일 락(`filelock`)으로 잡고 실행. 락을 잡을 수 있으면 `running` 상태로 남은 실험은 죽은 프로세스의 흔적으로 간주 | 중단된 실험과 실행 중인 실험을 구분 |
| 로그 | Python `logging` INFO. 콘솔과 실험 폴더의 `run.log`에 동시에 기록 | 실패 원인 추적 |
| 오류 코드 체계 | `E-1xxx` 데이터, `E-2xxx` 전처리·분할, `E-3xxx` 모델·학습, `E-4xxx` 실험 규칙·평가. 예외 클래스 `TPError(code, message)`, CLI 종료 코드 1 | 원인 영역을 번호로 구분 |
| 시간대 | 저장은 UTC. 분할·요일·그래프·푸리에 위상은 `Europe/Rome` | 전제 A4 |
| 산출물 유효성 | 코드 버전은 상수 `AGG_VERSION`(S1)과 `IMPUTE_VERSION`(S3)만 뜻함(git 커밋 아님). S1은 (날짜, `AGG_VERSION`)으로, S2는 (phase 날짜 범위, k)로, S3은 (square_id, phase 날짜 범위, `AGG_VERSION`, `IMPUTE_VERSION`)으로 유효성을 판정. 원본 파일명·크기는 기록용으로만 둠 | 원본을 압축하거나 옮겨도 캐시가 유지되게. K를 바꿔도 기존 실험이 무효가 되지 않게 |

## 4. 상세 설계

### 4.1 인터페이스 (CLI: `uv run python -m tp <명령>`)
| ID | 명령 | 기능 ID | 입력 | 출력(산출물) | 호출하는 구현 | 예외 |
|---|---|---|---|---|---|---|
| I-01 | `prepare --phase {dev,full} [--force]` | F-01, F-02, F-03, F-11 | `configs/phases.yaml`, `raw_dir`(D-12) | S1 일별 캐시, 진단 보고(`data/interim/inspect.json`: 날짜 범위 밖 행 수, 파일 전체에서 빠진 시각 목록), S2 `zones.json`, S3 구역 시계열 | M-15, M-01, M-02, M-03, M-04, M-05 | E-1001, E-1002, E-1003, E-2001, E-2002 |
| I-02 | `run --config <경로> [--retry]` | F-05, F-06, F-07, F-08 | 실험 설정 YAML(D-05) | `experiments/EXP-###_<name>/` 전체, `results/results.csv·md` 갱신 | M-11 → M-05~M-10, M-12, M-13 | E-2001, E-2003, E-3001, E-3002, E-3003, E-4001, E-4002, E-4003, E-4004, E-4007 |
| I-03 | `sweep --parent <EXP-###> --key <설정 키> --values <JSON 리스트> --start-id <EXP-###> --name <접두어> --reason <문장> --hypothesis <문장>` | F-06, F-07, F-08 | 부모 실험, 바꿀 키 1개와 값 목록(JSON으로 해석해 D-05 타입으로 검증) | 값마다 `configs/experiments/EXP-###.yaml`(번호는 `--start-id`부터 순서대로, `name`은 `<접두어>-<값>`)을 만들고 차례로 실행 | M-11 (`sweep`) | I-02와 같음 |
| I-04 | `results` | F-08, F-10 | `experiments/*/` | `results/results.csv·md`, `results/figures/*.png` 재생성. `meta.json`이 없거나 읽을 수 없는 폴더는 `status=corrupt`로 표시하고 건너뜀 | M-12 | 없음 |
| I-05 | `test --final configs/final.yaml --confirm` | F-09 | 최종 설정(D-06) | `results/test/metrics.csv·md`, `predictions.parquet`, `LOCK` | M-14 → M-05, M-07~M-10, M-13, M-17 | E-2003, E-4004, E-4005, E-4006, E-4007 |

- 모든 명령은 성공하면 종료 코드 0, `TPError`가 나면 1(메시지 `[E-xxxx] 설명`). 예상하지 못한 예외는 2.
- `run`은 먼저 설정을 검증하고, 그다음에 실험 폴더를 만든다. 검증에 실패하면 폴더를 만들지 않는다.
- `test` 순서: (0) `--confirm`이 없거나 작업 트리가 dirty면 거부(E-4006) (1) 락 획득, LOCK 존재 확인(E-4005) (2) `final.yaml`과 참조 실험·D-11 파일 전부 검증·로드(E-4006) (3) 모든 예측·지표를 메모리에서 계산 (4) 임시 폴더에 결과와 LOCK을 쓴 뒤 `results/test/`로 한 번에 이름 변경. (1)~(3)에서 실패하면 아무것도 쓰거나 출력하지 않는다.

### 4.2 구현 단위 (`src/tp/`)
| ID | 모듈·함수 | 책임 | 입력 → 출력 | 의존 |
|---|---|---|---|---|
| M-01 | `data/raw.py: read_raw_day(date) -> DataFrame` | 원본 한 파일 읽기와 형식 검증. `raw_dir`(D-12)에서 `.txt`를 먼저 찾고, 없으면 `.txt.gz`를 찾음. 파일 날짜 밖의 행은 오류가 아니라 개수만 집계 | 날짜 → D-01 행 | M-15 |
| M-02 | `data/cache.py: build_daily_cache(date, force)`, `write_inspect()` | 국가 코드 합산, S1 캐시와 `_meta.json` 쓰기, 진단 보고 | D-01 → D-02 | M-01 |
| M-03 | `data/zones.py: select_top_zones(phase_cfg, k)` | train 구간 총량 상위 K개 선택. dev와 full의 상위 구역이 다르면 경고 로그 | D-02 → D-03 | M-02, M-15 |
| M-04 | `prep/series.py: build_series(square_id, phase_cfg)` | phase 범위 ±1일의 캐시를 이어 붙여 10분 정규 인덱스 생성, 결측 판정·앞값 채우기, `is_imputed`, 구간 표시 | D-02 → D-04 | M-02, M-05 |
| M-05 | `prep/split.py: segment_of(ts, phase_cfg)`, `cut_until(series, segment)` | 대상 시각의 구간 판정, 계열을 특정 구간 끝에서 자르기 | 시각 → 구간 | M-15 |
| M-06 | `prep/scale.py: Scaler(kind).fit(train).transform/inverse` | LSTM 입력 스케일링(train에서만 적합). ARIMA는 스케일링하지 않음 | 배열 → 배열 | — |
| M-07 | `models/naive.py: predict_naive(series, lag, targets)` | y[t−lag] 예측 | D-04 → 예측 | — |
| M-08 | `models/arima.py: fit_arima(train, cfg)`, `predict_arima(params, series, targets)` | statsmodels `ARIMA`로 train에서 적합한 뒤, 같은 파라미터를 잘린 계열에 적용(`apply`, 재적합 없음)해 1스텝 예측. 계절성 처리는 아래 참고. 파라미터를 D-11로 저장·로드 | D-04 → 예측, D-11 | M-15 |
| M-09 | `models/lstm.py: train_lstm(series, cfg, seed)`, `predict_lstm(model, series, targets)` | 창 데이터셋(목표가 `is_imputed=False`인 창만), 모델, 조기 종료 학습(최저 val 손실 가중치 복원), 곡선 기록, 가중치·스케일러를 D-11로 저장·로드 | D-04 → 예측·곡선, D-11 | M-06, M-13 |
| M-10 | `eval/metrics.py: mae, rmse, rel_mae, eval_mask, evaluate(pred_frames, reference_times)` | 채운 시각 제외, 기준 시각 집합과 일치 검사, 지표·시드 평균·표준편차·lag-144 대비 상대 MAE 계산 | D-07 → D-08 | — |
| M-11 | `exp/registry.py: resolve_config, check_rules, run_experiment, sweep` | ID·루트·부모·한 요소 검증, 설정 풀기·차이 계산, 락, 시간 확인, 폴더·메타 쓰기, 모델 분기, sweep | D-05 → 실험 폴더 | M-05, M-07, M-08, M-09, M-10, M-12, M-13 |
| M-12 | `exp/results.py: rebuild_results()` | 결과 표·그림 재생성 | 실험 폴더들 → D-09 | M-10 |
| M-13 | `seed.py: set_seed(seed, threads)` | 3.3의 시드 정책 적용 | — | — |
| M-14 | `exp/testrun.py: run_test(final_cfg, confirm)` | 최종 설정 검증(선택 규칙 포함), 각 실험의 D-11 모델을 불러와(재학습 없음) test 구간 1회 평가, 부트스트랩 신뢰구간, 원자적 결과 쓰기(4.1) | D-06, D-11 → D-10 | M-05, M-07~M-10, M-13, M-17 |
| M-17 | `eval/bootstrap.py: block_bootstrap_diff_ci(err_model, err_ref, day_index, n=2000, seed=0)` | 일별 블록 부트스트랩으로 MAE 차이의 95% 신뢰구간 계산 | 오차 배열 → (차이, 하한, 상한) | M-13 |
| M-15 | `errors.py: TPError`, `config.py: load_phase, 경로·상수` | 오류 형식, 단계 설정 로드·검증, 경로(D-12), `AGG_VERSION`, `IMPUTE_VERSION`, `LSTM_SEEDS` | D-06a → phase_cfg | — |
| M-16 | `cli.py`, `__main__.py` | 스레드 환경 변수 설정(numpy import 전), 인자 파싱과 명령 분기, 종료 코드 | 인자 → I-01~I-05 | 전부 |

의존 방향: config(M-15)·seed(M-13) → data → prep → models → eval → exp → cli (한 방향).

**모델 초기 설정** (루트와 계열 전환 시의 값. 이후에는 실험으로 바꾼다)
- naive: `lag=144`(공식). lag-1은 루트의 자식(`changed: naive.lag`)
- ARIMA: `order=[2,1,2]`, `seasonal=none`, (`seasonal=fourier`일 때) `fourier_k=3`. 주기 144의 SARIMA 직접 적합은 30분 제한 때문에 쓰지 않는다
  - `diff144`: z_t = y_t − y_{t−144}로 직접 차분한 뒤 z에 ARIMA(order)를 적합. 예측은 ŷ_t = y_{t−144} + ẑ_t. train 앞 144칸은 적합에서 뺀다
  - `fourier`: Europe/Rome 기준 하루 안의 슬롯 s(0~143)로 sin(2πks/144), cos(2πks/144)(k=1..`fourier_k`)를 외생 변수로 넣음
  - 수렴 판정: `res.mle_retvals["converged"]`가 False이거나 `ConvergenceWarning`이 나면 E-3001
- LSTM: `window=144`, `hidden=64`, `layers=1`, `dropout=0.0`, `lr=1e-3`, `batch=256`, `max_epochs=50`, `patience=5`, `scaler=standard`, 손실 MSE, Adam, 시드 `LSTM_SEEDS`

### 4.3 오류·예외 목록 (`TPError(code, message)`, 종료 코드 1)
| 번호 | 상황 | 바깥에 보이는 형태 | 사용처 | 처리 |
|---|---|---|---|---|
| E-1001 | 캐시가 없는 날짜의 원본이 `raw_dir`(D-12)에 없음(`.txt`, `.txt.gz` 모두). `raw_dir` 폴더가 없는 경우도 캐시가 없는 날짜가 있을 때만 해당 | `[E-1001] 원본 없음: 2013-11-05, …` | M-01 (I-01) | 중단. 빠진 날짜를 모두 나열 |
| E-1002 | 원본의 열 개수가 8이 아님, 숫자로 읽을 수 없음, square_id가 1~10000 밖, internet이 음수 | `[E-1002] 형식 오류: <파일>:<줄>` | M-01 (I-01) | 중단 |
| E-1003 | 시각이 10분(600000ms) 배수가 아님 | `[E-1003] 시각 오류: <파일> <값>` | M-01 (I-01) | 중단 |
| E-2001 | 단계 설정 오류(날짜 순서, 구간 겹침, dev에 test, 허용 범위 11/01~12/22 밖, K가 1~3이 아님) | `[E-2001] 분할 설정 오류: <내용>` | M-15 (I-01, I-02) | 중단 |
| E-2002 | 구역 시계열에 연속 4칸 이상 결측(계열 맨 앞은 제외. 맨 앞 결측은 계열 시작을 뒤로 미루고 로그) | `[E-2002] 긴 결측: <구역> <시작>~<끝>` | M-04 (I-01) | 중단. 규칙 변경은 변경 통제 |
| E-2003 | 필요한 구역 목록(D-03)이나 구역 시계열(D-04)이 없거나, 3.3 "산출물 유효성"의 판정 항목이 현재 설정과 다름 | `[E-2003] prepare 먼저 실행: --phase <phase>` | M-11 (I-02), M-14 (I-05) | 중단 |
| E-3001 | ARIMA가 수렴하지 않음(4.2 판정 기준) | `[E-3001] ARIMA 수렴 실패: <order>` | M-08 (I-02) | 실험을 실패 상태로 기록 |
| E-3002 | LSTM 손실이 NaN 또는 inf | `[E-3002] 학습 발산: seed=<n> epoch=<e>` | M-09 (I-02) | 실험을 실패 상태로 기록 |
| E-3003 | 확인 시점(적합 뒤, 에폭마다)에 경과 시간이 `runtime.time_limit_min` 초과 | `[E-3003] 시간 초과: <경과>` | M-11 (I-02) | 실험을 실패 상태로 기록 |
| E-4001 | 설정 검증 실패: D-05 규칙 위반, ID가 최대+1이 아님, 루트 규칙 위반, 부모 대비 바뀐 키가 정확히 1개가 아님, `changed`가 실제 차이와 다름, 계열 전환 시 새 계열의 하위 키가 초기값이 아님, LSTM 학습 샘플 수(목표가 train이고 `is_imputed=False`인 창의 수) < `lstm.batch`, full 단계에서 작업 트리가 dirty | `[E-4001] 실험 규칙 위반: <내용>` | M-11 (I-02, I-03) | 폴더를 만들지 않고 중단 |
| E-4002 | 부모 실험이 없거나, 완료 상태가 아니거나, phase가 다름(`changed: phase`로 dev 부모를 쓰는 경우는 예외) | `[E-4002] 부모 오류: <EXP>` | M-11 (I-02, I-03) | 중단 |
| E-4003 | 이미 있는 ID를 `--retry` 없이 실행, 완료된 실험에 `--retry`, 재시도 시 설정 해시가 기존 `meta.json`과 다름 | `[E-4003] 재실행 불가: <EXP> <상태>` | M-11 (I-02) | 중단 |
| E-4004 | 평가 시각 집합이 기준(같은 phase·zone_rank에서 처음 완료된 실험, test에서는 계열 간)과 다름 | `[E-4004] 평가 시각 불일치: <차이 개수>` | M-10 (I-02, I-05) | 중단 |
| E-4005 | `results/test/LOCK`이 이미 있음 | `[E-4005] test는 이미 평가됨: <날짜>` | M-14 (I-05) | 중단 |
| E-4006 | 최종 설정 오류: `--confirm` 없음, 작업 트리 dirty, 계열 키 누락, 키와 실험의 `model.type`·`naive.lag` 불일치, full이 아님, 미완료, `post_test=true`, `zone_rank` 불일치, 구역 수 ≠ K, 선택 규칙(3.3)과 다름, D-11 파일 누락·로드 실패 | `[E-4006] 최종 설정 오류: <내용>` | M-14 (I-05) | 아무것도 쓰지 않고 중단 |
| E-4007 | 다른 `run`·`sweep`·`test`가 락을 잡고 있음 | `[E-4007] 다른 실행이 진행 중` | M-11 (I-02, I-03), M-14 (I-05) | 중단 |

### 4.4 데이터
검증 규칙은 이 절에만 적는다. 인터페이스와 구현은 이 규칙을 따른다.

**D-01 원본 행** (`<raw_dir>/sms-call-internet-mi-YYYY-MM-DD.txt[.gz]`, 탭 구분, 헤더 없음)
| 필드 | 타입 | 검증 |
|---|---|---|
| square_id | int | 1~10000 (E-1002) |
| time_ms | int64 | UTC epoch ms, 600000의 배수(E-1003). 파일 날짜 밖이어도 받아들이되 개수를 진단 보고에 기록 |
| country_code | int | 정수 (E-1002) |
| smsin, smsout, callin, callout | float | 읽기만 하고 사용하지 않음. 빈 값 허용 |
| internet | float | 빈 값 = 0. 음수면 E-1002 |

**D-02 일별 캐시** (`data/interim/daily/YYYY-MM-DD.parquet` + `_meta.json`. 파일 하나당 캐시 하나. 행의 실제 시각은 그대로 둠)
| 필드 | 타입 | 제약 | 보장 위치 |
|---|---|---|---|
| square_id | int16 | 1~10000 | 로직(M-02) |
| time_utc | datetime64[ns, UTC] | 10분 정각 | 로직 |
| internet | float64 | ≥ 0, 국가 코드 합 | 로직 |
| (키) | — | (square_id, time_utc) 유일 | 로직 + 테스트 |
`_meta.json`: `agg_version`, 원본 파일명·크기(기록용), **파일에 나온 시각 목록**(`present_times`), 날짜 밖 행 수.

**결측 판정 규칙:** 이어 붙인 캐시들의 `present_times` 합집합에 없는 시각 → 결측. 시각은 있지만 그 구역의 행만 없으면 → 활동 없음(0). 상위 구역 계열에서 0인 칸의 수를 로그로 남긴다. 이 규칙은 F-01의 진단 보고로 확인하고, 다르면 변경 통제를 거친다.

**D-03 구역 목록** (`data/processed/<phase>/zones.json`): `{phase, k, train_range, zones: [{rank, square_id, train_total}]}`. 동률이면 square_id 오름차순.

**D-04 구역 시계열** (`data/processed/<phase>/series_<square_id>.parquet` + `_meta.json`)
| 필드 | 타입 | 제약 |
|---|---|---|
| time_utc | datetime64[ns, UTC] | 계열 시작(보통 phase 시작, 맨 앞 결측이면 뒤로 밀림)부터 phase 끝까지 10분 간격, 빠짐 없음 |
| y | float64 | ≥ 0, NaN 없음(채운 뒤) |
| is_imputed | bool | 연속 3칸 이하 결측을 앞값으로 채운 칸만 True |
| segment | category | train / val / test (대상 시각 기준, Europe/Rome 날짜) |

**D-05 실험 설정** (`configs/experiments/EXP-###.yaml`. 실행 시 풀어 쓴 판이 실험 폴더의 `config.yaml`)
| 키 | 타입 | 필수 | 검증 (위반 시 E-4001) |
|---|---|---|---|
| id | str | 예 | `EXP-\d{3}`, 기존 최대 번호+1(재시도는 기존 ID) |
| name | str | 예 | `[a-z0-9-]{1,40}` (폴더 이름에 씀) |
| phase | str | 예 | dev \| full |
| parent | str\|null | 예 | phase의 루트만 null(3.3 루트 실험). 그 밖에는 E-4002 조건 |
| changed | str\|null | 예 | parent가 null이면 null, 아니면 실제로 바뀐 유일한 키(점 표기) 또는 `phase` |
| reason, hypothesis | str | 예 | 공백이 아닌 1자 이상 |
| zone_rank | int | 아니오(1) | 1 ≤ 값 ≤ phase의 K |
| model.type | str | 예 | naive \| arima \| lstm |
| naive.lag | int | type=naive | 1 \| 144 |
| arima.order | [p,d,q] | type=arima | p, q는 0~5, d는 0~2 |
| arima.seasonal | str | type=arima | none \| diff144 \| fourier |
| arima.fourier_k | int | seasonal=fourier (기본 3) | 1~10 |
| lstm.window | int | type=lstm | 6~1008 |
| lstm.hidden | int | type=lstm | 8~256 |
| lstm.layers | int | type=lstm | 1~3 |
| lstm.dropout | float | type=lstm | 0~0.5 (layers=1이면 0) |
| lstm.lr | float | type=lstm | 1e-5~1e-1 |
| lstm.batch | int | type=lstm | 32~1024 |
| lstm.max_epochs | int | type=lstm | 1~200 |
| lstm.patience | int | type=lstm | 1~20 |
| lstm.scaler | str | type=lstm | standard \| minmax |
| runtime.time_limit_min | int | 아니오(30) | 1~30 |
| runtime.threads | int | 아니오(4) | 1~8 |
비교 규칙은 3.3 "한 요소" 판정을 따른다.

**D-06 단계·최종 설정**
- D-06a `configs/phases.yaml`: `dev: {train: [2013-11-04, 2013-11-13], val: [2013-11-14, 2013-11-17], k: 1}`, `full: {train: [2013-11-01, 2013-12-08], val: [2013-12-09, 2013-12-15], test: [2013-12-16, 2013-12-22], k: 1}`. 검증은 E-2001 참고.
- D-06 `configs/final.yaml`: `{zones: {1: {naive_144: EXP-###, naive_1: EXP-###, arima: EXP-###, lstm: EXP-###}, 2: {...}, 3: {...}}}`. 키는 zone_rank이고 개수는 full의 K와 같아야 함. 검증은 E-4006. Must(K=1)에서는 키 1만 둠. F-11에서 구역 2·3의 실험은 구역 1의 최종 실험을 부모로 두고 `changed: zone_rank`로 만듦(한 요소 규칙 유지)

**D-07 예측** (`predictions.parquet`): `time_utc, segment(train|val), y_true, y_pred, seed`(LSTM이 아니면 −1), `is_imputed`. test 행은 없음.
**D-08 지표** (`metrics.json`): `{val: {mae, rmse, rel_mae, n}, val_std: {mae, rmse}|null, seeds: {"0": {mae, rmse}, …}|null}`. LSTM의 `val`은 시드별 지표의 평균, `val_std`는 ddof=1 표준편차. naive·ARIMA는 `val_std`와 `seeds`가 null. n은 채점한 시각의 수.
**D-09 결과 표** (`results/results.csv·md`): `exp_id, name, phase, parent, changed, model_type, zone_rank, square_id, val_mae, val_mae_std, val_rmse, val_rmse_std, val_rel_mae, n, compare_group, status, post_test, duration_s`. `compare_group`은 (phase, square_id, 평가 시각 집합, `AGG_VERSION`, `IMPUTE_VERSION`)의 해시 앞 8자리.
**D-10 test** (`results/test/`): `metrics.csv·md`(행: 계열 × zone_rank. 열: square_id, MAE, RMSE, rel_mae, mae_diff_vs_lag144, ci_low, ci_high. LSTM은 D-08과 같은 정의로 평균±표준편차이고, 신뢰구간은 시드 평균 예측 오차로 계산. F-11이면 구역 평균 행 추가(rel_mae 평균)), `predictions.parquet`(`family, zone_rank, square_id, time_utc, y_true, y_pred, seed, is_imputed`), `LOCK`(`{evaluated_at, final: {...}, git_commit}`).
**D-11 모델 산출물** (실험 폴더의 `model/`): naive는 없음. ARIMA는 `arima_params.json`(order, seasonal, fourier_k, 적합된 파라미터 벡터). LSTM은 `lstm_seed{n}.pt`(최저 val 손실 시점의 state_dict)와 `scaler.json`(train 통계). I-05는 이 파일만 불러와 예측하고 재학습하지 않음.
**D-12 경로 설정**: `raw_dir`는 환경 변수 `TP_RAW_DIR`로 정하고, 없으면 `data/raw`. 캐시·처리 데이터 경로(`data/interim`, `data/processed`)는 저장소 기준으로 고정.
**D-13 실험 메타** (`meta.json`): `id, parent, changed, reason, hypothesis, status(running|completed|failed), error_code, post_test, config_hash, git_commit, git_dirty, started_at, ended_at, duration_s, pid`.

### 4.5 기술 스택·폴더 구조·명령·실행 자산
**스택** (Python 3.12. uv로 설치하고 `uv.lock`에 정확한 버전을 고정. 3.14가 설치되어 있지만 torch·statsmodels 휠 호환성 때문에 3.12를 씀)
pandas ≥2.2, numpy ≥1.26, pyarrow ≥15, statsmodels ≥0.14, torch ≥2.3 (CPU 휠), matplotlib ≥3.8, pyyaml ≥6, filelock ≥3.13, pytest ≥8, ruff ≥0.6

**폴더**
```
traffic-predict/
├─ pyproject.toml, uv.lock, README.md, CLAUDE.md, .gitignore
├─ docs/traffic-predict_기획서.md
├─ configs/ phases.yaml, final.yaml, experiments/EXP-###.yaml
├─ src/tp/ __init__.py, __main__.py, cli.py, config.py, errors.py, seed.py
│   ├─ data/ raw.py, cache.py, zones.py
│   ├─ prep/ series.py, split.py, scale.py
│   ├─ models/ naive.py, arima.py, lstm.py
│   ├─ eval/ metrics.py
│   └─ exp/ registry.py, results.py, testrun.py
├─ tests/ fixtures/, test_*.py
├─ data/        (git 제외) raw/, interim/daily/, processed/<phase>/
├─ experiments/ (커밋) EXP-###_<slug>/
└─ results/     (커밋) results.csv·md, figures/, test/
```

**명령**
```bash
uv sync
uv run python -m tp prepare --phase dev
uv run python -m tp run --config configs/experiments/EXP-001.yaml
uv run python -m tp results
uv run pytest -q
uv run ruff check src tests
uv run ruff format --check src tests
```

**실행 자산**
- `tests/fixtures/make_fixtures.py`(시드 0): 가짜 원본 3일치(구역 4개, 국가 코드 2개, 결측 1칸짜리·4칸짜리 사례 포함)를 만들고, 손으로 계산한 기대 캐시값과 기대 지표를 `golden.json`에 둔다
- 고정 시드: 3.3 참고
- 기준선 결과: dev 단계의 EXP-001(lag-144), EXP-002(lag-1)가 이후 모든 dev 실험의 비교 기준

## 5. 설계 결정 기록
| 결정 | 대안 | 이유 |
|---|---|---|
| lag-1 참고 기준선 추가 | lag-144만 | 10분 뒤 예측에서 가장 강한 단순 기준선. 모델 우위가 진짜인지 확인 |
| 1단계는 val만, test는 전체 기간에서 1회 | 단계마다 test | 데이터를 늘리면 "마지막 구간"이 바뀌어 test 고정·1회 원칙이 깨짐 |
| test = 12/16~12/22 | 12/25~12/31, 12/16~12/31 | 연휴의 분포 변화를 피하고 평소 예측력을 잼 |
| 상위 구역은 train 총량으로 선택 | 전체 기간 총량 | test 정보 누수 방지 |
| Must 구역 1개, Should 3개 | 처음부터 3개, 1개만 | 파이프라인을 작게 완성한 뒤 넓힘 |
| LSTM 시드 3개, 평균±표준편차 | 시드 1개 | 한 요소 변경의 효과를 시드 노이즈와 구분 |
| 짧은 결측(3칸 이하)만 앞값으로 채움, 채운 시각은 채점 제외 | 모두 보간 후 채점 포함 / 선형 보간 | 만든 값으로 채점하지 않음. 선형 보간은 뒤의 실제 값을 입력에 섞어 누수를 만듦(레드팀 #1) |
| 실험당 30분(CPU) 상한 | 2시간 | 실험 반복 속도. 계절성 처리 방식의 제약이 됨 |
| test는 잠금 시점의 K로 1회 | 구역별 잠금 | 첫 test 결과를 본 뒤 설정을 고를 여지를 없앰 |
| 잠금 후 실험은 허용하되 `post_test` 표시 | 거부 | 사후 분석은 하되 test 결론과 분리 |
| 실험 기록은 평문 파일 | MLflow | GitHub에서 바로 보이고 의존성이 적음 |
| 원본은 수동 다운로드, F-01이 파일 확인 | 다운로드 스크립트 | 범위 최소화, 외부 연동 위험 제거 |
| 레드팀 1회(오류 37, 개선 15, 확장 2): 오류 모두 반영, 확장 2건은 한 줄 규칙으로 처리 | — | 5단계 절차 |
| val 편향은 명시만 | holdout 분리 | 공정 비교는 test 1회로 보장. 분할 단순성 유지 |
| 상대 MAE, 일별 블록 부트스트랩 CI 추가 | 원래 단위 MAE·RMSE만 | 구역 간 평균과 test 1주의 불확실성 표시 |
| `test --confirm`, full·test는 dirty 거부, LSTM 샘플 수 검사 | 정책으로만 둠 | 코드로 강제 |
| 모든 모델 train으로만 학습 | test 전 train+val 재학습 | 계열 간 조건 동일, 저장된 모델(D-11) 재사용 |
| 원본 경로는 `TP_RAW_DIR`, `.txt`/`.txt.gz` 모두 읽음 | 보관 방식 하나로 고정 | 디스크 8GB. 사용자가 공간을 마련 중이라 방식이 정해지지 않음 |
| test는 저장된 모델(D-11)을 불러와 평가 | test 때 재학습 | 재학습 비용(최대 30분 × 계열)과 재현 차이 위험 제거 |
| 주기 144 SARIMA는 쓰지 않고 `diff144`/`fourier`로 계절성 처리 | SARIMA(s=144) | 30분 제한 |
| Python 3.12 (uv) | 설치된 3.14 | torch·statsmodels 휠 호환성 |
| 결측 판정: 파일 전체에서 빠진 시각 = 결측, 구역 행만 없음 = 0 | 구역 행이 없으면 모두 결측 | 드문 구역은 활동이 없을 때 행이 없음. F-01에서 확인 |

## 동결 체크리스트
- [x] 핵심 흐름이 Must 기능만으로 끝까지 이어진다 (F-01 → F-02 → F-03 → F-05~F-07 + F-08 → F-04 → F-09)
- [x] 모든 Must·Should에 검증 가능한 완료 조건이 있다
- [x] 제외 목록에 이유가 있다
- [x] 3.1·3.2·3.3에 빈칸이 없다
- [x] 스크립트 인자 = 설정 키 = 코드 변수 (불일치 0: `--phase`↔`phases.yaml`↔`load_phase`, `--key`↔D-05 점 표기 키, `TP_RAW_DIR`↔D-12)
- [x] 인터페이스가 부르는 구현 = 구현 표 (M-01~M-17 모두 정의·사용)
- [x] 모든 오류 번호에 사용처가 있고, 모든 예외에 오류 번호가 있다 (E-1001~E-4007, 16개)
- [x] 모든 입력 값에 검증 규칙이 한 곳에 적혀 있다 (4.4 D-01, D-05, D-06)
- [x] 레드팀 1회 완료, 오류 항목 반영 (오류 37건 반영, 개선 7건 채택, 확장 2건 한 줄 규칙)
- [x] (평가 과제) 해당 없음
- [x] 기술 스택, 폴더 구조, 실행·테스트·린트 명령, 실행 자산이 정해졌다
- [x] 구현 진행표에 순서가 채워져 있다
- [x] CLAUDE.md에 프로필과 경계 3단(항상/물어보고/절대)이 있다

## 구현 진행표
| 순서 | 기능 ID | 범위 | 상태 | 비고 |
|---|---|---|---|---|
| 1 | 골격 | `pyproject.toml`·`uv.lock`(Python 3.12), M-15(config·errors·`load_phase`), M-13(seed), M-16(CLI 뼈대, 스레드 환경 변수), `tests/fixtures/make_fixtures.py`·`golden.json`, `.gitignore`, ruff·pytest 동작 | 완료 | E-2001 테스트 포함. pytest 25 통과, ruff 통과 (2026-09-23) |
| 2 | F-01 | M-01, M-02, 진단 보고 / I-01의 캐시 부분 / 테스트(픽스처 합산값, `.gz`, E-1001~1003) → **dev 원본(11/03~11/18)으로 전제 A1~A4 확인** | 완료 | 코드·테스트 완료. 실제 11-01 원본으로 확인(2026-09-23): A1 8열·검증 통과(484만 행, 읽기 2.7초) / A2 국가 코드별 여러 행 → 합산 필요 확인 / A3 정규화 값(최대 약 5000) / **A4 파일은 현지(Rome) 날짜 기준으로 나뉨**(현지 날짜 밖 0행, UTC 날짜 밖 15.6만 행) / 11-01은 144칸 모두 있음, 한 시각에 행이 없는 셀은 최대 4개. 11-01 1위 구역 5161 |
| 3 | F-02 | M-03 / I-01 구역 부분 / 테스트(train만 사용, 동률) | 완료 | pytest 54 통과. 인접 날짜 파일에 섞인 train 행도 현지 날짜 기준으로 집계 |
| 4 | F-03 | M-04, M-05 / I-01 완성 / 테스트(앞값 채우기, E-2002, 경계, 144칸, `cut_until`) | 완료 | pytest 69 통과. `load_series`(E-2003)도 여기서 구현. 테스트용 dev 단계는 train 11-04 / val 11-05(11-06의 4칸 결측은 E-2002 테스트에서 따로 사용) |
| 5 | F-04 | M-10 / 테스트(손계산 예제 1e-9, 채운 시각 제외, E-4004, rel_mae) | 완료 | pytest 78 통과 |
| 6 | F-08 + F-05 | M-11(규칙·락·메타·sweep), M-12, M-07 / I-02, I-03, I-04 / 테스트(루트·한 요소·계열 전환·ID·retry·dirty) → dev EXP-001(lag-144), EXP-002(lag-1) | 진행 중 | 코드·테스트 완료(pytest 117 통과). **남은 것: 실제 dev 데이터로 EXP-001(lag-144)·EXP-002(lag-1) 실행**. 구현 세부: dirty 판정은 코드 경로(`src`, `pyproject.toml`, `uv.lock`, `configs/phases.yaml`)만 봄 / `meta.json`에 `square_id·agg_version·impute_version·compare_group` 추가(실행 당시 버전으로 compare_group 계산) / `results/figures/*.png`는 F-10에서 구현 |
| 7 | F-06 | M-08(none·diff144·fourier, 수렴 판정, D-11) / 테스트(재현 1e-9, apply가 재적합 안 함) → dev ARIMA 첫 실험 | 완료 | pytest 131 통과. full 길이 합성 계열(train 38일+val 7일) 적합 시간: (2,1,2) none 1.3초 / diff144 0.9초 / fourier 2.4초, (5,1,5)+fourier K=10 23.9초 → 30분 제한 위험 없음. 실데이터 dev 실험은 원본 도착 후 |
| 8 | F-07 | M-06, M-09(D-11, 곡선, 샘플 수 검사) / 테스트(같은 시드 1e-6, 스케일러는 train만) → dev LSTM 첫 실험 | 완료 | pytest 145 통과. 같은 시드 재실행 1e-6 이내 동일(Windows CPU, deterministic). full 길이 합성 계열에서 초기 설정 시드 3개 학습 145초. 학습 곡선은 로그 y축, 팔레트 1·2번(train 파랑/val 주황), 최저 val 에폭 표시 |
| 9 | F-09 | M-17, M-14 / I-05 / 테스트(픽스처로만: LOCK·원자성·confirm·dirty·선택 규칙·부트스트랩 재현) | 대기 | 실제 test는 순서 11에서 |
| 10 | 운영 | 원본 11/01~12/22 확보 → `prepare --phase full` → full 실험(`changed: phase`로 옮기기) | 대기 | 디스크 공간 필요 |
| 11 | F-11 | K=3 `prepare`, 구역 2·3 자식 실험 | 대기 | Should. 실제 test 전에 끝낼지 결정 |
| 12 | 운영 | 실제 `test --confirm` 1회 | 대기 | **사용자 승인 필수** |
| 13 | F-10 | README 결과·한계 정리 | 대기 | Should |