# 밀라노 모바일 트래픽 예측 기획서

상태: 동결 (2026-09-23, v2.5)
프로필: 데이터·ML 실험

## 다음 세션에게
- 현재 단계: 진행표 1~11, 14~17 완료(2026-09-23). 목적 단계 2(예측 거리)·3(결정 지표)을 val로 끝냄. 다음은 목적 단계 4(켜기/끄기 시뮬레이션) 변경 통제 → F-09 개정 변경 통제 → 12(test, "설정 동결, test 실행" 지시 후) → 13(README)
- 용어: 이 문서에서 "N단계"는 목적 진행 순서(1 목적 확정 ~ 5 test)를 뜻한다. 데이터 기간은 "단계(phase)" dev/full로 부른다
- 최종 목적(v2): 예측을 **무선망 운용 결정**(혼잡 예방용 용량 셀 켜기)에 쓴다. 순서: 1) 목적 확정(v2) 2) 예측 거리 확장 비교(10/30/60분) 3) 결정 수준 평가 지표 4) 규칙 기반 켜기/끄기 시뮬레이션. **test는 1~4의 설정을 val로 모두 확정한 뒤 마지막에 한 번만**
- 확정된 결정:
  - 프로필: 데이터·ML 실험 / 개인 프로젝트, 1인, 직접 구현
  - 결과 소비자: GitHub 포트폴리오(README). 정해진 마감 없음
  - 기준선: lag-144(전날 같은 시각)가 공식 기준선, lag-1(직전 값)은 참고 기준선으로 함께 보고
  - dev 단계(phase, 1~2주치)는 train/val만 사용. test는 전체 기간으로 확장한 뒤 1회만 평가
  - 공식 test 구간: 2013-12-16 00:00 ~ 2013-12-22 23:50 (CET). 12/23 이후 데이터는 사용하지 않음
  - 비교 순서: seasonal naive → ARIMA → LSTM. 한 실험에서는 한 요소만 바꾸고 이유·가설을 기록
  - 지표: MAE, RMSE (원래 단위). 실험마다 시드 재고정, 결과 표·학습 곡선을 실험별로 저장
  - 구역: Must는 상위 1개, Should는 상위 3개(구역별로 따로 학습)
  - LSTM은 실험마다 시드 3개(0, 1, 2)로 돌려 평균±표준편차를 보고
  - 결측: 3칸(30분) 이하는 직전 관측값으로 채워(앞값 채우기, 레드팀 #1로 선형 보간에서 변경) 입력으로만 사용하고, 채운 시각은 모든 모델의 채점·학습 손실에서 제외. 더 긴 결측은 오류로 중단
  - 실행 시간 상한: 실험당 30분(CPU, 전체 기간, 구역 1개, 시드 모두 포함)
  - 평가 방식: 롤링 예측(거리 h: 시점 t − h까지의 관측으로 t를 예측, v1은 h=1). 평가 구간에서는 재적합하지 않고 실제 관측만 반영
  - test는 잠금 시점의 K로 1회만. 잠금 후 실험은 허용하되 `post_test` 표시
  - 실험 기록은 평문 파일(YAML/JSON/CSV). 원본은 수동 다운로드
  - 원본 보관: 사용자가 디스크 공간을 마련 중. 코드는 `TP_RAW_DIR` + `.txt`/`.txt.gz`를 모두 읽어 어떤 보관 방식이든 대응
  - (v2) 예측 거리 h ∈ {1, 3, 6}스텝(10/30/60분). 기준선에 지난주 같은 시각(lag-1008) 추가. 거리 확장 실험은 각 계열의 h=1 최선을 부모로 `changed: horizon`
  - (v2) 혼잡 임계치 = 0.7 × 구역별 train 99번째 백분위수(최댓값은 튀는 값에 민감해 기각). 혼잡 구간은 1칸 끊김을 이어 붙임. 탐지는 혼잡 시작 전에 발행된 경보만 인정. 결정 지표는 60분 뒤 예측을 기준으로 보고. (v2.1) 탐지는 발행 시각 < 혼잡 시작일 때만, 리드타임 10분~거리(60분 뒤 예측이면 10~60분)
  - (v2) 과소예측(혼잡을 놓침)이 과대예측(괜히 켬)보다 비용이 큼. 최소 유지 시간 값은 4단계에서 결정
- 남은 일: 목적 단계 4(최소 유지 시간 등 결정 값은 후보와 근거를 제시해 사용자가 정함) → F-09 개정(아래 F-09 참고) → 12 → 13.
- 제약: 마감 없음 · 채점 기준 없음 · 제출물 = GitHub 저장소 README · 구현함
- 최근 변경: v2.5(2026-09-23) test 차단 규칙의 범위를 실제 동작에 맞게 명확화(사용자 결정, 코드 변경 없음): test 값으로 예측·채점하는 것은 `test`뿐이고, test 값은 모델 입력·채점·선택·결정 지표·그래프에 들어가지 않음. 파이프라인이 계열을 만들거나 읽을 때 test 행을 메모리에 올렸다 잘라내는 것은 허용하되 그 값을 출력·요약하지 않음. v2.4(2026-09-23) 문구 정리(동작·결정 변화 없음): 문서 사이 불일치 검토 결과 반영. 구현 단위 시그니처·오류 사용처·파일 필드를 실제 코드에 맞춤, full 실험은 사용자 승인 후 실행으로 3.1을 CLAUDE.md와 맞춤, v1 표현(1스텝·보간) 정리, "단계" 용어 구분. v2.3(2026-09-23) 결정 6건 확정: (a) 기준선 이름 '마지막 관측값(10분 뒤 예측에서는 10분 전 값)' (b) 경보 = 예측 ≥ 임계치 유지 (c) 혼잡 시작 직전 h칸 안의 경보는 불필요 경보에서 제외 (d) LSTM 결정 지표는 시드별 계산 후 평균±표준편차 (e) 30·60분은 10분 뒤 예측에서 고른 하이퍼파라미터(차수·창 등) 재사용 (f) 채운 칸은 경보·혼잡에서 제외. v2.2(2026-09-23) 문구 수정: '셀 단위' → '격자 칸(약 235m) 단위, 지역별 수요 예측'(데이터 해석 주의 반영, 설계 영향 없음). v2.1(2026-09-23) 탐지 조건을 '발행 시각 < 혼잡 시작'으로 엄격화, 리드타임 10~60분. v2(2026-09-23) 목적 추가(무선망 운용 결정), F-12 예측 거리·F-13 결정 지표 추가, 제외 목록에서 '여러 스텝 앞 예측'·'lag-1008' 되살림. test(F-09) 형식은 4단계 변경 통제에서 개정 예정
- 마지막 실험: v2 full 거리 확장 EXP-026~058. v2.3 정의로 결정 지표 재계산(재학습 없음). 60분 뒤 LSTM이 세 구역 모두 MAE 최저(173.3±8.5 / 146.7±0.7 / 154.8±10.3). pytest 211 통과
- 구현 환경(2026-09-23 확인): 커밋은 기능 완료마다 자동(`feat(F-xx): …`, push 안 함). 원본은 기본 경로 `data/raw`(11/01~12/22 확보 완료). CPU만 사용. 디스크 여유 52GB. GitHub 비공개 저장소 `origin`, push는 사용자가 요청할 때만

## 1. 개요
- **문제:** 격자 칸(약 235m, CellID는 무선 셀이 아님) 단위 모바일 인터넷 트래픽의 단기(10·30·60분 뒤) 지역별 수요 예측. 값은 스케일된 활동량이며 절대량으로 해석하지 않음
- **최종 목적(v2):** 예측을 무선망 운용 결정에 쓴다. 상위 3개 구역은 **혼잡 예방** 시나리오: 구역마다 항상 켜진 커버리지 셀과 켜고 끌 수 있는 용량 셀이 있다고 가정하고, 예측 트래픽이 임계치를 넘을 것 같으면 용량 셀을 미리 켠다. 과소예측(혼잡을 놓침)이 과대예측(괜히 켬)보다 비용이 크다. 셀 용량은 절대 단위를 모르므로 구역별 train 피크 대비 비율로 정의한다
- **대상 사용자:** 본인(학습)과 포트폴리오를 보는 사람(채용 담당자·엔지니어). 결과 표·그래프·실험 기록으로 방법론의 엄밀함을 보여 주는 것이 목적
- **한 문장 요약:** 밀라노 격자 중 인터넷 트래픽 총량 상위 구역에서 10·30·60분 뒤 인터넷 활동량을 예측하고, naive 기준선 → ARIMA → LSTM을 같은 test 구간·같은 전처리 조건에서 MAE·RMSE와 결정 수준 지표(놓친 혼잡, 불필요 경보 시간, 리드타임)로 비교한다.

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
- 상위 구역의 트래픽이 하루 주기로 충분히 규칙적이어서 짧은 거리(v1 10분, v2 10·30·60분) 예측의 모델 간 차이가 의미 있게 드러난다.
- 원본의 10분 구간 결측이 적어 하나의 채우기 규칙(앞값 채우기)으로 처리할 수 있다.
- 1~2주치로 만든 파이프라인이 전체 기간에서도 설정 변경만으로 돌아간다(일별 사전 집계 캐시로 메모리를 제한).
- (v2) 셀 용량을 "구역별 train 99번째 백분위수 × 비율"로 대신할 수 있다(절대 용량을 모름).
- (v2) val 1주에 구역마다 비교할 만큼의 혼잡 구간이 있다(2026-09-23 확인: 0.7 × P99 기준 연속 구간(병합 전) 5161 13회, 5059 13회, 5259 5회. 1칸 끊김 병합 후 val 결과 표 기준 12·8·5회).

### 위험과 대응
| 위험 | 영향 | 대응 |
|---|---|---|
| 기준선이 약하면 모델 우위가 부풀려진다 | 결론이 틀림 | lag-1 참고 기준선을 함께 보고 |
| 연말 연휴의 분포 변화 | test가 평소 예측력을 반영하지 못함 | test를 12/16~12/22로 고정하고 12/23 이후는 제외 |
| test를 여러 번 보게 됨 | 누수·과대평가 | test는 1회만 평가, 잠금 파일로 강제 |
| SARIMA를 주기 144로 적합하면 매우 느림 | 실험이 멈춤 | 상세 설계(4.2)에서 계절성 처리 방식을 결정(계절 차분 또는 푸리에 항) |
| LSTM 결과가 시드에 따라 흔들림 | 한 요소 비교의 신뢰도가 떨어짐 | 시드 3개의 평균±표준편차로 보고 |
| 원본 용량: 61개 파일 20.5GB(하루 약 280~376MB). 사용 구간 11/01~12/22만 약 17GB인데, 디스크 여유는 당시 8GB(2026-09-23 기획 시점. 이후 52GB 확보) | 전체 기간 원본을 한 번에 둘 수 없음. 메모리(16GB)도 파일 하나씩 읽어야 함 | 파일 단위로 읽어 일별 집계 캐시(parquet)에 저장. 원본은 `TP_RAW_DIR` + `.gz` 지원으로 어떤 보관 방식이든 대응(D-12) |
| train에 공휴일(11/01, 12/07, 12/08)이 있고, test는 성탄 직전 쇼핑 주간 | 평소 패턴과 다른 날이 섞임 | README 한계 절에 명시(F-10) |
| (v2) 혼잡 임계치가 튀는 값에 민감함(5161 train 최댓값 8,044는 2위보다 15% 높음 → 0.7×최댓값이면 val 혼잡 2회) | 결정 지표 비교 불가 | 피크를 train 99번째 백분위수로 정의 |
| (v2) 거리 확장 실험은 h=1에서 고른 하이퍼파라미터를 그대로 씀 | 30·60분 뒤 성능이 과소평가될 수 있음 | 한계로 기록. 필요하면 거리별로 한 요소씩 추가 실험 |
| (v2) val 1주의 혼잡 구간 수가 5~12회(병합 후)로 적음 | 결정 지표 차이가 우연일 수 있음 | 횟수와 함께 보고, test 1회로 최종 판단 |
| Dataverse 목록은 61개인데 기간은 62일 | 하루치가 빠져 있을 수 있음 | F-01에서 사용 구간의 파일이 모두 있는지 확인. 빠진 날이 사용 구간 안이면 변경 통제 |

## 2. 범위
- **핵심 흐름:** 원본 일별 파일 → 구역·시각별 집계 캐시 → 상위 구역 선택(train 구간) → 전처리·분할 → 모델 학습(거리 h별, val로 선택) → 평가(MAE·RMSE + 결정 지표) → 결과 표·거리별 그래프·학습 곡선 → (3·4단계 뒤, 전체 기간에서) test 1회 평가

### 분할
| 단계 | train | val | test |
|---|---|---|---|
| dev (개발, 2주) | 11/04 ~ 11/13 (월~수, 10일) | 11/14 ~ 11/17 (목~일, 4일) | 없음 |
| full (전체 기간) | 11/01 ~ 12/08 | 12/09 ~ 12/15 (월~일) | 12/16 ~ 12/22 (월~일) |

### Must
| ID | 이름 | 설명 | 완료 조건 | 예외 |
|---|---|---|---|---|
| F-01 | 원본 로드·집계 | 일별 원본을 읽어 국가 코드를 합치고, (square_id, time) → internet 값을 일별 parquet 캐시로 저장 | (1) 캐시 스키마가 정의와 일치 (2) 무작위 3개 (구역, 시각)의 값이 원본에서 직접 합산한 값과 일치 (3) 캐시가 있으면 원본을 다시 읽지 않음 (4) 전제 A1~A4가 실제 파일과 맞는지 확인 결과를 로그로 남김 | E-1001, E-1002, E-1003 |
| F-02 | 상위 구역 선택 | train 구간의 internet 총량으로 구역을 정렬해 상위 K개를 저장(Must에서는 K=1) | 같은 입력이면 같은 목록이 나옴(동률이면 square_id가 작은 쪽). val·test 구간 데이터는 계산에 쓰이지 않음(테스트로 검증) | E-1001(train 구간 원본 없음) |
| F-03 | 전처리·분할 | Europe/Rome 기준 날짜 판정, 10분 정규 인덱스, 결측 처리, 고정 날짜로 분할 | (1) 구간 경계가 설정 날짜와 정확히 같고 겹침 0 (2) `run`에 넘기는 계열에 test 구간이 없음 (3) 연속 3칸 이하의 결측은 직전 관측값으로 채우고(앞값 채우기) `is_imputed=True`로 표시 (4) 연속 4칸 이상의 결측은 오류로 중단 (5) 사용 구간의 모든 날이 144칸 | E-2001, E-2002 |
| F-04 | 평가 | 원래 단위로 MAE·RMSE를 계산. 모든 모델을 같은 평가 시각 집합(채운 시각 제외)에서 비교 | (1) 손으로 계산한 예제와 1e-9 이내로 일치 (2) 채운(`is_imputed`) 목표 시각은 지표에서 빠짐 (3) 모델 간 예측 시각 집합이 다르면 오류 | E-4004 |
| F-05 | 기준선 | lag-144(공식), lag-1(참고), (v2) lag-1008(지난주 같은 시각) 예측. 거리 h에서는 ŷ_t = y[t − max(lag, h)] | 예측값이 y[t − max(lag, h)]와 정확히 같음 | E-4004(평가 시각에 필요한 과거 기록이 계열에 없을 때. 예측 파일에는 train 행도 있으나 채점은 val·test 시각만 하므로, full에서는 실제로 나지 않음) |
| F-06 | ARIMA | train에서 적합, val로 차수 선택. 평가 구간에서는 재적합 없이 실제 관측을 반영하며 롤링 예측(v1은 1스텝, (v2) 거리 h는 F-12) | (1) 지표·예측 파일 저장 (2) 같은 설정으로 다시 실행하면 지표가 동일 (3) 전체 기간·구역 1개 기준으로 30분 이내 | E-3001, E-3003 |
| F-07 | LSTM | 입력 창 → 다음 값 예측(v1 1스텝, (v2) 거리 h는 F-12). val 손실로 조기 종료, 시드 3개(0, 1, 2) 각각 학습 | (1) 시드마다 학습 곡선(train/val 손실) png·csv 저장. 스케일러 통계가 train에서만 계산됨 (2) 평균±표준편차 지표 저장 (3) 같은 시드로 다시 실행하면 지표가 1e-6 이내로 동일 (4) 시드 3개 합쳐 30분 이내 | E-3002, E-3003 |
| F-08 | 실험 기록·재현 | 실험마다 설정·부모 실험·바꾼 요소·이유·가설·시드·결과를 저장하고, 전체 결과 표를 자동 갱신 | (1) 실험 폴더에 설정·지표·예측·곡선이 있음 (2) 부모 대비 바뀐 설정 키가 정확히 1개가 아니면 실행 거부 (3) 결과 표 한 곳에 모든 실험이 나열됨 | E-4001, E-4002, E-4003 |
| F-09 | test 1회 평가 | 전체 기간에서 계열별 최종 설정(lag-144, lag-1, ARIMA 최선, LSTM 최선)을 한 번에 test로 평가하고 잠금. (v2) 거리·결정 지표·시뮬레이션을 포함하는 형식은 4단계 변경 통제에서 개정하며, 그 전에는 실행하지 않음 | (1) 두 번째 실행은 오류로 거부됨 (2) `--confirm`이 없거나 dirty면 거부됨 (3) test 결과 표에 MAE·RMSE·상대 MAE·lag-144 대비 차이의 95% 신뢰구간이 있음 (4) 부트스트랩이 같은 시드에서 같은 구간을 냄 (5) (v2) 선택 규칙의 후보는 같은 horizon의 실험만. 현재 코드는 개정 전까지 horizon=1만 받음(그 밖은 E-4006). **개정 때 정할 것:** 평가할 거리, `final.yaml`의 lag-1008 키, LSTM 신뢰구간 방식(지금은 시드 평균 예측 오차 → v2.3 결정 지표처럼 시드별로 할지) | E-4005, E-4006 |
| F-12 | (v2) 예측 거리 확장 | 실험 설정 `horizon` ∈ {1, 3, 6}(10/30/60분). 시점 o = t − h까지의 관측만으로 y[t]를 예측. naive는 ŷ_t = y[t − max(lag, h)], ARIMA는 train 적합 파라미터로 h스텝 앞 예측(재적합 없음), LSTM은 입력 창이 t − h에서 끝나고 거리마다 따로 학습(direct) | (1) 기존 실험은 재실행 없이 horizon=1로 간주 (2) naive 예측이 y[t − max(lag, h)]와 정확히 같음 (3) ARIMA h스텝 예측이 원점별 statsmodels `apply`+`forecast(h)` 결과와 1e-6 이내 일치, 재적합 없음 (4) LSTM·ARIMA 모두 y[t − h + 1] 이후 값을 바꿔도 ŷ_t가 그대로(인과성) (5) 비교·선택은 같은 horizon끼리 (6) `results`가 거리별 표 `results/horizon.md`와 그래프 `results/figures/horizon_mae.png` 생성 (7) 실험당 30분 제한 유지 | E-4001, E-4004 |
| F-13 | (v2) 결정 수준 평가 | 구역별 혼잡 임계치 θ = `threshold_ratio` × train `peak_quantile` 백분위수(기본 0.7 × P99). 저장된 예측에서 놓친 혼잡 횟수, 불필요 경보 시간, 평균 리드타임을 계산(정의는 3.3) | (1) 손으로 계산한 예제와 일치(구간 병합, 탐지·놓침, 리드타임, 불필요 경보 시간, (v2.1) 시작 시각에 발행된 경보는 놓침) (2) 결과 표에 모델·구역·거리별 결정 지표가 있음 (3) `configs/decision.yaml`을 바꾸고 `results`만 다시 돌리면 재학습 없이 반영됨 (4) (v2.3) LSTM은 시드별로 계산한 뒤 평균±표준편차(ddof=1) (5) 채운 칸은 경보·혼잡 판정에서 제외 (6) (v2.3) 혼잡 시작 직전 h칸 안의 경보는 불필요 경보로 세지 않음 | E-2001 |

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
| Transformer 등 다른 신경망 | 비교 순서는 naive → ARIMA → LSTM으로 확정 |
| 자동 하이퍼파라미터 탐색 | "한 실험에 한 요소" 원칙과 충돌 |
| GPU 학습 | 결정적 재현을 우선함 |
| 대시보드·웹 시각화 | 결과 소비처는 README |
| 원본 자동 다운로드 스크립트 | 수동 다운로드로 충분. 받는 방법은 README에 적음 |
| 조기 종료용 holdout 분리 | 편향을 명시하는 것으로 처리. 분할 규칙이 복잡해짐(레드팀 #3) |
| 결과 표의 모델별 입력 정보 열 | `config.yaml`에서 확인할 수 있음 |
| sweep 그룹 크기 열 | `changed`와 부모로 추적할 수 있음 |
| `MKL_CBWR=COMPATIBLE` 고정 | 같은 머신 재현이 목표. 다른 머신은 상대 1e-3 허용 |
| (v2) 절전 시나리오(한산한 지역 1~2곳 추가, 용량 셀 끄기) | 나중 단계. 현재 범위는 상위 3개 구역의 혼잡 예방 |
| (v2) 최소 유지 시간 규칙·켜기/끄기 시뮬레이션·비대칭 비용 반영 | 3·4단계에서 별도 변경 통제로 다룸 |
| (v2) 90분 이상 거리, 거리별 하이퍼파라미터 재탐색 | 목표 거리는 10/30/60분. 재탐색은 필요할 때 한 요소 실험으로 |

## 3. 뼈대

### 3.1 상호작용 주체와 권한
| 주체 | 하는 일 | 할 수 있는 것 | 할 수 없는 것 |
|---|---|---|---|
| 실행자(본인) | CLI로 파이프라인·실험 실행 | 모든 단계 실행, 실험 생성, test 실행 1회 | 잠금 파일을 지워 test 재평가, 완료된 실험 폴더 수정 |
| 구현 에이전트(Claude) | 코드 작성·테스트, 실험 실행 | 코드·테스트 수정, dev 단계 실험 실행, full 단계의 val 실험 실행(사용자 승인 후, CLAUDE.md "물어보고") | 사용자 승인 없이 test 실행, 원본 데이터 수정, 잠금 해제 |
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

각 산출물에는 3.3 "산출물 유효성"의 판정 항목(코드 버전, 날짜 범위 등. S1은 원본 파일명·크기도 기록용으로)을 담은 메타(`_meta.json`)를 함께 저장한다. 메타가 현재 설정과 다르면 다시 만든다.

**단계(phase)**: `dev`(11/04~11/17, test 없음) → `full`(11/01~12/22). dev 실험 결과는 full과 비교하지 않고, 최종 설정은 full 단계의 val에서 고른다(3.3 선택 규칙). dev에서 찾은 설정은 `changed: phase`로 full에 옮긴다.

**실험 상태**
| 상태 | 전이 | 주체 | 불허 전이 |
|---|---|---|---|
| 생성 → 실행 중 | `run` 명령 | 실행자·에이전트 | 설정 검증(E-4001, E-4002) 실패 시 생성 거부 |
| 실행 중 → 완료 | 지표·산출물 저장 성공 | 코드 | — |
| 실행 중 → 실패 | 예외·시간 초과 | 코드 | — |
| 실패 → 실행 중 | `run --retry` (같은 ID) | 실행자·에이전트 | 완료 → 실행 중 (완료된 실험은 불변. 바꾸려면 새 ID) |
| 실행 중(중단됨) → 실행 중 | `run --retry`. 프로세스가 죽어 `meta.json`의 상태가 `running`으로 남은 경우(락을 잡을 수 있으면 중단된 것으로 봄, 3.3 동시 실행) | 실행자·에이전트 | 재시도 시 설정 해시가 다르면 불가(E-4003). (v2) 해시에 `horizon`이 들어가므로 v1에서 실패한 실험(EXP-015)은 재시도할 수 없음. 새 ID로 만든다 |

**test 상태**: 미평가 → 평가됨(LOCK 생성). 평가됨 → 미평가는 불가.
- test는 잠금 시점의 K(구역 수)로 한 번만 평가한다. 그때까지 F-11이 끝나지 않았으면 test는 구역 1개로 진행하고 F-11은 제외 목록으로 옮긴다.
- 잠금 뒤에도 full 단계의 val 실험은 허용한다. 대신 `post_test=true`로 기록해 결과 표에 표시하고, test에는 쓰지 않는다.

### 3.3 공통 정책
| 항목 | 결정 | 이유 |
|---|---|---|
| 난수 시드 | 실험 시작 때와 시드별 학습 시작 때 `set_seed(seed, threads)`로 `random`·`numpy`·`torch`를 다시 고정. `torch.use_deterministic_algorithms(True)`, `torch.set_num_threads(threads)`, 미니배치 셔플(`randperm`)용 generator에도 시드 지정. `OMP_NUM_THREADS`·`MKL_NUM_THREADS`는 CLI 프로세스 시작 시 numpy import 전에 설정(M-16): `run --config`는 그 설정의 `runtime.threads`, 그 밖의 명령(`sweep` 포함)은 기본 4. 부트스트랩(M-17)은 `numpy.random.default_rng(seed)`를 따로 씀 | 요구: 실험마다 시드 재고정. CPU에서 결정적 재현 |
| 시드 값 | LSTM은 상수 `LSTM_SEEDS=(0, 1, 2)`(설정으로 바꿀 수 없음). naive·ARIMA는 시드를 쓰지 않으며 기록 값은 −1 | 시드 3개로 평균±표준편차. 시드 수가 다른 실험끼리 비교되는 일을 막음 |
| 분할 규칙 | 날짜는 Europe/Rome 기준 하루 단위, 양끝 포함. 구간은 **예측 대상 시각**으로 판정. 입력(과거 값)은 앞 구간 데이터를 써도 됨. 사용 구간(11/01~12/22)은 서머타임 종료(10/27) 뒤라 모든 날이 144칸(테스트로 확인) | 롤링 예측에서 val 첫 시각도 예측할 수 있게. 대상 시각이 겹치지 않으므로 누수 없음 |
| 미래 데이터 차단 | `run`(I-02)은 D-04 계열을 **val 마지막 시각에서 잘라** 모델에 넘기고, 예측 파일에는 train·val 행만 저장한다. test 구간의 y로 예측·채점하는 것은 `test`(I-05)뿐이다. (v2.5) test 값은 모델 입력·학습·채점·설정 선택·결정 지표(임계치는 train만)·그래프에 쓰지 않는다. 계열을 만들거나 읽는 과정에서 test 행을 메모리에 올렸다가 잘라내는 것은 허용한다: `prepare`(I-01)는 D-04를 phase 끝까지 만들고 test 날짜도 결측 검사(E-2002)·진단 보고(`inspect.json`의 빠진 시각 목록)에 넣으며, `run`·`results`는 계열을 읽은 뒤 자른다. 이 경로에서 test 값을 출력·요약하지 않는다 | test를 여러 번 보는 경로 차단 |
| 결측 채우기 | 연속 3칸 이하 결측은 **직전 관측값으로 채움**(앞값 채우기, 인과적)하고 `is_imputed=True`. 선형 보간은 쓰지 않음 | 선형 보간은 결측 뒤의 실제 값을 입력에 섞어 누수를 만듦 |
| 평가 지표 | MAE, RMSE. 원래 단위, 대상 시각 중 `is_imputed=False`만. 소수 4자리로 보고. LSTM은 시드별 지표를 계산한 뒤 평균과 표준편차(ddof=1)를 보고. 예측값은 음수여도 자르지 않음 | 요구 사항 + 결측 결정 |
| 학습 손실 | LSTM의 학습 손실과 조기 종료용 val 손실도 `is_imputed=False`인 목표만 사용 | 채운 값으로 채점하지 않는다는 원칙을 학습에도 적용 |
| val 수치의 성격 | val은 **선택용**이라 모든 계열이 낙관적으로 편향되고, 조기 종료까지 val로 하는 LSTM은 더 편향됨. 공정한 비교는 test뿐임을 결과 표 머리말과 README에 적음 | 레드팀 #3. 분할을 복잡하게 만들지 않고 명시로 처리 |
| 학습 데이터 범위 | 모든 모델은 train으로만 학습(ARIMA 적합, LSTM 가중치·스케일러). val·test 구간에서는 실제 관측을 입력으로만 반영하고, train+val로 다시 학습하지 않음 | 계열 간 조건을 같게 함. README에 명시 |
| 예측 거리 (v2) | 설정 `horizon` ∈ {1, 3, 6}. 대상 시각 t의 예측은 시점 o = t − h까지의 관측만 씀. 평가 대상 시각 집합은 거리와 무관하게 같음(val의 채우지 않은 칸). 기존 실험의 설정에 `horizon`이 없으면 1로 간주(파일은 고치지 않음). 루트 실험은 horizon=1 | 거리마다 같은 문제를 같은 조건에서 비교 |
| 결정 지표 (v2) | 임계치 θ = `threshold_ratio` × train의 `peak_quantile` 백분위수(구역별, `configs/decision.yaml`, 기본 0.7·0.99·병합 1칸. θ 계산에는 train의 채운 칸도 포함, 현재 채운 칸 0개). **경보**: ŷ_t ≥ θ이면 t에 대한 경보(발행 시각 t − h). **혼잡 구간**: 실제 y ≥ θ인 연속 칸, 사이의 끊김이 `merge_gap`칸 이하면 이어 붙임(끊김 칸도 구간에 포함). **탐지**: 구간 [s, e]에 대해 대상 시각 τ ∈ [s, min(e, s + h − 1)]인 경보가 있으면 탐지(발행 시각 τ − h < s, 즉 혼잡이 관측되기 전). **리드타임** = s − (가장 이른 그런 경보의 발행 시각), 10분~h×10분(v2.1: 발행 시각 = s는 이미 혼잡을 본 뒤라 탐지로 인정하지 않음). **놓친 혼잡** = 구간 수 − 탐지 수. **불필요 경보 시간** = 혼잡 구간 밖이면서 (v2.3) 어느 혼잡 구간의 시작 직전 h칸([s − h, s − 1])에도 들지 않는 경보 칸 수 × 10분(미리 켠 경보는 준비 시간으로 보고 벌점을 주지 않음). 채운 칸은 경보·혼잡에서 제외. (v2.3) LSTM은 시드별로 계산한 뒤 평균±표준편차(ddof=1), 리드타임 평균은 탐지가 있는 시드만. 경보 조건은 예측 ≥ θ(b1). val 구간 안에서만 계산 | 혼잡을 미리 알리는 목적에 맞춤. 저장된 예측에서 사후 계산하므로 정의를 바꿔도 재학습 불필요 |
| 상대 MAE | `rel_mae` = 모델 MAE ÷ 같은 계열·같은 평가 시각에서 계산한 lag-144 MAE. lag-144 MAE는 실험마다 M-10이 내부적으로 계산(따로 실험을 만들지 않음) | 구역 간 평균과 README 가독성 |
| 불확실성(test) | 계열마다 "모델 MAE − lag-144 MAE"의 95% 신뢰구간을 **일별 블록 부트스트랩**으로 계산(블록 = Europe/Rome 하루 144칸, 반복 2000, 시드 0, 백분위수 방식). test가 7일이라 블록이 7개뿐이어서 구간이 거칠다는 점을 한계로 적음 | test 1주 단일 구간에서 우연한 차이인지 판단 |
| 코드 상태 | full 단계의 `run`·`sweep`과 `test`는 git 작업 트리가 dirty면 거부(E-4001, E-4006). dirty 판정은 코드 경로(`src`, `pyproject.toml`, `uv.lock`, `configs/phases.yaml`)의 변경만 봄(문서·결과·실험 설정 변경은 제외). dev는 거부하지 않고 `meta.json`에 `git_dirty=true`로 기록 | 결과가 어느 커밋에서 나왔는지 보장 |
| 모델 비교 조건 | 결과 표의 `compare_group`(D-09)과 `horizon`이 모두 같은 행끼리만 비교. 같은 (phase, zone_rank) 안에서 평가 시각 집합이 기준 실험과 다르면 E-4004 | 공정한 비교 |
| 최종 설정 선택 규칙 | 계열마다 같은 `compare_group`·같은 `horizon`에서 full·완료·`post_test=false`인 실험 중 **val MAE(LSTM은 시드 평균)가 가장 작은 것**. 동률이면 ID가 작은 것. M-14가 `final.yaml`이 이 규칙과 맞는지 검증(E-4006) | test 전 선택을 사후 판단에 맡기지 않음 |
| 재현 허용 오차 | 같은 머신·같은 `uv.lock`·같은 `runtime.threads`에서: naive·ARIMA는 1e-9, LSTM은 같은 시드에서 1e-6. 다른 머신에서는 상대 오차 1e-3을 재현으로 인정 | BLAS·CPU에 따라 부동소수 결과가 달라짐 |
| 실험 이름·번호 | `EXP-###`(세 자리). 설정에 직접 적고, 기존 최대 번호+1이어야 함(실패한 실험 포함). 폴더 `experiments/EXP-###_<name>/`(`name`은 D-05) | 한 요소 변경 규칙을 추적 |
| 루트 실험 | phase마다 루트(부모 없음)는 정확히 1개이고, `model.type=naive, naive.lag=144, zone_rank=1, horizon=1`로 고정 | 부모 없는 실험을 여러 개 만들어 한 요소 규칙을 우회하는 일을 막음 |
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
| I-04 | `results` | F-08, F-10, F-12, F-13 | `experiments/*/`, (v2) `configs/decision.yaml`, 구역 시계열(D-04, 임계치 계산용) | `results/results.csv·md`, (v2) `results/horizon.md`, `results/figures/horizon_mae.png` 재생성. `meta.json`이 없거나 읽을 수 없는 폴더는 `status=corrupt`로 표시하고 건너뜀 | M-12 | E-2001(`decision.yaml` 오류) |
| I-05 | `test --final configs/final.yaml --confirm` | F-09 | 최종 설정(D-06) | `results/test/metrics.csv·md`, `predictions.parquet`, `LOCK` | M-14 → M-05, M-07~M-10, M-13, M-17 | E-2003, E-4004, E-4005, E-4006, E-4007 |

- 모든 명령은 성공하면 종료 코드 0, `TPError`가 나면 1(메시지 `[E-xxxx] 설명`). 예상하지 못한 예외는 2.
- `run`은 먼저 설정을 검증하고, 그다음에 실험 폴더를 만든다. 검증에 실패하면 폴더를 만들지 않는다.
- `test` 순서: (0) `--confirm`이 없거나 작업 트리가 dirty면 거부(E-4006) (1) 락 획득, LOCK 존재 확인(E-4005) (2) `final.yaml`과 참조 실험·D-11 파일 전부 검증·로드(E-4006) (3) 모든 예측·지표를 메모리에서 계산 (4) 임시 폴더에 결과와 LOCK을 쓴 뒤 `results/test/`로 한 번에 이름 변경. (1)~(3)에서 실패하면 아무것도 쓰거나 출력하지 않는다.

### 4.2 구현 단위 (`src/tp/`)
| ID | 모듈·함수 | 책임 | 입력 → 출력 | 의존 |
|---|---|---|---|---|
| M-01 | `data/raw.py: read_raw_day(date) -> DataFrame` | 원본 한 파일 읽기와 형식 검증. `raw_dir`(D-12)에서 `.txt`를 먼저 찾고, 없으면 `.txt.gz`를 찾음. 파일 날짜 밖의 행은 오류가 아니라 개수만 집계 | 날짜 → D-01 행 | M-15 |
| M-02 | `data/cache.py: build_daily_cache(day, force)`, `ensure_daily_caches(...)`, `write_inspect(days)` | 국가 코드 합산, S1 캐시와 `_meta.json` 쓰기, 진단 보고 | D-01 → D-02 | M-01 |
| M-03 | `data/zones.py: select_top_zones(phase, k)`, `ensure_zones(phase)` | train 구간 총량 상위 K개 선택. dev와 full의 상위 구역이 다르면 경고 로그 | D-02 → D-03 | M-02, M-15 |
| M-04 | `prep/series.py: build_series(square_id, phase)`, `ensure_series(phase, square_id)`, `load_series(phase, square_id)`(없거나 유효하지 않으면 E-2003) | phase 범위 ±1일의 캐시를 이어 붙여 10분 정규 인덱스 생성, 결측 판정·앞값 채우기, `is_imputed`, 구간 표시 | D-02 → D-04 | M-02, M-05 |
| M-05 | `prep/split.py: segment_of(ts, phase_cfg)`, `cut_until(series, segment)` | 대상 시각의 구간 판정, 계열을 특정 구간 끝에서 자르기 | 시각 → 구간 | M-15 |
| M-06 | `prep/scale.py: Scaler(kind).fit(train).transform/inverse` | LSTM 입력 스케일링(train에서만 적합). ARIMA는 스케일링하지 않음 | 배열 → 배열 | — |
| M-07 | `models/naive.py: predict_naive(series, lag, targets, horizon=1)` | y[t − max(lag, horizon)] 예측 | D-04 → 예측 | — |
| M-08 | `models/arima.py: fit_arima(train, cfg)`, `predict_arima(params, df, horizon=1)`(df는 잘린 D-04 계열, 대상 시각은 호출자가 고름) | statsmodels `ARIMA`로 train에서 적합한 뒤, 같은 파라미터를 잘린 계열에 적용(`apply`, 재적합 없음)해 1스텝 예측. (v2) h스텝: 칼만 필터의 예측 상태 a_{o+1|o}를 전이 행렬로 h − 1번 전개해 원점 o마다 h스텝 앞 예측(재적합 없음). 계절성 처리는 아래 참고. 파라미터를 D-11로 저장·로드 | D-04 → 예측, D-11 | M-15 |
| M-09 | `models/lstm.py: train_lstm(series, cfg, seed, check_time)`, `predict_lstm(model, series, targets)`, `count_train_samples(series, window, horizon=1)` | 창 데이터셋(목표가 `is_imputed=False`인 창만, (v2) 창은 t − h에서 끝남), 모델, 조기 종료 학습(최저 val 손실 가중치 복원), 곡선 기록, 가중치·스케일러를 D-11로 저장·로드 | D-04 → 예측·곡선, D-11 | M-06, M-13 |
| M-10 | `eval/metrics.py: mae, rmse, eval_mask, eval_times, lag144_mae, evaluate(pred, series, segment, reference_times=None)` | 채운 시각 제외, 기준 시각 집합과 일치 검사, 지표·시드 평균·표준편차·lag-144 대비 상대 MAE 계산 | D-07 → D-08 | — |
| M-11 | `exp/registry.py: resolve_config, check_rules, check_data, run_from_file(path, retry), sweep` | ID·루트·부모·한 요소 검증, 설정 풀기·차이 계산, 락, 시간 확인, 폴더·메타 쓰기, 모델 분기, sweep | D-05 → 실험 폴더 | M-05, M-07, M-08, M-09, M-10, M-12, M-13 |
| M-12 | `exp/results.py: rebuild_results()` | 결과 표·그림 재생성. (v2) horizon 열, 결정 지표 열(M-18), 거리별 표·그래프(D-15) | 실험 폴더들 → D-09, D-15 | M-04, M-10, M-15, M-18 |
| M-18 | (v2) `eval/decision.py: threshold(train_y, cfg)`, `decision_metrics(y_true, y_pred, imputed, thr, horizon, merge_gap)`, `episodes(congested, merge_gap)` | 3.3 결정 지표 정의대로 혼잡 구간·경보·탐지·리드타임·불필요 경보 시간 계산 | D-07 + θ → 결정 지표 | — |
| M-13 | `seed.py: set_seed(seed, threads)` | 3.3의 시드 정책 적용 | — | — |
| M-14 | `exp/testrun.py: run_test(final_cfg, confirm)` | 최종 설정 검증(선택 규칙 포함), 각 실험의 D-11 모델을 불러와(재학습 없음) test 구간 1회 평가, 부트스트랩 신뢰구간, 원자적 결과 쓰기(4.1) | D-06, D-11 → D-10 | M-05, M-07~M-10, M-13, M-17 |
| M-17 | `eval/bootstrap.py: block_bootstrap_diff_ci(err_model, err_ref, day_index, n=2000, seed=0)` | 일별 블록 부트스트랩으로 MAE 차이의 95% 신뢰구간 계산 | 오차 배열 → (차이, 하한, 상한) | — (자체 `default_rng(seed)`) |
| M-15 | `errors.py: TPError`, `config.py: load_phase, load_decision, 경로·상수` | 오류 형식, 단계 설정·(v2) 결정 설정 로드·검증, 경로(D-12), `AGG_VERSION`, `IMPUTE_VERSION`, `LSTM_SEEDS` | D-06a → phase_cfg | — |
| M-16 | `cli.py`, `__main__.py` | 스레드 환경 변수 설정(numpy import 전), 인자 파싱과 명령 분기, 종료 코드 | 인자 → I-01~I-05 | 전부 |

의존 방향: config(M-15)·seed(M-13) → data → prep → models → eval → exp → cli (한 방향).

**모델 초기 설정** (루트와 계열 전환 시의 값. 이후에는 실험으로 바꾼다)
- naive: `lag=144`(공식). lag-1·(v2) lag-1008은 루트의 자식(`changed: naive.lag`)
- (v2) 거리: 모든 계열의 기본 `horizon=1`. 30·60분 실험은 계열별 h=1 최선을 부모로 `changed: horizon`
- ARIMA: `order=[2,1,2]`, `seasonal=none`, (`seasonal=fourier`일 때) `fourier_k=3`. 주기 144의 SARIMA 직접 적합은 30분 제한 때문에 쓰지 않는다
  - `diff144`: z_t = y_t − y_{t−144}로 직접 차분한 뒤 z에 ARIMA(order)를 적합. 예측은 ŷ_t = y_{t−144} + ẑ_t. train 앞 144칸은 적합에서 뺀다
  - `fourier`: Europe/Rome 기준 하루 안의 슬롯 s(0~143)로 sin(2πks/144), cos(2πks/144)(k=1..`fourier_k`)를 외생 변수로 넣음
  - 수렴 판정: `res.mle_retvals["converged"]`가 False이거나 `ConvergenceWarning`이 나면 E-3001
- LSTM: `window=144`, `hidden=64`, `layers=1`, `dropout=0.0`, `lr=1e-3`, `batch=256`, `max_epochs=50`, `patience=5`, `scaler=standard`, 손실 MSE, Adam, 시드 `LSTM_SEEDS`

### 4.3 오류·예외 목록 (`TPError(code, message)`, 종료 코드 1)
| 번호 | 상황 | 바깥에 보이는 형태 | 사용처 | 처리 |
|---|---|---|---|---|
| E-1001 | 캐시가 없는 날짜의 원본이 `raw_dir`(D-12)에 없음(`.txt`, `.txt.gz` 모두). `raw_dir` 폴더가 없는 경우도 캐시가 없는 날짜가 있을 때만 해당 | `[E-1001] 원본 없음: 2013-11-05, …` | M-01, M-02, M-03 (I-01) | 중단. 빠진 날짜를 모두 나열 |
| E-1002 | 원본의 열 개수가 8이 아님, 숫자로 읽을 수 없음, square_id가 1~10000 밖, internet이 음수 | `[E-1002] 형식 오류: <파일>:<줄>` | M-01 (I-01) | 중단 |
| E-1003 | 시각이 10분(600000ms) 배수가 아님 | `[E-1003] 시각 오류: <파일> <값>` | M-01 (I-01) | 중단 |
| E-2001 | 설정 파일 오류. 단계 설정(날짜 순서, 구간 겹침, dev에 test, 허용 범위 11/01~12/22 밖, K가 1~3이 아님), (v2) 결정 설정(`threshold_ratio`가 (0, 1] 밖, 즉 0 이하 또는 1 초과, `peak_quantile` 0.5~1 밖, `merge_gap` 0~6 밖, `report_horizon`이 1·3·6이 아님, 키 누락) | `[E-2001] 분할 설정 오류: <내용>` / `[E-2001] 결정 설정 오류: <내용>` | M-15 (I-01, I-02, I-04) | 중단 |
| E-2002 | 구역 시계열에 연속 4칸 이상 결측(계열 맨 앞은 제외. 맨 앞 결측은 계열 시작을 뒤로 미루고 로그) | `[E-2002] 긴 결측: <구역> <시작>~<끝>` | M-04 (I-01) | 중단. 규칙 변경은 변경 통제 |
| E-2003 | 필요한 구역 목록(D-03)이나 구역 시계열(D-04)이 없거나, 3.3 "산출물 유효성"의 판정 항목이 현재 설정과 다름 | `[E-2003] prepare 먼저 실행: --phase <phase>` | M-04 `load_series`, M-11 (I-02), M-14 (I-05) | 중단 |
| E-3001 | ARIMA가 수렴하지 않음(4.2 판정 기준) | `[E-3001] ARIMA 수렴 실패: <order>` | M-08 (I-02) | 실험을 실패 상태로 기록 |
| E-3002 | LSTM 손실이 NaN 또는 inf | `[E-3002] 학습 발산: seed=<n> epoch=<e>` | M-09 (I-02) | 실험을 실패 상태로 기록 |
| E-3003 | 확인 시점(적합 뒤, 에폭마다)에 경과 시간이 `runtime.time_limit_min` 초과 | `[E-3003] 시간 초과: <경과>` | M-11 (I-02) | 실험을 실패 상태로 기록 |
| E-4001 | 설정 검증 실패: D-05 규칙 위반, ID가 최대+1이 아님, 루트 규칙 위반, 부모 대비 바뀐 키가 정확히 1개가 아님, `changed`가 실제 차이와 다름, 계열 전환 시 새 계열의 하위 키가 초기값이 아님, LSTM 학습 샘플 수(목표가 train이고 `is_imputed=False`이며 창이 t − h에서 끝나는 샘플 수) < `lstm.batch`, full 단계에서 작업 트리가 dirty | `[E-4001] 실험 규칙 위반: <내용>` | M-11 (I-02, I-03) | 폴더를 만들지 않고 중단 |
| E-4002 | 부모 실험이 없거나, 완료 상태가 아니거나, phase가 다름(`changed: phase`로 dev 부모를 쓰는 경우는 예외) | `[E-4002] 부모 오류: <EXP>` | M-11 (I-02, I-03) | 중단 |
| E-4003 | 이미 있는 ID를 `--retry` 없이 실행, 완료된 실험에 `--retry`, 재시도 시 설정 해시가 기존 `meta.json`과 다름 | `[E-4003] 재실행 불가: <EXP> <상태>` | M-11 (I-02) | 중단. (v2) v1에서 실패한 실험은 horizon이 해시에 들어가 재시도 불가 → 새 ID |
| E-4004 | 평가 시각 집합이 기준(같은 phase·zone_rank에서 처음 완료된 실험, test에서는 계열 간)과 다름 | `[E-4004] 평가 시각 불일치: <차이 개수>` | M-10, M-07·M-09(과거 기록 부족) (I-02, I-05) | 중단 |
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

**D-03 구역 목록** (`data/processed/<phase>/zones.json`): `{phase, k, train_range, phase_ranges, zones: [{rank, square_id, train_total}]}`(`phase_ranges`는 유효성 판정용 구간 날짜). 동률이면 square_id 오름차순.

**D-04 구역 시계열** (`data/processed/<phase>/series_<square_id>.parquet` + `_meta.json`: `square_id, phase, phase_ranges, agg_version, impute_version`)
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
| horizon | int | 아니오(1) | 1 \| 3 \| 6 (v2) |
| naive.lag | int | type=naive | 1 \| 144 \| 1008 (v2) |
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
- D-06a `configs/phases.yaml`: `dev: {train: [2013-11-04, 2013-11-13], val: [2013-11-14, 2013-11-17], k: 1}`, `full: {train: [2013-11-01, 2013-12-08], val: [2013-12-09, 2013-12-15], test: [2013-12-16, 2013-12-22], k: 3}`(F-11로 1→3). 검증은 E-2001 참고.
- D-06 `configs/final.yaml`: `{zones: {1: {naive_144: EXP-###, naive_1: EXP-###, arima: EXP-###, lstm: EXP-###}, 2: {...}, 3: {...}}}`(v1 형식. lag-1008 키는 F-09 개정 때 정함). 키는 zone_rank이고 개수는 full의 K와 같아야 함. 검증은 E-4006. Must(K=1)에서는 키 1만 둠. F-11에서 구역 2·3의 실험은 구역 1의 최종 실험을 부모로 두고 `changed: zone_rank`로 만듦(한 요소 규칙 유지)

**D-07 예측** (`predictions.parquet`): `time_utc, segment(train|val), y_true, y_pred, seed`(LSTM이 아니면 −1), `is_imputed`. test 행은 없음.
**D-08 지표** (`metrics.json`): `{val: {mae, rmse, rel_mae, n}, val_std: {mae, rmse}|null, seeds: {"0": {mae, rmse}, …}|null}`. LSTM의 `val`은 시드별 지표의 평균, `val_std`는 ddof=1 표준편차. naive·ARIMA는 `val_std`와 `seeds`가 null. n은 채점한 시각의 수.
**D-09 결과 표** (`results/results.csv·md`): `exp_id, name, phase, parent, changed, model_type, horizon, zone_rank, square_id, val_mae, val_mae_std, val_rmse, val_rmse_std, val_rel_mae, n, compare_group, dec_threshold, dec_episodes, dec_missed, dec_false_alarm_min, dec_lead_min, dec_missed_std, dec_false_alarm_min_std, dec_lead_min_std, status, post_test, duration_s`(v2: `horizon`, `dec_*` 추가. `dec_*`는 M-18로 계산, 구역 시계열이 없으면 빈칸. (v2.3) `dec_*_std`는 LSTM 시드 간 표준편차, 다른 계열은 빈칸). `compare_group`은 (phase, square_id, 평가 시각 집합, `AGG_VERSION`, `IMPUTE_VERSION`)의 해시 앞 8자리.
**D-10 test** (`results/test/`): `metrics.csv·md`(행: 계열 × zone_rank. 열: square_id, MAE, RMSE, rel_mae, mae_diff_vs_lag144, ci_low, ci_high. LSTM은 D-08과 같은 정의로 평균±표준편차이고, 신뢰구간은 시드 평균 예측 오차로 계산(v1 형식. F-09 개정 때 다시 정함). F-11이면 구역 평균 행 추가(rel_mae 평균)), `predictions.parquet`(`family, zone_rank, square_id, time_utc, y_true, y_pred, seed, is_imputed`), `LOCK`(`{evaluated_at, final: {...}, git_commit}`).
**D-11 모델 산출물** (실험 폴더의 `model/`): naive는 없음. ARIMA는 `arima_params.json`(order, seasonal, fourier_k, 적합된 파라미터 벡터). LSTM은 `lstm_seed{n}.pt`(최저 val 손실 시점의 state_dict)와 `scaler.json`(train 통계). I-05는 이 파일만 불러와 예측하고 재학습하지 않음.
**D-12 경로 설정**: `raw_dir`는 환경 변수 `TP_RAW_DIR`로 정하고, 없으면 `data/raw`. 캐시·처리 데이터 경로(`data/interim`, `data/processed`)는 저장소 기준으로 고정.
**D-13 실험 메타** (`meta.json`): `id, parent, changed, reason, hypothesis, status(running|completed|failed), error_code, post_test, config_hash, git_commit, git_dirty, started_at, ended_at, duration_s, pid`, 구현 추가 필드 `square_id, agg_version, impute_version, compare_group`.
**D-14 결정 설정** (v2, `configs/decision.yaml`): `{threshold_ratio: 0.7, peak_quantile: 0.99, merge_gap: 1, report_horizon: 6}`. 검증은 E-2001. `report_horizon`은 결정 지표를 대표로 보고하는 거리.
**D-15 거리별 비교** (v2, `results/horizon.md`, `results/figures/horizon_mae.png`): full 단계에서 (구역, 거리, 계열)마다 선택 규칙상 최선 실험의 val MAE(LSTM은 ±표준편차)와 `report_horizon`의 결정 지표. 계열은 마지막 관측값(lag-1. 10분 뒤 예측에서는 10분 전 값, 거리 h에서는 예측 시점의 값 y[t − h]), 어제 같은 시각(lag-144), 지난주 같은 시각(lag-1008), ARIMA, LSTM. LSTM 결정 지표는 평균±표준편차. 그래프는 구역별 패널 3개, x축 거리(10/30/60분), y축 val MAE, 계열별 선(팔레트 1~5번 고정 순서, 계열별 마커 모양)과 범례.

### 4.5 기술 스택·폴더 구조·명령·실행 자산
**스택** (Python 3.12. uv로 설치하고 `uv.lock`에 정확한 버전을 고정. 3.14가 설치되어 있지만 torch·statsmodels 휠 호환성 때문에 3.12를 씀)
pandas ≥2.2, numpy ≥1.26, pyarrow ≥15, statsmodels ≥0.14, torch ≥2.3 (CPU 휠), matplotlib ≥3.8, pyyaml ≥6, filelock ≥3.13, pytest ≥8, ruff ≥0.6

**폴더**
```
traffic-predict/
├─ pyproject.toml, uv.lock, README.md (F-10, 예정), CLAUDE.md, .gitignore
├─ docs/traffic-predict_기획서.md
├─ configs/ phases.yaml, decision.yaml (v2), final.yaml (test 전, 예정), experiments/EXP-###.yaml
├─ src/tp/ __init__.py, __main__.py, cli.py, config.py, errors.py, seed.py
│   ├─ data/ raw.py, cache.py, zones.py
│   ├─ prep/ series.py, split.py, scale.py
│   ├─ models/ naive.py, arima.py, lstm.py
│   ├─ eval/ metrics.py, bootstrap.py, decision.py (v2)
│   └─ exp/ registry.py, results.py, testrun.py, selection.py (3.3 선택 규칙, results·test 공용)
├─ tests/ fixtures/, test_*.py
├─ data/        (git 제외) raw/, interim/daily/, processed/<phase>/
├─ experiments/ (커밋) EXP-###_<name>/
└─ results/     (커밋) results.csv·md, horizon.md (v2), figures/, test/ (test 뒤 생성)
```

**명령**
```bash
uv sync
uv run python -m tp prepare --phase dev
uv run python -m tp prepare --phase full
uv run python -m tp run --config configs/experiments/EXP-001.yaml
uv run python -m tp sweep --parent EXP-### --key <키> --values '<JSON>' --start-id EXP-### --name <접두어> --reason <문장> --hypothesis <문장>
uv run python -m tp results
uv run python -m tp test --final configs/final.yaml --confirm   # 사용자 지시("설정 동결, test 실행") 후 1회만
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
| 레드팀 1회(오류 37, 개선 15, 확장 2): 오류 모두 반영, 확장 2건은 한 줄 규칙으로 처리 | — | plan-project 5단계(검토·동결) 절차 |
| val 편향은 명시만 | holdout 분리 | 공정 비교는 test 1회로 보장. 분할 단순성 유지 |
| 상대 MAE, 일별 블록 부트스트랩 CI 추가 | 원래 단위 MAE·RMSE만 | 구역 간 평균과 test 1주의 불확실성 표시 |
| `test --confirm`, full·test는 dirty 거부, LSTM 샘플 수 검사 | 정책으로만 둠 | 코드로 강제 |
| 모든 모델 train으로만 학습 | test 전 train+val 재학습 | 계열 간 조건 동일, 저장된 모델(D-11) 재사용 |
| 원본 경로는 `TP_RAW_DIR`, `.txt`/`.txt.gz` 모두 읽음 | 보관 방식 하나로 고정 | 당시 디스크 8GB. 사용자가 공간을 마련 중이라 방식이 정해지지 않음(이후 52GB 확보) |
| test는 저장된 모델(D-11)을 불러와 평가 | test 때 재학습 | 재학습 비용(최대 30분 × 계열)과 재현 차이 위험 제거 |
| 주기 144 SARIMA는 쓰지 않고 `diff144`/`fourier`로 계절성 처리 | SARIMA(s=144) | 30분 제한 |
| Python 3.12 (uv) | 설치된 3.14 | torch·statsmodels 휠 호환성 |
| 결측 판정: 파일 전체에서 빠진 시각 = 결측, 구역 행만 없음 = 0 | 구역 행이 없으면 모두 결측 | 드문 구역은 활동이 없을 때 행이 없음. F-01에서 확인 |
| (v2) 최종 목적을 무선망 운용 결정(혼잡 예방)으로 정하고 예측 거리 10/30/60분·결정 지표 추가 | 10분 예측만 비교 | 용량 셀을 "미리" 켜려면 더 먼 거리와 결정 수준 평가가 필요 |
| (v2) 피크 = train 99번째 백분위수 | train 최댓값(원안), 99.9번째 | 최댓값은 튀는 값 하나에 좌우됨(5161: val 혼잡 2회뿐). P99로 세 구역 모두 5~13회(병합 전) |
| (v2) 탐지는 혼잡 시작 전 경보만 인정 | 구간 안 경보도 인정 | 미리 켜는 목적에 맞춤 |
| (v2.1) 탐지는 발행 시각 < 혼잡 시작(엄격히 이전), 리드타임 10분~거리(60분 뒤면 10~60분) | 발행 시각 ≤ 시작(v2, 리드타임 0 허용) | 발행 시각 o에는 y[o]가 이미 관측됨. o = s면 혼잡을 본 뒤의 경보라 미리 알린 것이 아님. v2 정의에서는 마지막 관측값 기준선이 리드타임 0분으로 '탐지'되는 왜곡이 있었음(2026-09-23 사용자 승인) |
| (v2) 1칸 끊김은 이어 붙여 한 혼잡으로 셈 | 연속 구간 그대로 | 임계치 근처 흔들림으로 횟수가 부풀려지는 것을 막음 |
| (v2) 거리 h 기준선: ŷ_t = y[t − max(lag, h)] | lag 고정 | "10분 전 값"을 "예측 시점의 마지막 관측값"으로 일반화 |
| (v2) ARIMA h스텝은 칼만 예측 상태 전개 | 원점마다 apply+forecast | 결과는 같고(테스트로 대조) 훨씬 빠름 |
| (v2) LSTM은 거리마다 따로 학습(direct) | 1스텝 반복 적용(recursive) | 구현 단순, 오차 누적 없음 |
| (v2) 결정 지표는 저장된 예측에서 `results`가 계산 | 실험 실행 때 계산 | 임계치·정의 변경 시 재학습 불필요 |
| (v2.3) 기준선 이름 '마지막 관측값(10분 뒤 예측에서는 10분 전 값)' | 대상 시각의 10분 전 값 y[t − 1] | 먼 거리에서 y[t − 1]은 예측 시점 이후 값이라 누수. 계산은 그대로, 이름만 명확히 |
| (v2.3) 혼잡 시작 직전 h칸 안의 경보는 불필요 경보에서 제외 | 구간 밖 경보 모두 벌점(v2) | 미리 켠 경보를 벌점으로 세면 리드타임과 모순되어 일찍 경보하는 모델이 불리해짐 |
| (v2.3) LSTM 결정 지표는 시드별 계산 후 평균±표준편차 | 시드 평균 예측(v2) | 시드 평균 예측은 사실상 3개 모델 앙상블이라 ARIMA보다 유리함. MAE 표와 같은 방식으로 공정하게 비교 |
| (v2.3) 경보 조건 예측 ≥ θ, 30·60분은 10분 뒤에서 고른 하이퍼파라미터 재사용, 채운 칸 제외는 유지 | k×θ 경보, 거리별 재조정, 채운 값 사용 | 3단계는 같은 조건의 모델 비교. 비대칭 비용은 4단계에서 다룸 |
| (v2.4) 문서 문구를 실제 코드에 맞춤(시그니처·오류 사용처·파일 필드·dirty 범위·스레드 설정), full 실험은 사용자 승인 후로 3.1 통일, "단계"(목적 순서)와 "phase"(dev/full) 구분 | 그대로 둠 | 변경이 쌓이며 문서끼리 어긋남(2026-09-23 검토). 동작·결정 값은 바꾸지 않음 |
| (v2.5) test 차단은 "test 값을 모델·채점·선택·결정 지표·그래프에 쓰지 않음"으로 정의하고, 계열 생성·로드 중 메모리에 올렸다 자르는 것은 허용 | 계열을 val 끝까지만 만들어 test 행을 아예 읽지 않게 코드 변경 | 실제 누수 경로가 없고(모델 입력·채점에 안 들어감) 코드 변경은 캐시 재생성·재현 확인 비용이 큼. 문구가 동작보다 강해 생기던 혼동만 없앰(사용자 결정) |
| (v2) test(F-09) 형식 개정은 4단계 변경 통제로 미룸 | 지금 개정 | test는 1~4 확정 뒤 한 번. 시뮬레이션 설계 전에 형식을 정하면 다시 바뀜 |

## 동결 체크리스트
- [x] 핵심 흐름이 Must 기능만으로 끝까지 이어진다 (F-01 → F-02 → F-03 → F-05~F-07 + F-08 → F-04 → (v2) F-12 → F-13 → F-09. v2.4에서 다시 확인)
- [x] 모든 Must·Should에 검증 가능한 완료 조건이 있다
- [x] 제외 목록에 이유가 있다
- [x] 3.1·3.2·3.3에 빈칸이 없다
- [x] 스크립트 인자 = 설정 키 = 코드 변수 (불일치 0: `--phase`↔`phases.yaml`↔`load_phase`, `--key`↔D-05 점 표기 키, `TP_RAW_DIR`↔D-12)
- [x] 인터페이스가 부르는 구현 = 구현 표 (M-01~M-18 모두 정의·사용. v2.4에서 시그니처를 코드에 맞춤)
- [x] 모든 오류 번호에 사용처가 있고, 모든 예외에 오류 번호가 있다 (E-1001~E-4007, 16개)
- [x] 모든 입력 값에 검증 규칙이 한 곳에 적혀 있다 (4.4 D-01, D-05, D-06, D-14)
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
| 6 | F-08 + F-05 | M-11(규칙·락·메타·sweep), M-12, M-07 / I-02, I-03, I-04 / 테스트(루트·한 요소·계열 전환·ID·retry·dirty) → dev EXP-001(lag-144), EXP-002(lag-1) | 완료 | 코드·테스트 완료(pytest 117 통과). 실데이터 dev 기준선(2026-09-23): 상위 구역 5259, 결측 0칸 / EXP-001 lag-144 val MAE 503.76·RMSE 897.52 / EXP-002 lag-1 val MAE 101.44·RMSE 151.32(rel_mae 0.201). 독립 계산과 일치. lag-144 오차는 토요일(11/16, 금→토) MAE 1360이 대부분: 요일 효과(평일 평균 약 1750, 주말 약 520~780). 구현 세부: dirty 판정은 코드 경로(`src`, `pyproject.toml`, `uv.lock`, `configs/phases.yaml`)만 봄 / `meta.json`에 `square_id·agg_version·impute_version·compare_group` 추가(실행 당시 버전으로 compare_group 계산) / `results/figures/*.png`는 F-10에서 구현(이후 변경: `horizon_mae.png`는 F-13에서 만듦) |
| 7 | F-06 | M-08(none·diff144·fourier, 수렴 판정, D-11) / 테스트(재현 1e-9, apply가 재적합 안 함) → dev ARIMA 첫 실험 | 완료 | pytest 131 통과. full 길이 합성 계열(train 38일+val 7일) 적합 시간: (2,1,2) none 1.3초 / diff144 0.9초 / fourier 2.4초, (5,1,5)+fourier K=10 23.9초 → 30분 제한 위험 없음. 실데이터 dev 실험은 원본 도착 후 |
| 8 | F-07 | M-06, M-09(D-11, 곡선, 샘플 수 검사) / 테스트(같은 시드 1e-6, 스케일러는 train만) → dev LSTM 첫 실험 | 완료 | pytest 145 통과. 같은 시드 재실행 1e-6 이내 동일(Windows CPU, deterministic). full 길이 합성 계열에서 초기 설정 시드 3개 학습 145초. 학습 곡선은 로그 y축, 팔레트 1·2번(train 파랑/val 주황), 최저 val 에폭 표시 |
| 9 | F-09 | M-17, M-14 / I-05 / 테스트(픽스처로만: LOCK·원자성·confirm·dirty·선택 규칙·부트스트랩 재현) | 완료(v1 형식) | pytest 162 통과(픽스처 full 단계로 검증: LOCK·E-4005·--confirm·dirty·선택 규칙·D-11 누락·원자성·재학습 없음). 구현 중 발견한 버그: final.yaml 키 순서(알파벳)대로 처리해 부트스트랩 기준이 lag-144가 아니게 되던 문제 → 항상 FAMILIES 순서로 처리하도록 수정, 회귀 테스트 유지. 실제 test는 순서 12(사용자 승인 필수) |
| 10 | 운영 | 원본 11/01~12/22 확보 → `prepare --phase full` → full 실험(`changed: phase`로 옮기기) | 완료 | 2026-09-23. 53개 파일 검증 통과, 결측 0칸. full 상위 구역 5161(dev 5259와 다름, 경고 기록). EXP-013~017 실행(EXP-015 E-3001) |
| 11 | F-11 | K=3 `prepare`, 구역 2·3 자식 실험 | 완료 | 2026-09-23. full k=3 → 상위 구역 5161·5059·5259. 구역 2·3 자식 실험 EXP-018~025(changed zone_rank) 모두 완료. val MAE(lag144/lag1/ARIMA/LSTM): 5059 213.4/99.9/94.6/95.3±4.4, 5259 513.1/89.9/82.6/79.3±0.3. 세 구역 모두 ARIMA·LSTM이 lag-1보다 4.6~11.8% 나음. 5259(업무 지구)는 38일 학습에도 LSTM 주말 오차(63.0)가 lag-1(50.9)보다 큼 → 주말 약점은 데이터 양보다 구역 성격. test 표 구역별·평균 행은 픽스처 테스트로 검증, 실제 확인은 순서 12 |
| 12 | 운영 | 실제 `test --confirm` 1회 | 대기 | **사용자 승인 필수**. (v2) 3·4단계를 val로 확정하고 F-09 형식을 개정한 뒤에만 |
| 13 | F-10 | README 결과·한계 정리 | 대기 | Should |
| 14 | F-12 | (v2) M-07(horizon, lag-1008), M-08(h스텝), M-09(창 t − h), M-11(horizon 키·기본값·루트·샘플 수), M-14(선택 규칙 같은 horizon) / 테스트(naive 정확값, ARIMA 기준 방법 대조 1e-6, 인과성, 기존 실험 horizon=1 간주) | 완료 | 2026-09-23. pytest 190 통과. ARIMA h스텝(칼만 예측 상태 전개)이 원점별 filter+forecast와 1e-6 이내 일치(none·diff144·fourier × h 3·6), 인과성 확인. 기존 실험 25개는 설정에 horizon이 없어 1로 읽힘(파일 수정 없음). 완료 조건 (6) 거리별 표·그래프는 15(F-13, M-12)에서. 알려진 제약: horizon이 해시에 들어가 v1에서 실패한 실험(EXP-015)은 `--retry` 시 설정 해시가 달라 E-4003 |
| 15 | F-13 | (v2) M-18, M-15(`load_decision`), M-12(horizon·dec_* 열, D-15 표·그래프) / 테스트(손계산 예제, 설정 변경 반영, E-2001) | 완료 | 2026-09-23. pytest 208 통과. 손계산 예제(구간 병합·탐지·놓침·리드타임·불필요 경보) 일치, 설정 변경이 재학습 없이 반영됨을 확인. 구현 중 발견한 버그: 결과 표에서 naive lag가 실수(1.0)로 읽혀 거리별 표에서 기준선이 빠지던 문제 → 정수 변환, 테스트로 고정. 팔레트 검증기(Node) 없음 → 문서화된 기본 팔레트 1~5번 + 계열별 마커 모양으로 보완 |
| 16 | 운영 | (v2) full: lag-1008 h=1 3개 + h=3·6 × 5계열 × 3구역 30개 → 거리별 표·그래프·결정 지표 | 완료 | 2026-09-23. EXP-026~058 33개 모두 완료(약 10분). 60분 뒤 val MAE(구역 5161/5059/5259): 마지막 관측값 349/259/252, lag-144 295/213/513, lag-1008 214/203/185, ARIMA 258/199/205, LSTM 173/147/155. 결정 지표(60분, v2.1 정의)는 `results/horizon.md`(이후 17에서 v2.3 정의로 재계산) |
| 17 | F-13 개정 | (v2.3) M-18(시작 직전 h칸 경보 제외), M-12(LSTM 시드별 결정 지표·표준편차 열, 기준선 이름) / 테스트(손계산 예제 갱신, 사전 경보 제외, 시드별 평균·표준편차) | 완료 | 2026-09-23 사용자 결정. pytest 211 통과. 60분 뒤 결정 지표 재계산(`results/horizon.md`): 사전 경보 제외로 불필요 경보가 대부분 줄고(예: 5161 LSTM 180→130분), LSTM은 시드별 평균±표준편차(예: 5161 놓침 7.3±1.2/12) |
| 18 | 선택 규칙 일원화 | 3.3 최종 설정 선택 규칙을 `exp/selection.py` 한 곳에 두고 M-12(거리별 표)와 M-14(test 검증)가 같이 씀 / 테스트(최소 val MAE·ID 동률, 후보 조건, 구역·거리·계열·compare_group별 분리, 거리별 표가 구역 기준 compare_group 안에서만 고름) | 완료 | 2026-09-23. 발견한 버그: 거리별 표의 '최선'이 compare_group을 보지 않았음(기획서 3.3과 다름) → 재현 테스트로 실패 확인 후 수정. 거리별 표는 구역의 첫 완료 full 실험(E-4004 기준)의 compare_group 안에서만 고르고, 다른 그룹 실험은 경고 로그. 실제 결과는 구역마다 그룹이 하나라 `results/` 변화 없음. pytest 216 통과 |
| 19 | I-04 손상 폴더 | M-12 `load_rows`: 파일이 없거나 형식이 틀리거나 필수 키가 없는 실험 폴더를 `status=corrupt`로 표시하고 나머지는 계속 처리(경고 로그) / 테스트(빈 config, 목록 config, 키 누락, status 없는 meta, 목록 meta, 깨진·빈 metrics) | 완료 | 2026-09-23. 발견한 버그: 이런 폴더 하나로 `results` 전체가 종료 코드 2로 멈춤(I-04와 다름) → 재현 테스트 7개로 실패 확인 후 수정. 실제 `results/` 변화 없음. pytest 223 통과. 발견한 `registry.scan`의 같은 빈틈은 20에서 수정 |
| 20 | 실험 목록 손상 판정 | M-11 `scan`: `meta.json`에 D-13 필수 필드(`status`·`post_test`·`config_hash`·`compare_group`·`square_id`)가 없거나 `status`가 running·completed·failed가 아니거나, 설정에 `phase`·`parent`·`zone_rank`·`model.type`이 없으면 corrupt(경고 로그). corrupt 실험은 부모가 될 수 없고(E-4002) `--retry`도 불가(E-4003) / 테스트 5개(다음 실험 실행·재시도·부모 지정) | 완료 | 2026-09-23. 재현 테스트 3개가 먼저 실패함을 확인(필수 필드 누락·알 수 없는 status를 completed로 읽음). 실제 실험 58개는 모두 읽힘(완료 57, 실패 1), `results/` 변화 없음. pytest 228 통과 |
| 21 | 테스트 정리 | 테스트만: F-13 주석을 v2.1 정의(발행 시각 < 시작)로 수정 / "재학습 없음" 가드 강화(모델 학습·예측 진입점 6곳이 불리면 실패 + 실험 폴더 바이트 동일 확인) / horizon 없는 v1 폴더를 `results`와 `lstm.load_model`도 1로 읽는지 테스트 추가 / h=1 인과성 테스트에 h>1 위치 주석 | 완료 | 2026-09-23. 옛 가드는 `results`가 부르지 않는 함수만 막아 아무것도 검증하지 못했음. 새 가드는 `results`에 재예측 코드를 일부러 넣었을 때 실패함을 확인한 뒤 되돌림. 코드 변경 없음. pytest 229 통과 |
| 22 | test 차단 문구 | 문서만(v2.5): 3.3 "미래 데이터 차단", CLAUDE.md 핵심 원칙·작업 규칙에 test 사용의 범위를 명시 | 완료 | 2026-09-23 사용자 결정(문구를 좁힘). 코드 변경 없음. 작업 규칙 원문은 그대로 두고 범위 설명을 덧붙임 |
