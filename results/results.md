# 실험 결과 (val)

val 지표는 설정 **선택용**이라 낙관적으로 편향되어 있습니다(조기 종료까지 val로 하는 LSTM은 더 편향됨). 계열 간 공정한 비교는 test 결과(`results/test/`)만 해당합니다. `compare_group`이 같은 행끼리만 비교할 수 있습니다.

| exp_id | name | phase | parent | changed | model_type | zone_rank | square_id | val_mae | val_mae_std | val_rmse | val_rmse_std | val_rel_mae | n | compare_group | status | post_test | duration_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| EXP-001 | lag144 | dev |  |  | naive | 1 | 5259 | 503.7637 |  | 897.5233 |  | 1.0000 | 576 | e9ffdf9f | completed | False | 1.9370 |
| EXP-002 | lag1 | dev | EXP-001 | naive.lag | naive | 1 | 5259 | 101.4407 |  | 151.3227 |  | 0.2014 | 576 | e9ffdf9f | completed | False | 1.3750 |
| EXP-003 | arima-212 | dev | EXP-001 | model.type | arima | 1 | 5259 | 94.1631 |  | 137.0304 |  | 0.1869 | 576 | e9ffdf9f | completed | False | 1.8120 |
| EXP-004 | arima-seasonal-diff144 | dev | EXP-003 | arima.seasonal | arima | 1 | 5259 | 142.4173 |  | 199.1959 |  | 0.2827 | 576 | e9ffdf9f | completed | False | 1.5790 |
| EXP-005 | arima-seasonal-fourier | dev | EXP-003 | arima.seasonal | arima | 1 | 5259 | 105.8979 |  | 143.6543 |  | 0.2102 | 576 | e9ffdf9f | completed | False | 0.5000 |
