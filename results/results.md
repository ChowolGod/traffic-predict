# 실험 결과 (val)

val 지표는 설정 **선택용**이라 낙관적으로 편향되어 있습니다(조기 종료까지 val로 하는 LSTM은 더 편향됨). 계열 간 공정한 비교는 test 결과(`results/test/`)만 해당합니다. `compare_group`이 같은 행끼리만 비교할 수 있습니다.

| exp_id | name | phase | parent | changed | model_type | zone_rank | square_id | val_mae | val_mae_std | val_rmse | val_rmse_std | val_rel_mae | n | compare_group | status | post_test | duration_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| EXP-001 | lag144 | dev |  |  | naive | 1 | 5259 | 503.7637 |  | 897.5233 |  | 1.0000 | 576 | e9ffdf9f | completed | False | 1.9370 |
| EXP-002 | lag1 | dev | EXP-001 | naive.lag | naive | 1 | 5259 | 101.4407 |  | 151.3227 |  | 0.2014 | 576 | e9ffdf9f | completed | False | 1.3750 |
| EXP-003 | arima-212 | dev | EXP-001 | model.type | arima | 1 | 5259 | 94.1631 |  | 137.0304 |  | 0.1869 | 576 | e9ffdf9f | completed | False | 1.8120 |
| EXP-004 | arima-seasonal-diff144 | dev | EXP-003 | arima.seasonal | arima | 1 | 5259 | 142.4173 |  | 199.1959 |  | 0.2827 | 576 | e9ffdf9f | completed | False | 1.5790 |
| EXP-005 | arima-seasonal-fourier | dev | EXP-003 | arima.seasonal | arima | 1 | 5259 | 105.8979 |  | 143.6543 |  | 0.2102 | 576 | e9ffdf9f | completed | False | 0.5000 |
| EXP-006 | arima-202-diff144 | dev | EXP-004 | arima.order | arima | 1 | 5259 | 142.3831 |  | 199.6453 |  | 0.2826 | 576 | e9ffdf9f | completed | False | 1.8130 |
| EXP-007 | arima-order-1-1-1 | dev | EXP-003 | arima.order | arima | 1 | 5259 | 98.3630 |  | 146.1301 |  | 0.1953 | 576 | e9ffdf9f | completed | False | 1.6410 |
| EXP-008 | arima-order-3-1-3 | dev | EXP-003 | arima.order | arima | 1 | 5259 | 95.1271 |  | 138.2604 |  | 0.1888 | 576 | e9ffdf9f | completed | False | 0.4530 |
| EXP-009 | arima-order-5-1-5 | dev | EXP-003 | arima.order | arima | 1 | 5259 | 93.8156 |  | 136.9472 |  | 0.1862 | 576 | e9ffdf9f | completed | False | 2.2190 |
| EXP-010 | lstm-init | dev | EXP-001 | model.type | lstm | 1 | 5259 | 107.4369 | 1.6576 | 150.2999 | 2.2700 | 0.2133 | 576 | e9ffdf9f | completed | False | 48.4220 |
| EXP-011 | lstm-epochs100 | dev | EXP-010 | lstm.max_epochs | lstm | 1 | 5259 | 103.0124 | 2.3310 | 145.6739 | 3.0378 | 0.2045 | 576 | e9ffdf9f | completed | False | 62.6570 |
| EXP-012 | lstm-window24 | dev | EXP-011 | lstm.window | lstm | 1 | 5259 | 114.1215 | 4.4820 | 157.9242 | 5.9282 | 0.2265 | 576 | e9ffdf9f | completed | False | 10.2030 |
| EXP-013 | full-lag144 | full |  |  | naive | 1 | 5161 | 294.8942 |  | 525.4935 |  | 1.0000 | 1008 | aeb67261 | completed | False | 5.9370 |
| EXP-014 | full-lag1 | full | EXP-013 | naive.lag | naive | 1 | 5161 | 116.8333 |  | 175.1906 |  | 0.3962 | 1008 | aeb67261 | completed | False | 1.4060 |
| EXP-015 | full-arima-515 | full | EXP-009 | phase | arima | 1 | 5161 |  |  |  |  |  |  |  | failed | False | 14.0160 |
| EXP-016 | full-lstm-init-e100 | full | EXP-011 | phase | lstm | 1 | 5161 | 110.8343 | 3.9510 | 160.0432 | 3.1561 | 0.3758 | 1008 | aeb67261 | completed | False | 117.6560 |
| EXP-017 | full-arima-212 | full | EXP-003 | phase | arima | 1 | 5161 | 109.5478 |  | 164.5672 |  | 0.3715 | 1008 | aeb67261 | completed | False | 2.2340 |
