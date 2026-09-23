# 실험 결과 (val)

val 지표는 설정 **선택용**이라 낙관적으로 편향되어 있습니다(조기 종료까지 val로 하는 LSTM은 더 편향됨). 계열 간 공정한 비교는 test 결과(`results/test/`)만 해당합니다. `compare_group`과 `horizon`이 같은 행끼리만 비교할 수 있습니다. `dec_*`는 각 실험의 거리 기준 결정 지표입니다(`configs/decision.yaml`).

| exp_id | name | phase | parent | changed | model_type | horizon | zone_rank | square_id | val_mae | val_mae_std | val_rmse | val_rmse_std | val_rel_mae | n | compare_group | dec_threshold | dec_episodes | dec_missed | dec_false_alarm_min | dec_lead_min | status | post_test | duration_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| EXP-001 | lag144 | dev |  |  | naive | 1 | 1 | 5259 | 503.7637 |  | 897.5233 |  | 1.0000 | 576 | e9ffdf9f | 2648.6353 | 3 | 1 | 540 | 10.0000 | completed | False | 1.9370 |
| EXP-002 | lag1 | dev | EXP-001 | naive.lag | naive | 1 | 1 | 5259 | 101.4407 |  | 151.3227 |  | 0.2014 | 576 | e9ffdf9f | 2648.6353 | 3 | 0 | 30 | 0.0000 | completed | False | 1.3750 |
| EXP-003 | arima-212 | dev | EXP-001 | model.type | arima | 1 | 1 | 5259 | 94.1631 |  | 137.0304 |  | 0.1869 | 576 | e9ffdf9f | 2648.6353 | 3 | 1 | 40 | 0.0000 | completed | False | 1.8120 |
| EXP-004 | arima-seasonal-diff144 | dev | EXP-003 | arima.seasonal | arima | 1 | 1 | 5259 | 142.4173 |  | 199.1959 |  | 0.2827 | 576 | e9ffdf9f | 2648.6353 | 3 | 1 | 30 | 0.0000 | completed | False | 1.5790 |
| EXP-005 | arima-seasonal-fourier | dev | EXP-003 | arima.seasonal | arima | 1 | 1 | 5259 | 105.8979 |  | 143.6543 |  | 0.2102 | 576 | e9ffdf9f | 2648.6353 | 3 | 2 | 20 | 0.0000 | completed | False | 0.5000 |
| EXP-006 | arima-202-diff144 | dev | EXP-004 | arima.order | arima | 1 | 1 | 5259 | 142.3831 |  | 199.6453 |  | 0.2826 | 576 | e9ffdf9f | 2648.6353 | 3 | 1 | 30 | 0.0000 | completed | False | 1.8130 |
| EXP-007 | arima-order-1-1-1 | dev | EXP-003 | arima.order | arima | 1 | 1 | 5259 | 98.3630 |  | 146.1301 |  | 0.1953 | 576 | e9ffdf9f | 2648.6353 | 3 | 2 | 30 | 0.0000 | completed | False | 1.6410 |
| EXP-008 | arima-order-3-1-3 | dev | EXP-003 | arima.order | arima | 1 | 1 | 5259 | 95.1271 |  | 138.2604 |  | 0.1888 | 576 | e9ffdf9f | 2648.6353 | 3 | 1 | 50 | 0.0000 | completed | False | 0.4530 |
| EXP-009 | arima-order-5-1-5 | dev | EXP-003 | arima.order | arima | 1 | 1 | 5259 | 93.8156 |  | 136.9472 |  | 0.1862 | 576 | e9ffdf9f | 2648.6353 | 3 | 1 | 40 | 0.0000 | completed | False | 2.2190 |
| EXP-010 | lstm-init | dev | EXP-001 | model.type | lstm | 1 | 1 | 5259 | 107.4369 | 1.6576 | 150.2999 | 2.2700 | 0.2133 | 576 | e9ffdf9f | 2648.6353 | 3 | 1 | 30 | 10.0000 | completed | False | 48.4220 |
| EXP-011 | lstm-epochs100 | dev | EXP-010 | lstm.max_epochs | lstm | 1 | 1 | 5259 | 103.0124 | 2.3310 | 145.6739 | 3.0378 | 0.2045 | 576 | e9ffdf9f | 2648.6353 | 3 | 1 | 30 | 10.0000 | completed | False | 62.6570 |
| EXP-012 | lstm-window24 | dev | EXP-011 | lstm.window | lstm | 1 | 1 | 5259 | 114.1215 | 4.4820 | 157.9242 | 5.9282 | 0.2265 | 576 | e9ffdf9f | 2648.6353 | 3 | 1 | 30 | 5.0000 | completed | False | 10.2030 |
| EXP-013 | full-lag144 | full |  |  | naive | 1 | 1 | 5161 | 294.8942 |  | 525.4935 |  | 1.0000 | 1008 | aeb67261 | 3856.3743 | 12 | 6 | 490 | 5.0000 | completed | False | 5.9370 |
| EXP-014 | full-lag1 | full | EXP-013 | naive.lag | naive | 1 | 1 | 5161 | 116.8333 |  | 175.1906 |  | 0.3962 | 1008 | aeb67261 | 3856.3743 | 12 | 7 | 120 | 0.0000 | completed | False | 1.4060 |
| EXP-015 | full-arima-515 | full | EXP-009 | phase | arima | 1 | 1 | 5161 |  |  |  |  |  |  |  |  |  |  |  |  | failed | False | 14.0160 |
| EXP-016 | full-lstm-init-e100 | full | EXP-011 | phase | lstm | 1 | 1 | 5161 | 110.8343 | 3.9510 | 160.0432 | 3.1561 | 0.3758 | 1008 | aeb67261 | 3856.3743 | 12 | 9 | 150 | 10.0000 | completed | False | 117.6560 |
| EXP-017 | full-arima-212 | full | EXP-003 | phase | arima | 1 | 1 | 5161 | 109.5478 |  | 164.5672 |  | 0.3715 | 1008 | aeb67261 | 3856.3743 | 12 | 7 | 170 | 8.0000 | completed | False | 2.2340 |
| EXP-018 | z-lag144-2 | full | EXP-013 | zone_rank | naive | 1 | 2 | 5059 | 213.4478 |  | 307.9601 |  | 1.0000 | 1008 | f89e8093 | 2281.9846 | 8 | 4 | 280 | 10.0000 | completed | False | 1.2970 |
| EXP-019 | z-lag144-3 | full | EXP-013 | zone_rank | naive | 1 | 3 | 5259 | 513.0997 |  | 898.8109 |  | 1.0000 | 1008 | 3adef90e | 2631.7073 | 5 | 1 | 540 | 7.5000 | completed | False | 0.0310 |
| EXP-020 | z-lag1-2 | full | EXP-014 | zone_rank | naive | 1 | 2 | 5059 | 99.9003 |  | 145.7529 |  | 0.4680 | 1008 | f89e8093 | 2281.9846 | 8 | 1 | 80 | 0.0000 | completed | False | 1.2820 |
| EXP-021 | z-lag1-3 | full | EXP-014 | zone_rank | naive | 1 | 3 | 5259 | 89.9451 |  | 132.1991 |  | 0.1753 | 1008 | 3adef90e | 2631.7073 | 5 | 0 | 50 | 0.0000 | completed | False | 0.0310 |
| EXP-022 | z-arima212-2 | full | EXP-017 | zone_rank | arima | 1 | 2 | 5059 | 94.5509 |  | 134.4790 |  | 0.4430 | 1008 | f89e8093 | 2281.9846 | 8 | 1 | 110 | 2.8571 | completed | False | 2.2030 |
| EXP-023 | z-arima212-3 | full | EXP-017 | zone_rank | arima | 1 | 3 | 5259 | 82.6111 |  | 121.8600 |  | 0.1610 | 1008 | 3adef90e | 2631.7073 | 5 | 0 | 50 | 8.0000 | completed | False | 0.6870 |
| EXP-024 | z-lstm-2 | full | EXP-016 | zone_rank | lstm | 1 | 2 | 5059 | 95.2608 | 4.4247 | 137.5885 | 5.8229 | 0.4463 | 1008 | f89e8093 | 2281.9846 | 8 | 3 | 140 | 2.0000 | completed | False | 91.6400 |
| EXP-025 | z-lstm-3 | full | EXP-016 | zone_rank | lstm | 1 | 3 | 5259 | 79.2732 | 0.3231 | 113.6754 | 0.9005 | 0.1545 | 1008 | 3adef90e | 2631.7073 | 5 | 0 | 60 | 8.0000 | completed | False | 140.7030 |
