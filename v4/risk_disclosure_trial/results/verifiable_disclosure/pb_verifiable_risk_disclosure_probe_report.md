# PB x exposure -> verifiable risk disclosure probe

Date: 2026-05-06

Design: `PB_narrow x pre-policy exposure`, firm FE, city-year FE, industry-year FE, city-clustered SE. Strict specification adds `ExposureGroup x YearFE`.

## Coverage

|   year |         n |   has_risk |   verif |   np_verif |
|-------:|----------:|-----------:|--------:|-----------:|
|   2015 | 2290.0000 |     0.8000 | -0.0019 |    -0.0009 |
|   2016 | 2536.0000 |     0.8174 |  0.0091 |    -0.0000 |
|   2017 | 2761.0000 |     0.8254 | -0.0045 |     0.0004 |
|   2018 | 3256.0000 |     0.8409 |  0.0031 |    -0.0068 |
|   2019 | 3314.0000 |     0.8582 | -0.0041 |    -0.0056 |
|   2020 | 3428.0000 |     0.8708 | -0.0136 |    -0.0006 |
|   2021 | 3910.0000 |     0.8997 | -0.0081 |     0.0067 |
|   2022 | 4339.0000 |     0.9168 | -0.0001 |    -0.0015 |
|   2023 | 4662.0000 |     0.9213 |  0.0163 |     0.0052 |

## Decision table

| outcome                            | exposure                  | decision          |    coef |      p | sig   |   coef_strict |   p_strict | sig_strict   |   near_leads_p |   lag1_coef |   lag1_p |   lag2p_coef |   lag2p_p |   n_input | note                                           |
|:-----------------------------------|:--------------------------|:------------------|--------:|-------:|:------|--------------:|-----------:|:-------------|---------------:|------------:|---------:|-------------:|----------:|----------:|:-----------------------------------------------|
| risk_verif_debt_distress_per10k    | failure_cost_pre          | candidate         |  0.9682 | 0.0460 | **    |        1.1274 |     0.0832 | *            |         0.9526 |      0.0634 |   0.9612 |       1.5109 |    0.1729 |     26541 | 主规格、严格趋势控制和近端前趋势同时过关。     |
| np_liability_control_right_per10k  | high_lev_pre              | candidate         |  0.0932 | 0.0518 | *     |        0.1196 |     0.0768 | *            |         0.2092 |      0.7706 |   0.0344 |       0.8213 |    0.0250 |     26541 | 主规格、严格趋势控制和近端前趋势同时过关。     |
| risk_verifiability_index           | high_lev_pre              | candidate         |  0.0675 | 0.0566 | *     |        0.0642 |     0.0809 | *            |         0.3369 |      0.0735 |   0.0072 |       0.1596 |    0.0179 |     26541 | 主规格、严格趋势控制和近端前趋势同时过关。     |
| risk_verif_guarantee_pledge_per10k | ar_controller_pledge_pre  | candidate         |  1.5773 | 0.0872 | *     |        1.6196 |     0.0699 | *            |         0.3466 |      1.3306 |   0.1917 |       1.3035 |    0.1054 |     26541 | 主规格、严格趋势控制和近端前趋势同时过关。     |
| risk_verif_guarantee_pledge_per10k | ar_np_guarantee_pre       | fragile_candidate |  0.6385 | 0.0054 | ***   |        0.3737 |     0.2008 |              |         0.5780 |      0.6192 |   0.0001 |       0.6905 |    0.0265 |     26541 | 主规格和近端前趋势可用，但严格趋势控制未稳住。 |
| risk_verif_debt_distress_per10k    | natural_pre               | fragile_candidate |  0.7552 | 0.0728 | *     |       -0.1226 |     0.8540 |              |         0.6365 |     -0.1586 |   0.8854 |       0.6238 |    0.4332 |     26541 | 主规格和近端前趋势可用，但严格趋势控制未稳住。 |
| risk_verif_legal_process_per10k    | natural_highlev_pre       | base_only         |  0.9242 | 0.0081 | ***   |        0.6751 |     0.0974 | *            |         0.0437 |      0.1430 |   0.6963 |       1.7429 |    0.0754 |     26541 | 只有主规格方向显著，不能直接写。               |
| risk_verifiability_index           | failure_cost_pre          | base_only         |  0.1219 | 0.0226 | **    |        0.1115 |     0.0302 | **           |         0.0010 |      0.0415 |   0.5402 |       0.1903 |    0.0001 |     26541 | 只有主规格方向显著，不能直接写。               |
| risk_verifiability_index           | natural_highlev_pre       | base_only         |  0.0852 | 0.0479 | **    |        0.0491 |     0.2943 |              |         0.0237 |      0.0432 |   0.4311 |       0.1890 |    0.0029 |     26541 | 只有主规格方向显著，不能直接写。               |
| risk_verif_legal_process_per10k    | high_lev_pre              | base_only         |  0.7522 | 0.0636 | *     |        0.7544 |     0.1030 |              |         0.0911 |     -0.1467 |   0.6212 |       1.5050 |    0.0474 |     26541 | 只有主规格方向显著，不能直接写。               |
| risk_verif_legal_process_per10k    | ar_np_guarantee_pre       | no_signal         | -0.6634 | 0.0945 | *     |       -0.6940 |     0.1141 |              |       nan      |    nan      | nan      |     nan      |  nan      |     26541 | 当前口径下没有理论方向一致的可用信号。         |
| risk_verif_guarantee_pledge_per10k | ar_np_liability_index_pre | no_signal         |  2.0077 | 0.1015 |       |        1.6650 |     0.1476 |              |       nan      |    nan      | nan      |     nan      |  nan      |     26541 | 当前口径下没有理论方向一致的可用信号。         |
| risk_verif_legal_process_per10k    | failure_cost_pre          | no_signal         |  1.0202 | 0.1031 |       |        0.7436 |     0.2967 |              |       nan      |    nan      | nan      |     nan      |  nan      |     26541 | 当前口径下没有理论方向一致的可用信号。         |
| risk_verif_guarantee_pledge_per10k | high_lev_pre              | no_signal         |  0.5502 | 0.1212 |       |        0.6982 |     0.0885 | *            |       nan      |    nan      | nan      |     nan      |  nan      |     26541 | 当前口径下没有理论方向一致的可用信号。         |
| risk_verif_debt_distress_per10k    | ar_np_liability_index_pre | no_signal         |  1.1213 | 0.1338 |       |       -0.0550 |     0.9504 |              |       nan      |    nan      | nan      |     nan      |  nan      |     26541 | 当前口径下没有理论方向一致的可用信号。         |
| risk_verif_guarantee_pledge_per10k | ar_np_liability_text_pre  | no_signal         |  0.5666 | 0.1473 |       |        0.5709 |     0.1895 |              |       nan      |    nan      | nan      |     nan      |  nan      |     26541 | 当前口径下没有理论方向一致的可用信号。         |
| risk_verifiability_index           | natural_pre               | no_signal         |  0.0673 | 0.2323 |       |        0.0351 |     0.5408 |              |       nan      |    nan      | nan      |     nan      |  nan      |     26541 | 当前口径下没有理论方向一致的可用信号。         |
| np_liability_control_right_per10k  | ar_controller_pledge_pre  | no_signal         |  0.1286 | 0.2504 |       |       -0.0244 |     0.8777 |              |       nan      |    nan      | nan      |     nan      |  nan      |     26541 | 当前口径下没有理论方向一致的可用信号。         |
| np_liability_control_right_per10k  | ar_np_liability_text_pre  | no_signal         | -0.0678 | 0.2737 |       |       -0.1108 |     0.2887 |              |       nan      |    nan      | nan      |     nan      |  nan      |     26541 | 当前口径下没有理论方向一致的可用信号。         |
| np_liability_control_right_per10k  | ar_np_guarantee_pre       | no_signal         | -0.1123 | 0.3025 |       |       -0.1366 |     0.2662 |              |       nan      |    nan      | nan      |     nan      |  nan      |     26541 | 当前口径下没有理论方向一致的可用信号。         |
| risk_verif_debt_distress_per10k    | high_lev_pre              | no_signal         |  0.4918 | 0.3091 |       |        0.4328 |     0.4622 |              |       nan      |    nan      | nan      |     nan      |  nan      |     26541 | 当前口径下没有理论方向一致的可用信号。         |
| risk_verif_debt_distress_per10k    | ar_controller_pledge_pre  | no_signal         |  0.7520 | 0.3357 |       |        0.7899 |     0.3761 |              |       nan      |    nan      | nan      |     nan      |  nan      |     26541 | 当前口径下没有理论方向一致的可用信号。         |
| risk_verifiability_index           | ar_controller_pledge_pre  | no_signal         |  0.0548 | 0.3604 |       |        0.0535 |     0.3814 |              |       nan      |    nan      | nan      |     nan      |  nan      |     26541 | 当前口径下没有理论方向一致的可用信号。         |
| np_liability_control_right_per10k  | failure_cost_pre          | no_signal         |  0.1295 | 0.3741 |       |        0.1394 |     0.3337 |              |       nan      |    nan      | nan      |     nan      |  nan      |     26541 | 当前口径下没有理论方向一致的可用信号。         |
| risk_verif_debt_distress_per10k    | ar_np_guarantee_pre       | no_signal         |  0.2353 | 0.3897 |       |       -0.1120 |     0.8319 |              |       nan      |    nan      | nan      |     nan      |  nan      |     26541 | 当前口径下没有理论方向一致的可用信号。         |
| risk_verif_guarantee_pledge_per10k | failure_cost_pre          | no_signal         |  0.6504 | 0.4244 |       |        0.7424 |     0.3625 |              |       nan      |    nan      | nan      |     nan      |  nan      |     26541 | 当前口径下没有理论方向一致的可用信号。         |
| risk_verifiability_index           | ar_np_guarantee_pre       | no_signal         | -0.0323 | 0.4258 |       |       -0.0509 |     0.2203 |              |       nan      |    nan      | nan      |     nan      |  nan      |     26541 | 当前口径下没有理论方向一致的可用信号。         |
| risk_verifiability_index           | ar_np_liability_index_pre | no_signal         |  0.0404 | 0.4361 |       |        0.0200 |     0.7397 |              |       nan      |    nan      | nan      |     nan      |  nan      |     26541 | 当前口径下没有理论方向一致的可用信号。         |
| risk_verif_legal_process_per10k    | ar_controller_pledge_pre  | no_signal         |  0.3802 | 0.4606 |       |       -0.0063 |     0.9917 |              |       nan      |    nan      | nan      |     nan      |  nan      |     26541 | 当前口径下没有理论方向一致的可用信号。         |
| risk_verif_debt_distress_per10k    | natural_highlev_pre       | no_signal         |  0.5395 | 0.4624 |       |       -0.3740 |     0.6664 |              |       nan      |    nan      | nan      |     nan      |  nan      |     26541 | 当前口径下没有理论方向一致的可用信号。         |
| risk_verif_debt_distress_per10k    | ar_np_liability_text_pre  | no_signal         | -0.2385 | 0.4668 |       |       -0.0671 |     0.8671 |              |       nan      |    nan      | nan      |     nan      |  nan      |     26541 | 当前口径下没有理论方向一致的可用信号。         |
| risk_verif_guarantee_pledge_per10k | natural_highlev_pre       | no_signal         |  0.1640 | 0.5380 |       |        0.3004 |     0.4298 |              |       nan      |    nan      | nan      |     nan      |  nan      |     26541 | 当前口径下没有理论方向一致的可用信号。         |
| np_liability_verifiability_index   | failure_cost_pre          | no_signal         |  0.0311 | 0.6026 |       |        0.0485 |     0.4633 |              |       nan      |    nan      | nan      |     nan      |  nan      |     26541 | 当前口径下没有理论方向一致的可用信号。         |
| np_liability_control_right_per10k  | natural_pre               | no_signal         | -0.0330 | 0.6475 |       |       -0.0791 |     0.3082 |              |       nan      |    nan      | nan      |     nan      |  nan      |     26541 | 当前口径下没有理论方向一致的可用信号。         |
| np_liability_verifiability_index   | high_lev_pre              | no_signal         |  0.0189 | 0.6575 |       |        0.0561 |     0.2549 |              |       nan      |    nan      | nan      |     nan      |  nan      |     26541 | 当前口径下没有理论方向一致的可用信号。         |
| risk_verif_legal_process_per10k    | natural_pre               | no_signal         |  0.2610 | 0.6760 |       |        0.1667 |     0.7892 |              |       nan      |    nan      | nan      |     nan      |  nan      |     26541 | 当前口径下没有理论方向一致的可用信号。         |
| risk_verif_legal_process_per10k    | ar_np_liability_text_pre  | no_signal         |  0.2115 | 0.7163 |       |       -0.0339 |     0.9539 |              |       nan      |    nan      | nan      |     nan      |  nan      |     26541 | 当前口径下没有理论方向一致的可用信号。         |
| np_liability_verifiability_index   | ar_controller_pledge_pre  | no_signal         |  0.0100 | 0.7595 |       |       -0.0427 |     0.3570 |              |       nan      |    nan      | nan      |     nan      |  nan      |     26541 | 当前口径下没有理论方向一致的可用信号。         |
| np_liability_control_right_per10k  | natural_highlev_pre       | no_signal         |  0.0097 | 0.7893 |       |        0.0363 |     0.6148 |              |       nan      |    nan      | nan      |     nan      |  nan      |     26541 | 当前口径下没有理论方向一致的可用信号。         |
| np_liability_control_right_per10k  | ar_np_liability_index_pre | no_signal         |  0.0204 | 0.8054 |       |       -0.1044 |     0.4554 |              |       nan      |    nan      | nan      |     nan      |  nan      |     26541 | 当前口径下没有理论方向一致的可用信号。         |
| np_liability_verifiability_index   | ar_np_liability_text_pre  | no_signal         | -0.0132 | 0.8071 |       |       -0.0424 |     0.4885 |              |       nan      |    nan      | nan      |     nan      |  nan      |     26541 | 当前口径下没有理论方向一致的可用信号。         |
| risk_verifiability_index           | ar_np_liability_text_pre  | no_signal         |  0.0049 | 0.8286 |       |       -0.0073 |     0.7783 |              |       nan      |    nan      | nan      |     nan      |  nan      |     26541 | 当前口径下没有理论方向一致的可用信号。         |
| np_liability_verifiability_index   | ar_np_guarantee_pre       | no_signal         | -0.0133 | 0.8484 |       |       -0.0389 |     0.5927 |              |       nan      |    nan      | nan      |     nan      |  nan      |     26541 | 当前口径下没有理论方向一致的可用信号。         |
| np_liability_verifiability_index   | natural_highlev_pre       | no_signal         |  0.0034 | 0.9002 |       |        0.0424 |     0.2671 |              |       nan      |    nan      | nan      |     nan      |  nan      |     26541 | 当前口径下没有理论方向一致的可用信号。         |
| risk_verif_legal_process_per10k    | ar_np_liability_index_pre | no_signal         |  0.0713 | 0.9295 |       |       -0.1099 |     0.8982 |              |       nan      |    nan      | nan      |     nan      |  nan      |     26541 | 当前口径下没有理论方向一致的可用信号。         |
| risk_verif_guarantee_pledge_per10k | natural_pre               | no_signal         |  0.0383 | 0.9422 |       |       -0.0865 |     0.8861 |              |       nan      |    nan      | nan      |     nan      |  nan      |     26541 | 当前口径下没有理论方向一致的可用信号。         |
| np_liability_verifiability_index   | ar_np_liability_index_pre | no_signal         |  0.0065 | 0.9439 |       |       -0.0516 |     0.6172 |              |       nan      |    nan      | nan      |     nan      |  nan      |     26541 | 当前口径下没有理论方向一致的可用信号。         |
| np_liability_verifiability_index   | natural_pre               | no_signal         |  0.0027 | 0.9671 |       |       -0.0085 |     0.8974 |              |       nan      |    nan      | nan      |     nan      |  nan      |     26541 | 当前口径下没有理论方向一致的可用信号。         |

## Full base results

| outcome                            | exposure                  | spec                        |    coef |   se_city_cluster |       t |      p | sig   |   n_input |   treated_x_obs |   clusters_city |
|:-----------------------------------|:--------------------------|:----------------------------|--------:|------------------:|--------:|-------:|:------|----------:|----------------:|----------------:|
| risk_verif_guarantee_pledge_per10k | ar_np_guarantee_pre       | main_fe                     |  0.6385 |            0.2274 |  2.8077 | 0.0054 | ***   |     26541 |             444 |             406 |
| risk_verif_legal_process_per10k    | natural_highlev_pre       | main_fe                     |  0.9242 |            0.3463 |  2.6692 | 0.0081 | ***   |     26541 |             400 |             406 |
| risk_verifiability_index           | failure_cost_pre          | main_fe                     |  0.1219 |            0.0531 |  2.2940 | 0.0226 | **    |     26541 |            1210 |             406 |
| risk_verifiability_index           | failure_cost_pre          | plus_exposure_group_year_fe |  0.1115 |            0.0512 |  2.1793 | 0.0302 | **    |     26541 |            1210 |             406 |
| risk_verif_debt_distress_per10k    | failure_cost_pre          | main_fe                     |  0.9682 |            0.4828 |  2.0053 | 0.0460 | **    |     26541 |            1210 |             406 |
| risk_verifiability_index           | natural_highlev_pre       | main_fe                     |  0.0852 |            0.0429 |  1.9876 | 0.0479 | **    |     26541 |             400 |             406 |
| np_liability_control_right_per10k  | high_lev_pre              | main_fe                     |  0.0932 |            0.0477 |  1.9538 | 0.0518 | *     |     26541 |             595 |             406 |
| risk_verifiability_index           | high_lev_pre              | main_fe                     |  0.0675 |            0.0353 |  1.9148 | 0.0566 | *     |     26541 |             595 |             406 |
| risk_verif_legal_process_per10k    | high_lev_pre              | main_fe                     |  0.7522 |            0.4038 |  1.8628 | 0.0636 | *     |     26541 |             595 |             406 |
| risk_verif_guarantee_pledge_per10k | ar_controller_pledge_pre  | plus_exposure_group_year_fe |  1.6196 |            0.8897 |  1.8204 | 0.0699 | *     |     26541 |             380 |             406 |
| risk_verif_debt_distress_per10k    | natural_pre               | main_fe                     |  0.7552 |            0.4191 |  1.8018 | 0.0728 | *     |     26541 |             997 |             406 |
| np_liability_control_right_per10k  | high_lev_pre              | plus_exposure_group_year_fe |  0.1196 |            0.0673 |  1.7769 | 0.0768 | *     |     26541 |             595 |             406 |
| risk_verifiability_index           | high_lev_pre              | plus_exposure_group_year_fe |  0.0642 |            0.0366 |  1.7523 | 0.0809 | *     |     26541 |             595 |             406 |
| risk_verif_debt_distress_per10k    | failure_cost_pre          | plus_exposure_group_year_fe |  1.1274 |            0.6483 |  1.7390 | 0.0832 | *     |     26541 |            1210 |             406 |
| risk_verif_guarantee_pledge_per10k | ar_controller_pledge_pre  | main_fe                     |  1.5773 |            0.9188 |  1.7167 | 0.0872 | *     |     26541 |             380 |             406 |
| risk_verif_guarantee_pledge_per10k | high_lev_pre              | plus_exposure_group_year_fe |  0.6982 |            0.4083 |  1.7101 | 0.0885 | *     |     26541 |             595 |             406 |
| risk_verif_legal_process_per10k    | ar_np_guarantee_pre       | main_fe                     | -0.6634 |            0.3952 | -1.6784 | 0.0945 | *     |     26541 |             444 |             406 |
| risk_verif_legal_process_per10k    | natural_highlev_pre       | plus_exposure_group_year_fe |  0.6751 |            0.4058 |  1.6635 | 0.0974 | *     |     26541 |             400 |             406 |
| risk_verif_guarantee_pledge_per10k | ar_np_liability_index_pre | main_fe                     |  2.0077 |            1.2217 |  1.6434 | 0.1015 |       |     26541 |             686 |             406 |
| risk_verif_legal_process_per10k    | high_lev_pre              | plus_exposure_group_year_fe |  0.7544 |            0.4610 |  1.6366 | 0.1030 |       |     26541 |             595 |             406 |
| risk_verif_legal_process_per10k    | failure_cost_pre          | main_fe                     |  1.0202 |            0.6236 |  1.6360 | 0.1031 |       |     26541 |            1210 |             406 |
| risk_verif_legal_process_per10k    | ar_np_guarantee_pre       | plus_exposure_group_year_fe | -0.6940 |            0.4377 | -1.5854 | 0.1141 |       |     26541 |             444 |             406 |
| risk_verif_guarantee_pledge_per10k | high_lev_pre              | main_fe                     |  0.5502 |            0.3539 |  1.5548 | 0.1212 |       |     26541 |             595 |             406 |
| risk_verif_debt_distress_per10k    | ar_np_liability_index_pre | main_fe                     |  1.1213 |            0.7455 |  1.5040 | 0.1338 |       |     26541 |             686 |             406 |
| risk_verif_guarantee_pledge_per10k | ar_np_liability_text_pre  | main_fe                     |  0.5666 |            0.3898 |  1.4537 | 0.1473 |       |     26541 |             686 |             406 |
| risk_verif_guarantee_pledge_per10k | ar_np_liability_index_pre | plus_exposure_group_year_fe |  1.6650 |            1.1464 |  1.4523 | 0.1476 |       |     26541 |             686 |             406 |
| risk_verif_guarantee_pledge_per10k | ar_np_liability_text_pre  | plus_exposure_group_year_fe |  0.5709 |            0.4339 |  1.3157 | 0.1895 |       |     26541 |             686 |             406 |
| risk_verif_guarantee_pledge_per10k | ar_np_guarantee_pre       | plus_exposure_group_year_fe |  0.3737 |            0.2914 |  1.2826 | 0.2008 |       |     26541 |             444 |             406 |
| risk_verifiability_index           | ar_np_guarantee_pre       | plus_exposure_group_year_fe | -0.0509 |            0.0415 | -1.2287 | 0.2203 |       |     26541 |             444 |             406 |
| risk_verifiability_index           | natural_pre               | main_fe                     |  0.0673 |            0.0562 |  1.1972 | 0.2323 |       |     26541 |             997 |             406 |
| np_liability_control_right_per10k  | ar_controller_pledge_pre  | main_fe                     |  0.1286 |            0.1116 |  1.1521 | 0.2504 |       |     26541 |             380 |             406 |
| np_liability_verifiability_index   | high_lev_pre              | plus_exposure_group_year_fe |  0.0561 |            0.0492 |  1.1411 | 0.2549 |       |     26541 |             595 |             406 |
| np_liability_control_right_per10k  | ar_np_guarantee_pre       | plus_exposure_group_year_fe | -0.1366 |            0.1226 | -1.1143 | 0.2662 |       |     26541 |             444 |             406 |
| np_liability_verifiability_index   | natural_highlev_pre       | plus_exposure_group_year_fe |  0.0424 |            0.0381 |  1.1122 | 0.2671 |       |     26541 |             400 |             406 |
| np_liability_control_right_per10k  | ar_np_liability_text_pre  | main_fe                     | -0.0678 |            0.0618 | -1.0970 | 0.2737 |       |     26541 |             686 |             406 |
| np_liability_control_right_per10k  | ar_np_liability_text_pre  | plus_exposure_group_year_fe | -0.1108 |            0.1042 | -1.0632 | 0.2887 |       |     26541 |             686 |             406 |
| risk_verifiability_index           | natural_highlev_pre       | plus_exposure_group_year_fe |  0.0491 |            0.0467 |  1.0509 | 0.2943 |       |     26541 |             400 |             406 |
| risk_verif_legal_process_per10k    | failure_cost_pre          | plus_exposure_group_year_fe |  0.7436 |            0.7111 |  1.0458 | 0.2967 |       |     26541 |            1210 |             406 |
| np_liability_control_right_per10k  | ar_np_guarantee_pre       | main_fe                     | -0.1123 |            0.1087 | -1.0333 | 0.3025 |       |     26541 |             444 |             406 |
| np_liability_control_right_per10k  | natural_pre               | plus_exposure_group_year_fe | -0.0791 |            0.0775 | -1.0210 | 0.3082 |       |     26541 |             997 |             406 |
| risk_verif_debt_distress_per10k    | high_lev_pre              | main_fe                     |  0.4918 |            0.4825 |  1.0191 | 0.3091 |       |     26541 |             595 |             406 |
| np_liability_control_right_per10k  | failure_cost_pre          | plus_exposure_group_year_fe |  0.1394 |            0.1439 |  0.9685 | 0.3337 |       |     26541 |            1210 |             406 |
| risk_verif_debt_distress_per10k    | ar_controller_pledge_pre  | main_fe                     |  0.7520 |            0.7796 |  0.9646 | 0.3357 |       |     26541 |             380 |             406 |
| np_liability_verifiability_index   | ar_controller_pledge_pre  | plus_exposure_group_year_fe | -0.0427 |            0.0463 | -0.9227 | 0.3570 |       |     26541 |             380 |             406 |
| risk_verifiability_index           | ar_controller_pledge_pre  | main_fe                     |  0.0548 |            0.0599 |  0.9162 | 0.3604 |       |     26541 |             380 |             406 |
| risk_verif_guarantee_pledge_per10k | failure_cost_pre          | plus_exposure_group_year_fe |  0.7424 |            0.8138 |  0.9123 | 0.3625 |       |     26541 |            1210 |             406 |
| np_liability_control_right_per10k  | failure_cost_pre          | main_fe                     |  0.1295 |            0.1454 |  0.8904 | 0.3741 |       |     26541 |            1210 |             406 |
| risk_verif_debt_distress_per10k    | ar_controller_pledge_pre  | plus_exposure_group_year_fe |  0.7899 |            0.8908 |  0.8866 | 0.3761 |       |     26541 |             380 |             406 |
| risk_verifiability_index           | ar_controller_pledge_pre  | plus_exposure_group_year_fe |  0.0535 |            0.0610 |  0.8768 | 0.3814 |       |     26541 |             380 |             406 |
| risk_verif_debt_distress_per10k    | ar_np_guarantee_pre       | main_fe                     |  0.2353 |            0.2731 |  0.8615 | 0.3897 |       |     26541 |             444 |             406 |
| risk_verif_guarantee_pledge_per10k | failure_cost_pre          | main_fe                     |  0.6504 |            0.8129 |  0.8001 | 0.4244 |       |     26541 |            1210 |             406 |
| risk_verifiability_index           | ar_np_guarantee_pre       | main_fe                     | -0.0323 |            0.0405 | -0.7977 | 0.4258 |       |     26541 |             444 |             406 |
| risk_verif_guarantee_pledge_per10k | natural_highlev_pre       | plus_exposure_group_year_fe |  0.3004 |            0.3798 |  0.7908 | 0.4298 |       |     26541 |             400 |             406 |
| risk_verifiability_index           | ar_np_liability_index_pre | main_fe                     |  0.0404 |            0.0518 |  0.7800 | 0.4361 |       |     26541 |             686 |             406 |
| np_liability_control_right_per10k  | ar_np_liability_index_pre | plus_exposure_group_year_fe | -0.1044 |            0.1396 | -0.7475 | 0.4554 |       |     26541 |             686 |             406 |
| risk_verif_legal_process_per10k    | ar_controller_pledge_pre  | main_fe                     |  0.3802 |            0.5146 |  0.7389 | 0.4606 |       |     26541 |             380 |             406 |
| risk_verif_debt_distress_per10k    | high_lev_pre              | plus_exposure_group_year_fe |  0.4328 |            0.5877 |  0.7364 | 0.4622 |       |     26541 |             595 |             406 |
| risk_verif_debt_distress_per10k    | natural_highlev_pre       | main_fe                     |  0.5395 |            0.7330 |  0.7360 | 0.4624 |       |     26541 |             400 |             406 |
| np_liability_verifiability_index   | failure_cost_pre          | plus_exposure_group_year_fe |  0.0485 |            0.0660 |  0.7345 | 0.4633 |       |     26541 |            1210 |             406 |
| risk_verif_debt_distress_per10k    | ar_np_liability_text_pre  | main_fe                     | -0.2385 |            0.3272 | -0.7287 | 0.4668 |       |     26541 |             686 |             406 |
| np_liability_verifiability_index   | ar_np_liability_text_pre  | plus_exposure_group_year_fe | -0.0424 |            0.0611 | -0.6937 | 0.4885 |       |     26541 |             686 |             406 |
| risk_verif_guarantee_pledge_per10k | natural_highlev_pre       | main_fe                     |  0.1640 |            0.2659 |  0.6167 | 0.5380 |       |     26541 |             400 |             406 |
| risk_verifiability_index           | natural_pre               | plus_exposure_group_year_fe |  0.0351 |            0.0573 |  0.6124 | 0.5408 |       |     26541 |             997 |             406 |
| np_liability_verifiability_index   | ar_np_guarantee_pre       | plus_exposure_group_year_fe | -0.0389 |            0.0727 | -0.5356 | 0.5927 |       |     26541 |             444 |             406 |
| np_liability_verifiability_index   | failure_cost_pre          | main_fe                     |  0.0311 |            0.0596 |  0.5213 | 0.6026 |       |     26541 |            1210 |             406 |
| np_liability_control_right_per10k  | natural_highlev_pre       | plus_exposure_group_year_fe |  0.0363 |            0.0721 |  0.5038 | 0.6148 |       |     26541 |             400 |             406 |
| np_liability_verifiability_index   | ar_np_liability_index_pre | plus_exposure_group_year_fe | -0.0516 |            0.1031 | -0.5004 | 0.6172 |       |     26541 |             686 |             406 |
| np_liability_control_right_per10k  | natural_pre               | main_fe                     | -0.0330 |            0.0722 | -0.4578 | 0.6475 |       |     26541 |             997 |             406 |
| np_liability_verifiability_index   | high_lev_pre              | main_fe                     |  0.0189 |            0.0425 |  0.4438 | 0.6575 |       |     26541 |             595 |             406 |
| risk_verif_debt_distress_per10k    | natural_highlev_pre       | plus_exposure_group_year_fe | -0.3740 |            0.8665 | -0.4316 | 0.6664 |       |     26541 |             400 |             406 |
| risk_verif_legal_process_per10k    | natural_pre               | main_fe                     |  0.2610 |            0.6239 |  0.4184 | 0.6760 |       |     26541 |             997 |             406 |
| risk_verif_legal_process_per10k    | ar_np_liability_text_pre  | main_fe                     |  0.2115 |            0.5813 |  0.3638 | 0.7163 |       |     26541 |             686 |             406 |
| risk_verifiability_index           | ar_np_liability_index_pre | plus_exposure_group_year_fe |  0.0200 |            0.0602 |  0.3327 | 0.7397 |       |     26541 |             686 |             406 |
| np_liability_verifiability_index   | ar_controller_pledge_pre  | main_fe                     |  0.0100 |            0.0326 |  0.3065 | 0.7595 |       |     26541 |             380 |             406 |
| risk_verifiability_index           | ar_np_liability_text_pre  | plus_exposure_group_year_fe | -0.0073 |            0.0257 | -0.2819 | 0.7783 |       |     26541 |             686 |             406 |
| risk_verif_legal_process_per10k    | natural_pre               | plus_exposure_group_year_fe |  0.1667 |            0.6230 |  0.2677 | 0.7892 |       |     26541 |             997 |             406 |
| np_liability_control_right_per10k  | natural_highlev_pre       | main_fe                     |  0.0097 |            0.0364 |  0.2675 | 0.7893 |       |     26541 |             400 |             406 |
| np_liability_control_right_per10k  | ar_np_liability_index_pre | main_fe                     |  0.0204 |            0.0826 |  0.2466 | 0.8054 |       |     26541 |             686 |             406 |
| np_liability_verifiability_index   | ar_np_liability_text_pre  | main_fe                     | -0.0132 |            0.0542 | -0.2445 | 0.8071 |       |     26541 |             686 |             406 |
| risk_verifiability_index           | ar_np_liability_text_pre  | main_fe                     |  0.0049 |            0.0225 |  0.2168 | 0.8286 |       |     26541 |             686 |             406 |
| risk_verif_debt_distress_per10k    | ar_np_guarantee_pre       | plus_exposure_group_year_fe | -0.1120 |            0.5272 | -0.2125 | 0.8319 |       |     26541 |             444 |             406 |
| np_liability_verifiability_index   | ar_np_guarantee_pre       | main_fe                     | -0.0133 |            0.0694 | -0.1914 | 0.8484 |       |     26541 |             444 |             406 |
| risk_verif_debt_distress_per10k    | natural_pre               | plus_exposure_group_year_fe | -0.1226 |            0.6658 | -0.1842 | 0.8540 |       |     26541 |             997 |             406 |
| risk_verif_debt_distress_per10k    | ar_np_liability_text_pre  | plus_exposure_group_year_fe | -0.0671 |            0.4006 | -0.1675 | 0.8671 |       |     26541 |             686 |             406 |
| np_liability_control_right_per10k  | ar_controller_pledge_pre  | plus_exposure_group_year_fe | -0.0244 |            0.1583 | -0.1540 | 0.8777 |       |     26541 |             380 |             406 |
| risk_verif_guarantee_pledge_per10k | natural_pre               | plus_exposure_group_year_fe | -0.0865 |            0.6033 | -0.1434 | 0.8861 |       |     26541 |             997 |             406 |
| np_liability_verifiability_index   | natural_pre               | plus_exposure_group_year_fe | -0.0085 |            0.0657 | -0.1290 | 0.8974 |       |     26541 |             997 |             406 |
| risk_verif_legal_process_per10k    | ar_np_liability_index_pre | plus_exposure_group_year_fe | -0.1099 |            0.8587 | -0.1280 | 0.8982 |       |     26541 |             686 |             406 |
| np_liability_verifiability_index   | natural_highlev_pre       | main_fe                     |  0.0034 |            0.0274 |  0.1255 | 0.9002 |       |     26541 |             400 |             406 |
| risk_verif_legal_process_per10k    | ar_np_liability_index_pre | main_fe                     |  0.0713 |            0.8056 |  0.0886 | 0.9295 |       |     26541 |             686 |             406 |
| risk_verif_guarantee_pledge_per10k | natural_pre               | main_fe                     |  0.0383 |            0.5272 |  0.0726 | 0.9422 |       |     26541 |             997 |             406 |
| np_liability_verifiability_index   | ar_np_liability_index_pre | main_fe                     |  0.0065 |            0.0921 |  0.0704 | 0.9439 |       |     26541 |             686 |             406 |
| risk_verif_debt_distress_per10k    | ar_np_liability_index_pre | plus_exposure_group_year_fe | -0.0550 |            0.8832 | -0.0623 | 0.9504 |       |     26541 |             686 |             406 |
| risk_verif_legal_process_per10k    | ar_np_liability_text_pre  | plus_exposure_group_year_fe | -0.0339 |            0.5865 | -0.0579 | 0.9539 |       |     26541 |             686 |             406 |
| np_liability_verifiability_index   | natural_pre               | main_fe                     |  0.0027 |            0.0646 |  0.0413 | 0.9671 |       |     26541 |             997 |             406 |
| risk_verif_legal_process_per10k    | ar_controller_pledge_pre  | plus_exposure_group_year_fe | -0.0063 |            0.6051 | -0.0104 | 0.9917 |       |     26541 |             380 |             406 |

## Event-study summaries

| outcome                            | exposure                 | status   |   n_input |   clusters_city |   near_leads_p |   all_leads_p |   event0_coef |   event0_p |   lag1_coef |   lag1_p |   lag2p_coef |   lag2p_p |
|:-----------------------------------|:-------------------------|:---------|----------:|----------------:|---------------:|--------------:|--------------:|-----------:|------------:|---------:|-------------:|----------:|
| risk_verif_guarantee_pledge_per10k | ar_np_guarantee_pre      | ok       |     26541 |             406 |         0.5780 |        0.1535 |       -0.4236 |     0.5135 |      0.6192 |   0.0001 |       0.6905 |    0.0265 |
| risk_verif_legal_process_per10k    | natural_highlev_pre      | ok       |     26541 |             406 |         0.0437 |        0.0036 |       -0.8548 |     0.0684 |      0.1430 |   0.6963 |       1.7429 |    0.0754 |
| risk_verifiability_index           | failure_cost_pre         | ok       |     26541 |             406 |         0.0010 |        0.0007 |        0.0825 |     0.1404 |      0.0415 |   0.5402 |       0.1903 |    0.0001 |
| risk_verif_debt_distress_per10k    | failure_cost_pre         | ok       |     26541 |             406 |         0.9526 |        0.9528 |        1.7708 |     0.1097 |      0.0634 |   0.9612 |       1.5109 |    0.1729 |
| risk_verifiability_index           | natural_highlev_pre      | ok       |     26541 |             406 |         0.0237 |        0.0182 |        0.0281 |     0.3419 |      0.0432 |   0.4311 |       0.1890 |    0.0029 |
| np_liability_control_right_per10k  | high_lev_pre             | ok       |     26541 |             406 |         0.2092 |        0.0978 |        0.2110 |     0.6652 |      0.7706 |   0.0344 |       0.8213 |    0.0250 |
| risk_verifiability_index           | high_lev_pre             | ok       |     26541 |             406 |         0.3369 |        0.4286 |        0.0262 |     0.2680 |      0.0735 |   0.0072 |       0.1596 |    0.0179 |
| risk_verif_legal_process_per10k    | high_lev_pre             | ok       |     26541 |             406 |         0.0911 |        0.1100 |       -0.0992 |     0.8519 |     -0.1467 |   0.6212 |       1.5050 |    0.0474 |
| risk_verif_debt_distress_per10k    | natural_pre              | ok       |     26541 |             406 |         0.6365 |        0.4585 |        1.5731 |     0.1295 |     -0.1586 |   0.8854 |       0.6238 |    0.4332 |
| risk_verif_guarantee_pledge_per10k | ar_controller_pledge_pre | ok       |     26541 |             406 |         0.3466 |        0.0077 |        0.8689 |     0.3664 |      1.3306 |   0.1917 |       1.3035 |    0.1054 |

## Event coefficients

| outcome                            | exposure                 | event   |   rel_year |    coef |   se_city_cluster |      p | sig   |
|:-----------------------------------|:-------------------------|:--------|-----------:|--------:|------------------:|-------:|:------|
| risk_verif_guarantee_pledge_per10k | ar_np_guarantee_pre      | <=-4    |         -4 | -0.7511 |            0.5119 | 0.1436 |       |
| risk_verif_guarantee_pledge_per10k | ar_np_guarantee_pre      | -3      |         -3 | -0.4332 |            0.4597 | 0.3468 |       |
| risk_verif_guarantee_pledge_per10k | ar_np_guarantee_pre      | -2      |         -2 | -0.1017 |            0.5379 | 0.8502 |       |
| risk_verif_guarantee_pledge_per10k | ar_np_guarantee_pre      | 0       |          0 | -0.4236 |            0.6474 | 0.5135 |       |
| risk_verif_guarantee_pledge_per10k | ar_np_guarantee_pre      | +1      |          1 |  0.6192 |            0.1589 | 0.0001 | ***   |
| risk_verif_guarantee_pledge_per10k | ar_np_guarantee_pre      | >=+2    |          2 |  0.6905 |            0.3094 | 0.0265 | **    |
| risk_verif_legal_process_per10k    | natural_highlev_pre      | <=-4    |         -4 | -0.8954 |            1.0060 | 0.3743 |       |
| risk_verif_legal_process_per10k    | natural_highlev_pre      | -3      |         -3 | -1.3022 |            0.5773 | 0.0250 | **    |
| risk_verif_legal_process_per10k    | natural_highlev_pre      | -2      |         -2 | -0.7985 |            0.6366 | 0.2109 |       |
| risk_verif_legal_process_per10k    | natural_highlev_pre      | 0       |          0 | -0.8548 |            0.4671 | 0.0684 | *     |
| risk_verif_legal_process_per10k    | natural_highlev_pre      | +1      |          1 |  0.1430 |            0.3659 | 0.6963 |       |
| risk_verif_legal_process_per10k    | natural_highlev_pre      | >=+2    |          2 |  1.7429 |            0.9761 | 0.0754 | *     |
| risk_verifiability_index           | failure_cost_pre         | <=-4    |         -4 | -0.0291 |            0.0726 | 0.6890 |       |
| risk_verifiability_index           | failure_cost_pre         | -3      |         -3 | -0.1261 |            0.0751 | 0.0946 | *     |
| risk_verifiability_index           | failure_cost_pre         | -2      |         -2 |  0.0220 |            0.0564 | 0.6967 |       |
| risk_verifiability_index           | failure_cost_pre         | 0       |          0 |  0.0825 |            0.0558 | 0.1404 |       |
| risk_verifiability_index           | failure_cost_pre         | +1      |          1 |  0.0415 |            0.0677 | 0.5402 |       |
| risk_verifiability_index           | failure_cost_pre         | >=+2    |          2 |  0.1903 |            0.0467 | 0.0001 | ***   |
| risk_verif_debt_distress_per10k    | failure_cost_pre         | <=-4    |         -4 |  0.5358 |            1.2637 | 0.6719 |       |
| risk_verif_debt_distress_per10k    | failure_cost_pre         | -3      |         -3 | -0.1848 |            2.0837 | 0.9294 |       |
| risk_verif_debt_distress_per10k    | failure_cost_pre         | -2      |         -2 | -0.5772 |            2.0105 | 0.7743 |       |
| risk_verif_debt_distress_per10k    | failure_cost_pre         | 0       |          0 |  1.7708 |            1.1032 | 0.1097 |       |
| risk_verif_debt_distress_per10k    | failure_cost_pre         | +1      |          1 |  0.0634 |            1.3013 | 0.9612 |       |
| risk_verif_debt_distress_per10k    | failure_cost_pre         | >=+2    |          2 |  1.5109 |            1.1055 | 0.1729 |       |
| risk_verifiability_index           | natural_highlev_pre      | <=-4    |         -4 |  0.0125 |            0.0525 | 0.8127 |       |
| risk_verifiability_index           | natural_highlev_pre      | -3      |         -3 | -0.0996 |            0.0436 | 0.0232 | **    |
| risk_verifiability_index           | natural_highlev_pre      | -2      |         -2 |  0.0124 |            0.0218 | 0.5707 |       |
| risk_verifiability_index           | natural_highlev_pre      | 0       |          0 |  0.0281 |            0.0295 | 0.3419 |       |
| risk_verifiability_index           | natural_highlev_pre      | +1      |          1 |  0.0432 |            0.0548 | 0.4311 |       |
| risk_verifiability_index           | natural_highlev_pre      | >=+2    |          2 |  0.1890 |            0.0628 | 0.0029 | ***   |
| np_liability_control_right_per10k  | high_lev_pre             | <=-4    |         -4 |  0.6713 |            0.4541 | 0.1406 |       |
| np_liability_control_right_per10k  | high_lev_pre             | -3      |         -3 |  0.5248 |            0.4488 | 0.2434 |       |
| np_liability_control_right_per10k  | high_lev_pre             | -2      |         -2 |  0.5313 |            0.4317 | 0.2196 |       |
| np_liability_control_right_per10k  | high_lev_pre             | 0       |          0 |  0.2110 |            0.4869 | 0.6652 |       |
| np_liability_control_right_per10k  | high_lev_pre             | +1      |          1 |  0.7706 |            0.3625 | 0.0344 | **    |
| np_liability_control_right_per10k  | high_lev_pre             | >=+2    |          2 |  0.8213 |            0.3642 | 0.0250 | **    |
| risk_verifiability_index           | high_lev_pre             | <=-4    |         -4 |  0.0202 |            0.0506 | 0.6903 |       |
| risk_verifiability_index           | high_lev_pre             | -3      |         -3 |  0.0117 |            0.0693 | 0.8662 |       |
| risk_verifiability_index           | high_lev_pre             | -2      |         -2 | -0.0134 |            0.0284 | 0.6364 |       |
| risk_verifiability_index           | high_lev_pre             | 0       |          0 |  0.0262 |            0.0236 | 0.2680 |       |
| risk_verifiability_index           | high_lev_pre             | +1      |          1 |  0.0735 |            0.0271 | 0.0072 | ***   |
| risk_verifiability_index           | high_lev_pre             | >=+2    |          2 |  0.1596 |            0.0670 | 0.0179 | **    |
| risk_verif_legal_process_per10k    | high_lev_pre             | <=-4    |         -4 | -0.3462 |            0.5552 | 0.5335 |       |
| risk_verif_legal_process_per10k    | high_lev_pre             | -3      |         -3 | -1.5191 |            0.8154 | 0.0636 | *     |
| risk_verif_legal_process_per10k    | high_lev_pre             | -2      |         -2 | -0.3503 |            0.5202 | 0.5014 |       |
| risk_verif_legal_process_per10k    | high_lev_pre             | 0       |          0 | -0.0992 |            0.5308 | 0.8519 |       |
| risk_verif_legal_process_per10k    | high_lev_pre             | +1      |          1 | -0.1467 |            0.2965 | 0.6212 |       |
| risk_verif_legal_process_per10k    | high_lev_pre             | >=+2    |          2 |  1.5050 |            0.7553 | 0.0474 | **    |
| risk_verif_debt_distress_per10k    | natural_pre              | <=-4    |         -4 |  0.4162 |            1.1590 | 0.7198 |       |
| risk_verif_debt_distress_per10k    | natural_pre              | -3      |         -3 | -0.7605 |            1.1991 | 0.5265 |       |
| risk_verif_debt_distress_per10k    | natural_pre              | -2      |         -2 | -0.5242 |            1.7249 | 0.7615 |       |
| risk_verif_debt_distress_per10k    | natural_pre              | 0       |          0 |  1.5731 |            1.0343 | 0.1295 |       |
| risk_verif_debt_distress_per10k    | natural_pre              | +1      |          1 | -0.1586 |            1.0995 | 0.8854 |       |
| risk_verif_debt_distress_per10k    | natural_pre              | >=+2    |          2 |  0.6238 |            0.7948 | 0.4332 |       |
| risk_verif_guarantee_pledge_per10k | ar_controller_pledge_pre | <=-4    |         -4 | -0.7768 |            0.3434 | 0.0245 | **    |
| risk_verif_guarantee_pledge_per10k | ar_controller_pledge_pre | -3      |         -3 | -0.4856 |            0.4101 | 0.2375 |       |
| risk_verif_guarantee_pledge_per10k | ar_controller_pledge_pre | -2      |         -2 | -0.0691 |            0.4544 | 0.8792 |       |
| risk_verif_guarantee_pledge_per10k | ar_controller_pledge_pre | 0       |          0 |  0.8689 |            0.9602 | 0.3664 |       |
| risk_verif_guarantee_pledge_per10k | ar_controller_pledge_pre | +1      |          1 |  1.3306 |            1.0165 | 0.1917 |       |
| risk_verif_guarantee_pledge_per10k | ar_controller_pledge_pre | >=+2    |          2 |  1.3035 |            0.8020 | 0.1054 |       |