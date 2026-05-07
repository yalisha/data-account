# PB x natural-person liability exposure -> risk disclosure quality probe

Date: 2026-05-06

Design: `PB_narrow x pre-policy exposure`, firm FE, city-year FE, industry-year FE, city-clustered SE. Strict specification adds `ExposureGroup x YearFE`.

## Risk-disclosure coverage

|   year |         n |   has_risk |   risk_chars |   quality |
|-------:|----------:|-----------:|-------------:|----------:|
|   2015 | 2290.0000 |     0.8000 |     385.8485 |   -0.1086 |
|   2016 | 2536.0000 |     0.8174 |     412.3927 |   -0.0708 |
|   2017 | 2761.0000 |     0.8254 |     419.8160 |   -0.0596 |
|   2018 | 3256.0000 |     0.8409 |     411.8698 |   -0.0420 |
|   2019 | 3314.0000 |     0.8582 |     430.1548 |   -0.0306 |
|   2020 | 3428.0000 |     0.8708 |     459.1855 |   -0.0270 |
|   2021 | 3910.0000 |     0.8997 |     471.5174 |    0.0261 |
|   2022 | 4339.0000 |     0.9168 |     486.7686 |    0.0743 |
|   2023 | 4662.0000 |     0.9213 |     533.9247 |    0.1071 |

## Decision table

| sample       | outcome                | exposure            | decision          |    coef |      p | sig   |   coef_strict |   p_strict | sig_strict   |   near_leads_p |   lag1_coef |   lag1_p |   lag2p_coef |   lag2p_p |   n_input | note                                           |
|:-------------|:-----------------------|:--------------------|:------------------|--------:|-------:|:------|--------------:|-----------:|:-------------|---------------:|------------:|---------:|-------------:|----------:|----------:|:-----------------------------------------------|
| all          | risk_debt_per10k       | failure_cost_pre    | base_only         |  2.7976 | 0.0576 | *     |        3.0718 |     0.0402 | **           |       nan      |    nan      | nan      |     nan      |  nan      |     26541 | 只有主规格方向显著，不能直接写。               |
| nonfinancial | risk_debt_per10k       | failure_cost_pre    | base_only         |  2.7976 | 0.0576 | *     |        3.0718 |     0.0402 | **           |       nan      |    nan      | nan      |     nan      |  nan      |     26541 | 只有主规格方向显著，不能直接写。               |
| all          | has_risk_text          | natural_highlev_pre | base_only         |  0.0321 | 0.0810 | *     |        0.0171 |     0.3834 |              |       nan      |    nan      | nan      |     nan      |  nan      |     26541 | 只有主规格方向显著，不能直接写。               |
| nonfinancial | has_risk_text          | natural_highlev_pre | base_only         |  0.0321 | 0.0810 | *     |        0.0171 |     0.3834 |              |       nan      |    nan      | nan      |     nan      |  nan      |     26541 | 只有主规格方向显著，不能直接写。               |
| all          | risk_debt_per10k       | natural_pre         | base_only         |  1.7063 | 0.0956 | *     |        0.6687 |     0.5749 |              |       nan      |    nan      | nan      |     nan      |  nan      |     26541 | 只有主规格方向显著，不能直接写。               |
| nonfinancial | risk_debt_per10k       | natural_pre         | base_only         |  1.7063 | 0.0956 | *     |        0.6687 |     0.5749 |              |       nan      |    nan      | nan      |     nan      |  nan      |     26541 | 只有主规格方向显著，不能直接写。               |
| all          | risk_debt_per10k       | high_lev_pre        | base_only         |  1.3637 | 0.0959 | *     |        1.4606 |     0.0880 | *            |       nan      |    nan      | nan      |     nan      |  nan      |     26541 | 只有主规格方向显著，不能直接写。               |
| nonfinancial | risk_debt_per10k       | high_lev_pre        | base_only         |  1.3637 | 0.0959 | *     |        1.4606 |     0.0880 | *            |       nan      |    nan      | nan      |     nan      |  nan      |     26541 | 只有主规格方向显著，不能直接写。               |
| all          | has_risk_text          | private_pre         | candidate         |  0.0546 | 0.0009 | ***   |        0.0321 |     0.0742 | *            |         0.1671 |      0.0638 |   0.0083 |       0.0466 |    0.3789 |     26541 | 主规格、严格趋势控制和近端前趋势同时过关。     |
| nonfinancial | has_risk_text          | private_pre         | candidate         |  0.0546 | 0.0009 | ***   |        0.0321 |     0.0742 | *            |         0.1671 |      0.0638 |   0.0083 |       0.0466 |    0.3789 |     26541 | 主规格、严格趋势控制和近端前趋势同时过关。     |
| all          | has_risk_text          | private_highlev_pre | candidate         |  0.0492 | 0.0037 | ***   |        0.0353 |     0.0626 | *            |         0.4972 |      0.0431 |   0.0525 |       0.0534 |    0.0407 |     26541 | 主规格、严格趋势控制和近端前趋势同时过关。     |
| nonfinancial | has_risk_text          | private_highlev_pre | candidate         |  0.0492 | 0.0037 | ***   |        0.0353 |     0.0626 | *            |         0.4972 |      0.0431 |   0.0525 |       0.0534 |    0.0407 |     26541 | 主规格、严格趋势控制和近端前趋势同时过关。     |
| all          | risk_specificity_index | high_lev_pre        | candidate         |  0.0781 | 0.0064 | ***   |        0.0737 |     0.0255 | **           |         0.8453 |      0.0788 |   0.0043 |       0.1076 |    0.0465 |     26541 | 主规格、严格趋势控制和近端前趋势同时过关。     |
| nonfinancial | risk_specificity_index | high_lev_pre        | candidate         |  0.0781 | 0.0064 | ***   |        0.0737 |     0.0255 | **           |         0.8453 |      0.0788 |   0.0043 |       0.1076 |    0.0465 |     26541 | 主规格、严格趋势控制和近端前趋势同时过关。     |
| all          | risk_quality_index     | high_lev_pre        | candidate         |  0.0673 | 0.0431 | **    |        0.0622 |     0.0813 | *            |         0.9439 |      0.0755 |   0.1052 |       0.1017 |    0.0504 |     26541 | 主规格、严格趋势控制和近端前趋势同时过关。     |
| nonfinancial | risk_quality_index     | high_lev_pre        | candidate         |  0.0673 | 0.0431 | **    |        0.0622 |     0.0813 | *            |         0.9439 |      0.0755 |   0.1052 |       0.1017 |    0.0504 |     26541 | 主规格、严格趋势控制和近端前趋势同时过关。     |
| all          | risk_chars_ln          | private_pre         | fragile_candidate |  0.2132 | 0.0182 | **    |        0.0808 |     0.4481 |              |         0.3875 |      0.2638 |   0.0797 |       0.1221 |    0.6791 |     26541 | 主规格和近端前趋势可用，但严格趋势控制未稳住。 |
| nonfinancial | risk_chars_ln          | private_pre         | fragile_candidate |  0.2132 | 0.0182 | **    |        0.0808 |     0.4481 |              |         0.3875 |      0.2638 |   0.0797 |       0.1221 |    0.6791 |     26541 | 主规格和近端前趋势可用，但严格趋势控制未稳住。 |
| all          | risk_chars_ln          | private_highlev_pre | fragile_candidate |  0.2425 | 0.0261 | **    |        0.1321 |     0.2549 |              |         0.6382 |      0.1586 |   0.2113 |       0.2500 |    0.1116 |     26541 | 主规格和近端前趋势可用，但严格趋势控制未稳住。 |
| nonfinancial | risk_chars_ln          | private_highlev_pre | fragile_candidate |  0.2425 | 0.0261 | **    |        0.1321 |     0.2549 |              |         0.6382 |      0.1586 |   0.2113 |       0.2500 |    0.1116 |     26541 | 主规格和近端前趋势可用，但严格趋势控制未稳住。 |
| all          | has_risk_text          | failure_cost_pre    | fragile_candidate |  0.0760 | 0.0433 | **    |        0.0539 |     0.1399 |              |         0.8832 |      0.0791 |   0.0007 |       0.0518 |    0.3144 |     26541 | 主规格和近端前趋势可用，但严格趋势控制未稳住。 |
| nonfinancial | has_risk_text          | failure_cost_pre    | fragile_candidate |  0.0760 | 0.0433 | **    |        0.0539 |     0.1399 |              |         0.8832 |      0.0791 |   0.0007 |       0.0518 |    0.3144 |     26541 | 主规格和近端前趋势可用，但严格趋势控制未稳住。 |
| all          | has_risk_text          | natural_pre         | fragile_candidate |  0.0433 | 0.0547 | *     |        0.0164 |     0.4778 |              |         0.6509 |      0.0454 |   0.0316 |       0.0529 |    0.2844 |     26541 | 主规格和近端前趋势可用，但严格趋势控制未稳住。 |
| nonfinancial | has_risk_text          | natural_pre         | fragile_candidate |  0.0433 | 0.0547 | *     |        0.0164 |     0.4778 |              |         0.6509 |      0.0454 |   0.0316 |       0.0529 |    0.2844 |     26541 | 主规格和近端前趋势可用，但严格趋势控制未稳住。 |
| all          | risk_boilerplate_ratio | private_pre         | no_signal         |  0.0061 | 0.0015 | ***   |        0.0045 |     0.0426 | **           |       nan      |    nan      | nan      |     nan      |  nan      |     26541 | 当前口径下没有理论方向一致的可用信号。         |
| nonfinancial | risk_boilerplate_ratio | private_pre         | no_signal         |  0.0061 | 0.0015 | ***   |        0.0045 |     0.0426 | **           |       nan      |    nan      | nan      |     nan      |  nan      |     26541 | 当前口径下没有理论方向一致的可用信号。         |
| all          | risk_boilerplate_ratio | natural_pre         | no_signal         |  0.0044 | 0.0876 | *     |        0.0023 |     0.4034 |              |       nan      |    nan      | nan      |     nan      |  nan      |     26541 | 当前口径下没有理论方向一致的可用信号。         |
| nonfinancial | risk_boilerplate_ratio | natural_pre         | no_signal         |  0.0044 | 0.0876 | *     |        0.0023 |     0.4034 |              |       nan      |    nan      | nan      |     nan      |  nan      |     26541 | 当前口径下没有理论方向一致的可用信号。         |
| all          | risk_boilerplate_ratio | high_lev_pre        | no_signal         | -0.0026 | 0.1003 |       |       -0.0021 |     0.2922 |              |       nan      |    nan      | nan      |     nan      |  nan      |     26541 | 当前口径下没有理论方向一致的可用信号。         |
| nonfinancial | risk_boilerplate_ratio | high_lev_pre        | no_signal         | -0.0026 | 0.1003 |       |       -0.0021 |     0.2922 |              |       nan      |    nan      | nan      |     nan      |  nan      |     26541 | 当前口径下没有理论方向一致的可用信号。         |
| all          | has_risk_text          | high_lev_pre        | no_signal         |  0.0222 | 0.1216 |       |        0.0190 |     0.2768 |              |       nan      |    nan      | nan      |     nan      |  nan      |     26541 | 当前口径下没有理论方向一致的可用信号。         |
| nonfinancial | has_risk_text          | high_lev_pre        | no_signal         |  0.0222 | 0.1216 |       |        0.0190 |     0.2768 |              |       nan      |    nan      | nan      |     nan      |  nan      |     26541 | 当前口径下没有理论方向一致的可用信号。         |
| all          | risk_chars_ln          | natural_pre         | no_signal         |  0.1928 | 0.1261 |       |        0.0281 |     0.8366 |              |       nan      |    nan      | nan      |     nan      |  nan      |     26541 | 当前口径下没有理论方向一致的可用信号。         |
| nonfinancial | risk_chars_ln          | natural_pre         | no_signal         |  0.1928 | 0.1261 |       |        0.0281 |     0.8366 |              |       nan      |    nan      | nan      |     nan      |  nan      |     26541 | 当前口径下没有理论方向一致的可用信号。         |
| all          | risk_specificity_index | private_pre         | no_signal         | -0.0476 | 0.1346 |       |       -0.0679 |     0.0581 | *            |       nan      |    nan      | nan      |     nan      |  nan      |     26541 | 当前口径下没有理论方向一致的可用信号。         |
| nonfinancial | risk_specificity_index | private_pre         | no_signal         | -0.0476 | 0.1346 |       |       -0.0679 |     0.0581 | *            |       nan      |    nan      | nan      |     nan      |  nan      |     26541 | 当前口径下没有理论方向一致的可用信号。         |
| all          | risk_chars_ln          | failure_cost_pre    | no_signal         |  0.3336 | 0.1409 |       |        0.1784 |     0.4267 |              |       nan      |    nan      | nan      |     nan      |  nan      |     26541 | 当前口径下没有理论方向一致的可用信号。         |
| nonfinancial | risk_chars_ln          | failure_cost_pre    | no_signal         |  0.3336 | 0.1409 |       |        0.1784 |     0.4267 |              |       nan      |    nan      | nan      |     nan      |  nan      |     26541 | 当前口径下没有理论方向一致的可用信号。         |
| all          | risk_chars_ln          | natural_highlev_pre | no_signal         |  0.1676 | 0.1621 |       |        0.0528 |     0.6663 |              |       nan      |    nan      | nan      |     nan      |  nan      |     26541 | 当前口径下没有理论方向一致的可用信号。         |
| nonfinancial | risk_chars_ln          | natural_highlev_pre | no_signal         |  0.1676 | 0.1621 |       |        0.0528 |     0.6663 |              |       nan      |    nan      | nan      |     nan      |  nan      |     26541 | 当前口径下没有理论方向一致的可用信号。         |
| all          | risk_boilerplate_ratio | failure_cost_pre    | no_signal         |  0.0034 | 0.1748 |       |        0.0035 |     0.1767 |              |       nan      |    nan      | nan      |     nan      |  nan      |     26541 | 当前口径下没有理论方向一致的可用信号。         |
| nonfinancial | risk_boilerplate_ratio | failure_cost_pre    | no_signal         |  0.0034 | 0.1748 |       |        0.0035 |     0.1767 |              |       nan      |    nan      | nan      |     nan      |  nan      |     26541 | 当前口径下没有理论方向一致的可用信号。         |
| all          | risk_debt_per10k       | private_pre         | no_signal         |  0.9012 | 0.1960 |       |       -0.1360 |     0.8810 |              |       nan      |    nan      | nan      |     nan      |  nan      |     26541 | 当前口径下没有理论方向一致的可用信号。         |
| nonfinancial | risk_debt_per10k       | private_pre         | no_signal         |  0.9012 | 0.1960 |       |       -0.1360 |     0.8810 |              |       nan      |    nan      | nan      |     nan      |  nan      |     26541 | 当前口径下没有理论方向一致的可用信号。         |
| all          | risk_chars_ln          | high_lev_pre        | no_signal         |  0.1095 | 0.2316 |       |        0.0773 |     0.4493 |              |       nan      |    nan      | nan      |     nan      |  nan      |     26541 | 当前口径下没有理论方向一致的可用信号。         |
| nonfinancial | risk_chars_ln          | high_lev_pre        | no_signal         |  0.1095 | 0.2316 |       |        0.0773 |     0.4493 |              |       nan      |    nan      | nan      |     nan      |  nan      |     26541 | 当前口径下没有理论方向一致的可用信号。         |
| all          | risk_detail_per10k     | high_lev_pre        | no_signal         | 18.8394 | 0.2601 |       |       19.6863 |     0.2545 |              |       nan      |    nan      | nan      |     nan      |  nan      |     26541 | 当前口径下没有理论方向一致的可用信号。         |
| nonfinancial | risk_detail_per10k     | high_lev_pre        | no_signal         | 18.8394 | 0.2601 |       |       19.6863 |     0.2545 |              |       nan      |    nan      | nan      |     nan      |  nan      |     26541 | 当前口径下没有理论方向一致的可用信号。         |
| all          | risk_quality_index     | private_pre         | no_signal         | -0.0273 | 0.2787 |       |       -0.0587 |     0.0761 | *            |       nan      |    nan      | nan      |     nan      |  nan      |     26541 | 当前口径下没有理论方向一致的可用信号。         |
| nonfinancial | risk_quality_index     | private_pre         | no_signal         | -0.0273 | 0.2787 |       |       -0.0587 |     0.0761 | *            |       nan      |    nan      | nan      |     nan      |  nan      |     26541 | 当前口径下没有理论方向一致的可用信号。         |
| all          | risk_debt_per10k       | natural_highlev_pre | no_signal         |  1.4239 | 0.2862 |       |        0.6770 |     0.6371 |              |       nan      |    nan      | nan      |     nan      |  nan      |     26541 | 当前口径下没有理论方向一致的可用信号。         |
| nonfinancial | risk_debt_per10k       | natural_highlev_pre | no_signal         |  1.4239 | 0.2862 |       |        0.6770 |     0.6371 |              |       nan      |    nan      | nan      |     nan      |  nan      |     26541 | 当前口径下没有理论方向一致的可用信号。         |
| all          | risk_specificity_index | natural_pre         | no_signal         | -0.0398 | 0.3012 |       |       -0.0645 |     0.1055 |              |       nan      |    nan      | nan      |     nan      |  nan      |     26541 | 当前口径下没有理论方向一致的可用信号。         |
| nonfinancial | risk_specificity_index | natural_pre         | no_signal         | -0.0398 | 0.3012 |       |       -0.0645 |     0.1055 |              |       nan      |    nan      | nan      |     nan      |  nan      |     26541 | 当前口径下没有理论方向一致的可用信号。         |
| all          | risk_quality_index     | private_highlev_pre | no_signal         |  0.0417 | 0.3073 |       |        0.0178 |     0.6807 |              |       nan      |    nan      | nan      |     nan      |  nan      |     26541 | 当前口径下没有理论方向一致的可用信号。         |
| nonfinancial | risk_quality_index     | private_highlev_pre | no_signal         |  0.0417 | 0.3073 |       |        0.0178 |     0.6807 |              |       nan      |    nan      | nan      |     nan      |  nan      |     26541 | 当前口径下没有理论方向一致的可用信号。         |
| all          | risk_debt_per10k       | private_highlev_pre | no_signal         |  0.9453 | 0.3953 |       |        0.1632 |     0.8940 |              |       nan      |    nan      | nan      |     nan      |  nan      |     26541 | 当前口径下没有理论方向一致的可用信号。         |
| nonfinancial | risk_debt_per10k       | private_highlev_pre | no_signal         |  0.9453 | 0.3953 |       |        0.1632 |     0.8940 |              |       nan      |    nan      | nan      |     nan      |  nan      |     26541 | 当前口径下没有理论方向一致的可用信号。         |
| all          | risk_specificity_index | private_highlev_pre | no_signal         |  0.0279 | 0.3962 |       |        0.0089 |     0.8074 |              |       nan      |    nan      | nan      |     nan      |  nan      |     26541 | 当前口径下没有理论方向一致的可用信号。         |
| nonfinancial | risk_specificity_index | private_highlev_pre | no_signal         |  0.0279 | 0.3962 |       |        0.0089 |     0.8074 |              |       nan      |    nan      | nan      |     nan      |  nan      |     26541 | 当前口径下没有理论方向一致的可用信号。         |
| all          | risk_quality_index     | natural_pre         | no_signal         | -0.0281 | 0.3990 |       |       -0.0661 |     0.0700 | *            |       nan      |    nan      | nan      |     nan      |  nan      |     26541 | 当前口径下没有理论方向一致的可用信号。         |
| nonfinancial | risk_quality_index     | natural_pre         | no_signal         | -0.0281 | 0.3990 |       |       -0.0661 |     0.0700 | *            |       nan      |    nan      | nan      |     nan      |  nan      |     26541 | 当前口径下没有理论方向一致的可用信号。         |
| all          | risk_category_count    | natural_pre         | no_signal         | -0.0620 | 0.4074 |       |       -0.1988 |     0.0350 | **           |       nan      |    nan      | nan      |     nan      |  nan      |     26541 | 当前口径下没有理论方向一致的可用信号。         |
| nonfinancial | risk_category_count    | natural_pre         | no_signal         | -0.0620 | 0.4074 |       |       -0.1988 |     0.0350 | **           |       nan      |    nan      | nan      |     nan      |  nan      |     26541 | 当前口径下没有理论方向一致的可用信号。         |
| all          | risk_boilerplate_ratio | private_highlev_pre | no_signal         |  0.0015 | 0.4688 |       |        0.0016 |     0.5070 |              |       nan      |    nan      | nan      |     nan      |  nan      |     26541 | 当前口径下没有理论方向一致的可用信号。         |
| nonfinancial | risk_boilerplate_ratio | private_highlev_pre | no_signal         |  0.0015 | 0.4688 |       |        0.0016 |     0.5070 |              |       nan      |    nan      | nan      |     nan      |  nan      |     26541 | 当前口径下没有理论方向一致的可用信号。         |
| all          | risk_detail_per10k     | private_highlev_pre | no_signal         | 11.8323 | 0.5187 |       |       11.3050 |     0.5506 |              |       nan      |    nan      | nan      |     nan      |  nan      |     26541 | 当前口径下没有理论方向一致的可用信号。         |
| nonfinancial | risk_detail_per10k     | private_highlev_pre | no_signal         | 11.8323 | 0.5187 |       |       11.3050 |     0.5506 |              |       nan      |    nan      | nan      |     nan      |  nan      |     26541 | 当前口径下没有理论方向一致的可用信号。         |
| all          | risk_quality_index     | failure_cost_pre    | no_signal         |  0.0359 | 0.5645 |       |        0.0071 |     0.9150 |              |       nan      |    nan      | nan      |     nan      |  nan      |     26541 | 当前口径下没有理论方向一致的可用信号。         |
| nonfinancial | risk_quality_index     | failure_cost_pre    | no_signal         |  0.0359 | 0.5645 |       |        0.0071 |     0.9150 |              |       nan      |    nan      | nan      |     nan      |  nan      |     26541 | 当前口径下没有理论方向一致的可用信号。         |
| all          | risk_quality_index     | natural_highlev_pre | no_signal         |  0.0236 | 0.5952 |       |       -0.0068 |     0.8783 |              |       nan      |    nan      | nan      |     nan      |  nan      |     26541 | 当前口径下没有理论方向一致的可用信号。         |
| nonfinancial | risk_quality_index     | natural_highlev_pre | no_signal         |  0.0236 | 0.5952 |       |       -0.0068 |     0.8783 |              |       nan      |    nan      | nan      |     nan      |  nan      |     26541 | 当前口径下没有理论方向一致的可用信号。         |
| all          | risk_category_count    | private_highlev_pre | no_signal         |  0.0663 | 0.6478 |       |        0.0541 |     0.7151 |              |       nan      |    nan      | nan      |     nan      |  nan      |     26541 | 当前口径下没有理论方向一致的可用信号。         |
| nonfinancial | risk_category_count    | private_highlev_pre | no_signal         |  0.0663 | 0.6478 |       |        0.0541 |     0.7151 |              |       nan      |    nan      | nan      |     nan      |  nan      |     26541 | 当前口径下没有理论方向一致的可用信号。         |
| all          | risk_detail_per10k     | failure_cost_pre    | no_signal         | 12.3868 | 0.6493 |       |       11.9844 |     0.6471 |              |       nan      |    nan      | nan      |     nan      |  nan      |     26541 | 当前口径下没有理论方向一致的可用信号。         |
| nonfinancial | risk_detail_per10k     | failure_cost_pre    | no_signal         | 12.3868 | 0.6493 |       |       11.9844 |     0.6471 |              |       nan      |    nan      | nan      |     nan      |  nan      |     26541 | 当前口径下没有理论方向一致的可用信号。         |
| all          | risk_specificity_index | natural_highlev_pre | no_signal         |  0.0154 | 0.6626 |       |       -0.0084 |     0.8225 |              |       nan      |    nan      | nan      |     nan      |  nan      |     26541 | 当前口径下没有理论方向一致的可用信号。         |
| nonfinancial | risk_specificity_index | natural_highlev_pre | no_signal         |  0.0154 | 0.6626 |       |       -0.0084 |     0.8225 |              |       nan      |    nan      | nan      |     nan      |  nan      |     26541 | 当前口径下没有理论方向一致的可用信号。         |
| all          | risk_category_count    | high_lev_pre        | no_signal         |  0.0466 | 0.6745 |       |        0.0702 |     0.5144 |              |       nan      |    nan      | nan      |     nan      |  nan      |     26541 | 当前口径下没有理论方向一致的可用信号。         |
| nonfinancial | risk_category_count    | high_lev_pre        | no_signal         |  0.0466 | 0.6745 |       |        0.0702 |     0.5144 |              |       nan      |    nan      | nan      |     nan      |  nan      |     26541 | 当前口径下没有理论方向一致的可用信号。         |
| all          | risk_detail_per10k     | natural_highlev_pre | no_signal         |  7.7227 | 0.6832 |       |        5.9650 |     0.7605 |              |       nan      |    nan      | nan      |     nan      |  nan      |     26541 | 当前口径下没有理论方向一致的可用信号。         |
| nonfinancial | risk_detail_per10k     | natural_highlev_pre | no_signal         |  7.7227 | 0.6832 |       |        5.9650 |     0.7605 |              |       nan      |    nan      | nan      |     nan      |  nan      |     26541 | 当前口径下没有理论方向一致的可用信号。         |
| all          | risk_category_count    | private_pre         | no_signal         |  0.0187 | 0.7181 |       |       -0.0956 |     0.2724 |              |       nan      |    nan      | nan      |     nan      |  nan      |     26541 | 当前口径下没有理论方向一致的可用信号。         |
| nonfinancial | risk_category_count    | private_pre         | no_signal         |  0.0187 | 0.7181 |       |       -0.0956 |     0.2724 |              |       nan      |    nan      | nan      |     nan      |  nan      |     26541 | 当前口径下没有理论方向一致的可用信号。         |
| all          | risk_specificity_index | failure_cost_pre    | no_signal         |  0.0221 | 0.7201 |       |        0.0090 |     0.8906 |              |       nan      |    nan      | nan      |     nan      |  nan      |     26541 | 当前口径下没有理论方向一致的可用信号。         |
| nonfinancial | risk_specificity_index | failure_cost_pre    | no_signal         |  0.0221 | 0.7201 |       |        0.0090 |     0.8906 |              |       nan      |    nan      | nan      |     nan      |  nan      |     26541 | 当前口径下没有理论方向一致的可用信号。         |
| all          | risk_category_count    | failure_cost_pre    | no_signal         |  0.0468 | 0.7548 |       |        0.0192 |     0.8971 |              |       nan      |    nan      | nan      |     nan      |  nan      |     26541 | 当前口径下没有理论方向一致的可用信号。         |
| nonfinancial | risk_category_count    | failure_cost_pre    | no_signal         |  0.0468 | 0.7548 |       |        0.0192 |     0.8971 |              |       nan      |    nan      | nan      |     nan      |  nan      |     26541 | 当前口径下没有理论方向一致的可用信号。         |
| all          | risk_detail_per10k     | natural_pre         | no_signal         |  4.5227 | 0.8077 |       |       -2.2494 |     0.9062 |              |       nan      |    nan      | nan      |     nan      |  nan      |     26541 | 当前口径下没有理论方向一致的可用信号。         |
| nonfinancial | risk_detail_per10k     | natural_pre         | no_signal         |  4.5227 | 0.8077 |       |       -2.2494 |     0.9062 |              |       nan      |    nan      | nan      |     nan      |  nan      |     26541 | 当前口径下没有理论方向一致的可用信号。         |
| all          | risk_detail_per10k     | private_pre         | no_signal         | -1.7427 | 0.8959 |       |       -8.3245 |     0.5692 |              |       nan      |    nan      | nan      |     nan      |  nan      |     26541 | 当前口径下没有理论方向一致的可用信号。         |
| nonfinancial | risk_detail_per10k     | private_pre         | no_signal         | -1.7427 | 0.8959 |       |       -8.3245 |     0.5692 |              |       nan      |    nan      | nan      |     nan      |  nan      |     26541 | 当前口径下没有理论方向一致的可用信号。         |
| all          | risk_boilerplate_ratio | natural_highlev_pre | no_signal         |  0.0002 | 0.9237 |       |        0.0006 |     0.8304 |              |       nan      |    nan      | nan      |     nan      |  nan      |     26541 | 当前口径下没有理论方向一致的可用信号。         |
| nonfinancial | risk_boilerplate_ratio | natural_highlev_pre | no_signal         |  0.0002 | 0.9237 |       |        0.0006 |     0.8304 |              |       nan      |    nan      | nan      |     nan      |  nan      |     26541 | 当前口径下没有理论方向一致的可用信号。         |
| all          | risk_category_count    | natural_highlev_pre | no_signal         | -0.0118 | 0.9453 |       |       -0.0436 |     0.7932 |              |       nan      |    nan      | nan      |     nan      |  nan      |     26541 | 当前口径下没有理论方向一致的可用信号。         |
| nonfinancial | risk_category_count    | natural_highlev_pre | no_signal         | -0.0118 | 0.9453 |       |       -0.0436 |     0.7932 |              |       nan      |    nan      | nan      |     nan      |  nan      |     26541 | 当前口径下没有理论方向一致的可用信号。         |

## Full base results

| sample       | outcome                | exposure            | spec                        |    coef |   se_city_cluster |       t |      p | sig   |   n_input |   treated_x_obs |   clusters_city |
|:-------------|:-----------------------|:--------------------|:----------------------------|--------:|------------------:|--------:|-------:|:------|----------:|----------------:|----------------:|
| all          | has_risk_text          | private_pre         | main_fe                     |  0.0546 |            0.0163 |  3.3482 | 0.0009 | ***   |     26541 |             972 |             406 |
| nonfinancial | has_risk_text          | private_pre         | main_fe                     |  0.0546 |            0.0163 |  3.3482 | 0.0009 | ***   |     26541 |             972 |             406 |
| all          | risk_boilerplate_ratio | private_pre         | main_fe                     |  0.0061 |            0.0019 |  3.2118 | 0.0015 | ***   |     26541 |             972 |             406 |
| nonfinancial | risk_boilerplate_ratio | private_pre         | main_fe                     |  0.0061 |            0.0019 |  3.2118 | 0.0015 | ***   |     26541 |             972 |             406 |
| all          | has_risk_text          | private_highlev_pre | main_fe                     |  0.0492 |            0.0168 |  2.9325 | 0.0037 | ***   |     26541 |             410 |             406 |
| nonfinancial | has_risk_text          | private_highlev_pre | main_fe                     |  0.0492 |            0.0168 |  2.9325 | 0.0037 | ***   |     26541 |             410 |             406 |
| all          | risk_specificity_index | high_lev_pre        | main_fe                     |  0.0781 |            0.0284 |  2.7485 | 0.0064 | ***   |     26541 |             595 |             406 |
| nonfinancial | risk_specificity_index | high_lev_pre        | main_fe                     |  0.0781 |            0.0284 |  2.7485 | 0.0064 | ***   |     26541 |             595 |             406 |
| all          | risk_chars_ln          | private_pre         | main_fe                     |  0.2132 |            0.0897 |  2.3774 | 0.0182 | **    |     26541 |             972 |             406 |
| nonfinancial | risk_chars_ln          | private_pre         | main_fe                     |  0.2132 |            0.0897 |  2.3774 | 0.0182 | **    |     26541 |             972 |             406 |
| all          | risk_specificity_index | high_lev_pre        | plus_exposure_group_year_fe |  0.0737 |            0.0328 |  2.2464 | 0.0255 | **    |     26541 |             595 |             406 |
| nonfinancial | risk_specificity_index | high_lev_pre        | plus_exposure_group_year_fe |  0.0737 |            0.0328 |  2.2464 | 0.0255 | **    |     26541 |             595 |             406 |
| all          | risk_chars_ln          | private_highlev_pre | main_fe                     |  0.2425 |            0.1084 |  2.2374 | 0.0261 | **    |     26541 |             410 |             406 |
| nonfinancial | risk_chars_ln          | private_highlev_pre | main_fe                     |  0.2425 |            0.1084 |  2.2374 | 0.0261 | **    |     26541 |             410 |             406 |
| all          | risk_category_count    | natural_pre         | plus_exposure_group_year_fe | -0.1988 |            0.0938 | -2.1197 | 0.0350 | **    |     26541 |             997 |             406 |
| nonfinancial | risk_category_count    | natural_pre         | plus_exposure_group_year_fe | -0.1988 |            0.0938 | -2.1197 | 0.0350 | **    |     26541 |             997 |             406 |
| all          | risk_debt_per10k       | failure_cost_pre    | plus_exposure_group_year_fe |  3.0718 |            1.4894 |  2.0625 | 0.0402 | **    |     26541 |            1210 |             406 |
| nonfinancial | risk_debt_per10k       | failure_cost_pre    | plus_exposure_group_year_fe |  3.0718 |            1.4894 |  2.0625 | 0.0402 | **    |     26541 |            1210 |             406 |
| all          | risk_boilerplate_ratio | private_pre         | plus_exposure_group_year_fe |  0.0045 |            0.0022 |  2.0381 | 0.0426 | **    |     26541 |             972 |             406 |
| nonfinancial | risk_boilerplate_ratio | private_pre         | plus_exposure_group_year_fe |  0.0045 |            0.0022 |  2.0381 | 0.0426 | **    |     26541 |             972 |             406 |
| all          | risk_quality_index     | high_lev_pre        | main_fe                     |  0.0673 |            0.0331 |  2.0330 | 0.0431 | **    |     26541 |             595 |             406 |
| nonfinancial | risk_quality_index     | high_lev_pre        | main_fe                     |  0.0673 |            0.0331 |  2.0330 | 0.0431 | **    |     26541 |             595 |             406 |
| all          | has_risk_text          | failure_cost_pre    | main_fe                     |  0.0760 |            0.0374 |  2.0305 | 0.0433 | **    |     26541 |            1210 |             406 |
| nonfinancial | has_risk_text          | failure_cost_pre    | main_fe                     |  0.0760 |            0.0374 |  2.0305 | 0.0433 | **    |     26541 |            1210 |             406 |
| all          | has_risk_text          | natural_pre         | main_fe                     |  0.0433 |            0.0224 |  1.9304 | 0.0547 | *     |     26541 |             997 |             406 |
| nonfinancial | has_risk_text          | natural_pre         | main_fe                     |  0.0433 |            0.0224 |  1.9304 | 0.0547 | *     |     26541 |             997 |             406 |
| all          | risk_debt_per10k       | failure_cost_pre    | main_fe                     |  2.7976 |            1.4670 |  1.9071 | 0.0576 | *     |     26541 |            1210 |             406 |
| nonfinancial | risk_debt_per10k       | failure_cost_pre    | main_fe                     |  2.7976 |            1.4670 |  1.9071 | 0.0576 | *     |     26541 |            1210 |             406 |
| all          | risk_specificity_index | private_pre         | plus_exposure_group_year_fe | -0.0679 |            0.0357 | -1.9037 | 0.0581 | *     |     26541 |             972 |             406 |
| nonfinancial | risk_specificity_index | private_pre         | plus_exposure_group_year_fe | -0.0679 |            0.0357 | -1.9037 | 0.0581 | *     |     26541 |             972 |             406 |
| all          | has_risk_text          | private_highlev_pre | plus_exposure_group_year_fe |  0.0353 |            0.0189 |  1.8699 | 0.0626 | *     |     26541 |             410 |             406 |
| nonfinancial | has_risk_text          | private_highlev_pre | plus_exposure_group_year_fe |  0.0353 |            0.0189 |  1.8699 | 0.0626 | *     |     26541 |             410 |             406 |
| all          | risk_quality_index     | natural_pre         | plus_exposure_group_year_fe | -0.0661 |            0.0363 | -1.8198 | 0.0700 | *     |     26541 |             997 |             406 |
| nonfinancial | risk_quality_index     | natural_pre         | plus_exposure_group_year_fe | -0.0661 |            0.0363 | -1.8198 | 0.0700 | *     |     26541 |             997 |             406 |
| all          | has_risk_text          | private_pre         | plus_exposure_group_year_fe |  0.0321 |            0.0179 |  1.7927 | 0.0742 | *     |     26541 |             972 |             406 |
| nonfinancial | has_risk_text          | private_pre         | plus_exposure_group_year_fe |  0.0321 |            0.0179 |  1.7927 | 0.0742 | *     |     26541 |             972 |             406 |
| all          | risk_quality_index     | private_pre         | plus_exposure_group_year_fe | -0.0587 |            0.0330 | -1.7813 | 0.0761 | *     |     26541 |             972 |             406 |
| nonfinancial | risk_quality_index     | private_pre         | plus_exposure_group_year_fe | -0.0587 |            0.0330 | -1.7813 | 0.0761 | *     |     26541 |             972 |             406 |
| all          | has_risk_text          | natural_highlev_pre | main_fe                     |  0.0321 |            0.0183 |  1.7518 | 0.0810 | *     |     26541 |             400 |             406 |
| nonfinancial | has_risk_text          | natural_highlev_pre | main_fe                     |  0.0321 |            0.0183 |  1.7518 | 0.0810 | *     |     26541 |             400 |             406 |
| all          | risk_quality_index     | high_lev_pre        | plus_exposure_group_year_fe |  0.0622 |            0.0355 |  1.7500 | 0.0813 | *     |     26541 |             595 |             406 |
| nonfinancial | risk_quality_index     | high_lev_pre        | plus_exposure_group_year_fe |  0.0622 |            0.0355 |  1.7500 | 0.0813 | *     |     26541 |             595 |             406 |
| all          | risk_boilerplate_ratio | natural_pre         | main_fe                     |  0.0044 |            0.0025 |  1.7151 | 0.0876 | *     |     26541 |             997 |             406 |
| nonfinancial | risk_boilerplate_ratio | natural_pre         | main_fe                     |  0.0044 |            0.0025 |  1.7151 | 0.0876 | *     |     26541 |             997 |             406 |
| all          | risk_debt_per10k       | high_lev_pre        | plus_exposure_group_year_fe |  1.4606 |            0.8530 |  1.7124 | 0.0880 | *     |     26541 |             595 |             406 |
| nonfinancial | risk_debt_per10k       | high_lev_pre        | plus_exposure_group_year_fe |  1.4606 |            0.8530 |  1.7124 | 0.0880 | *     |     26541 |             595 |             406 |
| all          | risk_debt_per10k       | natural_pre         | main_fe                     |  1.7063 |            1.0201 |  1.6727 | 0.0956 | *     |     26541 |             997 |             406 |
| nonfinancial | risk_debt_per10k       | natural_pre         | main_fe                     |  1.7063 |            1.0201 |  1.6727 | 0.0956 | *     |     26541 |             997 |             406 |
| all          | risk_debt_per10k       | high_lev_pre        | main_fe                     |  1.3637 |            0.8160 |  1.6712 | 0.0959 | *     |     26541 |             595 |             406 |
| nonfinancial | risk_debt_per10k       | high_lev_pre        | main_fe                     |  1.3637 |            0.8160 |  1.6712 | 0.0959 | *     |     26541 |             595 |             406 |
| all          | risk_boilerplate_ratio | high_lev_pre        | main_fe                     | -0.0026 |            0.0015 | -1.6493 | 0.1003 |       |     26541 |             595 |             406 |
| nonfinancial | risk_boilerplate_ratio | high_lev_pre        | main_fe                     | -0.0026 |            0.0015 | -1.6493 | 0.1003 |       |     26541 |             595 |             406 |
| all          | risk_specificity_index | natural_pre         | plus_exposure_group_year_fe | -0.0645 |            0.0397 | -1.6246 | 0.1055 |       |     26541 |             997 |             406 |
| nonfinancial | risk_specificity_index | natural_pre         | plus_exposure_group_year_fe | -0.0645 |            0.0397 | -1.6246 | 0.1055 |       |     26541 |             997 |             406 |
| all          | has_risk_text          | high_lev_pre        | main_fe                     |  0.0222 |            0.0143 |  1.5532 | 0.1216 |       |     26541 |             595 |             406 |
| nonfinancial | has_risk_text          | high_lev_pre        | main_fe                     |  0.0222 |            0.0143 |  1.5532 | 0.1216 |       |     26541 |             595 |             406 |
| all          | risk_chars_ln          | natural_pre         | main_fe                     |  0.1928 |            0.1256 |  1.5348 | 0.1261 |       |     26541 |             997 |             406 |
| nonfinancial | risk_chars_ln          | natural_pre         | main_fe                     |  0.1928 |            0.1256 |  1.5348 | 0.1261 |       |     26541 |             997 |             406 |
| all          | risk_specificity_index | private_pre         | main_fe                     | -0.0476 |            0.0317 | -1.5009 | 0.1346 |       |     26541 |             972 |             406 |
| nonfinancial | risk_specificity_index | private_pre         | main_fe                     | -0.0476 |            0.0317 | -1.5009 | 0.1346 |       |     26541 |             972 |             406 |
| all          | has_risk_text          | failure_cost_pre    | plus_exposure_group_year_fe |  0.0539 |            0.0364 |  1.4809 | 0.1399 |       |     26541 |            1210 |             406 |
| nonfinancial | has_risk_text          | failure_cost_pre    | plus_exposure_group_year_fe |  0.0539 |            0.0364 |  1.4809 | 0.1399 |       |     26541 |            1210 |             406 |
| all          | risk_chars_ln          | failure_cost_pre    | main_fe                     |  0.3336 |            0.2259 |  1.4770 | 0.1409 |       |     26541 |            1210 |             406 |
| nonfinancial | risk_chars_ln          | failure_cost_pre    | main_fe                     |  0.3336 |            0.2259 |  1.4770 | 0.1409 |       |     26541 |            1210 |             406 |
| all          | risk_chars_ln          | natural_highlev_pre | main_fe                     |  0.1676 |            0.1195 |  1.4023 | 0.1621 |       |     26541 |             400 |             406 |
| nonfinancial | risk_chars_ln          | natural_highlev_pre | main_fe                     |  0.1676 |            0.1195 |  1.4023 | 0.1621 |       |     26541 |             400 |             406 |
| all          | risk_boilerplate_ratio | failure_cost_pre    | main_fe                     |  0.0034 |            0.0025 |  1.3606 | 0.1748 |       |     26541 |            1210 |             406 |
| nonfinancial | risk_boilerplate_ratio | failure_cost_pre    | main_fe                     |  0.0034 |            0.0025 |  1.3606 | 0.1748 |       |     26541 |            1210 |             406 |
| all          | risk_boilerplate_ratio | failure_cost_pre    | plus_exposure_group_year_fe |  0.0035 |            0.0026 |  1.3547 | 0.1767 |       |     26541 |            1210 |             406 |
| nonfinancial | risk_boilerplate_ratio | failure_cost_pre    | plus_exposure_group_year_fe |  0.0035 |            0.0026 |  1.3547 | 0.1767 |       |     26541 |            1210 |             406 |
| all          | risk_debt_per10k       | private_pre         | main_fe                     |  0.9012 |            0.6951 |  1.2965 | 0.1960 |       |     26541 |             972 |             406 |
| nonfinancial | risk_debt_per10k       | private_pre         | main_fe                     |  0.9012 |            0.6951 |  1.2965 | 0.1960 |       |     26541 |             972 |             406 |
| all          | risk_chars_ln          | high_lev_pre        | main_fe                     |  0.1095 |            0.0913 |  1.1991 | 0.2316 |       |     26541 |             595 |             406 |
| nonfinancial | risk_chars_ln          | high_lev_pre        | main_fe                     |  0.1095 |            0.0913 |  1.1991 | 0.2316 |       |     26541 |             595 |             406 |
| all          | risk_detail_per10k     | high_lev_pre        | plus_exposure_group_year_fe | 19.6863 |           17.2363 |  1.1421 | 0.2545 |       |     26541 |             595 |             406 |
| nonfinancial | risk_detail_per10k     | high_lev_pre        | plus_exposure_group_year_fe | 19.6863 |           17.2363 |  1.1421 | 0.2545 |       |     26541 |             595 |             406 |
| all          | risk_chars_ln          | private_highlev_pre | plus_exposure_group_year_fe |  0.1321 |            0.1157 |  1.1411 | 0.2549 |       |     26541 |             410 |             406 |
| nonfinancial | risk_chars_ln          | private_highlev_pre | plus_exposure_group_year_fe |  0.1321 |            0.1157 |  1.1411 | 0.2549 |       |     26541 |             410 |             406 |
| all          | risk_detail_per10k     | high_lev_pre        | main_fe                     | 18.8394 |           16.6927 |  1.1286 | 0.2601 |       |     26541 |             595 |             406 |
| nonfinancial | risk_detail_per10k     | high_lev_pre        | main_fe                     | 18.8394 |           16.6927 |  1.1286 | 0.2601 |       |     26541 |             595 |             406 |
| all          | risk_category_count    | private_pre         | plus_exposure_group_year_fe | -0.0956 |            0.0869 | -1.1000 | 0.2724 |       |     26541 |             972 |             406 |
| nonfinancial | risk_category_count    | private_pre         | plus_exposure_group_year_fe | -0.0956 |            0.0869 | -1.1000 | 0.2724 |       |     26541 |             972 |             406 |
| all          | has_risk_text          | high_lev_pre        | plus_exposure_group_year_fe |  0.0190 |            0.0174 |  1.0898 | 0.2768 |       |     26541 |             595 |             406 |
| nonfinancial | has_risk_text          | high_lev_pre        | plus_exposure_group_year_fe |  0.0190 |            0.0174 |  1.0898 | 0.2768 |       |     26541 |             595 |             406 |
| all          | risk_quality_index     | private_pre         | main_fe                     | -0.0273 |            0.0251 | -1.0855 | 0.2787 |       |     26541 |             972 |             406 |
| nonfinancial | risk_quality_index     | private_pre         | main_fe                     | -0.0273 |            0.0251 | -1.0855 | 0.2787 |       |     26541 |             972 |             406 |
| all          | risk_debt_per10k       | natural_highlev_pre | main_fe                     |  1.4239 |            1.3324 |  1.0687 | 0.2862 |       |     26541 |             400 |             406 |
| nonfinancial | risk_debt_per10k       | natural_highlev_pre | main_fe                     |  1.4239 |            1.3324 |  1.0687 | 0.2862 |       |     26541 |             400 |             406 |
| all          | risk_boilerplate_ratio | high_lev_pre        | plus_exposure_group_year_fe | -0.0021 |            0.0020 | -1.0554 | 0.2922 |       |     26541 |             595 |             406 |
| nonfinancial | risk_boilerplate_ratio | high_lev_pre        | plus_exposure_group_year_fe | -0.0021 |            0.0020 | -1.0554 | 0.2922 |       |     26541 |             595 |             406 |
| all          | risk_specificity_index | natural_pre         | main_fe                     | -0.0398 |            0.0384 | -1.0359 | 0.3012 |       |     26541 |             997 |             406 |
| nonfinancial | risk_specificity_index | natural_pre         | main_fe                     | -0.0398 |            0.0384 | -1.0359 | 0.3012 |       |     26541 |             997 |             406 |
| all          | risk_quality_index     | private_highlev_pre | main_fe                     |  0.0417 |            0.0408 |  1.0230 | 0.3073 |       |     26541 |             410 |             406 |
| nonfinancial | risk_quality_index     | private_highlev_pre | main_fe                     |  0.0417 |            0.0408 |  1.0230 | 0.3073 |       |     26541 |             410 |             406 |
| all          | has_risk_text          | natural_highlev_pre | plus_exposure_group_year_fe |  0.0171 |            0.0195 |  0.8731 | 0.3834 |       |     26541 |             400 |             406 |
| nonfinancial | has_risk_text          | natural_highlev_pre | plus_exposure_group_year_fe |  0.0171 |            0.0195 |  0.8731 | 0.3834 |       |     26541 |             400 |             406 |
| all          | risk_debt_per10k       | private_highlev_pre | main_fe                     |  0.9453 |            1.1102 |  0.8514 | 0.3953 |       |     26541 |             410 |             406 |
| nonfinancial | risk_debt_per10k       | private_highlev_pre | main_fe                     |  0.9453 |            1.1102 |  0.8514 | 0.3953 |       |     26541 |             410 |             406 |
| all          | risk_specificity_index | private_highlev_pre | main_fe                     |  0.0279 |            0.0328 |  0.8499 | 0.3962 |       |     26541 |             410 |             406 |
| nonfinancial | risk_specificity_index | private_highlev_pre | main_fe                     |  0.0279 |            0.0328 |  0.8499 | 0.3962 |       |     26541 |             410 |             406 |
| all          | risk_quality_index     | natural_pre         | main_fe                     | -0.0281 |            0.0332 | -0.8448 | 0.3990 |       |     26541 |             997 |             406 |
| nonfinancial | risk_quality_index     | natural_pre         | main_fe                     | -0.0281 |            0.0332 | -0.8448 | 0.3990 |       |     26541 |             997 |             406 |
| all          | risk_boilerplate_ratio | natural_pre         | plus_exposure_group_year_fe |  0.0023 |            0.0028 |  0.8370 | 0.4034 |       |     26541 |             997 |             406 |
| nonfinancial | risk_boilerplate_ratio | natural_pre         | plus_exposure_group_year_fe |  0.0023 |            0.0028 |  0.8370 | 0.4034 |       |     26541 |             997 |             406 |
| all          | risk_category_count    | natural_pre         | main_fe                     | -0.0620 |            0.0748 | -0.8299 | 0.4074 |       |     26541 |             997 |             406 |
| nonfinancial | risk_category_count    | natural_pre         | main_fe                     | -0.0620 |            0.0748 | -0.8299 | 0.4074 |       |     26541 |             997 |             406 |
| all          | risk_chars_ln          | failure_cost_pre    | plus_exposure_group_year_fe |  0.1784 |            0.2241 |  0.7961 | 0.4267 |       |     26541 |            1210 |             406 |
| nonfinancial | risk_chars_ln          | failure_cost_pre    | plus_exposure_group_year_fe |  0.1784 |            0.2241 |  0.7961 | 0.4267 |       |     26541 |            1210 |             406 |
| all          | risk_chars_ln          | private_pre         | plus_exposure_group_year_fe |  0.0808 |            0.1064 |  0.7597 | 0.4481 |       |     26541 |             972 |             406 |
| nonfinancial | risk_chars_ln          | private_pre         | plus_exposure_group_year_fe |  0.0808 |            0.1064 |  0.7597 | 0.4481 |       |     26541 |             972 |             406 |
| all          | risk_chars_ln          | high_lev_pre        | plus_exposure_group_year_fe |  0.0773 |            0.1020 |  0.7578 | 0.4493 |       |     26541 |             595 |             406 |
| nonfinancial | risk_chars_ln          | high_lev_pre        | plus_exposure_group_year_fe |  0.0773 |            0.1020 |  0.7578 | 0.4493 |       |     26541 |             595 |             406 |
| all          | risk_boilerplate_ratio | private_highlev_pre | main_fe                     |  0.0015 |            0.0020 |  0.7256 | 0.4688 |       |     26541 |             410 |             406 |
| nonfinancial | risk_boilerplate_ratio | private_highlev_pre | main_fe                     |  0.0015 |            0.0020 |  0.7256 | 0.4688 |       |     26541 |             410 |             406 |
| all          | has_risk_text          | natural_pre         | plus_exposure_group_year_fe |  0.0164 |            0.0231 |  0.7109 | 0.4778 |       |     26541 |             997 |             406 |
| nonfinancial | has_risk_text          | natural_pre         | plus_exposure_group_year_fe |  0.0164 |            0.0231 |  0.7109 | 0.4778 |       |     26541 |             997 |             406 |
| all          | risk_boilerplate_ratio | private_highlev_pre | plus_exposure_group_year_fe |  0.0016 |            0.0025 |  0.6645 | 0.5070 |       |     26541 |             410 |             406 |
| nonfinancial | risk_boilerplate_ratio | private_highlev_pre | plus_exposure_group_year_fe |  0.0016 |            0.0025 |  0.6645 | 0.5070 |       |     26541 |             410 |             406 |
| all          | risk_category_count    | high_lev_pre        | plus_exposure_group_year_fe |  0.0702 |            0.1075 |  0.6529 | 0.5144 |       |     26541 |             595 |             406 |
| nonfinancial | risk_category_count    | high_lev_pre        | plus_exposure_group_year_fe |  0.0702 |            0.1075 |  0.6529 | 0.5144 |       |     26541 |             595 |             406 |
| all          | risk_detail_per10k     | private_highlev_pre | main_fe                     | 11.8323 |           18.3108 |  0.6462 | 0.5187 |       |     26541 |             410 |             406 |
| nonfinancial | risk_detail_per10k     | private_highlev_pre | main_fe                     | 11.8323 |           18.3108 |  0.6462 | 0.5187 |       |     26541 |             410 |             406 |
| all          | risk_detail_per10k     | private_highlev_pre | plus_exposure_group_year_fe | 11.3050 |           18.9154 |  0.5977 | 0.5506 |       |     26541 |             410 |             406 |
| nonfinancial | risk_detail_per10k     | private_highlev_pre | plus_exposure_group_year_fe | 11.3050 |           18.9154 |  0.5977 | 0.5506 |       |     26541 |             410 |             406 |
| all          | risk_quality_index     | failure_cost_pre    | main_fe                     |  0.0359 |            0.0622 |  0.5769 | 0.5645 |       |     26541 |            1210 |             406 |
| nonfinancial | risk_quality_index     | failure_cost_pre    | main_fe                     |  0.0359 |            0.0622 |  0.5769 | 0.5645 |       |     26541 |            1210 |             406 |
| all          | risk_detail_per10k     | private_pre         | plus_exposure_group_year_fe | -8.3245 |           14.6051 | -0.5700 | 0.5692 |       |     26541 |             972 |             406 |
| nonfinancial | risk_detail_per10k     | private_pre         | plus_exposure_group_year_fe | -8.3245 |           14.6051 | -0.5700 | 0.5692 |       |     26541 |             972 |             406 |
| all          | risk_debt_per10k       | natural_pre         | plus_exposure_group_year_fe |  0.6687 |            1.1907 |  0.5616 | 0.5749 |       |     26541 |             997 |             406 |
| nonfinancial | risk_debt_per10k       | natural_pre         | plus_exposure_group_year_fe |  0.6687 |            1.1907 |  0.5616 | 0.5749 |       |     26541 |             997 |             406 |
| all          | risk_quality_index     | natural_highlev_pre | main_fe                     |  0.0236 |            0.0443 |  0.5320 | 0.5952 |       |     26541 |             400 |             406 |
| nonfinancial | risk_quality_index     | natural_highlev_pre | main_fe                     |  0.0236 |            0.0443 |  0.5320 | 0.5952 |       |     26541 |             400 |             406 |
| all          | risk_debt_per10k       | natural_highlev_pre | plus_exposure_group_year_fe |  0.6770 |            1.4335 |  0.4723 | 0.6371 |       |     26541 |             400 |             406 |
| nonfinancial | risk_debt_per10k       | natural_highlev_pre | plus_exposure_group_year_fe |  0.6770 |            1.4335 |  0.4723 | 0.6371 |       |     26541 |             400 |             406 |
| all          | risk_detail_per10k     | failure_cost_pre    | plus_exposure_group_year_fe | 11.9844 |           26.1469 |  0.4583 | 0.6471 |       |     26541 |            1210 |             406 |
| nonfinancial | risk_detail_per10k     | failure_cost_pre    | plus_exposure_group_year_fe | 11.9844 |           26.1469 |  0.4583 | 0.6471 |       |     26541 |            1210 |             406 |
| all          | risk_category_count    | private_highlev_pre | main_fe                     |  0.0663 |            0.1449 |  0.4574 | 0.6478 |       |     26541 |             410 |             406 |
| nonfinancial | risk_category_count    | private_highlev_pre | main_fe                     |  0.0663 |            0.1449 |  0.4574 | 0.6478 |       |     26541 |             410 |             406 |
| all          | risk_detail_per10k     | failure_cost_pre    | main_fe                     | 12.3868 |           27.2083 |  0.4553 | 0.6493 |       |     26541 |            1210 |             406 |
| nonfinancial | risk_detail_per10k     | failure_cost_pre    | main_fe                     | 12.3868 |           27.2083 |  0.4553 | 0.6493 |       |     26541 |            1210 |             406 |
| all          | risk_specificity_index | natural_highlev_pre | main_fe                     |  0.0154 |            0.0351 |  0.4369 | 0.6626 |       |     26541 |             400 |             406 |
| nonfinancial | risk_specificity_index | natural_highlev_pre | main_fe                     |  0.0154 |            0.0351 |  0.4369 | 0.6626 |       |     26541 |             400 |             406 |
| all          | risk_chars_ln          | natural_highlev_pre | plus_exposure_group_year_fe |  0.0528 |            0.1223 |  0.4318 | 0.6663 |       |     26541 |             400 |             406 |
| nonfinancial | risk_chars_ln          | natural_highlev_pre | plus_exposure_group_year_fe |  0.0528 |            0.1223 |  0.4318 | 0.6663 |       |     26541 |             400 |             406 |
| all          | risk_category_count    | high_lev_pre        | main_fe                     |  0.0466 |            0.1107 |  0.4205 | 0.6745 |       |     26541 |             595 |             406 |
| nonfinancial | risk_category_count    | high_lev_pre        | main_fe                     |  0.0466 |            0.1107 |  0.4205 | 0.6745 |       |     26541 |             595 |             406 |
| all          | risk_quality_index     | private_highlev_pre | plus_exposure_group_year_fe |  0.0178 |            0.0433 |  0.4120 | 0.6807 |       |     26541 |             410 |             406 |
| nonfinancial | risk_quality_index     | private_highlev_pre | plus_exposure_group_year_fe |  0.0178 |            0.0433 |  0.4120 | 0.6807 |       |     26541 |             410 |             406 |
| all          | risk_detail_per10k     | natural_highlev_pre | main_fe                     |  7.7227 |           18.9045 |  0.4085 | 0.6832 |       |     26541 |             400 |             406 |
| nonfinancial | risk_detail_per10k     | natural_highlev_pre | main_fe                     |  7.7227 |           18.9045 |  0.4085 | 0.6832 |       |     26541 |             400 |             406 |
| all          | risk_category_count    | private_highlev_pre | plus_exposure_group_year_fe |  0.0541 |            0.1480 |  0.3654 | 0.7151 |       |     26541 |             410 |             406 |
| nonfinancial | risk_category_count    | private_highlev_pre | plus_exposure_group_year_fe |  0.0541 |            0.1480 |  0.3654 | 0.7151 |       |     26541 |             410 |             406 |
| all          | risk_category_count    | private_pre         | main_fe                     |  0.0187 |            0.0517 |  0.3614 | 0.7181 |       |     26541 |             972 |             406 |
| nonfinancial | risk_category_count    | private_pre         | main_fe                     |  0.0187 |            0.0517 |  0.3614 | 0.7181 |       |     26541 |             972 |             406 |
| all          | risk_specificity_index | failure_cost_pre    | main_fe                     |  0.0221 |            0.0615 |  0.3587 | 0.7201 |       |     26541 |            1210 |             406 |
| nonfinancial | risk_specificity_index | failure_cost_pre    | main_fe                     |  0.0221 |            0.0615 |  0.3587 | 0.7201 |       |     26541 |            1210 |             406 |
| all          | risk_category_count    | failure_cost_pre    | main_fe                     |  0.0468 |            0.1495 |  0.3127 | 0.7548 |       |     26541 |            1210 |             406 |
| nonfinancial | risk_category_count    | failure_cost_pre    | main_fe                     |  0.0468 |            0.1495 |  0.3127 | 0.7548 |       |     26541 |            1210 |             406 |
| all          | risk_detail_per10k     | natural_highlev_pre | plus_exposure_group_year_fe |  5.9650 |           19.5459 |  0.3052 | 0.7605 |       |     26541 |             400 |             406 |
| nonfinancial | risk_detail_per10k     | natural_highlev_pre | plus_exposure_group_year_fe |  5.9650 |           19.5459 |  0.3052 | 0.7605 |       |     26541 |             400 |             406 |
| all          | risk_category_count    | natural_highlev_pre | plus_exposure_group_year_fe | -0.0436 |            0.1663 | -0.2624 | 0.7932 |       |     26541 |             400 |             406 |
| nonfinancial | risk_category_count    | natural_highlev_pre | plus_exposure_group_year_fe | -0.0436 |            0.1663 | -0.2624 | 0.7932 |       |     26541 |             400 |             406 |
| all          | risk_specificity_index | private_highlev_pre | plus_exposure_group_year_fe |  0.0089 |            0.0364 |  0.2440 | 0.8074 |       |     26541 |             410 |             406 |
| nonfinancial | risk_specificity_index | private_highlev_pre | plus_exposure_group_year_fe |  0.0089 |            0.0364 |  0.2440 | 0.8074 |       |     26541 |             410 |             406 |
| all          | risk_detail_per10k     | natural_pre         | main_fe                     |  4.5227 |           18.5669 |  0.2436 | 0.8077 |       |     26541 |             997 |             406 |
| nonfinancial | risk_detail_per10k     | natural_pre         | main_fe                     |  4.5227 |           18.5669 |  0.2436 | 0.8077 |       |     26541 |             997 |             406 |
| all          | risk_specificity_index | natural_highlev_pre | plus_exposure_group_year_fe | -0.0084 |            0.0375 | -0.2245 | 0.8225 |       |     26541 |             400 |             406 |
| nonfinancial | risk_specificity_index | natural_highlev_pre | plus_exposure_group_year_fe | -0.0084 |            0.0375 | -0.2245 | 0.8225 |       |     26541 |             400 |             406 |
| all          | risk_boilerplate_ratio | natural_highlev_pre | plus_exposure_group_year_fe |  0.0006 |            0.0026 |  0.2144 | 0.8304 |       |     26541 |             400 |             406 |
| nonfinancial | risk_boilerplate_ratio | natural_highlev_pre | plus_exposure_group_year_fe |  0.0006 |            0.0026 |  0.2144 | 0.8304 |       |     26541 |             400 |             406 |
| all          | risk_chars_ln          | natural_pre         | plus_exposure_group_year_fe |  0.0281 |            0.1363 |  0.2064 | 0.8366 |       |     26541 |             997 |             406 |
| nonfinancial | risk_chars_ln          | natural_pre         | plus_exposure_group_year_fe |  0.0281 |            0.1363 |  0.2064 | 0.8366 |       |     26541 |             997 |             406 |
| all          | risk_quality_index     | natural_highlev_pre | plus_exposure_group_year_fe | -0.0068 |            0.0442 | -0.1533 | 0.8783 |       |     26541 |             400 |             406 |
| nonfinancial | risk_quality_index     | natural_highlev_pre | plus_exposure_group_year_fe | -0.0068 |            0.0442 | -0.1533 | 0.8783 |       |     26541 |             400 |             406 |
| all          | risk_debt_per10k       | private_pre         | plus_exposure_group_year_fe | -0.1360 |            0.9082 | -0.1498 | 0.8810 |       |     26541 |             972 |             406 |
| nonfinancial | risk_debt_per10k       | private_pre         | plus_exposure_group_year_fe | -0.1360 |            0.9082 | -0.1498 | 0.8810 |       |     26541 |             972 |             406 |
| all          | risk_specificity_index | failure_cost_pre    | plus_exposure_group_year_fe |  0.0090 |            0.0653 |  0.1376 | 0.8906 |       |     26541 |            1210 |             406 |
| nonfinancial | risk_specificity_index | failure_cost_pre    | plus_exposure_group_year_fe |  0.0090 |            0.0653 |  0.1376 | 0.8906 |       |     26541 |            1210 |             406 |
| all          | risk_debt_per10k       | private_highlev_pre | plus_exposure_group_year_fe |  0.1632 |            1.2237 |  0.1334 | 0.8940 |       |     26541 |             410 |             406 |
| nonfinancial | risk_debt_per10k       | private_highlev_pre | plus_exposure_group_year_fe |  0.1632 |            1.2237 |  0.1334 | 0.8940 |       |     26541 |             410 |             406 |
| all          | risk_detail_per10k     | private_pre         | main_fe                     | -1.7427 |           13.3076 | -0.1310 | 0.8959 |       |     26541 |             972 |             406 |
| nonfinancial | risk_detail_per10k     | private_pre         | main_fe                     | -1.7427 |           13.3076 | -0.1310 | 0.8959 |       |     26541 |             972 |             406 |
| all          | risk_category_count    | failure_cost_pre    | plus_exposure_group_year_fe |  0.0192 |            0.1481 |  0.1295 | 0.8971 |       |     26541 |            1210 |             406 |
| nonfinancial | risk_category_count    | failure_cost_pre    | plus_exposure_group_year_fe |  0.0192 |            0.1481 |  0.1295 | 0.8971 |       |     26541 |            1210 |             406 |
| all          | risk_detail_per10k     | natural_pre         | plus_exposure_group_year_fe | -2.2494 |           19.0802 | -0.1179 | 0.9062 |       |     26541 |             997 |             406 |
| nonfinancial | risk_detail_per10k     | natural_pre         | plus_exposure_group_year_fe | -2.2494 |           19.0802 | -0.1179 | 0.9062 |       |     26541 |             997 |             406 |
| all          | risk_quality_index     | failure_cost_pre    | plus_exposure_group_year_fe |  0.0071 |            0.0669 |  0.1068 | 0.9150 |       |     26541 |            1210 |             406 |
| nonfinancial | risk_quality_index     | failure_cost_pre    | plus_exposure_group_year_fe |  0.0071 |            0.0669 |  0.1068 | 0.9150 |       |     26541 |            1210 |             406 |
| all          | risk_boilerplate_ratio | natural_highlev_pre | main_fe                     |  0.0002 |            0.0021 |  0.0959 | 0.9237 |       |     26541 |             400 |             406 |
| nonfinancial | risk_boilerplate_ratio | natural_highlev_pre | main_fe                     |  0.0002 |            0.0021 |  0.0959 | 0.9237 |       |     26541 |             400 |             406 |
| all          | risk_category_count    | natural_highlev_pre | main_fe                     | -0.0118 |            0.1713 | -0.0687 | 0.9453 |       |     26541 |             400 |             406 |
| nonfinancial | risk_category_count    | natural_highlev_pre | main_fe                     | -0.0118 |            0.1713 | -0.0687 | 0.9453 |       |     26541 |             400 |             406 |

## Event-study summaries

| sample       | outcome                | exposure            | status   |   n_input |   clusters_city |   near_leads_p |   all_leads_p |   event0_coef |   event0_p |   lag1_coef |   lag1_p |   lag2p_coef |   lag2p_p |
|:-------------|:-----------------------|:--------------------|:---------|----------:|----------------:|---------------:|--------------:|--------------:|-----------:|------------:|---------:|-------------:|----------:|
| all          | has_risk_text          | private_pre         | ok       |     26541 |             406 |         0.1671 |        0.2081 |        0.0815 |     0.0015 |      0.0638 |   0.0083 |       0.0466 |    0.3789 |
| nonfinancial | has_risk_text          | private_pre         | ok       |     26541 |             406 |         0.1671 |        0.2081 |        0.0815 |     0.0015 |      0.0638 |   0.0083 |       0.0466 |    0.3789 |
| all          | has_risk_text          | private_highlev_pre | ok       |     26541 |             406 |         0.4972 |        0.6730 |        0.0688 |     0.0000 |      0.0431 |   0.0525 |       0.0534 |    0.0407 |
| nonfinancial | has_risk_text          | private_highlev_pre | ok       |     26541 |             406 |         0.4972 |        0.6730 |        0.0688 |     0.0000 |      0.0431 |   0.0525 |       0.0534 |    0.0407 |
| all          | risk_specificity_index | high_lev_pre        | ok       |     26541 |             406 |         0.8453 |        0.7020 |        0.0430 |     0.1081 |      0.0788 |   0.0043 |       0.1076 |    0.0465 |
| nonfinancial | risk_specificity_index | high_lev_pre        | ok       |     26541 |             406 |         0.8453 |        0.7020 |        0.0430 |     0.1081 |      0.0788 |   0.0043 |       0.1076 |    0.0465 |
| all          | risk_chars_ln          | private_pre         | ok       |     26541 |             406 |         0.3875 |        0.5515 |        0.3977 |     0.0040 |      0.2638 |   0.0797 |       0.1221 |    0.6791 |
| nonfinancial | risk_chars_ln          | private_pre         | ok       |     26541 |             406 |         0.3875 |        0.5515 |        0.3977 |     0.0040 |      0.2638 |   0.0797 |       0.1221 |    0.6791 |
| all          | risk_chars_ln          | private_highlev_pre | ok       |     26541 |             406 |         0.6382 |        0.7761 |        0.3715 |     0.0000 |      0.1586 |   0.2113 |       0.2500 |    0.1116 |
| nonfinancial | risk_chars_ln          | private_highlev_pre | ok       |     26541 |             406 |         0.6382 |        0.7761 |        0.3715 |     0.0000 |      0.1586 |   0.2113 |       0.2500 |    0.1116 |
| all          | risk_quality_index     | high_lev_pre        | ok       |     26541 |             406 |         0.9439 |        0.8909 |        0.0466 |     0.1545 |      0.0755 |   0.1052 |       0.1017 |    0.0504 |
| nonfinancial | risk_quality_index     | high_lev_pre        | ok       |     26541 |             406 |         0.9439 |        0.8909 |        0.0466 |     0.1545 |      0.0755 |   0.1052 |       0.1017 |    0.0504 |
| all          | has_risk_text          | failure_cost_pre    | ok       |     26541 |             406 |         0.8832 |        0.5512 |        0.1093 |     0.0003 |      0.0791 |   0.0007 |       0.0518 |    0.3144 |
| nonfinancial | has_risk_text          | failure_cost_pre    | ok       |     26541 |             406 |         0.8832 |        0.5512 |        0.1093 |     0.0003 |      0.0791 |   0.0007 |       0.0518 |    0.3144 |
| all          | has_risk_text          | natural_pre         | ok       |     26541 |             406 |         0.6509 |        0.0823 |        0.0577 |     0.0084 |      0.0454 |   0.0316 |       0.0529 |    0.2844 |
| nonfinancial | has_risk_text          | natural_pre         | ok       |     26541 |             406 |         0.6509 |        0.0823 |        0.0577 |     0.0084 |      0.0454 |   0.0316 |       0.0529 |    0.2844 |

## Event coefficients

| sample       | outcome                | exposure            | event   |   rel_year |    coef |   se_city_cluster |      p | sig   |
|:-------------|:-----------------------|:--------------------|:--------|-----------:|--------:|------------------:|-------:|:------|
| all          | has_risk_text          | private_pre         | <=-4    |         -4 |  0.0055 |            0.0383 | 0.8857 |       |
| all          | has_risk_text          | private_pre         | -3      |         -3 |  0.0204 |            0.0237 | 0.3914 |       |
| all          | has_risk_text          | private_pre         | -2      |         -2 |  0.0326 |            0.0307 | 0.2897 |       |
| all          | has_risk_text          | private_pre         | 0       |          0 |  0.0815 |            0.0254 | 0.0015 | ***   |
| all          | has_risk_text          | private_pre         | +1      |          1 |  0.0638 |            0.0240 | 0.0083 | ***   |
| all          | has_risk_text          | private_pre         | >=+2    |          2 |  0.0466 |            0.0529 | 0.3789 |       |
| nonfinancial | has_risk_text          | private_pre         | <=-4    |         -4 |  0.0055 |            0.0383 | 0.8857 |       |
| nonfinancial | has_risk_text          | private_pre         | -3      |         -3 |  0.0204 |            0.0237 | 0.3914 |       |
| nonfinancial | has_risk_text          | private_pre         | -2      |         -2 |  0.0326 |            0.0307 | 0.2897 |       |
| nonfinancial | has_risk_text          | private_pre         | 0       |          0 |  0.0815 |            0.0254 | 0.0015 | ***   |
| nonfinancial | has_risk_text          | private_pre         | +1      |          1 |  0.0638 |            0.0240 | 0.0083 | ***   |
| nonfinancial | has_risk_text          | private_pre         | >=+2    |          2 |  0.0466 |            0.0529 | 0.3789 |       |
| all          | has_risk_text          | private_highlev_pre | <=-4    |         -4 |  0.0024 |            0.0415 | 0.9530 |       |
| all          | has_risk_text          | private_highlev_pre | -3      |         -3 |  0.0057 |            0.0218 | 0.7932 |       |
| all          | has_risk_text          | private_highlev_pre | -2      |         -2 |  0.0215 |            0.0184 | 0.2445 |       |
| all          | has_risk_text          | private_highlev_pre | 0       |          0 |  0.0688 |            0.0125 | 0.0000 | ***   |
| all          | has_risk_text          | private_highlev_pre | +1      |          1 |  0.0431 |            0.0221 | 0.0525 | *     |
| all          | has_risk_text          | private_highlev_pre | >=+2    |          2 |  0.0534 |            0.0259 | 0.0407 | **    |
| nonfinancial | has_risk_text          | private_highlev_pre | <=-4    |         -4 |  0.0024 |            0.0415 | 0.9530 |       |
| nonfinancial | has_risk_text          | private_highlev_pre | -3      |         -3 |  0.0057 |            0.0218 | 0.7932 |       |
| nonfinancial | has_risk_text          | private_highlev_pre | -2      |         -2 |  0.0215 |            0.0184 | 0.2445 |       |
| nonfinancial | has_risk_text          | private_highlev_pre | 0       |          0 |  0.0688 |            0.0125 | 0.0000 | ***   |
| nonfinancial | has_risk_text          | private_highlev_pre | +1      |          1 |  0.0431 |            0.0221 | 0.0525 | *     |
| nonfinancial | has_risk_text          | private_highlev_pre | >=+2    |          2 |  0.0534 |            0.0259 | 0.0407 | **    |
| all          | risk_specificity_index | high_lev_pre        | <=-4    |         -4 |  0.0042 |            0.0555 | 0.9400 |       |
| all          | risk_specificity_index | high_lev_pre        | -3      |         -3 | -0.0230 |            0.0795 | 0.7726 |       |
| all          | risk_specificity_index | high_lev_pre        | -2      |         -2 | -0.0195 |            0.0337 | 0.5646 |       |
| all          | risk_specificity_index | high_lev_pre        | 0       |          0 |  0.0430 |            0.0267 | 0.1081 |       |
| all          | risk_specificity_index | high_lev_pre        | +1      |          1 |  0.0788 |            0.0273 | 0.0043 | ***   |
| all          | risk_specificity_index | high_lev_pre        | >=+2    |          2 |  0.1076 |            0.0538 | 0.0465 | **    |
| nonfinancial | risk_specificity_index | high_lev_pre        | <=-4    |         -4 |  0.0042 |            0.0555 | 0.9400 |       |
| nonfinancial | risk_specificity_index | high_lev_pre        | -3      |         -3 | -0.0230 |            0.0795 | 0.7726 |       |
| nonfinancial | risk_specificity_index | high_lev_pre        | -2      |         -2 | -0.0195 |            0.0337 | 0.5646 |       |
| nonfinancial | risk_specificity_index | high_lev_pre        | 0       |          0 |  0.0430 |            0.0267 | 0.1081 |       |
| nonfinancial | risk_specificity_index | high_lev_pre        | +1      |          1 |  0.0788 |            0.0273 | 0.0043 | ***   |
| nonfinancial | risk_specificity_index | high_lev_pre        | >=+2    |          2 |  0.1076 |            0.0538 | 0.0465 | **    |
| all          | risk_chars_ln          | private_pre         | <=-4    |         -4 |  0.0646 |            0.2035 | 0.7510 |       |
| all          | risk_chars_ln          | private_pre         | -3      |         -3 |  0.0978 |            0.1397 | 0.4844 |       |
| all          | risk_chars_ln          | private_pre         | -2      |         -2 |  0.1374 |            0.1978 | 0.4881 |       |
| all          | risk_chars_ln          | private_pre         | 0       |          0 |  0.3977 |            0.1369 | 0.0040 | ***   |
| all          | risk_chars_ln          | private_pre         | +1      |          1 |  0.2638 |            0.1499 | 0.0797 | *     |
| all          | risk_chars_ln          | private_pre         | >=+2    |          2 |  0.1221 |            0.2947 | 0.6791 |       |
| nonfinancial | risk_chars_ln          | private_pre         | <=-4    |         -4 |  0.0646 |            0.2035 | 0.7510 |       |
| nonfinancial | risk_chars_ln          | private_pre         | -3      |         -3 |  0.0978 |            0.1397 | 0.4844 |       |
| nonfinancial | risk_chars_ln          | private_pre         | -2      |         -2 |  0.1374 |            0.1978 | 0.4881 |       |
| nonfinancial | risk_chars_ln          | private_pre         | 0       |          0 |  0.3977 |            0.1369 | 0.0040 | ***   |
| nonfinancial | risk_chars_ln          | private_pre         | +1      |          1 |  0.2638 |            0.1499 | 0.0797 | *     |
| nonfinancial | risk_chars_ln          | private_pre         | >=+2    |          2 |  0.1221 |            0.2947 | 0.6791 |       |
| all          | risk_chars_ln          | private_highlev_pre | <=-4    |         -4 |  0.0112 |            0.2509 | 0.9645 |       |
| all          | risk_chars_ln          | private_highlev_pre | -3      |         -3 | -0.0089 |            0.1387 | 0.9491 |       |
| all          | risk_chars_ln          | private_highlev_pre | -2      |         -2 |  0.0880 |            0.1077 | 0.4143 |       |
| all          | risk_chars_ln          | private_highlev_pre | 0       |          0 |  0.3715 |            0.0755 | 0.0000 | ***   |
| all          | risk_chars_ln          | private_highlev_pre | +1      |          1 |  0.1586 |            0.1266 | 0.2113 |       |
| all          | risk_chars_ln          | private_highlev_pre | >=+2    |          2 |  0.2500 |            0.1566 | 0.1116 |       |
| nonfinancial | risk_chars_ln          | private_highlev_pre | <=-4    |         -4 |  0.0112 |            0.2509 | 0.9645 |       |
| nonfinancial | risk_chars_ln          | private_highlev_pre | -3      |         -3 | -0.0089 |            0.1387 | 0.9491 |       |
| nonfinancial | risk_chars_ln          | private_highlev_pre | -2      |         -2 |  0.0880 |            0.1077 | 0.4143 |       |
| nonfinancial | risk_chars_ln          | private_highlev_pre | 0       |          0 |  0.3715 |            0.0755 | 0.0000 | ***   |
| nonfinancial | risk_chars_ln          | private_highlev_pre | +1      |          1 |  0.1586 |            0.1266 | 0.2113 |       |
| nonfinancial | risk_chars_ln          | private_highlev_pre | >=+2    |          2 |  0.2500 |            0.1566 | 0.1116 |       |
| all          | risk_quality_index     | high_lev_pre        | <=-4    |         -4 |  0.0109 |            0.0539 | 0.8403 |       |
| all          | risk_quality_index     | high_lev_pre        | -3      |         -3 |  0.0050 |            0.0862 | 0.9535 |       |
| all          | risk_quality_index     | high_lev_pre        | -2      |         -2 | -0.0114 |            0.0425 | 0.7882 |       |
| all          | risk_quality_index     | high_lev_pre        | 0       |          0 |  0.0466 |            0.0326 | 0.1545 |       |
| all          | risk_quality_index     | high_lev_pre        | +1      |          1 |  0.0755 |            0.0464 | 0.1052 |       |
| all          | risk_quality_index     | high_lev_pre        | >=+2    |          2 |  0.1017 |            0.0517 | 0.0504 | *     |
| nonfinancial | risk_quality_index     | high_lev_pre        | <=-4    |         -4 |  0.0109 |            0.0539 | 0.8403 |       |
| nonfinancial | risk_quality_index     | high_lev_pre        | -3      |         -3 |  0.0050 |            0.0862 | 0.9535 |       |
| nonfinancial | risk_quality_index     | high_lev_pre        | -2      |         -2 | -0.0114 |            0.0425 | 0.7882 |       |
| nonfinancial | risk_quality_index     | high_lev_pre        | 0       |          0 |  0.0466 |            0.0326 | 0.1545 |       |
| nonfinancial | risk_quality_index     | high_lev_pre        | +1      |          1 |  0.0755 |            0.0464 | 0.1052 |       |
| nonfinancial | risk_quality_index     | high_lev_pre        | >=+2    |          2 |  0.1017 |            0.0517 | 0.0504 | *     |
| all          | has_risk_text          | failure_cost_pre    | <=-4    |         -4 |  0.0124 |            0.0636 | 0.8461 |       |
| all          | has_risk_text          | failure_cost_pre    | -3      |         -3 |  0.0027 |            0.0241 | 0.9110 |       |
| all          | has_risk_text          | failure_cost_pre    | -2      |         -2 |  0.0168 |            0.0429 | 0.6947 |       |
| all          | has_risk_text          | failure_cost_pre    | 0       |          0 |  0.1093 |            0.0296 | 0.0003 | ***   |
| all          | has_risk_text          | failure_cost_pre    | +1      |          1 |  0.0791 |            0.0229 | 0.0007 | ***   |
| all          | has_risk_text          | failure_cost_pre    | >=+2    |          2 |  0.0518 |            0.0514 | 0.3144 |       |
| nonfinancial | has_risk_text          | failure_cost_pre    | <=-4    |         -4 |  0.0124 |            0.0636 | 0.8461 |       |
| nonfinancial | has_risk_text          | failure_cost_pre    | -3      |         -3 |  0.0027 |            0.0241 | 0.9110 |       |
| nonfinancial | has_risk_text          | failure_cost_pre    | -2      |         -2 |  0.0168 |            0.0429 | 0.6947 |       |
| nonfinancial | has_risk_text          | failure_cost_pre    | 0       |          0 |  0.1093 |            0.0296 | 0.0003 | ***   |
| nonfinancial | has_risk_text          | failure_cost_pre    | +1      |          1 |  0.0791 |            0.0229 | 0.0007 | ***   |
| nonfinancial | has_risk_text          | failure_cost_pre    | >=+2    |          2 |  0.0518 |            0.0514 | 0.3144 |       |
| all          | has_risk_text          | natural_pre         | <=-4    |         -4 |  0.0129 |            0.0429 | 0.7642 |       |
| all          | has_risk_text          | natural_pre         | -3      |         -3 | -0.0059 |            0.0283 | 0.8342 |       |
| all          | has_risk_text          | natural_pre         | -2      |         -2 |  0.0220 |            0.0240 | 0.3607 |       |
| all          | has_risk_text          | natural_pre         | 0       |          0 |  0.0577 |            0.0217 | 0.0084 | ***   |
| all          | has_risk_text          | natural_pre         | +1      |          1 |  0.0454 |            0.0210 | 0.0316 | **    |
| all          | has_risk_text          | natural_pre         | >=+2    |          2 |  0.0529 |            0.0493 | 0.2844 |       |
| nonfinancial | has_risk_text          | natural_pre         | <=-4    |         -4 |  0.0129 |            0.0429 | 0.7642 |       |
| nonfinancial | has_risk_text          | natural_pre         | -3      |         -3 | -0.0059 |            0.0283 | 0.8342 |       |
| nonfinancial | has_risk_text          | natural_pre         | -2      |         -2 |  0.0220 |            0.0240 | 0.3607 |       |
| nonfinancial | has_risk_text          | natural_pre         | 0       |          0 |  0.0577 |            0.0217 | 0.0084 | ***   |
| nonfinancial | has_risk_text          | natural_pre         | +1      |          1 |  0.0454 |            0.0210 | 0.0316 | **    |
| nonfinancial | has_risk_text          | natural_pre         | >=+2    |          2 |  0.0529 |            0.0493 | 0.2844 |       |