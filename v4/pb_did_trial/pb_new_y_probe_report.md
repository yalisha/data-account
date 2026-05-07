# PB new Y probe: accounting recognition and weak write-down proxies

Date: 2026-05-06

Design: `PB_narrow x pre-policy exposure`, firm FE, city-year FE, industry-year FE, city-clustered SE. Strict columns add `ExposureGroup x YearFE` to absorb nationwide high-exposure trends.

## What this run can and cannot test

- Can test with current local data: loss recognition propensity, loss magnitude, accrual conservatism-style proxies.
- Only weakly testable: decreases in intangible assets and development expenditure; these are not exact impairment-loss variables.
- Not testable in the current local parquet set: corporate risk-disclosure specificity, debt-risk KAM text, exact asset/credit impairment loss.

## Decision table

| outcome            | exposure            | decision          |    coef |      p | sig   |   coef_strict |   p_strict | sig_strict   |   near_leads_p |   lag1_coef |   lag1_p |   lag2p_coef |   lag2p_p |   n_input | note                                                             |
|:-------------------|:--------------------|:------------------|--------:|-------:|:------|--------------:|-----------:|:-------------|---------------:|------------:|---------:|-------------:|----------:|----------:|:-----------------------------------------------------------------|
| LossDummy          | natural_highlev_pre | base_only         |  0.0448 | 0.0009 | ***   |       -0.0042 |     0.8044 |              |         0.0175 |     -0.0058 |   0.8587 |       0.0093 |    0.7601 |     26541 | 只有基准显著，不能直接写。                                       |
| LossDummy          | natural_pre         | base_only         |  0.0579 | 0.0032 | ***   |        0.0306 |     0.1461 |              |         0.0565 |      0.0112 |   0.8441 |      -0.0105 |    0.7820 |     26541 | 只有基准显著，不能直接写。                                       |
| LossMagnitude      | high_lev_pre        | base_only         |  0.0033 | 0.0061 | ***   |        0.0003 |     0.8521 |              |         0.0043 |     -0.0031 |   0.4321 |      -0.0029 |    0.4884 |     26541 | 只有基准显著，不能直接写。                                       |
| LossMagnitude      | private_highlev_pre | base_only         |  0.0047 | 0.0074 | ***   |        0.0008 |     0.7061 |              |         0.0017 |     -0.0005 |   0.8913 |      -0.0026 |    0.5953 |     26541 | 只有基准显著，不能直接写。                                       |
| LossMagnitude      | natural_highlev_pre | base_only         |  0.0043 | 0.0082 | ***   |        0.0004 |     0.8392 |              |         0.0086 |     -0.0015 |   0.6566 |      -0.0022 |    0.6335 |     26541 | 只有基准显著，不能直接写。                                       |
| LossDummy          | private_highlev_pre | base_only         |  0.0406 | 0.0095 | ***   |       -0.0079 |     0.6950 |              |         0.0167 |     -0.0097 |   0.7731 |      -0.0310 |    0.4016 |     26541 | 只有基准显著，不能直接写。                                       |
| ConservAccrual     | natural_highlev_pre | base_only         |  0.0139 | 0.0117 | **    |        0.0049 |     0.4109 |              |       nan      |    nan      | nan      |     nan      |  nan      |     26541 | 只有基准显著，不能直接写。                                       |
| NegAccrual         | private_highlev_pre | base_only         |  0.0101 | 0.0145 | **    |        0.0067 |     0.1270 |              |       nan      |    nan      | nan      |     nan      |  nan      |     26541 | 只有基准显著，不能直接写。                                       |
| NegAccrual         | natural_highlev_pre | base_only         |  0.0102 | 0.0268 | **    |        0.0065 |     0.1657 |              |       nan      |    nan      | nan      |     nan      |  nan      |     26541 | 只有基准显著，不能直接写。                                       |
| LossMagnitude      | natural_pre         | base_only         |  0.0029 | 0.0831 | *     |        0.0026 |     0.1891 |              |       nan      |    nan      | nan      |     nan      |  nan      |     26541 | 只有基准显著，不能直接写。                                       |
| ConservAccrual     | natural_pre         | candidate         |  0.0180 | 0.0000 | ***   |        0.0165 |     0.0000 | ***          |         0.1797 |      0.0105 |   0.3208 |       0.0040 |    0.6507 |     26541 | 基准、ExposureGroup-Year FE 与近端前趋势同时过关，可进入下一轮。 |
| NegAccrual         | natural_pre         | candidate         |  0.0115 | 0.0000 | ***   |        0.0103 |     0.0001 | ***          |         0.3976 |      0.0065 |   0.3732 |      -0.0009 |    0.8716 |     26541 | 基准、ExposureGroup-Year FE 与近端前趋势同时过关，可进入下一轮。 |
| ConservAccrual     | failure_cost_pre    | candidate         |  0.0221 | 0.0003 | ***   |        0.0160 |     0.0045 | ***          |         0.3580 |      0.0111 |   0.4663 |       0.0018 |    0.9079 |     26541 | 基准、ExposureGroup-Year FE 与近端前趋势同时过关，可进入下一轮。 |
| LossDummy          | failure_cost_pre    | candidate         |  0.0733 | 0.0006 | ***   |        0.0478 |     0.0842 | *            |         0.1138 |      0.0081 |   0.8864 |      -0.0400 |    0.3837 |     26541 | 基准、ExposureGroup-Year FE 与近端前趋势同时过关，可进入下一轮。 |
| NegAccrual         | failure_cost_pre    | candidate         |  0.0165 | 0.0006 | ***   |        0.0148 |     0.0005 | ***          |         0.5101 |      0.0096 |   0.2838 |      -0.0008 |    0.9357 |     26541 | 基准、ExposureGroup-Year FE 与近端前趋势同时过关，可进入下一轮。 |
| ConservAccrual     | private_highlev_pre | fragile_candidate |  0.0134 | 0.0072 | ***   |        0.0049 |     0.3732 |              |         0.5289 |      0.0060 |   0.5238 |      -0.0038 |    0.7240 |     26541 | 基准和近端前趋势有信号，但严格趋势控制未稳住。                   |
| NegAccrual         | high_lev_pre        | no_signal         |  0.0059 | 0.1659 |       |        0.0032 |     0.4334 |              |       nan      |    nan      | nan      |     nan      |  nan      |     26541 | 当前本地数据下没有可用信号。                                     |
| LossMagnitude      | failure_cost_pre    | no_signal         |  0.0035 | 0.2175 |       |        0.0013 |     0.6514 |              |       nan      |    nan      | nan      |     nan      |  nan      |     26541 | 当前本地数据下没有可用信号。                                     |
| ConservAccrual     | high_lev_pre        | no_signal         |  0.0071 | 0.2288 |       |        0.0006 |     0.9163 |              |       nan      |    nan      | nan      |     nan      |  nan      |     26541 | 当前本地数据下没有可用信号。                                     |
| LossDummy          | high_lev_pre        | no_signal         |  0.0164 | 0.3433 |       |       -0.0124 |     0.5018 |              |       nan      |    nan      | nan      |     nan      |  nan      |     26541 | 当前本地数据下没有可用信号。                                     |
| DevExpDecrease     | high_lev_pre        | weak_proxy_only   |  0.0016 | 0.0000 | ***   |        0.0015 |     0.0002 | ***          |       nan      |    nan      | nan      |     nan      |  nan      |      7379 | 弱代理变量，不能直接写成减值或稳健性主证据。                     |
| DevExpDecrease     | natural_highlev_pre | weak_proxy_only   |  0.0015 | 0.0000 | ***   |        0.0012 |     0.0006 | ***          |       nan      |    nan      | nan      |     nan      |  nan      |      7379 | 弱代理变量，不能直接写成减值或稳健性主证据。                     |
| DevExpDecrease     | private_highlev_pre | weak_proxy_only   |  0.0014 | 0.0000 | ***   |        0.0010 |     0.0024 | ***          |       nan      |    nan      | nan      |     nan      |  nan      |      7379 | 弱代理变量，不能直接写成减值或稳健性主证据。                     |
| IntangibleDecrease | natural_pre         | weak_proxy_only   |  0.0003 | 0.1174 |       |       -0.0001 |     0.6090 |              |       nan      |    nan      | nan      |     nan      |  nan      |     26521 | 弱代理变量，不能直接写成减值或稳健性主证据。                     |
| IntangibleDecrease | failure_cost_pre    | weak_proxy_only   |  0.0005 | 0.1625 |       |        0.0005 |     0.2511 |              |       nan      |    nan      | nan      |     nan      |  nan      |     26521 | 弱代理变量，不能直接写成减值或稳健性主证据。                     |
| DevExpDecrease     | failure_cost_pre    | weak_proxy_only   |  0.0007 | 0.2318 |       |        0.0004 |     0.4856 |              |       nan      |    nan      | nan      |     nan      |  nan      |      7379 | 弱代理变量，不能直接写成减值或稳健性主证据。                     |
| IntangibleDecrease | high_lev_pre        | weak_proxy_only   | -0.0003 | 0.3848 |       |       -0.0003 |     0.5148 |              |       nan      |    nan      | nan      |     nan      |  nan      |     26521 | 弱代理变量，不能直接写成减值或稳健性主证据。                     |
| IntangibleDecrease | natural_highlev_pre | weak_proxy_only   | -0.0003 | 0.5225 |       |       -0.0005 |     0.3775 |              |       nan      |    nan      | nan      |     nan      |  nan      |     26521 | 弱代理变量，不能直接写成减值或稳健性主证据。                     |
| IntangibleDecrease | private_highlev_pre | weak_proxy_only   | -0.0003 | 0.5334 |       |       -0.0005 |     0.3459 |              |       nan      |    nan      | nan      |     nan      |  nan      |     26521 | 弱代理变量，不能直接写成减值或稳健性主证据。                     |
| DevExpDecrease     | natural_pre         | weak_proxy_only   | -0.0002 | 0.6256 |       |       -0.0005 |     0.3240 |              |       nan      |    nan      | nan      |     nan      |  nan      |      7379 | 弱代理变量，不能直接写成减值或稳健性主证据。                     |

## Full base results

| outcome            | exposure            | spec                        |    coef |   se_city_cluster |       t |      p | sig   |   n_input |   treated_x_obs |   clusters_city | weak_proxy   |
|:-------------------|:--------------------|:----------------------------|--------:|------------------:|--------:|-------:|:------|----------:|----------------:|----------------:|:-------------|
| ConservAccrual     | natural_pre         | main_fe                     |  0.0180 |            0.0027 |  6.7304 | 0.0000 | ***   |     26541 |             997 |             406 | False        |
| NegAccrual         | natural_pre         | main_fe                     |  0.0115 |            0.0020 |  5.6978 | 0.0000 | ***   |     26541 |             997 |             406 | False        |
| DevExpDecrease     | high_lev_pre        | main_fe                     |  0.0016 |            0.0003 |  5.0953 | 0.0000 | ***   |      7379 |             162 |             354 | True         |
| DevExpDecrease     | natural_highlev_pre | main_fe                     |  0.0015 |            0.0003 |  5.0820 | 0.0000 | ***   |      7379 |             103 |             354 | True         |
| DevExpDecrease     | private_highlev_pre | main_fe                     |  0.0014 |            0.0003 |  4.9689 | 0.0000 | ***   |      7379 |             112 |             354 | True         |
| ConservAccrual     | natural_pre         | plus_exposure_group_year_fe |  0.0165 |            0.0037 |  4.4916 | 0.0000 | ***   |     26541 |             997 |             406 | False        |
| NegAccrual         | natural_pre         | plus_exposure_group_year_fe |  0.0103 |            0.0026 |  3.9684 | 0.0001 | ***   |     26541 |             997 |             406 | False        |
| DevExpDecrease     | high_lev_pre        | plus_exposure_group_year_fe |  0.0015 |            0.0004 |  3.8422 | 0.0002 | ***   |      7379 |             162 |             354 | True         |
| ConservAccrual     | failure_cost_pre    | main_fe                     |  0.0221 |            0.0060 |  3.6961 | 0.0003 | ***   |     26541 |            1210 |             406 | False        |
| NegAccrual         | failure_cost_pre    | plus_exposure_group_year_fe |  0.0148 |            0.0042 |  3.5252 | 0.0005 | ***   |     26541 |            1210 |             406 | False        |
| DevExpDecrease     | natural_highlev_pre | plus_exposure_group_year_fe |  0.0012 |            0.0004 |  3.5467 | 0.0006 | ***   |      7379 |             103 |             354 | True         |
| LossDummy          | failure_cost_pre    | main_fe                     |  0.0733 |            0.0212 |  3.4649 | 0.0006 | ***   |     26541 |            1210 |             406 | False        |
| NegAccrual         | failure_cost_pre    | main_fe                     |  0.0165 |            0.0048 |  3.4593 | 0.0006 | ***   |     26541 |            1210 |             406 | False        |
| LossDummy          | natural_highlev_pre | main_fe                     |  0.0448 |            0.0133 |  3.3739 | 0.0009 | ***   |     26541 |             400 |             406 | False        |
| DevExpDecrease     | private_highlev_pre | plus_exposure_group_year_fe |  0.0010 |            0.0003 |  3.1045 | 0.0024 | ***   |      7379 |             112 |             354 | True         |
| LossDummy          | natural_pre         | main_fe                     |  0.0579 |            0.0195 |  2.9728 | 0.0032 | ***   |     26541 |             997 |             406 | False        |
| ConservAccrual     | failure_cost_pre    | plus_exposure_group_year_fe |  0.0160 |            0.0056 |  2.8667 | 0.0045 | ***   |     26541 |            1210 |             406 | False        |
| LossMagnitude      | high_lev_pre        | main_fe                     |  0.0033 |            0.0012 |  2.7653 | 0.0061 | ***   |     26541 |             595 |             406 | False        |
| ConservAccrual     | private_highlev_pre | main_fe                     |  0.0134 |            0.0049 |  2.7092 | 0.0072 | ***   |     26541 |             410 |             406 | False        |
| LossMagnitude      | private_highlev_pre | main_fe                     |  0.0047 |            0.0017 |  2.6977 | 0.0074 | ***   |     26541 |             410 |             406 | False        |
| LossMagnitude      | natural_highlev_pre | main_fe                     |  0.0043 |            0.0016 |  2.6662 | 0.0082 | ***   |     26541 |             400 |             406 | False        |
| LossDummy          | private_highlev_pre | main_fe                     |  0.0406 |            0.0155 |  2.6151 | 0.0095 | ***   |     26541 |             410 |             406 | False        |
| ConservAccrual     | natural_highlev_pre | main_fe                     |  0.0139 |            0.0055 |  2.5391 | 0.0117 | **    |     26541 |             400 |             406 | False        |
| NegAccrual         | private_highlev_pre | main_fe                     |  0.0101 |            0.0041 |  2.4606 | 0.0145 | **    |     26541 |             410 |             406 | False        |
| NegAccrual         | natural_highlev_pre | main_fe                     |  0.0102 |            0.0046 |  2.2271 | 0.0268 | **    |     26541 |             400 |             406 | False        |
| LossMagnitude      | natural_pre         | main_fe                     |  0.0029 |            0.0017 |  1.7397 | 0.0831 | *     |     26541 |             997 |             406 | False        |
| LossDummy          | failure_cost_pre    | plus_exposure_group_year_fe |  0.0478 |            0.0276 |  1.7336 | 0.0842 | *     |     26541 |            1210 |             406 | False        |
| IntangibleDecrease | natural_pre         | main_fe                     |  0.0003 |            0.0002 |  1.5710 | 0.1174 |       |     26521 |             996 |             406 | True         |
| NegAccrual         | private_highlev_pre | plus_exposure_group_year_fe |  0.0067 |            0.0044 |  1.5310 | 0.1270 |       |     26541 |             410 |             406 | False        |
| LossDummy          | natural_pre         | plus_exposure_group_year_fe |  0.0306 |            0.0210 |  1.4578 | 0.1461 |       |     26541 |             997 |             406 | False        |
| IntangibleDecrease | failure_cost_pre    | main_fe                     |  0.0005 |            0.0004 |  1.4007 | 0.1625 |       |     26521 |            1209 |             406 | True         |
| NegAccrual         | natural_highlev_pre | plus_exposure_group_year_fe |  0.0065 |            0.0047 |  1.3903 | 0.1657 |       |     26541 |             400 |             406 | False        |
| NegAccrual         | high_lev_pre        | main_fe                     |  0.0059 |            0.0043 |  1.3893 | 0.1659 |       |     26541 |             595 |             406 | False        |
| LossMagnitude      | natural_pre         | plus_exposure_group_year_fe |  0.0026 |            0.0020 |  1.3167 | 0.1891 |       |     26541 |             997 |             406 | False        |
| LossMagnitude      | failure_cost_pre    | main_fe                     |  0.0035 |            0.0029 |  1.2362 | 0.2175 |       |     26541 |            1210 |             406 | False        |
| ConservAccrual     | high_lev_pre        | main_fe                     |  0.0071 |            0.0059 |  1.2062 | 0.2288 |       |     26541 |             595 |             406 | False        |
| DevExpDecrease     | failure_cost_pre    | main_fe                     |  0.0007 |            0.0006 |  1.2024 | 0.2318 |       |      7379 |             265 |             354 | True         |
| IntangibleDecrease | failure_cost_pre    | plus_exposure_group_year_fe |  0.0005 |            0.0004 |  1.1503 | 0.2511 |       |     26521 |            1209 |             406 | True         |
| DevExpDecrease     | natural_pre         | plus_exposure_group_year_fe | -0.0005 |            0.0005 | -0.9908 | 0.3240 |       |      7379 |             201 |             354 | True         |
| LossDummy          | high_lev_pre        | main_fe                     |  0.0164 |            0.0173 |  0.9495 | 0.3433 |       |     26541 |             595 |             406 | False        |
| IntangibleDecrease | private_highlev_pre | plus_exposure_group_year_fe | -0.0005 |            0.0005 | -0.9444 | 0.3459 |       |     26521 |             410 |             406 | True         |
| ConservAccrual     | private_highlev_pre | plus_exposure_group_year_fe |  0.0049 |            0.0055 |  0.8920 | 0.3732 |       |     26541 |             410 |             406 | False        |
| IntangibleDecrease | natural_highlev_pre | plus_exposure_group_year_fe | -0.0005 |            0.0005 | -0.8840 | 0.3775 |       |     26521 |             400 |             406 | True         |
| IntangibleDecrease | high_lev_pre        | main_fe                     | -0.0003 |            0.0003 | -0.8706 | 0.3848 |       |     26521 |             595 |             406 | True         |
| ConservAccrual     | natural_highlev_pre | plus_exposure_group_year_fe |  0.0049 |            0.0059 |  0.8237 | 0.4109 |       |     26541 |             400 |             406 | False        |
| NegAccrual         | high_lev_pre        | plus_exposure_group_year_fe |  0.0032 |            0.0041 |  0.7846 | 0.4334 |       |     26541 |             595 |             406 | False        |
| DevExpDecrease     | failure_cost_pre    | plus_exposure_group_year_fe |  0.0004 |            0.0005 |  0.6998 | 0.4856 |       |      7379 |             265 |             354 | True         |
| LossDummy          | high_lev_pre        | plus_exposure_group_year_fe | -0.0124 |            0.0184 | -0.6726 | 0.5018 |       |     26541 |             595 |             406 | False        |
| IntangibleDecrease | high_lev_pre        | plus_exposure_group_year_fe | -0.0003 |            0.0004 | -0.6523 | 0.5148 |       |     26521 |             595 |             406 | True         |
| IntangibleDecrease | natural_highlev_pre | main_fe                     | -0.0003 |            0.0004 | -0.6405 | 0.5225 |       |     26521 |             400 |             406 | True         |
| IntangibleDecrease | private_highlev_pre | main_fe                     | -0.0003 |            0.0004 | -0.6237 | 0.5334 |       |     26521 |             410 |             406 | True         |
| IntangibleDecrease | natural_pre         | plus_exposure_group_year_fe | -0.0001 |            0.0002 | -0.5121 | 0.6090 |       |     26521 |             996 |             406 | True         |
| DevExpDecrease     | natural_pre         | main_fe                     | -0.0002 |            0.0004 | -0.4893 | 0.6256 |       |      7379 |             201 |             354 | True         |
| LossMagnitude      | failure_cost_pre    | plus_exposure_group_year_fe |  0.0013 |            0.0030 |  0.4523 | 0.6514 |       |     26541 |            1210 |             406 | False        |
| LossDummy          | private_highlev_pre | plus_exposure_group_year_fe | -0.0079 |            0.0201 | -0.3926 | 0.6950 |       |     26541 |             410 |             406 | False        |
| LossMagnitude      | private_highlev_pre | plus_exposure_group_year_fe |  0.0008 |            0.0022 |  0.3775 | 0.7061 |       |     26541 |             410 |             406 | False        |
| LossDummy          | natural_highlev_pre | plus_exposure_group_year_fe | -0.0042 |            0.0168 | -0.2480 | 0.8044 |       |     26541 |             400 |             406 | False        |
| LossMagnitude      | natural_highlev_pre | plus_exposure_group_year_fe |  0.0004 |            0.0021 |  0.2031 | 0.8392 |       |     26541 |             400 |             406 | False        |
| LossMagnitude      | high_lev_pre        | plus_exposure_group_year_fe |  0.0003 |            0.0017 |  0.1867 | 0.8521 |       |     26541 |             595 |             406 | False        |
| ConservAccrual     | high_lev_pre        | plus_exposure_group_year_fe |  0.0006 |            0.0057 |  0.1052 | 0.9163 |       |     26541 |             595 |             406 | False        |

## Event-study summaries

| outcome        | exposure            | status   |   n_input |   clusters_city |   near_leads_p |   all_leads_p |   event0_coef |   event0_p |   lag1_coef |   lag1_p |   lag2p_coef |   lag2p_p |
|:---------------|:--------------------|:---------|----------:|----------------:|---------------:|--------------:|--------------:|-----------:|------------:|---------:|-------------:|----------:|
| ConservAccrual | natural_pre         | ok       |     26541 |             406 |         0.1797 |        0.3092 |        0.0013 |     0.8246 |      0.0105 |   0.3208 |       0.0040 |    0.6507 |
| NegAccrual     | natural_pre         | ok       |     26541 |             406 |         0.3976 |        0.5908 |        0.0045 |     0.2575 |      0.0065 |   0.3732 |      -0.0009 |    0.8716 |
| ConservAccrual | failure_cost_pre    | ok       |     26541 |             406 |         0.3580 |        0.5515 |       -0.0094 |     0.4864 |      0.0111 |   0.4663 |       0.0018 |    0.9079 |
| LossDummy      | failure_cost_pre    | ok       |     26541 |             406 |         0.1138 |        0.0002 |       -0.0039 |     0.9252 |      0.0081 |   0.8864 |      -0.0400 |    0.3837 |
| NegAccrual     | failure_cost_pre    | ok       |     26541 |             406 |         0.5101 |        0.5371 |        0.0006 |     0.9423 |      0.0096 |   0.2838 |      -0.0008 |    0.9357 |
| LossDummy      | natural_highlev_pre | ok       |     26541 |             406 |         0.0175 |        0.0362 |       -0.0423 |     0.2728 |     -0.0058 |   0.8587 |       0.0093 |    0.7601 |
| LossDummy      | natural_pre         | ok       |     26541 |             406 |         0.0565 |        0.0001 |        0.0222 |     0.3815 |      0.0112 |   0.8441 |      -0.0105 |    0.7820 |
| LossMagnitude  | high_lev_pre        | ok       |     26541 |             406 |         0.0043 |        0.0004 |       -0.0044 |     0.2918 |     -0.0031 |   0.4321 |      -0.0029 |    0.4884 |
| ConservAccrual | private_highlev_pre | ok       |     26541 |             406 |         0.5289 |        0.2761 |        0.0016 |     0.8726 |      0.0060 |   0.5238 |      -0.0038 |    0.7240 |
| LossMagnitude  | private_highlev_pre | ok       |     26541 |             406 |         0.0017 |        0.0032 |       -0.0024 |     0.6511 |     -0.0005 |   0.8913 |      -0.0026 |    0.5953 |
| LossMagnitude  | natural_highlev_pre | ok       |     26541 |             406 |         0.0086 |        0.0210 |       -0.0034 |     0.5160 |     -0.0015 |   0.6566 |      -0.0022 |    0.6335 |
| LossDummy      | private_highlev_pre | ok       |     26541 |             406 |         0.0167 |        0.0337 |       -0.0346 |     0.4630 |     -0.0097 |   0.7731 |      -0.0310 |    0.4016 |

Detailed event coefficients are in `pb_new_y_probe_event_coefficients.csv`.

## Event coefficients

| outcome        | exposure            | event   |    coef |   se_city_cluster |      p | sig   |
|:---------------|:--------------------|:--------|--------:|------------------:|-------:|:------|
| ConservAccrual | natural_pre         | <=-4    | -0.0192 |            0.0145 | 0.1875 |       |
| ConservAccrual | natural_pre         | -3      | -0.0143 |            0.0109 | 0.1917 |       |
| ConservAccrual | natural_pre         | -2      | -0.0091 |            0.0080 | 0.2526 |       |
| ConservAccrual | natural_pre         | 0       |  0.0013 |            0.0059 | 0.8246 |       |
| ConservAccrual | natural_pre         | +1      |  0.0105 |            0.0106 | 0.3208 |       |
| ConservAccrual | natural_pre         | >=+2    |  0.0040 |            0.0089 | 0.6507 |       |
| NegAccrual     | natural_pre         | <=-4    | -0.0129 |            0.0100 | 0.1991 |       |
| NegAccrual     | natural_pre         | -3      | -0.0078 |            0.0066 | 0.2352 |       |
| NegAccrual     | natural_pre         | -2      | -0.0028 |            0.0053 | 0.5911 |       |
| NegAccrual     | natural_pre         | 0       |  0.0045 |            0.0039 | 0.2575 |       |
| NegAccrual     | natural_pre         | +1      |  0.0065 |            0.0073 | 0.3732 |       |
| NegAccrual     | natural_pre         | >=+2    | -0.0009 |            0.0058 | 0.8716 |       |
| ConservAccrual | failure_cost_pre    | <=-4    | -0.0332 |            0.0266 | 0.2123 |       |
| ConservAccrual | failure_cost_pre    | -3      | -0.0211 |            0.0160 | 0.1896 |       |
| ConservAccrual | failure_cost_pre    | -2      | -0.0162 |            0.0151 | 0.2849 |       |
| ConservAccrual | failure_cost_pre    | 0       | -0.0094 |            0.0135 | 0.4864 |       |
| ConservAccrual | failure_cost_pre    | +1      |  0.0111 |            0.0152 | 0.4663 |       |
| ConservAccrual | failure_cost_pre    | >=+2    |  0.0018 |            0.0152 | 0.9079 |       |
| LossDummy      | failure_cost_pre    | <=-4    | -0.1328 |            0.0369 | 0.0004 | ***   |
| LossDummy      | failure_cost_pre    | -3      | -0.0489 |            0.0410 | 0.2351 |       |
| LossDummy      | failure_cost_pre    | -2      | -0.0788 |            0.0395 | 0.0471 | **    |
| LossDummy      | failure_cost_pre    | 0       | -0.0039 |            0.0410 | 0.9252 |       |
| LossDummy      | failure_cost_pre    | +1      |  0.0081 |            0.0567 | 0.8864 |       |
| LossDummy      | failure_cost_pre    | >=+2    | -0.0400 |            0.0458 | 0.3837 |       |
| NegAccrual     | failure_cost_pre    | <=-4    | -0.0211 |            0.0153 | 0.1694 |       |
| NegAccrual     | failure_cost_pre    | -3      | -0.0105 |            0.0098 | 0.2835 |       |
| NegAccrual     | failure_cost_pre    | -2      | -0.0091 |            0.0098 | 0.3553 |       |
| NegAccrual     | failure_cost_pre    | 0       |  0.0006 |            0.0086 | 0.9423 |       |
| NegAccrual     | failure_cost_pre    | +1      |  0.0096 |            0.0090 | 0.2838 |       |
| NegAccrual     | failure_cost_pre    | >=+2    | -0.0008 |            0.0100 | 0.9357 |       |
| LossDummy      | natural_highlev_pre | <=-4    | -0.0962 |            0.0339 | 0.0049 | ***   |
| LossDummy      | natural_highlev_pre | -3      | -0.0266 |            0.0339 | 0.4326 |       |
| LossDummy      | natural_highlev_pre | -2      | -0.0870 |            0.0306 | 0.0048 | ***   |
| LossDummy      | natural_highlev_pre | 0       | -0.0423 |            0.0385 | 0.2728 |       |
| LossDummy      | natural_highlev_pre | +1      | -0.0058 |            0.0328 | 0.8587 |       |
| LossDummy      | natural_highlev_pre | >=+2    |  0.0093 |            0.0304 | 0.7601 |       |
| LossDummy      | natural_pre         | <=-4    | -0.0775 |            0.0284 | 0.0067 | ***   |
| LossDummy      | natural_pre         | -3      | -0.0387 |            0.0344 | 0.2613 |       |
| LossDummy      | natural_pre         | -2      | -0.0382 |            0.0195 | 0.0517 | *     |
| LossDummy      | natural_pre         | 0       |  0.0222 |            0.0253 | 0.3815 |       |
| LossDummy      | natural_pre         | +1      |  0.0112 |            0.0568 | 0.8441 |       |
| LossDummy      | natural_pre         | >=+2    | -0.0105 |            0.0379 | 0.7820 |       |
| LossMagnitude  | high_lev_pre        | <=-4    | -0.0074 |            0.0046 | 0.1064 |       |
| LossMagnitude  | high_lev_pre        | -3      | -0.0079 |            0.0061 | 0.1977 |       |
| LossMagnitude  | high_lev_pre        | -2      | -0.0121 |            0.0037 | 0.0011 | ***   |
| LossMagnitude  | high_lev_pre        | 0       | -0.0044 |            0.0042 | 0.2918 |       |
| LossMagnitude  | high_lev_pre        | +1      | -0.0031 |            0.0039 | 0.4321 |       |
| LossMagnitude  | high_lev_pre        | >=+2    | -0.0029 |            0.0041 | 0.4884 |       |
| ConservAccrual | private_highlev_pre | <=-4    | -0.0179 |            0.0117 | 0.1273 |       |
| ConservAccrual | private_highlev_pre | -3      | -0.0103 |            0.0107 | 0.3375 |       |
| ConservAccrual | private_highlev_pre | -2      | -0.0122 |            0.0108 | 0.2627 |       |
| ConservAccrual | private_highlev_pre | 0       |  0.0016 |            0.0100 | 0.8726 |       |
| ConservAccrual | private_highlev_pre | +1      |  0.0060 |            0.0093 | 0.5238 |       |
| ConservAccrual | private_highlev_pre | >=+2    | -0.0038 |            0.0108 | 0.7240 |       |
| LossMagnitude  | private_highlev_pre | <=-4    | -0.0083 |            0.0037 | 0.0238 | **    |
| LossMagnitude  | private_highlev_pre | -3      | -0.0024 |            0.0069 | 0.7235 |       |
| LossMagnitude  | private_highlev_pre | -2      | -0.0134 |            0.0047 | 0.0045 | ***   |
| LossMagnitude  | private_highlev_pre | 0       | -0.0024 |            0.0054 | 0.6511 |       |
| LossMagnitude  | private_highlev_pre | +1      | -0.0005 |            0.0037 | 0.8913 |       |
| LossMagnitude  | private_highlev_pre | >=+2    | -0.0026 |            0.0048 | 0.5953 |       |
| LossMagnitude  | natural_highlev_pre | <=-4    | -0.0080 |            0.0037 | 0.0299 | **    |
| LossMagnitude  | natural_highlev_pre | -3      | -0.0044 |            0.0065 | 0.4974 |       |
| LossMagnitude  | natural_highlev_pre | -2      | -0.0135 |            0.0049 | 0.0060 | ***   |
| LossMagnitude  | natural_highlev_pre | 0       | -0.0034 |            0.0052 | 0.5160 |       |
| LossMagnitude  | natural_highlev_pre | +1      | -0.0015 |            0.0035 | 0.6566 |       |
| LossMagnitude  | natural_highlev_pre | >=+2    | -0.0022 |            0.0047 | 0.6335 |       |
| LossDummy      | private_highlev_pre | <=-4    | -0.1012 |            0.0353 | 0.0045 | ***   |
| LossDummy      | private_highlev_pre | -3      | -0.0321 |            0.0360 | 0.3739 |       |
| LossDummy      | private_highlev_pre | -2      | -0.0941 |            0.0333 | 0.0051 | ***   |
| LossDummy      | private_highlev_pre | 0       | -0.0346 |            0.0470 | 0.4630 |       |
| LossDummy      | private_highlev_pre | +1      | -0.0097 |            0.0337 | 0.7731 |       |
| LossDummy      | private_highlev_pre | >=+2    | -0.0310 |            0.0369 | 0.4016 |       |
