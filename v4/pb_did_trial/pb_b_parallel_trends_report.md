# PB B-route parallel-trends screen

Date: 2026-05-04

Design: event-study version of `PB_narrow x pre-policy exposure`, with firm FE, city-year FE, industry-year FE, city-clustered standard errors. Base event year is -1.

Interpretation: the key pre-trend check is whether event years -3 and -2 are jointly insignificant. The <=-4 bin is reported as a far-pre-period diagnostic.

## Joint lead tests

| outcome          | exposure            | status   |   n_input |   clusters_city |   near_leads_chi2 |   near_leads_p |   all_leads_chi2 |   all_leads_p |
|:-----------------|:--------------------|:---------|----------:|----------------:|------------------:|---------------:|-----------------:|--------------:|
| short_liab_ratio | high_lev_pre        | ok       |     26541 |             406 |            2.5078 |         0.2854 |           4.8607 |        0.1823 |
| short_liab_ratio | private_highlev_pre | ok       |     26541 |             406 |            2.7242 |         0.2561 |           6.2893 |        0.0984 |
| short_liab_ratio | natural_highlev_pre | ok       |     26541 |             406 |            3.0580 |         0.2168 |           5.1386 |        0.1619 |
| Lev              | high_lev_pre        | ok       |     26541 |             406 |           53.7225 |         0.0000 |          56.7638 |        0.0000 |
| Lev              | private_highlev_pre | ok       |     26541 |             406 |           10.5558 |         0.0051 |          42.8000 |        0.0000 |
| Lev              | natural_highlev_pre | ok       |     26541 |             406 |           11.2605 |         0.0036 |          48.5479 |        0.0000 |
| InvestIneff      | natural_pre         | ok       |     22884 |             397 |           13.4679 |         0.0012 |          24.2764 |        0.0000 |
| InvestIneff      | failure_cost_pre    | ok       |     22884 |             397 |            2.6128 |         0.2708 |          16.7354 |        0.0008 |

## Lead coefficients

| outcome          | exposure            | event   |    coef |   se_city_cluster |       t |      p | sig   |   treated_exposure_obs |
|:-----------------|:--------------------|:--------|--------:|------------------:|--------:|-------:|:------|-----------------------:|
| short_liab_ratio | high_lev_pre        | <=-4    |  0.0137 |            0.0246 |  0.5562 | 0.5786 |       |                    622 |
| short_liab_ratio | high_lev_pre        | -3      |  0.0067 |            0.0141 |  0.4755 | 0.6348 |       |                    240 |
| short_liab_ratio | high_lev_pre        | -2      |  0.0141 |            0.0107 |  1.3190 | 0.1883 |       |                    229 |
| short_liab_ratio | private_highlev_pre | <=-4    | -0.0060 |            0.0106 | -0.5715 | 0.5682 |       |                    409 |
| short_liab_ratio | private_highlev_pre | -3      | -0.0112 |            0.0088 | -1.2693 | 0.2055 |       |                    168 |
| short_liab_ratio | private_highlev_pre | -2      | -0.0053 |            0.0055 | -0.9695 | 0.3332 |       |                    157 |
| short_liab_ratio | natural_highlev_pre | <=-4    | -0.0095 |            0.0106 | -0.8897 | 0.3745 |       |                    400 |
| short_liab_ratio | natural_highlev_pre | -3      | -0.0111 |            0.0096 | -1.1582 | 0.2479 |       |                    165 |
| short_liab_ratio | natural_highlev_pre | -2      | -0.0060 |            0.0054 | -1.1153 | 0.2658 |       |                    155 |
| Lev              | high_lev_pre        | <=-4    |  0.0168 |            0.0091 |  1.8379 | 0.0672 | *     |                    622 |
| Lev              | high_lev_pre        | -3      |  0.0290 |            0.0071 |  4.0791 | 0.0001 | ***   |                    240 |
| Lev              | high_lev_pre        | -2      |  0.0081 |            0.0094 |  0.8635 | 0.3887 |       |                    229 |
| Lev              | private_highlev_pre | <=-4    |  0.0012 |            0.0131 |  0.0939 | 0.9253 |       |                    409 |
| Lev              | private_highlev_pre | -3      |  0.0162 |            0.0136 |  1.1903 | 0.2350 |       |                    168 |
| Lev              | private_highlev_pre | -2      |  0.0018 |            0.0152 |  0.1158 | 0.9079 |       |                    157 |
| Lev              | natural_highlev_pre | <=-4    | -0.0015 |            0.0115 | -0.1320 | 0.8951 |       |                    400 |
| Lev              | natural_highlev_pre | -3      |  0.0146 |            0.0119 |  1.2250 | 0.2217 |       |                    165 |
| Lev              | natural_highlev_pre | -2      |  0.0011 |            0.0146 |  0.0771 | 0.9386 |       |                    155 |
| InvestIneff      | natural_pre         | <=-4    | -0.0109 |            0.0095 | -1.1517 | 0.2506 |       |                    620 |
| InvestIneff      | natural_pre         | -3      | -0.0185 |            0.0070 | -2.6187 | 0.0094 | ***   |                    258 |
| InvestIneff      | natural_pre         | -2      | -0.0097 |            0.0111 | -0.8774 | 0.3812 |       |                    312 |
| InvestIneff      | failure_cost_pre    | <=-4    | -0.0399 |            0.0151 | -2.6505 | 0.0086 | ***   |                    819 |
| InvestIneff      | failure_cost_pre    | -3      | -0.0125 |            0.0128 | -0.9696 | 0.3332 |       |                    328 |
| InvestIneff      | failure_cost_pre    | -2      | -0.0215 |            0.0155 | -1.3818 | 0.1683 |       |                    384 |

## Post coefficients

| outcome          | exposure            | event   |    coef |   se_city_cluster |       t |      p | sig   |   treated_exposure_obs |
|:-----------------|:--------------------|:--------|--------:|------------------:|--------:|-------:|:------|-----------------------:|
| short_liab_ratio | high_lev_pre        | 0       |  0.0342 |            0.0167 |  2.0503 | 0.0414 | **    |                    218 |
| short_liab_ratio | high_lev_pre        | +1      |  0.0292 |            0.0141 |  2.0713 | 0.0393 | **    |                    216 |
| short_liab_ratio | high_lev_pre        | >=+2    |  0.0501 |            0.0120 |  4.1576 | 0.0000 | ***   |                    161 |
| short_liab_ratio | private_highlev_pre | 0       |  0.0315 |            0.0157 |  2.0097 | 0.0455 | **    |                    147 |
| short_liab_ratio | private_highlev_pre | +1      |  0.0258 |            0.0186 |  1.3862 | 0.1669 |       |                    145 |
| short_liab_ratio | private_highlev_pre | >=+2    |  0.0464 |            0.0091 |  5.1171 | 0.0000 | ***   |                    118 |
| short_liab_ratio | natural_highlev_pre | 0       |  0.0279 |            0.0162 |  1.7187 | 0.0869 | *     |                    145 |
| short_liab_ratio | natural_highlev_pre | +1      |  0.0212 |            0.0190 |  1.1153 | 0.2658 |       |                    142 |
| short_liab_ratio | natural_highlev_pre | >=+2    |  0.0404 |            0.0069 |  5.8843 | 0.0000 | ***   |                    113 |
| Lev              | high_lev_pre        | 0       | -0.0100 |            0.0070 | -1.4136 | 0.1587 |       |                    218 |
| Lev              | high_lev_pre        | +1      | -0.0237 |            0.0081 | -2.9292 | 0.0037 | ***   |                    216 |
| Lev              | high_lev_pre        | >=+2    | -0.0330 |            0.0124 | -2.6517 | 0.0085 | ***   |                    161 |
| Lev              | private_highlev_pre | 0       | -0.0136 |            0.0056 | -2.4500 | 0.0150 | **    |                    147 |
| Lev              | private_highlev_pre | +1      | -0.0285 |            0.0094 | -3.0249 | 0.0027 | ***   |                    145 |
| Lev              | private_highlev_pre | >=+2    | -0.0265 |            0.0089 | -2.9603 | 0.0034 | ***   |                    118 |
| Lev              | natural_highlev_pre | 0       | -0.0127 |            0.0057 | -2.2398 | 0.0260 | **    |                    145 |
| Lev              | natural_highlev_pre | +1      | -0.0253 |            0.0105 | -2.4068 | 0.0168 | **    |                    142 |
| Lev              | natural_highlev_pre | >=+2    | -0.0196 |            0.0062 | -3.1537 | 0.0018 | ***   |                    113 |
| InvestIneff      | natural_pre         | 0       |  0.0059 |            0.0067 |  0.8830 | 0.3781 |       |                    340 |
| InvestIneff      | natural_pre         | +1      |  0.0033 |            0.0118 |  0.2761 | 0.7827 |       |                    341 |
| InvestIneff      | natural_pre         | >=+2    | -0.0064 |            0.0105 | -0.6154 | 0.5388 |       |                    276 |
| InvestIneff      | failure_cost_pre    | 0       |  0.0003 |            0.0126 |  0.0276 | 0.9780 |       |                    414 |
| InvestIneff      | failure_cost_pre    | +1      |  0.0069 |            0.0160 |  0.4291 | 0.6683 |       |                    416 |
| InvestIneff      | failure_cost_pre    | >=+2    | -0.0117 |            0.0180 | -0.6509 | 0.5157 |       |                    327 |

Plot: `/Users/mac/computerscience/0做完了/15会计研究/v4/pb_did_trial/pb_b_event_study_plot.png`
