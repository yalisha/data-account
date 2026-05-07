# PB all-candidate rerun

Date: 2026-05-04

Design: `PB_narrow x pre-policy exposure`, firm FE, city-year FE, industry-year FE, city-clustered SE. Event-study base year is t=-1.

## Top base DDD signals

| group                  | outcome          | exposure            |    coef |   se_city_cluster |       t |      p | sig   |   n_input |   treated_x_obs |   clusters_city |
|:-----------------------|:-----------------|:--------------------|--------:|------------------:|--------:|-------:|:------|----------:|----------------:|----------------:|
| B_debt_pressure        | InvestIneff      | natural_pre         |  0.0115 |            0.0024 |  4.8083 | 0.0000 | ***   |     22884 |        957.0000 |        397.0000 |
| B_debt_pressure        | Lev              | natural_highlev_pre | -0.0215 |            0.0046 | -4.7005 | 0.0000 | ***   |     26541 |        400.0000 |        406.0000 |
| A_positive_information | NCSKEW           | natural_pre         | -0.1333 |            0.0293 | -4.5426 | 0.0000 | ***   |     26541 |        997.0000 |        406.0000 |
| B_debt_pressure        | Lev              | private_highlev_pre | -0.0263 |            0.0061 | -4.3285 | 0.0000 | ***   |     26541 |        410.0000 |        406.0000 |
| B_debt_pressure        | Lev              | high_lev_pre        | -0.0343 |            0.0086 | -3.9882 | 0.0001 | ***   |     26541 |        595.0000 |        406.0000 |
| A_positive_information | NCSKEW           | failure_cost_pre    | -0.1691 |            0.0443 | -3.8132 | 0.0002 | ***   |     26541 |       1210.0000 |        406.0000 |
| A_positive_information | EarnQualAbs      | natural_highlev_pre |  0.0079 |            0.0022 |  3.5654 | 0.0004 | ***   |     25078 |        388.0000 |        403.0000 |
| B_debt_pressure        | InvestIneff      | failure_cost_pre    |  0.0221 |            0.0064 |  3.4241 | 0.0007 | ***   |     22884 |       1157.0000 |        397.0000 |
| A_positive_information | EarnQualAbs      | private_highlev_pre |  0.0066 |            0.0020 |  3.3770 | 0.0008 | ***   |     25078 |        398.0000 |        403.0000 |
| A_positive_information | NCSKEW           | private_pre         | -0.1221 |            0.0366 | -3.3363 | 0.0010 | ***   |     26541 |        972.0000 |        406.0000 |
| A_positive_information | DUVOL            | natural_pre         | -0.0545 |            0.0167 | -3.2724 | 0.0012 | ***   |     26541 |        997.0000 |        406.0000 |
| A_positive_information | EarnQualAbs      | high_lev_pre        |  0.0066 |            0.0020 |  3.2259 | 0.0014 | ***   |     25078 |        579.0000 |        403.0000 |
| A_positive_information | DUVOL            | failure_cost_pre    | -0.0767 |            0.0248 | -3.0965 | 0.0022 | ***   |     26541 |       1210.0000 |        406.0000 |
| B_debt_pressure        | short_liab_ratio | private_highlev_pre |  0.0387 |            0.0132 |  2.9446 | 0.0035 | ***   |     26541 |        410.0000 |        406.0000 |
| A_positive_information | DUVOL            | private_pre         | -0.0520 |            0.0180 | -2.8912 | 0.0042 | ***   |     26541 |        972.0000 |        406.0000 |
| A_positive_information | NCSKEW           | high_lev_pre        |  0.0858 |            0.0311 |  2.7534 | 0.0063 | ***   |     26541 |        595.0000 |        406.0000 |
| B_debt_pressure        | short_liab_ratio | high_lev_pre        |  0.0265 |            0.0098 |  2.7104 | 0.0072 | ***   |     26541 |        595.0000 |        406.0000 |
| B_debt_pressure        | short_liab_ratio | natural_highlev_pre |  0.0356 |            0.0140 |  2.5544 | 0.0112 | **    |     26541 |        400.0000 |        406.0000 |
| B_supply_chain         | CustConc         | private_pre         |  2.0674 |            0.8795 |  2.3506 | 0.0195 | **    |     25601 |        970.0000 |        406.0000 |
| B_supply_chain         | CustConc         | high_lev_pre        | -1.3842 |            0.6615 | -2.0924 | 0.0374 | **    |     25601 |        591.0000 |        406.0000 |
| A_positive_information | DUVOL            | high_lev_pre        |  0.0179 |            0.0087 |  2.0525 | 0.0411 | **    |     26541 |        595.0000 |        406.0000 |
| B_debt_pressure        | InvestIneff      | natural_highlev_pre |  0.0081 |            0.0042 |  1.9149 | 0.0567 | *     |     22884 |        389.0000 |        397.0000 |
| A_audit_risk           | AuditFee         | failure_cost_pre    |  0.0359 |            0.0189 |  1.9062 | 0.0577 | *     |     26490 |       1207.0000 |        406.0000 |
| B_supply_chain         | CustHHI          | high_lev_pre        | -0.6092 |            0.3224 | -1.8895 | 0.0602 | *     |     18164 |        513.0000 |        370.0000 |
| B_debt_pressure        | InvestIneff      | private_highlev_pre |  0.0085 |            0.0045 |  1.8700 | 0.0627 | *     |     22884 |        396.0000 |        397.0000 |
| A_audit_risk           | AuditFee         | private_pre         |  0.0200 |            0.0115 |  1.7404 | 0.0830 | *     |     26490 |        969.0000 |        406.0000 |
| A_positive_information | NCSKEW           | private_highlev_pre |  0.0552 |            0.0322 |  1.7150 | 0.0876 | *     |     26541 |        410.0000 |        406.0000 |
| B_debt_pressure        | InvestIneff      | private_pre         |  0.0062 |            0.0037 |  1.6685 | 0.0965 | *     |     22884 |        930.0000 |        397.0000 |
| A_audit_risk           | AuditDelayLn     | failure_cost_pre    |  0.0297 |            0.0180 |  1.6482 | 0.1005 |       |     26534 |       1210.0000 |        406.0000 |
| B_debt_pressure        | short_liab_ratio | failure_cost_pre    |  0.0423 |            0.0259 |  1.6345 | 0.1034 |       |     26541 |       1210.0000 |        406.0000 |
| B_debt_pressure        | short_liab_ratio | private_pre         |  0.0246 |            0.0151 |  1.6299 | 0.1043 |       |     26541 |        972.0000 |        406.0000 |
| B_debt_pressure        | InvestIneff      | high_lev_pre        |  0.0065 |            0.0041 |  1.6034 | 0.1102 |       |     22884 |        571.0000 |        397.0000 |
| B_supply_chain         | CustConc         | natural_pre         |  1.3758 |            0.8975 |  1.5328 | 0.1266 |       |     25601 |        995.0000 |        406.0000 |
| A_audit_risk           | AuditDelayLn     | natural_pre         |  0.0217 |            0.0154 |  1.4059 | 0.1610 |       |     26534 |        997.0000 |        406.0000 |
| B_supply_chain         | CustHHI          | natural_highlev_pre | -0.5596 |            0.4087 | -1.3692 | 0.1724 |       |     18164 |        362.0000 |        370.0000 |
| B_debt_pressure        | Lev              | failure_cost_pre    | -0.0192 |            0.0141 | -1.3600 | 0.1750 |       |     26541 |       1210.0000 |        406.0000 |
| A_audit_risk           | AuditDelayLn     | natural_highlev_pre |  0.0154 |            0.0113 |  1.3578 | 0.1757 |       |     26534 |        400.0000 |        406.0000 |
| A_audit_risk           | NonStdAudit      | private_highlev_pre |  0.0193 |            0.0143 |  1.3518 | 0.1776 |       |     26541 |        410.0000 |        406.0000 |
| B_supply_chain         | CustHHI          | failure_cost_pre    | -1.1225 |            0.8513 | -1.3186 | 0.1887 |       |     18164 |       1008.0000 |        370.0000 |
| A_audit_risk           | AuditFee         | natural_highlev_pre |  0.0099 |            0.0078 |  1.2693 | 0.2055 |       |     26490 |        399.0000 |        406.0000 |

## Decision table after event checks

| group                  | outcome          | exposure            |     coef |        p |   near_leads_p |   post0_p |   post1_p |   post2p_p | direction                | verdict       |
|:-----------------------|:-----------------|:--------------------|---------:|---------:|---------------:|----------:|----------:|-----------:|:-------------------------|:--------------|
| B_debt_pressure        | InvestIneff      | natural_pre         |   0.0115 |   0.0000 |         0.0012 |    0.3781 |    0.7827 |     0.5388 | good_if_negative         | bad_pretrend  |
| B_debt_pressure        | Lev              | natural_highlev_pre |  -0.0215 |   0.0000 |         0.0036 |    0.0260 |    0.0168 |     0.0018 | good_if_negative         | bad_pretrend  |
| B_debt_pressure        | Lev              | private_highlev_pre |  -0.0263 |   0.0000 |         0.0051 |    0.0150 |    0.0027 |     0.0034 | good_if_negative         | bad_pretrend  |
| B_debt_pressure        | Lev              | high_lev_pre        |  -0.0343 |   0.0001 |         0.0000 |    0.1587 |    0.0037 |     0.0085 | good_if_negative         | bad_pretrend  |
| A_positive_information | EarnQualAbs      | natural_highlev_pre |   0.0079 |   0.0004 |         0.0000 |    0.4359 |    0.2117 |     0.6011 | good_if_negative         | bad_pretrend  |
| A_positive_information | EarnQualAbs      | private_highlev_pre |   0.0066 |   0.0008 |         0.0000 |    0.5588 |    0.1831 |     0.7686 | good_if_negative         | bad_pretrend  |
| A_positive_information | NCSKEW           | private_pre         |  -0.1221 |   0.0010 |         0.0005 |    0.4360 |    0.1352 |     0.0164 | good_if_negative         | bad_pretrend  |
| A_positive_information | DUVOL            | natural_pre         |  -0.0545 |   0.0012 |         0.0478 |    0.1503 |    0.0236 |     0.0000 | good_if_negative         | bad_pretrend  |
| A_positive_information | EarnQualAbs      | high_lev_pre        |   0.0066 |   0.0014 |         0.0137 |    0.3849 |    0.2971 |     0.4233 | good_if_negative         | bad_pretrend  |
| A_positive_information | DUVOL            | private_pre         |  -0.0520 |   0.0042 |         0.0420 |    0.3427 |    0.1041 |     0.0065 | good_if_negative         | bad_pretrend  |
| B_debt_pressure        | InvestIneff      | natural_highlev_pre |   0.0081 |   0.0567 |         0.0009 |    0.0263 |    0.2787 |     0.0940 | good_if_negative         | bad_pretrend  |
| B_debt_pressure        | InvestIneff      | private_highlev_pre |   0.0085 |   0.0627 |         0.0001 |    0.0007 |    0.0965 |     0.0582 | good_if_negative         | bad_pretrend  |
| B_debt_pressure        | InvestIneff      | private_pre         |   0.0062 |   0.0965 |         0.0123 |    0.7189 |    0.9375 |     0.4697 | good_if_negative         | bad_pretrend  |
| A_positive_information | NCSKEW           | natural_pre         |  -0.1333 |   0.0000 |         0.5789 |    0.1241 |    0.0070 |     0.0000 | good_if_negative         | dynamic_pass  |
| A_positive_information | NCSKEW           | failure_cost_pre    |  -0.1691 |   0.0002 |         0.4254 |    0.2082 |    0.0990 |     0.0012 | good_if_negative         | dynamic_pass  |
| A_positive_information | DUVOL            | failure_cost_pre    |  -0.0767 |   0.0022 |         0.1522 |    0.2592 |    0.0852 |     0.0010 | good_if_negative         | dynamic_pass  |
| B_debt_pressure        | short_liab_ratio | private_highlev_pre |   0.0387 |   0.0035 |         0.2561 |    0.0455 |    0.1669 |     0.0000 | ambiguous                | dynamic_pass  |
| A_positive_information | NCSKEW           | high_lev_pre        |   0.0858 |   0.0063 |         0.6840 |    0.7220 |    0.0057 |     0.0174 | good_if_negative         | dynamic_pass  |
| B_debt_pressure        | short_liab_ratio | high_lev_pre        |   0.0265 |   0.0072 |         0.2854 |    0.0414 |    0.0393 |     0.0000 | ambiguous                | dynamic_pass  |
| B_debt_pressure        | short_liab_ratio | natural_highlev_pre |   0.0356 |   0.0112 |         0.2168 |    0.0869 |    0.2658 |     0.0000 | ambiguous                | dynamic_pass  |
| B_supply_chain         | CustConc         | private_pre         |   2.0674 |   0.0195 |         0.6625 |    0.0000 |    0.1541 |     0.3293 | ambiguous                | dynamic_pass  |
| B_supply_chain         | CustConc         | high_lev_pre        |  -1.3842 |   0.0374 |         0.9351 |    0.5415 |    0.0018 |     0.1064 | ambiguous                | dynamic_pass  |
| A_positive_information | DUVOL            | high_lev_pre        |   0.0179 |   0.0411 |         0.8135 |    0.9829 |    0.0330 |     0.0404 | good_if_negative         | dynamic_pass  |
| B_supply_chain         | CustHHI          | high_lev_pre        |  -0.6092 |   0.0602 |         0.8488 |    0.6258 |    0.0104 |     0.5828 | ambiguous                | dynamic_pass  |
| A_positive_information | NCSKEW           | private_highlev_pre |   0.0552 |   0.0876 |         0.7930 |    0.3972 |    0.0000 |     0.5365 | good_if_negative         | dynamic_pass  |
| A_positive_information | absDA            | private_pre         | nan      | nan      |       nan      |  nan      |  nan      |   nan      | good_if_negative         | not_estimated |
| A_positive_information | absDA            | natural_pre         | nan      | nan      |         0.0125 |  nan      |  nan      |   nan      | good_if_negative         | not_estimated |
| A_positive_information | absDA            | high_lev_pre        | nan      | nan      |       nan      |  nan      |  nan      |   nan      | good_if_negative         | not_estimated |
| A_positive_information | absDA            | private_highlev_pre | nan      | nan      |       nan      |  nan      |  nan      |   nan      | good_if_negative         | not_estimated |
| A_positive_information | absDA            | natural_highlev_pre | nan      | nan      |       nan      |  nan      |  nan      |   nan      | good_if_negative         | not_estimated |
| A_positive_information | absDA            | failure_cost_pre    | nan      | nan      |       nan      |  nan      |  nan      |   nan      | good_if_negative         | not_estimated |
| B_debt_pressure        | InvestIneff      | failure_cost_pre    |   0.0221 |   0.0007 |         0.2708 |    0.9780 |    0.6683 |     0.5157 | good_if_negative         | pooled_only   |
| A_audit_risk           | AuditFee         | failure_cost_pre    |   0.0359 |   0.0577 |         0.4277 |    0.5342 |    0.8345 |     0.4664 | risk_pricing_if_positive | pooled_only   |
| A_audit_risk           | AuditFee         | private_pre         |   0.0200 |   0.0830 |         0.7250 |    0.9690 |    0.7922 |     0.5585 | risk_pricing_if_positive | pooled_only   |

## Verdict labels
- `dynamic_pass`: base DDD significant, near leads clean, at least one post-event coefficient significant with same sign.
- `pooled_only`: base DDD significant and near leads clean, but event-study post coefficients are weak.
- `bad_pretrend`: base DDD significant but t=-3/-2 lead test fails.
- `weak_base`: base DDD not significant at 10%.
