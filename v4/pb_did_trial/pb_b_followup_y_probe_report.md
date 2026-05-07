# PB follow-up Y probe: supply chain and audit risk

Date: 2026-05-04

Design: same exposure DID/DDD as the B-route pilot: `PB_narrow x pre-policy debt exposure`, firm FE, city-year FE, industry-year FE, city-clustered SE.

## Base DDD results

| outcome       | exposure            |    coef |   se_city_cluster |       t |      p | sig   |   n_input |   treated_x_obs |   clusters_city |
|:--------------|:--------------------|--------:|------------------:|--------:|-------:|:------|----------:|----------------:|----------------:|
| CustConc      | private_pre         |  2.0674 |            0.8795 |  2.3506 | 0.0195 | **    |     25601 |             970 |             406 |
| CustConc      | high_lev_pre        | -1.3842 |            0.6615 | -2.0924 | 0.0374 | **    |     25601 |             591 |             406 |
| AuditFee      | failure_cost_pre    |  0.0359 |            0.0189 |  1.9062 | 0.0577 | *     |     26490 |            1207 |             406 |
| CustHHI       | high_lev_pre        | -0.6092 |            0.3224 | -1.8895 | 0.0602 | *     |     18164 |             513 |             370 |
| AuditFee      | private_pre         |  0.0200 |            0.0115 |  1.7404 | 0.0830 | *     |     26490 |             969 |             406 |
| AuditDelayLn  | failure_cost_pre    |  0.0297 |            0.0180 |  1.6482 | 0.1005 |       |     26534 |            1210 |             406 |
| CustConc      | natural_pre         |  1.3758 |            0.8975 |  1.5328 | 0.1266 |       |     25601 |             995 |             406 |
| AuditDelayLn  | natural_pre         |  0.0217 |            0.0154 |  1.4059 | 0.1610 |       |     26534 |             997 |             406 |
| CustHHI       | natural_highlev_pre | -0.5596 |            0.4087 | -1.3692 | 0.1724 |       |     18164 |             362 |             370 |
| AuditDelayLn  | natural_highlev_pre |  0.0154 |            0.0113 |  1.3578 | 0.1757 |       |     26534 |             400 |             406 |
| NonStdAudit   | private_highlev_pre |  0.0193 |            0.0143 |  1.3518 | 0.1776 |       |     26541 |             410 |             406 |
| CustHHI       | failure_cost_pre    | -1.1225 |            0.8513 | -1.3186 | 0.1887 |       |     18164 |            1008 |             370 |
| AuditFee      | natural_highlev_pre |  0.0099 |            0.0078 |  1.2693 | 0.2055 |       |     26490 |             399 |             406 |
| CustConc      | failure_cost_pre    |  1.3674 |            1.0965 |  1.2471 | 0.2135 |       |     25601 |            1205 |             406 |
| SuppConc      | private_highlev_pre |  0.6560 |            0.5318 |  1.2335 | 0.2185 |       |     25538 |             409 |             405 |
| SuppConc      | private_pre         | -1.1812 |            0.9732 | -1.2137 | 0.2260 |       |     25538 |             970 |             405 |
| NonStdAudit   | natural_highlev_pre |  0.0161 |            0.0139 |  1.1635 | 0.2457 |       |     26541 |             400 |             406 |
| AuditorSwitch | private_highlev_pre |  0.0133 |            0.0119 |  1.1197 | 0.2639 |       |     26541 |             410 |             406 |
| AuditFee      | private_highlev_pre |  0.0124 |            0.0111 |  1.1170 | 0.2651 |       |     26490 |             409 |             406 |
| SuppConc      | natural_highlev_pre |  0.7820 |            0.7291 |  1.0726 | 0.2845 |       |     25538 |             399 |             405 |
| AuditFee      | natural_pre         |  0.0155 |            0.0146 |  1.0609 | 0.2897 |       |     26490 |             994 |             406 |
| SuppConc      | natural_pre         | -0.7330 |            0.7007 | -1.0461 | 0.2965 |       |     25538 |             995 |             405 |
| SuppConc      | failure_cost_pre    | -1.0490 |            1.0561 | -0.9933 | 0.3215 |       |     25538 |            1206 |             405 |
| AuditDelayLn  | private_pre         |  0.0107 |            0.0110 |  0.9685 | 0.3337 |       |     26534 |             972 |             406 |
| AuditDelayLn  | private_highlev_pre |  0.0117 |            0.0123 |  0.9572 | 0.3394 |       |     26534 |             410 |             406 |
| SuppConc      | high_lev_pre        |  0.9958 |            1.0675 |  0.9328 | 0.3518 |       |     25538 |             592 |             405 |
| CustConc      | natural_highlev_pre | -0.4120 |            0.4595 | -0.8968 | 0.3707 |       |     25601 |             399 |             406 |
| SCConc        | private_highlev_pre |  0.3694 |            0.4139 |  0.8926 | 0.3729 |       |     25775 |             409 |             406 |
| AuditorSwitch | high_lev_pre        | -0.0219 |            0.0280 | -0.7829 | 0.4344 |       |     26541 |             595 |             406 |
| NonStdAudit   | natural_pre         |  0.0142 |            0.0232 |  0.6115 | 0.5414 |       |     26541 |             997 |             406 |
| CustHHI       | natural_pre         | -0.3939 |            0.7075 | -0.5568 | 0.5783 |       |     18164 |             839 |             370 |
| SCConc        | natural_pre         |  0.4144 |            0.8431 |  0.4915 | 0.6235 |       |     25775 |             995 |             406 |
| AuditorSwitch | failure_cost_pre    | -0.0269 |            0.0595 | -0.4525 | 0.6513 |       |     26541 |            1210 |             406 |
| SCConc        | private_pre         |  0.4452 |            1.0957 |  0.4063 | 0.6848 |       |     25775 |             970 |             406 |
| CustHHI       | private_highlev_pre | -0.1923 |            0.4823 | -0.3988 | 0.6905 |       |     18164 |             367 |             370 |
| CustConc      | private_highlev_pre |  0.2051 |            0.5171 |  0.3966 | 0.6920 |       |     25601 |             409 |             406 |
| SCConc        | natural_highlev_pre |  0.1774 |            0.4545 |  0.3903 | 0.6967 |       |     25775 |             399 |             406 |
| NonStdAudit   | private_pre         |  0.0075 |            0.0210 |  0.3557 | 0.7223 |       |     26541 |             972 |             406 |
| AuditorSwitch | private_pre         | -0.0111 |            0.0320 | -0.3483 | 0.7279 |       |     26541 |             972 |             406 |
| AuditorSwitch | natural_pre         | -0.0104 |            0.0351 | -0.2961 | 0.7674 |       |     26541 |             997 |             406 |
| SCConc        | failure_cost_pre    |  0.2653 |            0.9162 |  0.2895 | 0.7724 |       |     25775 |            1206 |             406 |
| NonStdAudit   | high_lev_pre        |  0.0025 |            0.0095 |  0.2632 | 0.7926 |       |     26541 |             595 |             406 |
| CustHHI       | private_pre         |  0.1805 |            0.7511 |  0.2403 | 0.8103 |       |     18164 |             817 |             370 |
| SCConc        | high_lev_pre        | -0.1932 |            0.8848 | -0.2183 | 0.8274 |       |     25775 |             592 |             406 |
| AuditDelayLn  | high_lev_pre        | -0.0013 |            0.0074 | -0.1817 | 0.8560 |       |     26534 |             595 |             406 |
| NonStdAudit   | failure_cost_pre    |  0.0034 |            0.0354 |  0.0959 | 0.9237 |       |     26541 |            1210 |             406 |
| AuditorSwitch | natural_highlev_pre |  0.0009 |            0.0107 |  0.0844 | 0.9328 |       |     26541 |             400 |             406 |
| AuditFee      | high_lev_pre        | -0.0008 |            0.0156 | -0.0537 | 0.9572 |       |     26490 |             594 |             406 |

## Event-study pretrend checks for strongest base signals

| outcome   | exposure         | status   |   n_input |   clusters_city |   near_leads_p |   event0_coef |   event0_p |   lag2p_coef |   lag2p_p |
|:----------|:-----------------|:---------|----------:|----------------:|---------------:|--------------:|-----------:|-------------:|----------:|
| CustConc  | private_pre      | ok       |     25601 |             406 |         0.6625 |        2.0336 |     0.0000 |       1.2636 |    0.3293 |
| CustConc  | high_lev_pre     | ok       |     25601 |             406 |         0.9351 |       -0.5977 |     0.5415 |      -2.3678 |    0.1064 |
| AuditFee  | failure_cost_pre | ok       |     26490 |             406 |         0.4277 |       -0.0253 |     0.5342 |       0.0260 |    0.4664 |
| CustHHI   | high_lev_pre     | ok       |     18164 |             370 |         0.8488 |       -0.2396 |     0.6258 |      -0.4140 |    0.5828 |
| AuditFee  | private_pre      | ok       |     26490 |             406 |         0.7250 |        0.0008 |     0.9690 |       0.0118 |    0.5585 |
