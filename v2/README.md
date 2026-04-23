# v16 Integrated Run Summary

更新时间：2026-04-19

## 产出路径

- 统一样本：[reg_sample_v16_integrated.dta](/Users/mac/computerscience/0做完了/15会计研究/data_stata/reg_sample_v16_integrated.dta)
- 样本说明：[sample_summary.csv](/Users/mac/computerscience/0做完了/15会计研究/results/v16_integrated/sample_summary.csv)
- H2a 五维分解：[h2a_value_chain.csv](/Users/mac/computerscience/0做完了/15会计研究/results/v16_integrated/h2a_value_chain.csv)
- H2b 质量 direct：[h2b_quality_direct.csv](/Users/mac/computerscience/0做完了/15会计研究/results/v16_integrated/h2b_quality_direct.csv)
- H2b 质量 joint：[h2b_quality_joint.csv](/Users/mac/computerscience/0做完了/15会计研究/results/v16_integrated/h2b_quality_joint.csv)
- H2c WashGap：[h2c_washgap.csv](/Users/mac/computerscience/0做完了/15会计研究/results/v16_integrated/h2c_washgap.csv)
- H3 下游机制：[h3_downstream.csv](/Users/mac/computerscience/0做完了/15会计研究/results/v16_integrated/h3_downstream.csv)
- 日志目录：[logs](</Users/mac/computerscience/0做完了/15会计研究/results/v16_integrated/logs>)

前版归档产物 [reg_sample_v16_quality.dta](/Users/mac/computerscience/0做完了/15会计研究/data_stata/reg_sample_v16_quality.dta) 和 [results/v16_quality](</Users/mac/computerscience/0做完了/15会计研究/results/v16_quality>) 未修改。

## 核心结果

### H2a 价值链五维分解

| 变量 | 本次系数 | t 值 | 结论 |
|---|---:|---:|---|
| `DU_stock_lag` | -0.01235 | -4.09 | 显著负向 |
| `DU_dev_lag` | -0.00477 | -2.97 | 显著负向 |
| `DU_app_lag` | -0.01163 | -4.46 | 显著负向 |
| `DU_value_lag` | -0.06168 | -1.51 | 不显著 |
| `DU_gov_lag` | -0.04273 | -1.66 | 边际显著 |

### H2b 质量变量 direct

| 变量 | v15 原值 | 本次 | 差异 |
|---|---:|---:|---:|
| `DUclosedloop_lag` | -0.00828, t=-2.74 | -0.00828, t=-2.74 | 约 0 |
| `DUcore_lag` | -0.00946, t=-3.46 | -0.00946, t=-3.46 | 约 0 |
| `DUchain_count_lag` | -0.00182, t=-2.13 | -0.00182, t=-2.13 | 约 0 |
| `DUkw_mda_lag` | -0.00056, t=-2.95 | -0.00056, t=-2.95 | 约 0 |

`DU_llm_lenstd_lag` 本次采用 v14 原公式后，Direct 结果为 `-0.01879***`，`t=-5.94`。对应历史 v14 原值为 `-0.01840***`，`t=-5.80`，量级一致。

### H2b Joint 附加观察

`DU_kw_lag + DU_llm_lenstd_lag` 联合规格中：

- `DU_kw_lag = -0.00155`，`t=-1.26`
- `DU_llm_lenstd_lag = -0.01457***`，`t=-3.76`

本轮不再把 “subsume DU_kw” 作为硬 gate，但该规格方向与历史 v14 结果一致。

### H2c WashGap

- Direct：`WashGap_lag = +0.00347**`，`t=2.09`
- Joint with `DU_kw_lag`：`WashGap_lag = +0.00575***`，`DU_kw_lag = -0.00448***`
- Joint with `DU_llm_lag`：`WashGap_lag = -0.00475`，`t=-1.59`，不显著

本次 direct 结果与历史 v14 `+0.00347**` 完全一致。

### H3 下游机制

| 渠道 | `DU_kw` | `DU_llm` | 与 v18 对比 |
|---|---:|---:|---|
| `ForecastDisp` | -0.00648*** | +0.00096 | 完全一致 |
| `CashFlowVol` | -0.00049** | -0.00048** | 完全一致 |
| `SCConc` | -0.27672*** | -0.46023*** | 与 v18 偏离 < 0.1% |

## DU_llm_lenstd 口径说明

### 新旧公式对比

- 错误旧公式：`llm_score / log1p(total_chars) * 1e4`
- 本次修正公式：`log1p(DU_kw) * (llm_score / 3.0)`
- 两种公式在统一样本上的相关系数：`0.6624`

结论：两者不是同一变量。此前错误旧公式下的回归结果，不能拿来与历史 `-0.0184***` 直接对比。历史 `-0.0184***` 对应的是本次修正后的 v14 原公式。

### 与 DU_kw 的相关性

- `corr(DU_llm_lenstd, DU_kw) = 0.8336`

该相关性较高，因此本轮将 `DU_llm_lenstd + DU_kw` 联合规格降级为附加观察，而不再把 “subsume” 作为核心识别卖点。

## Gate 检查

| Gate | 结果 | 说明 |
|---|---|---|
| 任务 1 样本 `N=43,735` | 通过 | 统一样本构造完成 |
| H2a 至少三维 `stock/dev/app` 显著负向 | 通过 | 三个主维度全部通过 |
| H2b Direct 显著 | 通过 | 五个 direct 规格均为负，且 `5/5` 显著 |
| H2b Joint `subsume` | 不再作为 gate | 仅保留为附加观察 |
| H2c WashGap Direct 正向显著 | 通过 | `+0.00347**` |
| H3 与 v18 偏离 < 20% | 通过 | 实际接近 0% 偏离 |

## 执行说明

- 本轮补跑主要集中在 `2026-04-19 16:20` 至 `16:32`。
- `OPEN_QUESTIONS.md` 已记录旧 spec 与新裁决的衔接、公式口径修正与 gate 变更。
