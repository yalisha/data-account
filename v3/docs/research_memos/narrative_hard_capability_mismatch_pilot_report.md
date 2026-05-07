# 年报数据要素叙事 - 会计验证/硬能力：试跑报告

日期：2026-05-02

## 1. 本次构造

这次不是继续用单纯的年报数据要素披露，而是把 X 改成：

> 年报数据要素叙事强度 - 可验证的硬能力/会计验证

脚本：

```text
src/stata/narrative_hard_capability_mismatch.do
```

核心输出：

```text
results/data/narrative_hard_capability_mismatch_panel.dta
results/stata/narrative_hard_x_summary.csv
results/stata/narrative_hard_y_screen_panel.csv
results/stata/narrative_hard_y_screen_2024.csv
results/logs/narrative_hard_capability_mismatch.log
results/logs/narrative_hard_capability_mismatch_mcp.log
results/logs/narrative_hard_capability_mismatch_mcp.smcl
```

Stata MCP 已成功跑完。Python 只用于读取结果 CSV 做汇总，没有做回归。

## 2. 变量口径

### 2.1 全样本版本：叙事 - 硬能力/潜力

叙事侧：

- `DU_kw`：年报数据要素叙事强度。

硬能力侧目前只能用已有变量拼一个 v0 代理：

- `HighTech`
- `StrategicEmerging`
- `DigEconCore`
- `IndustryCluster`
- `ln(1+LQ)`

构造：

- `HardBaseRaw`：上述硬能力代理标准化后的均值。
- `disc_pct_y`：同年内 `DU_kw` percentile，高值代表叙事强。
- `hard_base_pct_y`：同年内 `HardBaseRaw` percentile，高值代表硬能力强。
- `NarrHardGap_base = disc_pct_y - hard_base_pct_y`。
- `NarrHardWashing`：高叙事、低硬能力。
- `NarrHardHushing`：高硬能力、低叙事。
- `NarrHardVerified`：高叙事、高硬能力。
- strict 版本用 75/25 分位阈值。

重要实现细节：本机 Stata 的 `egen rank(...), field` 在这个用法下是高值拿更小 rank，所以脚本改为对负值排序，确保 percentile 方向是“原变量越大，percentile 越高”。因此本报告的新结果按这个方向解释。

### 2.2 2024 版本：叙事 - 硬能力 + 数据资产入表

会计验证侧：

- `DataAsset0`：从 `v1/data_parquet/panel.parquet` 抽取的数据资产变量，缺失设为 0。
- `BookEntry`：2024 年 `DataAsset0 > 0`。
- `HardAcctRaw`：2024 年 `HardBaseRaw`、`DataAsset_ln`、`BookEntry` 标准化后的均值。
- `NarrAcctHardGap = disc_pct_2024 - hard_acct_pct_2024`。
- `NarrAcctHardWashing / Hushing / Verified`：2024 年 accounting-augmented 版本。

## 3. X 的样本量

全样本：

| X | N | 均值/占比 | 正值/1 的数量 |
|---|---:|---:|---:|
| `NarrHardGap_base` | 43,735 | -0.0006 | 21,964 |
| `NarrHardWashing` | 43,735 | 0.2102 | 9,195 |
| `NarrHardHushing` | 43,735 | 0.2077 | 9,082 |
| `NarrHardVerified` | 43,735 | 0.2898 | 12,674 |
| `NarrHardWashing_strict` | 43,735 | 0.0297 | 1,299 |
| `NarrHardHushing_strict` | 43,735 | 0.0339 | 1,482 |
| `NarrHardVerified_strict` | 43,735 | 0.1094 | 4,786 |

2024：

| X | N | 均值/占比 | 正值/1 的数量 |
|---|---:|---:|---:|
| `BookEntry` | 4,598 | 0.0048 | 22 |
| `NarrAcctHardGap` | 4,598 | 0.0025 | 2,294 |
| `NarrAcctHardWashing` | 4,598 | 0.2114 | 972 |
| `NarrAcctHardHushing` | 4,598 | 0.2114 | 972 |
| `NarrAcctHardVerified` | 4,598 | 0.2886 | 1,327 |
| `NarrAcctHardWashing_strict` | 4,598 | 0.0287 | 132 |
| `NarrAcctHardHushing_strict` | 4,598 | 0.0348 | 160 |
| `NarrAcctHardVerified_strict` | 4,598 | 0.1053 | 484 |

会计入表样本太少：2024 年只有 22 个 `BookEntry=1`。这 22 家在 accounting-augmented hard score 中全部进入高硬能力组，其中 9 家属于高叙事，13 家属于低叙事。

`HardAcct` 对 `HardBase` 的增量有限：

- `corr(hard_base_pct_y, hard_acct_pct_2024) = 0.9909`
- `corr(NarrHardGap_base, NarrAcctHardGap) = 0.9937`

所以数据资产入表可以作为“会计验证”增强项，但目前不能单独撑起主 X。

## 4. 回归设定

全样本：

```text
Y_t = L.X + controls + firm FE + year FE, cluster(IndYear_num)
```

2024 横截面：

```text
Y_2024 = X_2024 + controls + industry FE, robust SE
```

控制变量沿用项目已有口径：

```text
Size Lev ROA TobinQ Age Growth IndepRatio Dual Top1Share SOE CFO
```

当 Y 本身是 `TobinQ` 时，控制变量中剔除 `TobinQ`。

## 5. 主要信号

### 5.1 全样本 `NarrHardGap_base`

`NarrHardGap_base` 可以理解为“叙事强度超过硬能力代理的程度”。主要结果：

| Y | coef | t | N | 解释 |
|---|---:|---:|---:|---|
| `TobinQ` | -0.1955 | -8.74 | 37,294 | 估值折价信号很强 |
| `Analyst` | 0.1471 | 5.33 | 37,294 | 分析师关注上升 |
| `RatingDisp` | 0.0262 | 3.28 | 22,842 | 评级分歧上升 |
| `ReportFreq` | 0.0704 | 2.32 | 24,046 | 研报频率上升 |
| `PriceDelay` | -0.0081 | -2.67 | 37,294 | 不适合继续主打定价效率 |
| `TFP` | 0.0635 | 2.60 | 5,951 | 方向不支持“空喊损害效率”的简单故事 |
| `SCConc` | -1.7317 | -5.25 | 36,321 | 供应链集中度下降，机制解释需谨慎 |
| `SuppConc` | -1.6136 | -4.20 | 34,366 | 同上 |

连续差值最强的是 `TobinQ`、分析师关注/分歧、供应链集中度。`PriceDelay` 虽显著，但不建议作为主 Y，因为它又回到定价效率赛道。

### 5.2 全样本离散型 washing/hushing

`NarrHardWashing`：

| Y | coef | t | N |
|---|---:|---:|---:|
| `TobinQ` | -0.0609 | -4.21 | 37,294 |
| `SA` | -0.0013 | -1.99 | 37,294 |

`NarrHardWashing_strict`：

| Y | coef | t | N |
|---|---:|---:|---:|
| `RatingDisp` | 0.0356 | 3.50 | 22,842 |
| `Analyst` | 0.0721 | 2.05 | 37,294 |
| `AuditFee` | 0.0194 | 1.96 | 37,211 |
| `TobinQ` | -0.0747 | -2.86 | 37,294 |
| `SCConc` | -1.2987 | -3.64 | 36,321 |
| `SuppConc` | -1.6358 | -3.91 | 34,366 |

`NarrHardHushing` 方向大体相反：低叙事但高硬能力企业的 `TobinQ` 更高、分析师关注/分歧更低。这个对“washing/hushing”框架是有用的，因为它不是单纯披露强度，而是“说得多但硬能力弱”和“硬能力强但少说”的对照。

### 5.3 2024 accounting-augmented 版本

`NarrAcctHardGap`：

| Y | coef | t | N | 解释 |
|---|---:|---:|---:|---|
| `ForecastDisp` | -0.0630 | -3.54 | 2,024 | 方向不支持“分歧更大” |
| `Analyst` | 0.1293 | 2.03 | 4,593 | 分析师关注上升 |
| `ReportFreq` | 0.1487 | 2.50 | 2,625 | 研报频率上升 |
| `InstStable` | -0.4847 | -2.19 | 4,591 | 长期机构稳定性下降 |
| `SCConc` | -6.4974 | -8.57 | 4,571 | 供应链集中度下降 |
| `SuppConc` | -7.0046 | -7.80 | 4,565 | 同上 |

`NarrAcctHardWashing`：

| Y | coef | t | N |
|---|---:|---:|---:|
| `ForecastDisp` | -0.0351 | -2.89 | 2,024 |
| `SA` | -0.0060 | -2.76 | 4,593 |
| `SCConc` | -2.9341 | -5.13 | 4,571 |
| `SuppConc` | -3.0792 | -4.57 | 4,565 |

`BookEntry` 单独作为 X 基本不行。只有 `InstHold` 边际显著，且 22 个正样本太少，不适合单独主打。

## 6. 判断

这个 X 比“年报数据要素披露”更像一个能写的变量，因为它把特刊里的 data/AI economy 叙事转成了“叙事与可验证基础是否一致”的问题。它也和 AI washing 文献的构造方式相近：披露侧减去行动/投入/硬证据侧。

但以目前数据看，有两个限制：

1. `HardBaseRaw` 仍然偏粗，更多是行业/地区/政策潜力，不是真实企业级硬能力。
2. 数据资产入表目前只有 2024 年 22 个正样本，适合做会计验证/增强项，不适合作为主识别。

最稳的下一步不是直接写“数据资产入表”，而是：

> 以 `NarrHardGap_base` 或 `NarrHardWashing` 为主 X，先讲“数据要素叙事-硬能力错配”；数据资产入表作为 2024 会计验证补充。

更好的最终版应把硬能力侧补强为企业级证据：

- 数据/AI/软件相关专利；
- 软件著作权；
- 数据相关招聘；
- AI/data capital expenditure；
- 数据产品、数据交易、数据资产确权等外部可验证事件。

如果这些能接上，X 的概念会比单纯年报披露更干净，也比直接拿 `DataAsset` 入表更稳。

## 7. 暂定可写 Y

不建议继续主打：

- `PriceDelay`：又回定价效率，且用户已明确觉得这条拥挤。
- `BookEntry` 单独作为 X：样本太小。

比较可继续看的 Y：

1. `TobinQ`：全样本最稳，适合写“资本市场估值折价/真实性折价”，但需要处理反向因果和行业结构。
2. `Analyst / RatingDisp / ReportFreq`：适合写“信息中介反应”，但方向不是纯粹坏消息，需要包装成关注与分歧，而不是简单 forecast error。
3. `AuditFee`：只在 strict washing 中边际显著，可作为验证成本/审计关注的辅助结果，不适合单独主线。

当前最像主线的一句话：

> 当企业年报中的数据要素叙事超过其可验证硬能力时，资本市场并非简单奖励这种叙事，反而表现为估值折价和外部信息中介更强的关注/分歧；2024 年数据资产入表能作为会计验证补充，但现阶段样本不足以单独承担识别。
