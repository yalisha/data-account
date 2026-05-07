# HardCap direct v1 试跑报告

日期：2026-05-04

## 1. 本次做了什么

本次开始处理“小表直连版”硬能力数据，目标是把 X 从：

```text
年报数据要素叙事 - 行业/地区潜力
```

升级为：

```text
年报数据要素叙事 - 企业级可验证硬能力
```

没有处理 20G 全量专利库；本轮只使用已经按股票代码整理好的小表。

## 2. 新增脚本和输出

Python 构造脚本：

```text
src/python/build_hardcap_direct_v1.py
```

Stata MCP 回归脚本：

```text
src/stata/hardcap_direct_v1_screen.do
```

核心输出：

```text
results/data/hardcap_direct_v1_components.csv
results/data/hardcap_direct_v1_panel.dta
results/stata/hardcap_direct_v1_component_summary.csv
results/stata/hardcap_direct_v1_x_summary.csv
results/stata/hardcap_direct_v1_y_screen.csv
results/logs/hardcap_direct_v1_screen.log
results/logs/hardcap_direct_v1_screen_mcp.log
results/logs/hardcap_direct_v1_screen_mcp.smcl
```

说明：

- Python 只负责解压小表、清洗、聚合 firm-year。
- 回归和 X 筛选使用 Stata MCP 完成。

## 3. 数据源

本轮使用这些小表：

| 数据 | 文件 | 主要变量 |
|---|---|---|
| 数字发明专利授权 | `数字发明专利授权情况表（年）190317870.zip` | `diginvpatautnum` |
| 数字人力投入 | `数字人力投入计划统计表（月）185606015.zip` | `releasedemandtimes`, `recruitmentsnumber` |
| 数字资本投入 | `数字资本投入计划统计表（年）185545091.zip` | `investitemnum`, `investtotalamount` |
| AI 投资水平 | `人工智能投资水平131920031(仅供沪江大学使用).zip` | `AIInvestTotal`, `AIInvestLevel` |
| 上市公司专利明细 | `专利明细情况093018090(仅供哈佛大学使用).zip` | `PatentName`, `ApplicationDate`, `GrantDate` |

专利明细仅使用专利名称做数据/AI/软件关键词筛选，没有使用 20G 全量专利库的摘要、IPC、主权项。

## 4. 组件覆盖

| component | nonmissing | nonzero | firms nonzero | years |
|---|---:|---:|---:|---|
| `DigInvPatAut` | 56,315 | 1,642 | 747 | 2011-2024 |
| `DigHumanDemand` | 19,617 | 9,063 | 3,355 | 2022-2026 |
| `DigHumanRecruit` | 19,617 | 9,051 | 3,351 | 2022-2026 |
| `DigCapexAmount` | 56,315 | 717 | 659 | 2011-2024 |
| `AIInvestTotal` | 65,873 | 53,203 | 5,577 | 2007-2025 |
| `AIInvestLevel` | 53,223 | 50,671 | 5,540 | 2007-2025 |
| `PatentTitleDataApp` | 67,621 | 932 | 346 | 1980-2025 |
| `PatentTitleDataGrant` | 67,621 | 738 | 311 | 1980-2025 |

注意：

1. 数字人力投入只有 2022 年后覆盖，所以不能把 2011-2021 缺失误填为 0；脚本已处理为覆盖期外缺失。
2. 数字发明专利授权在 2024 年样本中为 0，原因可能是“当年申请且已授权”在近年存在授权滞后。本轮主回归用 `L.X`，2024 的结果主要吃 2023 的 X。
3. direct hard capability 很稀疏，说明这个版本比 `HardBaseRaw` 更像“硬证据”，但也更容易把大量企业划为“叙事强、硬证据弱”。

## 5. 新硬能力指数

本轮构造三版：

### 5.1 主版本

```text
HardCap_direct_v1 =
    rowmean(
        z(DigInvPatAut_ln),
        z(DigHumanDemand_ln),
        z(AIInvestLevel),
        z(DigCapexAmount_ln),
        z(PatentTitleDataApp_ln)
    )
```

对应：

```text
NarrHardGap_direct_v1 = PctYear(DU_kw) - PctYear(HardCap_direct_v1)
NarrHardWashing_direct_v1
NarrHardHushing_direct_v1
NarrHardVerified_direct_v1
```

### 5.2 金额版本

```text
HardCap_direct_amt =
    rowmean(
        z(DigInvPatAut_ln),
        z(DigHumanDemand_ln),
        z(AIInvestTotal_ln),
        z(DigCapexAmount_ln),
        z(PatentTitleDataApp_ln)
    )
```

### 5.3 专利-人力版本

```text
HardCap_direct_patent_human =
    rowmean(
        z(DigInvPatAut_ln),
        z(DigHumanDemand_ln),
        z(PatentTitleDataApp_ln)
    )
```

## 6. 与旧 base 版的关系

新 hard capability 和旧 `HardBaseRaw` 的相关性很低：

| pair | corr |
|---|---:|
| `HardCap_direct_v1` vs `HardBaseRaw` | 0.0321 |
| `HardCap_direct_amt` vs `HardBaseRaw` | 0.0284 |
| `HardCap_direct_patent_human` vs `HardBaseRaw` | 0.0176 |

新 gap 和旧 gap 有中等相关：

| pair | corr |
|---|---:|
| `NarrHardGap_direct_v1` vs `NarrHardGap_base` | 0.3875 |
| `NarrHardGap_direct_amt` vs `NarrHardGap_base` | 0.3831 |
| `NarrHardGap_patent_human` vs `NarrHardGap_base` | 0.5134 |

判断：

```text
direct v1 不是旧行业/地区潜力变量的复制品。
```

这对论文很重要，因为它说明新的 X 确实带来了企业级硬证据。

## 7. X 分布

| X | N | mean | positive count |
|---|---:|---:|---:|
| `HardCap_direct_v1` | 43,735 | -0.0002 | 4,042 |
| `HardCap_direct_amt` | 43,735 | -0.0020 | 9,781 |
| `HardCap_direct_patent_human` | 43,735 | 0.0020 | 1,904 |
| `NarrHardGap_direct_v1` | 43,735 | 0.3075 | 33,714 |
| `NarrHardWashing_direct_v1` | 43,735 | 0.3901 | 17,062 |
| `NarrHardHushing_direct_v1` | 43,735 | 0.1047 | 4,577 |
| `NarrHardVerified_direct_v1` | 43,735 | 0.1099 | 4,807 |

解释：

direct v1 的硬证据比较稀疏，所以 `NarrHardGap_direct_v1` 多数为正。这不是坏事，但写作时要明确：这个 X 更像“企业是否存在可验证硬能力不足以支撑数据要素叙事”的严格度量。

## 8. 主结果信号

回归设定：

```text
Y_t = L.X + controls + firm FE + year FE, cluster(IndYear_num)
```

### 8.1 连续 gap：`NarrHardGap_direct_v1`

| Y | coef | t | N | 判断 |
|---|---:|---:|---:|---|
| `TobinQ` | -0.1673 | -9.08 | 37,294 | 最强主结果 |
| `Analyst` | 0.0834 | 3.84 | 37,294 | 信息中介关注增加 |
| `AuditFee` | 0.0187 | 3.20 | 37,211 | 审计验证成本/风险上升 |
| `SA` | -0.0039 | -4.06 | 37,294 | 方向需谨慎，不建议主写 |
| `PriceDelay` | -0.0087 | -3.76 | 37,294 | 显著但不建议回定价效率主线 |
| `TFP` | 0.0459 | 2.64 | 5,951 | 不支持“空喊损害生产率” |
| `SCConc` | -1.1963 | -4.67 | 36,321 | 供应链结果显著但机制发散 |
| `SuppConc` | -0.9096 | -2.99 | 34,366 | 同上 |

### 8.2 离散 washing：`NarrHardWashing_direct_v1`

| Y | coef | t | N |
|---|---:|---:|---:|
| `TobinQ` | -0.0776 | -6.26 | 37,294 |
| `Analyst` | 0.0651 | 4.54 | 37,294 |
| `AuditFee` | 0.0163 | 4.34 | 37,211 |
| `PriceDelay` | -0.0062 | -4.06 | 37,294 |
| `SCConc` | -0.5177 | -3.29 | 36,321 |
| `SuppConc` | -0.5560 | -2.98 | 34,366 |

### 8.3 离散 hushing：`NarrHardHushing_direct_v1`

| Y | coef | t | N |
|---|---:|---:|---:|
| `TobinQ` | 0.0717 | 3.69 | 37,294 |
| `ForecastDisp` | 0.0176 | 3.01 | 19,126 |
| `TFP` | -0.0583 | -2.84 | 5,951 |
| `CashFlowVol` | 0.0017 | 2.02 | 37,294 |

最有用的是 `TobinQ` 方向：washing 折价，hushing 溢价/更高估值。这给三分法提供了对照。

### 8.4 strict washing

`NHWash_direct_v1_s`：

| Y | coef | t | N |
|---|---:|---:|---:|
| `TobinQ` | -0.0746 | -4.36 | 37,294 |
| `Analyst` | 0.0967 | 5.42 | 37,294 |
| `RatingDisp` | 0.0159 | 2.59 | 22,842 |
| `ReportFreq` | 0.0604 | 2.51 | 24,046 |
| `AuditFee` | 0.0187 | 3.40 | 37,211 |
| `SCConc` | -0.9163 | -4.58 | 36,321 |
| `SuppConc` | -1.1217 | -4.57 | 34,366 |

strict 版本反而更像论文故事：

```text
高叙事、低硬证据的企业，估值更低，分析师关注/分歧/研报频率更高，审计费用更高。
```

## 9. 本轮判断

这轮结果比 `HardBaseRaw` 更有价值。

原因：

1. 新 hard capability 来自企业级可验证证据，不再只是行业/地区潜力。
2. 与旧 `HardBaseRaw` 相关性只有 0.0321，说明它确实是新信息。
3. 主结果 `TobinQ` 稳定显著为负，washing 版本也显著为负。
4. `Analyst / RatingDisp / ReportFreq / AuditFee` 给出信息中介和审计验证反应。

但也有一个必须承认的限制：

```text
direct v1 的硬证据很稀疏，许多企业会被判定为高叙事低硬能力。
```

这不是致命问题，但最终版需要用招聘大数据和 20G 全量专利库进一步细化：

- 招聘大数据：把泛数字人力投入细化成数据/AI/算法/数据库岗位。
- 全量专利库：用标题、摘要、主权项、IPC、统一社会信用代码构造更精确的 `DataPatent_custom`。

## 10. 下一步建议

现在可以把主 X 从 `NarrHardGap_base` 升级为：

```text
NarrHardGap_direct_v1
NarrHardWashing_direct_v1
NarrHardHushing_direct_v1
NarrHardVerified_direct_v1
```

主 Y 暂定：

```text
TobinQ
```

辅助 Y：

```text
Analyst
RatingDisp
ReportFreq
AuditFee
```

不建议主写：

```text
PriceDelay
SCConc
SuppConc
SA
TFP
```

一句话主线：

> 当企业年报中的数据要素叙事超过其企业级可验证硬能力时，资本市场给予估值折价，信息中介和审计师表现出更高关注和验证成本。

这是目前最像能写成特刊论文的版本。
