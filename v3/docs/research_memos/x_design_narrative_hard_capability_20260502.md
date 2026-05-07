# X 设计定稿：数据要素叙事-可验证硬能力错配

日期：2026-05-02

## 1. 定稿判断

当前 X 不再定义为“年报数据要素披露强度”，而是定义为：

> 企业年报中的数据要素叙事，与其可验证数据硬能力之间的错配程度。

核心变量名暂定：

```text
NarrHardGap
```

解释：

```text
NarrHardGap = 数据要素叙事强度 - 可验证硬能力
```

这个变量的含义不是“披露多不多”，而是“说得是否超过做得到的程度”。这比普通披露强度更接近 AI washing / data washing 文献，也更贴特刊的 artificial intelligence、data investment、digital economy 主题。

## 2. 为什么不再用单纯披露强度

单纯 `DU_kw` 的问题：

1. 文献已经很多，容易被归入“数字化转型披露/数据资产披露经济后果”。
2. 它只度量企业说了什么，不度量是否有能力支撑。
3. 如果 Y 是定价效率、分析师、融资约束，很容易被审稿人质疑为普通信息披露效应。

`NarrHardGap` 的优势：

1. X 本身带机制：叙事和实质之间是否一致。
2. 能自然连接 AI washing 的构造思路：disclosure minus action / investment / hard evidence。
3. 能把“数据资产入表”放进会计验证层，而不是把 2024 年 22 个入表样本硬撑成主识别。
4. 能形成三类企业：washing、hushing、verified，叙事空间比单一披露强度更大。

## 3. 当前可跑版本

脚本：

```text
src/stata/narrative_hard_capability_mismatch.do
```

当前样本：

```text
v1/data_stata/reg_sample_v18.dta
```

当前结果：

```text
results/data/narrative_hard_capability_mismatch_panel.dta
results/stata/narrative_hard_x_summary.csv
results/stata/narrative_hard_y_screen_panel.csv
results/stata/narrative_hard_y_screen_2024.csv
docs/research_memos/narrative_hard_capability_mismatch_pilot_report.md
```

### 3.1 叙事侧

当前用：

```text
DU_kw
```

含义：

```text
年报数据要素叙事强度
```

处理：

```text
disc_pct_y = within-year percentile(DU_kw)
```

注意：脚本中已经修正 Stata `egen rank(...), field` 的排序方向，确保原变量越大 percentile 越高。

### 3.2 硬能力侧 v0

当前可用的硬能力代理：

```text
HighTech
StrategicEmerging
DigEconCore
IndustryCluster
ln(1+LQ)
```

当前变量：

```text
HardBaseRaw = rowmean(z_HighTech, z_StrategicEmerging, z_DigEconCore, z_IndustryCluster, z_lnLQ)
hard_base_pct_y = within-year percentile(HardBaseRaw)
```

当前主 X：

```text
NarrHardGap_base = disc_pct_y - hard_base_pct_y
```

当前离散变量：

```text
NarrHardWashing  = high narrative, low hard capability
NarrHardHushing  = high hard capability, low narrative
NarrHardVerified = high narrative, high hard capability
```

建议主文先用：

```text
NarrHardGap_base
NarrHardWashing
NarrHardHushing
NarrHardVerified
```

strict 版本作为稳健性：

```text
NarrHardWashing_strict
NarrHardHushing_strict
NarrHardVerified_strict
```

## 4. 最终投稿版应该怎么做

当前 `HardBaseRaw` 只能算 v0。它最大的问题是：偏行业/地区/政策潜力，不够企业级。

最终投稿版应把硬能力侧升级成企业级、多来源、可验证的 `HardDataCapability`。

### 4.1 硬能力组件优先级

优先级 A：最应该补

```text
DataPatent
SoftwareCopyright
DataRelatedHiring
AIDataCapex
```

含义：

- `DataPatent`：数据、算法、人工智能、大数据、数据库、数据处理、数据安全等相关专利。
- `SoftwareCopyright`：软件著作权，尤其是数据平台、数据中台、算法系统、数据库系统、数据治理系统等。
- `DataRelatedHiring`：数据工程师、算法工程师、数据治理、数据安全、数据库、大数据平台等岗位招聘。
- `AIDataCapex`：AI / data / software / platform 相关资本化支出或数字化投资。

优先级 B：有则加入

```text
DataProduct
DataTransaction
DataAssetConfirmation
DigitalInfrastructure
```

含义：

- `DataProduct`：企业是否发布数据产品、数据服务、数据平台。
- `DataTransaction`：数据交易所挂牌、成交、数据产品登记。
- `DataAssetConfirmation`：数据确权、数据资源登记、数据知识产权登记。
- `DigitalInfrastructure`：云平台、数据库、ERP/MES/工业互联网平台等硬设施。

优先级 C：当前已有但不够强

```text
HighTech
StrategicEmerging
DigEconCore
IndustryCluster
LQ
```

这些可以保留为控制或补充维度，但不应长期作为唯一硬能力证据。

### 4.2 最终硬能力指数

推荐最终口径：

```text
HardDataCapability_it =
    rowmean(
        z(DataPatent_ln),
        z(SoftwareCopyright_ln),
        z(DataRelatedHiringShare),
        z(AIDataCapex_ln),
        z(DataProduct),
        z(DataTransaction),
        z(DataAssetVerification)
    )
```

如果不同数据年份覆盖不一致，采用分层版本：

```text
HardCap_core    = patents + software copyrights + hiring
HardCap_account = data asset booking / data resource recognition
HardCap_market  = data product / data transaction / data IP registration
```

主文用 `HardCap_core`，其他两个做增强和异质性。

## 5. X 的三种正式构造

### 5.0 2026-05-04 招聘细化试跑后的更新

已经处理招聘大数据：

```text
/Users/mac/computerscience/第三方资料/第三方数据资源/上市公司其他/上市公司招聘大数据2014-2026.3.rar
```

本轮结论：

```text
综合 hard capability 加入招聘后，结果稳，但和 direct v1 几乎重合。
纯招聘 hard capability 与 direct v1 有明显增量，更适合做“人力资本硬证据”分支。
```

关键证据：

| pair | rho |
|---|---:|
| `NHGap_hire_title` vs `NarrHardGap_direct_v1` | 0.9838 |
| `NHWash_hire_title` vs `NarrHardWashing_direct_v1` | 0.9927 |
| `HireCap_title` vs `HardCap_direct_v1` | 0.1654 |
| `NarrHireGap_title` vs `NarrHardGap_direct_v1` | 0.7006 |
| `NarrHireWash_title` vs `NarrHardWashing_direct_v1` | 0.8381 |

因此当前变量层级更新为：

```text
主 X：NarrHardGap_direct_v1 / NarrHardWashing_direct_v1 / NHWash_direct_v1_s
细化分支：NarrHireGap_title / NarrHireWash_title / NHireWash_title_s
稳健性：NHGap_hire_title / NHWash_hire_title
```

对应试跑报告：

```text
docs/research_memos/hiring_refined_hardcap_v1_report_20260504.md
```

### 5.1 连续差值

主变量：

```text
NarrHardGap_it = PctYear(DataNarrative_it) - PctYear(HardDataCapability_it)
```

优点：

- 解释直观。
- 不依赖量纲。
- 可以缓解不同年份数据要素叙事整体上升的问题。

推荐作为主回归 X。

### 5.2 分组变量

主文或机制中使用：

```text
DataWashing_it  = 1[DataNarrative high, HardCapability low]
DataHushing_it  = 1[HardCapability high, DataNarrative low]
DataVerified_it = 1[DataNarrative high, HardCapability high]
```

阈值建议：

```text
high = top 50% within year
low  = bottom 50% within year
```

稳健性：

```text
high = top 25%
low  = bottom 25%
```

这组三分法很重要，因为它能回答：

- 高叙事低能力是否被市场折价；
- 高能力低叙事是否被低估或被忽视；
- 高叙事高能力是否能获得正向验证。

### 5.3 残差错配

稳健性变量：

```text
DataNarrative_it = alpha + beta * HardDataCapability_it + controls + industry FE + year FE + error_it
ResidualGap_it = error_it
```

解释：

```text
在给定硬能力和企业特征后，超出正常水平的数据要素叙事。
```

优点：

- 审稿人如果质疑“差值变量太机械”，残差变量可以补上。
- 它更像“异常叙事”。

缺点：

- 解释不如 percentile 差值直观。
- 要避免把后果变量相关的控制过度放进第一阶段。

## 6. 会计验证怎么放

数据资产入表不建议单独做主 X。原因：

```text
2024 年 BookEntry=1 只有 22 个样本。
```

正确用法是验证层：

### 6.1 构造 accounting-augmented hard score

当前已试：

```text
HardAcctRaw = rowmean(z(HardBaseRaw), z(DataAsset_ln), z(BookEntry))
NarrAcctHardGap = PctYear(DataNarrative) - PctYear(HardAcctRaw)
```

结果显示：

```text
corr(NarrHardGap_base, NarrAcctHardGap) = 0.9937
```

这说明当前入表信息有概念价值，但由于样本太少，对全体排序增量有限。

### 6.2 更合适的论文用法

用作补充验证：

```text
High narrative + no data asset booking
High narrative + data asset booking
Low narrative + data asset booking
```

可以检验：

- 市场是否区分“只有叙事”与“叙事+会计确认”；
- 数据资产入表是否缓解叙事错配带来的折价；
- 低叙事但已入表企业是否存在 data hushing。

但由于样本小，这一块只能是 descriptive / supplemental evidence，不宜写成主识别。

## 7. 主线建议

当前最稳主线：

> 数据要素叙事-硬能力错配是否被资本市场和信息中介识别？

不要写：

```text
数据资产入表的经济后果
```

也不要写：

```text
数据要素披露是否提高定价效率
```

更适合写：

```text
When Data Narratives Outrun Hard Capabilities:
Evidence from Data-Element Disclosures in Annual Reports
```

中文题目暂定：

```text
数据要素叙事与可验证硬能力错配及其资本市场后果
```

更窄一点：

```text
数据要素叙事-硬能力错配、估值折价与信息中介反应
```

## 8. 目前最适合配的 Y

优先级 1：

```text
TobinQ
```

理由：

- 全样本结果最稳。
- 符合“市场不简单奖励叙事，而是折价识别错配”的故事。
- 避免再次回到定价效率拥挤赛道。

优先级 2：

```text
Analyst
RatingDisp
ReportFreq
```

理由：

- 能写信息中介反应。
- 高错配企业不是无人关注，而是引发更多关注和更高分歧。

优先级 3：

```text
AuditFee
```

理由：

- strict washing 中边际显著。
- 可以作为审计验证成本或会计信息风险的辅助结果。
- 不适合作为主 Y。

暂不建议：

```text
PriceDelay
ForecastDisp
SA
SCConc
SuppConc
```

理由：

- `PriceDelay` 回到定价效率，赛道拥挤。
- `ForecastDisp` 在 2024 accounting 版本方向不稳定。
- `SA` 的经济含义和方向需要重新确认，不适合现在主打。
- 供应链集中度结果显著但机制不直观，容易让主线发散。

## 9. 下一步执行清单

### Step 1：固定当前 X 版本

保留当前可跑版本：

```text
NarrHardGap_base
NarrHardWashing
NarrHardHushing
NarrHardVerified
```

在所有后续 Y 筛选和主回归中先统一使用这组变量。

### Step 2：补企业级硬能力数据

按优先级找：

```text
专利 -> 软件著作权 -> 招聘 -> AI/data 投资 -> 数据产品/交易/确权
```

先不要一口气追求完美，最少需要两个企业级硬证据：

```text
DataPatent
SoftwareCopyright
```

如果只能先拿到一个，就用它做 `HardCap_core_v1`，旧的 `HardBaseRaw` 做补充。

### Step 3：重构 X

得到新硬能力数据后，重新构造：

```text
HardCap_core
NarrHardGap_core
DataWashing_core
DataHushing_core
DataVerified_core
ResidualGap_core
```

### Step 4：跑 X validity tests

必须做：

```text
corr(DU_kw, HardCap_core)
tab DataWashing_core by industry/year
compare DataWashing_core vs DataWashing_base
reg HardCap_core DU_kw controls FE
```

目的：

- 证明 X 不是简单行业变量。
- 证明错配不是普通披露强度。
- 证明企业级硬能力确实带来新信息。

### Step 5：再定主 Y

先用当前结果暂定：

```text
TobinQ as main Y
Analyst / RatingDisp / ReportFreq as information-intermediary outcomes
AuditFee as accounting-risk auxiliary outcome
```

等企业级硬能力补齐后再最终决定。

## 10. 一句话版本

这篇文章的 X 应该被写成：

> 数据要素叙事是否超过了企业可验证的数据硬能力。

而不是：

> 企业是否披露了数据要素。

这是当前选题能从“普通年报披露经济后果”里跳出来的关键。
