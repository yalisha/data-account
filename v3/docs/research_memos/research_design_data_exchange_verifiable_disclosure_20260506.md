# 研究设计：数据交易所设立与企业数据要素披露可验证化

日期：2026-05-06

> 更新提示：本文档是 2026-05-06 早期研究设计 memo。当前主线已整理到 `docs/data_exchange_verifiable_disclosure/`。最新版主 Y 已从 `DU_kw / asset_trade_kw` 调整为剔除机构名词后的 `asset_trade_noinst_kw / strict_noinst_kw`，`verif_noinst_kw` 和真实 `BookEntry` 作为验证层。

## 0. 一页版结论

本文拟研究：

```text
城市数据交易所设立后，预先存在“数据要素叙事-硬能力错配”的企业，
是否会从泛化的数据要素叙事转向更可验证的数据要素披露？
```

更短的题目可以写成：

```text
数据交易所设立与企业数据要素披露可验证化
```

英文题目可暂定：

```text
Data Exchanges and Verifiable Data-Economy Disclosure:
Evidence from Firms with Prior Data-Washing Exposure
```

或者更锋利一点：

```text
Do Data Exchanges Discipline Data Washing?
Evidence from Verifiable Data-Economy Disclosure
```

核心识别不是普通的：

```text
数据交易所设立 -> 企业披露更多数据要素
```

而是三重差分：

```text
城市数据交易所设立 × 企业预先 data-washing 暴露
-> 后续可验证数据要素披露
```

当前最适合做主 Y 的变量是：

```text
DU_kw
asset_trade_kw
```

`strict_at_kw` 可作为更窄口径补充。`NarrHardGap_direct_v1` 和 `NarrHardWashing_direct_v1` 不建议当唯一主 Y，因为动态 pretrend 没过；它们更适合放在 supporting evidence 中，说明可验证化披露上升伴随错配收敛。

## 1. 研究问题

中国数据要素市场建设的一个核心制度变化，是城市数据交易所/数据交易中心在 2021-2024 年间分批设立。这类平台提供数据登记、合规审核、估值、挂牌、交易撮合等制度基础设施。对企业而言，数据交易所设立后，“数据要素”不再只是年报中的抽象战略叙事，而开始具有更强的可登记、可估值、可交易、可审查属性。

本文关注的问题是：

```text
当本地数据市场基础设施出现后，原先容易进行数据要素空泛叙事的企业，
是否会改变年报披露方式，增加更可验证的数据利用、数据资产、数据交易相关表达？
```

这里的重点不是“企业是否披露更多数据要素”，而是“披露是否更可验证”。这能避开已有数据资产披露、数字化转型披露、年报文本披露经济后果等拥挤路线。

## 2. 理论逻辑

### 2.1 数据交易所降低可验证披露的制度成本

数据交易所提供数据登记、确权、估值、合规审核、挂牌交易等服务。设立数据交易所后，企业可以更容易把数据资源、数据产品、数据利用场景和数据交易行为转化为可被外部观察的制度化表达。

因此，数据交易所可能提高企业年报中可验证数据要素披露的概率或强度。

### 2.2 数据交易所提高空泛叙事的相对成本

在缺乏本地数据市场基础设施时，企业可以用较低成本在年报中使用“数据要素、数据资产、AI、大数据、数字经济”等概念，但外部投资者、分析师和监管者较难判断这些表述是否有真实支撑。

数据交易所设立后，数据产品登记、挂牌、交易、估值和合规审核成为更明确的参照体系。企业继续进行泛化叙事而不提供更具体证据，可能面临更高的信息不一致成本。

因此，处理效应应主要出现在预先存在 data-washing 暴露的企业中。

### 2.3 披露可验证化不等于真实能力一定提升

本文不直接声称数据交易所短期内提高了企业真实数据能力。更稳妥的说法是：

```text
数据交易所设立促使预先 data-washing 暴露企业调整披露结构，
从泛化叙事转向更可验证的数据要素表达。
```

真实能力提升可以作为解释之一，但不是当前结果必须承担的强因果命题。

## 3. 核心变量

### 3.1 政策冲击：城市数据交易所设立

处理变量来自既有 `/16数据交易所` 项目的城市级数据交易所 DID：

```text
DID_city_ct = 1{企业注册城市 c 在年份 t 已设立数据交易所}
```

处理城市共 22 个，设立时间分布在 2021-2024 年。

基本窗口：

```text
2018-2024 年 A 股上市公司
```

### 3.2 企业预先 data-washing 暴露

预先暴露使用 2016-2017 年构造，确保早于 2018-2024 事件窗口：

```text
PreWashAny1617 = 1{企业在 2016 或 2017 年存在 NarrHardWashing_direct_v1}
PreStrictWashAny1617 = 1{企业在 2016 或 2017 年存在更严格口径 washing}
PreGap1617 = 2016-2017 年 NarrHardGap_direct_v1 均值
```

其中：

```text
NarrHardGap_direct_v1 = 年报数据要素叙事分位 - 企业级可验证硬能力分位
NarrHardWashing_direct_v1 = 高数据要素叙事、低可验证硬能力
```

硬能力 `HardCap_direct_v1` 来自企业级可验证数据，包括数字发明专利授权、数字人力投入、数字资本投入、AI 投资水平、专利名称中的数据/AI/软件关键词等小表聚合。它不是行业/地区潜力变量。

### 3.3 主因变量：可验证数据要素披露

建议主 Y 分三层：

```text
Y1: DU_kw
```

宽口径数据要素利用披露强度。它能捕捉企业年报是否更具体地谈到数据利用。

```text
Y2: asset_trade_kw
```

数据资产、数据交易、数据流通、数据产品等市场化/资产化相关表达。它更贴近数据交易所制度功能。

```text
Y3: strict_at_kw
```

更严格的数据资产/数据交易表达。该变量更窄，信号更干净，但样本稀疏，建议作为补充。

不建议把 `TobinQ`、`SYNCH`、`PriceDelay` 当主 Y。它们更像远端经济后果或定价效率结果，容易把论文拉回已有拥挤路线。

### 3.4 辅助结果：错配收敛

辅助 Y：

```text
NarrHardGap_direct_v1
NarrHardWashing_direct_v1
```

这些变量可以用来说明：可验证披露上升的同时，企业数据要素叙事与硬能力之间的错配有所下降。

但它们不适合做唯一主 Y，因为当前事件研究 pretrend 不够干净。

## 4. 识别设计

### 4.1 静态 DDD

主模型：

```text
Y_ict =
    beta DID_city_ct × PreWash_i
    + theta DID_city_ct
    + Controls_it
    + Firm FE
    + Year FE
    + error_ict
```

其中：

```text
Y_ict = 可验证数据要素披露
PreWash_i = PreWashAny1617 或 PreStrictWashAny1617
```

标准误聚类：

```text
cluster(city_id)
```

核心系数：

```text
beta
```

解释为：数据交易所设立后，预先 data-washing 暴露企业相对于其他企业，是否更明显地增加可验证数据要素披露。

### 4.2 动态 DDD

事件研究模型：

```text
Y_ict =
    sum_k beta_k 1{event_time = k} × PreWash_i
    + event-time main effects
    + Controls_it
    + Firm FE
    + Year FE
    + error_ict
```

省略基期：

```text
k = -1
```

重点检查：

```text
k = -3, -2 的联合 pretrend
k = 0, 1, 2, 3 的 post dynamics
```

### 4.3 Stacked not-yet / never-treated

为了避免 staggered TWFE 中 already-treated comparison 的问题，补充 stacked cohort 版本：

```text
每个处理 cohort 单独构造事件窗口
控制组只保留 not-yet-treated 和 never-treated
吸收 stack-firm FE 与 stack-year FE
```

当前 stacked static 结果支持主结论。

## 5. 当前初步结果

### 5.1 静态 DDD

以 `PreWashAny1617` 为企业预先暴露：

| Y | coef | t | 解释 |
|---|---:|---:|---|
| `DU_kw` | 0.1524 | 2.62 | 数据利用披露上升 |
| `asset_trade_kw` | 0.0438 | 2.68 | 数据资产/交易表达上升 |
| `strict_at_kw` | 0.0164 | 2.75 | 严格口径数据交易表达上升 |
| `NarrHardGap_direct_v1` | -0.0868 | -5.62 | 错配下降 |
| `NarrHardWashing_direct_v1` | -0.1194 | -6.70 | washing 下降 |

以 `PreStrictWashAny1617` 为企业预先暴露：

| Y | coef | t |
|---|---:|---:|
| `DU_kw` | 0.3394 | 4.62 |
| `asset_trade_kw` | 0.0956 | 3.55 |
| `strict_at_kw` | 0.0323 | 3.53 |
| `NarrHardGap_direct_v1` | -0.0560 | -4.36 |
| `NarrHardWashing_direct_v1` | -0.0894 | -7.63 |

### 5.2 更强固定效应

对 `PreWashAny1617`：

| 规格 | `NarrHardGap` | `NarrHardWashing` |
|---|---:|---:|
| baseline | -0.0868 / -5.62 | -0.1194 / -6.70 |
| city linear trend | -0.0863 / -5.13 | -0.1176 / -6.41 |
| province-year FE | -0.0859 / -5.08 | -0.1173 / -6.52 |
| industry-year FE | -0.1011 / -5.87 | -0.1295 / -7.18 |
| drop 2024 cohort | -0.0882 / -5.63 | -0.1207 / -6.73 |

静态错配收敛结果非常稳。

但机制型主 Y 对 industry-year FE 较敏感：

| Y | baseline coef/t | industry-year FE coef/t |
|---|---:|---:|
| `DU_kw` | 0.1524 / 2.62 | -0.0012 / -0.02 |
| `asset_trade_kw` | 0.0438 / 2.68 | 0.0135 / 0.91 |
| `strict_at_kw` | 0.0164 / 2.75 | 0.0070 / 1.37 |

这提示：可验证披露上升有行业年度共同冲击成分，写作中不能夸大。

### 5.3 Stacked not-yet / never-treated

| exposure | Y | coef | t | N |
|---|---|---:|---:|---:|
| `sdx_wany17` | `DU_kw` | 0.1646 | 2.72 | 65,534 |
| `sdx_wany17` | `asset_trade_kw` | 0.0445 | 2.67 | 65,530 |
| `sdx_wany17` | `strict_at_kw` | 0.0165 | 2.53 | 65,530 |
| `sdx_wany17` | `NarrHardGap_direct_v1` | -0.0803 | -4.55 | 65,534 |
| `sdx_wany17` | `NarrHardWashing_direct_v1` | -0.1064 | -5.49 | 65,534 |

说明静态结果不是 already-treated comparison 造成的。

### 5.4 动态 pretrend

Stacked event-study 的联合 pretrend 检验：

| exposure | Y | pretrend p | 判断 |
|---|---|---:|---|
| `wany17` | `DU_kw` | 0.5607 | 通过 |
| `wany17` | `asset_trade_kw` | 0.2248 | 通过 |
| `wany17` | `strict_at_kw` | 0.0348 | 边际不过 |
| `wany17` | `NarrHardGap_direct_v1` | 0.0072 | 不过 |
| `wany17` | `NarrHardWashing_direct_v1` | 0.000008 | 不过 |

因此，当前最适合主文的结果是：

```text
DU_kw 和 asset_trade_kw under PreWashAny1617
```

`strict_at_kw` 做补充，错配下降做 supporting evidence。

### 5.5 Leave-one-treated-city-out

对 `dx_wany17` 做 leave-one-treated-city-out：

| Y | mean coef | min coef | max coef | 5% significant share |
|---|---:|---:|---:|---:|
| `DU_kw` | 0.1521 | 0.0909 | 0.1690 | 22/22 |
| `asset_trade_kw` | 0.0437 | 0.0299 | 0.0511 | 21/22 |
| `strict_at_kw` | 0.0163 | 0.0099 | 0.0189 | 22/22 |
| `NarrHardGap_direct_v1` | -0.0868 | -0.0950 | -0.0739 | 22/22 |
| `NarrHardWashing_direct_v1` | -0.1195 | -0.1339 | -0.1094 | 22/22 |

结果不是北京、上海、深圳、广州等单一城市驱动。

## 6. 论文假设

### H1：数据交易所设立促进预先 data-washing 暴露企业的可验证数据要素披露

```text
相对于其他企业，2016-2017 年存在数据要素叙事-硬能力错配的企业，
在注册城市设立数据交易所后，更可能增加数据利用、数据资产、数据交易相关披露。
```

对应主检验：

```text
DID_city × PreWashAny1617 -> DU_kw
DID_city × PreWashAny1617 -> asset_trade_kw
```

### H2：披露可验证化伴随数据要素叙事错配收敛

```text
数据交易所设立后，预先 data-washing 暴露企业的数据要素叙事-硬能力错配下降。
```

对应辅助检验：

```text
DID_city × PreWashAny1617 -> NarrHardGap_direct_v1
DID_city × PreWashAny1617 -> NarrHardWashing_direct_v1
```

注意：H2 不建议写成主因果结论，因为错配 Y 的动态 pretrend 没完全通过。

### H3：制度功能越接近数据资产化/交易化，效果越强

可从两个方向写：

1. 因变量层面：`asset_trade_kw` 和 `strict_at_kw` 更直接反映数据交易所制度功能。
2. 异质性层面：预先 `PreStrictWashAny1617` 企业的反应更强。

## 7. 计划表格结构

### Table 1：描述统计

变量组：

```text
DID_city
PreWashAny1617
DU_kw
asset_trade_kw
strict_at_kw
NarrHardGap_direct_v1
NarrHardWashing_direct_v1
controls
```

### Table 2：主回归

主 Y：

```text
DU_kw
asset_trade_kw
strict_at_kw
```

核心 X：

```text
DID_city × PreWashAny1617
DID_city × PreStrictWashAny1617
```

### Figure 1：动态 DDD

画：

```text
DID event × PreWashAny1617 -> DU_kw
DID event × PreWashAny1617 -> asset_trade_kw
```

错配 Y 动态图可以放附录，避免主文暴露 pretrend 问题。

### Table 3：Stacked not-yet / never-treated

主 Y 仍用：

```text
DU_kw
asset_trade_kw
strict_at_kw
```

### Table 4：错配收敛辅助结果

Y：

```text
NarrHardGap_direct_v1
NarrHardWashing_direct_v1
```

定位：

```text
supporting evidence, not sole main outcome
```

### Table 5：稳健性

包括：

```text
city linear trend
province-year FE
drop 2024 cohort
no launch year
leave-one-treated-city-out
```

industry-year FE 可以作为非常强的吸收规格，但当前会吸收主 Y 的行业年度共同变化，建议放附录并谨慎解释。

## 8. 新意边界

本文不是普通的：

```text
数据资产披露 -> 企业价值
数字化转型披露 -> 定价效率
数据交易所 -> 股票同步性
AI washing -> 崩盘风险
```

本文的新意在于：

1. 把数据交易所作为数据要素市场基础设施冲击。
2. 企业层面不是简单处理组，而是预先 data-washing 暴露企业。
3. 主 Y 不是远端资本市场后果，而是披露结构的可验证化。
4. 错配变量来自“年报数据要素叙事 - 企业级可验证硬能力”，比普通关键词披露更接近 data washing。

最清楚的贡献表述：

```text
本文说明，数据要素市场基础设施不仅影响数据流通和资本市场定价，
也会改变企业如何在年报中表述数据要素活动：
原先叙事强、硬证据弱的企业，在本地数据交易所设立后，
更倾向于使用可验证的数据利用和数据交易相关披露。
```

## 9. 主要风险

### 风险 1：主 Y 仍可能被认为是文本披露强度

应对：

```text
把 DU_kw 与 asset_trade_kw 分开。
DU_kw 是宽口径，asset_trade_kw 是更贴交易所制度功能的窄口径。
```

写作中强调“可验证化”来自更具体、更制度化的数据资产/交易表达，而不是单纯字数增加。

### 风险 2：行业年度冲击

当前主 Y 在 industry-year FE 下变弱。说明行业年度共同冲击可能解释一部分可验证披露上升。

应对：

1. 不把 industry-year FE 作为唯一生死规格。
2. 解释数据交易所是城市制度冲击，industry-year FE 可能吸收掉一部分真实的城市-行业集聚效应。
3. 进一步补充行业组别趋势或数字行业排除检验。

### 风险 3：错配 Y 的 pretrend 不干净

应对：

```text
不把 NarrHardGap / NarrHardWashing 当主 Y。
把它们放到辅助表，作为错配收敛证据。
```

### 风险 4：数据交易所设立时间并非完全外生

应对：

1. 使用 firm FE 和 year FE。
2. 加 city linear trend、province-year FE。
3. stacked not-yet / never-treated。
4. leave-one-treated-city-out。
5. 后续可补：城市层预趋势、数字经济基础控制、处理城市匹配。

## 10. 需要外部评审重点判断的问题

请重点评价以下问题：

1. 这个题目是否比“年报数据要素披露 -> 企业价值/定价效率”有明显新意？
2. 主 Y 定为 `DU_kw` 和 `asset_trade_kw`，把错配下降作为 supporting evidence，是否合理？
3. “数据交易所促进预先 data-washing 暴露企业披露可验证化”这个叙事是否成立，还是仍然会被认为只是文本披露增加？
4. `industry-year FE` 吸收后主 Y 变弱，这个问题是否致命？有没有更好的解释或替代检验？
5. 英文题目用 `Do Data Exchanges Discipline Data Washing?` 是否过强？是否应改成更稳妥的 `Data Exchanges and Verifiable Data-Economy Disclosure`？
6. 这篇更像会计、金融，还是 digital economy / economic modelling 方向？特刊适配度如何？

## 11. 当前建议定稿方向

建议目前按以下方式推进：

```text
主线：
数据交易所设立 -> 预先 data-washing 暴露企业 -> 可验证数据要素披露上升

主 Y：
DU_kw + asset_trade_kw

辅助 Y：
strict_at_kw
NarrHardGap_direct_v1
NarrHardWashing_direct_v1

不主打：
TobinQ
SYNCH
PriceDelay
```

最稳妥的题目：

```text
数据交易所设立与企业数据要素披露可验证化
```

最稳妥的英文题目：

```text
Data Exchanges and Verifiable Data-Economy Disclosure:
Evidence from Firms with Prior Data-Washing Exposure
```

## 12. 外部 Pro 评审后的修订重点

外部评审认可当前主线：

```text
城市数据交易所设立 × 企业预先 data-washing 暴露
-> 更制度化、更可核验的数据要素披露
```

但建议进一步收紧三点。

### 12.1 把“可验证化”降格为“更可核验、更制度化、更具体”

不要把结果写成：

```text
企业真实数据能力提升
数据交易所治理了虚假陈述
```

更稳妥的表述：

```text
预先 data-washing 暴露企业调整披露结构，
在既有数据要素叙事中增加更具体、更制度化、更可核验的表达。
```

### 12.2 补充比例型/残差型 Y

为了避免 `DU_kw` 和 `asset_trade_kw` 被理解为“企业多写了几句数据”，应增加：

```text
asset_trade_share = asset_trade_kw / (DU_kw + small constant)
strict_at_share   = strict_at_kw / (DU_kw + small constant)
```

也可以构造残差型 Y：

```text
residualized asset_trade_kw after controlling for DU_kw / annual-report length / generic data narrative
```

这会把问题从“数据披露总量是否增加”推进到：

```text
更制度化、更可核验的数据资产/交易表达，在全部数据要素叙事中的占比是否上升。
```

### 12.3 新增更强城市内识别规格

建议新增：

```text
Y_ict = beta DID_city_ct × PreWash_i
      + Firm FE
      + City-Year FE
      + PreWash_i × Year FE
      + Controls_it
      + error_ict
```

这个规格吸收：

1. 城市-年份共同冲击，包括城市数字经济政策、营商环境变化、地方监管加强、数据交易所本身主效应。
2. 预先 data-washing 企业在全国层面的共同年度变化。

如果 `asset_trade_kw` 或比例型 Y 在这个规格中仍有方向一致结果，论文识别会明显升级。

### 12.4 单独处理 2024 数据资源会计规则

财政部 2024 年企业数据资源会计处理规则可能直接影响 `asset_trade_kw` 和 `strict_at_kw`。

需要新增至少两个检验：

```text
drop year 2024
add PreWash × 2024
```

更强版本：

```text
industry × 2024 FE
```

否则评审可能认为 2024 年结果来自全国统一会计规则，而不是数据交易所。

### 12.5 industry-year FE 的解释边界

当前 `industry-year FE` 后主 Y 变弱，这不是立即否决，但不能略过。

后续要做：

1. 宽行业-year FE 替代细行业-year FE。
2. 行业线性趋势。
3. 剔除软件、互联网、计算机、通信、传媒、金融科技等热门数字行业。

如果 `asset_trade_kw` 方向仍一致，即可把细行业-year FE 作为过强压力测试放附录。
