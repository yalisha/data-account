# AI washing 构造方式与数据资产入表可用性判断

日期：2026-05-02

## 1. 别人的 AI washing 怎么构造

### 1.1 Zhou et al. (2026, Journal of Business Research)

本地 PDF：

```text
/Users/mac/computerscience/0做完了/15会计研究/v3/bib/origin/Zhou 等 - 2026 - Contractual arrangements and information consistency How ESG executive compensation incentives affe.pdf
```

核心思路：

```text
AI washing = standardized AI symbolic disclosure - standardized AI substantive investment
```

具体做法：

- “说”：年报 AI symbolic disclosure。作者用 AI 词典做关键词匹配和频率统计，并用类似 TF-IDF/文本长度调整的方式得到 AI 披露强度。
- “做”：AI substantive investment。作者使用 CSMAR 的人工智能投资数据，口径是企业财务报表中 AI 相关资本化支出；再减去行业-年均值，得到 `AI_invest`。
- AI washing 指标：`z(AI_disclosure) - z(AI_invest)`。
- 这个指标本质是 speech-action gap，不识别管理层主观动机，只识别行为结果上的“披露超过投入”。

这个做法和我们现在想做的“年报叙事 - 数据资产/能力端”非常接近。

### 1.2 Song et al. (2026, Finance Research Letters)

论文：AI washing: Strategic disclosure and backlash  
来源：[University of Edinburgh Research Explorer](https://www.research.ed.ac.uk/en/publications/ai-washing-strategic-disclosure-and-backlash/)

核心思路：

- 用 BERT 对 10-K 里的 AI 句子分类。
- 区分 forward-looking / planned adoption 的 `Will-Do AI` 和有可验证实施证据的 `Done AI`。
- `Done AI` 用产品部署、专业招聘、AI 相关收购、AI 专利等验证。
- 还使用 USPTO/PATSTAT 的 AI 专利数据作为外部验证。
- overclaiming 的典型定义是：AI disclosure 高于行业-年均值，但 AI patenting 低于行业-年均值。

这篇给我们的启发是：不要只做连续差值，还可以做四象限或二元 overclaim：

```text
High narrative + Low hard capability = washing / overclaim
High narrative + High hard capability = verified claim
Low narrative + High hard capability = hushing / underclaim
```

### 1.3 Sun et al. (2026, Technological Forecasting and Social Change)

论文：Unveiling AI washing: Bridging corporate technological gaps through a cognitive dissonance lens  
来源：[Strathprints](https://strathprints.strath.ac.uk/95265/)；PDF 可得。

核心思路：

```text
AI washing = z(AI disclosure keyword frequency) - z(AI patent grants)
```

具体做法：

- “说”：年报 AI 关键词频率。
- “做”：企业当年 AI 授权专利数量。
- 标准化后相减，数值越大表示 AI 宣传相对真实 AI 技术成果越高。
- 稳健性用 MD&A 部分 AI 关键词替代全文关键词，用专利申请替代授权专利。

这篇说明专利端已经是 AI washing 文献里很常见的“硬能力端”。

### 1.4 Liu and Li (2026, Finance Research Letters)

论文：The Impact of AI Washing on Enterprises' Access to Bank Loans  
来源：[ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1544612326004149)

公开摘要显示它也是从“words / information disclosure”和“deeds / actual practice”两个维度刻画 AI washing。也就是说，AI washing 文献已经基本收敛到：

```text
叙事/披露 - 实际行动/投资/能力
```

### 1.5 Xing et al. (2026, Technovation)

论文：AI technology, AI narrative, and firm value  
来源：[ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0166497225001816)

这篇不是直接叫 AI washing，但它把 AI technology 和 AI narrative 分开，并讨论二者脱钩。它进一步说明“技术能力”和“叙事表达”已经成为 AI 经济后果文献里的标准拆分。

## 2. 对我们的启发

别人构造 AI washing 的共识已经很清楚：

1. 不能只数 AI/数据词。
2. 必须有一个“做”的端口。
3. “做”的端口最好是外部可验证或财务可观察数据。
4. 主变量可以是连续差值，也可以是四象限。

所以我们现在如果继续只用 `DU_kw` 或 `DU_sub_ln`，就会显得弱。更有希望的是：

```text
Data narrative in annual report - hard data capability / hard data investment / balance-sheet recognition
```

## 3. 数据资产入表怎么样

结论：能用，但不适合直接单独当主变量；更适合作为“硬验证端”或 2024 后的验证机制。

优点：

- 和特刊主题高度贴合：AI、data investment、digital economy。
- 比年报内部文本 `DU_sub_ln` 更硬，因为它进入会计确认和披露体系。
- 和 AI washing 文献里的 “capitalized AI expenditures” 口径非常接近，理论上可以作为 data-side substantive action。
- 可以提供制度背景：财政部《企业数据资源相关会计处理暂行规定》自 2024 年 1 月 1 日起施行，目标是规范数据资源相关会计处理并强化披露。

硬伤：

- 当前本地 `v1/data_parquet/panel.parquet` 里 `DataAsset` 只有 2024 年有正值，正样本 89 个。
- 只有一年正值，不能支撑标准双向固定效应面板主回归。
- 入表不是纯“真实能力”：未入表可能是没有数据资产，也可能是保守会计处理、确认条件不满足、商业秘密或合规风险导致不愿入表。
- 数据资产披露/数据资产经济后果文献已经不少，不能写成“数据资产入表有什么后果”。

## 4. 推荐用法

### 4.1 不推荐

不建议写：

```text
数据资产入表 -> 资本市场反应 / 估值 / 融资 / 定价效率
```

这个方向太像已有数据资产披露和入表经济后果文献，而且当前样本太短。

### 4.2 推荐

推荐把数据资产入表放进 X 的“硬验证端”：

```text
DataWashing_book = high data narrative + no/low data asset booking
VerifiedData_book = high data narrative + positive/high data asset booking
DataHushing_book = low data narrative + positive/high data asset booking
NarrativeBookGap = rank(data narrative) - rank(log(1 + DataAsset / TotalAssets))
```

更稳的写法是把它作为验证项：

```text
Main X = annual-report data narrative - multi-proxy hard data capability
DataAsset entry = accounting-verification component / 2024 validation test
```

也就是说，主能力端不要只靠 `DataAsset`，最好组合：

- 数据资产入表；
- 数字/AI/数据专利；
- 软件著作权；
- 数据岗位招聘；
- AI/数据资本化投资；
- 数据产品/数据交易/数据平台项目。

数据资产入表在这个组合里权重很高，但不应单独承担全部识别。

## 5. 目前最适合的论文口径

建议主线不要叫 AI washing，而叫：

```text
数据要素叙事-会计验证错配
```

或者英文：

```text
Data narrative-accounting recognition mismatch
```

更完整一点：

```text
When data narratives outpace accounting recognition: Evidence from Chinese listed firms
```

对应变量：

```text
Narrative side: annual-report data element / data asset / AI-data terms
Verification side: data asset booking + patent/job/software/investment proxies
Mismatch: z(narrative) - z(verification)
```

这样比直接写“数据资产入表的经济后果”干净，也比普通 AI washing 更贴特刊。

## 6. 下一步最小可跑方案

当前就能先做 2024 横截面试跑：

```text
TalkNoBook_2024 = high DU_kw or data narrative, DataAsset == 0
BookNoTalk_2024 = DataAsset > 0, low DU_kw or data narrative
VerifiedBook_2024 = high DU_kw or data narrative, DataAsset > 0
```

可能的 Y：

- `TobinQ`：市场是否奖励 TalkNoBook / VerifiedBook。
- `Analyst` / `ForecastDisp`：资本市场是否识别入表验证。
- `InvestIneff` / `TFP`：真实经营后果。
- `BankLoan` / 债务融资：如果有可用数据，可贴近 FRL 银行贷款文献，但 novelty 要小心。

但这只能是 pilot。真正写论文，要么补 2025 年入表数据，要么把 `DataAsset` 放进 multi-proxy hard capability index。
