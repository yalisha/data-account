# Codex handoff — 机制重构文献搜索（方案 B + 资产配置复活）

**产出**：`/Users/mac/computerscience/0做完了/15会计研究/v2/literature_mechanism_rebuild.md`（markdown 综述报告）+ `/Users/mac/computerscience/0做完了/15会计研究/v2/literature_mechanism_rebuild.bib`（BibTeX）

**不要做**：不写 Zotero / Notion / Linear 等外部系统；只产出本地 md + bib。不写正文草稿，只做文献综述素材。

**上下文**：
合作老师对当前 4 机制（Amihud 流动性 + CashFlowVol + SCConc + Comparability）提了两条意见——缺乏创新（怀疑别人做过）、机制太散（要统一到一个与数据要素强相关的理论下）。Claude 和用户已决定方案 B：砍 Amihud，保留 CashFlowVol + SCConc + Comparability 三机制，理论主线统一到"数据要素作为生产要素的价值释放"（Jones-Tonetti 2020 + Farboodi-Veldkamp 2021 + Crouzet-Eberly-Eisfeldt-Papanikolaou 2022）+ 数据要素政策情境。同时保留资产配置作为可能的第四候选——前版 FinRatio（朱康 11 项加总）Step 2 完全 null，但换口径 / 换 MV 还有空间。

**你的任务**：围绕以下 4 条主线做文献检索 + 素材整理，重点回答两个问题——(1) 别人是否已做过我们的机制变量？(2) 有哪些新角度 / 替代 MV 可以尝试？

---

## 检索工具

- 优先 `mcp__semantic-scholar__search_papers` / `search_snippets` / `get_paper_references` / `get_paper_citations`
- 补充 `mcp__zotero__zotero_search_items`（查用户已有库存）+ `mcp__zotero__zotero_semantic_search`
- 必要时 WebSearch / WebFetch（CNKI 中文文献搜顶刊《会计研究》《经济研究》《管理世界》《金融研究》《中国工业经济》数据要素 / 数字化相关论文）

## 检索主线

### 主线 1：数据要素 / 数据作为生产要素的理论基础

**目标**：找清楚"数据作为生产要素"的经典理论文献 + 实证 paper 用了什么机制变量，判断我们的 CashFlowVol + SCConc 有没有被抢跑。

**关键词组合（英文）**：
- "data as production factor" + productivity
- "non-rival data" + firm value
- "data capital" + operational uncertainty
- "data-driven decision making" + firm performance
- "information capital" + intangibles + valuation

**必查锚点论文（从这些拉 references / citations）**：
- Jones, C. & Tonetti, C. (2020). Nonrivalry and the Economics of Data. AER.
- Farboodi, M. & Veldkamp, L. (2021/2023). A Model of the Data Economy. 工作论文 / Restud.
- Crouzet, N., Eberly, J., Eisfeldt, A., Papanikolaou, D. (2022). The Economics of Intangible Capital. JEP.
- Brynjolfsson, E., Hitt, L., Kim, H. (2011). Strength in Numbers: How Does Data-Driven Decisionmaking Affect Firm Performance?
- Tambe, P. (2014). Big data investment, skills, and firm value. Management Science.
- Veldkamp, L. (2023). Valuing Data as an Asset. RFS / working paper.
- Peters, R. & Taylor, L. (2017). Intangible capital and the investment-q relation. JFE.

**关键词组合（中文，走 CNKI / 百度学术）**：
- "数据要素" + 企业价值 / 企业绩效
- "数据资本化" + 投资效率
- "数据资产" + 会计
- "数据驱动" + 经营不确定性 / 风险
- "数据要素利用" + 披露 / 信息环境

**必查锚点论文（CNKI）**：
- 朱康 & 唐勇（2025）. 数据要素利用与企业金融资产配置. 会计研究.
- 蒋为, 龚思豪, 黄玖立（2023/2024）. 数据要素相关文献（经济研究）
- 戚聿东、肖旭等的数字经济理论文献
- 陈德球、陈运森等的数字化披露与信息环境文献

**重点整理**：
1. 这些文献**用了什么中介变量 / 机制变量**（列一张表：变量名 + 定义 + 样本 + 核心结论）
2. 如果某文献已用过 CashFlowVol 或供应链集中度作为数据要素 / 数字化的机制，标红——这是"被抢跑"的直接证据
3. 找 5-10 篇 paper 的 gap 表述，帮我们定位本文的 contribution

---

### 主线 2：CashFlowVol（现金流波动 / 基本面不确定性）作为机制变量

**目标**：(1) 在数据要素 / 数字化披露文献中查 CashFlowVol 有没有被用过；(2) 如果被用过是什么口径。

**关键词组合**：
- "digital transformation" + cash flow volatility / operating risk
- "data disclosure" + earnings smoothness / cash flow uncertainty
- "digitalization" + operational stability / performance variability
- "数字化转型" + 现金流波动 / 经营风险 / 盈余平滑
- "数据要素" + 经营不确定性 / 现金流

**锚点**：
- Epstein, L. & Schneider, M. (2008). Ambiguity, Information Quality, and Asset Pricing. JF.
- Dechow, P. & Dichev, I. (2002). The Quality of Accruals and Earnings. TAR.（盈余质量的母文献，跟 CashFlowVol 相近）

**产出**：列出 3-5 篇直接用 CashFlowVol 做数字化机制的 paper（如果找到），标明定义差异。我们的定义：3 年滚动标准差 / 资产总额。

---

### 主线 3：SCConc（供应链集中度）作为数字化 / 数据要素机制

**目标**：查在数字化 / 数据要素文献里 SCConc 做机制被用过没有；数据要素 → 供应链韧性的理论支撑。

**关键词组合**：
- "digital transformation" + supply chain concentration / customer concentration / supplier diversification
- "data disclosure" + supply chain risk / value chain
- "big data" + supply chain resilience / diversification
- "数字化转型" + 供应链集中度 / 客户集中度 / 供应商集中度
- "数据要素" + 供应链
- "数字化" + 价值链韧性

**锚点**：
- Cen, L., Dasgupta, S., Elkamhi, R., Pungaliya, R. (2016). Reputation and loan contract terms: The role of principal customers. RF.
- Ellis, J., Fee, C., Thomas, S. (2012). Proprietary costs and the disclosure of information about customers. JAR.
- 企业数字化转型 × 供应链相关 CNKI 文献

**产出**：查 3-5 篇数字化 / 数据 × 供应链 paper，判断"数据要素披露 → SCConc ↓"是我们原创还是已被做过。

---

### 主线 4：会计可比性（Comparability）在数字化 / 数据要素语境下

**目标**：查 De Franco 2011 后续文献，尤其是数字化披露 × 可比性的实证；理论上数据要素 → 可比性提升的逻辑是什么（是否只是"更多披露 → 更多可比"的弱逻辑）。

**关键词组合**：
- "accounting comparability" + disclosure quality
- "comparability" + digital transformation / data disclosure
- "De Franco" + extension / digital / intangibles
- "会计可比性" + 数字化 / 披露质量
- "可比性" + 数据要素

**锚点**：
- De Franco, G., Kothari, S.P., Verdi, R. (2011). The Benefits of Financial Statement Comparability. JAR.
- 陈运森等 / 袁蓉丽等中文可比性文献

**产出**：核心是回答——我们的"数据要素利用披露 → Comparability ↑"在理论上是否比"通用披露 → Comparability ↑"有**增量**？如果没有，可能要把 Comparability 降级到稳健性而不是主机制。

---

### 主线 5：资产配置机制复活 —— 非 FinRatio 口径的候选 MV

**目标**：前版朱康 11 项 FinRatio 失败（Step 2 FinRatio → Delay 完全 null）。找其他金融化 / 去金融化 / 主业投资的 MV 口径，理论支撑 + 实证证据。

**关键词组合**：
- "corporate financialization" + real investment / firm value
- "financial asset holding" + stock price informativeness / delay / synchronicity
- "企业金融化" + 定价效率 / 信息不对称 / 股价
- "脱实向虚" + 披露 / 信息环境
- "投资效率" + 数字化 / 数据要素

**锚点论文（按口径分类）**：
| 口径方向 | 代表文献 |
|---|---|
| 狭口径金融化（排除长期股权 + 投资性房地产） | 彭俞超（2017, 金融研究）；杜勇等（2017, 管理世界） |
| 交易性金融资产占比（短期套利动机） | 杜勇等（2019）；王红建等（2017） |
| 金融化增量（一阶差分） | 张成思、郑宁（2020, 世界经济）|
| 主业投资 vs 金融投资替代 | 王红建等（2017, 南开管理评论）"实业投资率 / 金融资产比"；彭俞超-黄志刚（2018, 经济研究） |
| 无形资产 / 资本支出 | Peters-Taylor 2017；国内数字无形资产相关文献 |
| 金融资产波动 | 查有无"金融资产稳定性 / 资产组合波动"的定义 |

**产出**（最重要）：
1. 列一张**候选 MV 清单表**：MV 名 + 公式定义 + 代表文献 + 是否在数字化 / 数据要素文献中出现过
2. 特别关注：**彭俞超 2017 狭口径 FinRatio** 和**杜勇 2017 交易性金融资产占比**的定义（CSMAR 字段）
3. 查"金融资产 → 股价信息效率 / 定价效率"的实证证据——有没有 paper 直接跑过 FinRatio → PriceDelay / SYNCH？这直接影响我们 Step 2 失败是"机制不存在"还是"口径不对"

---

## 产出文件结构（`literature_mechanism_rebuild.md`）

```markdown
# 机制重构文献综述（方案 B + 资产配置复活）

## 1. 数据要素 / 数据作为生产要素的理论基础
### 1.1 核心理论文献（含定义、主要结论、引用）
### 1.2 已有实证 paper 用过的机制变量清单表（重要）
### 1.3 研究 gap 与本文定位

## 2. CashFlowVol 作为机制 — 是否被抢跑
### 2.1 数字化 / 数据要素 × CashFlowVol 的文献
### 2.2 口径差异
### 2.3 判断：原创 / 部分已做 / 完全重复

## 3. SCConc 作为机制
### 3.1 数字化 × 供应链的文献综述
### 3.2 口径差异
### 3.3 判断

## 4. Comparability 作为机制
### 4.1 De Franco 2011 后续 + 数字化语境扩展
### 4.2 理论增量判断
### 4.3 建议去留（主机制 vs 稳健性）

## 5. 资产配置候选 MV 清单（复活路径）
### 5.1 狭口径 FinRatio（彭俞超 2017）
### 5.2 交易性金融资产占比（杜勇 2017）
### 5.3 ΔFinRatio（张成思-郑宁 2020）
### 5.4 研发 / 实业投资 vs 金融替代（王红建 2017）
### 5.5 金融资产占比 × 波动
### 5.6 哪些 MV 在数字化 / 数据要素文献中被验证过

## 6. 综合判断与建议
### 6.1 保留 / 剔除哪些机制
### 6.2 推荐的理论统一主线（供 Claude 写第二章）
### 6.3 资产配置复活的最优先 MV（top 3）
```

**BibTeX 文件**：所有引用的论文抓 bibtex 存 `literature_mechanism_rebuild.bib`。

---

## 验证 Gate（自检清单，跑完必须 pass）

- [ ] 主线 1-5 每条都至少有 5 篇具体文献（不是泛泛"这个领域有很多研究"）
- [ ] 每篇关键论文都抓了 bibtex
- [ ] 资产配置 MV 候选清单**至少 5 个**有文献锚点和 CSMAR 字段映射
- [ ] 明确回答了"CashFlowVol / SCConc / Comparability 是否被数据要素文献抢跑"
- [ ] 每条主线有判断，不是只列文献——我们要的是决策素材不是综述大纲
- [ ] **不**改项目其他文件，只产出 md + bib
- [ ] 报告 OPEN_QUESTIONS 段，列 3-5 条你在搜索中产生的不确定点

---

## 不要做

- 不跑 Stata、不碰 `.do` 文件、不碰数据
- 不改 outline.md / outline_paper.md
- 不写论文正文
- 不往 Zotero / Notion / Linear 同步
- 不做"综合改写建议"之类的发挥——只做素材收集

时间预算：semantic-scholar + zotero 检索应该 1-2 小时内能完成。超过 3 小时未结束在 OPEN_QUESTIONS 标记卡点，回来给 Claude / 用户。
