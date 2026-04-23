# Codex 文献检索 TODO（v2-1 理论层现代化）

**创建时间**：2026-04-19
**目标**：为估值不确定性 + 数据要素披露框架补充近十年（2015-2024）核心文献，替换 Merton 1987 / Akerlof 1970 / Verrecchia 2001 过度主导引言的问题
**目标刊**：Economic Modelling（Elsevier, SSCI Q2）
**输出**：`v2/literature_modern.md` + Zotero 条目（带 citation key）

---

## 检索任务清单（按主题分组）

### 主题 1：无形资产估值与估值不确定性（最关键）

**检索关键词**：intangible capital valuation, intangible assets uncertainty, intangible assets Tobin's Q, R&D capitalization valuation

**目标论文**（已知必须命中）：
- Peters & Taylor (2017, JFE) — "Intangible capital and the investment-q relation"
- Crouzet, Eberly, Eisfeldt, Papanikolaou (2022, QJE) — "The economics of intangible capital"
- Ewens, Peters, Wang (2024, RFS) — "Measuring intangible capital with market prices"
- Lev & Gu (2016, 书) — The End of Accounting
- Falato, Kadyrzhanova, Sim, Steri (2022, JF) — "Rising intangible capital, shrinking debt capacity"

**请补充检索**：2018-2024 期间 JF / JFE / RFS / JAE / JAR / TAR 上"intangibles + valuation uncertainty / disclosure / price efficiency"主题的重要论文，需 Semantic Scholar 按 citation count 排序取 top 15

### 主题 2：数据作为经济要素（理论锚）

**检索关键词**：economics of data, nonrival data, data as asset, information good economics

**目标论文**（已知必须命中）：
- Jones & Tonetti (2020, AER) — "Nonrivalry and the Economics of Data"
- Farboodi, Mihet, Philippon, Veldkamp (2019, AER P&P) — "Big data and firm dynamics"
- Farboodi & Veldkamp (2022, Review of Economic Studies) — "A model of the data economy"（需核对刊名：也可能是 AER 或 REStud working paper）
- Abis & Veldkamp (2024, JF) — "The changing economics of knowledge production"
- Veldkamp & Chung (2024) — data economics survey

**请补充检索**：Veldkamp 团队近五年全部 working papers + AER / QJE / JF / RES 上"data economy / big data firm"主题 top-cited 文献

### 主题 3：现代披露理论（替换 Verrecchia 2001）

**检索关键词**：disclosure processing cost, voluntary disclosure intangibles, disclosure price efficiency, soft information disclosure

**目标论文**（已知必须命中）：
- Blankespoor, deHaan, Marinovic (2020, JAE) — "Disclosure processing costs, investors' information choice, and equity market outcomes"
- Bertomeu & Marinovic (2016, JAR) — "A theory of hard and soft information"
- Bushee, Gow, Taylor (2018, JAR) — "Linguistic complexity in firm disclosures"
- Chen, Cohen, Gurun, Lou, Malloy (2022, JFE) — "Shrouded transaction costs"
- Israeli, Kasznik, Sridharan (2022, JAR) — information dissemination via social media

**请补充检索**：2018-2024 JAE / JAR / TAR 上"disclosure + price efficiency / investor attention / intangibles"主题 top-cited 文献

### 主题 4：模糊性定价与估值不确定性（金融理论支撑）

**检索关键词**：ambiguity aversion asset pricing, Knightian uncertainty stock returns, valuation uncertainty risk premium

**目标论文**（已知必须命中）：
- Epstein & Schneider (2008, JF) — "Ambiguity, information quality, and asset prices"
- Illeditsch (2011, JF) — "Ambiguous information, portfolio inertia, and excess volatility"
- Brenner & Izhakian (2018, JFE) — "Asset pricing and ambiguity"
- Bianchi & Tallon (2019) — ambiguity and asset pricing
- Izhakian (2020, JFQA) — ambiguity measurement

**请补充检索**：2015-2024 JF / JFE / RFS / JFQA 上"ambiguity + asset pricing / disclosure / information quality"主题论文

### 主题 5：AI / 机器学习文本与披露（方法论支撑）

**检索关键词**：textual analysis annual report, LLM financial disclosure, machine learning disclosure quality, semantic disclosure measure

**目标论文**（已知必须命中）：
- Loughran & McDonald (2011, JF) — 金融文本情感词典
- Hoberg & Phillips (2016, JPE) — 文本相似度产品市场
- Li, Mai, Shen, Yan (2021, RFS) — machine learning in accounting
- Chen, Cohen, Lou, Malloy (2020, MS) — 文本驱动的信息事件

**请补充检索**：2020-2024 年"LLM / large language model / transformer + accounting disclosure / textual analysis"主题论文 top 10

### 主题 6：EM 本刊互引（投稿偏好，必须 3-5 篇）

**检索范围**：Economic Modelling 2018-2024
**检索关键词**：disclosure, intangible, price efficiency, information asymmetry, China listed firms

**要求**：
- 按 citation count 排序取 top 20
- 过滤出与 disclosure / intangibles / China / information asymmetry 主题最相关的 8-10 篇
- 标注每篇在 v2-1 稿件中可用的位置（引言 / 理论 / 实证对照）

### 主题 7：中国情境文献（仅用于"中国情境"段落）

**检索关键词**：数据要素 披露 中国，数字经济 上市公司 披露质量，数据资产 估值

**要求**：
- 限定 2022-2025 中文 CSSCI 期刊
- 目标会计研究 / 管理世界 / 金融研究 / 经济研究 / 中国工业经济
- 10-15 篇核心文献，按主题分类
- 特别注意：张新民团队数据资产披露系列、黄群慧数字经济、陈诗一数字化转型

---

## 产出格式（请严格按此输出 `v2/literature_modern.md`）

```markdown
# v2-1 现代文献清单（2015-2024）

## 按主题分组

### 主题 1：无形资产估值
| # | 作者 (年份) | 标题 | 刊物 | Citation | 核心贡献（一句话） | 用于哪条假说 |
|---|---|---|---|---:|---|---|
| 1.1 | Crouzet et al. (2022) | The economics of intangible capital | QJE | 234 | 无形资本估值偏差与企业价值 | H1 引言 + 2.1 理论 |
| 1.2 | ... | ... | ... | ... | ... | ... |

### 主题 2：数据作为经济要素
...

### 主题 3：现代披露理论
...

（依此类推）

## 按假说分配
- H1（主效应）必用文献：主题 1.1 / 1.3 / 2.1 / 3.1 / 4.1
- H2a（价值链）必用文献：主题 1.2 / 2.2 / 2.3
- H2b（质量 direct+joint）必用文献：主题 3.1 / 3.2 / 5.1 / 5.3
- H2c（WashGap）必用文献：主题 3.2 / 3.3（信息模糊、披露复杂度）
- H3（下游 channels）必用文献：主题 4.1 / 4.2（估值不确定性 vs 模糊性）
- H4（异质性）必用文献：主题 1.4 / 2.4（数据经济学动态）

## Zotero 操作
- 全部 35-40 篇通过 DOI 加入 Zotero，collection 名为 "v2-1_theory_modernization"
- citation key 格式：作者首字母+年份+首词（例：crouzet2022economics）
- 导出 BibTeX 存为 v2/literature_modern.bib
```

---

## 约束

1. **不要覆盖 Merton / Akerlof / Verrecchia 这三篇经典**，只做补充不做替换。经典文献保留在 naming_conventions.md 引用规范里
2. **每条必须能在 Semantic Scholar / Scholar Gateway 查到 DOI + citation count**，否则标注 `[需人工核对]`
3. **Citation count 截止 2025-12**，请按这个时间点的数据做排序
4. **不要引入 2015 年之前的文献**（除非是同一作者的序列工作如 Verrecchia 综述），主题 1-5 严格 2015+
5. **中文文献单独一章**，不混入主清单，避免翻译英文版时混乱

---

## 验证

Codex 交付后 Claude 会审核：
- [ ] 每个主题至少 5-8 篇，避免单主题空心
- [ ] 主题 1-5 全部英文文献，来自 SSCI / top 5 economics 期刊
- [ ] EM 本刊互引 ≥ 3 篇
- [ ] 中文文献 10-15 篇，CSSCI
- [ ] 所有条目带 DOI + citation count
- [ ] Zotero collection 已建，BibTeX 已导出
