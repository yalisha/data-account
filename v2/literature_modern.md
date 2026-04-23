# OPEN QUESTIONS

1. `Semantic Scholar` 的 `snippet_search` 在本机 MCP 上对主题关键词和精确题名查询均出现超时；本轮检索改用 `search_authors_by_name` + `get_paper` + `get_citations` 的组合完成主题扩展，并以 OpenAlex/Crossref 做元数据回填。
2. 主题 2 指定的 `Farboodi & Veldkamp (2022, Review of Economic Studies)` 在当前可核验结果中仍主要对应 `2021 NBER Working Paper` 版本；正式发表刊名与年份仍需人工复核。
3. Zotero MCP 在本机是 `local-only mode`，读操作可用，但写入 collection / DOI 导入不可用；因此本轮采用“本地 `.bib` 导入 Zotero + Better BibTeX 回导出”的回退链路。
4. 主题 7 要求中文 CSSCI 文献同时带 DOI + citation count，但部分中文期刊在 Semantic Scholar / OpenAlex 上缺 DOI 或引用数据；无法稳定核验者统一标记 `[需人工核对]`。
5. spec 要求 citation count 截止 `2025-12`，但当前可访问接口仅返回当前累计引用，无法回溯历史快照；表内引用数均为 `2026-04-19` 当前可核验值。
6. 主题 1 的 `Crouzet et al. (2022) The Economics of Intangible Capital` 当前可核验来源为 `Journal of Economic Perspectives`，不是 spec 中写的 `QJE`。
7. 主题 1 的 `Ewens, Peters, Wang, Measuring Intangible Capital with Market Prices` 当前先核到的是 `2019 NBER working paper` 版本；spec 所写 `2024 RFS` 版本未稳定核到。
8. 主题 4 与主题 5 的经典锚点（如 `Epstein & Schneider 2008`、`Illeditsch 2011`、`Loughran & McDonald 2011`）与“英文文献 2015+”约束冲突；本清单只将它们保留为写作背景，不纳入主清单。
9. 主题 2 的 `Abis & Veldkamp` 条目在当前可核验结果中对应 `2023 Review of Financial Studies`，不是 spec 中写的 `2024 JF`。

# v2-1 现代文献清单（2015-2024）

注：Citation 列为 `2026-04-19` 当前可核验累计引用，不是 `2025-12` 历史快照。每个主题均先做 top 15 检索池，再筛 5 篇核心；表内仅保留核心结果。

## 按主题分组

### 主题 1：无形资产估值与估值不确定性
筛选备注：top 15 检索池中保留 5 篇核心；spec 指定的 5 篇全部命中，但 `Ewens et al.` 当前仅稳定核到 2019 NBER working paper 版本。
| # | 作者 (年份) | 标题 | 刊物 | Citation | 核心贡献（一句话） | 用于哪条假说 |
|---|---|---|---|---:|---|---|
| 1.1 | Peters & Taylor (2017) | Intangible capital and the investment-q relation | JFE | 1146 | 将无形资本资本化后重写 investment-q 关系，是“账面遗漏资产”进入估值框架的基准文献。 | H1 引言 + 2.1 理论 |
| 1.2 | Lev & Gu (2016) | The End of Accounting and the Path Forward for Investors and Managers | Book | 435 | 系统论证传统财报对无形资本与平台型价值创造的低覆盖，是“会计锚点失灵”的直接出处。 | H1 引言 + 问题提出 |
| 1.3 | Ewens, Peters & Wang (2019 WP) | Measuring Intangible Capital with Market Prices | NBER WP | 49 | 用市场价格反推无形资本存量，为“价格端识别账外资产”提供可操作方法。 | H1 2.1 理论 + H4 扩展 |
| 1.4 | Crouzet et al. (2022) | The Economics of Intangible Capital | JEP | 151 | 把无形资本放回宏观与公司金融共同框架，解释其对估值、融资与竞争结构的系统影响。 | H1 引言 + 2.1 理论 |
| 1.5 | Falato et al. (2022) | Rising Intangible Capital, Shrinking Debt Capacity, and the U.S. Corporate Savings Glut | JF | 188 | 说明无形资本上升会压缩债务容量并改变融资结构，支持“高无形资本=更高估值不确定性”的外推。 | H1 理论 + H4 异质性 |

### 主题 2：数据作为经济要素（理论锚）
筛选备注：top 15 检索池中保留 5 篇核心；`Farboodi & Veldkamp` 与 `Abis & Veldkamp` 的年份/刊名按当前可核验结果纠偏。
| # | 作者 (年份) | 标题 | 刊物 | Citation | 核心贡献（一句话） | 用于哪条假说 |
|---|---|---|---|---:|---|---|
| 2.1 | Jones & Tonetti (2020) | Nonrivalry and the Economics of Data | AER | 724 | 明确数据的非竞争性与排他权配置如何改变资源配置效率，是“数据不同于传统要素”的核心理论锚。 | H1 引言 + 2.2 理论 |
| 2.2 | Farboodi et al. (2019) | Big Data and Firm Dynamics | AEA P&P | 151 | 把大数据引入企业动态与竞争结构分析，解释数据积累为何改变规模收益与市场集中。 | H2a 价值链 + H4 异质性 |
| 2.3 | Farboodi & Veldkamp (2021 WP) | A Model of the Data Economy | NBER WP | 134 | 构建数据经济的一般均衡框架，说明数据规模、学习与利润分配如何共同作用于企业价值。 | H1 2.2 理论 + H4 动态 |
| 2.4 | Abis & Veldkamp (2023) | The Changing Economics of Knowledge Production | RFS | 115 | 将知识生产函数与数据/AI 驱动的知识复制联系起来，适合支撑“披露帮助市场理解数据生产能力”的论证。 | H1 引言 + H4 动态 |
| 2.5 | Veldkamp & Chung (2024) | Data and the Aggregate Economy | JEL | 108 | 给出数据经济学的综述性框架，适合在引言中概括“数据作为资产、投入和信息品”的三重角色。 | H1 引言 + 文献综述收束 |

### 主题 3：现代披露理论（替换 Verrecchia 2001 的现代补充）
筛选备注：top 15 检索池中保留 5 篇核心；spec 中 `Chen et al. (2022)` 与 `Israeli et al. (2022)` 当前未稳定解析到精确题名，故未进入核心清单。
| # | 作者 (年份) | 标题 | 刊物 | Citation | 核心贡献（一句话） | 用于哪条假说 |
|---|---|---|---|---:|---|---|
| 3.1 | Blankespoor, deHaan & Marinovic (2020) | Disclosure processing costs, investors’ information choice, and equity market outcomes: A review | JAE | 1041 | 把“披露有效性”从供给端推进到投资者处理成本端，是把披露与价格效率重新接起来的现代综述。 | H1 2.2 理论 + H2b |
| 3.2 | Bertomeu & Marinovic (2015) | A Theory of Hard and Soft Information | TAR | 178 | 区分 hard / soft information 的可验证性与均衡后果，为“数据利用披露兼具软硬信息属性”提供理论抓手。 | H2b 质量 + H2c WashGap |
| 3.3 | Bushee, Gow & Taylor (2017) | Linguistic Complexity in Firm Disclosures: Obfuscation or Information? | JAR | 447 | 把披露复杂度与信息含量/模糊性分开识别，是 H2b 与 H2c 的直接写作母文献。 | H2b 质量 + H2c WashGap |
| 3.4 | Bourveau et al. (2021) | The Role of Disclosure and Information Intermediaries in an Unregulated Capital Market: Evidence from Initial Coin Offerings | JAR | 146 | 强调披露与中介共同作用于价格发现，适合支撑“披露不是自动生效，而要经过市场处理”的论点。 | H1 2.2 理论 + H3 channels |
| 3.5 | Goldstein, Yang & Zuo (2023) | The Real Effects of Modern Information Technologies: Evidence from the EDGAR Implementation | JAR | 104 | 用 EDGAR 展示信息技术基础设施如何改变披露传播与市场响应，是“现代披露基础设施”最好的制度证据。 | H1 理论 + H3 channels |

### 主题 4：模糊性定价与估值不确定性（金融理论支撑）
筛选备注：按“英文文献 2015+”约束，`Epstein & Schneider (2008)` 与 `Illeditsch (2011)` 只保留为背景锚点，不进入主清单。
| # | 作者 (年份) | 标题 | 刊物 | Citation | 核心贡献（一句话） | 用于哪条假说 |
|---|---|---|---|---:|---|---|
| 4.1 | Dimmock et al. (2016) | Ambiguity aversion and household portfolio choice puzzles: Empirical evidence | JFE | 381 | 从投资组合选择角度提供 ambiguity aversion 的强经验支持，说明模糊性会显著改变资产需求。 | H3 下游 + H4 异质性 |
| 4.2 | Baltussen, van Bekkum & van der Grient (2018) | Unknown Unknowns: Uncertainty About Risk and Stock Returns | JFQA | 155 | 把“对风险分布本身的不确定”与股票收益联系起来，直接支持本文的估值不确定性叙事。 | H1 理论 + H3 下游 |
| 4.3 | Brenner & Izhakian (2018) | Asset pricing and ambiguity: Empirical evidence | JFE | 206 | 把 ambiguity measure 带入资产定价实证，是“模糊性可被计量并进入风险溢价”的关键桥梁。 | H1 理论 + H3 下游 |
| 4.4 | Gallant, Jahan-Parvar & Liu (2018) | Does Smooth Ambiguity Matter for Asset Pricing? | RFS | 20 | 说明 smooth ambiguity 偏好能够进入标准资产定价框架，适合在理论部分提升金融学说服力。 | H1 2.5 理论补强 |
| 4.5 | Izhakian (2020) | A theoretical foundation of ambiguity measurement | JET | 108 | 给出 ambiguity measure 的理论基础，为后续把“估值不确定性”写成可测、可比较对象提供方法背书。 | H3 下游 + 方法论补强 |

### 主题 5：AI / 机器学习文本与披露（方法论支撑）
筛选备注：按 2015+ 约束，`Loughran & McDonald (2011)` 不纳入主清单；同作者的 2020 综述保留。spec 中 `Li et al. (2021)` 与 `Chen et al. (2020)` 当前未稳定核到精确题名。
| # | 作者 (年份) | 标题 | 刊物 | Citation | 核心贡献（一句话） | 用于哪条假说 |
|---|---|---|---|---:|---|---|
| 5.1 | Hoberg & Phillips (2016) | Text-Based Network Industries and Endogenous Product Differentiation | JPE | 2027 | 展示文本相似度如何刻画企业竞争位置，是把文本度量带入经济学主流识别的代表作。 | H2a 价值链 + 方法论 |
| 5.2 | Loughran & McDonald (2020) | Textual Analysis in Finance | Annual Review of Financial Economics | 244 | 总结金融文本分析主流方法与陷阱，适合给关键词法与语义法并行测度做方法学交代。 | H2b 质量 + 方法论 |
| 5.3 | Frankel, Jennings & Lee (2021) | Disclosure Sentiment: Machine Learning vs. Dictionary Methods | Management Science | 41 | 直接比较机器学习与词典法在披露文本上的表现，是“关键词法 + LLM 双测度”最贴切的近邻。 | H2b 质量 + 测度说明 |
| 5.4 | Zhao, Xu & Ji (2023) | Predicting financial distress of Chinese listed companies using machine learning: To what extent does textual disclosure matter? | IRFA | 38 | 在中国上市公司情境下证明文本披露对机器学习预测有独立信息增量，适合连接中文数据环境。 | H2b 质量 + 中国情境 |
| 5.5 | Eulerich et al. (2024) | Is it all hype? ChatGPT’s performance and disruptive potential in the accounting and auditing industries | Review of Accounting Studies | 78 | 把生成式 AI 正式带入会计与审计研究议程，适合说明本文使用 LLM 评分并非方法越界。 | H2b 质量 + 方法论边界 |

### 主题 6：EM 本刊互引（Economic Modelling，2018-2024）
筛选备注：已按 Economic Modelling 2018-2024 检索池筛出 5 篇最可直接嵌入 v2-1 稿件的文献，满足“EM 本刊互引 ≥ 3 篇”的硬约束。
| # | 作者 (年份) | 标题 | 刊物 | Citation | 核心贡献（一句话） | 用于哪条假说 |
|---|---|---|---|---:|---|---|
| 6.1 | Du et al. (2024) | Public data access and stock price synchronicity: Evidence from China | Economic Modelling | 50 | 直接把公共数据开放与股价同步性连到一起，是最贴近本文因变量的 EM 同刊文献。 | 引言互引 + 实证对照 |
| 6.2 | Kong, Shi & Zhang (2020) | Explain or conceal? Causal language intensity in annual report and stock price crash risk | Economic Modelling | 72 | 从年报文本的因果语言强度切入信息含量与风险后果，是 H2b/H2c 的同刊近邻。 | H2b 质量 + H2c |
| 6.3 | Wen, Fang & Gao (2023) | How FinTech improves financial reporting quality? Evidence from earnings management | Economic Modelling | 56 | 把数字技术与财务报告质量连接起来，适合放在“数字技术改善信息环境”的共同背景中。 | H1 引言 + H2b 对照 |
| 6.4 | He, Feng & Hao (2023) | Corporate ESG rating and stock market liquidity: Evidence from China | Economic Modelling | 93 | 虽然主题是 ESG，但核心识别仍是信息环境改进如何进入市场流动性与价格效率。 | 引言互引 + 市场后果对照 |
| 6.5 | Zhang et al. (2023) | Digitalization, financial inclusion, and small and medium-sized enterprise financing: Evidence from China | Economic Modelling | 84 | 从数字化改善融资可得性切入，适合支撑 H4 中“融资约束越高，外部信息价值越大”的外推。 | H4 异质性 + 实证对照 |

## 中文情境文献（主题 7，单列）

注：中文条目不混入英文主清单；DOI 或 citation count 无法被当前接口稳定核验者统一标记 `[需人工核对]`。
| # | 作者 (年份) | 标题 | 刊物 | Citation | 核心贡献（一句话） | 用于哪条假说 |
|---|---|---|---|---:|---|---|
| 7.1 | 李世刚等（2025） | 企业数据资产信息披露与资本市场定价效率 | 中国工业经济 | [需人工核对] | 中文最直接竞争文献，从“数据资产披露”切入资本市场定价效率。 | 中国情境段 + 文献综述对比 |
| 7.2 | 朱康、唐勇（2025） | 数据要素利用与企业金融资产配置——基于机器学习和文本分析的证据 | 会计研究 | [需人工核对] | 提供数据要素利用测度、中文写作范式与文本识别路径。 | 中国情境段 + 研究设计 |
| 7.3 | 李姝、赵灿、谢雁翔（2025） | 数据资产与企业金融化：数据治理还是概念炒作？ | 外国经济与管理 | [需人工核对] | 从数据资产视角讨论资本配置与融资约束，为本文机制池提供中文近邻。 | 中国情境段 + H4 |
| 7.4 | 洪永淼、史九领（2024） | 数据要素与数据经济学 | 经济理论与经济管理 | [需人工核对] | 系统梳理中国语境下的数据要素经济学，为第二章背景段提供中文权威表述。 | 中国情境段 + 背景制度 |
| 7.5 | 黄世忠、叶丰滢、陈朝琳（2023） | 数据资产的确认、计量和报告——基于商业模式视角 | 财会月刊 | [需人工核对] | 直接讨论数据资产确认与计量难题，适合支撑“账面缺位”与估值偏差。 | 中国情境段 + 2.1 理论 |
| 7.6 | 郭家堂（2025） | 公共数据开放与中国绿色全要素生产率 | 经济研究 | [需人工核对] | 从公共数据开放角度证明数据要素配置能够改变实体经济绩效，是政策环境写作的高质量来源。 | 中国情境段 + 政策背景 |
| 7.7 | 陆瑶等（2025） | 中国企业数字技术风险暴露对企业价值的影响 | 经济研究 | [需人工核对] | 把数字技术风险显式引入企业价值，有助于说明中国市场对数字相关信息仍存在高解释摩擦。 | 中国情境段 + 风险补充 |
| 7.8 | 姚加权等（2024） | 人工智能如何提升企业生产效率 | 管理世界 | [需人工核对] | 从 AI 与生产率切入数字技术的真实经营后果，适合承接“利用”而非“概念喊话”的论证。 | 中国情境段 + 利用逻辑 |
| 7.9 | 陈素云、李怡舒（2024） | 企业数字化转型与分析师预测准确性 | 会计之友 | [需人工核对] | 直接落在分析师信息环境上，适合为 H3 中分析师分歧渠道补中文近邻。 | 中国情境段 + H3 |
| 7.10 | 刘翰林、黄佳玲（2024） | 企业数字化转型对融资约束的影响研究——基于“信息论”与“资源论”的视角 | 杭州电子科技大学学报（社会科学版） | [需人工核对] | 把数字化与融资约束联系起来，适合补强 H4 的中国制度语境。 | 中国情境段 + H4 |

## 按假说分配

- H1（主效应）必用文献：1.1 / 1.4 / 2.1 / 3.1 / 4.3 / 6.1。
- H2a（价值链）必用文献：2.2 / 2.3 / 5.1。
- H2b（质量 direct + joint）必用文献：3.1 / 3.2 / 3.3 / 5.2 / 5.3 / 5.5。
- H2c（WashGap）必用文献：3.2 / 3.3 / 6.2。
- H3（下游 channels）必用文献：3.4 / 3.5 / 4.2 / 4.3 / 7.9。
- H4（异质性）必用文献：1.5 / 2.4 / 2.5 / 6.5 / 7.3 / 7.10。

## Zotero 操作

- 英文核心 30 篇 + 中文情境 10 篇，共 40 篇。
- 目标 collection：`v2-1_theory_modernization`。
- 导入文件：`v2/v2-1_theory_modernization.bib`；导出文件：`v2/literature_modern.bib`。
- 由于 Zotero MCP 写入在本机处于 `local-only mode`，本轮通过本地 Zotero 导入 `.bib` 建 collection，再用 Better BibTeX 从 Zotero 回导出 BibTeX。
- 已用 Better BibTeX 回读验证 40 个 citation key，`v2-1_theory_modernization` 中缺失数为 `0`。
- Zotero 回导出的最终 BibTeX 已落盘到 `v2/literature_modern.bib`；若后续补全中文条目元数据，可直接在该 collection 内重导出覆盖。

## 清单外发现的惊喜文献

- Liu, Ma & Veldkamp (2025)，Data Sales and Data Dilution，JFE：把“数据可卖出”与“数据一旦扩散就被稀释”放进同一框架，极适合后续扩展数据披露的双刃剑叙事。
- Gao, Lyu & Zhang (2024)，Disclosure Regulation and Price Informativeness: Evidence from Industry-Information Disclosure Guidelines in China，EAR：虽然不是数据资产题材，但它把中国行业披露指引与价格信息含量严密接上，适合写制度段。
- Wei et al. (2025)，Does data asset disclosure contribute to the market efficiency? Evidence from China，RIBAF：把分析师预测分歧、噪声交易与市场效率放进同一篇里，和你的 H3 写法贴合度很高。
- Lopez-Lira & Tang (2023)，Can ChatGPT Forecast Stock Price Movements? Return Predictability and Large Language Models，SSRN：不是披露论文，但它让“LLM 能否提炼金融文本信息”在资本市场语境下变得可引用。
- Xie et al. (2023)，PIXIU: A Large Language Model, Instruction Data and Evaluation Benchmark for Finance，arXiv：如果后面要单开方法附录或补模型选择理由，这篇是金融大模型基准文献。
