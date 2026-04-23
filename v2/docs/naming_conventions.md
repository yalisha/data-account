# 命名与叙事口径规范（中英对照）

**目标刊**：Economic Modelling（Elsevier, SSCI Q2）
**作用**：锁定核心概念与术语，防止中英文版本前后打架，便于翻译阶段一致性审查

---

## 零、三层概念链强制规范（最高优先级）

本文涉及三个相关但截然不同的概念层，全文必须严格区分：

| 层 | 中文 | 英文 | 属性 | 是否可观测 |
|---|---|---|---|---|
| 第一层 | 数据要素 | Data (as factor of production) | 资源 | 外部不可观测 |
| 第二层 | 数据要素利用 | Data utilization | 经营行为 | 外部不可观测 |
| 第三层 | **数据要素利用披露** | **Data-element utilization disclosure** | 信息行为 | 可观测（进入定价函数） |

### 使用规则

- **自变量名、假说表述、实证结果解读** — 必须使用第三层"数据要素利用披露"。例："数据要素利用披露提升股价定价效率"✓；"数据要素提升股价定价效率"❌；"数据要素利用提升股价定价效率"❌（后两者把不可观测变量当自变量）。
- **背景讨论 / 政策引用** — 可以单用"数据要素"（指资源本身）或"数据要素利用"（指使用行为），但必须在上下文中明确是哪一层，不得与第三层混用。例：引述国务院文件时用"数据要素的市场化配置"✓。
- **理论推导段落** — 说明"数据要素 → 利用 → 披露"的映射关系时三者可并列，但必须明确每层的角色。本文 2.2 节专门做此界定。
- **英文版本** — 标题、摘要、关键词、假说、主文献综述必须是 "data-element utilization disclosure"；缩写为 "data disclosure" 或 "data utilization" 是红线错误。

### 违例识别（校对时重点检查）

- ❌ "数据要素降低股价延迟" — 混淆第一层与第三层
- ❌ "数据要素利用的定价效应" — 第二层冒充第三层
- ❌ "Data disclosure affects price delay" — 英文版必须是 "data-element utilization disclosure"
- ❌ "企业数据披露越多，股价越有效" — "数据披露"指代不明
- ✓ "企业数据要素利用披露程度越高，股价延迟越低"
- ✓ "数据要素作为新型生产要素（此处指第一层概念），其价值实现依赖企业披露"

### 识别策略的理论对应

本文自变量（DU<sub>kw</sub>、DU<sub>llm</sub> 等）测量第三层。之所以定价效应存在，是因为 Merton (1987) / Verrecchia (2001) 的价格调整通过信息触达投资者发生。识别上，H2b 与 H2c 的披露质量指标专门用于分离"第二层能力"与"第三层信息行为"——两个数据能力接近的企业若披露质量不同且对应不同延迟，该差异只能归因于信息维度。

---

## 一、核心概念中英对照

| 中文 | 英文 | 备注 |
|---|---|---|
| 数据要素利用披露 | Data-element utilization disclosure | 不用 "data disclosure"（太泛） |
| 股价定价效率 | Stock price efficiency | 不用 "capital pricing efficiency" |
| 生产要素价值释放 | Production factor value release | 本文主理论锚点（Jones-Tonetti 2020 + Crouzet 2022 + Brynjolfsson 2011） |
| 估值不确定性 | Valuation uncertainty | 下位机制理论（Merton 1987 + Akerlof 1970） |
| 经营不确定性 | Operational uncertainty | 经营侧机制统一表述（基本面 + 供应链两维） |
| 经营侧机制 | Operational-side channels | 本文机制定位（vs 既有文献 information-side） |
| 基本面稳定 | Fundamental stability | CashFlowVol 对应（Epstein-Schneider 2008） |
| 供应链优化 | Supply chain optimization | SCConc 对应（Cen 2016；Ellis 2012） |
| 披露密度 | Disclosure intensity | 不用 "disclosure quantity"（草率） |
| 披露强度 | Disclosure intensity | 同上 |
| 语义质量 | Semantic quality | 不用 "semantic depth"（强 claim） |
| 信息含量 | Information content | 会计文献惯用语 |
| 广度-深度错配 | Breadth-depth mismatch | WashGap；避免 "greenwashing"（中文语境不同） |
| 沿价值链拆分披露 | Value chain decomposition | 五环节（资源/开发/应用/价值化/治理） |
| 下游传导（中文）/ Channels（英文） | Channels Consistent with ... | 英文严禁用 "mechanism" |
| 异质性 | Heterogeneity / Contextual variation | 后者更 EM 风格 |
| 增量解释力 | Incremental explanatory power | 代替 subsume/dominate |
| 双测度 | Dual measurement | DU_kw + DU_llm |

---

## 二、假说对应的中英叙事

### H1 主效应

| 中文 | 英文 |
|---|---|
| 企业数据要素利用披露降低股价定价效率的延迟程度 | Firms' data-element utilization disclosure reduces stock price delay, thereby improving price efficiency |

### H2a 价值链内容异质性

| 中文 | 英文 |
|---|---|
| 数据价值链应用层、开发层、资源层的披露内容独立降低股价延迟 | Disclosure content from the application, development, and resource stages of the data value chain independently reduces stock price delay |

### H2b 披露质量的增量解释力

| 中文 | 英文 |
|---|---|
| 在控制披露密度后，披露的闭环性、核心业务嵌入、链条完整性、MD&A 嵌入、语义×密度复合指标**仍具有增量解释力** | After controlling for disclosure intensity, the closed-loop nature, core-business embeddedness, chain completeness, MD&A embeddedness, and semantic-intensity composite of disclosure **retain incremental explanatory power** over price delay |

**绝对禁止**：
- ❌ 语义深度 subsume 关键词数量 / Semantic depth subsumes keyword intensity
- ❌ 质量 dominate 数量 / Quality dominates quantity
- ❌ 同等密度下语义评分越高效应越强 / Higher semantic scores amplify the disclosure effect at the same intensity level（听起来像 moderation，但我们没跑交互）
- ❌ 只有一个最重要 / Only one matters

**安全表述**：
- ✓ 在控制披露密度后，语义质量仍具有增量解释力
- ✓ Semantic quality retains incremental explanatory power beyond disclosure intensity
- ✓ 多测度从不同角度捕捉披露信息含量（convergent validity）
- ✓ Multiple quality proxies provide convergent validity for the information content of disclosure

### H2c 广度-深度错配

| 中文 | 英文 |
|---|---|
| 披露的广度-深度错配（WashGap）增加股价延迟 | Disclosure breadth-depth mismatch (WashGap) increases stock price delay |

### H3 下游 channels

| 中文标题 | 中文正文 | 英文 |
|---|---|---|
| 下游传导机制（用江艇范式） | 与估值不确定性下降**相一致**的下游 channels | Channels consistent with valuation uncertainty reduction |

**绝对禁止**：
- ❌ 披露通过 X 机制降低股价延迟（mediation claim）/ Disclosure reduces delay through mechanism X
- ❌ 严格识别机制链条 / Identified mediation pathway

**安全表述**：
- ✓ 与估值不确定性下降相一致的下游表现
- ✓ Consistent with the valuation uncertainty reduction hypothesis
- ✓ 三条 channels 对应估值不确定性的三个维度
- ✓ Three channels corresponding to the three dimensions of valuation uncertainty

### H4 情境异质性

| 统一框架（中文） | 统一框架（英文） |
|---|---|
| 当企业原始估值不确定性更高、市场信息处理摩擦更大、或外部监督更强时，数据要素披露的边际定价作用更强 | The marginal pricing effect of data-element disclosure is stronger when firms face greater ex-ante valuation uncertainty, higher information-processing frictions, or more intensive external monitoring |

**四个子维度映射**：
- Post2020 → 信息摩擦（信号稀缺性随政策减弱）
- DigEconCore / HighTech → 基准不确定性（业务依赖度）
- SA → 外部信息价值（融资约束）
- InstHold → 外部监督（机构持股）

---

## 三、变量名规范

### Stata / Python 变量名（代码层）
保持现有：`DU_kw`、`DU_llm`、`DUclosedloop`、`DUcore` 等，不改。

### 正文表述（写作层）

| 代码名 | 中文正文 | 英文正文 |
|---|---|---|
| DU_kw | 关键词披露密度 | Keyword-based disclosure intensity |
| DU_llm | LLM 语义加权披露 | LLM-weighted disclosure |
| DU_llm_lenstd | 语义×密度复合指标 | Semantic-intensity composite |
| DUclosedloop | 闭环披露 | Closed-loop disclosure |
| DUcore | 核心业务嵌入披露 | Core-business embeddedness |
| DUchain_count | 数据链条完整性 | Chain completeness |
| DUkw_mda | MD&A 嵌入披露 | MD&A-embedded disclosure |
| WashGap | 广度-深度错配 | Breadth-depth mismatch |
| DU_stock/dev/app/value/gov | 数据价值链各环节（资源/开发/应用/价值化/治理） | Five stages of the data value chain (resource / development / application / value realization / governance) |
| PriceDelay | 股价延迟（Hou-Moskowitz 2005） | Price delay (Hou-Moskowitz 2005) |
| ForecastDisp | 分析师预测分歧 | Analyst forecast dispersion |
| CashFlowVol | 现金流波动率 | Cash flow volatility |
| SCConc | 供应链集中度 | Supply chain concentration |

---

## 四、数字表达规范

- 系数保留小数 4 位：-0.0046
- t 值保留小数 2 位：-5.94
- 显著性标记：* p<0.1, ** p<0.05, *** p<0.01
- 英文显著性表达：significant at the 1% (/5%/10%) level
- N 使用千分位：43,735 / 37,294

---

## 五、禁用词汇总

### 一般禁用词

**中文**：subsume、dominate、主导、替代、击败、优于（比较两个变量时）、决定（强因果词）、严格识别、机制链条、mediation、中介检验（除非真做了 Baron-Kenny）

**英文**：subsume、dominate、replace、beat、superior to（比较两个变量时）、determine、strictly identify、mediation pathway、mediation proof

### 与竞品文献重叠的机制侧用语禁用（方案 B 新增）

**直接竞品**：Sun & Du (2024, IRFA)；李世刚等 (2025, 中国工业经济)。两篇机制全落在信息侧，其核心词汇为"信息供给 / 信息扩散 / 信息融入 / 信息不对称缓解 / 机构投资者持股 / 融资约束缓解 / 特质信息含量"。

**本文定位为经营侧机制**。下列信息侧核心词**在第六章机制正文禁用**（除非在"对照段"明确归因给既有文献）：

❌ 信息供给效应 / information supply effect
❌ 信息扩散效应 / information diffusion effect
❌ 信息融入 / information incorporation
❌ 特质信息含量 ↑ / firm-specific information content ↑
❌ 披露通过降低信息不对称提升定价效率（此句是李世刚/Sun-Du 核心叙事）
❌ 披露通过吸引机构投资者 / 缓解融资约束提升定价效率

**安全替代**（经营侧语汇）：
✓ 基本面稳定 / fundamental stability
✓ 经营不确定性下降 / reduction in operational uncertainty
✓ 供应链优化 / 供应链韧性提升 / supply chain optimization (resilience)
✓ 数据要素真实嵌入生产函数 / data factors genuinely embedded in production function
✓ 数据要素的市场识别信号 / market-recognition signal of data-factor embeddedness
✓ 与估值不确定性下降相一致 / consistent with reduction in valuation uncertainty

### SYNCH 处理红线（方案 B 新增）

**绝对禁用**：
❌ "SYNCH ↓ = 定价效率 ↑"（MYY 2000 传统读法，Sun-Du/李世刚立场）
❌ "SYNCH 是定价效率对偶 DV / 第二个定价效率指标"
❌ 把 SYNCH 放进基准主回归表
❌ "披露提高股价同步性 / 降低股价同步性" 的简单方向表述（我们结果与 Sun-Du/李世刚方向相反，简单表述会引起方向误读）

**安全表述**：
✓ "股价同步性作为信息含量结构的补充证据"（第五章 5.6 定位）
✓ "按 Chen et al. (2021, A&F) public vs private information 框架，SYNCH 上升反映披露强化 public info 进价渠道，不等于定价效率恶化"
✓ "Delay 测信息吸收速度，SYNCH 测公开信息成分占比，两者从不同维度刻画'应当定价的信息更快进价'"
✓ 诚实脚注引用 Li-Liu-Pursiainen (2022, MiFID II) + Hu-Zhao-Zhang (2019) 的 caveat

### 贡献叙事红线（方案 B 新增）

**直接竞品已占位**：Sun-Du 2024 / 李世刚等 2025 已做"数据（资产）披露 → 定价效率 / 股价同步性"主链。

**禁用的贡献表述**（在引言、摘要、结论反复使用的三句式）：
❌ "本文首次发现数据披露降低股价延迟 / 提升定价效率"
❌ "本文首次揭示数据披露的定价效应"
❌ "本文开创数据要素披露的实证研究"
❌ "本文首次采用文本分析 / 机器学习测度数据披露"（朱康 2025 已做）

**安全的贡献三句式**：
✓ 贡献一（方法论）："提出关键词 + LLM 语义评分双测度框架，结合价值链分解、质量多指标与 WashGap 反向测度，为文本式披露研究提供可复用模板"
✓ 贡献二（理论定位）："从'数据要素作为生产要素的价值释放'理论切入，识别经营侧机制（基本面稳定 + 供应链优化），区别于既有文献的信息侧路径"
✓ 贡献三（独家发现）："WashGap 反向效应揭示概念堆砌型披露加剧定价延迟，这一发现在单测度文献中不可检验"

---

## 六、引用规范

### 中文论文（投中文期刊版本）
- 国际：作者 (年份)，页码；如 Merton (1987)
- 中文：作者 等 (年份)；如 张新民等 (2023)
- 同一括号多引用：Merton (1987); Amihud & Mendelson (1986)

### 英文论文（投 EM 版本）
- 在文中引用：Merton (1987) argues that...
- 括号内引用：(Merton, 1987)
- 多作者：3+ 用 et al.：Chernozhukov et al. (2018)
- 页码：(Merton, 1987, p. 486)

### 引用优先级（EM 版本）
1. 顶刊：JF / JFE / RFS / JAR / TAR / JAE / AER / JPE / QJE
2. EM 本刊历史文献（至少引 2-3 篇）
3. 中文文献：只在中国情境段落引用，正文主体避免

---

## 七、校对清单（写完一章后）

- [ ] 三层概念链无混用："数据要素 / 数据要素利用 / 数据要素利用披露"三者每次出现对应正确层级（第零节）
- [ ] 所有"subsume/dominate/主导/替代"已替换
- [ ] H2 机制叙事未用"信息供给/扩散/融入"这组竞品词（第五节）
- [ ] H2 机制叙事未 claim mediation（一律用"与 X 相一致"）
- [ ] 贡献叙事未用"首次发现披露降定价效率"（第五节贡献三句式）
- [ ] SYNCH 未被表述为"对偶 DV / 第二定价效率指标"；未出现在主回归表
- [ ] "资本定价效率"全部改"股价定价效率"
- [ ] "披露"一词前加"数据要素利用"前缀
- [ ] 无 AI 痕迹词："五维分解 / 多维质量 / 多维设计 / 结构化分析 / consistent-with 证据"（变量层面技术描述除外）
- [ ] 数字显著性 */**/*** 规范
- [ ] 直接竞品 Sun-Du 2024 / 李世刚等 2025 / 朱康唐勇 2025 在引言必引
- [ ] 英文无 Chinese-style："我们发现" / "本文认为"

