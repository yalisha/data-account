# Economic Modelling 特刊潜在选题备忘录

特刊：**Artificial Intelligence, Data Investment and the Digital Economy**  
核心自变量：**年报中的数据要素披露 / 数据资源披露 / 数据资产披露**  
核心定位：不要把文章写成一般的“数字化披露影响企业绩效”，而要把 X 解释为企业在数据经济中对数据资产、数据能力和数据治理的外部可见信号。

---

## 一、三个优先选题总览

| 优先级 | 题目方向 | 推荐程度 | 核心 Y | 主要贡献点 | 主要风险 |
|---|---:|---:|---|---|---|
| 1 | 年报数据要素披露与创新原创性 | 高 | 专利原创性、远距离知识重组、跨技术域组合 | 从“创新数量”推进到“创新质量与知识重组” | 专利滞后、R&D 内生性 |
| 2 | 年报数据要素披露与盈余公告后漂移 | 高 | PEAD、公告后 CAR、盈余信息吸收速度 | 从宽泛定价效率收窄到“盈余新闻进入价格的动态过程” | 容易被误写成一般定价效率，需要明确区分 |
| 3 | 年报数据要素披露、加成率与劳动收入份额 | 中高但难 | Markup、劳动收入份额、毛利率、市场势力 | 最贴合特刊中的 mark-up dynamics、market concentration、inequality | Markup 测度难，识别压力大 |

---

# 题目一：年报数据要素披露与创新原创性

## 拟题目

**中文题目：**  
年报数据要素披露能否提升企业创新原创性？来自远距离知识重组的证据

**英文题目：**  
Can Data Element Disclosure Enhance Corporate Innovation Originality? Evidence from Distant Knowledge Recombination

## 核心问题

已有研究较容易做到“数据资产披露提高创新投入或创新产出”，但这类题目容易停留在创新数量层面。真正更有增量的问题是：

**企业披露更多、更具体、更可信的数据要素信息之后，是否不仅产生更多专利，而且产生更原创、更跨域、更难复制的创新？**

## 理论逻辑

数据要素披露可以被视为企业数据能力和数据治理成熟度的外部信号。高质量数据披露意味着企业可能更有能力收集、清洗、整合和复用异质信息资源。数据能力越强，企业越可能进行跨部门、跨技术域、跨场景的知识重组，从而提高创新原创性。

机制可以写成：

**数据要素披露 → 数据治理能力 / 数据资源可见性 → 知识搜索边界扩大 → 远距离知识重组 → 创新原创性提高**

## 可能的因变量

主 Y 可以是：

- 专利原创性；
- 专利新颖性；
- 跨 IPC 技术组合；
- 技术距离；
- 引用分散度；
- 授权发明专利占比；
- 高质量专利数量。

不要只用“专利申请数”或“发明专利数”作为主 Y。那些指标太常规，容易被认为只是又一篇“数据披露促进创新”的文章。

## 基准模型

```math
Originality_{i,t+1}
=
\beta X_{i,t}
+
\gamma Controls_{i,t}
+
\mu_i
+
\tau_t
+
\varepsilon_{i,t}
```

其中：

- \(X_{i,t}\)：年报数据要素披露质量；
- \(Originality_{i,t+1}\)：下一期创新原创性；
- \(\mu_i\)：企业固定效应；
- \(\tau_t\)：年份固定效应。

## 识别增强

可以利用 2024 年数据资源会计处理规则作为制度节点，构造：

```math
Originality_{i,t}
=
\delta (Post2024_t \times HighDataPotential_i)
+
\gamma Controls_{i,t}
+
\mu_i
+
\tau_t
+
\varepsilon_{i,t}
```

更强一点可以做 DDD：

```math
Originality_{i,t}
=
\theta (Post2024_t \times HighDataPotential_i \times HighDisclosureImprovement_i)
+
\gamma Controls_{i,t}
+
\mu_i
+
\tau_t
+
\varepsilon_{i,t}
```

## 适合特刊的写法

这题应当主动贴合以下关键词：

- data as an asset；
- data investment；
- intangible capital；
- R&D and returns to innovation；
- AI, software, and data。

核心叙事不是“中国上市公司披露文本影响创新”，而是：

**数据作为新型无形资本，是否通过扩大企业知识搜索与重组能力，提高创新产出的原创性？**

## 风险

第一，专利结果有滞后性。  
第二，高数据披露企业本身可能就是高科技企业。  
第三，简单词频可能混入“数字化叙事”或“数据概念包装”。

## 应对

- 使用企业固定效应和行业×年份固定效应；
- 控制 R&D、员工结构、企业规模、现金流、融资约束；
- 使用滞后一期或两期的创新质量指标；
- 把 X 拆成“披露强度、会计确认、具体性、一致性”几个维度；
- 区分真实数据资源披露与泛泛数字化口号。

---

# 题目二：年报数据要素披露与盈余公告后漂移

## 拟题目

**中文题目：**  
年报数据要素披露会降低盈余公告后漂移吗？基于盈余信息吸收速度的证据

**英文题目：**  
Does Data Element Disclosure Reduce Post-Earnings Announcement Drift? Evidence from the Speed of Earnings News Incorporation

## 这个题目必须和“定价效率”区分

这题最大的写作风险是被审稿人理解成：

**数据要素披露 → 定价效率提高**

如果这样写，就会和已有的资本市场信息效率、股价同步性、错定价、价格延迟等研究发生重叠。

所以这题不能主打“定价效率”这个宽概念，而要主打：

**盈余新闻进入价格的速度**  
或者  
**投资者对盈余信息的动态学习过程**

也就是说，PEAD 不是泛泛的“价格是否有效”，而是一个更具体的问题：

**当企业公布盈余消息之后，市场是否需要很长时间继续消化这些信息？**

如果高质量数据要素披露降低 PEAD，说明它改善的不是抽象意义上的“定价效率”，而是：

**降低了投资者理解和处理盈余信息的摩擦，使盈余新闻更快进入股价。**

## PEAD 与定价效率的边界

| 概念 | 问题 | 指标层级 | 和本文的关系 |
|---|---|---|---|
| 定价效率 | 股价是否充分反映信息 | 大概念 | 不能作为本文主标题，否则太宽 |
| 股价同步性 | 股价中公司特质信息含量是否更高 | 宽口径指标 | 可作为补充结果 |
| Price Delay | 股价是否滞后反应市场或行业信息 | 中口径指标 | 可作为稳健性 |
| Mispricing | 价格是否偏离基本面价值 | 宽口径指标 | 不宜作为主 Y |
| PEAD | 盈余公告后股价是否继续沿盈余意外方向漂移 | 窄口径、事件型指标 | 适合作为主 Y |
| Earnings news incorporation speed | 盈余新闻被市场吸收的速度 | 机制型概念 | 应作为本文核心定位 |

因此，本文题目和摘要中应该尽量少用：

> capital market pricing efficiency

而应更多使用：

> earnings information processing friction  
> speed of earnings news incorporation  
> investor learning after earnings announcements  
> delayed market response to earnings news

## 核心问题

年报中的数据要素披露是否帮助投资者更好理解企业的数据能力、经营模式、信息系统和盈利生成过程，从而降低后续盈余公告发布后的信息处理摩擦？

换句话说：

**同样一份盈余公告，在高数据要素披露企业中，市场是否更快理解其含义？**

## 理论逻辑

年报数据要素披露提供的是一种“背景信息”。它未必直接告诉投资者下一期利润是多少，但它帮助投资者理解企业如何使用数据、如何组织生产、如何管理客户、供应链、库存和风险。

因此，当后续盈余公告出现时，投资者面对的不是孤立的利润数字，而是可以放入企业数据能力和经营系统中的信息。这样会降低投资者的解读成本，减少公告后的慢反应。

机制链条可以写成：

**数据要素披露 → 企业数据能力与经营模式更可理解 → 盈余消息解释成本下降 → 公告后价格漂移减弱**

## 主回归设计

最核心的模型应当抓住：

**SUE × 数据要素披露**

```math
CAR_{i,[2,60],t}
=
\beta_1 SUE_{i,t}
+
\beta_2 X_{i,t-1}
+
\beta_3 (SUE_{i,t} \times X_{i,t-1})
+
\gamma Controls_{i,t}
+
\mu_i
+
\tau_t
+
\varepsilon_{i,t}
```

其中：

- \(CAR_{i,[2,60],t}\)：盈余公告后第 2 天到第 60 天的累计异常收益；
- \(SUE_{i,t}\)：标准化未预期盈余；
- \(X_{i,t-1}\)：上一期年报数据要素披露；
- \(SUE \times X\)：核心交互项。

## 系数解释

如果存在 PEAD，则 \(SUE\) 应该能够预测公告后的异常收益。  
如果数据要素披露降低 PEAD，则高 X 企业中，SUE 对公告后 CAR 的预测力应当减弱。

因此，关键不是简单看 \(X\) 对 \(CAR\) 的影响，而是看：

```math
\beta_3 < 0
```

解释为：

**数据要素披露越充分，盈余意外对公告后漂移的预测作用越弱，说明盈余信息更快被市场吸收。**

## 可替代 Y

主 Y：

- \(CAR[2,60]\)；
- \(CAR[2,30]\)；
- \(CAR[2,90]\)；
- PEAD 强度；
- delayed response to earnings news。

辅助 Y：

- Price Delay；
- 公告日 CAR；
- 公告日前信息泄露；
- 分析师 forecast dispersion；
- forecast error；
- analyst revision speed。

## 机制检验

可以做三类机制。

第一，分析师信息处理机制：

```math
ForecastDisp_{i,t+1}
=
\beta X_{i,t}
+
Controls
+
FE
+
\varepsilon
```

预期：数据要素披露降低分析师预测分歧。

第二，投资者注意力机制：

可以用百度指数、东方财富股吧、研报数量、媒体报道数量等指标，检验结果是否不是单纯由注意力驱动。

第三，披露质量机制：

把 X 拆成：

- 数据要素披露强度；
- 数据资源会计确认；
- 具体应用场景；
- 可量化披露；
- 文本—会计一致性。

如果只有“具体性”和“一致性”显著，而泛泛词频不显著，文章会更有说服力。

## 与定价效率文献的区分写法

不能这样写：

> 本文研究数据要素披露对资本市场定价效率的影响。

应该这样写：

> 本文研究数据要素披露是否影响投资者对后续盈余新闻的动态吸收过程。不同于使用股价同步性、错定价或价格延迟等宽口径定价效率指标的研究，本文聚焦盈余公告后的价格漂移，考察数据要素披露是否降低盈余信息处理摩擦，使盈余新闻更快进入股价。

中文摘要里的关键句可以写成：

> 本文并不将盈余公告后漂移简单视为一般定价效率指标，而是将其作为投资者处理企业盈余信息速度的动态证据。若数据要素披露能够降低 PEAD，则说明其作用机制并非仅仅改变价格水平或市场估值，而是改善了市场对后续盈余新闻的解释与吸收过程。

英文摘要里的关键句可以写成：

> Rather than treating post-earnings announcement drift as a generic proxy for pricing efficiency, we interpret it as evidence on the speed with which investors incorporate earnings news into prices. This distinction allows us to examine whether data element disclosure reduces information-processing frictions surrounding subsequent earnings announcements.

## 风险

第一，PEAD 容易被看成定价效率的一个子指标。  
第二，数据要素披露可能只是吸引市场注意力，而不是真正提高信息处理能力。  
第三，年报披露和盈余公告之间可能存在时间错配。  
第四，数字化热门叙事可能影响短期收益。

## 应对

- 标题中避免使用“定价效率”作为主概念；
- 使用“盈余信息吸收速度”作为核心机制；
- 控制媒体关注、投资者情绪、年报语调、数字化语调；
- 做 placebo：非盈余公告窗口、非数据类热词、随机公告日；
- 做机制：分析师分歧下降、预测修正更快；
- 做异质性：信息不透明企业、低分析师关注企业、非高科技企业中效果更强。

## 适合特刊的写法

这题应当贴合：

- AI, data, and financial markets；
- data investment and information frictions；
- data as an informational asset；
- empirical studies with causal inference。

最好的定位是：

**企业数据要素披露是否改变资本市场处理企业基本面新闻的方式？**

而不是：

**数据要素披露是否提高资本市场定价效率？**

---

# 题目三：年报数据要素披露、加成率与劳动收入份额

## 拟题目

**中文题目：**  
年报数据要素披露如何重塑企业市场势力？来自加成率与劳动收入份额的证据

**英文题目：**  
How Does Data Element Disclosure Reshape Firm Market Power? Evidence from Markups and Labor Share

## 核心问题

数据是新型无形资产。企业拥有、治理和使用数据的能力，可能改变其成本结构、客户锁定能力、产品定价能力和劳动需求结构。

因此，这题要问：

**数据要素披露是否反映了企业数据能力的增强，并进一步改变企业的市场势力和要素收入分配？**

## 理论逻辑

存在两种相反机制。

第一种是效率机制：

**数据能力提高 → 预测更准确、库存更低、匹配更好、边际成本下降 → 企业经营效率提高**

第二种是市场势力机制：

**数据能力提高 → 客户画像更精细、转换成本更高、平台/网络效应更强 → 企业加成率提高**

第三种是分配机制：

**数据能力提高 → 自动化与算法管理增强 → 劳动议价能力下降或劳动份额变化**

所以这题最好不要一开始就假设“数据披露一定提高 markup”。更高级的写法是：

**数据要素披露可能同时带来效率提升和市场势力扩张，劳动收入份额可以帮助区分两种机制。**

## 可能的因变量

主 Y：

- markup；
- Lerner index；
- 毛利率；
- 营业利润率；
- 劳动收入份额；
- 行业相对市场份额。

辅助 Y：

- TFP；
- 单位成本；
- 销售费用率；
- 管理费用率；
- 市场集中度；
- 客户集中度。

## 基准模型

```math
Markup_{i,t}
=
\beta X_{i,t-1}
+
\gamma Controls_{i,t}
+
\mu_i
+
\tau_t
+
\varepsilon_{i,t}
```

劳动收入份额模型：

```math
LaborShare_{i,t}
=
\phi X_{i,t-1}
+
\gamma Controls_{i,t}
+
\mu_i
+
\tau_t
+
\varepsilon_{i,t}
```

异质性模型：

```math
Markup_{i,t}
=
\beta_1 X_{i,t-1}
+
\beta_2 (X_{i,t-1} \times LowCompetition_{j,t})
+
\gamma Controls
+
FE
+
\varepsilon_{i,t}
```

如果 \(\beta_2 > 0\)，说明数据能力在低竞争行业中更可能转化为市场势力，而不只是效率提升。

## 识别设计

可以使用：

- 2024 年数据资源会计规则作为制度节点；
- 事前数据潜能高低作为处理强度；
- 行业竞争程度作为异质性；
- 地方公共数据开放水平作为外部数据环境；
- 数字基础设施或数据交易平台作为外部条件。

一个 DDD 思路：

```math
Y_{i,t}
=
\theta
(Post2024_t \times HighDataPotential_i \times LowCompetition_{j,t})
+
\gamma Controls
+
\mu_i
+
\tau_t
+
\varepsilon_{i,t}
```

## 适合特刊的写法

这题非常贴合特刊中的：

- mark-up dynamics；
- market concentration；
- income inequality；
- data investment；
- data as intangible capital；
- digital transformation and market structure。

它的优势是学术辨识度强，不像普通企业金融题，而更像数字经济结构性问题。

## 风险

第一，markup 测度复杂，容易被审稿人攻击。  
第二，劳动收入份额数据口径可能不稳定。  
第三，X 是披露变量，不一定代表真实数据能力。  
第四，结果解释可能在“效率提升”和“市场势力扩张”之间摇摆。

## 应对

- 不依赖单一 markup estimator；
- 同时使用生产函数法、毛利率、营业利润率、Lerner proxy；
- 用劳动收入份额辅助区分效率机制和分配机制；
- 用行业竞争程度、市场集中度、公共数据开放程度做异质性；
- 把 X 做成“高质量披露”而不是简单词频；
- 明确区分“数据能力带来的效率红利”和“数据壁垒带来的市场势力”。

---

# 最终推荐排序

## 如果目标是稳妥发表

首选：

**题目二：年报数据要素披露与盈余公告后漂移**

原因是数据可得性强，识别路径清楚，和你现有的资本市场信息效率研究基础最接近。但写作时必须避免把它写成宽泛定价效率题，而要把它写成“盈余新闻吸收速度”。

## 如果目标是做出更强增量

首选：

**题目一：年报数据要素披露与创新原创性**

原因是已有文献可能已经做了创新数量，但创新原创性、远距离知识重组、跨技术域组合更有新意，也更贴合 data as intangible capital。

## 如果目标是最贴 Economic Modelling 特刊

首选：

**题目三：年报数据要素披露、加成率与劳动收入份额**

原因是它直接对应特刊中的 mark-up dynamics、market concentration 和 inequality。但执行难度最高，需要较强的经济学建模和测度能力。

---

# 我对三个题目的最终判断

| 题目 | 创新性 | 可执行性 | 特刊适配 | 综合建议 |
|---|---:|---:|---:|---|
| 创新原创性 | 8.5 | 8.0 | 9.0 | 最均衡，可作为主推 |
| PEAD / 盈余信息吸收速度 | 8.0 | 9.0 | 8.5 | 最稳，但必须和定价效率区分 |
| Markup / 劳动收入份额 | 9.5 | 6.5 | 10.0 | 最像特刊，但难度最大 |

最终建议：

**主推题目一或题目二。**

如果主人想最快做出一篇可投 Economic Modelling 特刊的文章，我建议选：

**年报数据要素披露会降低盈余公告后漂移吗？基于盈余信息吸收速度的证据**

但这篇的标题、摘要和理论部分必须坚决避免泛化成“定价效率”。它的真正卖点是：

**数据要素披露改变了投资者处理后续盈余新闻的速度。**

这才是它和一般定价效率研究的边界。
