# 个人破产制度研究设计 v2：外部评审吸收版

更新日期：2026-05-04

## 一、吸收后的总判断

这个题继续，但只能继续成“自然人责任边界冲击”题，不能继续成“个人破产制度影响企业某个 Y”题。

最终研究问题应收窄为：

> 个人破产制度改变自然人责任边界后，高自然人责任暴露企业是否减少坏消息隐藏；同时，对高度依赖自然人追偿价值的企业，制度是否诱发信贷期限收缩，从而削弱这种治理改善效应。

这意味着：

- `PB -> CrashRisk` 不能作为主贡献；
- `PB -> DebtMaturity` 也不能作为主贡献；
- 主贡献必须来自 `PB x NaturalPersonLiabilityExposure`；
- Crash risk 可以用，但应降级为坏消息集中释放的资本市场后果；
- 债务短期化可以用，但应作为信贷收缩/追偿价值下降机制；
- 论文更适合走会计方向，主问题写“坏消息隐藏、披露、稳健性、信息风险”，而不是单纯公司金融的融资结构。

## 二、最优研究问题排序

### 第一优先级：自然人责任边界与坏消息隐藏

题目：

> 自然人责任边界与企业坏消息隐藏：来自个人破产制度试点的证据

X：

`PB_city,t x NaturalPersonLiabilityExposure_i,pre`

Y：

主 Y 应是坏消息隐藏/信息风险，而不是只写股价崩盘风险。优先顺序：

1. 风险披露具体性、风险披露模板化程度；
2. 会计稳健性、坏消息确认及时性；
3. 资产减值/信用减值及时性；
4. NCSKEW、DUVOL、Crash dummy 作为资本市场后果。

机制：

个人破产制度降低自然人控制人、自然人保证人和自然人股东失败后的毁灭性成本，使其隐瞒坏消息、强撑企业和拖延风险释放的私人收益下降。

最相近文献：

陈怡欣、张俊瑞（2026）《个人破产制度与企业盈余平滑》。他们的主线是城市 PB 平均效应、员工财务忧虑、盈余平滑，并把股价崩盘风险作为经济后果。本文要避开它，必须强调自然人责任暴露、控制人/保证人失败成本和同城同年高低暴露差异。

适合领域：

会计，尤其是资本市场会计、财务报告质量、披露和审计方向。

### 第二优先级：自然人追偿价值下降与债务期限压力

题目：

> 自然人追偿价值与企业债务期限压力：来自个人破产制度试点的证据

X：

`PB_city,t x NaturalPersonGuaranteeExposure_i,pre`

最低也要做到：

`PB x NaturalController x HighLeverage`

或：

`PB x Private x HighLeverage x Pledge`

Y：

债务期限结构、短期借款占比、一年内到期非流动负债、短贷长投、融资成本、抵押/质押贷款占比、商业信用替代。

机制：

个人破产制度降低自然人保证和个人连带责任的事后追偿价值，银行用更短期限、更强滚动审查和更高抵押要求应对。

最相近文献：

纪翔阁等《个人破产制度与信贷资源供给》。该文已经研究 PB 对银行信贷供给、担保结构和贷款期限的影响。因此本路线不能停在“债务短期化”，必须进一步接企业应对行为、会计/审计后果或坏消息隐藏。

适合领域：

金融或会计。如果只做融资结构，更偏金融；如果接审计风险、披露或财务报告，才适合会计。

### 第三优先级：失败容错、信贷收缩与企业风险承担/创新

题目：

> 失败容错还是信贷收缩？个人破产制度与企业风险承担

X：

`PB_city,t x FailureCostExposure_i,pre`

Y：

风险承担、研发投入、探索式创新、发明专利、研发资本化、创新披露质量。

机制：

失败保险效应鼓励风险承担；信贷收缩效应抑制外部融资依赖企业创新。

最相近文献：

汤旭东等已经做城市科技创业活跃度；Cerqueiro et al. 已经指出债务人保护可能通过外部融资收缩抑制小企业创新。因此这条线必须写成竞争机制，不能单向写“PB 促进创新”。

适合领域：

管理学或公司金融。当前不建议作为主线，因为城市产业政策、高新区政策和科创政策干扰较强。

## 三、X 构造：全题成败核心

### 最理想口径

核心变量：

```text
NPGuaranteeExposure_i,pre =
政策前自然人为公司债务提供保证担保金额 / 政策前有息债务
```

自然人范围包括：

- 实际控制人；
- 控股股东中的自然人；
- 法定代表人；
- 董事长、总经理、董监高；
- 自然人股东；
- 配偶或亲属；
- 企业经营者个人连带保证人。

进一步可拆：

```text
ControllerGuarantee_i,pre
ManagerGuarantee_i,pre
ShareholderGuarantee_i,pre
JointLiabilityGuarantee_i,pre
GuaranteeAmountRatio_i,pre
GuaranteeDummy_i,pre
```

这个 X 最硬，因为它直接连接个人破产制度改变自然人追偿价值。

### 可落地口径

分三组构造，不要混成一个黑箱指数：

1. `NPGuaranteeExposure`：自然人担保金额或 dummy；
2. `ControllerFailureCostExposure`：自然人实控、创始人控制、控制人持股、股权质押、控制权风险；
3. `DebtDependenceExposure`：高杠杆、银行债务占比高、短债占比高、低抵押资产、低现金持有。

对应机制应分开：

- `PB x ControllerFailureCostExposure` 应使坏消息隐藏下降；
- `PB x NPGuarantee/DebtDependenceExposure` 应使债务短期化上升。

### 最低可接受口径

如果自然人担保金额暂时拿不到，最低版本为：

```text
NaturalController_pre
Private_pre x NaturalController_pre
NaturalController_pre x HighLeverage_pre
NaturalController_pre x Pledge_pre
FailureCostIndex = NaturalController + Founder + Pledge + Private + HighLeverage
```

但这只能作为过渡。若正式论文最后只有 `natural_pre` 和 `failure_cost_pre`，贡献会明显下降。

### 不建议口径

不建议：

- 单独把 `high_lev_pre` 叫自然人责任暴露；
- 单独把 `private_pre` 叫自然人责任暴露；
- 使用政策后担保或政策后质押作为 Exposure；
- 把股权质押和自然人担保混为同一个概念；
- 用动态更新的 Exposure。

高杠杆是债务风险暴露，不是自然人责任暴露。民营企业也不等于自然人责任暴露。

## 四、Y 层级：主 Y、机制 Y、附录 Y

### 主 Y：坏消息隐藏/信息风险

主文应围绕“坏消息隐藏”而不是“股价崩盘风险”展开。

推荐主 Y 组合：

1. 风险披露具体性；
2. 风险披露模板化下降；
3. 会计稳健性；
4. 坏消息确认及时性；
5. 资产减值/信用减值及时性；
6. NCSKEW/DUVOL 作为资本市场后果。

### Crash risk 的使用方式

可以使用 NCSKEW/DUVOL，但不能作为唯一主 Y，也不建议出现在标题中。

合理写法：

> NCSKEW/DUVOL 是坏消息累积后集中释放的资本市场代理。

不合理写法：

> 个人破产制度降低股价崩盘风险。

因为这个平均效应已经被相邻会计文献触及。

### 机制 Y：债务期限压力

债务短期化是强机制，不是主故事。

推荐变量：

- 流动负债 / 总负债；
- 短期借款 / 有息债务；
- 长期借款 / 有息债务；
- 一年内到期非流动负债；
- 短贷长投；
- 融资成本；
- 抵押/质押贷款占比；
- 保证贷款占比；
- 商业信用替代。

### 辅助 Y

审计费用、非标意见、KAM、审计报告滞后适合作为审计/披露治理机制。

真实盈余管理适合作企业应对行为。

创新和风险承担适合作第三路线，不建议塞入主文。

## 五、基准模型和识别解释

主模型：

```text
Y_i,c,j,t =
  beta PB_c,t x Exposure_i,pre
  + Controls_i,t-1
  + FirmFE_i
  + CityYearFE_c,t
  + IndustryYearFE_j,t
  + epsilon_i,c,j,t
```

`beta` 的含义：

> 同一城市、同一年内，政策实施后高自然人责任暴露企业相对于低暴露企业的额外变化。

这不是 PB 的平均处理效应，而是政策冲击在企业层面自然人责任暴露上的异质影响。

`CityYearFE` 必须保留。它吸收：

- 城市营商环境改革；
- 法院资源变化；
- 城市金融环境变化；
- 城市疫情冲击；
- 当地创业政策和破产审判基础。

还必须增加一版：

```text
ExposureGroup x YearFE
```

理由是高自然人实控、高杠杆、民营或股权质押企业可能存在全国共同趋势。`CityYearFE` 吸收不了这种“高暴露企业全国趋势”。

正式主表建议三列：

1. FirmFE + YearFE + CityFE + IndustryYearFE；
2. FirmFE + CityYearFE + IndustryYearFE；
3. FirmFE + CityYearFE + IndustryYearFE + ExposureGroupYearFE。

第三列站住，论文才有说服力。

标准误：

- 主口径按政策赋值层级聚类，即城市或中级法院辖区；
- 如果政策按中院辖区编码，则按中院辖区聚类；
- 由于 treated clusters 可能较少，必须报告 wild cluster bootstrap 或随机化推断。

动态效应：

```text
Y_i,c,j,t =
  sum_k beta_k EventTime_c,t^k x Exposure_i,pre
  + Controls_i,t-1
  + FirmFE_i
  + CityYearFE_c,t
  + IndustryYearFE_j,t
  + epsilon_i,c,j,t
```

基准期 `k = -1`。

窗口：

```text
k <= -4, k = -3, k = -2, k = 0, k = +1, k = +2, k >= +3
```

可接受：

- `k = -3`、`k = -2` 单独不显著；
- `k = -3`、`k = -2` 联合不显著；
- 政策前没有同向单调漂移。

不可接受：

- `k = -2` 已经显著同向；
- 政策前连续上升或下降；
- 政策前系数幅度接近政策后主效应。

Staggered DID：

基准 TWFE 可以保留，但动态效应不能只靠传统 TWFE。可做：

- 高/低 Exposure 分组后的 Sun-Abraham；
- stacked DID；
- cohort-specific DDD；
- Callaway-Sant'Anna 作为高暴露组 ATT 稳健性。

## 六、机制与异质性证据链

### 失败容错机制

检验对象：

自然人控制人失败后的私人损失是否下降。

变量：

- 自然人实控；
- 创始人控制；
- 控制人股权质押；
- 控制人高持股；
- 高控制权风险；
- 无国资背书。

预期：

这些组中 `PB x NPExposure` 对坏消息隐藏的负向影响更强。

### 信贷收缩/追偿价值下降机制

检验对象：

自然人保证价值是否被债权人重新定价。

变量：

- 自然人担保金额；
- 连带责任保证；
- 银行债务占比；
- 高杠杆；
- 低抵押资产；
- 低现金持有；
- 外部融资依赖。

预期：

```text
PB x NPGuarantee/DebtExposure -> 短期债务占比上升
PB x NPGuarantee/DebtExposure -> 长期借款占比下降
PB x NPGuarantee/DebtExposure -> 融资成本上升
PB x NPGuarantee/DebtExposure -> 抵押/质押贷款要求上升
```

更关键的是，在这些企业中，PB 对坏消息隐藏的改善效应应当减弱，甚至反向。

### 审计/披露治理机制

失败容错主导时：

- 披露具体性上升；
- 模板化风险披露下降；
- 坏消息确认更及时；
- 会计稳健性增强。

信贷收缩主导时：

- 审计费用上升；
- 债务相关 KAM 增加；
- 持续经营段落或非标意见上升；
- 审计报告滞后增加。

因此审计变量不要简单假设单向，而应分组解释。

### 控制人激励机制

检验对象：

坏消息隐藏是否来自控制人强撑风险、维持股价和控制权。

变量：

- 股权质押比例；
- 控制权与现金流权分离；
- 控制人持股；
- 分析师覆盖；
- 机构持股；
- 媒体关注；
- 内部控制质量。

预期：

在外部监督弱、机构持股低、分析师少、质押高、控制权风险高的企业中，PB 对坏消息隐藏的降低效应更强。

## 七、最危险攻击与硬应对

1. PB 只是城市营商环境改革代理。应对：主模型保留 `CityYearFE`，并控制/剔除破产法庭、营商环境示范、金融法院、知识产权示范、高新区政策。

2. Exposure 内生。应对：政策前固定 Exposure、FirmFE、IndustryYearFE、ExposureGroupYearFE、前趋势、SOE placebo、多种 pre-window。

3. Crash risk 已经被做过。应对：标题和主 Y 不写 CrashRisk，改写坏消息隐藏、披露和稳健性；Crash risk 作为经济后果。

4. 中国 PB 怎么影响上市公司。应对：必须补自然人担保、控制人连带责任、股权质押等证据，证明上市公司存在自然人责任链条。

5. 政策城市和年份太少。应对：报告 treated city/court、treated firms；做 narrow/broad/formal/first-case/month-intensity；wild cluster bootstrap；leave-one-city-out。

6. COVID 干扰。应对：CityYearFE + ExposureGroupYearFE；剔除 2020-2022；控制疫情强度与行业冲击。

7. 自然人担保数据不可得。应对：先抽样验证可提取性；若只能用 `natural_pre`，文章层级下降；若连股权质押也没有，不建议冲高。

8. 结果只是基本面风险或股价波动。应对：控制收益波动、平均收益、换手率、现金流波动、经营风险；用披露、稳健性、减值及时性证明坏消息隐藏机制。

9. 政策口径混乱。应对：主口径用 narrow policy table，formal PB、debt clean-up、first-case、month intensity 做稳健性。

10. 多重试跑挑显著。应对：预设 Y 层级；附录报告全量候选 Y；必要时做 FDR 或 family-wise 调整。

## 八、下一步实证优先级

### 必须补的数据

第一优先级：

- 自然人为本企业债务提供担保的金额或 dummy；
- 实际控制人/控股股东/法定代表人/董监高个人连带保证；
- 股权质押；
- 自然人实控和创始人控制。

第二优先级：

- 风险披露文本；
- 风险披露具体性；
- 审计费用、KAM、非标意见；
- 会计稳健性；
- 减值及时性。

第三优先级：

- 债务期限细项；
- 短期借款、长期借款、一年内到期非流动负债；
- 抵押/质押/保证贷款；
- 融资成本；
- 商业信用替代。

### 必须先跑

1. 用硬 X 跑 `PB x NPGuaranteeExposure -> 披露具体性/稳健性/NCSKEW/DUVOL`；
2. 加 `ExposureGroup x YearFE` 后复跑；
3. 只针对硬 X 和主 Y 画事件研究；
4. 跑一阶机制：`PB x NPGuarantee/HighDebt -> short_liab_ratio/短期借款占比/长期借款占比`；
5. 剔除深圳、剔除浙江、剔除疫情年份；
6. wild cluster bootstrap。

### 暂时不要主打

- 裸 `PB -> 某个 Y`；
- `PB x high_lev_pre -> short_liab_ratio` 作为主线；
- 创新、投资效率、审计费用、真实盈余管理同时塞进主文；
- 没有自然人担保或股权质押数据前冲英文高刊。

### 放弃信号

出现以下两三项，应降级或停止：

- 硬 X 不显著，只有 `natural_pre` 显著；
- 加入 `ExposureGroup x YearFE` 后全部消失；
- `k = -2` 或 `k = -3` 前趋势显著且方向与政策后相同；
- SOE placebo 也显著；
- 剔除深圳或疫情年份后经济量级消失；
- 债务期限机制和坏消息隐藏机制方向完全无法协调。

### 继续推进信号

出现以下证据，可以继续：

- `NPGuaranteeExposure` 或 `Pledge x NaturalController` 显著；
- `PB x FailureCostExposure` 使坏消息隐藏下降；
- `PB x DebtDependence/NPGuarantee` 使债务短期化上升；
- 披露、稳健性、减值及时性与 crash risk 方向一致；
- SOE placebo 无效；
- wild bootstrap 仍稳；
- 剔除深圳/浙江后主方向不变。

## 九、当前版本的正式结构

主标题：

> 自然人责任边界与企业坏消息隐藏：来自个人破产制度试点的证据

主假说：

H1：个人破产制度实施后，自然人责任暴露越高的企业，坏消息隐藏程度越低。

H2：H1 在自然人实控、创始人控制、控制人股权质押和自然人担保企业中更强。

H3：对高自然人保证或高债务依赖企业，个人破产制度会提高债务短期化和融资滚动压力，并削弱 H1 的信息风险改善效应。

贡献写法：

本文不把个人破产制度视为一般城市政策冲击，而是将其解释为自然人责任边界变化。通过构造企业政策前自然人责任暴露，并在企业固定效应、城市年份固定效应和行业年份固定效应下比较同城同年高低暴露企业，本文识别个人法制度如何影响上市公司坏消息隐藏和信息风险。文章进一步区分失败容错的治理改善效应与自然人追偿价值下降的信贷收缩效应，从而解释同一制度冲击为何可能同时降低部分企业的信息风险并加剧另一类企业的债务期限压力。

## 十、2026-05-06 新 Y 试跑结果

根据关键词共联结果，当前最值得抢救的 Y 不是再找一个孤立经济后果，而是围绕坏消息隐藏过程变量展开：风险披露具体性、会计稳健性 / 损失确认、减值及时性、审计风险反应。

本轮只用本地已有数据试跑了可立即构造的会计确认类代理。风险披露具体性、KAM 文本、精确资产减值 / 信用减值损失在当前 parquet 数据集中尚不可得，因此没有强行跑。

产出文件：

```text
/Users/mac/computerscience/0做完了/15会计研究/v4/pb_did_trial/run_pb_new_y_probe.py
/Users/mac/computerscience/0做完了/15会计研究/v4/pb_did_trial/pb_new_y_probe_report.md
/Users/mac/computerscience/0做完了/15会计研究/v4/pb_did_trial/pb_new_y_probe_decisions.csv
```

当前可用代理：

```text
LossDummy        = 净利润 < 0
LossMagnitude    = max(-净利润 / 期初总资产, 0)
NegAccrual       = max(-(净利润 - 经营现金流) / 期初总资产, 0)
ConservAccrual   = -(净利润 - 经营现金流) / 期初总资产
IntangibleDecrease / DevExpDecrease = 无形资产或开发支出下降代理，仅作弱代理
```

初步结果：

- `PB x natural_pre -> ConservAccrual`：主规格系数 0.0180，p < 0.001；加入 `ExposureGroup x YearFE` 后系数 0.0165，p < 0.001；近端前趋势 p = 0.180。
- `PB x natural_pre -> NegAccrual`：主规格系数 0.0115，p < 0.001；加入 `ExposureGroup x YearFE` 后系数 0.0103，p < 0.001；近端前趋势 p = 0.398。
- `PB x failure_cost_pre -> ConservAccrual`：主规格系数 0.0221，p < 0.001；加入 `ExposureGroup x YearFE` 后系数 0.0160，p = 0.004；近端前趋势 p = 0.358。
- `PB x failure_cost_pre -> NegAccrual`：主规格系数 0.0165，p < 0.001；加入 `ExposureGroup x YearFE` 后系数 0.0148，p < 0.001；近端前趋势 p = 0.510。
- `PB x failure_cost_pre -> LossDummy`：主规格系数 0.0733，p < 0.001；加入 `ExposureGroup x YearFE` 后系数 0.0478，p = 0.084；近端前趋势 p = 0.114。

保守解释：

1. 这轮结果说明“会计确认 / 坏消息确认”比单纯债务短期化更有救。自然人责任暴露高的企业在 PB 后出现更强的负向应计和损失确认倾向，与“失败成本下降后不再强行隐藏坏消息”一致。
2. 但它还不是可直接投稿的主证据。事件研究中，后期单点系数并不强，说明当前代理更适合做筛查证据，不能替代更硬的风险披露、会计稳健性或减值及时性变量。
3. `DevExpDecrease` 在高债务暴露下很显著，但它只是开发支出余额下降，不能写成精确减值或稳健性主结果。
4. 下一轮应优先补三个硬 Y：风险披露具体性、Basu / C-score 式会计稳健性、资产减值 / 信用减值及时性。若这些和 `NCSKEW/DUVOL` 同方向，文章可以从“Crash risk 被抢跑”转向“自然人责任边界影响坏消息确认过程”。

因此，当前判断不是“换 Y 已经成功”，而是：

```text
可抢救，但主线应从 Crash risk 后果转为坏消息确认过程；
现有会计确认代理有信号；
需要补风险披露或精确会计确认数据来把 Y 做硬。
```

## 十一、2026-05-06 风险披露质量第一版推进

根据“不要围着崩盘风险打转”的新约束，已正式把新终点 Y 切到：

```text
自然人责任边界与企业风险披露质量
```

本轮已从 2015-2023 年 A 股年报 TXT 压缩包中抽取风险披露段落，并合并到 PB DID 面板。抽取口径偏保守，只承认明确风险标题下的段落，不使用泛风险关键词段落兜底。

产出文件：

```text
/Users/mac/computerscience/0做完了/15会计研究/v4/risk_disclosure_trial/extract_risk_disclosure_features.py
/Users/mac/computerscience/0做完了/15会计研究/v4/risk_disclosure_trial/run_pb_risk_disclosure_probe.py
/Users/mac/computerscience/0做完了/15会计研究/v4/risk_disclosure_trial/risk_disclosure_features_2015_2023.parquet
/Users/mac/computerscience/0做完了/15会计研究/v4/risk_disclosure_trial/pb_risk_disclosure_probe_report.md
/Users/mac/computerscience/0做完了/15会计研究/v4/risk_disclosure_trial/风险披露质量_第一版实证备忘_20260506.md
```

第一版发现：

- 风险披露指标可构造，2015-2023 年共 30,496 个 firm-year，4,938 家公司；
- 明确风险段识别率从 2015 年的 80.0% 上升至 2023 年的 92.1%；
- `PB x high_lev_pre -> risk_specificity_index` 较稳：主规格系数 0.0781，p = 0.006；加入 `ExposureGroup x YearFE` 后系数 0.0737，p = 0.026；近端前趋势 p = 0.845；
- `PB x high_lev_pre -> risk_quality_index` 有弱稳健信号：主规格系数 0.0673，p = 0.043；严格规格系数 0.0622，p = 0.081；近端前趋势 p = 0.944；
- `PB x natural_pre / failure_cost_pre -> has_risk_text` 有动态信号，但严格趋势控制下不够稳；
- `PB x failure_cost_pre -> risk_debt_per10k` 显示政策后两年债务风险披露上升，近端前趋势较干净。

当前解释必须保守：

```text
风险披露方向可以继续；
但第一版最强质量结果来自 high_lev_pre，而不是 natural_pre；
因此还不能直接宣称自然人责任边界提高风险披露质量。
```

更稳的中期表述是：

```text
个人破产制度改变自然人责任边界和债权追偿预期后，
高责任/债务暴露企业更倾向披露风险；
其中债务暴露高的企业风险披露更具体，债务风险内容更多。
```

下一步硬门槛：

1. 补 `自然人担保 / 个人连带保证 / 股权质押`，否则 X 仍偏软；
2. 人工校验 100-200 条风险披露段，确认不是财报附注或金融工具风险管理误抓；
3. 将风险披露 Y 收敛为三类：`has_risk_text`、`risk_specificity_index/risk_quality_index`、`risk_debt_per10k`；
4. 补硬 X 后重跑，不再围绕 `NCSKEW/DUVOL` 组织主线。

## 十二、2026-05-06 晚：合作反馈后的冻结版主线

合作老师初步认可“自然人责任边界与企业风险披露质量 / 可核验性”方向后，当前研究设计正式从“坏消息隐藏”进一步收束为：

```text
自然人责任边界与企业风险披露可核验性：来自个人破产制度试点的证据
```

这个版本不再把股价崩盘风险、债务期限结构、审计费用、创新或会计稳健性作为主 Y。
这些变量后续只作为机制、边界条件或经济后果。

### 1. 新研究问题

```text
当本地个人破产制度或个人债务集中清理机制出现后，
原先企业风险与自然人风险高度绑定的企业，
是否会减少空泛、模板化风险叙事，
增加更具体、可外部核验的责任风险、债务风险、担保风险、诉讼执行风险和偿债安排表达？
```

### 2. 理论靠近概念

“企业预先自然人责任暴露”不是现成的标准术语，因此论文中不宜直接当作成熟概念使用。
更合适的写法是把它靠到以下既有文献概念：

```text
business-personal risk non-separation
personal guarantee / owner guarantee
contracting out of limited liability
entrepreneurial downside risk / failure cost
personal recourse exposure
```

建议中文定义：

```text
借鉴 business-personal risk non-separation、personal guarantee 和 limited liability 文献，
本文将企业债务风险通过自然人担保、控制人个人连带责任和股权质押等安排传导至自然人主体的程度，
定义为企业政策前自然人责任暴露。
```

### 3. 新主模型

```text
RiskDisclosureVerifiability_i,c,j,t
= beta PB_c,t × NP_LiabilityExposure_i,pre
+ FirmFE_i + CityYearFE_c,t + IndustryYearFE_j,t
+ Controls_i,t-1 + epsilon_i,c,j,t
```

严格规格继续加入：

```text
ExposureGroup_i × YearFE_t
```

识别含义是：

```text
同一城市、同一年内，政策实施后自然人责任暴露更高的企业，
相对于低暴露企业，风险披露可核验性是否提高。
```

### 4. Y 的新层级

主 Y 不再是一般风险披露质量，而是：

```text
risk_verifiability_index
risk_verif_per10k
risk_verif_legal_process_per10k
risk_verif_guarantee_pledge_per10k
risk_verif_debt_distress_per10k
```

更贴自然人责任边界的主 Y 是：

```text
np_liability_verifiability_index
np_liability_risk_verif_per10k
np_liability_np_actor_per10k
np_liability_np_guarantee_per10k
np_liability_recourse_capacity_per10k
np_liability_control_right_per10k
```

对照 Y 是：

```text
has_risk_text
risk_chars_ln
risk_specificity_index
risk_quality_index
risk_boilerplate_ratio
risk_debt_per10k
```

### 5. X 的硬门槛

当前临时 X 包括：

```text
natural_pre
failure_cost_pre
high_lev_pre
natural_highlev_pre
private_highlev_pre
```

但正式文章必须补：

```text
自然人担保 dummy / 金额
实际控制人、控股股东、法定代表人、董监高个人连带保证
控制人股权质押比例
自然人实控 × 股权质押
自然人实控 × 银行债务依赖
```

其中：

```text
high_lev_pre 不是自然人责任暴露，只能解释为债务风险 / 追偿压力暴露。
natural_pre 和 failure_cost_pre 是低成本代理，不能单独支撑高贡献文章。
```

新的冻结版设计文档已单独保存：

```text
/Users/mac/computerscience/0做完了/15会计研究/v4/docs/自然人责任边界_风险披露可核验性_研究设计_2026-05-06.md
```

正在新增的可核验披露特征构造脚本：

```text
/Users/mac/computerscience/0做完了/15会计研究/v4/risk_disclosure_trial/build_verifiable_risk_disclosure_features.py
```

## 十三、2026-05-06 晚：可核验披露第二版实证结果

本轮新增两类数据：

```text
1. 风险披露可核验性 Y：
   /Users/mac/computerscience/0做完了/15会计研究/v4/risk_disclosure_trial/risk_disclosure_verifiable_features_2015_2023.parquet

2. 政策前年报全文自然人责任暴露文本代理：
   /Users/mac/computerscience/0做完了/15会计研究/v4/risk_disclosure_trial/np_liability_text_exposure_pre_2016_2018.csv
```

政策前文本代理覆盖 3,387 家公司：

```text
ar_np_guarantee_pre          0.388
ar_controller_pledge_pre     0.298
ar_personal_recourse_pre     0.147
ar_np_liability_text_pre     0.561
ar_np_liability_index_pre    0.278
```

需要注意，这是年报全文文本代理，不是正式担保金额数据库。它比 `natural_pre` 更靠近自然人责任链条，但仍需人工核验误抓。

第二版回归结果显示，当前最稳的不是“自然人实控企业提高自然人责任披露”，而是：

```text
高债务/控制人质押暴露企业，在个人破产制度冲击后，
提高更可核验的债务、担保、质押和控制权风险披露。
```

较可用结果包括：

```text
PB × high_lev_pre -> risk_verifiability_index
主规格 coef = 0.0675, p = 0.0566
严格规格 coef = 0.0642, p = 0.0809
近端前趋势 p = 0.3369
政策后 +1 年 p = 0.0072；>=+2 年 p = 0.0179
```

```text
PB × high_lev_pre -> np_liability_control_right_per10k
主规格 coef = 0.0932, p = 0.0518
严格规格 coef = 0.1196, p = 0.0768
近端前趋势 p = 0.2092
政策后 +1 年 p = 0.0344；>=+2 年 p = 0.0250
```

```text
PB × failure_cost_pre -> risk_verif_debt_distress_per10k
主规格 coef = 0.9682, p = 0.0460
严格规格 coef = 1.1274, p = 0.0832
近端前趋势 p = 0.9526
```

```text
PB × ar_controller_pledge_pre -> risk_verif_guarantee_pledge_per10k
主规格 coef = 1.5773, p = 0.0872
严格规格 coef = 1.6196, p = 0.0699
近端前趋势 p = 0.3466
```

同时，`ar_np_guarantee_pre` 对 `risk_verif_guarantee_pledge_per10k` 的主规格和动态信号较强，但加入 `ExposureGroup × YearFE` 后不稳：

```text
主规格 coef = 0.6385, p = 0.0054
严格规格 coef = 0.3737, p = 0.2008
近端前趋势 p = 0.5780
政策后 +1 年 p = 0.0001；>=+2 年 p = 0.0265
```

当前写法应收束为：

```text
个人破产制度改变自然人责任边界和债权追偿预期后，
债务压力较高、控制人质押/冻结风险较高的企业，
会从空泛风险叙事转向更具体、可核验的风险披露，
尤其是债务风险、担保质押风险和控制权风险披露。
```

不能写成：

```text
个人破产制度普遍提高企业风险披露质量。
```

也不能写成：

```text
自然人责任暴露企业一定提高自然人责任披露。
```

第二版实证备忘已保存：

```text
/Users/mac/computerscience/0做完了/15会计研究/v4/risk_disclosure_trial/风险披露可核验性_第二版实证备忘_20260506.md
```
