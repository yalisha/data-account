# VSI 扩展 Y 候选池：年报数据要素 X

日期：2026-04-30

## 判断前提

- 特刊主题更偏 `AI / data investment / data economy` 对生产率、金融市场、市场结构、韧性、绿色转型的经济后果。本文不要把 X 直接写成真实 `data investment`，更稳的说法是：**年报披露的数据要素利用、数据资源经营闭环、数据治理/应用实质信息**。
- X 口径建议主推 `DU_llm`、`DUcore`、`DUclosedloop`、`DUchain_count`、`DUkw_mda`，`DU_kw` 做基准。单纯关键词频率容易被审稿人理解成数字化叙事。
- 主 Y 的筛选标准：第一，能和“数据作为资产/生产要素”形成机制闭环；第二，本地已有数据或轻量构造能跑；第三，避开已经拥挤的 `innovation/R&D/patent`、`investment efficiency`、`SYNCH` 主线。

## 最优先候选

| 优先级 | Y | 口径 | 预期方向 | 本地可行性 | 判断 |
|---:|---|---|---|---|---|
| 1 | `PEAD60_abs` | 公告后 60 日绝对异常收益 / residual post-announcement adjustment | `DU -> Y < 0` | 已构造、已显著 | 当前最稳。不是 signed PEAD，而是公告后残余价格调整幅度更小，最贴“信息更快进入价格”。 |
| 2 | `AnnouncementAbsorptionRatio` | `PEAD60_abs / EA_absCAR02` 或 `PEAD60_abs / (abs(EA_CAR02)+epsilon)` | `DU -> Y < 0` | 可由现有事件层数据轻量构造 | 比单独 `PEAD60_abs` 更像“公告信息吸收效率”。如果显著，主线比 PriceDelay 更新。 |
| 3 | `CS_Spread` | Corwin-Schultz 高低价隐含价差，年均 | `DU -> Y < 0` | 已构造，`DU_llm` 强、`DU_kw` 边际 | 微观结构摩擦下降。适合做第二主结果或补充 Y。 |
| 4 | `RetAutoCorr` / delay-style autocorrelation | 年度收益自相关或反转强度 | `DU -> Y < 0` | `panel_supply_chain_spillover.parquet` 已有 | 和 `PriceDelay` 同属信息扩散，但构造不同。适合做稳健性，不一定比 `PEAD60_abs` 新。 |
| 5 | `ImpliedCostOfEquity` | PEG / MPEG / Ohlson-Juettner 型隐含权益资本成本，基于分析师预测和价格 | `DU -> Y < 0` | 需要用 `analyst_forecast` + 股价构造 | 很贴 finance/economic resilience。若样本够大，可能比分析师误差更像经济后果。 |

## 第二梯队：能跑，但更像机制/扩展

| 优先级 | Y | 口径 | 预期方向 | 本地可行性 | 判断 |
|---:|---|---|---|---|---|
| 6 | `AF_Error_abs` / `SUE_abs_price` | 分析师绝对预测误差 / 绝对盈余意外 | `DU -> Y < 0` | 已构造，一显著一边际 | 可解释“信息环境改善”，但样本较小，不适合单独当主 Y。 |
| 7 | `AF_Coverage` / `ReportFreq` | 分析师覆盖度或报告频率 | `DU -> Y > 0` | 已有，当前偏弱 | 可以做可见性机制。不要当主线。 |
| 8 | `ForecastDisp` / `RatingDisp` | 分析师预测分歧 / 评级分歧 | `DU -> Y < 0` | 已有，但现有结果不稳 | 老师之前已觉得链条远，除非和 PEAD/CoE 联动，不建议主推。 |
| 9 | `EarnQualAbs` / `absDA` | 盈余质量、绝对应计盈余操纵 | `DU -> Y < 0` | 已有 | “数据治理提高财务透明度”逻辑成立，但已有 big data policy -> earnings management 文献，作为辅线更稳。 |
| 10 | `AuditFee` / non-standard audit opinion | 审计费用、审计风险 | `DU -> Y < 0` 或混合 | 已有审计数据 | 可以讲数据治理降低审计不确定性，但离特刊主旨略偏会计。 |
| 11 | `InstStable` / `InstHold` | 稳定机构持股或机构持股比例 | `DU -> Y > 0` | 已有 | 可作为资本市场吸收条件或机制，不宜作为主 Y。 |
| 12 | `IdioVol` / `RetVol` | 特质波动、收益波动 | `DU -> Y < 0` | 已有 | 可讲金融韧性，但方向容易被“更多信息释放导致波动上升”反驳。 |

## 第三梯队：贴特刊，但要防撞题或补数据

| 优先级 | Y | 口径 | 预期方向 | 本地可行性 | 判断 |
|---:|---|---|---|---|---|
| 13 | `TFP` | 企业全要素生产率 | `DU -> Y > 0` | 已有 | 最贴 Economic Modelling 的 productivity，但数字化/数据要素 -> TFP 很拥挤，创新压力大。 |
| 14 | `CashFlowVol` | 现金流波动 | `DU -> Y < 0` | 已有、历史机制较稳 | 可改写成 operational resilience。适合放机制或第二篇分支，不建议抢主 Y。 |
| 15 | `SCConc` / `CustConc` / `SuppConc` | 供应链集中度、客户/供应商集中度 | `DU -> Y < 0` | 已有、历史机制较稳 | 贴 global value chains / resilience，但更像管理机制。 |
| 16 | `InvestIneff` | 投资不足/过度投资的绝对偏离 | `DU -> Y < 0` | 已有 | 和 data element marketization -> investment efficiency 已撞得比较近，不建议主推。 |
| 17 | `FinAsset` / `Financialization` | 金融资产配置或金融化 | `DU -> Y < 0` | 已有，但前期机制未闭合 | 本地三步不理想，除非重新做成“数据利用减少短期金融套利”，否则风险高。 |
| 18 | `RealInvestment` / `DevOutlayRatio` | 实体投资、研发资本化/开发支出占比 | `DU -> Y > 0` | 可由财务报表轻量构造 | 比普通 R&D 稍新，但仍容易滑向 innovation/investment crowded lane。 |

## 需要外部补数才可能变强

| Y | 需要数据 | 为什么值得想 | 风险 |
|---|---|---|---|
| 数据安全/隐私处罚风险 | 行政处罚、网信办处罚、交易所问询、诉讼公告 | 贴 `data regulation and competition policy`，能把数据要素从“好东西”写成治理约束 | 数据整理成本高，事件稀疏 |
| 数据资源入表 / 2024 数据资产会计确认 | 2024 后“数据资源”会计科目、年报附注 | 最接近真实 data investment / data assetization | 窗口短，样本年少，难做主检验 |
| AI/data 人才招聘 | 招聘文本、岗位技能、数字人才 | 能区分“说数据”与“真投入数据能力” | 需要外部爬取或第三方库 |
| 绿色转型/能效 | 绿色专利、碳排、环境绩效 | 特刊有 green transition 入口 | 绿色创新已拥挤，且会偏离现有资本市场主线 |
| 市场竞争/markup | 行业竞争、企业 markups | 贴 market structure / competition | A 股企业 markups 构造复杂，识别压力大 |

## 不建议主推

- `innovation / R&D / patent`：外部近文献已经很多，且 ScienceDirect 已有数据要素与企业创新文章。
- `SYNCH`：显著但太容易撞“data assets disclosure -> stock price synchronicity”。
- `PEAD_signed`：本地结果不支持，不能写“降低传统盈余漂移”。
- `crash risk`：`NCSKEW/DUVOL` 当前主要单测度显著，`DU_llm` 不稳，不适合主推。
- `investment efficiency`：特刊贴合度高，但已有 data element marketization -> corporate investment efficiency 近文献，除非换成完全不同的识别和机制，否则不划算。

## 当前推荐组合

1. 主线：`DU_realized_disclosure -> PEAD60_abs`，标题口径用“post-announcement price adjustment / capital-market information assimilation”。
2. 补充：`CS_Spread`、`RetAutoCorr`、`PriceDelay`，形成价格吸收速度和交易摩擦的证据链。
3. 机制：`AF_Error_abs`、`ImpliedCostOfEquity`、`InstStable` 三选二；其中 `ImpliedCostOfEquity` 最值得新增构造。
4. 扩展：`CashFlowVol` / `SCConc` 作为 operational resilience / value-chain resilience，不要和主线抢位置。
