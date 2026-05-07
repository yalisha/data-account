# 数据补充列表：用于强化 NarrHardGap 的硬能力侧

日期：2026-05-03

目标：把当前 `NarrHardGap_base` 从“年报叙事 - 行业/地区潜力”升级为“年报叙事 - 企业级可验证硬能力”。

核心思路：

```text
NarrHardGap = PctYear(DataNarrative) - PctYear(HardDataCapability)
```

其中 `HardDataCapability` 尽量来自年报之外，至少要有企业级硬证据。

## A. 必须优先补

| 优先级 | 数据 | 推荐变量名 | 作用 | 来源候选 | 备注 |
|---:|---|---|---|---|---|
| A1 | 数据/AI/软件相关专利 | `DataPatent_ln` | 最核心的企业级硬能力 | CNRDS、CSMAR 专利、国家知识产权局、incoPat、智慧芽 | 最像 AI washing 文献里的 “disclosure minus patent/action” |
| A2 | 软件著作权 | `SoftwareCopyright_ln` | 数据系统、平台、算法系统的硬证据 | CNRDS、CSMAR、企查查/天眼查、国家版权局相关数据 | 比专利更贴数据平台、数据中台、软件系统 |
| A3 | 数据相关招聘 | `DataHiringShare` | 当期投入和真实需求 | CSMAR 招聘、Wind 招聘、BOSS/智联/猎聘爬取、上市公司招聘公告 | 能补专利的滞后性，适合解释“正在建设数据能力” |
| A4 | AI/data/digital capex | `AIDataCapex_ln` | 投资强度/资本化投入 | CSMAR 数字化投资、年报附注、在建工程/无形资产明细、管理层讨论文本抽取 | 难度较高，但概念最贴 data investment |

最低可接受组合：

```text
DataPatent_ln + SoftwareCopyright_ln
```

更好的组合：

```text
DataPatent_ln + SoftwareCopyright_ln + DataHiringShare
```

投稿版理想组合：

```text
DataPatent_ln + SoftwareCopyright_ln + DataHiringShare + AIDataCapex_ln
```

## B. 强烈建议补

| 优先级 | 数据 | 推荐变量名 | 作用 | 来源候选 | 备注 |
|---:|---|---|---|---|---|
| B1 | 数据产品/数据服务 | `DataProduct` | 证明企业把数据能力产品化 | 数据交易所披露、企业官网、公告、新闻、招股书/年报文本 | 可作为可见市场验证 |
| B2 | 数据交易所挂牌/成交 | `DataTransaction` | 数据要素市场参与证据 | 各地数据交易所、公开挂牌产品目录、企业公告 | 样本可能小，但很贴特刊 |
| B3 | 数据确权/数据知识产权/数据资源登记 | `DataAssetConfirmation` | 数据资源被制度化确认 | 地方数据知识产权登记平台、数据局/交易所公告 | 适合 2022 后制度背景 |
| B4 | 数据安全/隐私/数据治理认证 | `DataGovernanceCert` | 数据治理能力证据 | ISO/等保/信通院认证、企业公告、官网 | 可补“有数据但不一定能合规使用” |

这些不一定要全补。比较适合作为：

```text
HardCap_market
HardCap_governance
```

或者做异质性：

```text
数据要素市场制度环境更强的企业/地区，错配后果是否更明显。
```

## C. 会计验证层

| 优先级 | 数据 | 推荐变量名 | 作用 | 当前状态 | 备注 |
|---:|---|---|---|---|---|
| C1 | 数据资产入表金额 | `DataAsset_ln` | 会计确认/会计验证 | 已有，2024 正样本 22 个 | 不适合作主 X，只能做验证层 |
| C2 | 是否入表 | `BookEntry` | 会计确认虚拟变量 | 已有，2024 正样本 22 个 | 可区分“有叙事无入表”和“有叙事有入表” |
| C3 | 年报数据资源附注披露 | `DataResourceNote` | 比金额更宽的会计披露证据 | 年报附注文本抽取 | 可能比 `BookEntry` 样本大 |
| C4 | 数据资产相关审计强调/问询/更正 | `DataAssetAuditFlag` | 会计不确定性/验证成本 | 审计报告、问询函、更正公告 | 适合作补充，不适合主识别 |

会计验证层建议写法：

```text
HardCap_account = DataAsset_ln / BookEntry / DataResourceNote
```

但主文不要把它作为唯一硬能力。

## D. 可作为控制或补充，不建议做主硬能力

| 数据 | 当前/推荐变量名 | 用法 | 原因 |
|---|---|---|---|
| 高新技术企业 | `HighTech` | 控制/补充 | 偏企业资质，不专指数据能力 |
| 战略新兴产业 | `StrategicEmerging` | 控制/补充 | 行业标签偏粗 |
| 数字经济行业 | `DigEconCore` | 控制/补充 | 行业属性，不是企业行动 |
| 数据产业集群/产业位置 | `IndustryCluster` | 控制/补充 | 地区/行业环境 |
| 区位熵 | `LQ` | 控制/补充 | 更像外部生态，不是企业硬能力 |

这些可以保留在 `HardBaseRaw`，但最终论文不能只靠它们。

## E. 推荐最终指数结构

### E1. 主指数

```text
HardCap_core =
    rowmean(
        z(DataPatent_ln),
        z(SoftwareCopyright_ln),
        z(DataHiringShare)
    )
```

如果拿不到招聘：

```text
HardCap_core =
    rowmean(
        z(DataPatent_ln),
        z(SoftwareCopyright_ln)
    )
```

### E2. 增强指数

```text
HardCap_extended =
    rowmean(
        z(DataPatent_ln),
        z(SoftwareCopyright_ln),
        z(DataHiringShare),
        z(AIDataCapex_ln),
        z(DataProduct),
        z(DataTransaction),
        z(DataAssetConfirmation)
    )
```

### E3. 会计验证指数

```text
HardCap_account =
    rowmean(
        z(DataAsset_ln),
        z(BookEntry),
        z(DataResourceNote)
    )
```

## F. 最终 X 变量清单

主文：

```text
NarrHardGap_core = PctYear(DU_kw) - PctYear(HardCap_core)
DataWashing_core = 1[DU_kw high, HardCap_core low]
DataHushing_core = 1[HardCap_core high, DU_kw low]
DataVerified_core = 1[DU_kw high, HardCap_core high]
```

稳健性：

```text
NarrHardGap_extended
NarrHardGap_residual
NarrHardGap_indyear
DataWashing_core_strict
DataHushing_core_strict
DataVerified_core_strict
```

会计验证：

```text
NarrAcctHardGap
TalkNoBook
TalkWithBook
BookNoTalk
```

## G. 数据补充优先顺序

第一批，最值得马上找：

```text
1. 数据/AI/软件相关专利
2. 软件著作权
```

第二批，能显著增强说服力：

```text
3. 数据相关招聘
4. AI/data/digital capex
```

第三批，适合做制度背景和补充验证：

```text
5. 数据产品/数据交易
6. 数据确权/数据知识产权登记
7. 年报数据资源附注披露
```

第四批，只做辅助：

```text
8. 数据治理/安全认证
9. 企业官网数据平台/数据产品文本
10. 媒体报道/新闻中的数据项目
```

## H. 数据拿到后的最低验证

补完每个数据后，都要先做这些检查：

```text
1. firm-year coverage
2. nonzero count by year
3. correlation with DU_kw
4. correlation with current HardBaseRaw
5. distribution by industry
6. whether it mechanically overlaps with Y
```

必须输出：

```text
results/stata/hardcap_data_coverage.csv
results/stata/hardcap_component_corr.csv
results/stata/hardcap_component_summary.csv
```

## I. 最小可发表版本

如果资源有限，最低版本可以是：

```text
DU_kw
DataPatent_ln
SoftwareCopyright_ln
DataAsset_ln / BookEntry as accounting verification
```

对应主 X：

```text
HardCap_core = rowmean(z(DataPatent_ln), z(SoftwareCopyright_ln))
NarrHardGap_core = PctYear(DU_kw) - PctYear(HardCap_core)
```

这比当前 `HardBaseRaw` 稳，因为它已经是企业级硬能力。

## J. 不建议花太多时间的数据

暂不优先：

```text
纯新闻情绪
普通数字化转型词频
宽泛 IT 投入
是否高新技术企业
是否数字经济行业
地方数字经济政策数量
```

原因：这些会把题目又拉回普通数字化转型或政策环境，不足以支撑“数据要素叙事-硬能力错配”。
