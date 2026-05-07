# 本地第三方资料中的数据资产与数据知识产权变量盘点

日期：2026-05-07

## 结论

本地资料里已经有可以直接强化 v3 设计的硬证据。最重要的是资产负债表中的三个数据资源入表金额字段：

- `A001123101`：数据资源（存货）
- `A001218201`：数据资源（无形资产）
- `A001219101`：数据资源（开发支出）

字段说明均写明“2024年起开始使用”。这三个字段比年报文本中的“数据资产/数据入表”关键词更硬，适合作为可核验披露的会计验证层，但由于只从 2024 年开始，不能直接作为 2016-2017 事前 `PreWash` 的硬能力基准。

专利可以算广义知识产权，但不等同于政策语境里的“数据知识产权”。在本项目中，专利更适合进入“真实数字/数据能力”的 hard capability 侧；只有“数据知识产权登记、数据产品登记、数据资源确权、数据产品挂牌交易”等才更接近数据资产权属或可交易性的外部验证。

## 可用资料

### 1. 数据资源入表金额

路径：

`/Users/mac/computerscience/第三方资料/第三方数据资源/上市公司财务信息/FS_Combas.xlsx`

字段说明：

`/Users/mac/computerscience/第三方资料/第三方数据资源/上市公司财务信息/FS_Combas[DES][xlsx].txt`

关键字段：

- `A001123101 [其中：数据资源（存货）]`：资产负债表日确认为存货的数据资源的期末账面价值。2024年起开始使用。
- `A001218201 [其中：数据资源（无形资产）]`：资产负债表日确认为无形资产的数据资源的期末账面价值。2024年起开始使用。
- `A001219101 [其中：数据资源（开发支出）]`：资产负债表日正在进行数据资源研究开发项目满足资本化条件的支出金额。2024年起开始使用。

初步扫描结果：

- 合并报表 `2024-12-31` 样本：5,484 条。
- 任一数据资源金额为正：91 条。
- 分项为正：
  - 数据资源（存货）：3 条。
  - 数据资源（无形资产）：78 条。
  - 数据资源（开发支出）：28 条。
- 合计金额：
  - 数据资源（存货）：107,328,363.10。
  - 数据资源（无形资产）：1,205,723,391.95。
  - 数据资源（开发支出）：767,826,252.33。

初步用途：

- `BookDataResource = 1[A001123101 + A001218201 + A001219101 > 0]`
- `BookDataResourceAmount = A001123101 + A001218201 + A001219101`
- `lnBookDataResource = ln(1 + BookDataResourceAmount)`
- `BookDataResourceRatio = BookDataResourceAmount / TotalAssets`

在论文中建议作为 external validation 或 2024 后补充结果，而不是主 Y。

### 2. 数字发明专利授权量

路径：

`/Users/mac/computerscience/第三方资料/第三方数据资源/上市公司数据相关信息/数字发明专利授权情况表（年）190317870.zip`

字段：

- `sgnyear`：统计年度。
- `symbol`：证券代码。
- `shortname`：证券简称。
- `diginvpatautnum`：数字发明专利授权量。

初步用途：

该变量可以继续作为 hard capability 的组成部分。它比普通数字化文本硬，但仍然只是数据/数字技术能力，不是数据资源权属或数据产品可交易性的直接证据。

### 3. 专利明细

路径：

`/Users/mac/computerscience/第三方资料/第三方数据资源/上市公司数据相关信息/专利明细情况093018090(仅供哈佛大学使用).zip`

字段：

- `Symbol`
- `EndDate`
- `PatentName`
- `PatentType`
- `ApplicationNumber`
- `PatentNumber`
- `ApplicationDate`
- `GrantDate`
- `LegalStatus`
- `Applicant`

初步用途：

可以从 `PatentName` 中构造数据/AI/软件相关专利，例如标题含“数据、数据库、数据处理、数据采集、数据分析、人工智能、机器学习、算法、平台、云计算”等。这个变量适合加强 `HardCap_direct_v1`，尤其是把“数字发明专利授权量”进一步细化成“数据相关专利/AI相关专利/软件平台相关专利”。

但不建议把它直接命名为“数据知识产权”。更准确的说法是“数据相关技术知识产权”或“数据技术专利”。

### 4. 企业数据要素披露与开发利用指数

路径：

- `/Users/mac/computerscience/第三方资料/第三方数据资源/上市公司数据相关信息/企业数据要素信息披露程度明细表121712248(仅供哈佛大学使用).zip`
- `/Users/mac/computerscience/第三方资料/第三方数据资源/上市公司数据相关信息/企业数据要素开发利用指数表121115351(仅供哈佛大学使用).zip`
- `/Users/mac/computerscience/第三方资料/第三方数据资源/上市公司财务信息/企业数据要素开发利用情况.7z;filename_=企业数据要素开发利用情况`

这些资料本质上仍是年报关键词、句频、句子字数和报告总字数统计。它们可以帮助复核文本口径，但不能单独证明会计入表、数据权属或真实交易。

## 对 v3 的放置建议

## 已接入 v3 的验证结果

已新增脚本：

- `src/python/extract_balance_sheet_data_resource_for_stata.py`
- `src/stata/did_data_exchange_bs_data_resource_validation_v1.do`

已生成结果：

- `results/data/balance_sheet_data_resource_annual.csv`
- `results/data/balance_sheet_data_resource_period.csv`
- `results/data/did_data_exchange_bs_data_resource_validation_v1_panel.dta`
- `results/stata/did_data_exchange_bs_data_resource_validation_v1_measurement.csv`

合并到当前 v3 面板后，2024 年共有 4,596 个公司样本，其中 84 个存在数据资源入表。旧 `BookEntry` 与资产负债表标准字段完全一致：

| `BookDataResource_bs` | `BookEntry_old` | 公司年 |
|---:|---:|---:|
| 0 | 0 | 4,512 |
| 1 | 1 | 84 |

关键验证结果：

| X | t | 正入表中 X 也正 |
|---|---:|---:|
| `asset_trade_noinst_pos` | 6.86 | 56 / 84 |
| `strict_noinst_pos` | 6.89 | 55 / 84 |
| `verif_noinst_pos` | 6.85 | 48 / 84 |
| `asset_trade_noinst_share` | 3.45 | 56 / 84 |
| `strict_noinst_share` | 3.22 | 55 / 84 |
| `verif_noinst_share` | 3.82 | 48 / 84 |
| `product_tx_noinst_kw` | 4.08 | 36 / 84 |

这说明当前可核验披露 Y 可以被资产负债表标准字段外部验证。

### X 侧

继续使用：

`城市数据交易所设立 × 事前数据叙事-硬能力错配`

但可以把 hard capability 更新为更硬版本：

- 原有数字发明专利授权量。
- 新增专利明细中的数据/AI/软件相关专利。
- 数字人力、数字资本、AI 投资等原有硬能力。

不建议把 2024 数据资源入表金额放进 2016-2017 的 `PreWash`，因为时间不对。

### Y 侧

主 Y 仍建议维持：

- `asset_trade_noinst_kw`
- `strict_noinst_kw`

同时新增外部验证：

- 主文本可核验披露是否预测 2024 年真实入表：
  - `BookDataResource`
  - `lnBookDataResource`
  - `BookDataResourceRatio`

这样可以把“可核验化披露”做实：不是只数关键词，而是这些关键词确实与资产负债表中的数据资源确认相关。

### 论文表述边界

可以说：

数据交易所设立后，事前存在数据叙事-硬能力错配的企业，更倾向于从泛化叙事转向资产化、交易化、会计可验证的数据要素披露；并且这种披露与 2024 年数据资源入表金额存在外部一致性。

不要说：

数据交易所直接提升了企业真实数据资产，或者专利本身就是数据知识产权。
