# 资产负债表数据资源入表字段验证

日期：2026-05-07

## 1. 目的

本轮把第三方财务资料中的资产负债表标准字段接入 v3，用来验证当前“可核验数据要素披露”Y 是否真的对应会计层面的数据资源入表。

这不是主 DID 的新 Y。原因是数据资源入表字段从 2024 年开始使用，不能形成 2018-2024 的长期结果变量。

## 2. 数据与脚本

原始资料：

`/Users/mac/computerscience/第三方资料/第三方数据资源/上市公司财务信息/FS_Combas.xlsx`

字段：

- `A001123101`：数据资源（存货）
- `A001218201`：数据资源（无形资产）
- `A001219101`：数据资源（开发支出）

抽取脚本：

`src/python/extract_balance_sheet_data_resource_for_stata.py`

Stata 验证脚本：

`src/stata/did_data_exchange_bs_data_resource_validation_v1.do`

输出：

- `results/data/balance_sheet_data_resource_annual.csv`
- `results/data/balance_sheet_data_resource_period.csv`
- `results/data/did_data_exchange_bs_data_resource_validation_v1_panel.dta`
- `results/stata/did_data_exchange_bs_data_resource_validation_v1_coverage.csv`
- `results/stata/did_data_exchange_bs_data_resource_validation_v1_old_new_crosstab_2024.csv`
- `results/stata/did_data_exchange_bs_data_resource_validation_v1_measurement.csv`
- `results/stata/did_data_exchange_bs_data_resource_validation_v1_did.csv`

## 3. 覆盖率

全量资产负债表年末合并报表中，2024 年 `12-31` 有 5,484 条，其中任一数据资源金额为正的有 91 条。

合并到当前 v3 DDD 面板后：

| 年份 | N | 入表正样本 | 正样本率 | 入表金额合计 |
|---|---:|---:|---:|---:|
| 2018 | 3,256 | 0 | 0.00% | 0 |
| 2019 | 3,310 | 0 | 0.00% | 0 |
| 2020 | 3,420 | 0 | 0.00% | 0 |
| 2021 | 3,909 | 0 | 0.00% | 0 |
| 2022 | 4,338 | 0 | 0.00% | 0 |
| 2023 | 4,662 | 0 | 0.00% | 0 |
| 2024 | 4,596 | 84 | 1.83% | 2,013,982,867.38 |

## 4. 新旧入表变量一致性

当前 v3 面板中，旧 `BookEntry` 与资产负债表标准字段 `BookDataResource_bs` 完全一致：

| `BookDataResource_bs` | `BookEntry_old` | 公司年 |
|---:|---:|---:|
| 0 | 0 | 4,512 |
| 1 | 1 | 84 |

这说明旧 `DataAsset / BookEntry` 不是纯文本误识别，至少在当前样本内与资产负债表标准字段完全对齐。

## 5. 文本 Y 对真实入表的验证

2024 年横截面规格：

```text
BookDataResource_i = TextMeasure_i + Controls_i + Industry FE + error_i
```

核心结果：

| X | coef | t | p | 正入表中 X 也正 |
|---|---:|---:|---:|---:|
| `asset_trade_noinst_kw` | 0.1497 | 1.82 | 0.069 | 56 / 84 |
| `asset_trade_noinst_pos` | 0.0901 | 6.86 | <0.001 | 56 / 84 |
| `asset_trade_noinst_share` | 3.2550 | 3.45 | <0.001 | 56 / 84 |
| `strict_noinst_kw` | 0.1574 | 1.73 | 0.084 | 55 / 84 |
| `strict_noinst_pos` | 0.0946 | 6.89 | <0.001 | 55 / 84 |
| `strict_noinst_share` | 3.4538 | 3.22 | 0.001 | 55 / 84 |
| `verif_noinst_kw` | 0.3693 | 2.31 | 0.021 | 48 / 84 |
| `verif_noinst_pos` | 0.1423 | 6.85 | <0.001 | 48 / 84 |
| `verif_noinst_share` | 5.2962 | 3.82 | <0.001 | 48 / 84 |
| `product_tx_noinst_kw` | 0.6505 | 4.08 | <0.001 | 36 / 84 |

金额型验证也成立，尤其是 `lnBookDataResource_bs`；但资产占比 `BookDataResourceRatio_bs` 明显更弱。原因大概率是入表金额极端稀疏，且金额大小受行业资产规模和会计确认口径影响更大。

## 6. 对主设计的含义

这组结果支持当前 Y 的命名：

```text
可核验数据要素披露 / verifiable data-economy disclosure
```

因为这些文本变量不只是政策词频，它们能预测资产负债表中的真实数据资源确认。

但这组结果不能支持：

```text
数据交易所导致企业真实数据资产入表增加
```

探索性 DDD 中，`BookDataResource_bs` 和 `lnBookDataResource_bs` 只在弱规格下对严格 exposure 有边际信号；加入 `City-Year FE + PreWash-Year FE` 后消失。因此入表金额应留在 external validation，而不是主结果。

## 7. 写作建议

主文可以写：

> 为进一步验证文本指标确实反映披露的可核验性，本文将其与资产负债表中 2024 年开始列示的“数据资源（存货）”“数据资源（无形资产）”和“数据资源（开发支出）”字段进行匹配。结果显示，资产化、交易化和严格可核验披露指标均显著预测企业是否确认数据资源入表。

不要把这部分写成机制或因果后果。它是 measurement validation。
