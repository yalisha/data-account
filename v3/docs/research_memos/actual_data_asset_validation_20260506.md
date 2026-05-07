# 实际数据资产入表验证：可验证披露变量能否坐实

生成时间：2026-05-06  
对应脚本：`src/stata/did_data_exchange_actual_data_asset_validation_v1.do`  
核心输出：

- `results/data/did_data_exchange_actual_data_asset_validation_v1_panel.dta`
- `results/stata/did_data_exchange_actual_data_asset_validation_v1_coverage.csv`
- `results/stata/did_data_exchange_actual_data_asset_validation_v1_crosstab_2024.csv`
- `results/stata/did_data_exchange_actual_data_asset_validation_v1_measurement.csv`
- `results/stata/did_data_exchange_actual_data_asset_validation_v1_did.csv`

## 1. 当前能用的“实际数据资产”是什么

本轮使用旧 panel 中的 `DataAsset` 字段，含义是企业实际披露/入表的数据资源或数据资产金额。构造为：

```text
DataAsset0 = DataAsset，缺失置 0
DataAsset_ln = ln(1 + DataAsset0)
BookEntry = 1{DataAsset0 > 0}
```

刷新抽取后，原始全库为 48,217 个公司年，正值 89 个，全部集中在 2024 年。并入当前数据交易所 DDD 面板后，2018-2023 年没有正值，2024 年有 84 个正值。

因此，这个变量不适合用来构造 2016-2017 年事前 `PreWash`，因为事前窗口内没有可用正值。它适合做“会计验证层”或“外部验证 Y”。

## 2. 它能不能支持 `asset_trade_kw / strict_at_kw`

可以。2024 年交叉验证很强。

2024 年 DDD 面板中：

```text
BookEntry = 1 的企业数：84
asset_trade_pos = 1 的企业数：663
strict_at_pos = 1 的企业数：493
```

交叉分布：

| BookEntry | asset_trade_pos | strict_at_pos | firm_years |
|---:|---:|---:|---:|
| 0 | 0 | 0 | 3910 |
| 1 | 0 | 0 | 23 |
| 0 | 1 | 0 | 164 |
| 1 | 1 | 0 | 6 |
| 0 | 1 | 1 | 438 |
| 1 | 1 | 1 | 55 |

由此可见：

```text
P(BookEntry=1 | asset_trade_pos=1) = 61 / 663 = 9.20%
P(BookEntry=1 | asset_trade_pos=0) = 23 / 3933 = 0.58%

P(BookEntry=1 | strict_at_pos=1) = 55 / 493 = 11.16%
P(BookEntry=1 | strict_at_pos=0) = 29 / 4103 = 0.71%
```

这说明 `asset_trade_kw` 和 `strict_at_kw` 不是普通数据叙事，它们确实更接近真实数据资产入表。

## 3. 2024 横截面回归验证

规格：

```text
BookEntry_i,2024 = keyword_i,2024 + controls + industry FE + robust SE
DataAsset_ln_i,2024 = keyword_i,2024 + controls + industry FE + robust SE
```

主要结果：

| Y | X | coef | t |
|---|---|---:|---:|
| BookEntry | DU_kw | 0.0110 | 5.78 |
| BookEntry | asset_trade_kw | 0.0774 | 2.66 |
| BookEntry | asset_trade_pos | 0.0736 | 6.96 |
| BookEntry | asset_trade_ratio | 1.1137 | 3.23 |
| BookEntry | asset_trade_share | 1.7409 | 4.86 |
| BookEntry | strict_at_kw | 0.1574 | 1.73 |
| BookEntry | strict_at_pos | 0.0946 | 6.89 |
| BookEntry | strict_at_ratio | 2.3376 | 2.26 |
| BookEntry | strict_at_share | 3.4538 | 3.22 |

金额型 `DataAsset_ln` 的结果弱很多，只有 `asset_trade_share` 在 5% 水平显著，其他多为 10% 或不显著。原因大概率是入表金额极小、分布极端稀疏。因此目前更应使用 `BookEntry` 作为验证，而不是主推金额。

## 4. 把实际数据资产直接当 DID 的 Y 是否可行

不适合作主线。

探索性 DDD 中，`BookEntry` 和 `DataAsset_ln` 只在最弱的 firm FE + year FE 规格下对 `dx_wstr17` 有边际信号：

| Y | exposure | spec | coef | t |
|---|---|---|---:|---:|
| BookEntry | dx_wstr17 | baseline firm/year FE | 0.0059 | 1.67 |
| DataAsset_ln | dx_wstr17 | baseline firm/year FE | 0.0000136 | 2.36 |

但一旦加入 `PreWash × 2024` 或 `City-Year FE + PreWash-Year FE`，结果基本消失。这个结论很重要：实际数据资产入表是全国 2024 会计规则下的稀疏实现，不应包装成数据交易所 DID 的主结果。

## 5. 现在最稳的写法

实际数据资产应该放在验证层：

```text
L0: DU_kw
L1: asset_trade_kw / asset_trade_ratio / asset_trade_share
L2: strict_at_kw / strict_at_ratio / strict_at_share
L3: BookEntry / DataAsset_ln
```

论文主张应写为：

> 数据交易所设立促使事前 data-washing 暴露企业的年报披露从泛数据叙事转向更制度化、可核验的数据资产/交易披露；2024 年真实数据资源入表结果进一步验证这些文本指标确实更接近可核验披露。

不能写成：

> 数据交易所显著促进企业真实数据资产入表。

也不能把 `DataAsset` 并入 2016-2017 事前硬能力，因为数据现实不支持。
