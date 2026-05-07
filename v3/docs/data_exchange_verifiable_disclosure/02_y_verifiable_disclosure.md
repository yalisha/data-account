# Y 构造：数据要素披露可核验化

更新日期：2026-05-07

## 1. Y 的经济含义

本文的主 Y 不是“企业是否更多谈数据”。更准确的 Y 是：

```text
企业是否从泛化数据叙事，转向更具资产化、交易化和可核验特征的数据要素披露。
```

这类披露更容易被外部证据核验，例如：

```text
数据资产
数据产品
数据确权
数据产权
数据定价
数据估值
数据挂牌
数据入表
数据资源会计处理
```

## 2. 为什么不用旧 DU_kw 当主 Y

`DU_kw` 的优点是覆盖广，缺点是太宽。它更像 L0 层面的泛数据叙事，不能直接证明“可核验化”。

因此当前层级是：

```text
L0: DU_kw
    泛数据叙事，对照变量

L1: asset_trade_noinst_kw
    剔除机构名词后的资产化/交易化披露，主 Y

L2: strict_noinst_kw
    更严格的数据资产、确权、定价、入表披露，主 Y

L3: verif_noinst_kw / verif_noinst_share
    更窄的可核验披露指数，补充与验证

L4: BookDataResource_bs / lnBookDataResource_bs
    资产负债表标准字段中的真实数据资源入表验证层
```

## 3. 机构名词屏蔽

为了避免机械相关，抽词前先屏蔽：

```text
数据交易所
数据交易中心
数据要素市场
大数据交易所
大数据交易中心
```

这一步很重要。否则企业所在城市设立数据交易所后，年报中提到“数据交易所”可能只是机构名词同步，而不是更可核验披露。

对应脚本：

```text
src/python/build_verifiable_disclosure_y_noinst_v1.py
```

## 4. 主 Y

### 4.1 asset_trade_noinst_kw

定义：

```text
asset_trade_noinst_kw
  = 屏蔽机构名词后，资产化/交易化关键词出现次数 / 年报全文长度 x 10000
```

关键词包括：

```text
数据资产入表
数据资源入表
数据资源会计处理
数据资产化
数据产权
数据确权
数据定价
数据估值
数据交易
数据流通
数据挂牌
数据产品
数据资产
数据入表
```

它相对旧 `asset_trade_kw` 的改进是：`数据交易` 不再从 `数据交易所` 里机械切出来。

### 4.2 strict_noinst_kw

定义：

```text
strict_noinst_kw
  = 屏蔽机构名词后，严格资产/确权/定价/入表关键词出现次数 / 年报全文长度 x 10000
```

关键词包括：

```text
数据资产入表
数据资源入表
数据资源会计处理
数据资产化
数据产权
数据确权
数据定价
数据产品
数据资产
数据入表
```

这个变量本质上接近旧 `strict_at_kw`，因为旧严格口径已经排除了机构名词。

## 5. 补充 Y 和组件

更窄的验证型 Y：

```text
verif_noinst_kw = acct_kw + rights_kw + pricing_kw + product_tx_noinst_kw
```

四个组件：

```text
acct_kw
  数据资源/数据资产入表、会计处理、确认、计量、列示、披露

rights_kw
  数据产权、确权、权属、登记、数据知识产权、授权运营

pricing_kw
  数据定价、估值、评估、价格、价值计量

product_tx_noinst_kw
  数据产品、挂牌、交易、流通等，且剔除机构名词
```

## 6. 覆盖率

当前 DDD 面板 27,491 个公司年：

| 变量 | 正值率 | 均值 |
|---|---:|---:|
| `DU_kw` | 98.95% | 1.586 |
| 旧 `asset_trade_kw` | 8.48% | 0.0237 |
| 旧 `strict_at_kw` | 6.49% | 0.0108 |
| `inst_term_kw` | 1.59% | 0.0020 |
| `asset_trade_noinst_kw` | 6.95% | 0.0121 |
| `strict_noinst_kw` | 6.49% | 0.0108 |
| `verif_noinst_kw` | 4.17% | 0.0058 |
| `acct_kw` | 0.33% | 0.0003 |
| `rights_kw` | 0.83% | 0.0008 |
| `pricing_kw` | 0.83% | 0.0005 |
| `product_tx_noinst_kw` | 3.42% | 0.0042 |

解释：`asset_trade_noinst_kw` 和 `strict_noinst_kw` 是主文最合适的平衡点；`verif_noinst_kw` 更干净，但较稀疏。

## 7. 当前主结果

`asset_trade_noinst_kw`：

| exposure | baseline | City-Year + PreWash-Year FE | Drop 2024 | Industry-Year FE |
|---|---:|---:|---:|---:|
| `dx_wany17` | 0.0175 / 2.77 | 0.0125 / 2.27 | 0.0173 / 2.85 | 0.0066 / 1.22 |
| `dx_wstr17` | 0.0361 / 3.64 | 0.0226 / 2.71 | 0.0351 / 4.22 | 0.0140 / 1.95 |

`strict_noinst_kw`：

| exposure | baseline | City-Year + PreWash-Year FE | Drop 2024 | Industry-Year FE |
|---|---:|---:|---:|---:|
| `dx_wany17` | 0.0164 / 2.75 | 0.0121 / 2.39 | 0.0169 / 2.84 | 0.0070 / 1.37 |
| `dx_wstr17` | 0.0323 / 3.53 | 0.0211 / 2.83 | 0.0326 / 4.13 | 0.0132 / 1.98 |

`verif_noinst_kw`：

| exposure | baseline | City-Year + PreWash-Year FE | Drop 2024 | Industry-Year FE |
|---|---:|---:|---:|---:|
| `dx_wany17` | 0.0085 / 1.89 | 0.0062 / 1.62 | 0.0076 / 1.83 | 0.0034 / 0.88 |
| `dx_wstr17` | 0.0180 / 2.83 | 0.0113 / 2.26 | 0.0157 / 3.04 | 0.0076 / 1.67 |

## 8. 真实数据资源入表验证

2024 年横截面中，真实数据资源入表变量改用资产负债表标准字段：

```text
BookDataResource_bs
  = 1{数据资源（存货） + 数据资源（无形资产） + 数据资源（开发支出） > 0}

lnBookDataResource_bs
  = ln(1 + 数据资源入表金额合计)
```

对应字段来自资产负债表：

```text
A001123101: 数据资源（存货）
A001218201: 数据资源（无形资产）
A001219101: 数据资源（开发支出）
```

合并到当前 v3 面板后，2024 年有 4,596 个公司样本，其中 84 个 `BookDataResource_bs=1`。旧 `BookEntry` 与资产负债表标准字段在当前样本内完全一致：4,512 个共同为 0，84 个共同为 1。

`BookDataResource_bs` 验证结果：

| X | coef | t | 正入表中 X 也正 |
|---|---:|---:|---:|
| `asset_trade_noinst_kw` | 0.1497 | 1.82 | 56 / 84 |
| `asset_trade_noinst_pos` | 0.0901 | 6.86 | 56 / 84 |
| `asset_trade_noinst_share` | 3.2550 | 3.45 | 56 / 84 |
| `strict_noinst_kw` | 0.1574 | 1.73 | 55 / 84 |
| `strict_noinst_pos` | 0.0946 | 6.89 | 55 / 84 |
| `strict_noinst_share` | 3.4538 | 3.22 | 55 / 84 |
| `verif_noinst_kw` | 0.3693 | 2.31 | 48 / 84 |
| `verif_noinst_pos` | 0.1423 | 6.85 | 48 / 84 |
| `verif_noinst_share` | 5.2962 | 3.82 | 48 / 84 |
| `product_tx_noinst_kw` | 0.6505 | 4.08 | 36 / 84 |

这说明新版 Y 确实更接近真实数据资源入表附近的披露，而不是纯政策词频。金额型 `lnBookDataResource_bs` 结果方向类似；资产占比 `BookDataResourceRatio_bs` 较弱，更适合作稳健性而不是主验证。

## 9. 论文表述

推荐中文：

```text
可核验数据要素披露
```

英文：

```text
verifiable data-economy disclosure
```

避免使用：

```text
可检验性披露
```

因为“可检验性”容易被理解为理论命题是否能被实证检验，不如“可核验/可验证”贴近披露可被外部证据交叉验证。
