# 研究总计划：数据交易所设立与企业数据要素披露可核验化

更新日期：2026-05-07

## 1. 当前判断

这个选题可以继续推进，但主张要收紧。

最稳的论文定位不是：

```text
数据交易所治理了 data washing
```

也不是：

```text
数据交易所促进企业真实数据资产入表
```

而是：

```text
数据交易所作为数据要素市场基础设施，是否促使事前 data-washing 暴露企业
从泛化数据叙事转向更具资产化、交易化和可核验特征的数据要素披露。
```

## 2. 研究问题

中国各地数据交易所在 2021-2024 年间分批设立，提供数据登记、确权、估值、合规审核、挂牌和交易撮合等制度功能。本文关心的是：本地数据交易所出现后，原先高喊数据要素、但缺少硬能力支撑的企业，是否会调整年报披露结构。

核心问题：

```text
Local data exchanges -> more verifiable data-economy disclosure
especially among firms with prior data-washing exposure.
```

## 3. 贡献边界

已有文献已经研究了：

1. 数据资产披露的经济后果；
2. hard disclosure 与 soft disclosure；
3. 文本披露与实际账面数据资产之间的 gap；
4. 2024 年数据资源会计规则后的披露不一致；
5. 基于财政部条款的 VDI / clause-level verifiability index。

所以本文不能写“首次研究数据要素披露可验证性”。更稳的贡献是：

```text
不同于既有文献主要考察数据资产披露的经济后果、软硬披露差异
或披露不一致的市场反应，本文关注数据交易所这一市场基础设施
是否改变事前 data-washing 暴露企业的披露结构，使其从泛化叙事
转向资产化、交易化和可核验披露。
```

## 4. 识别设计

主识别是 DDD：

```text
Y_ict =
    beta DID_city_ct x PreWash_i
    + DID_city_ct
    + Controls_it
    + Firm FE
    + Year FE
    + error_ict
```

其中：

```text
DID_city_ct = 企业注册城市 c 在年份 t 是否已设立数据交易所
PreWash_i = 企业 2016-2017 年是否存在事前 data-washing 暴露
Y_ict = 后续可核验数据要素披露
```

更强规格：

```text
Y_ict =
    beta DID_city_ct x PreWash_i
    + Firm FE
    + City-Year FE
    + PreWash-Year FE
    + Controls_it
    + error_ict
```

这个规格吸收城市年份层面的共同冲击，也吸收事前 washing 企业全国层面的年度共同变化。

## 5. 当前变量层级

### X

```text
PreWashAny1617
PreStrictWashAny1617
PreGap1617
```

详见 `01_x_pre_data_washing_exposure.md`。

### Y

主文建议使用：

```text
asset_trade_noinst_kw
strict_noinst_kw
```

补充与验证：

```text
verif_noinst_kw
verif_noinst_share
product_tx_noinst_kw
acct_kw
pricing_kw
BookEntry
DataAsset_ln
BookDataResource_bs
lnBookDataResource_bs
```

详见 `02_y_verifiable_disclosure.md`。

### 支持证据

```text
NarrHardGap_direct_v1
NarrHardWashing_direct_v1
```

它们只作为错配收敛证据，不作为唯一主 Y。

## 6. 当前结果层级

### 主 Y 可用

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

### 动态可用，但要分层

宽口径 `PreWashAny1617` 下：

| Y | pretrend p |
|---|---:|
| `asset_trade_noinst_kw` | 0.396 |
| `strict_noinst_kw` | 0.548 |
| `verif_noinst_kw` | 0.785 |

严格 `PreStrictWashAny1617` 下：

| Y | pretrend p |
|---|---:|
| `verif_noinst_kw` | 0.609 |
| `verif_noinst_share` | 0.801 |
| `product_tx_noinst_kw` | 0.340 |

严格 exposure 下 `asset_trade_noinst_kw` 和 `strict_noinst_kw` 的 pretrend 边际失败，不适合作主动态图。

## 7. 论文表格计划

建议主文表格：

| 表 | 内容 |
|---|---|
| Table 1 | 变量定义、样本分布、处理城市与处理年份 |
| Table 2 | 主 DDD：`asset_trade_noinst_kw`、`strict_noinst_kw` |
| Table 3 | 动态 DDD：宽 exposure 下主 Y；严格 exposure 下 `verif_noinst_kw` |
| Table 4 | 强识别规格：City-Year FE + PreWash-Year FE、drop 2024、industry-2024 FE |
| Table 5 | 组件检验：`acct_kw`、`pricing_kw`、`product_tx_noinst_kw` |
| Table 6 | 真实数据资源入表验证：资产负债表标准字段 `BookDataResource_bs` / `lnBookDataResource_bs` |
| Table 7 | 支持证据：错配收敛，但标明 pretrend 限制 |

附录：

```text
旧 asset_trade_kw 对照
DU_kw 对照
leave-one-city-out
stacked not-yet/never
行业年度 FE 压力测试
2024 会计规则处理
```

## 8. 下一步计划

优先级：

1. 抽取一批命中句子，人工/LLM 标注其是否为真正可核验披露，做 measurement validity。
2. 将新版 Y 替换进主研究设计文档和表格脚本。
3. 画主动态图，只画 pretrend 过关的组合。
4. 做 `asset_trade_noinst_kw` 与 `strict_noinst_kw` 的行业剔除检验。
5. 2024 年后尝试构造财政部 clause-level VDI，但只作为补充，不抢主线。

## 9. 当前风险

1. `industry-year FE` 会吃掉大量效应，说明行业年度共同冲击仍重要。
2. `verif_noinst_kw` 最干净但稀疏，不能单独承载全文。
3. 数据资源入表正值集中在 2024 年，只能验证文本指标，不能当长期主 Y；当前样本内资产负债表标准字段与旧 `BookEntry` 完全一致。
4. 需要避免把“可核验化”写成“真实能力提升”。
