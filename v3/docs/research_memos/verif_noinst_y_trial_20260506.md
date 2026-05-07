# 剔除机构名词后的可核验披露 Y 试跑

生成时间：2026-05-06  
抽取脚本：`src/python/build_verifiable_disclosure_y_noinst_v1.py`  
Stata 脚本：`src/stata/did_data_exchange_verif_noinst_y_v1.do`

核心输出：

- `results/data/verifiable_disclosure_y_noinst_v1.csv`
- `results/data/did_data_exchange_verif_noinst_y_v1_panel.dta`
- `results/stata/did_data_exchange_verif_noinst_y_v1_summary.csv`
- `results/stata/did_data_exchange_verif_noinst_y_v1_static.csv`
- `results/stata/did_data_exchange_verif_noinst_y_v1_event.csv`
- `results/stata/did_data_exchange_verif_noinst_y_v1_pretrend_joint.csv`
- `results/stata/did_data_exchange_verif_noinst_y_v1_dataasset_validation.csv`

## 1. 这版 Y 怎么构造

先把明显会和处理变量机械相关的机构名词从文本中屏蔽：

```text
数据交易所
数据交易中心
数据要素市场
大数据交易所
大数据交易中心
```

然后在屏蔽后的文本中计数，按年报全文长度标准化为每万字频率。

主要变量：

```text
asset_trade_noinst_kw
```

含数据资产入表、数据资源入表、数据资源会计处理、数据资产化、数据产权、数据确权、数据定价、数据估值、数据交易、数据流通、数据挂牌、数据产品、数据资产、数据入表。关键区别是：`数据交易` 不再从 `数据交易所` 里机械切出来。

```text
strict_noinst_kw
```

对应旧 `strict_at_kw` 的严格口径。由于旧严格口径本身已经删除了机构名词，所以这版结果几乎等同于 `strict_at_kw`。

```text
verif_noinst_kw
```

四个更窄组件之和：

```text
acct_kw: 数据资源/数据资产入表、会计处理、确认、计量、列示、披露
rights_kw: 数据产权、确权、权属、登记、数据知识产权、授权运营
pricing_kw: 数据定价、估值、评估、价格、价值计量
product_tx_noinst_kw: 数据产品、挂牌、交易、流通等，且剔除机构名词污染
```

## 2. 覆盖率

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

解释：新版 Y 明显更窄。`verif_noinst_kw` 很干净，但会牺牲统计功效。

## 3. DDD 主结果

规格沿用：

```text
Y_ict = DID_city_ct × PreWash_i + DID_city_ct + controls
        + Firm FE + Year FE
```

并补充：

```text
City-Year FE + PreWash-Year FE
Drop 2024
PreWash × 2024
Industry × 2024 FE
Industry-Year FE
```

### 3.1 `asset_trade_noinst_kw`

| exposure | baseline | City-Year + PreWash-Year FE | Drop 2024 | Industry-Year FE |
|---|---:|---:|---:|---:|
| `dx_wany17` | 0.0175 / 2.77 | 0.0125 / 2.27 | 0.0173 / 2.85 | 0.0066 / 1.22 |
| `dx_wstr17` | 0.0361 / 3.64 | 0.0226 / 2.71 | 0.0351 / 4.22 | 0.0140 / 1.95 |

这个变量比旧 `asset_trade_kw` 更干净，而且强 FE 下仍能站住。

### 3.2 `strict_noinst_kw`

| exposure | baseline | City-Year + PreWash-Year FE | Drop 2024 | Industry-Year FE |
|---|---:|---:|---:|---:|
| `dx_wany17` | 0.0164 / 2.75 | 0.0121 / 2.39 | 0.0169 / 2.84 | 0.0070 / 1.37 |
| `dx_wstr17` | 0.0323 / 3.53 | 0.0211 / 2.83 | 0.0326 / 4.13 | 0.0132 / 1.98 |

严格口径很稳。缺点是它并不是全新的变量，本质上就是原来的 `strict_at_kw`。

### 3.3 `verif_noinst_kw`

| exposure | baseline | City-Year + PreWash-Year FE | Drop 2024 | Industry-Year FE |
|---|---:|---:|---:|---:|
| `dx_wany17` | 0.0085 / 1.89 | 0.0062 / 1.62 | 0.0076 / 1.83 | 0.0034 / 0.88 |
| `dx_wstr17` | 0.0180 / 2.83 | 0.0113 / 2.26 | 0.0157 / 3.04 | 0.0076 / 1.67 |

这个变量适合给严格事前 washing 企业用。对宽口径 `PreWashAny` 功效不够。

## 4. 动态与 pretrend

联合 pretrend p 值：

| exposure | Y | pretrend p |
|---|---|---:|
| `wany17` | `asset_trade_noinst_kw` | 0.396 |
| `wany17` | `strict_noinst_kw` | 0.548 |
| `wany17` | `verif_noinst_kw` | 0.785 |
| `wany17` | `asset_trade_noinst_share` | 0.680 |
| `wany17` | `strict_noinst_share` | 0.510 |
| `wany17` | `verif_noinst_share` | 0.586 |
| `wstr17` | `asset_trade_noinst_kw` | 0.021 |
| `wstr17` | `strict_noinst_kw` | 0.043 |
| `wstr17` | `verif_noinst_kw` | 0.609 |
| `wstr17` | `asset_trade_noinst_share` | 0.464 |
| `wstr17` | `strict_noinst_share` | 0.693 |
| `wstr17` | `verif_noinst_share` | 0.801 |
| `wstr17` | `product_tx_noinst_kw` | 0.340 |

解释：

1. 宽口径 exposure `wany17` 下，新版主 Y 的 pretrend 都干净。
2. 严格 exposure `wstr17` 下，`asset_trade_noinst_kw` 和 `strict_noinst_kw` 的 pretrend 边际失败，主要来自 `m3` 负向差异。
3. `wstr17 × verif_noinst_kw` 的动态更干净，且静态显著。

因此，主文不能简单说所有严格 exposure 的动态都好。更稳的是：

```text
主 Y: asset_trade_noinst_kw / strict_noinst_kw under PreWashAny1617
严格 exposure 补充: verif_noinst_kw / product_tx_noinst_kw under PreStrictWashAny1617
```

## 5. 真实数据资产入表验证

2024 年横截面：

```text
BookEntry_i = keyword_i + controls + industry FE
```

主要结果：

| X | t | BookEntry 正样本中 X 也正的数量 |
|---|---:|---:|
| `DU_kw` | 5.78 | 84 / 84 |
| 旧 `asset_trade_kw` | 2.66 | 61 / 84 |
| 旧 `strict_at_kw` | 1.73 | 55 / 84 |
| `inst_term_kw` | 2.18 | 29 / 84 |
| `asset_trade_noinst_kw` | 1.82 | 56 / 84 |
| `asset_trade_noinst_pos` | 6.86 | 56 / 84 |
| `asset_trade_noinst_share` | 3.45 | 56 / 84 |
| `strict_noinst_pos` | 6.89 | 55 / 84 |
| `strict_noinst_share` | 3.22 | 55 / 84 |
| `verif_noinst_kw` | 2.31 | 48 / 84 |
| `verif_noinst_pos` | 6.85 | 48 / 84 |
| `verif_noinst_share` | 3.82 | 48 / 84 |
| `product_tx_noinst_kw` | 4.08 | 36 / 84 |

这说明新版 Y 的确抓到了真实数据资产入表附近的披露，不只是政策名词。

但 `inst_term_kw` 也能预测入表，说明“提到数据交易所/交易中心”不完全是噪音。不过主 Y 剔除机构名后仍成立，可以回应机械相关担忧。

## 6. 当前判断

这版比旧版更适合写主文。

最稳组合：

```text
主 Y 1: asset_trade_noinst_kw
主 Y 2: strict_noinst_kw
验证 Y: verif_noinst_kw / verif_noinst_share / BookEntry
组件: acct_kw, pricing_kw, product_tx_noinst_kw
不主推: rights_kw, inst_term_kw, verif_noinst_share 的 DDD 主结果
```

一句话结论：

> 数据交易所设立后，事前 data-washing 暴露企业并不只是更多提到“数据交易所”这个机构名词，而是在剔除机构名词后，仍显著增加数据资产、数据产品、确权、定价、入表等更可核验的数据要素披露。

保守边界：

1. `industry-year FE` 下大部分主变量衰减，说明行业年度共同冲击仍然重要。
2. `verif_noinst_kw` 很干净但稀疏，宽口径 exposure 下功效不足。
3. 真实数据资产入表只能做 2024 验证层，不能做长期主 Y。
