# 机制、识别检验与验证计划

更新日期：2026-05-06

## 1. 理论机制

本文机制不是“企业真实数据能力立刻提升”。更稳的机制是披露结构调整。

### 1.1 可核验披露成本下降

数据交易所提供：

```text
数据登记
确权
估值
合规审核
挂牌
交易撮合
```

这些制度功能降低了企业披露数据资产、数据产品、数据交易和数据入表信息的组织成本与验证成本。

可检验含义：

```text
DID_city x PreWash -> asset_trade_noinst_kw / strict_noinst_kw 上升
```

### 1.2 空泛叙事相对成本上升

数据交易所出现后，数据要素不再只是抽象政策概念，而有了更明确的外部参照：能否登记、能否确权、能否估值、能否挂牌、能否交易、能否入表。

对事前 data-washing 暴露企业而言，继续泛泛谈数据要素但不提供更具体证据，信息不一致成本更高。

可检验含义：

```text
处理效应主要集中在 PreWashAny1617 / PreStrictWashAny1617 企业
```

### 1.3 披露从泛化叙事转向制度化表达

本文不要求企业短期内真的增加数据资产数量。主张是：

```text
数据交易所改变企业披露结构，
从 DU_kw 这类泛叙事转向资产化、交易化、可核验表达。
```

## 2. 机制组件

可核验披露可以拆成四个组件：

```text
acct_kw
rights_kw
pricing_kw
product_tx_noinst_kw
```

当前结果：

| Y | exposure | baseline | City-Year + PreWash-Year FE | Drop 2024 | Industry-Year FE |
|---|---|---:|---:|---:|---:|
| `acct_kw` | `dx_wany17` | 0.0015 / 2.15 | 0.0008 / 1.69 | 0.0017 / 2.71 | 0.0009 / 1.84 |
| `acct_kw` | `dx_wstr17` | 0.0026 / 2.42 | 0.0014 / 1.95 | 0.0030 / 3.01 | 0.0015 / 2.23 |
| `pricing_kw` | `dx_wany17` | 0.0013 / 1.96 | 0.0005 / 1.14 | 0.0014 / 1.98 | 0.0005 / 1.20 |
| `pricing_kw` | `dx_wstr17` | 0.0025 / 2.52 | 0.0010 / 1.49 | 0.0026 / 2.72 | 0.0010 / 2.16 |
| `product_tx_noinst_kw` | `dx_wany17` | 0.0042 / 1.68 | 0.0041 / 1.70 | 0.0034 / 1.50 | 0.0016 / 0.70 |
| `product_tx_noinst_kw` | `dx_wstr17` | 0.0093 / 2.61 | 0.0077 / 2.28 | 0.0073 / 2.54 | 0.0040 / 1.33 |

当前最像机制的组件是：

```text
acct_kw
pricing_kw
product_tx_noinst_kw
```

`rights_kw` 目前不稳，不主推。

## 3. 动态检验

主动态图建议只画 pretrend 过关的组合。

宽 exposure：

| Y | exposure | pretrend p |
|---|---|---:|
| `asset_trade_noinst_kw` | `wany17` | 0.396 |
| `strict_noinst_kw` | `wany17` | 0.548 |
| `verif_noinst_kw` | `wany17` | 0.785 |
| `asset_trade_noinst_share` | `wany17` | 0.680 |
| `strict_noinst_share` | `wany17` | 0.510 |
| `verif_noinst_share` | `wany17` | 0.586 |

严格 exposure：

| Y | exposure | pretrend p |
|---|---|---:|
| `verif_noinst_kw` | `wstr17` | 0.609 |
| `verif_noinst_share` | `wstr17` | 0.801 |
| `product_tx_noinst_kw` | `wstr17` | 0.340 |

不建议主画：

```text
wstr17 x asset_trade_noinst_kw
wstr17 x strict_noinst_kw
```

因为联合 pretrend p 值分别为 0.021 和 0.043。

## 4. 真实数据资源入表验证

真实数据资源入表不能当长期主 Y，但可以验证文本 Y。

当前验证：

```text
BookDataResource_i,2024 = keyword_i,2024 + controls + industry FE
```

`BookDataResource` 由资产负债表标准字段构造：

```text
A001123101 数据资源（存货）
A001218201 数据资源（无形资产）
A001219101 数据资源（开发支出）
```

在当前 v3 面板中，2024 年有 4,596 个公司样本，其中 84 个确认数据资源入表。旧 `BookEntry` 与标准字段完全一致。

关键结果：

| X | t | 解释 |
|---|---:|---|
| `asset_trade_noinst_pos` | 6.86 | 有资产化/交易化可核验表达的企业更可能真实入表 |
| `strict_noinst_pos` | 6.89 | 严格口径同样有效 |
| `verif_noinst_pos` | 6.85 | 更窄可核验指数同样有效 |
| `verif_noinst_share` | 3.82 | 在泛数据叙事中的可核验占比也有效 |
| `product_tx_noinst_kw` | 4.08 | 数据产品/挂牌/交易表达与入表相关 |

这部分应写成 measurement validation，而不是 causal result。

金额型 `lnBookDataResource` 的方向与入表虚拟变量一致；资产占比 `BookDataResourceRatio` 较弱。探索性 DDD 只在弱规格下有边际信号，加入 `City-Year FE + PreWash-Year FE` 后消失，因此不能把入表金额写成主因果结果。

## 5. 错配收敛证据

可以保留：

```text
NarrHardGap_direct_v1
NarrHardWashing_direct_v1
```

作用：

```text
说明可核验披露上升伴随叙事-硬能力错配下降。
```

但不能放主假设中心，因为动态 pretrend 不干净。建议放在 supporting evidence 或附录表。

## 6. 必做稳健性

主文或附录至少保留：

```text
1. Drop 2024
2. PreWash x 2024
3. Industry x 2024 FE
4. City-Year FE + PreWash-Year FE
5. Industry-Year FE 压力测试
6. Stacked not-yet/never controls
7. Leave-one-city-out
8. 剔除热门数字行业
9. 剔除金融/互联网/软件/通信等行业
10. 旧 asset_trade_kw 与新版 noinst Y 对照
```

## 7. 当前边界

必须明确：

1. 本文不证明数据交易所短期内提高真实数据能力。
2. 本文不证明数据交易所完全治理 data washing。
3. 本文证明的是披露结构变化：更偏向资产化、交易化、可核验表达。
4. 行业年度冲击很强，`industry-year FE` 下效应明显衰减，因此主文需要解释行业扩散和行业政策模板可能是机制的一部分，而不是简单忽略。

## 8. 后续可以增强的验证

优先做：

```text
句子级抽样标注
```

把命中 `asset_trade_noinst_kw` / `strict_noinst_kw` / `verif_noinst_kw` 的句子抽出来，标注：

```text
0 = 政策口号/机构名词
1 = 泛数据叙事
2 = 具体数据利用
3 = 数据资产/入表/会计处理
4 = 确权/登记/权属
5 = 定价/估值/评估
6 = 数据产品/挂牌/交易/流通
```

然后报告：

```text
新版 noinst Y 中 3-6 类占比显著高于 DU_kw 命中句子。
```

这会把“可核验化”从词频进一步做实。
