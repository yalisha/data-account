# 数据交易所 × 企业预先错配 DDD 试跑报告

日期：2026-05-06

## 1. 本轮目的

本轮把 `/16数据交易所` 的城市级数据交易所 DID 接到 v3 的企业级错配变量上，检验一个更贴企业层面的识别思路：

```text
数据交易所设立是否约束企业数据要素叙事-硬能力错配？
```

核心不再是：

```text
数据交易所设立 -> SYNCH
```

而是：

```text
数据交易所设立 × 企业预先 data washing 暴露
-> 后续 NarrHardGap / DataWashing 是否下降
```

## 2. 数据合并

使用数据：

```text
v3/results/data/hardcap_direct_v1_panel.dta
/Users/mac/computerscience/0做完了/16数据交易所/data/did_sample_llm.dta
/Users/mac/computerscience/0做完了/16数据交易所/data/asset_trade_keywords.csv
```

关键修正：

```text
合并必须用真实股票代码 Stkcd + year_num，不能用 Stkcd_num。
```

原因是 v3 的 `Stkcd_num` 是内部编码，不是真实股票代码。最初错误使用 `Stkcd_num` 只匹配到 6,892 行；已改为 `Stkcd + year_num`。

正确合并后，主静态样本：

```text
v1 2018-2020 暴露版本：N = 23,526
v2 2016-2017 暴露版本：N = 19,222
```

## 3. 新增脚本和输出

脚本：

```text
src/stata/did_data_exchange_mismatch_ddd_v1.do
src/stata/did_data_exchange_mismatch_ddd_v2_pre2017.do
```

核心输出：

```text
results/data/did_data_exchange_mismatch_ddd_v1_panel.dta
results/data/did_data_exchange_mismatch_ddd_v2_pre2017_panel.dta
results/stata/did_data_exchange_mismatch_ddd_v1_main.csv
results/stata/did_data_exchange_mismatch_ddd_v1_event.csv
results/stata/did_data_exchange_mismatch_ddd_v2_pre2017_main.csv
results/stata/did_data_exchange_mismatch_ddd_v2_pre2017_event.csv
```

Stata MCP 日志：

```text
results/logs/did_data_exchange_mismatch_ddd_v1_mcp.log
results/logs/did_data_exchange_mismatch_ddd_v2_pre2017_mcp.log
```

## 4. 设计

### 4.1 静态 DDD

模型：

```text
Y_ict =
    beta DID_city_ct × PreMismatch_i
    + theta DID_city_ct
    + controls
    + firm FE + year FE + error_ict
```

标准误：

```text
cluster(city_id)
```

### 4.2 暴露变量

v1 暴露：

```text
PreGap1820
PreWashAny1820
PreWashMaj1820
PreStrictWashAny1820
```

v1 的问题：

```text
2018-2020 暴露期与 2018-2024 事件研究窗口重叠，pre-event 系数有机械污染风险。
```

因此又跑了 v2。

v2 暴露：

```text
PreGap1617
PreWashAny1617
PreWashMaj1617
PreStrictWashAny1617
```

v2 优点：

```text
2016-2017 暴露完全早于 /16 数据交易所样本的 2018-2024 事件窗口。
```

## 5. v1 结果：2018-2020 暴露

静态结果很强：

| treatment | Y | coef | t |
|---|---|---:|---:|
| `dx_wany` | `NarrHardGap_direct_v1` | -0.1023 | -7.67 |
| `dx_wany` | `NarrHardWashing_direct_v1` | -0.1863 | -19.52 |
| `dx_wany` | `NarrHardVerified_direct_v1` | 0.0329 | 2.68 |
| `dx_wany` | `DU_kw` | 0.1535 | 3.38 |
| `dx_wany` | `asset_trade_kw` | 0.0429 | 3.51 |
| `dx_wany` | `strict_at_kw` | 0.0167 | 3.58 |
| `dx_wany` | `SYNCH` | -0.0549 | -2.30 |

但是事件研究中，`m3/m2` 有不少显著项。由于暴露变量本身用 2018-2020 构造，这里的 pre-trend 不能直接解释为平行趋势失败，但足以说明 v1 不能作为最终主规格。

## 6. v2 结果：2016-2017 暴露

v2 更干净，因为暴露期不与事件窗口重叠。

### 6.1 预先暴露分布

| exposure | N | mean | positive |
|---|---:|---:|---:|
| `PreGap1617` | 2,851 | 0.3441 | 2,328 |
| `PreWashAny1617` | 2,851 | 0.4735 | 1,350 |
| `PreStrictWashAny1617` | 2,851 | 0.2406 | 686 |
| `PreHardCap1617` | 2,851 | 0.0065 | 192 |
| `PreDU1617` | 2,851 | 0.9257 | 2,784 |

### 6.2 静态 DDD：错配 Y

| treatment | Y | coef | t | N |
|---|---|---:|---:|---:|
| `dx_gap17` | `NarrHardGap_direct_v1` | -0.1130 | -7.83 | 19,222 |
| `dx_gap17` | `NarrHardWashing_direct_v1` | -0.0923 | -4.81 | 19,222 |
| `dx_wany17` | `NarrHardGap_direct_v1` | -0.0868 | -5.62 | 19,222 |
| `dx_wany17` | `NarrHardWashing_direct_v1` | -0.1194 | -6.70 | 19,222 |
| `dx_wstr17` | `NarrHardGap_direct_v1` | -0.0560 | -4.36 | 19,222 |
| `dx_wstr17` | `NarrHardWashing_direct_v1` | -0.0894 | -7.63 | 19,222 |

判断：

```text
核心 first-stage 成立：数据交易所设立后，预先错配/预先 washing 企业的后续错配显著下降。
```

### 6.3 静态 DDD：可验证化/数据交易表述

| treatment | Y | coef | t |
|---|---|---:|---:|
| `dx_gap17` | `DU_kw` | 0.2217 | 2.62 |
| `dx_gap17` | `asset_trade_kw` | 0.0760 | 2.55 |
| `dx_gap17` | `strict_at_kw` | 0.0277 | 2.61 |
| `dx_wany17` | `DU_kw` | 0.1524 | 2.62 |
| `dx_wany17` | `asset_trade_kw` | 0.0438 | 2.68 |
| `dx_wany17` | `strict_at_kw` | 0.0164 | 2.75 |
| `dx_wstr17` | `DU_kw` | 0.3394 | 4.62 |
| `dx_wstr17` | `asset_trade_kw` | 0.0956 | 3.55 |
| `dx_wstr17` | `strict_at_kw` | 0.0323 | 3.53 |

解释：

```text
预先 washing 企业在数据交易所设立后，不只是少了错配，还更集中地使用数据利用、数据资产/交易相关表达。
```

这和 `/16数据交易所` 原来的机制证据一致，但企业层面更细。

### 6.4 静态 DDD：资本市场后果

资本市场结果不是主识别，但有一些信号：

| treatment | Y | coef | t |
|---|---|---:|---:|
| `dx_gap17` | `TobinQ` | 0.0802 | 2.11 |
| `dx_wstr17` | `TobinQ` | 0.1159 | 3.00 |
| `dx_gap17` | `Analyst` | -0.1536 | -2.53 |
| `dx_wany17` | `Analyst` | -0.1428 | -3.15 |
| `dx_wstr17` | `Analyst` | -0.2016 | -4.17 |

`SYNCH` 在 v2 静态里不显著；这说明 v3 这条企业错配线不能简单复刻 `/16数据交易所` 的 SYNCH 主结果。

## 7. 动态结果与风险

v2 动态结果的好消息：

1. `DU_kw / asset_trade_kw / strict_at_kw` 的 post 系数基本为正，并随时间增强。
2. `NarrHardGap / NarrHardWashing` 的 post 系数基本转负，且 t+1 到 t+3 较强。

例如 `wany17`：

| Y | p0 | p1 | p2 | p3 |
|---|---:|---:|---:|---:|
| `NarrHardGap_direct_v1` | -0.0202 | -0.0651 | -0.0929 | -0.0944 |
| `NarrHardWashing_direct_v1` | -0.0047 | -0.0723 | -0.0935 | -0.1290 |
| `DU_kw` | 0.0757 | 0.0941 | 0.1899 | 0.2335 |
| `asset_trade_kw` | 0.0249 | 0.0390 | 0.0477 | 0.0734 |

但还有风险：

```text
m3 仍然经常显著，说明处理城市中的高预先错配企业在更早期已经存在不同轨迹。
```

这不是致命否决，因为 m2 通常弱于 m3，post 方向也发生明显反转；但它意味着当前版本还不能直接写成“平行趋势完全干净”的主 DID。

## 8. 当前判断

这条路比 2024 会计规则年度 DID 和省级数据知识产权试点更值得推进。

原因：

1. 冲击来自 `/16数据交易所` 已经验证过的城市级 staggered DID。
2. 企业层处理强度是 v3 的预先 data washing / mismatch，不是单纯城市处理。
3. 静态 DDD 对 `NarrHardGap` 和 `NarrHardWashing` 结果很稳。
4. `DU_kw / asset_trade_kw / strict_at_kw` 提供了“错配下降 -> 可验证数据表达上升”的解释。

但现在还不是最终版。

核心缺口：

```text
事件研究 pre-period，尤其 m3，仍有显著差异。
```

因此当前定位应是：

```text
可推进的主 DID 候选，而不是已经过关的主 DID。
```

## 9. 下一步建议

下一步不要再换政策，应该继续打磨这条企业层 DDD：

1. 加城市线性趋势或城市组别趋势，看静态 DDD 是否保留。
2. 用 `not-yet-treated / never-treated` 样本做更干净的事件窗口。
3. 把暴露变量改成非 outcome 构造：例如 2016-2017 的硬能力不足、低 `HardCap`、低数据招聘/专利，而不是直接用 `NarrHardWashing`。
4. 跑 cohort-specific 版本，检查是否由 2021 早期城市驱动。
5. 加 leave-one-city-out，看是否由北京、上海、深圳等城市驱动。

如果这些能过，题目可以明确改成：

```text
数据交易所设立、企业数据叙事错配与披露可验证化
```

或者：

```text
Can Data Exchanges Discipline Data Washing?
Evidence from Annual Reports and City-Level Data Market Infrastructure
```

这比原来的“年报数据披露 -> TobinQ”强很多。
