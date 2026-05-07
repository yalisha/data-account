# 数据交易所 DDD 稳健性检查

日期：2026-05-06

## 1. 本轮目的

上一轮发现：

```text
数据交易所 × 2016-2017 预先 data washing 暴露
-> 后续 NarrHardGap / NarrHardWashing 下降
```

静态 DDD 很强，但事件研究的 pre-period 仍显著。本轮专门检查这个设计能不能扛住更强识别要求。

## 2. 新增脚本与输出

脚本：

```text
src/stata/did_data_exchange_mismatch_ddd_robust_v1.do
src/stata/did_data_exchange_mismatch_ddd_pretrend_slope_v1.do
```

主要输出：

```text
results/stata/did_data_exchange_mismatch_ddd_robust_v1_static.csv
results/stata/did_data_exchange_mismatch_ddd_robust_v1_stacked_static.csv
results/stata/did_data_exchange_mismatch_ddd_robust_v1_stacked_event.csv
results/stata/did_data_exchange_mismatch_ddd_robust_v1_stacked_pretrend_joint.csv
results/stata/did_data_exchange_mismatch_ddd_robust_v1_jackknife.csv
results/stata/did_data_exchange_mismatch_ddd_robust_v1_jackknife_summary.csv
results/stata/did_data_exchange_mismatch_ddd_pretrend_slope_v1_static.csv
results/stata/did_data_exchange_mismatch_ddd_pretrend_slope_v1_stacked_event.csv
results/stata/did_data_exchange_mismatch_ddd_pretrend_slope_v1_stacked_pretrend_joint.csv
```

Stata MCP 日志：

```text
results/logs/did_data_exchange_mismatch_ddd_robust_v1_mcp.log
results/logs/did_data_exchange_mismatch_ddd_pretrend_slope_v1_mcp.log
```

## 3. 静态 DDD：结果很稳

核心暴露用：

```text
dx_wany17 = DID_city × PreWashAny1617
dx_wstr17 = DID_city × PreStrictWashAny1617
```

### 3.1 `dx_wany17`

| 规格 | Y | coef | t |
|---|---|---:|---:|
| baseline | `NarrHardGap_direct_v1` | -0.0868 | -5.62 |
| city linear trend | `NarrHardGap_direct_v1` | -0.0863 | -5.13 |
| province-year FE | `NarrHardGap_direct_v1` | -0.0859 | -5.08 |
| industry-year FE | `NarrHardGap_direct_v1` | -0.1011 | -5.87 |
| drop 2024 cohort | `NarrHardGap_direct_v1` | -0.0882 | -5.63 |
| baseline | `NarrHardWashing_direct_v1` | -0.1194 | -6.70 |
| city linear trend | `NarrHardWashing_direct_v1` | -0.1176 | -6.41 |
| province-year FE | `NarrHardWashing_direct_v1` | -0.1173 | -6.52 |
| industry-year FE | `NarrHardWashing_direct_v1` | -0.1295 | -7.18 |
| drop 2024 cohort | `NarrHardWashing_direct_v1` | -0.1207 | -6.73 |

判断：

```text
静态错配下降不是固定效应选择造成的，也不是 2024 cohort 单独驱动。
```

但机制变量对 industry-year FE 敏感：

| Y | baseline coef/t | industry-year FE coef/t |
|---|---:|---:|
| `DU_kw` | 0.1524 / 2.62 | -0.0012 / -0.02 |
| `asset_trade_kw` | 0.0438 / 2.68 | 0.0135 / 0.91 |
| `strict_at_kw` | 0.0164 / 2.75 | 0.0070 / 1.37 |

解释：

```text
可验证表达上升有行业年度共同冲击成分；不能把它写得过强。
```

## 4. Stacked not-yet / never-treated 静态检验：仍成立

使用 stacked cohort 样本，只用 not-yet-treated 和 never-treated 作为控制组。

| exposure | Y | coef | t | N |
|---|---|---:|---:|---:|
| `sdx_wany17` | `NarrHardGap_direct_v1` | -0.0803 | -4.55 | 65,534 |
| `sdx_wany17` | `NarrHardWashing_direct_v1` | -0.1064 | -5.49 | 65,534 |
| `sdx_wany17` | `DU_kw` | 0.1646 | 2.72 | 65,534 |
| `sdx_wany17` | `asset_trade_kw` | 0.0445 | 2.67 | 65,530 |
| `sdx_wany17` | `strict_at_kw` | 0.0165 | 2.53 | 65,530 |
| `sdx_wstr17` | `TobinQ` | 0.0827 | 2.35 | 65,534 |

判断：

```text
静态结果不是 TWFE 使用 already-treated comparison 导致的。
```

## 5. 动态 pre-trend：错配 Y 仍不过关

Stacked event-study 的联合 pretrend 检验：

| exposure | Y | pretrend p |
|---|---|---:|
| `wany17` | `DU_kw` | 0.5607 |
| `wany17` | `asset_trade_kw` | 0.2248 |
| `wany17` | `NarrHardGap_direct_v1` | 0.0072 |
| `wany17` | `NarrHardWashing_direct_v1` | 0.000008 |
| `wany17` | `strict_at_kw` | 0.0348 |
| `wstr17` | `DU_kw` | 0.3919 |
| `wstr17` | `asset_trade_kw` | 0.0218 |
| `wstr17` | `NarrHardGap_direct_v1` | 0.0002 |
| `wstr17` | `NarrHardWashing_direct_v1` | 0.0003 |

关键区别：

```text
DU_kw 和 asset_trade_kw 在 wany17 口径下 pretrend 是干净的。
NarrHardGap / NarrHardWashing 的 pretrend 不干净。
```

这说明如果把主 Y 写成“可验证数据表达/数据交易表达增加”，识别更像样；如果把主 Y 写成“错配下降”，目前动态证据不够干净。

## 6. 处理组×预先暴露线性趋势

为了排除“预先 washing 企业本来就在下降”的解释，加入：

```text
TreatedEver × PreWashAny1617 × linear trend
TreatedEver × PreStrictWashAny1617 × linear trend
```

静态结果：

| exposure | Y | coef | t |
|---|---|---:|---:|
| `dx_wany17` | `NarrHardGap_direct_v1` | -0.0626 | -3.76 |
| `dx_wany17` | `NarrHardWashing_direct_v1` | -0.0572 | -2.24 |
| `dx_wany17` | `DU_kw` | -0.0169 | -0.22 |
| `dx_wany17` | `asset_trade_kw` | 0.0083 | 0.88 |
| `dx_wany17` | `strict_at_kw` | 0.0028 | 0.97 |
| `dx_wstr17` | `asset_trade_kw` | 0.0304 | 2.32 |

判断：

```text
错配下降在 wany17 口径下仍保留，但幅度明显变小。
机制变量在这个强规格下大多被吸收。
```

趋势校正后的 stacked event pretrend 仍显示：

| exposure | Y | pretrend p |
|---|---|---:|
| `wany17` | `DU_kw` | 0.5964 |
| `wany17` | `asset_trade_kw` | 0.3302 |
| `wany17` | `NarrHardGap_direct_v1` | 0.0109 |
| `wany17` | `NarrHardWashing_direct_v1` | 0.000012 |
| `wany17` | `strict_at_kw` | 0.0601 |

因此：

```text
线性趋势不能完全解决错配 Y 的动态 pretrend。
```

## 7. Leave-one-treated-city-out：不是单一城市驱动

对 `dx_wany17` 做 leave-one-treated-city-out。

| Y | mean coef | min coef | max coef | 5% significant share |
|---|---:|---:|---:|---:|
| `NarrHardGap_direct_v1` | -0.0868 | -0.0950 | -0.0739 | 22/22 |
| `NarrHardWashing_direct_v1` | -0.1195 | -0.1339 | -0.1094 | 22/22 |
| `DU_kw` | 0.1521 | 0.0909 | 0.1690 | 22/22 |
| `asset_trade_kw` | 0.0437 | 0.0299 | 0.0511 | 21/22 |
| `strict_at_kw` | 0.0163 | 0.0099 | 0.0189 | 22/22 |
| `TobinQ` | 0.0395 | 0.0086 | 0.0541 | 0/22 |

判断：

```text
结果不是北京、上海、深圳、广州等单个城市驱动。
TobinQ 不适合作主 Y。
```

## 8. 当前总判断

这条 DID 线可以推进，但主叙事要收窄。

不建议写成：

```text
数据交易所设立显著抑制企业 data washing / 错配
```

原因：

```text
错配 Y 的事件研究 pretrend 没过。
```

更建议写成：

```text
数据交易所设立促进预先 data-washing 暴露企业转向可验证的数据要素表达。
```

对应主 Y：

```text
DU_kw
asset_trade_kw
strict_at_kw
```

其中更稳的是：

```text
DU_kw 和 asset_trade_kw under PreWashAny1617
```

因为它们在 stacked event 的 pretrend 上是干净的。

错配变量可以保留，但定位应改成：

```text
辅助结果 / 收敛性证据 / 描述性 disciplining pattern
```

而不是唯一主因变量。

## 9. 建议的新题目

更干净的题目可以是：

```text
数据交易所设立与企业数据要素披露可验证化
```

英文可写：

```text
Data Exchanges and Verifiable Data-Economy Disclosure:
Evidence from Firms with Prior Data-Washing Exposure
```

或者更锋利一点：

```text
Do Data Exchanges Discipline Data Washing?
Evidence from Verifiable Data-Economy Disclosure
```

但第二个题目里，`Discipline Data Washing` 不能只靠 `NarrHardWashing` 当主 Y；应把 `verifiable disclosure` 作为主结果，把 `washing decline` 放在附属表。
