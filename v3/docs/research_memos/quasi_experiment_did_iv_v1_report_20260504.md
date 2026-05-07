# DID / IV 轻量试跑报告

日期：2026-05-04

## 1. 本轮问题

当前主结果是 firm-year 面板相关性：

```text
Y_t = L.NarrHardMismatch + controls + firm FE + year FE
```

问题是没有 DID 或 IV，识别看起来原始。因此本轮不碰 20G 专利库，先试两个可直接用现有面板跑的准实验方向：

1. `2024 数据资源会计处理规则 × 预先 data washing`
2. `数据知识产权试点地区 × 预先 data washing`

本轮所有回归使用 Stata MCP。

## 2. 政策口径

### 2.1 2024 数据资源会计处理规则

财政部《企业数据资源相关会计处理暂行规定》自 2024 年 1 月 1 日起施行，官方表述是规范企业数据资源相关会计处理、强化相关会计信息披露。

本轮构造：

```text
PreGap2123      = mean(NarrHardGap_direct_v1), 2021-2023
PreWashAny2123  = any(NarrHardWashing_direct_v1), 2021-2023
PreWashMaj2123  = mean(washing) >= 0.5, 2021-2023
PreStrictWashAny2123 = any(NHWash_direct_v1_s), 2021-2023
Post2024        = 1[year >= 2024]
```

主交互项：

```text
acct_gap_post  = PreGap2123 x Post2024
acct_wany_post = PreWashAny2123 x Post2024
acct_wmaj_post = PreWashMaj2123 x Post2024
acct_wstr_post = PreStrictWashAny2123 x Post2024
```

样本：

```text
2021-2024
```

动态检验：

```text
2023 为基准年，估计 2021、2022、2024 的 exposure × year
```

### 2.2 数据知识产权试点

国家知识产权局 2022 年确定北京市、上海市、江苏省、浙江省、福建省、山东省、广东省、深圳市等 8 个首批试点地方，试点期限为 2022 年 11 月至 2023 年 12 月。

2024 年试点在原 8 个地方基础上新增天津市、河北省、山西省、安徽省、河南省、湖北省、湖南省、贵州省、陕西省等 9 个地方，试点期限为 2023 年 12 月至 2024 年 11 月。

现有面板只有省份字段，没有城市字段，因此深圳无法单独识别，只能把广东省整体作为首批试点地区。

本轮构造：

```text
FirstDIP = 北京、上海、江苏、浙江、福建、山东、广东
NewDIP2024 = 天津、河北、山西、安徽、河南、湖北、湖南、贵州、陕西
DIPilot = FirstDIP x 1[year >= 2023] + NewDIP2024 x 1[year >= 2024]
```

预先暴露：

```text
PreGap2021      = mean(NarrHardGap_direct_v1), 2020-2021
PreWashAny2021  = any(NarrHardWashing_direct_v1), 2020-2021
PreStrictWashAny2021 = any(NHWash_direct_v1_s), 2020-2021
```

样本：

```text
2019-2024
```

动态检验：

```text
首批试点地区，2022 为基准年，估计 2019、2020、2021、2023、2024 的 FirstDIP × exposure × year
```

## 3. 新增脚本和输出

脚本：

```text
src/stata/did_accounting_2024_v1.do
src/stata/did_data_ip_pilot_v1.do
src/stata/did_policy_first_stage_v1.do
src/stata/did_policy_first_stage_event_v1.do
```

核心输出：

```text
results/data/did_accounting_2024_v1_panel.dta
results/data/did_data_ip_pilot_v1_panel.dta
results/stata/did_accounting_2024_v1_main.csv
results/stata/did_accounting_2024_v1_event.csv
results/stata/did_data_ip_pilot_v1_main.csv
results/stata/did_data_ip_pilot_v1_event.csv
results/stata/did_policy_first_stage_v1.csv
results/stata/did_policy_first_stage_event_v1.csv
```

日志：

```text
results/logs/did_accounting_2024_v1_mcp.log
results/logs/did_data_ip_pilot_v1_mcp.log
results/logs/did_policy_first_stage_v1_mcp.log
results/logs/did_policy_first_stage_event_v1_mcp.log
```

## 4. 2024 会计规则 DID：对 Y 的 reduced form

设定：

```text
Y_it = exposure_i x Post2024_t + controls + firm FE + year FE
cluster(IndYear_num)
```

主结果并不支持“2024 后 washing 企业被估值折价”：

| treatment | Y | coef | t | 判断 |
|---|---|---:|---:|---|
| `acct_gap_post` | `TobinQ` | 0.0430 | 1.42 | 不显著 |
| `acct_wany_post` | `TobinQ` | 0.0209 | 0.81 | 不显著 |
| `acct_wmaj_post` | `TobinQ` | 0.0355 | 1.11 | 不显著 |
| `acct_wstr_post` | `TobinQ` | 0.0481 | 1.18 | 不显著 |
| `acct_gap_post` | `ForecastDisp` | 0.0281 | 2.47 | 显著上升 |
| `acct_wstr_post` | `ForecastDisp` | 0.0495 | 4.26 | 显著上升 |
| `acct_gap_post` | `PriceDelay` | 0.0246 | 5.05 | 显著上升 |
| `acct_wany_post` | `PriceDelay` | 0.0197 | 4.49 | 显著上升 |
| `acct_gap_post` | `AuditFee` | -0.0514 | -8.85 | 显著下降，方向不适合主写 |

解释：

```text
这个 reduced form 不适合写成资本市场识别 DID。
```

原因：

1. `TobinQ` 没有政策后折价效应。
2. `AuditFee` 方向为负，和“验证成本上升”叙事相反。
3. `PriceDelay` 显著为正，但这又回到定价效率赛道，而且部分动态项已有预趋势。

## 5. 2024 会计规则 DID：对 X 的 first-stage

first-stage 很强，尤其是 `PreWashAny2123 x Post2024`：

| treatment | X outcome | coef | t |
|---|---|---:|---:|
| `acct_wany_post` | `NarrHardGap_direct_v1` | -0.0609 | -10.86 |
| `acct_wany_post` | `NarrHardWashing_direct_v1` | -0.1551 | -13.73 |
| `acct_wany_post` | `DU_kw` | 0.0677 | 2.49 |
| `acct_wany_post` | `HardCap_direct_v1` | 0.0602 | 4.36 |
| `acct_wstr_post` | `NarrHardGap_direct_v1` | -0.0369 | -6.37 |
| `acct_wstr_post` | `NarrHardWashing_direct_v1` | -0.0450 | -4.79 |

动态 first-stage 中，`PreWashAny2123` 对核心 X 的政策前项相对干净：

| X outcome | 2021 | 2022 | 2024 |
|---|---:|---:|---:|
| `NarrHardGap_direct_v1` | 0.0037, t=0.55 | 0.0074, t=1.36 | -0.0575, t=-9.19 |
| `NarrHardWashing_direct_v1` | 0.0184, t=1.54 | 0.0101, t=0.96 | -0.1465, t=-12.10 |

判断：

```text
2024 会计规则最适合写成“制度强化后，既有 data washing 企业减少错配/转向可验证化”的 first-stage / supplementary DID。
```

但它不适合作为“错配导致 Y 变化”的主因果识别，因为 reduced form 对 `TobinQ` 不成立。

## 6. 数据知识产权试点 DDD：对 Y 的 reduced form

设定：

```text
Y_it = DIPilot_pt + exposure_i x DIPilot_pt + controls + firm FE + year FE
cluster(Prov_num)
```

主结果表面显著：

| treatment | Y | coef | t |
|---|---|---:|---:|
| `dip_gap` | `TobinQ` | 0.0680 | 2.52 |
| `dip_gap` | `ForecastDisp` | 0.0317 | 4.34 |
| `dip_gap` | `AuditFee` | -0.0441 | -5.05 |
| `dip_gap` | `PriceDelay` | 0.0133 | 6.78 |
| `dip_wstr` | `TobinQ` | 0.0934 | 3.07 |
| `dip_wstr` | `Analyst` | -0.1176 | -3.79 |
| `dip_wstr` | `AuditFee` | -0.0245 | -4.28 |
| `dip_wstr` | `PriceDelay` | 0.0087 | 3.43 |

但动态检验不支持干净 DID：

| exposure | Y | pre-trend problem |
|---|---|---|
| `gap` | `TobinQ` | 2019、2020、2021 均显著为负 |
| `gap` | `PriceDelay` | 2020、2021、2023 已显著 |
| `wany` | `PriceDelay` | 2020、2021 已显著 |
| `wstr` | `ForecastDisp` | 2019、2020 已显著 |

判断：

```text
数据知识产权试点不适合作为主 DID。
```

它的问题不是“没结果”，而是“结果太像原本地区趋势延续”。北京、上海、江苏、浙江、广东等首批试点地区本来就是数字经济强省，pre-trend 很难讲干净。

## 7. 数据知识产权试点：对 X 的 first-stage

first-stage 也显著：

| treatment | X outcome | coef | t |
|---|---|---:|---:|
| `dip_wany` | `NarrHardGap_direct_v1` | -0.0830 | -17.54 |
| `dip_wany` | `NarrHardWashing_direct_v1` | -0.1922 | -15.89 |
| `dip_wany` | `DU_kw` | 0.1306 | 3.02 |
| `dip_wstr` | `NarrHardGap_direct_v1` | -0.0498 | -11.53 |
| `dip_wstr` | `NarrHardWashing_direct_v1` | -0.0710 | -8.22 |

但动态 first-stage 有明显预趋势：

| exposure | X outcome | pre-trend problem |
|---|---|---|
| `wany` | `NarrHardGap_direct_v1` | 2020、2021 已显著为正 |
| `wany` | `NarrHardWashing_direct_v1` | 2020、2021 已显著为正 |
| `gap` | `HardCap_direct_v1` | 2019、2020 已显著 |
| `wstr` | `DU_kw` | 2019、2020、2021 已显著 |

因此数据知识产权试点只能保留为探索性政策背景，不建议写成主识别。

## 8. IV 判断

不建议用这两个政策交互项做 IV。

### 8.1 2024 会计规则 IV

优点：

```text
PreWashAny2123 x Post2024 对 NarrHardGap / Washing 有强 first-stage。
```

问题：

```text
reduced form 对 TobinQ 不显著；对 AuditFee 和 PriceDelay 的方向也不适合主线。
```

如果拿它做 IV，2SLS 的经济含义会很别扭。

### 8.2 数据知识产权试点 IV

优点：

```text
first-stage 很强。
```

问题：

```text
试点地区本身直接影响数据交易、数字经济、融资、估值和分析师关注，排除限制很难成立。
动态项还有明显预趋势。
```

这个 IV 被审稿人攻击的概率很高。

## 9. 当前推荐写法

不要把 DID/IV 硬塞成主识别。更稳的结构是：

### 主结果

继续写：

```text
NarrHardGap / DataWashing 与 TobinQ、Analyst、AuditFee 的面板关系
```

### 准实验补充

只保留 2024 会计规则作为 supplementary evidence：

```text
2024 数据资源会计规则施行后，预先 data washing 企业的错配显著下降。
```

这句话有数据支持：

```text
PreWashAny2123 x Post2024 -> NarrHardWashing_direct_v1:
coef = -0.1465, t = -12.10 in dynamic first-stage.
```

它的含义是：

```text
制度验证压力会约束企业继续保持“高叙事、低硬证据”的状态。
```

但不要写：

```text
2024 会计规则导致资本市场惩罚 data washing 企业。
```

因为 `TobinQ` reduced form 不支持。

## 10. 下一步

如果一定想做强 DID，当前最有希望的不是这两个全市场/地区政策，而是更窄的事件：

1. 交易所年报问询中的“数据资源/数据资产”主题问询事件；
2. 企业首次披露数据资源入表或会计政策变更事件；
3. 企业所在地区数据交易所上线或数据产品登记制度启动，但要精确到城市和启动日期；
4. 企业被纳入地方数据知识产权登记名单，而不是省级试点。

这些事件能更直接冲击“可验证性”，比省级试点更干净。

目前在 20G 专利库之前，最值得保留的是：

```text
2024 accounting-rule first-stage DID
```

它不能救主因果，但能显著增强“错配是会被制度约束的”这一段。

## Sources

- 财政部印发《企业数据资源相关会计处理暂行规定》，中国政府网：<https://www.gov.cn/lianbo/bumen/202308/content_6899425.htm>
- 国家知识产权局办公室关于确定数据知识产权工作试点地方的通知：<https://www.cnipa.gov.cn/art/2022/11/30/art_543_180519.html>
- 国家知识产权局办公室关于确定2024年数据知识产权试点地方的通知：<https://www.cnipa.gov.cn/art/2023/12/29/art_75_189406.html>
