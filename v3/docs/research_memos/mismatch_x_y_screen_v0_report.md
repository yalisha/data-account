# 年报叙事-能力错配 X 与备选 Y：v0 试跑报告

日期：2026-04-30

## 1. 这次做了什么

这次不是继续救“年报数据要素披露 -> 定价效率/经营韧性”，而是把 X 改成更接近“年报叙事 - 可验证能力/潜在能力”的错配变量。

执行方式：Stata MCP 跑通。

主脚本：

```text
/Users/mac/computerscience/0做完了/15会计研究/v3/src/stata/mismatch_x_y_screen_v0.do
```

输入数据：

```text
/Users/mac/computerscience/0做完了/15会计研究/v1/data_stata/reg_sample_v18.dta
```

主要输出：

```text
/Users/mac/computerscience/0做完了/15会计研究/v3/results/data/mismatch_x_panel_v0.dta
/Users/mac/computerscience/0做完了/15会计研究/v3/results/stata/mismatch_x_summary_v0.csv
/Users/mac/computerscience/0做完了/15会计研究/v3/results/stata/mismatch_x_y_screen_v0.csv
/Users/mac/computerscience/0做完了/15会计研究/v3/results/logs/mismatch_x_y_screen_v0.log
/Users/mac/computerscience/0做完了/15会计研究/v3/results/logs/mismatch_x_y_screen_v0_mcp_rerun.log
```

模型：

```text
Y_t = L.X + controls + firm FE + year FE, cluster(IndYear_num)
```

控制变量：

```text
Size Lev ROA TobinQ Age Growth IndepRatio Dual Top1Share SOE CFO
```

当 Y 本身是控制变量时，脚本会自动把该 Y 从控制变量里移除。

## 2. X 怎么构造

第一组是“年报内部错配”：

- `Mismatch_sub`：年报数据叙事强度分位数 - 年报实质表达分位数。
- `DataWashing_sub`：高叙事、低实质表达。
- `DataHushing_sub`：高实质表达、低叙事。

这一组的硬伤是：两边都来自年报文本，只能做快速试跑，不能当最终“真实能力”。

第二组是“年报叙事 - 潜在数据能力”：

- `DataPotentialRaw = HighTech + DigEconCore + StrategicEmerging + IndustryCluster + z(ln(1+LQ))`
- `Mismatch_potential`：年报数据叙事强度分位数 - 潜在数据能力分位数。
- `DataWashing_pot`：高叙事、低潜在数据能力。
- `DataHushing_pot`：高潜在数据能力、低叙事。
- `VerifiedData_pot`：高叙事、高潜在数据能力。

这一组更像“差值 X”，但目前仍是粗代理。它更适合用来判断路线能不能跑，不能替代后续的专利、软著、数据岗位、AI/数据投资、数据资产入表等硬能力数据。

## 3. X 的可用样本与分布

| X | N | mean | 说明 |
|---|---:|---:|---|
| `DataWashing_sub` | 43,735 | 0.097 | 高叙事、低年报实质表达 |
| `DataHushing_sub` | 43,735 | 0.063 | 高年报实质表达、低叙事 |
| `DataWashing_sub_strict` | 43,735 | 0.00005 | 样本几乎没有，不建议用 |
| `DataHushing_sub_strict` | 43,735 | 0.0009 | 样本太少，不建议用 |
| `DataWashing_pot` | 43,735 | 0.220 | 可用 |
| `DataHushing_pot` | 43,735 | 0.217 | 可用 |
| `VerifiedData_pot` | 43,735 | 0.280 | 可用 |
| `DataWashing_pot_strict` | 43,735 | 0.048 | 可做稳健性 |
| `DataHushing_pot_strict` | 43,735 | 0.031 | 可做稳健性，但样本偏少 |
| `VerifiedData_pot_strict` | 43,735 | 0.082 | 可做稳健性 |

## 4. 初步显著信号

下面只列 `|t| >= 1.96` 的重点结果。完整结果见：

```text
/Users/mac/computerscience/0做完了/15会计研究/v3/results/stata/mismatch_x_y_screen_v0.csv
```

### 4.1 连续错配：`Mismatch_potential`

这是当前最强的 X。

| Y | coef | t | p | 初步含义 |
|---|---:|---:|---:|---|
| `TobinQ` | 0.194 | 8.681 | <0.001 | 叙事相对潜在能力越高，估值越高 |
| `SCConc` | 1.702 | 5.331 | <0.001 | 综合供应链集中度更高 |
| `SuppConc` | 1.435 | 3.778 | <0.001 | 供应商集中度更高 |
| `Analyst` | -0.132 | -4.707 | <0.001 | 分析师关注更低 |
| `RatingDisp` | -0.024 | -2.936 | 0.003 | 评级分歧更低 |
| `ReportFreq` | -0.077 | -2.510 | 0.012 | 研报频率更低 |
| `InvestIneff` | 0.006 | 2.358 | 0.019 | 投资效率更差 |
| `SA` | 0.003 | 2.788 | 0.005 | SA 方向需回原定义确认 |
| `TFP` | -0.065 | -2.572 | 0.011 | 生产率更低 |

这个变量有信号，但也最容易被质疑为“行业/区位潜在能力的反面”或“概念炒作被估值奖励”。后续必须把能力端换成更硬的数据。

### 4.2 四象限：`DataWashing_pot`

| Y | coef | t | p | 初步含义 |
|---|---:|---:|---:|---|
| `TobinQ` | 0.064 | 4.089 | <0.001 | 高叙事、低潜力企业反而估值更高 |
| `SCConc` | 0.543 | 2.787 | 0.005 | 供应链集中度更高 |
| `Analyst` | -0.036 | -2.020 | 0.044 | 分析师关注更低 |
| `TFP` | -0.036 | -1.993 | 0.047 | 生产率更低 |

这个结果更像“数据要素叙事泡沫/叙事溢价”：真实潜力弱但披露强，估值更高、TFP 更低、分析师覆盖更少。这个方向比单纯年报数据披露更有空间。

### 4.3 四象限：`DataHushing_pot`

| Y | coef | t | p | 初步含义 |
|---|---:|---:|---:|---|
| `TobinQ` | -0.051 | -3.696 | <0.001 | 高潜力、低叙事企业估值更低 |
| `TFP` | 0.025 | 2.035 | 0.043 | 生产率更高 |
| `SA` | -0.001 | -2.023 | 0.043 | SA 方向需回原定义确认 |

严格版还出现：

| Y | coef | t | p |
|---|---:|---:|---:|
| `RatingDisp` | 0.032 | 3.343 | 0.001 |
| `SCConc` | -1.436 | -4.191 | <0.001 |
| `SuppConc` | -1.665 | -4.179 | <0.001 |
| `TobinQ` | -0.085 | -3.593 | <0.001 |

这个比 `DataWashing_pot` 更有论文味：有潜在能力但不说，TFP 更高、估值更低，像“沉默披露/低估”。但现在的“能力”还不够硬，必须补强。

### 4.4 四象限：`VerifiedData_pot`

| Y | coef | t | p | 初步含义 |
|---|---:|---:|---:|---|
| `TobinQ` | 0.051 | 3.636 | <0.001 | 高叙事、高潜力企业估值更高 |
| `Analyst` | -0.069 | -4.607 | <0.001 | 分析师关注更低 |
| `InstHold` | 0.312 | 2.372 | 0.018 | 机构持股更高 |
| `SA` | 0.002 | 3.053 | 0.002 | SA 方向需回原定义确认 |

这个可以作为对照组：不是所有高叙事都等于 washing。高潜力和高披露同时存在时，估值与机构持股也有反应。

## 5. 不建议主打的结果

`sub` 版本暂时不适合主线：

- `DataWashing_sub_strict` 和 `DataHushing_sub_strict` 样本几乎为零，不能作为主变量。
- `DataWashing_sub` 对 `TobinQ`、`InstStable` 有信号，但两边都来自年报文本，容易被批评为文本口径内部重排。
- `DataHushing_sub` 主要只在分析师预测准确度上有信号，主线不够稳。

经营韧性和定价效率也不建议继续当主 Y：

- `CashFlowVol` 只在极小样本的 `DataWashing_sub_strict` 上显著，不稳。
- 定价效率方向已经过于拥挤，且这次新的 X 更适合讲“叙事-能力错配”带来的资本市场识别、资源配置或生产率差异。

## 6. 当前判断

可以跑。

最值得继续的是：

1. `DataHushing_pot -> TobinQ / TFP / analyst uncertainty`
2. `DataWashing_pot -> TobinQ / TFP / Analyst`
3. `Mismatch_potential -> analyst information environment / supply-chain concentration / investment inefficiency`

但这里的 `pot` 只是 v0。真正要写特刊，下一步应当把能力端替换为更硬的“真实数据能力/数据投资”：

- 数字/AI/数据相关专利。
- 软件著作权。
- 数据岗位招聘。
- 数据资产入表金额或首次入表。
- AI/数据资本开支或无形资产明细。
- 数据产品、数据平台、数据合作或数据交易记录。

如果能把 X 固化为：

```text
Data narrative in annual report - hard data capability / data investment
```

那么主题会比“年报数据要素披露有没有经济后果”干净很多。
