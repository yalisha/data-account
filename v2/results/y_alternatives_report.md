# 候选主因变量 Y 诊断报告

## OPEN_QUESTIONS

1. `spec` 对 `SPI` 采用的是经典 `1 - R²` 读法，因此 Step B gate 把 `expected_sign` 设为正；但当前 `outline.md` 已接受 `SYNCH > 0` 的 public-info 解释，这会让 `SPI` 在经验上机械失败。这个冲突不是代码问题，是口径冲突。
2. `Turnover_year` 文献上没有单边方向。为了让 gate 可执行，回归脚本临时把它按 `expected_sign = positive` 记账；因此它的 `pass / fail` 只能做弱判断，不能像 `Amihud` 或 `NCSKEW` 那样硬读。
3. `CS_Spread` 当前无法构造。`v1/data_parquet/` 下未找到股票层面的日高价/低价字段，因此该项只能记为 unavailable；若后续补到高低价日行情，应第一时间重跑。

## 1. 文献调研摘要

### 1.1 候选 Y 长名单（Step A 摘要）

| 候选 Y | 家族 | 披露 -> Y 预期 | 当前可构造度 | 与李世刚 / Sun-Du 冲撞 |
|---|---|---|---|---|
| `Amihud_year` | 流动性 / 信息不对称 | `-` | 高 | 低 |
| Bid-Ask Spread | 流动性 / 信息不对称 | `-` | 低 | 低 |
| `ZeroRet_ratio` | 流动性 / 交易阻滞 | `-` | 高 | 低 |
| Roll spread | 流动性 / 微观结构 | `-` | 中 | 低 |
| `CS_Spread` | 流动性 / 微观结构 | `-` | 当前为零 | 低 |
| `Turnover_year` | 交易行为 | mixed | 高 | 低 |
| Abnormal volume / `TurnoverVol` | 交易行为 | mixed | 中 | 低 |
| `SPI = 1 - R²` | 股价信息含量 | `+`（传统读法） | 高 | 高 |
| FSV / idiosyncratic variation | 股价信息含量 | `+` | 中 | 高 |
| FERC | 盈余信息提前反映 | `+` | 低 | 中 |
| `NCSKEW` | 崩盘风险 | `-` | 高 | 低 |
| `DUVOL` | 崩盘风险 | `-` | 高 | 低 |

### 1.2 竞品占位结论

- 李世刚等（2025）主 Y 是 `SYN`，稳健性还是 `SYN_M` 和 lagged `SYN`，没有切到 `PriceDelay`、`Amihud`、`ZeroRet` 或 crash-risk。
- Sun and Du（2024）主 Y 也是 `SYN`，稳健性是 `SYN_NEW`，仍然属于同一家族。
- 因此，真正拥挤的是**同步性家族**；`PriceDelay` 之外最能拉开差距的，是流动性/价差和崩盘风险两大类。

### 1.3 Top 3 推荐（文献 + 数据可构造度）

1. `Amihud_year`
2. `NCSKEW`
3. `ZeroRet_ratio`

保留意见：

- `CS_Spread` 如果有高低价数据，本应进入前三；但当前数据缺口让它只能退出实际执行列表。
- `SPI` 虽然文献经典，但和 `SYNCH` 过近，不适合作为“差异化主 Y”。

## 2. 候选 Y 构造口径 + CSMAR 字段映射

| Y | 构造口径 | 数据字段 / 文件 | 备注 |
|---|---|---|---|
| `Amihud_year` | 年均 `ILLIQ` | `amihud_daily.parquet: ILLIQ` | 与现有样本中的 `Amihud` 同源 |
| `Turnover_year` | 年均 `Dnvaltrd / Dsmvosd` | `daily_return.parquet: Dnvaltrd, Dsmvosd` | 与现有样本中的 `Turnover` 同源 |
| `ZeroRet_ratio` | `count(Dretnd == 0) / N` | `daily_return.parquet: Dretnd` | 只保留年内至少 60 个交易日的 firm-year |
| `SPI` | `1 - R2_synch` | `price_synchronicity.parquet: R2_synch` | 与 `SYNCH` 同家族，不是独立新信息 |
| `NCSKEW` | 市场模型残差 `log(1 + abnormal return)` 的负条件偏度 | `daily_return.parquet + market_index.parquet` | 市场基准取 `Indexcd = 1` |
| `DUVOL` | `log(((n_up-1) * sum_down W^2) / ((n_down-1) * sum_up W^2))` | `daily_return.parquet + market_index.parquet` | 与 `NCSKEW` 共用中间残差 |
| `CS_Spread` | Corwin-Schultz 高低价隐含价差 | 需要股票层面日高/低价 | 当前 unavailable |

### 2.1 构造后样本覆盖

| y_name | nonmissing_candidate_frame | nonmissing_reg_sample | mean | sd | p01 | median | p99 | note |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `Amihud_year` | 43,735 | 43,735 | 0.045839 | 0.047697 | 0.001832 | 0.031549 | 0.294076 | - |
| `Turnover_year` | 43,735 | 43,735 | 27.318987 | 22.968445 | 2.528880 | 20.234564 | 121.435902 | - |
| `ZeroRet_ratio` | 43,735 | 43,735 | 0.026532 | 0.024298 | 0.000000 | 0.020492 | 0.123967 | - |
| `SPI` | 43,677 | 43,677 | 0.605348 | 0.170560 | 0.226582 | 0.610293 | 0.948807 | - |
| `NCSKEW` | 43,735 | 43,735 | -0.471767 | 0.623536 | -2.589614 | -0.436738 | 1.171408 | - |
| `DUVOL` | 43,735 | 43,735 | -0.307740 | 0.320488 | -1.122984 | -0.313933 | 0.530365 | - |
| `CS_Spread` | 0 | 0 | - | - | - | - | - | unavailable |

## 3. Step B 批量回归结果表

设定：

- 样本：`reg_sample_iv_v16.dta` 上按 `tsset Stkcd_num year_num` 生成 `DU_kw_lag` / `DU_llm_lag`
- 固定效应：Firm FE + Year FE
- 聚类：`cluster(IndYear_num)`
- 控制变量：`Size Lev ROA TobinQ Age Growth IndepRatio Dual Top1Share SOE CFO`
- 有效回归样本：`37,294`（`PriceDelay` / candidate Y），`37,245`（`SYNCH` / `SPI`）

| y_name | du_measure | expected_sign | coef | se | t | p | N | status | baseline_direction_match |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| `PriceDelay` | `DU_kw` | `-` | -0.003874 | 0.001001 | -3.869890 | 0.000116 | 37,294 | pass | baseline |
| `SYNCH` | `DU_kw` | `+` | 0.015851 | 0.005395 | 2.937803 | 0.003388 | 37,245 | pass | baseline |
| `PriceDelay` | `DU_llm` | `-` | -0.004527 | 0.000852 | -5.311208 | 0.000000 | 37,294 | pass | baseline |
| `SYNCH` | `DU_llm` | `+` | 0.026369 | 0.004759 | 5.540806 | 0.000000 | 37,245 | pass | baseline |
| `Amihud_year` | `DU_kw` | `-` | -0.001478 | 0.000320 | -4.617582 | 0.000004 | 37,294 | pass | match_delay |
| `Amihud_year` | `DU_llm` | `-` | -0.000267 | 0.000228 | -1.171738 | 0.241592 | 37,294 | fail | match_delay |
| `Turnover_year` | `DU_kw` | `+` | 0.376379 | 0.156630 | 2.402985 | 0.016450 | 37,294 | pass | mismatch |
| `Turnover_year` | `DU_llm` | `+` | -0.574561 | 0.140705 | -4.083446 | 0.000048 | 37,294 | fail | match_delay |
| `ZeroRet_ratio` | `DU_kw` | `-` | -0.000460 | 0.000137 | -3.364953 | 0.000796 | 37,294 | pass | match_delay |
| `ZeroRet_ratio` | `DU_llm` | `-` | 0.000095 | 0.000136 | 0.700135 | 0.484012 | 37,294 | fail | mismatch |
| `SPI` | `DU_kw` | `+` | -0.003292 | 0.001138 | -2.892615 | 0.003911 | 37,245 | fail | match_delay |
| `SPI` | `DU_llm` | `+` | -0.005485 | 0.000981 | -5.591418 | 0.000000 | 37,245 | fail | match_delay |
| `NCSKEW` | `DU_kw` | `-` | -0.014414 | 0.004478 | -3.219293 | 0.001328 | 37,294 | pass | match_delay |
| `NCSKEW` | `DU_llm` | `-` | 0.006532 | 0.005051 | 1.293213 | 0.196248 | 37,294 | fail | mismatch |
| `DUVOL` | `DU_kw` | `-` | -0.007278 | 0.002348 | -3.099901 | 0.001992 | 37,294 | pass | match_delay |
| `DUVOL` | `DU_llm` | `-` | 0.005185 | 0.002541 | 2.040099 | 0.041613 | 37,294 | fail | mismatch |
| `CS_Spread` | `DU_kw` | `-` | - | - | - | - | 0 | unavailable | unavailable |
| `CS_Spread` | `DU_llm` | `-` | - | - | - | - | 0 | unavailable | unavailable |

## 4. 分类判断

### 4.1 过关（pass）候选

- `Amihud_year ~ DU_kw_lag`
- `Turnover_year ~ DU_kw_lag`
- `ZeroRet_ratio ~ DU_kw_lag`
- `NCSKEW ~ DU_kw_lag`
- `DUVOL ~ DU_kw_lag`

说明：

- 所有 candidate 的 `pass` 都**只出现在 `DU_kw` 一侧**，没有任何一个新 Y 能复制 baseline 那种“双测度都过”的稳定性。
- `Turnover_year` 虽然在 `DU_kw` 下过关，但它和 `PriceDelay` baseline 的原始符号不一致，而且 `DU_llm` 一侧显著翻负，稳定性最差。

### 4.2 Marginal

- 无。

### 4.3 Fail / unavailable

- `Amihud_year ~ DU_llm_lag`
- `Turnover_year ~ DU_llm_lag`
- `ZeroRet_ratio ~ DU_llm_lag`
- `SPI ~ DU_kw_lag`
- `SPI ~ DU_llm_lag`
- `NCSKEW ~ DU_llm_lag`
- `DUVOL ~ DU_llm_lag`
- `CS_Spread` 两个测度都 unavailable

## 5. 最终建议

### 5.1 三选一结论

**建议：保持 `PriceDelay` 主 + `SYNCH` 讨论段不变。**

### 5.2 为什么不是“替代”

要替代 `PriceDelay`，至少要满足两条：

1. 候选 Y 在 `DU_kw` 和 `DU_llm` 下都方向稳、显著稳。
2. 它与李世刚 / Sun-Du 的 `SYNCH` 家族要拉得更开。

当前没有一个候选同时满足这两条：

- `Amihud_year`、`ZeroRet_ratio`、`NCSKEW`、`DUVOL` 都只在 `DU_kw` 下过关；
- `Turnover_year` 甚至出现一正一负；
- `SPI` 虽然显著，但它只是 `SYNCH` 的反向写法，本身就不适合拿来替代主 DV。

### 5.3 为什么也不建议正式“补充”为第三稳健性 DV

如果要把新 Y 放进正文作为第三 DV，它至少应该满足“方向解释清楚 + 双测度别互相打架”。当前最接近这个标准的是 `Amihud_year`，但它仍只有 `DU_kw` 通过，`DU_llm` 完全不跟。

因此，更稳的判断是：

- **正文层面不新增第三主 DV。**
- 如果你后续一定要在附录里放一个“补充指标”，优先顺序可以是：
  1. `Amihud_year`
  2. `NCSKEW`
  3. `ZeroRet_ratio`

这里的“补充”只适合放附录或 very light robustness，不足以改写主故事。

### 5.4 最后的 bottom line

- **主线最稳的仍然是 `PriceDelay`。**
- **竞品真正占的是 `SYNCH` 家族，不是 `PriceDelay`。**
- **本轮候选 Y 没有跑出足够强的新主 DV。**
- 因此，最优动作不是换主 Y，而是继续坚持：
  - `PriceDelay` 做主结果；
  - `SYNCH` 留在 discussion / robustness 段；
  - 如需补一个 appendix Y，只考虑 `Amihud_year` 或 `NCSKEW`，且明说“单测度有效、稳定性不如主结果”。
