# 专刊投稿的宽口径 Y pivot memo

日期：2026-04-30

## 结论先行

不要继续在定价效率内部换 Y。`PriceDelay`、`SYNCH`、`PEAD60_abs`、`CS_Spread`、`mispricing`、`ICOE` 都会被归到资本市场信息效率 / 信息不对称 / 定价摩擦这一大框，近邻文献太多。

如果要投 `Artificial Intelligence, Data Investment and the Digital Economy` 这类专刊，主线应从“市场怎么看披露”改为“披露是否对应真实经营后果”。当前最可继续的三条是：

1. **经营韧性 / 基本面稳定**：`CashFlowVol`。
2. **供应链韧性 / 价值实现稳定**：`SCConc` 或 `SuppConc`。
3. **真实数据资产化 / 数据资源入表**：需要补 2024 年后数据，最贴 `data investment`，但不是现成面板。

## 本地 DU_kw 宽筛结果

规格：`Y_t = L.DU_kw + controls + firm FE + year FE`，cluster 到 `IndYear_num`。已用 Stata MCP / `reghdfe` 跑完。

| 家族 | Y | 方向 | 结果 | 初判 |
|---|---|---:|---|---|
| resilience | `CashFlowVol` | - | -0.000510, t=-2.81, p=0.005 | 过关，且非定价效率。 |
| supply chain | `SCConc` | - | -0.273847, t=-3.86, p<0.001 | 过关，但供应链风险/集中度近邻较多。 |
| supply chain | `SuppConc` | - | -0.407074, t=-4.90, p<0.001 | 过关，供应商侧比综合 SCConc 可再看。 |
| capital allocation | `InvestIneff` | - | -0.002679, t=-3.35, p<0.001 | 过关，但投资效率已被 data element marketization 文献占位。 |
| financing | `SA` | - | -0.001477, t=-5.47, p<0.001 | 过关，但融资约束已是常见机制，不适合主线。 |
| analyst | `ForecastDisp`, `Analyst`, `FcstAcc`, `ReportFreq` | expected | 多数过关 | 又回到信息环境，不适合主 Y。 |
| crash risk | `NCSKEW`, `DUVOL` | - | 过关 | 仍属资本市场风险/信息不对称，且前面双测度不稳。 |
| productivity | `TFP` | + | t=1.42, p=0.158 | 本地不过关，先不推。 |
| investor base | `InstHold`, `InstStable` | + | 方向相反 | 不推。 |

输出文件：

- `v3/results/broad_y_screen_stata.csv`
- `v3/scripts/broad_y_screen_stata.do`

## 文献拥挤度

### 1. 定价效率 / 资本市场信息效率：放弃做主线

已有直接近邻：

- Sun and Du (2024), `data assets disclosure -> stock price synchronicity`。
- 李世刚等（2025）, `企业数据资产信息披露 -> 资本市场定价效率`。
- 新近还有 `data asset disclosure -> stock mispricing`。

因此 `PEAD60_abs` 即使精确变量没撞，也只是事件窗口版定价效率，不够新。

### 2. 投资效率 / 融资约束 / 企业价值：拥挤

已有：

- `data element marketization -> corporate investment efficiency`，机制含 market competition、financing constraints、management efficiency。
- `data element utilization -> firm value`，机制含 absorptive capacity，调节为 ESG disclosure。
- `data elements -> innovation`、`data element and big data policy -> enterprise innovation` 文献很多。

因此 `InvestIneff`、`firm value`、`innovation/R&D/patent` 不适合新主线。

### 3. 供应链：有空间但已有强近邻

已有：

- `data asset information disclosure -> supply chain risk`，且用了 DML。
- `数据资产化 -> 供应链集中度`。

所以 `SCConc/SuppConc` 可以作为经营韧性分支，但不应声称供应链方向没人做。若要写，最好强调“年报中的数据要素利用披露 -> 供应商依赖下降 / 价值实现稳定”，不要写成泛泛供应链风险。

### 4. 现金流波动：相对更干净

暂未查到精确的 `data asset/data element disclosure -> CashFlowVol` 主效应组合。供应链风险文献会提到 cash-flow bullwhip 或把 cash-flow volatility 作为调节变量，但不是本文这种年度 firm-level 主 Y。

这条与数据要素利用的业务逻辑也最顺：

`数据资源进入经营决策/风控/库存/供应链/客户管理 -> 现金流预测和调度能力增强 -> 经营现金流波动下降`

## 当前推荐

### 主推方案 A：经营韧性

中文题目口径：

> 年报中的数据要素利用披露与企业经营韧性

英文题目口径：

> Data-Element Utilization Disclosure and Corporate Operational Resilience

核心 Y：

- 主 Y：`CashFlowVol`
- 辅助 Y：`SCConc` / `SuppConc`
- 不再把 `PriceDelay/PEAD/CS_Spread` 放主文，最多放附录说明资本市场也能识别。

优势：

- 跳出定价效率拥挤区。
- 本地 DU_kw 结果过关。
- 特刊接受度高：AI/data investment/digital economy 与 resilience、resource allocation、operations 都能接。

风险：

- 现金流波动是机制变量旧结果，需要重写成主结果，而不是定价效率机制。
- 要补一组 robustness：用 `CashFlowVol` 的替代口径、分行业需求波动控制、经营现金流/销售波动对比。

### 备选方案 B：供应链韧性

主 Y：

- `SuppConc`
- `SCConc`

优势：

- 本地结果很强。
- 很贴数据要素在供应链预测、库存、采购和协同中的应用。

风险：

- 供应链方向近邻较强，尤其是 `data asset disclosure -> supply chain risk` 和 `数据资产化 -> 供应链集中度`。

### 备选方案 C：真实数据资产化

主 Y：

- 2024 后数据资源入表虚拟变量 / 数据资源金额 / 数据资源附注质量。

优势：

- 最接近 `data investment` 和数据资产化。
- 与年报 X 的机制最短：披露的数据要素利用是否转化为可确认的数据资源。

风险：

- 数据需要新抓，窗口短，主回归可能只能做 2024-2025 截面/短面板。

## 下一步建议

1. 不再继续跑新的定价效率变量。
2. 先把 `CashFlowVol` 当主 Y 重跑一套完整 H1 风格表：同期、滞后、控制 `GenericNarr`、控制行业需求波动或销售波动、按数据应用/治理/闭环分解。
3. 同时把 `SuppConc/SCConc` 作为第二 Y，看是否能组成“经营韧性双证据”。
4. 若结果稳，再单独检索 `CashFlowVol` 精确撞题，确认能否写成主线。
