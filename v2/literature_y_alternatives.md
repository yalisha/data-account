# 候选主因变量 Y 文献调研

## OPEN_QUESTIONS

1. `spec` 对 `SPI = 1 - R²` 采用的是传统“公司特质信息含量”读法，因此预期 `DU -> SPI > 0`；但当前 `outline.md` 已把 `SYNCH > 0` 放进 `Chen et al. (2021)` 的 public-info 框架。两套读法并不兼容，所以 `SPI` 在 Step B 的 gate 只能按传统文献口径执行，结果需单独解释。
2. `Turnover_year` 的理论方向并不单边。流动性改善时换手率可能上升，但投机噪声被压缩时也可能下降。为让 Step B 的 `pass / fail` 规则可执行，do 文件暂按“效率改善对应换手率上升”记为 `expected_sign = positive`，报告里会把它当作 mixed-sign 指标处理。
3. `CS_Spread` 需要股票层面的日高价/低价。当前 `v1/data_parquet/` 下未找到含 `Stkcd + high/low` 的日行情文件，因此 Step B 只能把它记为 unavailable；如果后续补到 CSMAR 日高低价，再优先回补这一项。

## 1. 直接竞品原文复核

### 1.1 李世刚等（2025，中国工业经济）到底用了哪些 Y

- 主因变量是**下一期股价同步性 `SYN`**，定义为 `ln(R² / (1 - R²))`。
- 稳健性并没有切到 `PriceDelay`、流动性、价差或崩盘风险，而是仍留在**同一同步性家族**：
  - `SYN_M`：把市场收益和行业收益的加权口径从总市值权重替换为流通市值权重；
  - 滞后 `SYN`：检验动态持续性。
- 结论：李世刚等**只占了 SYNCH 这条主 Y 赛道**，并没有占到 `PriceDelay / Amihud / Zero-return / Crash-risk / Spread` 这些替代 Y。

### 1.2 Sun and Du（2024，IRFA）到底用了哪些 Y

- 主因变量同样是**股价同步性 `SYN`**。
- 稳健性仍是**同步性口径内部替换**（文中 `SYN_NEW`），并未扩展到 `PriceDelay`、流动性或崩盘风险。
- 结论：Sun-Du 也主要占了**同步性家族**，没有把 microstructure / crash-risk 这几类 Y 占满。

### 1.3 竞品占位结论

| 文献 | 主 Y | 稳健性 Y | 是否碰到 PriceDelay | 是否碰到流动性/价差 | 是否碰到崩盘风险 |
|---|---|---|---|---|---|
| 李世刚等（2025） | `SYN` | `SYN_M`、lagged `SYN` | 否 | 否 | 否 |
| Sun and Du（2024） | `SYN` | `SYN_NEW` | 否 | 否 | 否 |

最关键的占位事实只有一条：**`SYNCH` 及其反向变体（`1-R²` / nonsynch）已经高度拥挤。**

## 2. 候选 Y 长名单（Step A）

| 候选 Y | 家族 | 核心文献锚点 | 公式/定义简写 | 披露 -> Y 预期 | 现有数据可构造度 | 与李世刚 / Sun-Du 冲撞 |
|---|---|---|---|---|---|---|
| `Amihud_year` | 流动性 / 信息不对称 | Amihud (2002); Welker (1995) | 年均 `|r_d| / volume_d` | `-` | 高 | 低 |
| Quoted / effective bid-ask spread | 流动性 / 信息不对称 | Welker (1995); Roll (1984) | 报价或成交价差 | `-` | 低 | 低 |
| `ZeroRet_ratio` | 流动性 / 交易阻滞 | Lesmond (2005) | `count(ret=0) / N` | `-` | 高 | 低 |
| Roll spread | 流动性 / 微观结构 | Roll (1984) | `2 * sqrt(-cov(Δp_t, Δp_{t-1}))` | `-` | 中 | 低 |
| `CS_Spread` | 流动性 / 微观结构 | Corwin and Schultz (2012); Fong et al. (2017) | 基于日高低价的隐含价差 | `-` | 当前为零 | 低 |
| `Turnover_year` | 交易行为 | Easley et al. (1996) | 年均 `Dnvaltrd / Dsmvosd` | mixed | 高 | 低 |
| Abnormal trading volume / `TurnoverVol` | 交易行为 | Easley et al. (1996); Fong et al. (2017) | 交易活跃度偏离或换手波动 | mixed | 中 | 低 |
| `SPI = 1 - R²` | 股价信息含量 | Durnev et al. (2003); Ferreira et al. (2011) | `1 - R²_synch` | `+`（传统读法） | 高 | 高 |
| FSV / idiosyncratic return variation | 股价信息含量 | Morck et al. (2000); Durnev et al. (2003) | firm-specific return variation | `+` | 中 | 高 |
| FERC / future earnings response coefficient | 盈余信息提前反映 | Durnev et al. (2003) | 未来盈余对当期回报的映射强度 | `+` | 低 | 中 |
| `NCSKEW` | 崩盘风险 | Chen et al. (2001); Hutton et al. (2009); 许年行等（2012） | 负条件偏度 | `-` | 高 | 低 |
| `DUVOL` | 崩盘风险 | Hutton et al. (2009); 许年行等（2012） | down/up 波动率对数比 | `-` | 高 | 低 |

## 3. 为什么这些 Y 值得看

### 3.1 真正能和竞品拉开距离的，是哪几类

最能拉开距离的不是 `SPI / FSV` 这类“同步性换壳”指标，而是下面两类：

1. **microstructure / liquidity 家族**：`Amihud`、`ZeroRet_ratio`、`Roll spread`、`CS_Spread`。
2. **crash-risk 家族**：`NCSKEW`、`DUVOL`。

理由很直接：

- 李世刚和 Sun-Du 都在讲“公司特质信息进入价格后，同步性下降”。
- 如果你改成 `SPI = 1 - R²` 或 `FSV`，概念上仍在同一大类里，差异化有限。
- 如果改成流动性、价差或崩盘风险，叙事会从“同步性 / 信息含量”转到“市场摩擦 / 交易成本 / 极端负尾风险”，更能与两篇竞品错位。

### 3.2 哪些 Y 虽然经典，但不适合你现在这篇

- `SPI`：文献上经典，但它本质上就是 `SYNCH` 的反向变体；在你当前 `outline` 已经接受 `SYNCH > 0` 的前提下，`SPI` 会自动与现有读法冲突。
- `FERC`：学术上有意思，但不是一个可以像 `PriceDelay` 那样直接单独扔进 `reghdfe` 的独立 Y，更像一整套额外建模任务。
- `Turnover_year`：构造最容易，但方向最容易被“流动性提升”和“噪声交易收缩”两套机制同时拉扯。

## 4. Top 3 推荐（按“差异化 + 文献支撑 + 当前可构造度”综合）

### Top 1: `Amihud_year`

- **差异化**：和 `SYNCH`、`PriceDelay` 都不是同一构念，能把故事切到“价格发现摩擦/交易成本”。
- **文献支撑**：最成熟，国际文献接受度最高，公式干净。
- **可构造度**：当前仓库已有 `amihud_daily.parquet`，直接年聚合即可。
- **不足**：这条线“太通用”，不是数据要素情境独有；更适合做替代/补充 DV，不够适合翻主线。

### Top 2: `NCSKEW`

- **差异化**：属于负尾风险，不和两篇竞品主 DV 重叠。
- **文献支撑**：`Chen-Hong-Stein -> Hutton -> 中文 crash-risk` 的线非常成熟。
- **可构造度**：当前数据可做，且与 `DUVOL` 共用同一套日频残差中间量。
- **不足**：崩盘风险赛道本身已经很热，不算“冷门新 Y”，更像“可辩护的外延结果”。

### Top 3: `ZeroRet_ratio`

- **差异化**：比 `SPI` 更远离同步性家族，能落到交易阻滞/不活跃交易。
- **文献支撑**：`Lesmond` 线足够经典，和流动性文献兼容。
- **可构造度**：只靠当前 `daily_return.parquet` 就能完成。
- **不足**：解释力度弱于 `Amihud` 和 `crash-risk`，更像简洁的低成本补充指标。

## 5. 没进 Top 3 的几个关键原因

| 候选 Y | 为什么没进前三 |
|---|---|
| `CS_Spread` | 概念上本来很强，但当前数据不可构造，综合分被直接拉低。 |
| `SPI` | 和 `SYNCH` 一体两面，差异化不足，而且和当前 `outline` 的 `SYNCH > 0` 读法冲突。 |
| `Turnover_year` | 构造虽稳，但理论方向混合，容易在实证里出现“一个测度正、另一个测度负”的尴尬。 |
| `DUVOL` | 与 `NCSKEW` 同类，但常比 `NCSKEW` 更容易出现方向不稳。 |
| FERC | 不是当前 spec 这一轮应优先投入的“直接替代 Y”。 |

## 6. Step A 结论

1. **李世刚原文实际只用了 `SYN` 及其同家族稳健性口径**，没有碰 `PriceDelay`、流动性、价差或崩盘风险。
2. **Sun-Du 也是 `SYNCH` 主线**，同样没有把 `PriceDelay / Amihud / NCSKEW` 这几类占满。
3. 从纯文献和当前数据角度看，最值得进入 Step B 的 actionable Top 3 是：
   - `Amihud_year`
   - `NCSKEW`
   - `ZeroRet_ratio`
4. 如果只看概念差异化而不管当前数据，`CS_Spread` 本应进入候选前列；但在当前仓库条件下，它只能留作**补数据后的优先回补项**。
