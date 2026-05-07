# 四个 Y 的 X-Y 组合撞题检索

日期：2026-04-30

X 口径：年报中的数据要素 / 数据资产 / 数据要素利用披露。判定时把“精确 X-Y 组合”和“邻近 Y 家族”分开。

## 总表

| Y | 是否已有别人做 X-Y 组合 | 撞题程度 | 判断 |
|---|---|---|---|
| `PEAD60_abs` / post-announcement absolute CAR | 未查到精确命中 | 低 | 可主推。已有 PEAD 文献很多，但没有查到“数据要素/数据资产披露 -> 公告后绝对残余调整”的精确组合。 |
| `ICOE_EY` | 已有成本权益资本家族 | 高 | 不建议当唯一主 Y。可作为机制或第二结果。 |
| `ICOE_PEG` | 已有 PEG/MPEG/OJ 权益资本成本组合 | 高 | 该 Y 家族已被直接做过，理论标签强但 novelty 弱。 |
| `CS_Spread` / spread/liquidity | 已有股票流动性家族 | 中高 | 未查到 Corwin-Schultz spread 精确口径，但“数据资产披露 -> 股票流动性”已被做。适合补充，不适合主推。 |

## 1. `PEAD60_abs`

### 检索词

- `"数据要素" "公告后" "CAR"`
- `"数据资产" "PEAD"`
- `"data asset disclosure" "post-earnings-announcement drift"`
- `"data asset information disclosure" "earnings announcement"`
- `"data asset information disclosure" "cumulative abnormal return"`

### 结果

未查到精确组合：`数据要素/数据资产/数据要素利用披露 -> PEAD60_abs / post-announcement absolute CAR / earnings-announcement residual adjustment`。

查到的邻近项主要是：

- 数据资源披露不一致与市场反应：事件研究，关注 2024 数据资源入表相关披露不一致的市场反应，不是年度 `DU_kw -> 盈余公告后 60 日绝对残余调整`。
- 数据资产入表现状与市场反应：用政策发布/实施日、数据资产入表文本和金额披露解释 CAR，但不是 earnings announcement 后的 PEAD/ACAR。
- 一般 PEAD 文献和年报文本/电话会议文本文献很多，但 X 不是数据要素披露。

### 判断

这是四个里面最干净的。写作时不要叫传统 signed PEAD，应叫 `post-announcement absolute CAR` 或 `post-announcement residual price adjustment`。

## 2. `ICOE_EY`

### 检索词

- `"数据资产信息披露" "权益资本成本"`
- `"数据要素利用" "权益资本成本"`
- `"data asset disclosure" "cost of equity"`
- `"digital transformation" "implied cost of equity"`

### 结果

已存在直接或高度近邻文献。

- 牛彪、于翔《数据资产获得投资者偏好了吗？——基于权益资本成本视角》：以 2007-2021 年 A 股为样本，检验数据资产与权益资本成本，主因变量就是权益资本成本；稳健性还使用 PEG/OJ。
- 《数据资产信息披露对企业价值创造的影响机制研究——企业盈余水平与权益资本成本双重视角》：明确将权益资本成本作为机制变量，且使用 MPEG/PEG 口径。
- 英文数字化转型文献也已有 `digital transformation -> cost of equity capital`。

### 判断

如果用 `ICOE_EY`，它只是权益资本成本家族的另一个代理，不足以形成主创新。可作为 `PEAD60_abs` 主线下的“估值风险溢价下降”机制。

## 3. `ICOE_PEG`

### 检索词

- `"数据资产" "MPEG" "权益资本成本"`
- `"数据资产信息披露" "MPEG"`
- `"数据资产信息披露" "Easton"`
- `"数据资产" "PEG" "权益资本成本"`

### 结果

撞得更近。已有文献不仅做权益资本成本，还明确用 MPEG/PEG/OJ 作为测度。

- 牛彪、于翔（2024）主测度为 MPEG，并用 OJ、PEG 做替代测度。
- 价值创造机制文献也使用 MPEG，并用 Easton PEG 公式做稳健性。

### 判断

`ICOE_PEG` 理论标签强，但 novelty 最弱。除非本文把 X 明确收窄为“数据要素利用披露/经营闭环披露”并把 Y 放在附加机制，否则容易被认为只是重复“数据资产 -> 权益资本成本”。

## 4. `CS_Spread`

### 检索词

- `"data asset disclosure" "bid-ask spread"`
- `"data asset disclosure" "stock liquidity"`
- `"数据资产披露" "股票流动性"`
- `"数据资产信息披露" "价差"`
- `"Does Data Asset Disclosure Affect Stock Liquidity"`

### 结果

股票流动性家族已有文献，但暂未查到 `Corwin-Schultz spread` 这个精确口径。

- 盛明泉、刘泽源《数据资产与企业资本市场表现：基于股票流动性的视角》：直接检验数据资产信息披露对股票流动性的影响，使用 Amihud 反向指标度量股票流动性。
- Korean Accounting Association 2025 会议手册列出论文 `Does Data Asset Disclosure Affect Stock Liquidity?: Evidence from China`。
- 数据资产披露与股价同步性文献也把信息不对称、长期机构投资者、融资约束等作为机制，和市场微观结构/流动性解释非常接近。

### 判断

`CS_Spread` 的精确变量可能还有一点空间，但“X -> liquidity”这条已经有人做。建议只放稳健性或交易摩擦补充，不要当主 Y。

## 推荐定位

1. 主 Y：`PEAD60_abs`。目前精确 X-Y 组合最干净。
2. 机制/补充：`ICOE_EY` + `ICOE_PEG`。不要声称首创，只说“估值风险溢价机制”。
3. 交易摩擦补充：`CS_Spread`。强调精确口径为 Corwin-Schultz spread，但承认股票流动性家族已有近邻。
4. 不推进：`AnnouncementAbsorptionRatio`，本地结果不显著，文献上也不如 `PEAD60_abs` 好解释。

## 主要证据链接

- Sun and Du (2024), `data assets disclosure -> stock price synchronicity`: https://www.sciencedirect.com/science/article/pii/S1059056024003289
- 牛彪、于翔（2024），`数据资产 -> 权益资本成本`: https://docs.static.szse.cn/www/aboutus/research/secuities/daily/W020240719367676655088.pdf
- 数据资产信息披露与企业价值创造，含 `MPEG/PEG` 权益资本成本机制: https://www.kjjb.org/article/2026/1001-7348/2026-43-6-001.htm
- 盛明泉、刘泽源（2025），`数据资产信息披露 -> 股票流动性`: https://www.ynufe.edu.cn/__local/5/60/9C/F0594D5F01F141597B0A58AC7B4_D5AA8830_107FE7.pdf
- 数据资产入表与市场反应，CAR 邻近但非 PEAD: https://m.fx361.com/news/2024/1206/25347574.html
