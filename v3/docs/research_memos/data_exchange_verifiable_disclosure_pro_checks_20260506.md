# Pro 反馈后的新增检验：可验证数据要素披露

日期：2026-05-06

## 1. 本轮检验目的

外部 Pro 反馈认为当前题目值得推进，但要求把“文本披露增加”进一步证明为：

```text
更制度化、更可核验、更具体的数据要素披露
```

本轮新增三类检验：

1. `City-Year FE + PreWash × Year FE`，做更强城市内识别。
2. 比例型可核验披露 Y，避免只是“数据文本总量增加”。
3. 单独处理 2024 年企业数据资源会计规则。

## 2. 新增脚本和输出

脚本：

```text
src/stata/did_data_exchange_verifiable_disclosure_pro_v1.do
```

输出：

```text
results/stata/did_data_exchange_verifiable_disclosure_pro_v1.csv
results/stata/did_data_exchange_verifiable_disclosure_pro_v1.dta
results/logs/did_data_exchange_verifiable_disclosure_pro_v1_mcp.log
```

## 3. 新增 Y

除原始披露强度外，新增比例型/相对型 Y：

```text
asset_trade_ratio = asset_trade_kw / (DU_kw + 1)
strict_at_ratio   = strict_at_kw / (DU_kw + 1)
asset_trade_share = asset_trade_kw / (asset_trade_kw + DU_kw + 1)
strict_at_share   = strict_at_kw / (strict_at_kw + DU_kw + 1)
```

这些变量试图回答：

```text
在广义数据要素叙事中，数据资产/数据交易等更制度化表达的占比是否上升？
```

## 4. 城市内识别：City-Year FE + PreWash × Year FE

规格：

```text
Y_ict =
    beta DID_city_ct × PreWash_i
    + Firm FE
    + City-Year FE
    + PreWash_i × Year FE
    + Controls_it
    + error_ict
```

其中 `DID_city` 主效应被 `City-Year FE` 吸收。

### 4.1 PreWashAny1617

| Y | coef | t | p |
|---|---:|---:|---:|
| `DU_kw` | 0.0546 | 0.77 | 0.443 |
| `asset_trade_kw` | 0.0246 | 1.65 | 0.101 |
| `strict_at_kw` | 0.0121 | 2.39 | 0.018 |
| `asset_trade_ratio` | 0.0019 | 1.33 | 0.185 |
| `asset_trade_share` | 0.0014 | 1.26 | 0.208 |

判断：

```text
最强城市内识别下，宽口径 PreWashAny 的 DU_kw 不保留；
asset_trade_kw 方向保留但只到 10% 边际；
strict_at_kw 保留 5% 显著。
```

### 4.2 PreStrictWashAny1617

| Y | coef | t | p |
|---|---:|---:|---:|
| `DU_kw` | 0.1523 | 1.55 | 0.122 |
| `asset_trade_kw` | 0.0496 | 2.09 | 0.038 |
| `strict_at_kw` | 0.0211 | 2.83 | 0.005 |
| `asset_trade_ratio` | 0.0038 | 1.74 | 0.084 |
| `asset_trade_share` | 0.0029 | 1.76 | 0.080 |
| `strict_at_ratio` | 0.0014 | 1.91 | 0.057 |
| `strict_at_share` | 0.0011 | 1.81 | 0.071 |

判断：

```text
严格预先漂数据暴露下，城市内识别结果更好。
asset_trade_kw 和 strict_at_kw 能保住，比例型 Y 方向一致但主要是 10% 边际。
```

这说明论文更稳的识别表述可能是：

```text
数据交易所主要促使“严格事前 data-washing 暴露企业”
增加更制度化、更可核验的数据资产/交易披露。
```

而不是笼统说所有预先 data-washing 企业都显著增加全部数据披露。

## 5. 2024 会计规则检查

### 5.1 剔除 2024 年全部观测

| exposure | Y | coef | t | p |
|---|---|---:|---:|---:|
| `dx_wany17` | `DU_kw` | 0.1573 | 2.41 | 0.016 |
| `dx_wany17` | `asset_trade_kw` | 0.0411 | 2.70 | 0.007 |
| `dx_wany17` | `strict_at_kw` | 0.0169 | 2.84 | 0.005 |
| `dx_wany17` | `asset_trade_ratio` | 0.0039 | 2.65 | 0.009 |
| `dx_wany17` | `asset_trade_share` | 0.0031 | 2.88 | 0.004 |
| `dx_wstr17` | `DU_kw` | 0.3401 | 5.12 | 0.000 |
| `dx_wstr17` | `asset_trade_kw` | 0.0879 | 3.82 | 0.000 |
| `dx_wstr17` | `strict_at_kw` | 0.0326 | 4.13 | 0.000 |
| `dx_wstr17` | `asset_trade_ratio` | 0.0082 | 3.95 | 0.000 |
| `dx_wstr17` | `asset_trade_share` | 0.0066 | 4.54 | 0.000 |

判断：

```text
剔除 2024 年后结果不但没有消失，反而相当稳。
全国 2024 数据资源会计规则不是主结果的唯一来源。
```

### 5.2 加 PreWash × 2024

| exposure | Y | coef | t | p |
|---|---|---:|---:|---:|
| `dx_wany17` | `DU_kw` | 0.1305 | 2.07 | 0.040 |
| `dx_wany17` | `asset_trade_kw` | 0.0398 | 2.54 | 0.012 |
| `dx_wany17` | `strict_at_kw` | 0.0161 | 2.76 | 0.006 |
| `dx_wstr17` | `DU_kw` | 0.2986 | 4.06 | 0.000 |
| `dx_wstr17` | `asset_trade_kw` | 0.0841 | 3.42 | 0.001 |
| `dx_wstr17` | `strict_at_kw` | 0.0309 | 3.73 | 0.000 |

判断：

```text
显式控制预先 washing 企业在 2024 年的全国性变化后，主结果仍保留。
```

### 5.3 Industry × 2024 FE

| exposure | Y | coef | t | p |
|---|---|---:|---:|---:|
| `dx_wany17` | `DU_kw` | 0.1129 | 1.87 | 0.063 |
| `dx_wany17` | `asset_trade_kw` | 0.0360 | 2.25 | 0.025 |
| `dx_wany17` | `strict_at_kw` | 0.0142 | 2.57 | 0.011 |
| `dx_wany17` | `asset_trade_ratio` | 0.0034 | 2.21 | 0.027 |
| `dx_wstr17` | `DU_kw` | 0.2805 | 3.91 | 0.000 |
| `dx_wstr17` | `asset_trade_kw` | 0.0793 | 3.24 | 0.001 |
| `dx_wstr17` | `strict_at_kw` | 0.0281 | 3.63 | 0.000 |
| `dx_wstr17` | `asset_trade_ratio` | 0.0074 | 3.29 | 0.001 |

判断：

```text
吸收行业层面的 2024 会计规则冲击后，数据资产/交易表达仍较稳。
```

## 6. 当前判断是否升级

Pro 提出的三项要求跑完后，当前判断可以从：

```text
有趣但需要补识别
```

升级为：

```text
值得继续推进的黄绿色主线。
```

但还不能完全绿灯。

### 支持推进的证据

1. 2024 会计规则不是结果的唯一来源：drop 2024、PreWash × 2024、Industry × 2024 FE 都能保住。
2. 比例型 Y 在 baseline、drop 2024、Industry × 2024 FE 下方向一致且显著。
3. 城市内识别下，严格预先漂数据暴露企业的 `asset_trade_kw` 和 `strict_at_kw` 仍显著。

### 仍需谨慎的地方

1. `City-Year FE + PreWash × Year FE` 下，`PreWashAny1617 -> DU_kw` 不显著。
2. 比例型 Y 在最强城市内识别下主要是 10% 边际。
3. `strict_at_kw` 虽然城市内识别更好，但之前 event pretrend 边际不过，不能单独当主 Y。

## 7. 写作口径调整

不建议写：

```text
数据交易所普遍提高企业数据披露
数据交易所治理 data washing
数据交易所提升企业真实数据能力
```

建议写：

```text
数据交易所设立后，事前 data-washing 暴露较强的企业，
尤其是严格错配企业，更倾向于增加与数据资产、数据交易、数据流通、
数据产品等制度化场景相关的可核验披露。
```

主文最稳结构：

```text
主结果 1：asset_trade_kw
主结果 2：DU_kw
补充：strict_at_kw
比例型 Y：asset_trade_ratio / asset_trade_share
辅助：NarrHardGap / NarrHardWashing 收敛
```

如果要更保守：

```text
把 asset_trade_kw 放在第一主 Y，DU_kw 作为宽口径补充。
```

这比原来 “DU_kw + asset_trade_kw 并列主 Y” 更符合 Pro 的意见，也更贴近数据交易所制度功能。
