# 数据资产入表验证端：2024 横截面试跑

日期：2026-05-02

## 1. 执行状态

已跑通，回归由 Stata MCP 执行。

Python 只用于把 parquet 中的 `DataAsset` 抽成 Stata 可读 CSV：

```text
/Users/mac/computerscience/0做完了/15会计研究/v3/src/python/extract_data_asset_for_stata.py
/Users/mac/computerscience/0做完了/15会计研究/v3/results/data/data_asset_from_panel.csv
```

Stata 主脚本：

```text
/Users/mac/computerscience/0做完了/15会计研究/v3/src/stata/data_asset_entry_mismatch_2024.do
```

输出：

```text
/Users/mac/computerscience/0做完了/15会计研究/v3/results/data/data_asset_entry_mismatch_2024_panel.dta
/Users/mac/computerscience/0做完了/15会计研究/v3/results/stata/data_asset_entry_x_summary_2024.csv
/Users/mac/computerscience/0做完了/15会计研究/v3/results/stata/data_asset_entry_y_screen_2024.csv
/Users/mac/computerscience/0做完了/15会计研究/v3/results/logs/data_asset_entry_mismatch_2024_mcp.log
```

模型是 2024 横截面：

```text
Y_2024 = X_2024 + controls + industry FE, robust SE
```

控制变量：

```text
Size Lev ROA TobinQ Age Growth IndepRatio Dual Top1Share SOE CFO
```

当 Y 本身是 `TobinQ` 时，控制变量中自动移除 `TobinQ`。

## 2. 构造的 X

会计验证端：

- `BookEntry`：2024 年 `DataAsset > 0`。
- `DataAsset_ln`：`ln(1 + DataAsset)`。

叙事端：

- 用 `DU_kw` 在 2024 年横截面中做分位数，得到 `disc_pct_2024`。

错配端：

- `NarrativeBookGap_z`：`z(DU_kw) - z(DataAsset_ln)`。
- `NarrativeBookGap_pct`：`rank(DU_kw) - rank(DataAsset_ln)` 的分位差。
- `TalkNoBook`：高数据叙事、未入表。
- `VerifiedBook`：高数据叙事、已入表。
- `BookNoTalk`：已入表、低数据叙事。

## 3. 样本量

当前 `reg_sample_v18` 中 2024 年可用样本是 4,598 个 firm-year。

| X | N | mean | positive obs |
|---|---:|---:|---:|
| `BookEntry` | 4,598 | 0.0048 | 22 |
| `TalkNoBook` | 4,598 | 0.4972 | 2,286 |
| `TalkNoBook_strict` | 4,598 | 0.2495 | 1,147 |
| `VerifiedBook` | 4,598 | 0.0028 | 13 |
| `VerifiedBook_strict` | 4,598 | 0.0007 | 3 |
| `BookNoTalk` | 4,598 | 0.0020 | 9 |
| `BookNoTalk_strict` | 4,598 | 0.0013 | 6 |

关键判断：

- `BookEntry` 正样本只有 22 个。
- `VerifiedBook` 只有 13 个，`BookNoTalk` 只有 9 个。
- strict 版本正样本太少，不能解释。
- `TalkNoBook` 样本充足，但因为入表企业极少，它基本接近“高数据叙事且绝大多数未入表”，不能单独证明“入表缺位”。

## 4. 主要试跑结果

下面只列有解释价值的结果。完整结果见：

```text
/Users/mac/computerscience/0做完了/15会计研究/v3/results/stata/data_asset_entry_y_screen_2024.csv
```

### 4.1 `TalkNoBook`

| Y | coef | t | p | N | positive obs |
|---|---:|---:|---:|---:|---:|
| `ForecastDisp` | 0.0313 | 2.70 | 0.007 | 2,024 | 953 |
| `ReportFreq` | -0.1159 | -3.02 | 0.003 | 2,625 | 1,262 |
| `SA` | 0.0037 | 2.09 | 0.037 | 4,593 | 2,286 |
| `SCConc` | 3.5420 | 7.22 | <0.001 | 4,571 | 2,276 |
| `SuppConc` | 3.8900 | 6.63 | <0.001 | 4,565 | 2,272 |

严格版：

| Y | coef | t | p | N | positive obs |
|---|---:|---:|---:|---:|---:|
| `Analyst` | -0.1083 | -2.43 | 0.015 | 4,593 | 1,147 |
| `ReportFreq` | -0.1433 | -3.34 | 0.001 | 2,625 | 618 |
| `AuditFee` | 0.0335 | 2.17 | 0.030 | 4,590 | 1,144 |
| `SCConc` | 2.8525 | 5.19 | <0.001 | 4,571 | 1,143 |
| `SuppConc` | 3.5456 | 5.37 | <0.001 | 4,565 | 1,142 |

初步读法：

高数据叙事但未入表的企业，分析师覆盖和研报频率更低、预测分歧更高、审计费用更高、供应链集中度更高。这个方向比较像“叙事强但会计验证不足，外部中介识别困难/风险更高”。

但注意：因为 2024 已入表企业太少，`TalkNoBook` 不是一个干净的“未入表惩罚”变量，更像“高数据叙事但尚未获得会计确认”的粗代理。

### 4.2 `NarrativeBookGap_z`

| Y | coef | t | p | N |
|---|---:|---:|---:|---:|
| `RatingDisp` | 0.0044 | 1.99 | 0.047 | 2,550 |
| `InvestIneff` | -0.0024 | -2.93 | 0.003 | 4,097 |
| `InstStable` | -0.1648 | -2.25 | 0.025 | 4,591 |
| `CashFlowVol` | -0.0007 | -2.03 | 0.043 | 4,593 |
| `SuppConc` | -1.0197 | -2.05 | 0.040 | 4,565 |

这个结果方向比较杂，不如 `TalkNoBook` 好讲。`NarrativeBookGap_pct` 不建议主用，因为 `DataAsset=0` 的企业太多，分位差很容易退化成叙事分位本身。

### 4.3 `VerifiedBook` / `BookNoTalk`

`VerifiedBook -> TobinQ` 为负，系数 -0.237，t=-2.05，但正样本只有 13 个。

`BookNoTalk -> ForecastDisp` 为负，t=-3.76，但该回归里正样本只有 3 个。

这些结果不能解释，最多说明变量能生成、能跑。strict 版本更不能看，`VerifiedBook_strict` 只有 3 个正样本，有些 Y 的正样本甚至只有 1 个。

## 5. 当前判断

这条路“能构造”，但当前数据不能把 `DataAsset` 单独做成主变量。

可以保留的变量：

1. `TalkNoBook`
2. `TalkNoBook_strict`
3. `NarrativeBookGap_z`

暂时不建议解释的变量：

1. `VerifiedBook`
2. `VerifiedBook_strict`
3. `BookNoTalk`
4. `BookNoTalk_strict`
5. `AnyBook`

原因很简单：已入表正样本太少。

## 6. 推荐下一步

不要写：

```text
数据资产入表 -> Y
```

可以写成：

```text
年报数据要素叙事强，但缺乏会计验证 -> 信息中介识别困难 / 风险溢价 / 审计成本
```

更稳的下一步是把 `DataAsset` 作为 hard capability index 的一部分，而不是唯一能力端：

```text
HardDataCapability = 数据资产入表 + 数据/AI专利 + 软件著作权 + 数据岗位招聘 + 数据投资
DataNarrativeGap = z(annual-report data narrative) - z(HardDataCapability)
```

如果只用现有数据，现在最值得继续打磨的是：

```text
TalkNoBook -> ForecastDisp / ReportFreq / Analyst / AuditFee
```

这比 `TobinQ` 更像“信息一致性/可验证性”的后果，也比经营韧性和定价效率更不拥挤。
