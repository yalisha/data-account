# 资产配置多 MV 诊断报告

## 1. 候选 MV 构造口径

数据源：
- `v1/data_parquet/balance_sheet.parquet`
- `v1/data_stata/reg_sample_iv_v16.dta`

执行口径：
- 只保留 `Typrep == "A"` 且 `Accper` 为年报年末 (`12-31`) 的资产负债表记录
- 按 `Stkcd + year` 与 lagged 主样本左连接
- 每个候选 MV 单独做 1% / 99% winsorize
- 缺失分项按现有项目脚本口径视为 `0`，分母 `A001000000` 为 `0` 时置缺

CSMAR 字段映射：

| 字段 | 含义 |
|:--|:--|
| `A001107000` | 交易性金融资产 |
| `A001202000` | 可供出售金融资产 |
| `A001211000` | 投资性房地产 |
| `A001229000` | 其他非流动金融资产 |
| `A001218000` | 无形资产净额 |
| `A001219000` | 开发支出 |
| `A001000000` | 资产总计 |

候选 MV：

| MV | 公式 | Step 2 预期方向 |
|:--|:--|:--|
| `FinRatio_narrow` | (`A001107000 + A001202000 + A001229000`) / `A001000000` | 正 |
| `FinRatio_trading` | `A001107000 / A001000000` | 正 |
| `FinRatio_realEstate` | `A001211000 / A001000000` | 正 |
| `FinRatio_other` | `A001229000 / A001000000` | 正（按执行假设） |
| `FinRatio_v4` | (`A001107000 + A001202000 + A001211000 + A001229000`) / `A001000000` | 正 |
| `IntangibleRatio` | `A001218000 / A001000000` | 负 |
| `DevOutlayRatio` | `A001219000 / A001000000` | 负 |
| `FinRatio_v4_delta` | `Δ FinRatio_v4` | 正 |
| `FinRatio_v4_vol3y` | `3y rolling sd(FinRatio_v4)` | 正（按执行假设） |
| `Fin_to_RD` | `FinRatio_v4 / (DevOutlayRatio + 0.001)` | 正 |

样本覆盖检查：
- `reg_sample_asset_mv.dta` 成功保存，合并后样本为 `43,735 x 56`
- 10 个候选在主样本中的当前期 non-missing 均超过 `25,000`
- `FinRatio_v4_vol3y` 覆盖最低，但仍有 `40,523` 个当前期观测、`34,004` 个 lagged 有效观测
- 新构造 `FinRatio_v4` 与现有 `FinAsset` 基本一致：`mean_abs_diff = 0.00002177`，`max_abs_diff = 0.00219787`

## 2. Step 2 批量诊断结果表

模型：

```stata
reghdfe PriceDelay MV_lag Size Lev ROA TobinQ Age Growth IndepRatio Dual Top1Share SOE CFO, ///
    absorb(Stkcd_num year_num) cluster(IndYear_num)
```

Gate：
- `pass`：`|t| >= 1.96` 且符号方向与预期一致
- `marginal`：`1.50 <= |t| < 1.96` 且方向一致
- `fail`：其余情况

| mv | expected_sign | coef | se | t | p | N | status |
|:--|:--|--:|--:|--:|--:|--:|:--|
| `FinRatio_narrow` | positive | -0.0173 | 0.0117 | -1.4743 | 0.1407 | 37294 | fail |
| `FinRatio_trading` | positive | -0.0139 | 0.0144 | -0.9654 | 0.3346 | 37294 | fail |
| `FinRatio_realEstate` | positive | 0.0894 | 0.0326 | 2.7394 | 0.0063 | 37294 | pass |
| `FinRatio_other` | positive | -0.0265 | 0.0528 | -0.5010 | 0.6165 | 37294 | fail |
| `FinRatio_v4` | positive | -0.0052 | 0.0104 | -0.4983 | 0.6184 | 37294 | fail |
| `IntangibleRatio` | negative | 0.0067 | 0.0236 | 0.2844 | 0.7762 | 37294 | fail |
| `DevOutlayRatio` | negative | 0.2254 | 0.2437 | 0.9247 | 0.3553 | 37294 | fail |
| `FinRatio_v4_delta` | positive | -0.0034 | 0.0120 | -0.2846 | 0.7760 | 37294 | fail |
| `FinRatio_v4_vol3y` | positive | -0.0455 | 0.0255 | -1.7845 | 0.0747 | 34004 | fail |
| `Fin_to_RD` | positive | -0.0000 | 0.0000 | -0.4355 | 0.6633 | 37294 | fail |

Step 2 结论：
- 10 个候选里只有 `FinRatio_realEstate` 过关
- 没有触发“Step 2 全 fail -> 机制路线死亡”
- 但“资产配置复活”的有效口径被收缩到**房地产金融化占比**这一条，其他宽口径、交易性口径、无形资产/开发支出口径都未通过

## 3. 过关候选的三步完整结果

仅对 `FinRatio_realEstate` 跑完整三步，分别对应 `DU_kw_lag` 与 `DU_llm_lag`。

### 3.1 `DU_kw_lag`

| 步骤 | 关键系数 | 结果 |
|:--|:--|:--|
| Step 1 | `FinRatio_realEstate = 0.0000601 * DU_kw_lag` (`t = 0.42`, `p = 0.673`) | fail |
| Step 2 | `PriceDelay = 0.0894213 * FinRatio_realEstate_lag` (`t = 2.74`, `p = 0.006`) | pass |
| Step 3 | `PriceDelay = -0.0038793 * DU_kw_lag + 0.0883561 * FinRatio_realEstate` | 两者均显著 |
| Step 3b | `PriceDelay = -0.0038728 * DU_kw_lag + 0.0893236 * FinRatio_realEstate_lag` | 两者均显著 |

### 3.2 `DU_llm_lag`

| 步骤 | 关键系数 | 结果 |
|:--|:--|:--|
| Step 1 | `FinRatio_realEstate = -0.0002033 * DU_llm_lag` (`t = -1.11`, `p = 0.267`) | fail |
| Step 2 | `PriceDelay = 0.0894213 * FinRatio_realEstate_lag` (`t = 2.74`, `p = 0.006`) | pass |
| Step 3 | `PriceDelay = -0.0045096 * DU_llm_lag + 0.0868453 * FinRatio_realEstate` | 两者均显著 |
| Step 3b | `PriceDelay = -0.0045053 * DU_llm_lag + 0.0880728 * FinRatio_realEstate_lag` | 两者均显著 |

三步诊断结论：
- `FinRatio_realEstate` 的 **M -> Y** 路径稳定成立
- 但 **DU -> M** 的 Step 1 对两个 DU 测度都不显著
- 因此它**通过了 Step 2 复活 gate，但没有形成完整江艇三步机制链**

## 4. 结论与建议

直接结论：
- 资产配置路线**没有死亡**
- 但也**没有按“完整中介机制”复活**

更具体地说：
- 如果只看 `MV_lag -> PriceDelay` 的筛选标准，`FinRatio_realEstate` 是唯一存活口径
- 如果按完整江艇三步标准看，关键的 `DU -> FinRatio_realEstate` 第一步不成立，因此不能把它写成强机制闭环

建议：
- 不建议把“宽口径 FinAsset”继续作为第四机制
- 若论文必须保留资产配置支线，最多保留 `FinRatio_realEstate` 这一条，且应降格为“房地产类金融化配置与价格延迟相关”的补充证据
- 若坚持严格机制叙事，则本轮结果不足以支持“DU 通过资产配置影响价格延迟”的完整因果链

## 5. OPEN_QUESTIONS

1. `balance_sheet.parquet` 缺少 `A001212000`，因此本轮无法补齐更完整的长期股权投资口径；若用户认为这一项是关键，需要先回到预处理层补字段后再重跑。
2. `spec` 同时写了“Step 2 只跑前 8 个”和“Step 2 输出必须 10 行”。本轮按 10 个全跑执行，并已满足输出 gate。
3. `FinRatio_other` 与 `FinRatio_v4_vol3y` 的理论方向在 `spec` 中写为“不确定”，但 Step 2 gate 需要方向判定；本轮按正向执行，结果两者均未过关，不影响主结论。
