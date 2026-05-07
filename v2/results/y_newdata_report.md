# 新数据候选 Y 构造与诊断报告

## OPEN_QUESTIONS

1. `PEAD` 事件按 `IAR_Rept.Accper` 的会计年度并入 firm-year 样本，公告日 `Annodt` 往往发生在下一自然年；本轮仍按主 H1 的 `L.DU` 口径回归。若要写进正文，需要决定是否额外做“按公告自然年合并”的敏感性检验。
2. `PEAD_signed` 依赖盈余意外方向，本轮两个 DU 测度下方向为正，不能支持“更少同向漂移”。真正稳的是 `PEAD_abs`，解释应写成“公告后残余价格调整幅度更小”，不要写成标准 signed PEAD 方向更弱。
3. 分析师预测类 Y 只覆盖有预测的 firm-year，回归样本约 20,729-23,998，明显小于主样本；这些结果更适合做机制/补充，不适合替代主 Y。
4. 本轮使用 `/Users/mac/computerscience/第三方资料/第三方数据资源/` 下本地 CSMAR Excel/zip，只新建输出文件，未改现有 parquet、outline、正文、Zotero、Notion 或 Linear。

## 1. 已构造的新 Y

| 家族 | Y | 构造口径 | 样本覆盖 |
|---|---|---|---:|
| 流动性 / 交易摩擦 | `CS_Spread` | `TRD_Dalyr.Hiprc/Loprc` 的 Corwin-Schultz 高低价隐含价差，年均 | 43,735 |
| 流动性 / 波动范围 | `HL_Range_year` | 年均 `ln(Hiprc/Loprc)` | 43,735 |
| 公告后漂移 | `PEAD20_signed`, `PEAD60_signed` | `sign(SUE_price) * CAR_abnormal[+2,+20/+60]` | 28,534-28,536 |
| 公告后残余调整 | `PEAD20_abs`, `PEAD60_abs` | `abs(CAR_abnormal[+2,+20/+60])` | 43,706-43,728 |
| 公告即时反应 | `EA_CAR02`, `EA_absCAR02` | 公告后 `[0,+2]` 异常收益及绝对值 | 43,734 |
| 盈余意外 / 分析师误差 | `SUE_price`, `SUE_abs_price`, `AF_Error_abs` | 实际 EPS 与公告前一年内一致预测 EPS 的差，按公告前收盘价缩放 | 28,538 |
| 分析师信息环境 | `AF_Dispersion`, `AF_Coverage` | 公告前一致预测离散度、预测覆盖度 | 24,958 / 28,538 |
| 分析师预测偏差 | `AF_Optimism`, `Bench_Accuracy`, `Bench_Optimism`, `Bench_BuyShare` | `AF_Forecast` / `AF_Bench` 汇总 | 28,821-28,973 |
| 官方崩盘风险 | `NCSKEW_off`, `DUVOL_off`, `CRASH_off` | CSMAR `BF_CRASHRISK` 年度指标 | 43,514 |

## 2. 回归口径

- 样本：`reg_sample_y_newdata.dta`
- 解释变量：`L.DU_kw`、`L.DU_llm`
- 固定效应：firm FE + year FE
- 聚类：`cluster(IndYear_num)`
- 控制变量：`Size Lev ROA TobinQ Age Growth IndepRatio Dual Top1Share SOE CFO`
- gate：方向符合预期且 `|t| >= 1.96` 为 `pass`，`1.50 <= |t| < 1.96` 为 `marginal`

## 3. 核心结果

| Y | `DU_kw` | `DU_llm` | 判断 |
|---|---:|---:|---|
| `CS_Spread` | -0.000037, t=-1.945, p=0.052 | -0.000093, t=-6.122, p<0.001 | **可补充**：一个 marginal，一个强 pass |
| `PEAD20_abs` | -0.001705, t=-2.833, p=0.005 | -0.001480, t=-1.948, p=0.052 | 可补充，但弱于 60 日 |
| `PEAD60_abs` | -0.003667, t=-2.554, p=0.011 | -0.003698, t=-2.887, p=0.004 | **本轮最强新 Y** |
| `EA_absCAR02` | -0.000968, t=-2.282, p=0.023 | -0.000807, t=-1.857, p=0.064 | 可作事件反应幅度补充 |
| `SUE_abs_price` / `AF_Error_abs` | -0.000940, t=-4.338, p<0.001 | -0.000510, t=-1.793, p=0.073 | 更像分析师信息环境机制 |
| `AF_Coverage` | 0.013122, t=1.777, p=0.076 | 0.012522, t=1.670, p=0.095 | 方向一致但仅 marginal |
| `NCSKEW_off` | -0.013868, t=-2.454, p=0.014 | 0.014183, t=2.190, p=0.029 | 仍是单边，不能主推 |
| `DUVOL_off` | -0.007408, t=-2.115, p=0.035 | 0.010847, t=2.635, p=0.009 | 仍是单边，不能主推 |
| `CRASH_off` | 0.000714, t=0.446, p=0.656 | 0.006146, t=2.486, p=0.013 | fail |

完整逐项结果见 `y_newdata_step1.csv`。

## 4. 结论

三选一建议从上一轮的“保持”调整为：**补充，不替代**。

理由：

- `PriceDelay` 仍是最干净的主 Y，因为它和文章主线“价格吸收速度”直接对应，且 baseline 在两个 DU 测度下都强显著。
- 新数据已经能补上一个真正差异化的微观结构 Y：`CS_Spread`。它与 `SYNCH` 家族不撞，`DU_llm` 很强，`DU_kw` 接近 5%。
- 最有增量的是 `PEAD60_abs`：两个 DU 测度都显著为负，说明数据要素利用披露越强，公告后 60 个交易日的残余异常调整幅度越小。这比年度均值型 Y 更贴近“信息更快进入价格”的机制。
- `PEAD_signed` 不支持，因此写作时不要说“漂移方向更弱”；应写“公告后残余价格调整的绝对幅度更小”。

建议写作位置：

1. 主文仍保留 `PriceDelay`。
2. 稳健性或补充结果新增 `CS_Spread`。
3. 机制/进一步结果新增 `PEAD60_abs`，可搭配 `PEAD20_abs`。
4. 分析师类指标只作解释性补充：`SUE_abs_price` / `AF_Error_abs` 和 `AF_Coverage`。

## 5. 产出文件

| 文件 | 内容 |
|---|---|
| `/Users/mac/computerscience/0做完了/15会计研究/v2/stata_do/build_y_newdata.py` | 新数据 Y 构造脚本 |
| `/Users/mac/computerscience/0做完了/15会计研究/v2/stata_do/y_newdata_diagnose.do` | 新数据 Y 回归诊断脚本 |
| `/Users/mac/computerscience/0做完了/15会计研究/v1/data_stata/reg_sample_y_newdata.dta` | 合并后的 firm-year 样本 |
| `/Users/mac/computerscience/0做完了/15会计研究/v2/results/y_newdata_build_summary.csv` | 新 Y 覆盖与描述统计 |
| `/Users/mac/computerscience/0做完了/15会计研究/v2/results/y_newdata_event_level.csv` | 公告事件层面的 PEAD / SUE 中间结果 |
| `/Users/mac/computerscience/0做完了/15会计研究/v2/results/y_newdata_step1.csv` | 批量回归结果 |
| `/Users/mac/computerscience/0做完了/15会计研究/v2/results/y_newdata_diagnose.log` | Stata 日志 |
