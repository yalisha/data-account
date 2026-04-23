# Codex Spec v2：v16 整合版（替代前版"质量交互"spec）

**更新日期**：2026-04-19
**替代**：前版 `codex_spec.md` 已失效（"质量交互"路线 null，放弃）

---

## 0. 任务总览

前版 spec 跑的 `DU_kw × Quality` 交互全部 null。但 Claude 挖出 `v15_analysis_sample.parquet` 发现 **v15 measurement 已用 direct + joint + lagged spec 跑出显著结果**。本版任务：**在统一样本（reg_sample_v18 口径，N=43,735）上复现 v15 measurement 的显著结果**，让所有实证可以在一张表上整合。

**不要做**：
- 不要跑交互（interaction）
- 不要构造新变量（所有变量都在 `v15_analysis_sample.parquet` 已有）
- 不要动 `data_parquet/` 或 `data_stata/reg_sample_v18.dta`
- 不要动前版产物 `reg_sample_v16_quality.dta`（作为历史归档保留）
- 不要改 `manuscript/` 任何文件
- 不要 git commit

**做**：
- 统一样本口径下，跑 direct / joint / lagged 回归
- 结果入 `results/v16_integrated/`
- 所有 reghdfe 规格：Firm+Year FE，cluster(IndYear)，11 控制变量

---

## 1. 输入（绝对路径）

### 主样本（首选）
`/Users/mac/computerscience/0做完了/15会计研究/results/v15_measurement/v15_analysis_sample.parquet`
- 43,735 obs × 116 cols
- 已含所有需要变量：
  - 主测度：`DU_kw, DU_llm, DU_kw_lag`
  - 价值链：`DU_stock, DU_dev, DU_app, DU_value, DU_gov` + `_lag`
  - 质量：`DUclosedloop, DUcore, DUchain_count, DUkw_mda, DUbalance, DUevent, DUevent_ratio, GenericNarr` + `_lag`
  - 下游：`ForecastDisp, CashFlowVol, SCConc`
  - 异质性：`Post2020, DigEconCore, HighTech, SA, InstHold`
  - 控制：`Size, Lev, ROA, TobinQ, Age, Growth, IndepRatio, Dual, Top1Share, SOE, CFO`
  - FE：`Stkcd_num, year_num, IndYear_num, Ind2_num`

### DU_llm_lenstd（需构造，**按 v14 历史公式**）
**2026-04-19 晚 patch**：前版 spec 写错，需改为 v14 原公式（出处：`scripts/run_v14_analysis.py:174`）：
```python
DU_llm_lenstd = np.log1p(DU_kw.clip(lower=0)) * (llm_score / 3.0)
DU_llm_lenstd_z = z-score(DU_llm_lenstd)   # 用整体样本均值/标准差
DU_llm_lenstd_lag = group_by(Stkcd).shift(1)
```
这是历史上 `DU_llm_lenstd_lag = -0.0184***, t=-5.80` 的同一变量（见 `results/v14/construct_boundary_results.json`）。

**⚠ 诚实警告**：此变量和 DU_kw 代数相关（都是 kw_per10k 的函数），joint regression 里 "subsume DU_kw" 的结果部分来自代数耦合——这不是 clean moderation。**README 对比部分必须明确此点**，不要把它说成"语义深度独立于关键词数量"。

### WashGap（需构造，**按 v14 历史公式**）
**2026-04-19 晚 patch**：前版 spec 写错，需改为 v14 原公式（出处：`scripts/run_v14_analysis.py:186`）：
```python
DU_kw_z = z-score(DU_kw)
DU_llm_lenstd_z = z-score(DU_llm_lenstd)
WashGap = winsorize(DU_kw_z - DU_llm_lenstd_z, 1%, 99%)
WashGap_lag = group_by(Stkcd).shift(1)
```
这是历史上 `WashGap_lag = +0.0035**` 的同一变量，概念是"广度-深度 z-score 差"（对应 ChatGPT Pro 叙事里"广而浅"）。

**⚠ 注意**：前版 Codex 在 `reg_sample_v16_quality.dta` 里构造的 WashGap 用的是不同公式（洗稿/实质 比例），与 v14 不是同一变量。**不要读前版产物的 WashGap**，按 v14 公式重构。

### 辅助样本（参考用）
`/Users/mac/computerscience/0做完了/15会计研究/data_stata/reg_sample_v18.dta` — 作为 v18 样本口径对齐参考。

---

## 2. 产出（必须创建）

```
results/v16_integrated/
├── h2a_value_chain.csv        # 5 维分解单独 lagged
├── h2b_quality_direct.csv     # 5 个质量变量单独 lagged
├── h2b_quality_joint.csv      # 质量 + DU_kw 联合（6 个规格）
├── h2c_washgap.csv            # WashGap 单独 + 联合 DU_kw
├── h3_downstream.csv          # 下游机制统一口径（确认已有 v18 结果）
├── sample_summary.csv         # 样本口径、N、控制变量清单
└── logs/                      # Stata log

scripts/v16_integrated/
├── build_integrated_sample.py  # 构造 DU_llm_lenstd + WashGap，合并到 v15_analysis_sample.parquet
├── h2a_value_chain.do
├── h2b_quality.do
├── h2c_washgap.do
└── h3_downstream.do            # 可选：如 v18 结果口径一致可跳过

data_stata/
└── reg_sample_v16_integrated.dta   # 新统一样本

v2/
├── outline.md           # Claude 已写，不动
├── codex_spec.md        # 本文件，不动
├── OPEN_QUESTIONS.md    # 按需更新
└── README.md            # 完工后写
```

---

## 3. 任务 1：构造统一样本

**脚本**：`scripts/v16_integrated/build_integrated_sample.py`

```python
import pandas as pd
import numpy as np
from scipy.stats.mstats import winsorize

EPS = 1e-6

# 1. 读 v15_analysis_sample（主样本，已含大部分变量）
df = pd.read_parquet("results/v15_measurement/v15_analysis_sample.parquet")
df = df.sort_values(['Stkcd','year']).reset_index(drop=True)

# 2. 构造 DU_llm_lenstd（v14 原公式，出处 run_v14_analysis.py:174）
df['DU_llm_lenstd'] = np.log1p(df['DU_kw'].clip(lower=0)) * (df['llm_score'] / 3.0)

# 3. Z-score（整体样本口径，和 v14 一致）
def zscore(s):
    return (s - s.mean()) / s.std(ddof=0)
df['DU_kw_z'] = zscore(df['DU_kw'])
df['DU_llm_lenstd_z'] = zscore(df['DU_llm_lenstd'])

# 4. 构造 WashGap（v14 原公式，出处 run_v14_analysis.py:186）
wg_raw = df['DU_kw_z'] - df['DU_llm_lenstd_z']
# winsorize 1% / 99%
lo, hi = wg_raw.quantile([0.01, 0.99])
df['WashGap'] = wg_raw.clip(lo, hi)

# 5. 构造 DU_llm_lag（spec §5.2 joint 要用）
df['DU_llm_lag'] = df.groupby('Stkcd')['DU_llm'].shift(1)

# 6. Lag 版本
for col in ['DU_llm_lenstd', 'WashGap']:
    df[f'{col}_lag'] = df.groupby('Stkcd')[col].shift(1)

# 7. 保存
df.to_stata("data_stata/reg_sample_v16_integrated.dta", write_index=False, version=118)

# 8. Smoke tests
assert df.shape[0] == 43735, f"样本量 {df.shape[0]} != 43,735"
assert df['DU_llm_lenstd'].notna().mean() > 0.95
assert df['WashGap'].notna().mean() > 0.95
assert df['DU_llm_lenstd_lag'].notna().mean() > 0.85
assert df['WashGap_lag'].notna().mean() > 0.85
assert df['DU_llm_lag'].notna().mean() > 0.85

# 9. 和 v14 历史值快速对比（print 即可）
print(f"DU_llm_lenstd mean={df['DU_llm_lenstd'].mean():.4f}, std={df['DU_llm_lenstd'].std():.4f}")
print(f"WashGap mean={df['WashGap'].mean():.4f}, std={df['WashGap'].std():.4f}")
print("OK: reg_sample_v16_integrated.dta 构造完毕（v14 公式）")
```

**Gate**：
- 所有 assert 通过
- 产出 dta 含 118+ 列（原 116 + 新增 DU_llm_lenstd, DU_llm_lenstd_lag, WashGap, WashGap_lag, placebo_count）

---

## 4. 任务 2：H2a 价值链五维分解

**脚本**：`scripts/v16_integrated/h2a_value_chain.do`

```stata
use "data_stata/reg_sample_v16_integrated.dta", clear

local controls "Size Lev ROA TobinQ Age Growth IndepRatio Dual Top1Share SOE CFO"

foreach V in DU_stock DU_dev DU_app DU_value DU_gov {
    reghdfe PriceDelay `V'_lag `controls', abs(Stkcd_num year_num) cluster(IndYear_num)
    est store m_`V'
}

esttab m_* using "results/v16_integrated/h2a_value_chain.csv", replace ///
    b(%9.4f) se(%9.4f) stats(N r2) csv nonumbers ///
    keep(DU_stock_lag DU_dev_lag DU_app_lag DU_value_lag DU_gov_lag) label
```

**预期**：`DU_stock_lag, DU_dev_lag, DU_app_lag` 显著负（参考 v15 measurement：-0.0124***/-0.0048***/-0.0116***），`DU_value_lag, DU_gov_lag` 可能不显著（披露稀疏）。

**Gate**：
- 三个主维度 |t| > 2 且符号为负 → 通过
- 若统一口径后有维度翻符号或失显著：**停下来报告**，不要粉饰

---

## 5. 任务 3：H2b 质量变量 direct + joint

**脚本**：`scripts/v16_integrated/h2b_quality.do`

### 5.1 Direct（5 个质量变量各自单独）
```stata
use "data_stata/reg_sample_v16_integrated.dta", clear

local controls "Size Lev ROA TobinQ Age Growth IndepRatio Dual Top1Share SOE CFO"

foreach Q in DUclosedloop DUcore DUchain_count DUkw_mda DU_llm_lenstd {
    reghdfe PriceDelay `Q'_lag `controls', abs(Stkcd_num year_num) cluster(IndYear_num)
    est store d_`Q'
}

esttab d_* using "results/v16_integrated/h2b_quality_direct.csv", replace ///
    b(%9.4f) se(%9.4f) stats(N r2) csv nonumbers ///
    keep(DUclosedloop_lag DUcore_lag DUchain_count_lag DUkw_mda_lag DU_llm_lenstd_lag) label
```

### 5.2 Joint（每个质量变量 + DU_kw 同时入方程）
```stata
foreach Q in DUclosedloop DUcore DUchain_count DUkw_mda DU_llm_lenstd {
    reghdfe PriceDelay DU_kw_lag `Q'_lag `controls', abs(Stkcd_num year_num) cluster(IndYear_num)
    est store j_`Q'
}

* DU_llm 与 DU_llm_lenstd 联合（看是否共存）
gen DU_llm_lag = L.DU_llm  // 若不存在 DU_llm_lag
reghdfe PriceDelay DU_llm_lag DU_llm_lenstd_lag `controls', abs(Stkcd_num year_num) cluster(IndYear_num)
est store j_DUllm_lenstd

esttab j_* using "results/v16_integrated/h2b_quality_joint.csv", replace ///
    b(%9.4f) se(%9.4f) stats(N r2) csv nonumbers ///
    keep(DU_kw_lag DU_llm_lag DUclosedloop_lag DUcore_lag DUchain_count_lag DUkw_mda_lag DU_llm_lenstd_lag) label
```

**核心 finding 预期**（来自 v15 measurement）：
- **Direct**：5 个质量变量全部显著负
- **Joint**：`DU_llm_lenstd_lag + DU_kw_lag` 规格中，DU_kw 变不显著（t=-0.46~-1.29 左右），DU_llm_lenstd 保持显著（t≈-5.75）→ 语义深度 **subsume** 数量

**Gate**：
- Direct：至少 4/5 显著 |t|>2 → 通过
- Joint 里 DU_llm_lenstd 保持显著 |t|>3 → 通过（这是本文核心创新卖点）
- 若 Direct 主要变量失显著：停下来报告

---

## 6. 任务 4：H2c WashGap

**脚本**：`scripts/v16_integrated/h2c_washgap.do`

```stata
use "data_stata/reg_sample_v16_integrated.dta", clear
local controls "Size Lev ROA TobinQ Age Growth IndepRatio Dual Top1Share SOE CFO"

* Direct
reghdfe PriceDelay WashGap_lag `controls', abs(Stkcd_num year_num) cluster(IndYear_num)
est store w_direct

* Joint DU_kw
reghdfe PriceDelay DU_kw_lag WashGap_lag `controls', abs(Stkcd_num year_num) cluster(IndYear_num)
est store w_joint_kw

* Joint DU_llm
reghdfe PriceDelay DU_llm_lag WashGap_lag `controls', abs(Stkcd_num year_num) cluster(IndYear_num)
est store w_joint_llm

esttab w_* using "results/v16_integrated/h2c_washgap.csv", replace ///
    b(%9.4f) se(%9.4f) stats(N r2) csv nonumbers ///
    keep(DU_kw_lag DU_llm_lag WashGap_lag) label
```

**预期**（v15 measurement：WashGap_lag = +0.0035**）：direct 正向显著，联合 DU_kw 后仍正向显著。

---

## 7. 任务 5：H3 下游机制口径确认

**脚本**：`scripts/v16_integrated/h3_downstream.do`

```stata
use "data_stata/reg_sample_v16_integrated.dta", clear
local controls "Size Lev ROA TobinQ Age Growth IndepRatio Dual Top1Share SOE CFO"

* ForecastDisp（分析师预测分歧）
reghdfe ForecastDisp DU_kw `controls', abs(Stkcd_num year_num) cluster(IndYear_num)
est store m_fd_kw
reghdfe ForecastDisp DU_llm `controls', abs(Stkcd_num year_num) cluster(IndYear_num)
est store m_fd_llm

* CashFlowVol
reghdfe CashFlowVol DU_kw `controls', abs(Stkcd_num year_num) cluster(IndYear_num)
est store m_cfv_kw
reghdfe CashFlowVol DU_llm `controls', abs(Stkcd_num year_num) cluster(IndYear_num)
est store m_cfv_llm

* SCConc
reghdfe SCConc DU_kw `controls', abs(Stkcd_num year_num) cluster(IndYear_num)
est store m_sc_kw
reghdfe SCConc DU_llm `controls', abs(Stkcd_num year_num) cluster(IndYear_num)
est store m_sc_llm

esttab m_* using "results/v16_integrated/h3_downstream.csv", replace ///
    b(%9.4f) se(%9.4f) stats(N r2) csv nonumbers ///
    keep(DU_kw DU_llm) label
```

**预期**（v18 A 级结果）：
- ForecastDisp：DU_kw -0.0065*** / DU_llm +0.001（仅 KW 显著）
- CashFlowVol：双显著 -0.0005**
- SCConc：双显著 -0.28***/-0.46***

**Gate**：若和 v18 口径出现系数大差异（> 20% 偏离），停下来报告，可能是控制变量或样本切片差异。

---

## 8. 任务 6：样本与变量清单

**脚本**：同上，输出 `sample_summary.csv`

内容：
- 样本 N
- 控制变量清单
- FE 规格
- 每个假说对应的变量列表
- 每张表的读法

---

## 9. 完工后写 `v2/README.md`

内容：
1. 产出文件列表 + 路径
2. H2a, H2b, H2c, H3 核心系数摘要
3. 与 v15 measurement 原结果的对比（系数差异、样本差异）
4. 是否通过全部 Gate
5. 用时

---

## 10. Gate 总览

- [ ] 任务 1 构造样本 N=43,735
- [ ] 任务 2 H2a：至少 3 维度（app/dev/stock）|t|>2 负向显著
- [ ] 任务 3 H2b Direct：至少 4/5 质量变量 |t|>2 负向显著
- [ ] 任务 3 H2b Joint：DU_llm_lenstd subsume DU_kw 的关键规格 |t|>3
- [ ] 任务 4 H2c WashGap：Direct |t|>2 正向显著
- [ ] 任务 5 H3：与 v18 口径差异<20%
- [ ] `v2/README.md` 写完

任一 Gate 不达：停下来在 `OPEN_QUESTIONS.md` 报告，不要粉饰。

---

## 11. 分工

| 阶段 | 任务 | 执行者 |
|---|---|---|
| 本 spec | 6 个任务 | **Codex（你）** |
| 下一步 | 审结果 + 写第二章理论 + 第五章实证 | Claude |
| 再下一步 | 表格 PDF + Word 生成 | Codex |

