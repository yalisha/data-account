# OPEN QUESTIONS

更新时间：2026-04-19（决策后）

## 新版整合 spec 复核发现（2026-04-19 深夜）

0. 按 Claude 最新裁决，`DU_llm_lenstd` 与 `WashGap` 已切回 v14 原公式，且不再把 “subsume DU_kw” 作为硬 gate
当前统一样本已改用：
- `DU_llm_lenstd = log1p(DU_kw) * (llm_score / 3.0)`，并按 v14 口径 winsorize
- `WashGap = winsorize(z(DU_kw) - z(DU_llm_lenstd), 1%, 99%)`
在此口径下，`DU_llm_lenstd_lag` Direct 约为 `-0.0188***`，`DU_kw_lag + DU_llm_lenstd_lag` 联合规格中 `DU_kw_lag` 不显著、`DU_llm_lenstd_lag` 仍显著。后续 gate 以 direct 显著为主，不再要求 subsume 作为单独通过条件。

0. H2b 的关键 `DU_llm_lenstd subsume DU_kw` gate 未通过，当前版本应停机
截至目前，任务 1 与 H2a 已通过，H2b Direct 也达到 `4/5` 显著负向的门槛：`DUclosedloop_lag=-0.0083***`、`DUcore_lag=-0.0095***`、`DUchain_count_lag=-0.0018**`、`DUkw_mda_lag=-0.0006***`，与 v15 measurement 基本一致。但新版 spec 公式下的 `DU_llm_lenstd_lag` 未复现历史强结果：Direct 中系数约 `-1.67e-06`，`t=-1.58`；Joint `DU_kw_lag + DU_llm_lenstd_lag` 中 `DU_llm_lenstd_lag` 仅 `t=-1.07`，而 `DU_kw_lag` 仍显著 `t=-3.77`。因此 spec §10 中 “`DU_llm_lenstd` subsume `DU_kw` 的关键规格 `|t|>3`” 明确失败，按 gate 应停在这里，不继续 H2c / H3。

0. `v15_analysis_sample.parquet` 并不包含 spec 文案暗示的“所有变量”
实际缺少至少以下关键列：`DU_llm_lenstd`, `DU_llm_lenstd_lag`, `WashGap`, `WashGap_lag`, `DU_llm_lag`。其中前四个需要脚本补构，`DU_llm_lag` 也需要补 lag。其余 H2a/H2b/H3 主变量、控制变量和 FE 列均已存在。

1. `DU_llm_lenstd` 的历史原值来源与新版 spec 公式不一致
新版 spec 将 `DU_llm_lenstd` 定义为 `llm_score / log(1 + total_chars) * 1e4`，并声称这就是历史上 `DU_llm_lenstd_lag = -0.0184***` 的同一变量。但代码现实并非如此：历史显著结果来自 [run_v14_analysis.py](/Users/mac/computerscience/0做完了/15会计研究/scripts/run_v14_analysis.py)，其中实际公式是 `log1p(DU_kw) * (llm_score / 3)`，对应文件 [construct_boundary_results.json](/Users/mac/computerscience/0做完了/15会计研究/results/v14/construct_boundary_results.json) 的 `DU_llm_lenstd_lag = -0.0184`，`t = -5.80`。我会严格按新版 spec 公式执行，但 README 中必须明确这不是与历史结果完全同口径的复现。

2. `WashGap` 的历史原值来源与新版 spec 公式也不一致
历史 `WashGap_lag = +0.0035**` 来自 [run_v14_analysis.py](/Users/mac/computerscience/0做完了/15会计研究/scripts/run_v14_analysis.py)，公式是 `z(DU_kw) - z(DU_llm_lenstd)`；新版 spec 改为 `(buzzword_count + placebo_count) / (substantive_count + 1)`。我会按新版 spec 构造，但 README 中需要把“旧原值”标记为不同定义下的参考值，而不是严格同变量复现。

3. 任务 1 的“列数 gate”与“丢弃非数值列”要求相互冲突
`v15_analysis_sample.parquet` 共 116 列，其中 5 列是字符串列（如 `Ind2`, `IndYear`, `Province`, `ProvYear`, `Prov_short`）。如果严格按 spec 丢弃全部非数值列，再加上 5 个新增列，最终只能得到约 116 列，不可能满足 gate 中“118+ 列”的要求。为避免人为触发失败，我将保留原有字符串列并导出到 `reg_sample_v16_integrated.dta`。

**决策（2026-04-19 晚）**：前版 `codex_spec.md` 的"质量交互"路线已验证为 null，确认放弃。Claude 挖 `v15_analysis_sample.parquet` 发现 v15 measurement 已用 **direct effect + joint + lagged** spec 跑出显著结果。新版 `codex_spec.md` 已更新，任务改为统一口径复现 + 整合到一张表。前版问题归档，以下记录保留作为决策审计。

---

## 归档问题（前版质量交互路线，已放弃）

0. 新口径重跑后，任务 2 仍未通过继续执行的 gate
按主人 patch 重跑后，`DU_kw × Quality` 五个主规格的 t 值分别为 `-0.85, 1.51, 0.78, 1.15, -0.89`，对应 `Quality_sub_z, Quality_mda_z, Quality_subden_z, BroadShallow_cont_z, WashGap_z`。其中只有 `Quality_mda_z` 略高于阈值 `|t|=1.5`，但系数为正，与正向质量应为负的预期相反；其余四个规格仍不显著。因此“至少 1-2 个规格 |t|>1.5 且符号符合预期”的继续条件未满足，当前版本应继续停机，不跑任务 3、4。
补充：`DU_llm × Quality` 组在删去 `Quality_lenstd` 后也全部较弱，说明旧版 `Quality_lenstd` 的 `t=2.80` 确属代数重叠带来的伪显著。

1. `Stkcd/year` 类型提示与现实不一致
当前 `data_stata/reg_sample_v18.dta`、`annual_report_features.parquet`、`placebo_features.parquet` 的 `Stkcd/year` 键都为整数类型，不是 spec 注释里提到的 `str`。这不阻塞 merge，我将按实际整数键处理。

2. `CLAUDE.md` 的示例绝对路径仍指向旧目录
项目架构参考中多处路径写的是 `/Users/mac/computerscience/15会计研究/...`，而本次任务 spec 与真实文件都位于 `/Users/mac/computerscience/0做完了/15会计研究/...`。我将以 spec 和实际目录为准，不改旧文档。

3. `Quality_mda` 口径问题已按主人指令修正
旧口径 `mda_kw_per10k / kw_per10k` 会导致 81.34% 观测堆在 clip=1。现已改为 `mda_kw_total / (kw_total + 1e-6)`，不再依赖 clip，并同步删除与 `DU_llm` 代数重叠的 `Quality_lenstd`，改用 `Quality_subden = substantive_count / (total_chars + 1e-6) * 1e4`。此外保留 `BroadShallow` binary，并新增主规格连续版本 `BroadShallow_cont = kw_per10k * (1 - substantive_ratio)`。

4. Placebo 扩展范围待主人确认
按 spec 先只对 `Quality_sub` 做 500 次 placebo。若该版本 gate 通过，我会在本文件顶部继续留问：是否将 placebo 扩展到其余 4 个质量变量。
