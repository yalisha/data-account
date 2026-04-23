# Asset Allocation MV Diagnose

本 README 记录本轮 `asset allocation MV diagnose` 的执行约束、`OPEN QUESTIONS` 与结果目录说明。脚本文件写在 `v2/stata_do/`，结果文件写在 `v2/results/`。

## OPEN QUESTIONS

1. **候选口径存在字段缺口**
   `balance_sheet.parquet` 缺少 `A001212000`（长期股权投资）。因此本轮只能按现有字段构造 `spec` 列出的 10 个候选 MV，无法把朱康 11 项或更完整狭口径补齐到更宽版本；报告里将明确标记为“口径不全，未补数据”。

2. **`spec` 对 Step 2 候选数的描述有冲突**
   同一份 `spec` 前文写“Step 2 只跑前 8 个，后 2 个备用”，但后文又要求：
   - `asset_allocation_mv_step2.csv` 必须有 10 行
   - 报告第 2 节展示 10 个候选
   - 验证 gate 明确要求 “10 行（每个 MV 一行）”
   
   本轮默认执行假设：**Step 2 实跑全部 10 个候选**，以满足输出 gate；同时在报告中注明最后 2 个原本属于低优先级备用口径。

3. **Step 2 的符号 gate 对部分候选并不完全明确**
   `spec` 明确写了 “FinRatio 系正 / IntangibleRatio 系负”，但候选表里 `FinRatio_other` 与 `FinRatio_v4_vol3y` 的理论预期写的是“不确定”。本轮默认执行假设：
   - `FinRatio_*` 和 `Fin_to_RD` 统一按 **正向** `PriceDelay` 解释
   - `IntangibleRatio` 与 `DevOutlayRatio` 统一按 **负向** `PriceDelay` 解释
   
   若这两个“不确定”口径最终出现高 t 但反向结果，报告会单列说明，不会强行解释为通过。

4. **README 放置位置的执行假设**
   用户要求“在产出目录的 README.md 列 OPEN QUESTIONS”，而本任务有两个产出目录（`v2/stata_do/` 与 `v2/results/`）。本轮将 `README.md` 放在 `v2/results/`，作为整条诊断流水线的统一说明页。

## Scope Lock

- 只新增脚本与结果文件
- 不修改既有 `do` / `csv`
- 不修改 `outline` / 正文 / Zotero / Notion / Linear
- 样本口径固定为 lagged 样本，对齐 `reg_sample_iv_v16.dta`

## Planned Outputs

- `v2/stata_do/build_asset_allocation_mv.py`
- `v2/stata_do/asset_allocation_mv_diagnose.do`
- `v2/stata_do/asset_allocation_mv_threestep.do`
- `v1/data_stata/reg_sample_asset_mv.dta`
- `v2/results/asset_allocation_mv_step2.csv`
- `v2/results/asset_allocation_mv_threestep.csv` 或 gate-skip 日志
- `v2/results/asset_allocation_mv_report.md`
