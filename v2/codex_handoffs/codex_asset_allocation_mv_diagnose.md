# Codex handoff — 资产配置复活：多 MV 批量诊断

**产出**：
- `/Users/mac/computerscience/0做完了/15会计研究/v2/stata_do/build_asset_allocation_mv.py`（构造候选 MV 并合入 lagged 样本）
- `/Users/mac/computerscience/0做完了/15会计研究/v2/stata_do/asset_allocation_mv_diagnose.do`（Step 2 批量诊断）
- `/Users/mac/computerscience/0做完了/15会计研究/v2/stata_do/asset_allocation_mv_threestep.do`（对 Step 2 过关的候选跑完整江艇三步）
- `/Users/mac/computerscience/0做完了/15会计研究/v2/results/asset_allocation_mv_*.{csv,log}` 输出

**不要做**：不碰 outline / 正文 / 其他机制脚本；不改 panel_dml.parquet / reg_sample_iv_v16.dta；不做 Zotero/Notion/Linear 写入。

---

## 上下文

前版资产配置机制（`v2/stata_do/h2_asset_allocation_lagged.do`）用朱康 11 项加总口径 FinAsset，Step 2（FinAsset_lag → PriceDelay）完全 null（t=-0.50）。决策：不放弃资产配置路线，**换 MV 口径**批量诊断。核心逻辑：如果任一 MV 能过 Step 2（FinRatio → Delay 显著），资产配置机制就能成立；全过不了 Step 2 则该路线确诊死亡。

**样本**：lagged 样本 N≈37,294（与 H1/H2 主检验口径一致）。

**理论锚点**（来自 `v2/codex_handoffs/codex_mechanism_search.md` 主线 5）：
- 彭俞超（2017, 金融研究）狭口径金融化：排除投资性房地产和长期股权投资
- 杜勇等（2017, 管理世界）交易性金融资产主导
- 张成思-郑宁（2020, 世界经济）金融化增量
- 王红建等（2017, 南开管理评论）主业 vs 金融替代
- 朱康-唐勇（2025, 会计研究）11 项加总宽口径

---

## 第一步：候选 MV 构造（Python）

**数据源**：`/Users/mac/computerscience/0做完了/15会计研究/v1/data_parquet/balance_sheet.parquet`

**已知 CSMAR 字段**（`v1/scripts/preprocess_all.py:67-83`）：
| 字段 | 含义 |
|---|---|
| A001107000 | 交易性金融资产 |
| A001202000 | 可供出售金融资产 |
| A001211000 | 投资性房地产 |
| A001229000 | 其他非流动金融资产 |
| A001218000 | 无形资产净额 |
| A001218201 | 数据资源（无形资产） |
| A001219000 | 开发支出 |
| A001000000 | 资产总计 |
| A001100000 | 流动资产合计 |

**检查其他可能需要的字段**：先 `python3 -c "import pandas as pd; print(pd.read_parquet('v1/data_parquet/balance_sheet.parquet').columns.tolist())"` 确认现有列。如果发现缺长期股权投资（A001212000）、持有至到期投资、发放贷款及垫款、应收利息/股利 等朱康 11 项中的其他分项，**先不补数据**——在产出报告中标记"口径不全"，后续由用户决定是否重新预处理。优先用现有字段构造能构造的 MV。

**候选 MV 清单**（按优先级，Step 2 只跑前 8 个，后 2 个备用）：

| MV 名 | 公式 | 理论锚点 | 预期符号 |
|---|---|---|---|
| **FinRatio_narrow** | (A001107000 + A001202000 + A001229000) / A001000000 | 彭俞超 2017 | DU→MV 负 |
| **FinRatio_trading** | A001107000 / A001000000 | 杜勇 2017 | DU→MV 负 |
| **FinRatio_realEstate** | A001211000 / A001000000 | 房地产金融化 | DU→MV 负 |
| **FinRatio_other** | A001229000 / A001000000 | 其他非流动 | 不确定 |
| **FinRatio_v4** | FinAsset（朱康 4 项现有）| 项目既定 | DU→MV 负（已验证） |
| **IntangibleRatio** | A001218000 / A001000000 | Peters-Taylor 2017 | DU→MV **正**（反向） |
| **DevOutlayRatio** | A001219000 / A001000000 | 主业 R&D 投入 | DU→MV **正**（反向） |
| **FinRatio_v4_delta** | ΔFinRatio_v4 一阶差分 | 张成思-郑宁 2020 | DU→MV 负 |
| **FinRatio_v4_vol3y** | 3 年滚动 std(FinRatio_v4) | 金融资产波动 | 不确定 |
| **Fin_to_RD** | FinRatio_v4 / (DevOutlayRatio + 0.001) | 王红建 2017 主业替代比 | DU→MV 负 |

**Winsorize**：每个 MV 1% / 99%（和现有 FinAsset 一致口径）。

**输出**：`/Users/mac/computerscience/0做完了/15会计研究/v1/data_stata/reg_sample_asset_mv.dta`——在 reg_sample_iv_v16.dta 基础上 merge 进上述候选 MV（按 Stkcd + year），保留全部原列。

---

## 第二步：Step 2 批量诊断（Stata）

**数据**：`v1/data_stata/reg_sample_asset_mv.dta`
**样本**：lagged（设置 tsset 后构造 L.DU_kw / L.DU_llm / L.MV）

**模型**：
```stata
reghdfe PriceDelay MV_lag `ctrls', absorb(Stkcd_num year_num) cluster(IndYear_num)
```
其中 `ctrls = Size Lev ROA TobinQ Age Growth IndepRatio Dual Top1Share SOE CFO`

**对每个候选 MV 跑 Step 2（MV_lag → Delay），记录**：系数、标准误、t、p、N、R²。

**Gate 标准**（**必须在 do 文件里显式打印**）：
- |t| ≥ 1.96 且符号方向合理（FinRatio 系正 / IntangibleRatio 系负，表示"金融化加剧延迟 / 无形资产降低延迟"）→ **pass**（进入三步检验）
- 1.5 ≤ |t| < 1.96 → **marginal**（标记，供用户决定是否保留）
- |t| < 1.5 → **fail**（报告数字但不进入三步）

**产出 csv**：`v2/results/asset_allocation_mv_step2.csv`，列：MV 名 / 系数 / se / t / p / N / status(pass/marginal/fail)

---

## 第三步：对过关 MV 跑完整江艇三步（Stata）

**只对 Step 2 status∈{pass, marginal} 的 MV 跑**。若全部 fail，**跳过第三步**并在 log 里明确写"资产配置全部 MV Step 2 fail，机制路线确诊死亡，建议降级"。

**三步结构**（每个过关的 MV_x 都跑 2 个 DU 测度，共 2×n 套）：

```
Step 1: MV_x = DU_{t-1} + controls + Firm FE + Year FE + cluster(IndYear)
Step 2: PriceDelay = MV_x_lag + controls + Firm FE + Year FE + cluster(IndYear)  [第二步已跑过，这里再跑一次保留]
Step 3: PriceDelay = DU_{t-1} + MV_x + controls + Firm FE + Year FE + cluster(IndYear)  [江艇同期 MV]
Step 3b: PriceDelay = DU_{t-1} + MV_x_lag + ...  [全滞后，最保守]
```

**产出**：
- `v2/results/asset_allocation_mv_threestep.csv`（esttab 汇总全部系数 + SE + N + R²）
- 单独的 `asset_allocation_mv_{name}.log` 每个过关 MV 一个

---

## 第四步：综合判断报告（markdown）

产出 `/Users/mac/computerscience/0做完了/15会计研究/v2/results/asset_allocation_mv_report.md`，结构：

```markdown
# 资产配置多 MV 诊断报告

## 1. 候选 MV 构造口径（含 CSMAR 字段映射表）
## 2. Step 2 批量诊断结果表（10 个候选）
## 3. 过关候选的三步完整结果
## 4. 结论与建议
   - 若至少 1 个 MV 过关：建议把它作为方案 B 的第四机制，理论锚点引用 XX 文献
   - 若全部 fail：确认资产配置机制在本数据下不成立，降级为"描述性对比"段落
## 5. OPEN_QUESTIONS（如有数据缺失、口径不全、意外结果）
```

---

## 技术细节

1. **Stkcd 类型**：reg_sample_iv_v16.dta 里 Stkcd 是 int（见 `generate_tables_v18.py` merge 逻辑），构造 MV 时保持一致
2. **年度取值**：balance_sheet 里用 Accper 末（Typrep=='A' 合并报表），年 = Accper.year
3. **tsset**：`tsset Stkcd_num year_num` 后 `gen MV_lag = L.MV`
4. **missing handling**：如果某 MV 因滞后损失太多观测（<30,000），在报告中标记
5. **CSV 输出编码**：用 `esttab ... using "...", csv replace` 保证可打开
6. **脚本先写后跑**：Python 构造脚本写完先 dry-run 打印 shape 和前 5 行再保存 dta
7. **不删旧文件**：保留 h2_asset_allocation_lagged.do 与相关 csv，新脚本另起目录

## 验证 Gate

- [ ] Python 构造脚本跑通，reg_sample_asset_mv.dta 保存成功，shape 合理（N≈43,735，含滞后后 effective N≈37,000+）
- [ ] 每个候选 MV 的 non-missing 观测数 > 25,000
- [ ] Step 2 批量诊断产出 csv 含 10 行（每个 MV 一行）
- [ ] 报告明确给出"过关 / marginal / fail"分类
- [ ] 若有过关 MV，三步结果 csv 就位
- [ ] 报告最后有明确建议（是否用作第四机制）

## 不要做

- 不改 outline.md / outline_paper.md
- 不改现有机制 do 文件
- 不重新预处理 CSMAR（如果缺字段就标记不补）
- 不碰 Zotero / Notion / Linear
- 不发挥写"这个机制对论文的意义"——只报诊断结果

时间预算：Python 构造 + Stata 批量跑 Step 2 应该 1 小时内完成；若有过关 MV，三步再 30 分钟。超过 2 小时未完成在 OPEN_QUESTIONS 报告卡点。
