# v16 integrated 草稿集（v2/v16_draft/）

**创建时间**：2026-04-19
**目标刊**：Economic Modelling（Elsevier, SSCI Q2）
**路径**：中文先写 → 英文版后翻

---

## 产出清单

### 已完成

| 文件 | 内容 | 字数估算 |
|---|---|---|
| `tables_v16_integrated.md` | 全套 10 张回归表 + 1 张图的规范（按 v17 朱康会计研究格式） | 表格规范 |
| `chapter2_theory.md` | 第二章 理论分析与研究假说完整草稿 | ~3500 字 |
| `README.md` | 本索引 | - |

### 待完成（按优先级）

| 文件 | 内容 | 预估字数 | 优先级 |
|---|---|---|---|
| `chapter1_introduction.md` | 第一章 引言 | 1000 字 | 🟡 高 |
| `chapter5_content_quality.md` | 第五章 披露内容异质性与信息含量差异（H2a/H2b/H2c 实证） | 2000 字 | 🔴 最高 |
| `chapter6_channels.md` | 第六章 下游 channels（H3） | 1200 字 | 🟡 高 |
| `chapter3_design.md` | 第三章 研究设计 | 1500 字 | 🟢 中（可从 v15/v18 迁移） |
| `chapter4_baseline.md` | 第四章 基准回归与稳健性 | 1500 字 | 🟢 中（可从 v15/v18 迁移） |
| `chapter7_heterogeneity.md` | 第七章 异质性分析 | 800 字 | 🟢 中 |
| `chapter8_conclusion.md` | 第八章 结论 | 500 字 | 🟢 低 |

---

## 表格编号对照（相对 v17）

| v17 | v16 integrated | 备注 |
|---|---|---|
| 表 1 描述性统计 | **表 1**（新增 11 个新变量） | 扩表 |
| 表 2 基准回归 | **表 2**（不动） | 保留 |
| 表 3 DML-PLR | **表 3**（不动） | 保留 |
| 表 4 内生性 | **表 4**（不动） | 保留 |
| 表 5 稳健性 | **表 5**（不动） | 保留 |
| 图 1 PSM 核密度 | **图 1**（不动） | 保留 |
| 表 6 旧传导机制 | ❌ 删除 | 整体替换 |
| — | **表 6** 价值链五维分解（H2a） | 新增 |
| — | **表 7** 多维质量 direct + joint（H2b） | 新增 |
| — | **表 8** 广度-深度错配（H2c） | 新增 |
| — | **表 9** 下游 channels（H3，替代旧表 6） | 新增 |
| — | **表 10** 异质性 Fisher（H4） | 新增 |

---

## 生成 PDF 的 Codex handoff prompt（后续用）

```
请参考 v17 PDF 样式（朱康会计研究格式，宋体 SongtiSC，三线表），
根据 v2/v16_draft/tables_v16_integrated.md 规范生成 v16 integrated 回归表 PDF。

输入：
- 表格规范：/Users/mac/computerscience/0做完了/15会计研究/v2/v16_draft/tables_v16_integrated.md
- 参考脚本：/Users/mac/computerscience/0做完了/15会计研究/scripts/generate_tables_v17.py
- 数据源：
  - 表 1：data_stata/reg_sample_v16_integrated.dta
  - 表 2-5 + 图 1：保持 v17 现有
  - 表 6：results/v16_integrated/h2a_value_chain.csv
  - 表 7 Panel A：results/v16_integrated/h2b_quality_direct.csv
  - 表 7 Panel B：results/v16_integrated/h2b_quality_joint.csv（过滤 joint_dukw_*）
  - 表 8：results/v16_integrated/h2c_washgap.csv
  - 表 9：results/v16_integrated/h3_downstream.csv
  - 表 10：results/v18/heterogeneity_v18.csv

输出：
- results/v16_integrated/regression_tables_v16_integrated.pdf
- scripts/generate_tables_v16_integrated.py（新生成器脚本，基于 v17 改造）

要求：
1. 表号 1-10，图 1
2. 表题居中，宋体 SongtiSC
3. 三线表：top rule / mid rule at column header / bottom rule
4. 变量名左侧，系数 + */**/*** 显著性，括号内标准误
5. 控制变量、Firm/Year FE、N、R² 在表底
6. 脚注字号 9pt，左对齐
7. v17 已有的表 2-5 + 图 1 直接复用 v17 文件
```

---

## 叙事红线（写作时必须遵守）

来自 `../naming_conventions.md` 与 `../outline.md`：

1. 不用 subsume / dominate / 主导 → 用"增量解释力"
2. H3 不 claim mediation → 用"与估值不确定性下降相一致的 channels"
3. 统一"股价定价效率"，不用"资本定价效率"
4. 统一"数据要素利用披露"，不单用"数据要素利用"
5. H4 用统一框架（原始不确定性 / 信息摩擦 / 外部监督）
6. 数字显著性：*/**/*** 对应 10%/5%/1%

详细规范见 `../naming_conventions.md`。

