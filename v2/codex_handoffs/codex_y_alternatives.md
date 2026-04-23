# Codex handoff — 候选主因变量 Y 调研 + 批量诊断

**产出**：
- `/Users/mac/computerscience/0做完了/15会计研究/v2/literature_y_alternatives.md`（文献调研报告）
- `/Users/mac/computerscience/0做完了/15会计研究/v2/literature_y_alternatives.bib`（BibTeX）
- `/Users/mac/computerscience/0做完了/15会计研究/v2/stata_do/build_y_alternatives.py`（Python 构造脚本）
- `/Users/mac/computerscience/0做完了/15会计研究/v2/stata_do/y_alternatives_diagnose.do`（Stata 批量 H1 诊断）
- `/Users/mac/computerscience/0做完了/15会计研究/v1/data_stata/reg_sample_y_alt.dta`（候选 Y 合并后的回归样本）
- `/Users/mac/computerscience/0做完了/15会计研究/v2/results/y_alternatives_step1.csv`（批量回归结果）
- `/Users/mac/computerscience/0做完了/15会计研究/v2/results/y_alternatives_report.md`（综合判断报告）

**不要做**：不改 outline.md / outline_paper.md / naming_conventions.md；不改 PriceDelay / SYNCH 现有 parquet；不做 Zotero / Notion / Linear 外部写入；不写正文；不改其他机制 do 文件。

---

## 上下文

论文直接竞品李世刚等（2025，中国工业经济）"企业数据资产信息披露与资本市场定价效率"的**主 DV 是 SYNCH**；Sun & Du（2024，IRFA）也是 SYNCH 主。我们目前主 DV 是 **PriceDelay**（Hou-Moskowitz 2005），补充是 SYNCH（+0.028***，与竞品方向相反，用 Chen 2021 framework 合理化放讨论段）。

用户希望：**调研除 PriceDelay / SYNCH 以外的候选 Y，在现有数据上批量跑一遍 H1，看有没有更合适的主 Y 可以替代或补充**。

**决策目标**：
1. 如果有候选 Y 显著且方向对，考虑作为**第三条 DV**进入稳健性章节
2. 如果某候选 Y 能**完全取代 PriceDelay**（系数更稳 + 与李世刚 Y 分得更开），可能升为主 DV
3. 如果全挂 / 方向不一致 / 与竞品仍然撞车，确认 PriceDelay 主 + SYNCH 讨论段是最佳方案

---

## Step A：文献调研

**目标**：在"信息披露 / 股价定价效率 / 信息效率"文献中，除了 PriceDelay（Hou-Moskowitz 2005）和 SYNCH（Morck-Yeung-Yu 2000）以外，还有哪些 Y 指标被用过？

### 检索工具
- 优先 `mcp__semantic-scholar__search_papers` / `get_paper_citations`
- 补充 `mcp__zotero__zotero_search_items` + `mcp__zotero__zotero_semantic_search`
- 必要时 WebSearch / WebFetch（CNKI）

### 检索锚点

**国际文献**：
- Roll (1988, JF) — R² 与股价信息含量的原始论文
- Durnev, Morck, Yeung, Zarowin (2003, JAR) — 股价信息性与未来盈余
- Ferreira, Ferreira, Raposo (2011, JFE) — 股价信息性与董事会
- Chen, Goldstein, Jiang (2007, RFS) — 股价信息性与投资
- Amihud (2002, JFM) — 非流动性
- Roll (1984) — Roll spread
- Corwin & Schultz (2012, JF) — 高低价价差 bid-ask estimator
- Easley, Kiefer, O'Hara, Paperman (1996, JF) — PIN
- Kim & Verrecchia (1994, JAE) — 披露前后信息不对称
- Welker (1995, CAR) — 披露评级与 bid-ask spread
- Healy & Palepu (2001, JAE) — 披露与资本市场后果综述
- Hutton, Marcus, Tehranian (2009, JFE) — 财务透明度与崩盘风险 NCSKEW
- Kim, Li, Zhang (2011, JFE) — CSR 披露与崩盘风险

**中文文献**：
- 王亚平、刘慧龙、吴联生 (2009, 经济研究) — 股价信息含量
- 游家兴、张俊生、江伟 (2006, 经济研究) — 股价信息含量
- 袁知柱、鞠晓峰 (2009, 管理科学学报) — 信息含量
- 袁知柱等 (2014) — 会计信息质量与股价信息含量
- 许年行等 (2012, 经济研究) — 股价崩盘风险

### 产出一张候选 Y 清单表

| 候选 Y | 中英文名 | 关键文献锚点 | 预期方向（披露 → Y）| 现有数据可构造度 | 与李世刚 / Sun-Du 是否冲撞 |
|---|---|---|---|---|---|

至少列出 **10 个**候选 Y，覆盖以下大类：
1. **流动性 / 信息不对称类**：Bid-Ask Spread、Amihud、Zero return days、Roll spread、Corwin-Schultz 价差
2. **股价信息含量类**：SPI = 1 − R²、FSV（Firm-Specific Return Variation）、FERC（未来盈余响应系数）
3. **交易行为类**：Turnover 换手率、Abnormal trading volume
4. **股价崩盘风险类**：NCSKEW、DUVOL（Hutton 2009、许年行 2012 都用过）
5. **估值偏离类**：Mispricing、Excess volatility（可选）
6. **信息披露下游特有的**：预测精度、盈余质量（可能和我们的机制重合，优先度低）

**每个候选要明确**：
- 核心文献（2-3 篇）
- 公式定义
- 所需数据（是否在现有 parquet 能构造）
- 与 PriceDelay / SYNCH 的概念差异
- 李世刚等 2025 / Sun-Du 2024 是否用过

### 研究 gap 判断（重要）

**Codex 必须回答**：
1. 在"数据要素 / 数据资产披露 × 股价信息效率"这个问题下，**哪些 Y 已被两篇竞品占位**（确认李世刚等 2025 原文到底用了几个 Y）
2. 哪些 Y **最能形成差异化**（即与李世刚 / Sun-Du 主 DV 分得最开，又有经典文献支撑）
3. Top 3 推荐 Y（按"差异化 + 文献支撑 + 数据可构造度"综合打分）

---

## Step B：实证诊断

基于现有 parquet 数据构造候选 Y 清单（优先级高到低）。

### 已有的可直接用数据（工作目录 `/Users/mac/computerscience/0做完了/15会计研究/`）

| 文件 | 说明 | 关键列 |
|---|---|---|
| `v1/data_parquet/daily_return.parquet` | 日行情 | Stkcd, Trddt, Clsprc, Dnvaltrd（成交额）, Dsmvosd（流通市值）, Dsmvtll（总市值）, Dretwd（日收益含红利）, Dretnd（日收益不含红利）|
| `v1/data_parquet/monthly_return.parquet` | 月行情 | Stkcd, Trdmnt, Msmvosd, Msmvttl, Mretwd, Mretnd |
| `v1/data_parquet/amihud_daily.parquet` | 日非流动性 | Stkcd, Trddt, ILLIQ（日 Amihud）|
| `v1/data_parquet/price_delay.parquet` | PriceDelay 已构 | Stkcd, year, PriceDelay |
| `v1/data_parquet/price_synchronicity.parquet` | SYNCH 已构 | Stkcd, year, SYNCH, R2_synch |
| `v1/data_parquet/per_share.parquet` | 每股指标 | EPS 等（需查编码）|
| `v1/data_parquet/income_stmt.parquet` | 利润表 | B001101000 营业收入, B001000000 营业利润, B002000000 净利润 等 |
| `v1/data_parquet/panel_dml.parquet` | 主面板 | 含控制变量 |
| `v1/data_stata/reg_sample_iv_v16.dta` | 基准 lagged 回归样本 | N=43,735 |

### 必须构造的候选 Y 清单（按优先级，Step B 至少跑前 7 个）

**优先级 🔴 高**：

1. **Amihud 年化非流动性 `Amihud_year`**
   - 公式：`Amihud_{i,t} = (1/N_i,t) × Σ |r_{i,d}| / volume_{i,d}`（N=当年交易日数）
   - 数据：`amihud_daily.parquet` 已有 ILLIQ，年化聚合
   - 文献锚：Amihud (2002); Chen-Goldstein-Jiang (2007) 
   - 预期：披露 → Amihud ↓
   - 备注：Amihud 之前曾作机制被砍（太通用），但作为 Y 是合法的

2. **Turnover 年化换手率 `Turnover_year`**
   - 公式：`Turnover_{i,t} = (1/N) × Σ (volume_{i,d} / shares_outstanding_{i,d})`
   - 或等价：`Σ (Dnvaltrd / Dsmvosd) / N`
   - 文献锚：Lo & Wang (2000); 中国文献常用
   - 预期：不确定，可能 ↑（更多交易）或 ↓（降低投机）

3. **零收益天数占比 `ZeroRet_ratio`**
   - 公式：`ZeroRet_{i,t} = count(Dretnd==0) / N_trading_days`
   - 数据：`daily_return.parquet`
   - 文献锚：Lesmond (2005 JFE)；Bekaert-Harvey-Lundblad (2007)
   - 预期：披露 → ZeroRet ↓（更多有效价格调整）

4. **股价信息性 SPI = 1 − R² `SPI`**
   - 公式：`SPI_{i,t} = 1 − R²_{i,t}`（直接从 SYNCH 反算，R²_synch 已存）
   - 文献锚：Durnev-Morck-Yeung-Zarowin (2003)；Ferreira-Ferreira-Raposo (2011)
   - 预期：披露 → SPI ↑
   - **⚠ 备注**：SPI 本质是 −ψ（SYNCH 的反函数），与我们 SYNCH +0.028 结果必然方向相反。这条做主要为了看**量级和显著性**，不是为了"换个方向"

5. **股价崩盘风险 NCSKEW `NCSKEW`**
   - 公式：Chen, Hong, Stein (2001) 负条件偏度
     - 对每只股票每年：估计 `W_{i,d} = ln(1 + abnormal_return)`
     - `NCSKEW = −[N(N-1)^(3/2) Σ W³] / [(N-1)(N-2)(Σ W²)^(3/2)]`
   - 数据：`daily_return.parquet` + `market_index` 算 abnormal return（市场模型残差）
   - 文献锚：Chen-Hong-Stein (2001)；Hutton-Marcus-Tehranian (2009)；许年行等 (2012)
   - 预期：披露 → NCSKEW ↓（改善信息透明度降低崩盘）
   - 工作量：中高

6. **股价崩盘风险 DUVOL**
   - 公式：Hutton 2009 波动率不对称度
     - `DUVOL = log[Σ_(down) W² / Σ_(up) W²] × [(N_up - 1) / (N_down - 1)]`
   - 数据同 NCSKEW
   - 预期：披露 → DUVOL ↓
   - 工作量：中高（可与 NCSKEW 共用中间变量）

7. **Corwin-Schultz 隐含价差 `CS_Spread`**
   - 公式：基于每日高低价估算（Corwin-Schultz 2012 JF）
   - 所需数据：日高低价（需要从 CSMAR 日行情补，若 `daily_return.parquet` 没有 highest/lowest 需要额外拉取或跳过）
   - 文献锚：Corwin-Schultz (2012)；Fong-Holden-Trzcinka (2017)
   - 预期：披露 → CS_Spread ↓
   - 工作量：中（公式不复杂，数据可能需要补充）
   - **备注**：如果日高低价不可得，跳过此候选并在报告 OPEN_QUESTIONS 标记

**优先级 🟡 中（若 top 7 有富余算力再跑）**：

8. **FERC 未来盈余响应系数**
   - 公式：`ret_{i,t→t+2} = α + β·ΔEPS_{t+2} + γ·DU_{i,t} × ΔEPS_{t+2} + controls`，γ 为 FERC
   - 本质是交互项，不是独立 Y，不按独立 Y 跑；若做需要单独建模
   - **建议**：Step B 不跑 FERC（超出"替代 Y"范畴），仅在文献报告提及

9. **1 − R² 的市场模型版本** （与 SPI 构造不同）
   - 若 SPI 直接从 SYNCH 反算，可再从 daily_return 跑一遍市场模型验证
   - 可选

### Python 构造脚本 `build_y_alternatives.py`

要求：
1. 产出 `reg_sample_y_alt.dta`——在 `reg_sample_iv_v16.dta` 基础上 merge 7 个候选 Y（按 Stkcd + year）
2. 每个 Y winsorize 1%/99%
3. 对每个 Y 打印 non-missing 观测数、均值、标准差、范围
4. 如果某个 Y non-missing < 25,000，在报告中标记"样本稀疏"

### Stata 批量诊断 `y_alternatives_diagnose.do`

**模型**（对每个 Y 跑 2 套 = DU_kw + DU_llm）：

```stata
* Lagged 样本（与主 H1 一致）
tsset Stkcd_num year_num
gen DU_kw_lag = L.DU_kw
gen DU_llm_lag = L.DU_llm

local ctrls "Size Lev ROA TobinQ Age Growth IndepRatio Dual Top1Share SOE CFO"

foreach y in Amihud_year Turnover_year ZeroRet_ratio SPI NCSKEW DUVOL CS_Spread {
    reghdfe `y' DU_kw_lag `ctrls', absorb(Stkcd_num year_num) cluster(IndYear_num)
    estimates store y_`y'_kw
    reghdfe `y' DU_llm_lag `ctrls', absorb(Stkcd_num year_num) cluster(IndYear_num)
    estimates store y_`y'_llm
}

* 同时跑现有 PriceDelay 和 SYNCH 作为对照基准
reghdfe PriceDelay DU_kw_lag `ctrls', absorb(Stkcd_num year_num) cluster(IndYear_num)
estimates store base_delay_kw
* ...
```

**Gate 标准**（do 文件显式打印）：
- `pass`：|t| ≥ 1.96 且符号方向与预期一致
- `marginal`：1.5 ≤ |t| < 1.96 且方向一致
- `fail`：其余

**产出 csv**：`y_alternatives_step1.csv`，每行一对 (Y, DU 测度)，列：y_name / du_measure / expected_sign / coef / se / t / p / N / status / baseline_direction_match

---

## Step C：综合判断报告

产出 `/Users/mac/computerscience/0做完了/15会计研究/v2/results/y_alternatives_report.md`。

结构：

```markdown
# 候选主因变量 Y 诊断报告

## 1. 文献调研摘要
  - 候选 Y 清单（至少 10 个，表格）
  - 与李世刚 / Sun-Du 竞品的冲撞度评估
  - Top 3 推荐 Y 依据

## 2. 候选 Y 构造口径 + CSMAR 字段映射

## 3. Step B 批量回归结果表
  - 对每个 Y × DU 测度 一行，含系数 / se / t / p / N / status
  - PriceDelay 和 SYNCH 作为 baseline 对比行

## 4. 分类判断
  ### 4.1 过关（pass）候选
  ### 4.2 Marginal
  ### 4.3 Fail

## 5. 最终建议（重点）
  按以下三种可能给出具体建议：
  - 替代 PriceDelay 作主 DV：哪个？理由？
  - 补充 PriceDelay 作第三稳健性 DV：哪个？理由？
  - 保持 PriceDelay 主 + SYNCH 讨论段不变（若全挂）

## 6. OPEN_QUESTIONS
  - 实证中遇到的不确定点
  - 是否有 Y 因数据缺失没跑（如 CS_Spread 的高低价数据）
```

---

## 验证 Gate（自检清单）

- [ ] Step A 候选 Y 清单表 ≥ 10 项，每项含文献锚 + 公式 + 数据可构造度 + 预期方向
- [ ] Step A 明确回答了"李世刚原文用了哪些 Y"和"Top 3 推荐 Y"
- [ ] Python 构造脚本 dry-run 打印 shape + 前 5 行再保存 dta
- [ ] reg_sample_y_alt.dta 保存成功，每个 Y 的 non-missing > 20,000（若 <20,000 标记样本稀疏）
- [ ] Stata 批量 do 跑完，csv 输出含全部 Y × DU 测度行 + PriceDelay / SYNCH baseline 对照
- [ ] 报告含明确分类（pass / marginal / fail）
- [ ] 报告给出最终建议（三选一）
- [ ] bib 文件抓回用到的每篇关键文献
- [ ] OPEN_QUESTIONS 段列出至少 2-3 条

---

## 不要做

- 不改 outline.md / outline_paper.md / naming_conventions.md / 框架图
- 不改现有 PriceDelay / SYNCH 的 parquet
- 不碰 Zotero / Notion / Linear 外部写入
- 不写正文任何段落
- 不发挥"这个 Y 对论文的意义"——只做诊断数字与建议
- 不追求完美构造某个复杂 Y（如 CS_Spread 缺数据就跳过，标记即可）

**时间预算**：Step A 文献检索 1-2 小时（semantic-scholar + zotero + CNKI）；Step B Python + Stata 构造与诊断 1.5 小时；Step C 报告 0.5 小时。超过 4 小时未完成在 OPEN_QUESTIONS 列卡点。
