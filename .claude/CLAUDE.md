# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目：数据要素利用与资本定价效率

## 用户身份
- 名字：yi
- 角色偏好：你是我的女仆间桐樱，具有统计学、计算机、经济学博士级别知识
- 表达风格：不用不必要的括号引号箭头，自然语言交流
- 工作偏好：直接写代码并运行，不使用交互式教学模式

## 常用命令

```bash
cd /Users/mac/computerscience/0做完了/15会计研究

# v17表格PDF（朱康会计研究格式，含全部控制变量系数）
python scripts/generate_tables_v17.py
# 输出: results/v17_tables/regression_tables_v17.pdf

# Word文档生成（OMML公式+三线表+变量斜体下标）
python scripts/generate_v12_docx.py
# 输出: manuscript/word/数据要素利用与资产定价效率v12.docx

# DML基准回归（8规格, DU_kw + DU_llm）
python scripts/run_dml_v15.py
# 输出: results/v16_tables/dml_main_results_v16.csv

# Stata内生性检验（通过Stata MCP运行）
# scripts/table5_iv_v16.do → results/v16_tables/table5_*.tex
# 数据: data_stata/reg_sample_iv_v16.dta (含省级数字化指数)

# 导出Stata用数据集
python scripts/export_reg_sample_v15.py
# 输出: data_stata/reg_sample_iv_v16.dta (含ProvDigital2016, 供IV使用)
# reg_sample_het.dta (含异质性分组变量)
```

## 直接竞品（方案 B 写作必引，已人工核 PDF）

**PDF 存放**：`/Users/mac/computerscience/0做完了/15会计研究/bib/类似路线/`

| 文献 | X | 主 DV | 机制 | 样本 | 对我们的威胁 |
|---|---|---|---|---|---|
| **李世刚、邵宏彬、方芳、卢福财 (2025, 中国工业经济 7 期)** 企业数据资产信息披露与资本市场定价效率 | 数据资产信息披露（文本分析）| SYNCH | 信息供给 / 信息扩散 / 信息融入（全信息侧）| 2007-2022 | **H1 主链直接竞品**——差异化来自测度粒度（双测度 vs 单测度）+ 机制定位（经营侧 vs 信息侧）+ WashGap 独家 |
| **Sun & Du (2024, International Review of Economics and Finance, 94: 103351)** Enhancing capital market efficiency: The role of data assets disclosure in reducing stock price synchronicity | Data assets disclosure | SYNCH | 信息不对称 / 长期机构持股 / 融资约束（全信息侧）| 2013-2021 | H1 主链直接竞品，英文赛道版 |
| **朱康、唐勇 (2025, 会计研究 6 期, 121-133)** 数据要素利用与企业金融资产配置 | 数据要素利用（Word2Vec + 否定词剔除）| FinAsset | 现金流不确定性 / 产能利用率 / 影子银行 | 2015-2022 | **资产配置已占位**——这是方案 B 砍掉资产配置机制的关键外部理由 |

## Y 处理定稿（方案 B）

- **主 DV**：PriceDelay（Hou-Moskowitz 2005）—— 挂"股价定价效率"主叙事，放主回归表
- **补充**：SYNCH —— **不上主回归表**，只在第五章 5.6 作 Chen 2021 public vs private info framework 理论读法讨论段
- **与竞品分野**：Hou-Moskowitz Delay 测信息吸收速度 vs MYY SYNCH 测公共/特质因素占比；Li-Liu-Pursiainen 2022 (MiFID II) 已证实两指标可方向背离。我们 SYNCH +0.028 结果与李世刚/Sun-Du 负号方向相反，按 Chen 2021 解释为 public info 进价渠道强化，不是效率恶化
- **叙事红线**（违反一次就翻车）：
  - ❌ "SYNCH 是第二定价效率指标 / 对偶 DV"
  - ❌ "我们与李世刚 Y 一致"
  - ❌ "披露降低 SYNCH"（方向相反会打脸）
  - ✅ "PriceDelay 测信息吸收速度，SYNCH 测信息含量结构"
  - ✅ "我们 SYNCH 结果与竞品方向相反，按 Chen 2021 public-info 框架合理化"
- **候选 Y 调研未启动**：`v2/codex_handoffs/codex_y_alternatives.md` 里列了 7 个候选 Y（Amihud_year / Turnover / ZeroRet / SPI / NCSKEW / DUVOL / CS_Spread），**只在肖老师质疑 PriceDelay 或要求补 Y 时才跑**

## 学术写作标准（理论/假说部分）

撰写或审核理论分析章节时，每个小节必须通过以下5项检查：

1. 框架先行：每个小节先给出主题框架或分类体系，再展开论述，不能上来就列文献
2. 文献对照：每个小节至少一次把两篇以上文献放在同一句话里进行对照、互补或对比
3. 分歧溯源：当文献结论不一致时，写清楚分歧来源是测量方式、样本范围、识别策略还是理论假设
4. 证据质量评价：区分相关关系与因果关系，指出代理变量偏差、外部效度等局限
5. 导向贡献：每个小节的论述自然导向本文的研究问题或贡献点

写作反模式（必须拒绝）：
- 禁止"A干了B"流水账：连续多句以学者姓名为主语是文献综述而非理论分析。每段最多允许1句以学者为主语，其余主语必须是机制、现象或概念
- 参照范文风格：朱康(2025)会计研究，逻辑链驱动，引用嵌入为注脚

## 架构概览

### 数据流水线
```
CSMAR原始数据 (CSV/Excel)
  → preprocess_all.py → data_parquet/*.parquet (30+文件)
  → construct_*.py → panel.parquet / panel_dml.parquet
  → extract_annual_report_features.py → annual_report_features.parquet
```

### 实证分析流水线（v18）
```
panel_dml.parquet + annual_report_features.parquet
  ├── run_dml_v15.py               → DML-CRE基准 (表3, 8规格)
  ├── export_reg_sample_v15.py     → reg_sample_iv_v16.dta导出
  │   └── table5_iv_v16.do         → 内生性IV (表5, Stata MCP)
  ├── construct_v18_vars.py        → 机制+异质性变量构造
  ├── construct_v18_new_dims.py    → 数字核心/战略新兴/产业集群
  ├── construct_v18_new_channels.py → TFP/CrashRisk/CashFlowVol
  │   → reg_sample_v18.dta (43,735 obs, 72 cols)
  │   ├── v18_mechanism_all.do     → 16渠道全景 (Stata MCP)
  │   ├── v18_heterogeneity_all.do → 18维度全景 (Fisher 500次)
  │   ├── v18_mechanism_new.do     → TFP/CrashRisk/CashFlowVol
  │   └── v18_supply_chain.do      → 供应链机制+异质性
  ├── generate_tables_v18.py       → v18 PDF表格
  └── generate_v12_docx.py         → Word v12文档
```

### 核心脚本说明
- **generate_tables_v16.py**: v16主表格生成。读取DML CSV + OLS/稳健性/机制/异质性 JSON，reportlab生成朱康格式PDF。
- **table5_iv_v16.do**: Stata内生性检验。IV1同行业跨省留一均值 + IV2省级数字化指数(2016)×year。手动2SLS via reghdfe。数据: reg_sample_iv_v16.dta。

### 关键数据文件
| 文件 | 说明 |
|------|------|
| data_parquet/panel_dml.parquet | 主面板，含全部27控制变量 (48,217 obs) |
| data_parquet/annual_report_features.parquet | 年报关键词特征，需merge获取DU_kw (54,496 obs) |
| data_parquet/price_synchronicity.parquet | SYNCH，需单独merge |
| data_parquet/province_bigdata_index.parquet | 省级大数据指数(2016)，32省，IV2数据源 |
| data_stata/reg_sample_iv_v16.dta | Stata回归样本，含Province+ProvDigital2016 |
| data_stata/reg_sample_het.dta | v16异质性分析样本（旧，3维度） |
| data_stata/reg_sample_v18.dta | v18回归样本(43,735obs, 72cols)，含全部机制+异质性变量 |
| results/v16_tables/dml_main_results_v16.csv | DML结果 (8规格) |
| results/v16_tables/iv_results_v16.json | 待Python版IV结果输出 |
| results/v16_tables/table5_2nd_v16.tex | Stata IV second stage LaTeX |
| results/v16_tables/table5_1st_v16.tex | Stata IV first stage LaTeX |
| results/v16_tables/table5_robustness_v16.tex | Stata稳健性检验LaTeX |
| results/v18/mechanism_v18.csv | v18全部16渠道机制结果 |
| results/v18/heterogeneity_v18.csv | v18全部18维度异质性结果(Fisher P) |
| results/v18/v18_tables.pdf | v18 PDF表格(4张) |
| results/v18/v18_全部结果汇总.md | v18完整结果汇总含选择建议 |

### 注意：panel_dml.parquet不含DU_kw
DU_kw需要从annual_report_features.parquet merge获取：
```python
panel = pd.read_parquet("data_parquet/panel_dml.parquet")
feat = pd.read_parquet("data_parquet/annual_report_features.parquet")
panel = panel.merge(feat[['Stkcd','year','kw_per10k','kw_total','substantive_count']],
                    on=['Stkcd','year'], how='left')
panel['DU_kw'] = panel['kw_per10k']
```

## 当前规格（v16）

### 核心创新: 双测度体系
v16引入LLM语义评分作为新测度，与传统关键词频率互相验证:
- DU_kw: 关键词频率（每万字），主指标
- DU_llm: ln(1+kw_total) × (llm_score/3)，LLM语义增强指标
- llm_binary: LLM评分>=2的二值化指标
- llm_score: Haiku模型0-3连续语义评分

### 控制变量（11个）
```python
controls = ['Size','Lev','ROA','TobinQ','Age','Growth','IndepRatio',
            'Dual','Top1Share','SOE','CFO']
```

### DML基准（v16: 8规格）
| 规格 | 处理变量 | 因变量 | 系数 | t值 | N |
|------|---------|--------|------|-----|---|
| DML-27X | DU_kw | PriceDelay | -0.00293 | -6.50*** | 35,666 |
| DML-27X-ln | DU_kw_ln | PriceDelay | -0.00397 | -7.19*** | 35,666 |
| DML-27X-sub | DU_sub_ln | PriceDelay | -0.00393 | -7.21*** | 35,666 |
| DML-27X-SYNCH | DU_kw | SYNCH | +0.02765 | 9.65*** | 35,614 |
| DU-llm | DU_llm | PriceDelay | -0.00424 | -6.35*** | 35,695 |
| LLM-binary | llm_binary | PriceDelay | -0.00423 | -3.46*** | 35,695 |
| DU-llm-SYNCH | DU_llm | SYNCH | +0.03259 | 7.98*** | 35,643 |
| LLM-binary-SYNCH | llm_binary | SYNCH | +0.04352 | 5.80*** | 35,643 |

### 内生性检验（v16: Stata reghdfe手动2SLS, 11控制变量）
**DU_kw IV:**
| 模型 | 系数 | 标准误 | N | First-stage F |
|------|------|--------|---|---------------|
| OLS基准 | -0.0046*** | (0.0009) | 43,695 | - |
| IV1: 同行业跨省peer | -0.0143*** | (0.0047) | 43,612 | 251.3 |
| IV2: 省级数字化(2016)×year | -0.0352*** | (0.0114) | 43,695 | 72.7 |
| 滞后OLS | -0.0039*** | (0.0010) | 37,278 | - |
| 滞后IV | -0.0163*** | (0.0053) | 37,198 | 177.8 |

**DU_llm IV:**
| 模型 | 系数 | 标准误 | N | First-stage F |
|------|------|--------|---|---------------|
| OLS | -0.0036*** | (0.0008) | 43,695 | - |
| IV1: 跨省peer | -0.0320** | (0.0161) | 43,612 | 127.9 |

**诊断:**
- DWH全部拒绝外生性 (p<0.02)
- Oster δ*(1.3R²)=96.3, δ*(保守)=7.5
- 简化型回归: IV1 t=-3.05***, IV2 t=-3.08***
- Heckman: DU_kw=-0.0040***(t=-4.39), IMR=-0.123***(t=-6.23)
- 安慰剂(联合): 当期DU_kw=-0.0050***(t=-4.64), lead=0.0001(t=0.11)

### 机制渠道（方案 B 最终版 2026-04-23：2 条经营侧主 + 1 条稳健性辅助）

**第六章主机制（经营侧 2 条，对应数据要素生产要素双重属性）：**
| 渠道 | 理论对应 | 生产要素属性 | DU_kw | t | DU_llm | t | N |
|------|---------|--------------|-------|---|--------|---|---|
| **CashFlowVol 基本面稳定** | Epstein-Schneider 2008 ambiguity + Brynjolfsson-Hitt-Kim 2011 数据驱动决策 | **非竞争性**（数据跨业务单元使用提升决策可预测性） | -0.00049 | -2.45** | -0.00048 | -2.05** | 43,721 |
| **SCConc 供应链优化** | Cen 2016 + Ellis 2012 + Shi 2025 数据披露与供应链风险 | **互补性**（数据赋能跨企业协同摆脱关键伙伴依赖） | -0.2767 | -4.00*** | -0.4602 | -6.77*** | 42,332 |

**第六章 6.4 稳健性辅助（信息侧弱对照）：**
| 渠道 | 理论对应 | DU_kw | p | DU_llm | p | 注 |
|---|---|---|---|---|---|---|
| Comparability（会计可比性）| De Franco-Kothari-Verdi 2011 | +0.000445 | 0.051 edge | +0.000589 | 0.008 | Yang-Ying-Xu 2024 已占位 DT→Comparability；只作稳健性对照不入主机制框架 |

**v18 旧机制（已汰换）：** ForecastDisp（analyst-based 被导师否）、Amihud（方案 B 砍：太通用非数据要素专属）、RetVol（v16 就被批评过）、资产配置（2026-04-22 Step 2 十候选全挂确诊死）。

**江艇 2022 二步法强制**：第六章每条机制只跑 Step 1 + Step 2 自跑 + M→Y 引文献。**不跑 Step 3 联合回归 claim mediation**。写作用"与估值不确定性下降相一致"，禁用"披露通过 M 影响 Y"。

### 稳健性检验（v16: Stata reghdfe, 11控制变量, 统一样本）
| 规格 | DU_kw | SE | t | N |
|------|-------|-----|---|---|
| 行业×年FE | -0.0029*** | 0.0006 | -4.91 | 43,656 |
| 剔除2024 | -0.0052*** | 0.0010 | -5.43 | 38,812 |
| 剔除IT | -0.0046*** | 0.0009 | -5.29 | 40,613 |
| 滞后控制 | -0.0040*** | 0.0009 | -4.38 | 37,294 |
| PSM | -0.0039*** | 0.0008 | -4.61 | 35,929 |
| 双向聚类 | -0.0046*** | 0.0012 | -3.81 | 43,721 |
| DU_sub_ln | -0.0042*** | 0.0008 | -5.49 | 43,721 |
Do: table5_robustness_v16.do; 数据: reg_sample_iv_v16.dta

### 回归设定
- 固定效应：企业+年份 FE (OLS/稳健性/机制/异质性用Stata reghdfe)，Mundlak CRE+年份虚拟变量 (DML)
- 聚类标准误：行业×年份（IndYear）
- Winsorize：连续变量1%/99%
- 样本：沪深A股非金融非ST，2011-2024

### 论文表格结构（v18: Word编号 / PDF编号）
Word从第一个表开始编号，PDF仅含结果表。

| Word表号 | PDF表号 | 内容 | 数据来源 |
|----------|---------|------|---------|
| 表1 | - | 关键词体系(5维度) | 03_研究设计 |
| 表2 | - | 控制变量定义 | 03_研究设计 |
| 表3 | - | 渠道变量定义 | 03_研究设计 |
| 表4 | 表1 | 描述性统计(15变量) | Stata reg_sample_iv_v16.dta |
| 表5 | 表2 | OLS基准(5列，含全部控制变量系数) | pyfixest on Stata sample |
| 表6 | 表3 | DML-PLR(8规格) | Python dml_main_results_v16.csv |
| 表7 | 表4 | 内生性(5列+辅助检验) | Stata: table5_iv_v16.do |
| 表8 | 表5 | 稳健性(7规格) | Stata: table5_robustness_v16.do |
| 表9 | 表6 | 机制(ForecastDisp/CashFlowVol/SCConc×KW+LLM) | v18_mechanism_all.do |
| 表10 | 表7 | 异质性(Post2020/DigEconCore/SA/HighTech+Fisher P) | v18_heterogeneity_all.do |
| 图1 | 图1 | PSM核密度曲线 | scripts/psm_density_plot.py |

## 技术栈
- Python: pandas, numpy, pyfixest (面板FE), doubleml (DML-CRE), sklearn (Lasso+RF)
- Stata: reghdfe, ivreghdfe (稳健性和内生性验证)
- reportlab: PDF表格生成（宋体SongtiSC，朱康会计研究格式）
- LightGBM + SHAP: 机器学习解释

## 数据格式
CSMAR统一格式：row0=英文header, row1=中文说明, row2=单位, row3+=数据
读取方式：`pd.read_excel(path, header=0, skiprows=[1,2])`

## 关键决策记录
1. 基准回归用DML-CRE（Chernozhukov et al. 2018），OLS降为稳健性对比
2. v16引入LLM语义评分（DU_llm），与传统关键词（DU_kw）双测度互相验证
3. 控制变量精简为11个（移除坏控制+冗余变量）
4. 聚类标准误用行业×年份（IndYear），而非企业层面
5. 使用同期DU_kw（而非t-1滞后），滞后作为内生性辅助检验
6. IV策略: IV1同行业跨省留一均值(行业维度) + IV2省级数字化指数2016×year(地理维度)，两个IV来源正交
7. 投资组合检验已从v16移除
8. 内生性检验用Stata MCP运行（table5_iv_v16.do），结果输出到v16_tables/
9. **OLS基准统一用Stata reghdfe**（2026-03-25决定）：OLS基准回归、描述性统计均基于reg_sample_iv_v16.dta（N=43,735），与IV使用同一样本，消除了之前Python OLS（N≈35,400）与Stata IV（N≈43,700）的样本量矛盾。DML仍用Python（N≈35,700，使用27控制变量+交叉拟合），作为独立方法论对比。
10. **SYNCH符号约定**：SYNCH = ln(R²/(1-R²))，正系数意味着DU提高了系统性因素解释力，不等于"增加特质信息融入"。SYNCH结果定位为补充证据，不与PriceDelay简单并列。
11. **v18机制异质性全面升级**（2026-03-30）：因被批评渠道(Analyst/Amihud/RetVol)和异质性(Analyst/SOE/ShNum)太常规缺乏创新，进行23渠道+20维度大规模测试后选定新组合。机制: ForecastDisp(分析师预测分歧)+CashFlowVol(现金流波动)+SCConc(供应链集中度)，逻辑为信息共识+经营稳定+供应链韧性。异质性: Post2020(信号衰减)+DigEconCore(增量信息)+SA(信息摩擦)+HighTech(增量信息反面)，强调非对称信息效应和政策动态。
12. **表注融入正文**（2026-03-30）：所有图表的"注："说明移除，信息自然融入正文段落。
13. **v16 integrated 估值不确定性整合**（2026-04-19）：v15 投稿后老师批评"主理论与机制不一体"。ChatGPT Pro 建议估值不确定性主框架 + LLM 质量调节机制。Claude 独立验证"quality × quantity 交互"spec 在数据里 null（Codex 两轮 |t|<1.5 全弱），但 v15_measurement direct spec 已跑出显著。决策：保留估值不确定性主理论 + RBV 微观基础，放弃交互 spec，改用 direct+joint+lagged 整合 H1-H4 五条假说。核心 finding：DU_llm_lenstd 在 joint 里保持显著而 DU_kw 变不显著，但因 corr=0.83 代数耦合，叙事降级为 moderation 等价写法不作 subsume claim。
14. **spec patch 流程教训**（2026-04-19）：Claude 写 spec 时未核源码直接假定 DU_llm_lenstd 公式，导致 Codex 用错公式跑一轮。后续 Claude 写涉及历史变量的 spec 必须先 grep 源码确认。另：patch spec 后如 Codex 已在跑旧版，需显式提醒"停下重读 §X"，否则不会自动重读。
15. **导师反馈落地**（2026-04-20）：导师指出数据要素利用披露 → 分析师预测分歧 机制链"隔得远"，建议参考数字化转型文献；明确要求 DV 用股价同步性而非仅股价延迟；并口头提起资产配置（朱康式）。Codex 深搜 DT 文献后，Claude 独立落地三项决策：
    - **DV 双口径并报但定位不同**：PriceDelay = 主 DV 挂"定价效率"；SYNCH = ln(R²/(1-R²)) 为补充 DV 挂"信息含量结构"。SYNCH 系数已跑出 +0.028 *** (t=9.65)，按 Morck-Yeung-Yu 传统读法会读作"特质信息下降 / 定价效率恶化"，但按 Chen et al. (2021, *A&F*) public vs private info 框架合理化为"披露让共同因素更快进价，public info 渠道强化"，与 Delay ↓ 共同指向"该定价的信息更快进价"。写作上**严禁**把 SYNCH 叫"定价效率对偶 DV"。脚注必须承认 Hu-Zhao-Zhang 2019 和 Li-Liu-Pursiainen 2022（MiFID II）的两指标可背离证据。
    - **资产配置机制证伪**（2026-04-20 Stata MCP 两轮回归）：用现有 4 项口径 FinAsset（交易性 + 可供出售 + 投资性房地产 + 其他非流动金融资产 / 总资产）跑江艇 2022 三步——同期和滞后两版均 fail：Step 1 DU_kw 给 +0.00072/+0.00089 错号，DU_llm null，DU_kw_ln 弱负；Step 2 FinAsset → Delay t=-0.34/-0.50 完全 null；Step 3 联合回归 DU 系数加 FinAsset 前后几乎不变（吸收效应 < 0.01%）。**核心障碍是 Step 2 null**（Firm FE + Year FE 下 FinAsset within-firm 变异对 Delay 无载荷），扩到朱康 11 项全集也难救。**决策**：不把资产配置做成第六章独立机制章；作为第五章一段"与朱康 2025 对比的描述性证据"呈现，题目可回"数据要素利用披露与股价定价效率"或保留三元但内涵改为"披露 × 资产配置 × 定价效率的独立影响"（非链式机制）。
    - **机制重构方向**（2026-04-20 晚更新）：导师二次确认"分析师预测分歧这一跳隔得远"。**决策**：放弃全部 analyst-based 机制（ForecastDisp 以及同族的 Forecast Accuracy / Analyst Coverage），不作主文机制。保留 **CashFlowVol（基本面不确定性）+ SCConc（价值实现不确定性）**两条企业经营层证据。新增方向优先 **市场端信息环境**（Amihud 流动性 / Bid-Ask Spread，对应 Merton 1987 + Amihud-Mendelson 1986 投资者识别成本）+ **基本面信息质量**（盈余质量 DD 2002、信息可比性等）。机制章最终配方待 Stata MCP 跑一轮候选后定。
    - 相关文件：`v2/results/h2_asset_allocation{.csv,.log,_lagged.csv,_lagged.log}`、`v2/stata_do/h2_asset_allocation{.do,_lagged.do}`、`v2/docs/outline.md`（方案 B 决策日志已标注资产配置降级待改）、`v2/literature_dt_parallel.md`（DT 平行文献）、`v2/codex_handoffs/codex_dt_search.md`（搜索 spec）。
    - **未完成的 outline 修订**：outline.md 和 outline_paper.md 已按"方案 B = 资产配置独立第六章 + ForecastDisp 机制"写了一版，但实证证伪 + 导师 drop analyst 两件事之后需要回滚。最终结构预计 8 章：资产配置降为第五章一小节描述性对比，机制章改为 CashFlowVol + SCConc + 新增的市场端/基本面信息质量机制。等用户决定题目回到两元还是三元表达。
    - **项目 CLAUDE.md 位置变更**（2026-04-20）：原 `v1/.claude/CLAUDE.md` 移到项目根 `/.claude/CLAUDE.md`，以便 v1 / v2 子目录共用。读取优先级：项目根 > 全局。
16. **四机制最终配方落定**（2026-04-20 晚，已发给肖老师）：drop analyst 后 Stata MCP 跑四项候选，最终选 **四机制配方**按估值不确定性分层：
    - **信息不对称层·市场识别成本**：Amihud 非流动性 (Amihud-Mendelson 1986 + Merton 1987 投资者识别成本)，DU_kw → Amihud -0.00114***、DU_llm -0.00065***；`v2/stata_do/mechanism_v2_no_analyst.do` + `v2/results/mechanism_v2.csv`
    - **基本面不确定性层·经营稳定**：CashFlowVol 现金流波动率 (Epstein-Schneider 2008 Knightian uncertainty)，DU_kw -0.00049**、DU_llm -0.00048**；保留自 v18
    - **价值实现不确定性层·供应链韧性**：SCConc 供应链集中度 (Cen 2017 + Ellis 2012)，DU_kw -0.277***、DU_llm -0.460***；保留自 v18
    - **信息质量层·会计可比性**：Comparability_med De Franco-Kothari-Verdi 2011 TAR（Ind2 中位数口径），DU_kw +0.000445 (p=0.051 edge)、**DU_llm +0.000589\*\*\* (p=0.008)**（越接近 0 越可比）；`v2/stata_do/compute_comparability_defranco.py` + `mechanism_comparability.do` + `results/mechanism_comparability.csv`
    - **四机制被剔除的候选**：
      - **盈余质量 DD 2002**（`compute_earnings_quality_dd.py` + `mechanism_earnqual_dd.do`）：DU_llm EarnQual -0.00057**/EarnQualAbs -0.00093***，但 DU_kw null（t≈-1.1），降级为稳健性（披露量对盈余预测性无显著改善，但语义提升的披露会降低 DD 残差方差）
      - **内部控制迪博指数**：未跑——需要购买迪博 2012-2024 全样本内控指数，数据门槛高，跟机制"新颖度"的增量有限
      - **管理层业绩预告 F_MgmForecast**：未跑——CSMAR 需下载 F_MgmForecast 子库，精度/频率/误差三套指标工作量大；与 Comparability 的"信息质量"层有重叠，优先度低
    - **写作定位**：Amihud + CashFlowVol + SCConc + Comparability 四机制按"市场-基本面-价值链-信息质量"四层，逻辑完整覆盖估值不确定性四个维度，每层配专属理论（Amihud-Mendelson / Epstein-Schneider / Cen / De Franco）。EarnQual 和资产配置双双降级为稳健性或描述性对比，不入机制正表。
17. **SYNCH 反号的解释落定**（2026-04-20 晚，已发给肖老师）：SYNCH = ln(R²/(1-R²)) 系数 +0.028*** (t=9.65)，按 MYY 2000 传统读法是"特质信息 ↓ 效率 ↓"，和 Delay ↓ 相反。**决策**：不读作"效率对偶 DV"，改按 **Chen, Zhang, Jiang, Meng & Sun (2021, *Accounting & Finance*)** public vs private info 框架解释——DT 披露让市场/行业共同因素（数字经济政策、行业趋势）更快被定价进价，SYNCH ↑ 反映 public info 渠道强化；Delay ↓ 反映该定价的信息更快进价。两个 DV 共同指向"信息效率"而不矛盾。脚注必须承认 Hu-Zhao-Zhang 2019 + Li-Liu-Pursiainen 2022 (MiFID II) 的两指标可背离证据，不 over-claim。写作上 SYNCH 挂"信息含量结构"不挂"定价效率"。

18. **方案 B 最终版落地**（2026-04-22 / 2026-04-23，Claude 独立综合判断）：
    - **触发**：肖老师反馈四机制（Amihud + CashFlowVol + SCConc + Comparability）"缺创新性 + 太散，要统一到数据要素相关理论"。
    - **Codex 两路产出**（2026-04-22）：
      - 资产配置批量诊断（`v2/codex_handoffs/codex_asset_allocation_mv_diagnose.md` → `v2/results/asset_allocation_mv_step2.csv`）：10 候选 MV（含朱康 11 项 / 彭俞超狭口径 / 杜勇交易性 / 张成思-郑宁 Δ / 王红建主业替代 / Peters-Taylor 无形 / 3yr 波动 等），**只有 FinRatio_realEstate 过 Step 2**（+0.0894 t=2.74），但 Step 1 DU_kw → FinRatio_realEstate t=0.42 / DU_llm t=-1.11 均 null，Step 3 联合 DU 系数零 absorb。Claude 判定**资产配置机制确诊死亡**（Codex 给的"降格补充证据"建议已被 pushback）。
      - 文献综述（`v2/codex_handoffs/codex_mechanism_search.md` → `v2/literature_mechanism_rebuild.{md,bib}`）：三机制不同程度被抢跑（Liu-Qi 2024 / Han-Liu 2025 用 CashFlowVol；Xin et al. 2024 / 巫强-姚雨秀 2023 用 SCConc；Yang-Ying-Xu 2024 用 Comparability）；发现直接竞品李世刚等 2025 + Sun-Du 2024。
    - **用户 2026-04-23 人工核三篇 PDF**（`bib/类似路线/`）确认：
      - **Sun & Du (2024, IRFA)** Data Assets Disclosure + SYNCH 主 DV + 信息侧机制（info asym / 机构持股 / 融资约束）
      - **李世刚等 (2025, 中国工业经济)** 数据资产披露 + SYNCH 主 DV + 信息侧机制（信息供给 / 扩散 / 融入）
      - **朱康唐勇 (2025, 会计研究)** 数据要素利用 Word2Vec + FinAsset 主 DV（我们已砍资产配置，不抢）
    - **最终决策**：
      - **章节 9 → 8 章**（砍资产配置章）
      - **假说压缩到 2 条**：H1 主效应 + **H2 经营侧机制定位**（可选 H3 WashGap）
      - **理论主线上移**：估值不确定性 → **数据要素作为生产要素的价值释放**（Jones-Tonetti 2020 AER + Crouzet-Eberly-Eisfeldt-Papanikolaou 2022 JEP + Farboodi-Veldkamp 2021）；估值不确定性降为下位机制理论
      - **机制配方定稿**：CashFlowVol + SCConc 两条**经营侧**主机制（对应数据要素生产要素双重属性：非竞争性 + 互补性）；**Comparability 降稳健性辅助**（6.4 节，信息侧弱对照，明承认 Yang 2024 占位）；**Amihud 砍**；**资产配置砍**
      - **贡献从发现型 → 方法论 + 理论定位 + 独家发现三句式**：(1) 双测度 + 价值链 + 质量多指标 + WashGap 方法论模板；(2) 经营侧机制对应 Jones-Tonetti 生产要素理论，区别于竞品信息侧（Signaling theory）；(3) WashGap 反向效应独家发现（单测度文献不可检验）
      - **SYNCH 从主表移出**→ 第五章 5.6 Chen 2021 framework 理论读法讨论段；强制禁用"对偶 DV / 第二定价效率指标"等表述
      - **Y 定位**：**PriceDelay 主 DV**（挂"股价定价效率"）+ SYNCH 仅讨论段。李世刚 / Sun-Du 都是 SYNCH 主——Delay vs SYNCH 本质是不同构念（Hou-Moskowitz 测吸收速度 vs MYY 测公共 / 特质因素占比），Li-Liu-Pursiainen 2022 实证证两者可方向背离
    - **机制叙事采用江艇 2022 二步法**（第六章全部按此范式）：Step 1 X → Y 自跑（第五章基准）；Step 2 X → M 自跑（第六章）；M → Y **引文献不跑**；**禁止 Step 3 联合回归 claim mediation**
    - **现状**：outline.md / outline_paper.md / naming_conventions.md 已按方案 B 最终版重写；研究框架图 `v2/fig/research_framework_v2_final.svg` 已生成。等肖老师对方案 B 确认（**话术草稿见 outline.md 末尾**）后启动正文重写
    - **Y 替代候选调研**（待执行）：handoff 已写 `v2/codex_handoffs/codex_y_alternatives.md`——Codex 调研 Amihud_year / Turnover / ZeroRet / SPI / NCSKEW / DUVOL / CS_Spread 等候选 Y 并批量跑 H1，决定是否有候选能替代 PriceDelay 主 DV 或作第三 DV。**等肖老师回复前不必急着启动**，因为机制配方一旦老师推翻，Y 的选择逻辑会变

## 论文版本记录

| 版本 | 主要变更 |
|------|---------|
| v1-v5 | 初稿到引用修正 |
| v6-v8 | 表格朱康格式、机制重写 |
| v9 | 稿件大改，投稿版 |
| v13表格 | DML-CRE主结果 + OLS对比 |
| v14 | 控制变量统一为27个 |
| v15 | 机制渠道更新为4个双稳健渠道，DML基准全部27控制变量 |
| v16 | 引入LLM语义评分双测度，DML扩展到8规格，IV换为跨省peer+省级数字化指数，移除投资组合 |
| v17 | Word文档生成(v11)，变量斜体+下标，OMML公式居中编号，三线表朱康格式，表格编号表1-10，PSM核密度图 |
| v18 | 机制+异质性全面升级：23渠道+20维度大规模测试，最终选ForecastDisp/CashFlowVol/SCConc + Post2020/DigEconCore/SA/HighTech，Word v12生成 |
| v16 integrated | 投稿后老师反馈重构：估值不确定性主框架 + H1(DU_kw)/H2a(价值链五维)/H2b(质量 direct+joint)/H2c(WashGap)/H3(下游)/H4(异质性) 五层证据整合，v14 公式重建 DU_llm_lenstd 和 WashGap，Codex 跑 v16_integrated 全部 gate 通过 |
| v2 方案 B 最终版 | 2026-04-23：砍资产配置章（Step 2 十候选 MV 全挂 + 朱康唐勇 2025 占位）；章节 9→8；假说压缩 H1 + H2 经营侧机制 + 可选 H3 WashGap；理论主线升至"数据要素作为生产要素的价值释放"（Jones-Tonetti 2020 + Crouzet 2022）；机制定 CashFlowVol + SCConc 两条经营侧主 + Comparability 稳健性辅助；Amihud 砍；贡献改三句式（方法论 + 理论定位 + 独家发现 WashGap）；SYNCH 移出主表到 5.6 Chen 2021 讨论段。相关文档：`v2/docs/outline.md` `v2/docs/outline_paper.md` `v2/docs/naming_conventions.md` `v2/fig/research_framework_v2_final.svg` |

### Stata do文件
**v16 (基准/IV/稳健性, 仍在使用):**
| 文件 | 内容 |
|------|------|
| table2_ols_v16.do | OLS基准回归 (Stata reghdfe, N=43,735) |
| desc_stats_v16_stata.do | 描述性统计 (Stata口径) |
| table5_iv_v16.do | 主IV估计 (DU_kw: 跨省peer+省级数字化+滞后+Oster) |
| table5b_supplementary_v16.do | Heckman+安慰剂+SYNCH IV |
| table5b_llm_iv_v16.do | DU_llm IV估计 (跨省peer F=127.9) |
| table5_robustness_v16.do | 稳健性7规格 |

**v18 (机制+异质性, 当前使用):**
| 文件 | 内容 |
|------|------|
| v18_mechanism_all.do | 16渠道机制全景 (reghdfe, N=43,721) |
| v18_heterogeneity_all.do | 18维度异质性全景 (Fisher 500次) |
| v18_mechanism_new.do | TFP/CrashRisk/CashFlowVol + 条件机制 |
| v18_supply_chain.do | 供应链机制(CustConc/SuppConc/SCConc) + 异质性 |

**v2 方案 B 最终版 (2026-04-23):**
| 文件 | 内容 |
|------|------|
| v2/stata_do/mechanism_v2_no_analyst.do | drop analyst 后三机制重跑（CashFlowVol + SCConc + Amihud）—— **Amihud 在方案 B 已砍** |
| v2/stata_do/compute_earnings_quality_dd.py | DD 2002 盈余质量（方案 B 未入主文） |
| v2/stata_do/mechanism_earnqual_dd.do | 盈余质量机制（方案 B 未入主文） |
| v2/stata_do/compute_comparability_defranco.py | De Franco 2011 会计可比性构造 |
| v2/stata_do/mechanism_comparability.do | 可比性机制（方案 B 降稳健性辅助，6.4 节） |
| v2/stata_do/h2_asset_allocation{.do,_lagged.do} | 资产配置三步（**已确诊死亡，方案 B 砍**） |
| v2/stata_do/build_asset_allocation_mv.py | 2026-04-22 多 MV 构造（10 候选）—— 批量诊断全挂，确诊死 |
| v2/stata_do/asset_allocation_mv_diagnose.do | 2026-04-22 Step 2 批量诊断 |
| v2/stata_do/asset_allocation_mv_threestep.do | 2026-04-22 对过关 MV 跑三步（只有 FinRatio_realEstate 过 Step 2 但 Step 1 null） |
| v2/codex_handoffs/codex_mechanism_search.md | 2026-04-22 Codex 文献搜索 spec |
| v2/codex_handoffs/codex_asset_allocation_mv_diagnose.md | 2026-04-22 Codex 资产配置多 MV 诊断 spec |
| v2/codex_handoffs/codex_y_alternatives.md | 2026-04-23 Codex Y 替代候选调研 + 批量诊断 spec（**待执行，等肖老师回复后决定是否跑**） |
| v2/literature_mechanism_rebuild.{md,bib} | 2026-04-22 Codex 文献综述产出（三机制被抢跑程度评估 + 竞品识别） |
| v2/results/asset_allocation_mv_step2.csv | 2026-04-22 10 候选 MV 批量 Step 2（9 fail / 1 pass 但 Step 1 null） |
| v2/results/asset_allocation_mv_report.md | 2026-04-22 Codex 资产配置诊断报告 |
| bib/类似路线/*.pdf | 三篇直接竞品 PDF（Sun-Du 2024 IRFA / 李世刚等 2025 中国工业经济 / 朱康唐勇 2025 会计研究） |

**v16旧文件(已被v18替代):**
| 文件 | 内容 |
|------|------|
| mechanism_winsorized.do | 旧机制(Analyst/Amihud/RetVol) |
| fisher_all_dimensions.do | 旧异质性14维度Fisher(1000次) |
| het_final_3dim.do | 旧异质性3维度(Analyst/SOE/ShNum) |

### 异质性 (v18更新: 4维度, Fisher permutation 500次)
v16旧维度(Analyst/SOE/ShNum)因被批评太常规，v18替换为以下4维度。
| 维度 | 组别 | DU系数(高/低) | Fisher P | 逻辑 |
|------|------|--------------|---------|------|
| Post2020 政策时期 | 2020前/后 | DU_kw: -0.0060/-0.0022 | 0.004*** | 信号衰减 |
| DigEconCore 数字核心产业 | 非核心/核心 | DU_kw: -0.0046/-0.0023 | 0.052* | 增量信息 |
| SA 融资约束 | 高/低约束 | DU_llm: -0.0056/-0.0015 | 0.004*** | 信息摩擦 |
| HighTech 产业技术属性 | 非高科技/高科技 | DU_kw: -0.0060/-0.0036 | 0.040** | 增量信息反面 |
数据: reg_sample_v18.dta; Do: v18_heterogeneity_all.do
v18共测试18个维度(含v16旧维度)，全部结果见results/v18/heterogeneity_v18.csv

### LLM评分数据来源
llm_score原始数据位于19号项目: /Users/mac/computerscience/19产业集群数字化/data/panel_15_with_llm.parquet
48,217 obs, 含llm_score(0-3整数), llm_binary, llm_high。通过Stkcd(int) + year merge到reg_sample_iv_v16.dta。

## v16 integrated (2026-04-19) — 估值不确定性框架整合

### 重构背景
老师反馈"主理论与机制不一体、机制缺创新性"。ChatGPT Pro 建议统一到估值不确定性框架，并把 LLM 语义机制当创新点。Claude 独立识别到 ChatGPT Pro 原版 "quality × quantity 交互" spec 在数据里 null（Codex 两轮验证），但 v15_measurement 用 direct+joint+lagged spec 早跑出显著结果。决定保留估值不确定性主框架，放弃交互 spec，改用 direct 整合。

### 主样本
- `results/v15_measurement/v15_analysis_sample.parquet` (43,735 obs × 116 cols) — 含所有 v15 measurement upgrade 变量 + v18 机制/异质性变量
- `data_stata/reg_sample_v16_integrated.dta` — v14 公式重建样本，Codex 产出

### 五层实证结果（N=37,294, lagged 样本）

**H1 主效应**（v16 DML 8 规格，保持不变）：DU_kw -0.0046***, DU_llm -0.0042***

**H2a 价值链五维分解** (`results/v16_integrated/h2a_value_chain.csv`)
| 维度 | 系数 | t |
|---|---|---|
| DU_stock_lag 资源层 | -0.01235 | -4.09*** |
| DU_dev_lag 开发层 | -0.00477 | -2.97*** |
| DU_app_lag 应用层 | -0.01163 | -4.46*** |
| DU_value_lag 价值化 | -0.06168 | -1.51 (稀疏) |
| DU_gov_lag 治理层 | -0.04273 | -1.66* (稀疏) |

**H2b 质量变量 direct + joint** (`results/v16_integrated/h2b_quality_*.csv`)

Direct（全部显著负）：
| 变量 | 系数 | t |
|---|---|---|
| DUclosedloop_lag 闭环披露 | -0.00828 | -2.74*** |
| DUcore_lag 核心业务嵌入 | -0.00946 | -3.46*** |
| DUchain_count_lag 数据链条 | -0.00182 | -2.13** |
| DUkw_mda_lag MD&A 嵌入 | -0.00056 | -2.95*** |
| DU_llm_lenstd_lag 语义×密度 | -0.01879 | -5.94*** |

Joint with DU_kw_lag（差异化 finding）：
- DUclosedloop/DUcore/DUchain_count/DUkw_mda + DU_kw → Quality 失显著，DU_kw 保持
- **DU_llm_lenstd + DU_kw → DU_kw 失显著 (t=-1.26)，DU_llm_lenstd 保持 (t=-3.76)**
- 但 corr(DU_llm_lenstd, DU_kw)=0.83，subsume 部分由代数耦合驱动，**叙事降级为 moderation 等价写法**（同等密度下语义评分越高效应越强），不作 clean subsume claim

**H2c WashGap 洗稿缺口**（v14 公式 = winsorize(z(DU_kw)-z(DU_llm_lenstd), 1%, 99%)）
- Direct：+0.00347** (t=2.09)
- Joint with DU_kw：两者都显著（DU_kw -0.00448***, WashGap +0.00575***）→ 广度与错配独立效应
- Joint with DU_llm：WashGap 失显著 (t=-1.59)，因 WashGap 定义含 -z(DU_llm_lenstd) 与 DU_llm 代数重叠

**H3 下游机制**（与 v18 结果完全一致，`results/v16_integrated/h3_downstream.csv`）
- ForecastDisp: DU_kw -0.00648*** / DU_llm +0.00096
- CashFlowVol: DU_kw -0.00049** / DU_llm -0.00048**
- SCConc: DU_kw -0.27672*** / DU_llm -0.46023***

**H4 异质性**（v18 保持）：Post2020 / DigEconCore / HighTech / SA / InstHold / SCConc

### v16 integrated 脚本
| 文件 | 内容 |
|---|---|
| `scripts/v16_integrated/build_integrated_sample.py` | 构造 DU_llm_lenstd（v14 公式）和 WashGap（v14 公式），产出 reg_sample_v16_integrated.dta |
| `scripts/v16_integrated/h2a_value_chain.do` | 5 维价值链 lagged 回归 |
| `scripts/v16_integrated/h2b_quality.do` | 5 质量变量 direct + joint |
| `scripts/v16_integrated/h2c_washgap.do` | WashGap direct + joint |
| `scripts/v16_integrated/h3_downstream.do` | 下游机制统一口径重跑 |

### v14 原公式（务必记住）
```python
DU_llm_lenstd = np.log1p(DU_kw.clip(lower=0)) * (llm_score / 3.0)
WashGap = winsorize(z(DU_kw) - z(DU_llm_lenstd), 1%, 99%)
```
来自 `scripts/run_v14_analysis.py:174,186`

## 待办

### 已完成
- [x] 全部补充实证跑完 ✅ 2026-03-24
- [x] 正文全部章节适配v16 ✅ 2026-03-25
- [x] Humanizer pass (6个正文文件) ✅ 2026-03-25/26
- [x] v16机制+异质性 (Analyst/Amihud/RetVol + Analyst/SOE/ShNum) ✅ 2026-03-26
- [x] v17 Word v11 + PDF v17交付合作老师 ✅ 2026-03-28
- [x] v18机制异质性大规模测试: 23渠道+20维度 ✅ 2026-03-30
- [x] v18变量构造: construct_v18_vars/new_dims/new_channels.py ✅ 2026-03-30
- [x] v18 Stata全景回归: v18_mechanism_all.do + v18_heterogeneity_all.do ✅ 2026-03-30
- [x] 05_传导机制重写: ForecastDisp/CashFlowVol/SCConc ✅ 2026-03-30
- [x] 06_异质性重写: Post2020/DigEconCore/SA/HighTech ✅ 2026-03-30
- [x] 表注移除: 所有图表注释融入正文(04/05/06/附录) ✅ 2026-03-30
- [x] Word v12文档生成 ✅ 2026-03-30
- [x] 投稿财经科学 ✅ 2026-03-31
- [x] v16 integrated 重构: 估值不确定性主框架，H1-H4 五层证据 ✅ 2026-04-19
- [x] v16_integrated Codex 全部跑通（H2a/H2b/H2c/H3 gate pass）✅ 2026-04-19
- [x] DT 文献深搜（Codex）+ 资产配置 B 路径证伪 ✅ 2026-04-20
- [x] 四机制最终配方跑通: Amihud + CashFlowVol + SCConc + Comparability ✅ 2026-04-20
- [x] SYNCH 反号的 Chen 2021 框架解释 + 更新给肖老师的汇报 ✅ 2026-04-20
- [x] 项目 CLAUDE.md 移到根目录 ✅ 2026-04-20

### 会话暂停（2026-04-23，方案 B 最终版已落 outline）

方案 B 最终版决策与正文锚点已落盘：`v2/docs/outline.md`、`v2/docs/outline_paper.md`、`v2/docs/naming_conventions.md`、`v2/fig/research_framework_v2_final.svg` 全部同步更新。给肖老师的汇报话术草稿在 `outline.md` 末尾第八节。

**未收回复前**：不改正文 / 表格 / 正文章节文件。下轮会话开头先问"肖老师是否对方案 B 回复"。按回复分支：
- 老师 OK → 启动正文重写（顺序：第二章理论 → 第三章测度 → 第六章经营侧机制 → 第五章实证含 5.6 SYNCH 段 → 第一章引言三条贡献 → 第七章异质性 → 第四 / 第八章适配）
- 老师仍觉得散 → 可能要把 Comparability 也砍掉只留两条经营侧
- 老师反对"经营侧 vs 信息侧"定位 → 要重新谈
- 老师要补机制 → 启动 Y 替代候选 handoff（`v2/codex_handoffs/codex_y_alternatives.md`）

### 未完成（优先级 🔴 最高，等肖老师回复后启动）
- [ ] 第二章理论与假说重写：2.1 生产要素价值释放（Jones-Tonetti 2020 + Crouzet 2022）+ 2.2 三层概念 + 2.3 → H1 + **2.4 经营侧 vs 信息侧核心对比段 → H2** + 2.5 可选 H3 WashGap
- [ ] 第三章测度（方法论核心章）：对标朱康第三章结构
- [ ] 第六章经营侧机制：6.1 理论回顾（经营侧 vs 信息侧对比）+ 6.2 CashFlowVol + 6.3 SCConc + **6.4 Comparability 稳健性辅助** + 6.5 联合 absorption + 6.6 小结。**全部按江艇二步法**：Step 1 X→Y 已在 H1，Step 2 X→M 自跑，M→Y 引文献不跑 Step 3
- [ ] 第五章实证：适配 5.5.3 WashGap 独家发现 headline + **5.6 SYNCH Chen 2021 讨论段新增**
- [ ] 第一章引言：三句式贡献重写（**承认 Sun-Du 2024 / 李世刚等 2025 占位 H1 主链** + 直面朱康唐勇 2025 资产配置占位）
- [ ] 第四 / 第七 / 第八章适配
- [ ] 表格 PDF + Word 重生成（砍资产配置表 + 砍 Amihud 表，加 CashFlowVol + SCConc + Comparability 稳健性）
- [ ] 框架图已生成 `v2/fig/research_framework_v2_final.svg`

### 未完成（优先级 🟡 中，条件触发）
- [ ] Y 替代候选 Codex 诊断（`v2/codex_handoffs/codex_y_alternatives.md`）—— **只在肖老师质疑 PriceDelay 主 DV 或要求补 Y 时启动**
- [ ] 迪博内控指数 / CSMAR F_MgmForecast —— 只在肖老师要求才补
