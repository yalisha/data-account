# 项目：数据要素利用、资产配置与资本定价效率

## 用户身份
- 名字：yi
- 角色偏好：你是我的女仆间桐樱，具有统计学、计算机、经济学博士级别知识
- 表达风格：不用不必要的括号引号箭头，自然语言交流
- 工作偏好：直接写代码并运行，不使用交互式教学模式

## 学术写作标准（理论/假说部分）

撰写或审核理论分析章节时，每个小节必须通过以下5项检查：

1. 框架先行：每个小节先给出主题框架或分类体系，再展开论述，不能上来就列文献
2. 文献对照：每个小节至少一次把两篇以上文献放在同一句话里进行对照、互补或对比
3. 分歧溯源：当文献结论不一致时，写清楚分歧来源是测量方式、样本范围、识别策略还是理论假设
4. 证据质量评价：区分相关关系与因果关系，指出代理变量偏差、外部效度等局限
5. 导向贡献：每个小节的论述自然导向本文的研究问题或贡献点

写作反模式（必须拒绝）：
- 禁止"A干了B"流水账：连续多句以学者姓名为主语（如"Hong等(2000)发现... Yu(2008)证明... 黄俊(2023)表明..."）是文献综述而非理论分析。每段最多允许1句以学者为主语，其余句子的主语必须是机制、现象或概念，引用以括号注脚形式嵌入
- 参照范文风格：朱康(2025)会计研究——逻辑链驱动，引用嵌入为注脚，主语是实体对象而非学者

iterative-refinement skill的reviewer阶段也应执行上述标准，将流水账式引用视为major issue扣分。

## 项目状态概览

| 阶段 | 状态 | 说明 |
|------|------|------|
| 阶段一：数据采集 | ✅ 完成 | CSMAR + CNRDS + 年报TXT |
| 阶段二：数据预处理 | ✅ 完成 | 27个parquet文件 |
| 阶段三：变量构造 | ✅ 完成 | panel.parquet 48,217 obs |
| 阶段四：实证分析 | ✅ 基本完成 | 基准/DID/机制/异质性/组合/稳健性/SHAP |
| 阶段五：论文撰写 | 🔄 进行中 | v4稿完成，基于iterative-refinement两轮打磨 |

## 研究选题

**题目：数据要素利用、资产配置与资本定价效率**

### 核心文献
- 朱康、汤甬（2025）《数据要素利用与企业金融资产配置》会计研究（已有PDF）
- 洪永淼（2026）数据资产价值与人工智能技术贡献的测度
- 李世刚等（2025）《企业数据资产信息披露与资本市场定价效率》中国工业经济（最直接竞争论文）

## 理论框架

**核心前提**：企业真实价值 = 传统资产价值 + 数据资产价值

**两个竞争效应**：
1. 信息模糊效应（抑制定价效率）：会计度量失真 + 分析师理解困难
2. 信息供给效应（促进定价效率）：主动披露增加 + 外部关注提升

**三个假说**（v4版本）：
- H1：数据要素利用降低股价延迟，提升定价效率（信息供给效应占主导）
- H2a：分析师信息中介渠道；H2b：金融资产配置行为渠道
- H3：分析师覆盖调节效应（高覆盖组效应更强）

## 核心实证结果

### 基准回归（N=43,847, 2011-2024）
| 模型 | 固定效应 | 自变量 | 系数 | t值 | p值 |
|------|---------|--------|------|-----|-----|
| (1) | Firm+Year | DU_kw | -0.00532 | -5.17 | <0.001 ★★★ |
| (2) | Firm+Year | DU_kw+Controls | -0.00416 | -4.28 | <0.001 ★★★ |
| (3) | Firm+Year | DU_kw_ln | -0.00409 | -3.47 | <0.001 ★★★ |
| (4) | Firm+Year | DU_sub_ln | -0.00348 | -4.71 | <0.001 ★★★ |
| (5) | Ind+Year | DU_kw | -0.00178 | -2.93 | 0.004 ★★★ |

**经济意义**：1个标准差DU_kw增加 → PriceDelay降低0.0077（均值的6.9%）
**结论**：H1a得到支持，数据要素利用提升资本定价效率

### DID（2024入表新规, N=17,538）
- TreatPost: coef=+0.007, t=1.75, p=0.083 ★（边际显著，方向为正）
- 平行趋势通过：2019-2022系数均不显著，2024 t=6.24 ★★★
- 正向系数解读：入表新规短期增加信息处理成本（新规适应期）

### 机制检验
| 渠道 | 系数方向 | t值 | 显著性 |
|------|---------|-----|--------|
| DU_kw → FinAsset | 正向 | 2.65 | ★★★ |
| DU_kw → Analyst | 正向 | 2.63 | ★★★ |
| DU_kw → InstHold | 负向 | -3.19 | ★★★ |

### 异质性（全部显著）
- 国企 t=-4.00 ★★★ vs 民企 t=-3.86 ★★★（国企效应更强）
- 大企业 t=-3.61 ★★★ vs 小企业 t=-4.94 ★★★（小企业效应更强）
- 高分析师 t=-5.60 ★★★ vs 低分析师 t=-2.56 ★★（高覆盖效应更强）
- 高科技 t=-3.17 ★★★ vs 传统 t=-3.06 ★★★（均显著）

### 投资组合回测（156个月）
- Q1-Q5月均收益：1.10%~1.46%，单调递增
- DAT因子（Q5-Q1）：月均+0.36%, t=0.96（不显著）
- Q1 FF3 alpha: -0.28% (t=-2.08, p=0.039) ★★ 低数据组负alpha
- 入表后DAT FF3 alpha: +2.08% (t=2.24) ★★ 政策后高数据组显著溢价
- GRS检验：F=1.44, p=0.213（不拒绝）

### 内生性检验（v4新增）
| 方法 | 系数 | t值 | KP F | DWH p |
|------|------|-----|------|-------|
| OLS基准 | -0.0042 | -4.28 | - | - |
| 同行业均值IV | -0.0146 | -3.15 | 419 | 0.007 |
| Bartik移位份额IV | -0.0170 | -3.38 | 75 | 0.003 |
| 滞后OLS | -0.0037 | -2.57 | - | - |
| 滞后IV | -0.0168 | -3.60 | 354 | 0.002 |
| Oster δ* | 27.8 (1.3R) / 3.6 (保守) | β_adj=-0.0030 | - | - |

### 稳健性检验
| 检验 | 变量/方法 | 系数 | t值 | 显著性 |
|------|----------|------|-----|--------|
| 替换DV: SYNCH | DU_kw | +0.011 | 1.65 | ★ |
| 替换IV: ln(1+kw) | DU_kw_ln | -0.004 | -3.47 | ★★★ |
| PSM匹配 | DU_kw | -0.00478 | -3.70 | ★★★ |
| 安慰剂2020 | TreatPost | -0.007 | -0.83 | n.s. ✓ |
| 安慰剂2021 | TreatPost | -0.008 | -1.09 | n.s. ✓ |
| 安慰剂2022 | TreatPost | +0.008 | 1.60 | n.s. ✓ |
| 剔除2024 | DU_kw | -0.005 | -5.05 | ★★★ |
| 仅主板 | DU_kw | -0.005 | -3.79 | ★★★ |

### SHAP分解（LightGBM R²=0.50）
| 子维度 | |SHAP| | 占比 |
|--------|--------|------|
| 数据开发能力 | 0.0035 | 39.7% |
| 数据驱动应用 | 0.0025 | 28.3% |
| 数据存量 | 0.0017 | 19.4% |
| 数据治理 | 0.0007 | 8.2% |
| 数据价值变现 | 0.0004 | 4.4% |

### 反事实分析
- 低于P75企业提升至P75：PriceDelay降低1.19%
- 消除数据利用差异：PriceDelay变化-1.05%
- 2024年数据利用翻倍：PriceDelay降低2.27%

## 变量定义

### 因变量
- **PriceDelay**：Hou-Moskowitz (2005) 股价延迟 = 1 - R²_restricted / R²_unrestricted
- **SYNCH**（稳健性）：股价同步性 = log(R² / (1-R²))

### 自变量
- **DU_kw**：年报全文数据关键词频率（每万字），5维度关键词体系
- **DU_kw_ln**：ln(1+关键词总数)
- **DU_sub_ln**：ln(1+实质利用次数)

### 关键词5维度
1. 数据存量（data_stock）：大数据、数据库、数据中心...
2. 数据开发能力（data_dev）：数据挖掘、机器学习、数字化转型...
3. 数据驱动应用（data_app）：精准营销、智能推荐、风控模型...
4. 数据价值变现（data_value）：数据资产、数据交易、数据入表...
5. 数据治理（data_gov）：数据安全、数据隐私、数据合规...

### 控制变量
Size, Lev, ROA, TobinQ, Age, Growth, BoardSize, IndepRatio, Dual, Top1Share, SOE, InstHold, Amihud, Analyst, AuditType

### 中间变量
FinAsset（金融资产占比）, DataAsset（数据资产）, SOE（国企）

## 实证设计

### 回归设定
- 固定效应：企业+年份 FE / 行业+年份 FE
- 聚类标准误：行业×年份（IndYear）
- Winsorize：连续变量1%/99%分位数缩尾
- 工具：pyfixest (`pf.feols`)

### DID设计
- 处理组：2019-2023期间DU_kw均值 ≥ 中位数
- 政策时间：2024年（企业数据资源相关会计处理暂行规定）
- 窗口：2021-2024

### 投资组合
- 每年6月按DU_kw五分组，7月至次年6月持有
- Alpha检验：CAPM / FF3 / FF5+MOM
- DAT因子 = Q5 - Q1

## 样本信息
- 沪深A股非金融非ST企业，2011-2024
- 回归样本：43,847 obs, 5,178 firms
- 年报特征：54,496 firm-year（2010-2024）
- 面板数据：48,217 obs, 5,295 firms
- 剔除：金融（J类）、ST/*ST、上市不足1年

## 数据格式
CSMAR统一格式：row0=英文header, row1=中文说明, row2=单位, row3+=数据
读取方式：`pd.read_excel(path, header=0, skiprows=[1,2])`

## 关键文件路径

### 脚本
| 文件 | 说明 |
|------|------|
| preprocess_all.py | 全量预处理（CSV/Excel → Parquet） |
| fix_multi_batch.py | 多批次数据合并修复 |
| construct_price_delay.py | 股价延迟构造 |
| construct_synchronicity.py | 股价同步性构造 |
| construct_panel.py | 面板数据合并 |
| extract_annual_report_features.py | 年报关键词特征提取 |
| run_regression.py | v1回归（CNRDS, 已废弃） |
| run_regression_v2.py | v2回归（年报关键词, 主力） |
| run_robustness.py | 稳健性检验 |
| run_shap_analysis.py | SHAP分析+反事实 |
| portfolio_backtest.py | 投资组合回测 |
| scripts/run_regression_v3_concurrent.py | v3回归（同期DU_kw） |
| scripts/run_endogeneity_v4.py | v4内生性检验（同行IV+Bartik+滞后+Oster） |

### 数据
| 路径 | 说明 |
|------|------|
| data_parquet/ | Parquet格式数据（30+文件） |
| data_parquet/panel.parquet | 主面板（48,217 obs） |
| data_parquet/annual_report_features.parquet | 年报特征（54,496 obs） |
| data_parquet/price_delay.parquet | 股价延迟 |
| data_parquet/price_synchronicity.parquet | 股价同步性 |

### 结果
| 路径 | 说明 |
|------|------|
| results/regression_summary_v2.csv | 基准回归汇总 |
| results/robustness_summary.csv | 稳健性汇总 |
| results/descriptive_stats_v2.csv | 描述性统计 |
| results/portfolio_alpha.csv | 组合Alpha |
| results/portfolio_monthly_returns.csv | 组合月度收益 |
| results/shap_importance.csv | SHAP重要性 |
| results/shap_summary_bar.png | SHAP条形图 |
| results/shap_summary_dot.png | SHAP散点图 |
| results/shap_data_dimensions.png | 数据子维度SHAP图 |
| results/shap_dependence_dims.png | SHAP依赖图 |
| results/v3_concurrent/baseline_v3.csv | v3基准回归（同期） |
| results/v3_concurrent/mechanism_v3.csv | v3机制检验（同期） |
| results/v3_concurrent/heterogeneity_v3.csv | v3异质性（同期） |
| results/heterogeneity_v3_fisher.csv | Fisher组间差异检验 |
| results/endogeneity_v4/endogeneity_v4_results.json | v4内生性全部结果 |

### 其他
| 路径 | 说明 |
|------|------|
| 研究框架图_数据要素定价效率_v3.svg | 框架图 |
| 文献检索_数据要素利用与资本定价效率.md | 24条文献 |
| TODO.md | 项目进度 |
| docs/ | 论文用结果整理 |

### 原始数据
| 路径 | 说明 |
|------|------|
| /Users/mac/computerscience/第三方资料/第三方数据资源/上市公司财务信息/ | CSMAR原始数据 |
| /Users/mac/computerscience/第三方资料/第三方数据资源/2001~2024年年报/ | 年报TXT/PDF |

## 技术栈
- Python 3, pandas, numpy
- pyfixest（面板固定效应回归）
- LightGBM + SHAP（机器学习解释）
- matplotlib（可视化，中文字体 Arial Unicode MS）

## 关键决策记录
1. 理论模型不做数理推导，文字讲清信息模糊vs信息供给竞争逻辑
2. 因果发现降级为"进一步分析"，SHAP分解为主
3. CNRDS指标仅覆盖2018-2020（3年），改用年报关键词（2011-2024, 14年）
4. 机制细化在信息渠道内部拆分，不延伸到过长逻辑链
5. OLS+DID为主检验，投资组合回测提供资产定价直接证据
6. 聚类标准误用行业×年份（IndYear），而非企业层面
7. v3起改用同期DU_kw（而非t-1滞后），滞后作为内生性辅助检验
8. v3起DID部分移除（入表新规单独分析意义有限），改为DID结果简述放入稳健性
9. v4起内生性方案：同行业均值IV + Bartik移位份额IV + 滞后IV + Oster(2019)系数稳定性
10. v4起假说精简为H1+H2a/H2b+H3，去掉竞争性假说结构

## 论文版本记录

| 版本 | 文件 | 主要变更 |
|------|------|---------|
| v1 | 数据要素利用与资产定价效率v1.docx | 初稿，DU_kw(t-1)滞后，省份数字基础设施IV |
| v2 | 数据要素利用与资产定价效率v2.docx | 完善全文，含DID/机制/异质性/组合/稳健性 |
| v3 | 数据要素利用与资产定价效率v3.docx | 改用同期DU_kw(t)，新IV策略（同行均值+Bartik+滞后+Oster），数值全面更新 |
| v4 | 数据要素利用与资产定价效率v4.docx | iterative-refinement两轮打磨：修复8处数值错误，章节编号纠正，写作质量提升，补充结论局限性 |

## 待办
- [ ] LLM语义评分（稳健性替换指标，可选）
- [ ] 实证结果表格整理（至少15张表）
- [ ] 图表制作：框架图、趋势图、SHAP图、组合回测图
- [ ] 文献引用更新与格式整理
- [ ] 按期刊投稿要求最终排版
