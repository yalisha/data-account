# 数据要素利用、资产配置与资本定价效率 — 项目进度

## 阶段一：数据采集 ✅

- [x] 确认年报数据格式 — PDF和TXT两种格式都有，TXT完整覆盖2001-2024，PDF缺2011/2015/2016
- [x] 确认财务信息数据覆盖范围
- [x] 从CSMAR下载核心财务报表：资产负债表、利润表、现金流量表、所有者权益变动表
- [x] 下载交易数据：日个股回报率(2011-2026三批)、月个股回报率
- [x] 下载市场指数：综合指数 + 行业指数日行情(四批)
- [x] 下载因子数据：三因子(日)、五因子(日)、动量因子
- [x] 下载公司特征：基本信息年度表、每股指标、相对价值指标
- [x] 下载治理数据：治理综合信息、高管人数持股薪酬表、股权性质、十大股东、股本结构
- [x] 下载机构持股：机构持股分类统计表
- [x] 下载分析师数据：分析师预测指标文件(两批)、分析师评级及变动(两批)
- [x] 下载流动性：个股Amihud指标(日)(三批)
- [x] 下载审计意见表
- [x] 从CNRDS下载：企业数据要素开发利用指数、开发利用情况、各省份大数据发展指数、DMSP灯光数据

### 数据缺口（暂不影响主分析）

- [ ] 工具变量数据：城市互联网基础设施（可用省份大数据发展指数暂替）
- [ ] 2025年年报（尚未披露）
- [ ] 无风险利率序列（可从因子数据反推或后续补充Shibor）

## 阶段二：数据预处理 ✅

- [x] 初步检查所有数据文件的格式、字段和覆盖范围
- [x] 确认CSMAR表统一格式：row0英文header, row1中文说明, row2单位, row3+数据
- [x] 编写全量预处理脚本 preprocess_all.py
- [x] VM中小文件已成功转换为parquet
- [x] 本机运行 preprocess_all.py，27个parquet文件生成
- [x] 修复多批次数据覆盖问题 (fix_multi_batch.py)
- [x] 验证parquet转换结果

## 阶段三：变量构造 ✅

- [x] 股价延迟 Hou-Moskowitz (2005): 53,360 firm-year obs (construct_price_delay.py)
- [x] DataUsage (CNRDS TWFre_count/Term_Only): 10,887 obs (2018-2024)
- [x] 全部控制变量: Size, Lev, ROA, TobinQ, Age, Growth, BoardSize, IndepRatio, Dual, Top1Share, InstHold, Amihud, Analyst, AuditType
- [x] 中间变量: FinAsset, DataAsset, SOE
- [x] 面板合并: panel.parquet = 48,217 obs, 5,295 firms, 2011-2024 (construct_panel.py)

## 阶段四：实证分析 🔄 进行中

### 主检验一v1：面板回归 (CNRDS指标, 2018-2020) — 不显著

- [x] CNRDS仅覆盖2018-2020, 基准回归不显著 (run_regression.py)

### 主检验一v2：面板回归 (年报关键词, 2011-2024) ✅ 显著

- [x] 从年报TXT提取关键词特征: 54,496 firm-year (extract_annual_report_features.py)
  - 5维关键词: 数据存量/开发能力/商业应用/价值变现/数据治理
  - 实质利用比例: 区分实质应用vs概念提及
- [x] 基准回归 (run_regression_v2.py, N=43,847, 14 years)
  - **Firm+Year FE: DU_kw coef=-0.00416, t=-4.28, p<0.001 ★★★**
  - **Industry+Year FE: coef=-0.00178, t=-2.93, p=0.004 ★★★**
  - ln(1+kw): coef=-0.00409, t=-3.47, p<0.001 ★★★
  - 实质利用(ln): coef=-0.00348, t=-4.71, p<0.001 ★★★
  - 经济意义: 1 std DU_kw增加 → PriceDelay降低0.0077 (均值的6.9%)
- [x] DID (2024入表新规, N=17,538)
  - TreatPost coef=+0.007, t=1.75, p=0.083 (边际显著, 方向为正)
  - **平行趋势通过: 2019-2022系数均不显著, 2024 t=6.24 ★★★**
  - 正向系数解读: 入表新规短期增加信息处理成本
- [x] 机制检验
  - **DataUsage → FinAsset: 显著正向 (t=2.65, p=0.008) ★★★**
  - **DataUsage → Analyst: 显著正向 (t=2.63, p=0.009) ★★★**
  - **DataUsage → InstHold: 显著负向 (t=-3.19, p=0.002) ★★★**
- [x] 异质性 (全部显著)
  - **国企 t=-4.00 ★★★, 民企 t=-3.86 ★★★** (国企效应更强)
  - **大企业 t=-3.61 ★★★, 小企业 t=-4.94 ★★★** (小企业效应更强)
  - **高分析师 t=-5.60 ★★★, 低分析师 t=-2.56 ★★** (高分析师覆盖效应更强)
  - **高科技 t=-3.17 ★★★, 传统 t=-3.06 ★★★** (均显著)

### 核心结论

H1a得到支持: 数据要素利用**提升**资本定价效率 (降低股价延迟)
- 信息供给效应占主导: 数据利用吸引分析师关注, 改善信息环境
- FinAsset正向: 数据利用高的企业金融资产配置也更多
- InstHold下降: 可能反映信息模糊效应的另一面
- DID正向: 2024入表新规短期增加信息处理负担 (新规适应期)
- 所有异质性组均显著, 效应稳健

### 主检验二：投资组合回测 ✅

- [x] 每年6月按DataUsage_KW五分组, 7月至次年6月持有 (portfolio_backtest.py)
- [x] 156个月, Q1-Q5月均收益: 1.10%~1.46%, 单调递增
- [x] DAT因子 (Q5-Q1): 月均+0.36%, t=0.96 (不显著)
- [x] Alpha检验:
  - DAT CAPM alpha: +0.29% (t=0.77)
  - DAT FF3 alpha: +0.37% (t=1.17)
  - DAT FF5+MOM alpha: +0.23% (t=0.71)
  - Q1 FF3 alpha: **-0.28% (t=-2.08, p=0.039)** ★★ 低数据组负alpha
- [x] GRS检验: F=1.44, p=0.213 (不拒绝, 符合预期)
- [x] 子期对比:
  - 入表前DAT FF3 alpha: +0.21% (t=0.63)
  - **入表后DAT FF3 alpha: +2.08% (t=2.24) ★★** 2024后高数据组显著溢价

### 稳健性检验 ✅

- [x] 替换因变量: SYNCH (股价同步性, construct_synchronicity.py)
  - **Firm+Year FE: DU_kw coef=+0.011, t=1.79, p=0.074 ★** 方向为正(边际显著)
  - Industry+Year FE: 不显著 (t=1.61, p=0.108)
  - 解读: 数据利用降低delay但略增同步性, 信息效率提升但市场联动增强
- [x] 替换自变量: ln(1+关键词总数)
  - **DU_kw_ln coef=-0.004, t=-3.47, p=0.0005 ★★★** 一致显著
- [x] PSM匹配 (1:1最近邻, caliper=0.05, 17,646对)
  - 平衡性良好 (14/14变量SMD<12%, 13/14<10%)
  - **PSM回归: DU_kw coef=-0.004, t=-3.99, p<0.001 ★★★** 更显著
- [x] 安慰剂检验 (虚假政策年DID)
  - 虚假2020: t=-0.83, p=0.407 (不显著 ✓)
  - 虚假2021: t=-1.09, p=0.276 (不显著 ✓)
  - 虚假2022: t=1.60, p=0.110 (不显著 ✓)
  - **真实2024: t=1.75, p=0.083 ★** (边际显著, 与基准一致)
- [x] 子样本: 剔除2024年
  - **DU_kw coef=-0.005, t=-5.05, p<0.001 ★★★** 剔除政策年后更显著
- [x] 子样本: 仅主板
  - **DU_kw coef=-0.005, t=-3.79, p<0.001 ★★★** 主板样本一致

### 进一步分析 ✅

- [x] SHAP分解 (run_shap_analysis.py, LightGBM R²=0.50)
  - **数据开发能力**: |SHAP|=0.0035 (39.7%), 最重要子维度
  - **数据驱动应用**: |SHAP|=0.0025 (28.3%)
  - **数据存量**: |SHAP|=0.0017 (19.4%)
  - 数据治理: 8.2%, 数据价值变现: 4.4%
  - SHAP图: shap_summary_bar.png, shap_data_dimensions.png
- [x] 反事实分析
  - 低于P75企业提升至P75: PriceDelay降低1.19%
  - 消除数据利用差异: PriceDelay变化-1.05%
  - 2024年数据利用翻倍: PriceDelay降低2.27%

### 待办

- [ ] LLM语义评分 (稳健性替换指标)

## 阶段五：论文撰写

- [ ] 在 Goldmanuscript-2 模板基础上改写为本项目论文
- [ ] 实证结果表格整理（至少15张表）
- [ ] 图表制作：框架图、趋势图、SHAP图、组合回测图
- [ ] 文献引用更新

## 关键文件路径

| 文件 | 路径 |
|------|------|
| 预处理脚本 | /Users/mac/computerscience/15会计研究/preprocess_all.py |
| parquet输出 | /Users/mac/computerscience/15会计研究/data_parquet/ |
| 原始CSMAR数据 | /Users/mac/computerscience/第三方资料/第三方数据资源/上市公司财务信息/ |
| 年报TXT | /Users/mac/computerscience/第三方资料/第三方数据资源/2001~2024年年报/2001-2024年A股年报TXT格式/ |
| 研究框架图 | /Users/mac/computerscience/15会计研究/研究框架图_数据要素定价效率_v3.svg |
| 同步性脚本 | /Users/mac/computerscience/15会计研究/construct_synchronicity.py |
| 稳健性脚本 | /Users/mac/computerscience/15会计研究/run_robustness.py |
| SHAP分析 | /Users/mac/computerscience/15会计研究/run_shap_analysis.py |
| 年报特征提取 | /Users/mac/computerscience/15会计研究/extract_annual_report_features.py |
| CLAUDE.md | /Users/mac/computerscience/15会计研究/.claude/CLAUDE.md |
