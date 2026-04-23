# 当前结果总结（2026-02-18）

本文件基于最新重跑结果（已完成时点对齐、主板筛选修正、SHAP v2重构）整理，可直接用于论文写作口径统一。

## 一、总判断

- 主结论仍成立：数据要素利用提升与更高定价效率相关（`DU_kw` 对 `PriceDelay` 显著负向）。
- 因果识别仍偏弱：DID 在当前设定下不显著，且存在显著安慰剂年份。
- 投资组合证据为补充而非核心：DAT 因子不显著，主要亮点是低数据组 `Q1` 负 alpha。
- SHAP 机制证据应降级为探索性：外样本拟合弱（R²<0），不宜做强机制宣称。

## 二、核心回归结果（`results/regression_summary_v2.csv`）

- M2（Firm+Year，主模型）：
  - `DU_kw = -0.00364`, `t = -2.57`, `p = 0.010`, `N = 38,614`
- 其余设定：
  - `DU_kw_ln`、`DU_sub_ln` 均显著负向
  - `Ind+Year` 下仍显著负向

结论：主效应稳健但强度较旧版有所下降（修正前视后属于预期）。

## 三、DID与稳健性（`results/robustness_summary.csv`）

- 真实政策年 DID（2024）：
  - `TreatPost = +0.00750`, `t = 1.38`, `p = 0.170`（不显著）
- 安慰剂：
  - 2020、2021不显著
  - 2022显著：`p = 0.046`（识别存在风险信号）
- 其他稳健性：
  - PSM后主效应仍显著：`DU_kw = -0.00478`, `p < 0.001`
  - 剔除2024后效应增强：`DU_kw = -0.00503`, `p < 0.001`
  - 仅主板后仍显著：`DU_kw = -0.00389`, `p = 0.016`

## 四、投资组合结果（`results/portfolio_alpha.csv`）

- 全样本 DAT（Q5-Q1）：
  - FF3 alpha `= +0.374%`, `t = 1.17`, `p = 0.244`（不显著）
- 亮点：
  - Q1 在 FF3 下显著负 alpha：`-0.285%`, `t = -2.08`, `p = 0.039`

结论：组合结果可作为“市场对低数据利用企业定价偏差”的补充证据，但不支持强因子叙事。

## 五、SHAP v2 结果（新增）

对应文件：
- `results/shap_oos_metrics.csv`
- `results/shap_importance.csv`
- `results/shap_dim_rank_stability.csv`
- `results/shap_counterfactual.csv`

### 1. 外样本表现（walk-forward）

- 加权平均外样本 `R² = -0.0321`
- 最终 holdout（2023-2024）`Test R² = -0.0232`

含义：模型对残差部分几乎无可泛化解释力，SHAP解释应谨慎。

### 2. 维度重要性（外样本SHAP）

排序：
1. 数据开发能力（27.2%）
2. 数据驱动应用（26.1%）
3. 数据存量（25.4%）
4. 数据治理（13.6%）
5. 数据价值变现（7.7%）

### 3. 稳定性（bootstrap=200）

- 数据开发能力：`top1_share = 0.685`，`top2_share = 0.93`
- 数据驱动应用：`top2_share = 0.61`
- 数据存量：`top2_share = 0.46`
- 数据治理/价值变现长期稳定靠后

### 4. 反事实（现实边界）

- `P25 -> P75`：`+0.10%`（对 PriceDelay 均值影响极小）
- `+1σ capped`：`+3.72%`（方向需谨慎解读，仅为模型内推演）

## 六、建议的论文写作口径（当前版本）

- 可以主打：
  - “数据利用与定价效率提升显著相关（面板主效应稳健）”
- 应弱化：
  - “2024新规带来明确因果冲击”
  - “SHAP已清晰识别机制”
- 推荐定位：
  - DID：探索性/辅助识别
  - SHAP：补充性异质结构描述（非核心识别证据）

## 七、当前可直接引用的结果文件

- 主回归：`results/regression_summary_v2.csv`
- 稳健性：`results/robustness_summary.csv`
- 组合：`results/portfolio_alpha.csv`
- SHAP v2：`results/shap_oos_metrics.csv`, `results/shap_importance.csv`, `results/shap_dim_rank_stability.csv`, `results/shap_counterfactual.csv`

