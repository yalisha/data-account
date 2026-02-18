"""
SHAP分析: 数据要素各子维度对PriceDelay的边际贡献

子维度:
  - kw_data_stock: 数据存量
  - kw_data_dev: 数据开发能力
  - kw_data_app: 数据驱动商业应用
  - kw_data_value: 数据价值变现
  - kw_data_gov: 数据治理

方法: LightGBM预测PriceDelay, SHAP分解各特征贡献
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import shap
import warnings, os
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

OUT_DIR = "/Users/mac/computerscience/15会计研究/data_parquet"
RES_DIR = "/Users/mac/computerscience/15会计研究/results"
os.makedirs(RES_DIR, exist_ok=True)

# ================================================================
# 1. 准备数据
# ================================================================
print("=" * 60)
print("SHAP分析: 数据要素子维度贡献")
print("=" * 60)

panel = pd.read_parquet(f"{OUT_DIR}/panel.parquet")
ar_feat = pd.read_parquet(f"{OUT_DIR}/annual_report_features.parquet")

panel = panel.merge(
    ar_feat[['Stkcd', 'year', 'kw_data_stock', 'kw_data_dev', 'kw_data_app',
             'kw_data_value', 'kw_data_gov', 'substantive_count', 'substantive_ratio',
             'kw_per10k', 'kw_total', 'total_chars']],
    on=['Stkcd', 'year'], how='left'
)

# 合并行业
fi = pd.read_parquet(f"{OUT_DIR}/firm_info.parquet",
                     columns=['Symbol', 'EndDate', 'IndustryCodeC', 'LISTINGSTATE'])
fi = fi.rename(columns={'Symbol': 'Stkcd'})
fi['EndDate'] = pd.to_datetime(fi['EndDate'])
fi['year'] = fi['EndDate'].dt.year
fi = fi.sort_values(['Stkcd', 'year', 'EndDate']).drop_duplicates(
    subset=['Stkcd', 'year'], keep='last')
panel = panel.merge(fi[['Stkcd', 'year', 'IndustryCodeC', 'LISTINGSTATE']],
                    on=['Stkcd', 'year'], how='left')

# 筛选
mask_fin = panel['IndustryCodeC'].str.startswith('J', na=False)
mask_st = panel['LISTINGSTATE'].isin(['ST', '*ST'])
mask_new = panel['Age'] <= 0
panel = panel[~mask_fin & ~mask_st & ~mask_new]

# 归一化子维度 (每万字)
for dim in ['kw_data_stock', 'kw_data_dev', 'kw_data_app', 'kw_data_value', 'kw_data_gov']:
    panel[dim + '_p10k'] = panel[dim] / panel['total_chars'].clip(lower=1) * 10000

controls = ['Size', 'Lev', 'ROA', 'TobinQ', 'Age', 'Growth',
            'BoardSize', 'IndepRatio', 'Dual', 'Top1Share',
            'SOE', 'InstHold', 'Amihud', 'Analyst', 'AuditType']

data_dims = ['kw_data_stock_p10k', 'kw_data_dev_p10k', 'kw_data_app_p10k',
             'kw_data_value_p10k', 'kw_data_gov_p10k']

# SHAP解释聚焦在企业特征层面, 不把year作为可解释特征
features = data_dims + controls
target = 'PriceDelay'

df = panel.dropna(subset=[target] + features).copy()

# 留出集评估: 默认按年份切分, 避免样本内高估
train_df = df[df['year'] <= 2022].copy()
test_df = df[df['year'] >= 2023].copy()
if len(train_df) < 2000 or len(test_df) < 1000:
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

print(f"  总样本: {len(df):,} obs")
print(f"  训练集: {len(train_df):,} obs")
print(f"  测试集: {len(test_df):,} obs")

X_train = train_df[features].values
y_train = train_df[target].values
X_test = test_df[features].values
y_test = test_df[target].values
feature_names = features

# 漂亮的中文标签
label_map = {
    'kw_data_stock_p10k': '数据存量',
    'kw_data_dev_p10k': '数据开发能力',
    'kw_data_app_p10k': '数据驱动应用',
    'kw_data_value_p10k': '数据价值变现',
    'kw_data_gov_p10k': '数据治理',
    'Size': '企业规模',
    'Lev': '资产负债率',
    'ROA': '资产收益率',
    'TobinQ': 'Tobin Q',
    'Age': '上市年限',
    'Growth': '营收增长率',
    'BoardSize': '董事会规模',
    'IndepRatio': '独董比例',
    'Dual': '两职合一',
    'Top1Share': '第一大股东',
    'SOE': '国企',
    'InstHold': '机构持股',
    'Amihud': 'Amihud非流动性',
    'Analyst': '分析师覆盖',
    'AuditType': '标准审计意见',
}

# ================================================================
# 2. 训练LightGBM
# ================================================================
print("\n--- 训练LightGBM ---")

model = lgb.LGBMRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    verbose=-1,
    n_jobs=-1
)
model.fit(X_train, y_train)

# 训练集与留出集表现
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)
r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
print(f"  Train R² = {r2_train:.4f}")
print(f"  Test  R² = {r2_test:.4f}")
print(f"  Test  RMSE = {rmse_test:.6f}")

# ================================================================
# 3. SHAP分析
# ================================================================
print("\n--- 计算SHAP值 ---")

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

print(f"  SHAP values shape: {shap_values.shape}")

# 特征重要性排序 (mean |SHAP|)
mean_abs_shap = np.abs(shap_values).mean(axis=0)
importance = pd.DataFrame({
    'feature': feature_names,
    'mean_abs_shap': mean_abs_shap,
    'mean_shap': shap_values.mean(axis=0),
}).sort_values('mean_abs_shap', ascending=False)

importance['label'] = importance['feature'].map(label_map)
print("\n--- 特征重要性 (mean |SHAP|) ---")
print(importance[['label', 'mean_abs_shap', 'mean_shap']].round(6).to_string(index=False))

# 数据要素子维度排名
print("\n--- 数据要素子维度排名 ---")
data_imp = importance[importance['feature'].isin(data_dims)]
total_data_shap = data_imp['mean_abs_shap'].sum()
for _, row in data_imp.iterrows():
    pct = row['mean_abs_shap'] / total_data_shap * 100
    direction = '降低delay' if row['mean_shap'] < 0 else '增加delay'
    print(f"  {row['label']}: |SHAP|={row['mean_abs_shap']:.6f} ({pct:.1f}%), {direction}")

# ================================================================
# 4. 可视化
# ================================================================
print("\n--- 生成图表 ---")

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 4a. 全特征SHAP summary (bar)
fig, ax = plt.subplots(figsize=(10, 8))
labels = [label_map.get(f, f) for f in feature_names]
shap.summary_plot(shap_values, X_test, feature_names=labels, plot_type='bar',
                  show=False, max_display=20)
plt.tight_layout()
plt.savefig(f"{RES_DIR}/shap_summary_bar.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  保存: shap_summary_bar.png")

# 4b. SHAP summary (dot)
fig, ax = plt.subplots(figsize=(10, 8))
shap.summary_plot(shap_values, X_test, feature_names=labels,
                  show=False, max_display=20)
plt.tight_layout()
plt.savefig(f"{RES_DIR}/shap_summary_dot.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  保存: shap_summary_dot.png")

# 4c. 数据要素子维度单独图
data_idx = [feature_names.index(d) for d in data_dims]
data_shap = shap_values[:, data_idx]
data_X = X_test[:, data_idx]
data_labels = [label_map[d] for d in data_dims]

fig, ax = plt.subplots(figsize=(8, 5))
shap.summary_plot(data_shap, data_X, feature_names=data_labels,
                  show=False)
plt.title('数据要素子维度SHAP值分布', fontsize=14)
plt.tight_layout()
plt.savefig(f"{RES_DIR}/shap_data_dimensions.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  保存: shap_data_dimensions.png")

# 4d. 各子维度SHAP dependence plots
fig, axes = plt.subplots(1, 5, figsize=(20, 4))
for i, (dim, label) in enumerate(zip(data_dims, data_labels)):
    idx = feature_names.index(dim)
    ax = axes[i]
    ax.scatter(X_test[:, idx], shap_values[:, idx], alpha=0.1, s=2, c='steelblue')
    ax.set_xlabel(label, fontsize=10)
    ax.set_ylabel('SHAP value' if i == 0 else '', fontsize=10)
    ax.axhline(y=0, color='k', linewidth=0.5, linestyle='--')
    ax.set_title(label, fontsize=11)
plt.suptitle('数据要素子维度 SHAP Dependence Plots', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(f"{RES_DIR}/shap_dependence_dims.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  保存: shap_dependence_dims.png")

# ================================================================
# 5. 反事实分析
# ================================================================
print("\n" + "=" * 60)
print("反事实分析: 量化经济幅度")
print("=" * 60)

# 情景1: 所有企业数据利用提升至P75
p75_vals = {}
for dim in data_dims:
    p75_vals[dim] = train_df[dim].quantile(0.75)

X_cf = X_test.copy()
for dim in data_dims:
    idx = feature_names.index(dim)
    mask = X_cf[:, idx] < p75_vals[dim]
    X_cf[mask, idx] = p75_vals[dim]

y_cf1 = model.predict(X_cf)
delay_reduction = y_test.mean() - y_cf1.mean()
pct_reduction = delay_reduction / y_test.mean() * 100
print(f"\n情景1: 低于P75的企业数据利用提升至P75")
print(f"  基线PriceDelay: {y_test.mean():.6f}")
print(f"  反事实PriceDelay: {y_cf1.mean():.6f}")
print(f"  降幅: {delay_reduction:.6f} ({pct_reduction:.2f}%)")

# 情景2: 完全消除数据要素差异 (所有企业=均值)
X_cf2 = X_test.copy()
for dim in data_dims:
    idx = feature_names.index(dim)
    X_cf2[:, idx] = train_df[dim].mean()

y_cf2 = model.predict(X_cf2)
delay_change2 = y_test.mean() - y_cf2.mean()
pct_change2 = delay_change2 / y_test.mean() * 100
print(f"\n情景2: 消除数据利用差异 (所有企业=均值)")
print(f"  反事实PriceDelay: {y_cf2.mean():.6f}")
print(f"  变化: {delay_change2:.6f} ({pct_change2:.2f}%)")

# 情景3: 2024后所有企业数据利用翻倍
is_2024 = test_df['year'].values == 2024
X_cf3 = X_test.copy()
for dim in data_dims:
    idx = feature_names.index(dim)
    X_cf3[is_2024, idx] = X_cf3[is_2024, idx] * 2

y_cf3 = model.predict(X_cf3)
if is_2024.sum() > 0:
    delay_change3 = y_test[is_2024].mean() - y_cf3[is_2024].mean()
    pct_change3 = delay_change3 / y_test[is_2024].mean() * 100
    print(f"\n情景3: 2024年企业数据利用翻倍")
    print(f"  2024基线: {y_test[is_2024].mean():.6f}")
    print(f"  反事实: {y_cf3[is_2024].mean():.6f}")
    print(f"  降幅: {delay_change3:.6f} ({pct_change3:.2f}%)")

# 保存
importance.to_csv(f"{RES_DIR}/shap_importance.csv", index=False)
print(f"\n已保存所有结果到 {RES_DIR}/")
print("完成！")
