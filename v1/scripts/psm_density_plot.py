"""
PSM Kernel Density Plot: 匹配前 vs 匹配后
数据要素利用与资本市场定价效率

Treatment: DU_kw > industry-year median (Ind2 × year)
Controls: Size, Lev, ROA, TobinQ, Age, Growth, IndepRatio, Dual, Top1Share, SOE, CFO
Matching: 1:1 nearest-neighbor, caliper=0.05
Output: results/v16_tables/psm_density.pdf, psm_density.png
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from scipy.stats import gaussian_kde
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import warnings
import os
warnings.filterwarnings('ignore')

# ── Working directory ──
os.chdir('/Users/mac/computerscience/15会计研究')

# ── Font setup ──
font_candidates = ['Songti SC', 'STSong', 'SimSun', 'Heiti SC', 'PingFang SC', 'Arial Unicode MS']
chosen_font = None
available_fonts = set(f.name for f in fm.fontManager.ttflist)
for fc in font_candidates:
    if fc in available_fonts:
        chosen_font = fc
        break

if chosen_font:
    plt.rcParams['font.family'] = chosen_font
    print(f"Using font: {chosen_font}")
else:
    plt.rcParams['font.sans-serif'] = ['Songti SC', 'STSong', 'SimSun', 'Heiti SC', 'PingFang SC']
    plt.rcParams['font.family'] = 'sans-serif'
    print("Using fallback font list")

plt.rcParams['axes.unicode_minus'] = False

# ── Load data ──
df = pd.read_stata('data_stata/reg_sample_iv_v16.dta')
print(f"Raw data: {len(df)} obs")

controls = ['Size', 'Lev', 'ROA', 'TobinQ', 'Age', 'Growth', 'IndepRatio',
            'Dual', 'Top1Share', 'SOE', 'CFO']

# ── Define treatment: DU_kw > industry-year median ──
df['ind_year'] = df['Ind2'].astype(str) + '_' + df['year'].astype(str)
median_map = df.groupby('ind_year')['DU_kw'].median()
df['DU_kw_median'] = df['ind_year'].map(median_map)
df['treat'] = (df['DU_kw'] > df['DU_kw_median']).astype(int)

print(f"Treatment: {df['treat'].sum()} treated, {(df['treat']==0).sum()} control")

# ── Drop missing values ──
cols_needed = controls + ['treat', 'DU_kw']
df_clean = df[cols_needed].dropna().reset_index(drop=True).copy()
print(f"After dropna: {len(df_clean)} obs")

X = df_clean[controls].values
y = df_clean['treat'].values

# ── Estimate propensity scores ──
logit = LogisticRegression(max_iter=5000, C=1.0, solver='lbfgs')
logit.fit(X, y)
df_clean['pscore'] = logit.predict_proba(X)[:, 1]

print(f"Propensity score range: [{df_clean['pscore'].min():.4f}, {df_clean['pscore'].max():.4f}]")
print(f"  Treated mean: {df_clean.loc[df_clean['treat']==1, 'pscore'].mean():.4f}")
print(f"  Control mean: {df_clean.loc[df_clean['treat']==0, 'pscore'].mean():.4f}")

# ── 1:1 nearest-neighbor matching with caliper 0.05 ──
treated = df_clean[df_clean['treat'] == 1].reset_index(drop=True).copy()
control = df_clean[df_clean['treat'] == 0].reset_index(drop=True).copy()

nn = NearestNeighbors(n_neighbors=1, metric='euclidean')
nn.fit(control[['pscore']].values)

distances, indices = nn.kneighbors(treated[['pscore']].values)
caliper = 0.05

# Apply caliper
mask = distances.flatten() <= caliper
matched_treated = treated.iloc[mask].copy()
matched_control_idx = indices.flatten()[mask]
matched_control = control.iloc[matched_control_idx].copy()

print(f"\nMatching results:")
print(f"  Treated matched: {len(matched_treated)} / {len(treated)}")
print(f"  Caliper: {caliper}")
print(f"  Match rate: {len(matched_treated)/len(treated)*100:.1f}%")

# ── KDE computation ──
def compute_kde(data, grid):
    kde = gaussian_kde(data, bw_method='silverman')
    return kde(grid)

# Common grid for plotting
x_grid = np.linspace(0, 1, 500)

# Before matching
kde_treat_before = compute_kde(df_clean.loc[df_clean['treat']==1, 'pscore'].values, x_grid)
kde_control_before = compute_kde(df_clean.loc[df_clean['treat']==0, 'pscore'].values, x_grid)

# After matching
kde_treat_after = compute_kde(matched_treated['pscore'].values, x_grid)
kde_control_after = compute_kde(matched_control['pscore'].values, x_grid)

# ── Plot ──
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2), dpi=300)

# Colors
color_treat = '#1f77b4'   # blue
color_ctrl  = '#d62728'   # red

# Left panel: Before Matching
ax1.plot(x_grid, kde_treat_before, color=color_treat, linewidth=1.5, label='处理组')
ax1.plot(x_grid, kde_control_before, color=color_ctrl, linewidth=1.5, linestyle='--', label='对照组')
ax1.set_title('匹配前', fontsize=13)
ax1.set_xlabel('倾向得分', fontsize=11)
ax1.set_ylabel('核密度', fontsize=11)
ax1.legend(loc='upper right', fontsize=10, frameon=False)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.set_xlim(0, 1)
ax1.set_ylim(bottom=0)
ax1.tick_params(labelsize=9)

# Right panel: After Matching
ax2.plot(x_grid, kde_treat_after, color=color_treat, linewidth=1.5, label='处理组')
ax2.plot(x_grid, kde_control_after, color=color_ctrl, linewidth=1.5, linestyle='--', label='对照组')
ax2.set_title('匹配后', fontsize=13)
ax2.set_xlabel('倾向得分', fontsize=11)
ax2.set_ylabel('核密度', fontsize=11)
ax2.legend(loc='upper right', fontsize=10, frameon=False)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.set_xlim(0, 1)
ax2.set_ylim(bottom=0)
ax2.tick_params(labelsize=9)

plt.tight_layout(w_pad=2.5)

# ── Save ──
os.makedirs('results/v16_tables', exist_ok=True)
output_pdf = 'results/v16_tables/psm_density.pdf'
output_png = 'results/v16_tables/psm_density.png'
fig.savefig(output_pdf, bbox_inches='tight', dpi=300)
fig.savefig(output_png, bbox_inches='tight', dpi=300)
plt.close()

print(f"\nSaved: {output_pdf}")
print(f"Saved: {output_png}")
print("Done.")
