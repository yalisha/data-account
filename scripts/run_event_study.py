"""
Event Study: 2024入表新规的动态效应

设计:
- 处理组: 2017-2021年 DU_kw 均值 >= 中位数 (完全在政策公告前)
- 窗口: 2015-2024
- FE对比: Firm+Year vs Firm+Industry×Year
- 主报告: binary base=2023 + continuous base=2023
- 稳健性: binary base=2022 + continuous base=2022
- 输出: CSV + 事件研究图(含FE对比)
"""

import pandas as pd
import numpy as np
import pyfixest as pf
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import warnings, os
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

OUT_DIR = "/Users/mac/computerscience/15会计研究/data_parquet"
RES_DIR = "/Users/mac/computerscience/15会计研究/results"
os.makedirs(RES_DIR, exist_ok=True)

YEAR_LO, YEAR_HI = 2015, 2024

# ================================================================
# 1. 数据准备
# ================================================================
print("1. 加载数据...")

panel = pd.read_parquet(f"{OUT_DIR}/panel.parquet")
ar_feat = pd.read_parquet(f"{OUT_DIR}/annual_report_features.parquet")

panel = panel.merge(
    ar_feat[['Stkcd', 'year', 'kw_total', 'kw_per10k', 'substantive_ratio',
             'substantive_count']],
    on=['Stkcd', 'year'], how='left'
)

fi = pd.read_parquet(f"{OUT_DIR}/firm_info.parquet",
                     columns=['Symbol', 'EndDate', 'IndustryCodeC', 'LISTINGSTATE'])
fi = fi.rename(columns={'Symbol': 'Stkcd'})
fi['EndDate'] = pd.to_datetime(fi['EndDate'])
fi['year'] = fi['EndDate'].dt.year
fi = fi.sort_values(['Stkcd', 'year', 'EndDate']).drop_duplicates(
    subset=['Stkcd', 'year'], keep='last')
panel = panel.merge(fi[['Stkcd', 'year', 'IndustryCodeC', 'LISTINGSTATE']],
                    on=['Stkcd', 'year'], how='left')

mask_fin = panel['IndustryCodeC'].str.startswith('J', na=False)
mask_st = panel['LISTINGSTATE'].isin(['ST', '*ST'])
mask_new = panel['Age'] <= 0
panel = panel[~mask_fin & ~mask_st & ~mask_new]

panel['Ind2'] = panel['IndustryCodeC'].str[:3]
panel['IndYear'] = panel['Ind2'].astype(str) + '_' + panel['year'].astype(str)
panel['DU_kw'] = panel['kw_per10k']
panel['DU_kw_ln'] = np.log1p(panel['kw_total'])

panel = panel.sort_values(['Stkcd', 'year'])
for v in ['DU_kw', 'DU_kw_ln']:
    panel[v] = panel.groupby('Stkcd')[v].shift(1)

def winsorize(s, lower=0.01, upper=0.99):
    q_lo, q_hi = s.quantile([lower, upper])
    return s.clip(q_lo, q_hi)

cont_vars = ['PriceDelay', 'DU_kw', 'DU_kw_ln',
             'Lev', 'FinAsset', 'DataAsset', 'ROA', 'Growth', 'Size',
             'TobinQ', 'Age', 'BoardSize', 'IndepRatio', 'Top1Share',
             'InstHold', 'Amihud', 'Analyst']
for v in cont_vars:
    if v in panel.columns and panel[v].notna().any():
        panel[v] = winsorize(panel[v])

controls = ['Size', 'Lev', 'ROA', 'TobinQ', 'Age', 'Growth',
            'BoardSize', 'IndepRatio', 'Dual', 'Top1Share',
            'SOE', 'InstHold', 'Amihud', 'Analyst', 'AuditType']
ctrl_str = ' + '.join(controls)

reg_df = panel.dropna(subset=['PriceDelay', 'DU_kw'] + controls).copy()
reg_df['Stkcd_str'] = reg_df['Stkcd'].astype(str)
reg_df['year_int'] = reg_df['year']
print(f"  全样本: {len(reg_df):,} obs, {reg_df.year.min()}-{reg_df.year.max()}")

# ================================================================
# 2. 处理组定义
# ================================================================
print("\n2. 处理组定义 (2017-2021)")

pre_firms = reg_df[(reg_df.year_int >= 2017) & (reg_df.year_int <= 2021)]
firm_kw_mean = pre_firms.groupby('Stkcd')['DU_kw'].mean()
treat_th = firm_kw_mean.median()
treat_set = set(firm_kw_mean[firm_kw_mean >= treat_th].index)
firm_kw_dict = firm_kw_mean.to_dict()
print(f"  阈值: {treat_th:.4f}, 处理组: {len(treat_set)} firms")

# ================================================================
# 3. Event Study 核心函数
# ================================================================
def run_event_study(df, treat_type, base_year, year_lo, year_hi, ctrl_str,
                    fe_type='firm_year'):
    """
    fe_type: 'firm_year' = Stkcd + Year FE
             'firm_indyear' = Stkcd + Industry×Year FE
    """
    es = df[(df.year_int >= year_lo) & (df.year_int <= year_hi)].copy()

    if treat_type == 'binary':
        es['Treat'] = es['Stkcd'].isin(treat_set).astype(float)
    else:
        es['Treat'] = es['Stkcd'].map(firm_kw_dict).fillna(0)
        mu, sigma = es['Treat'].mean(), es['Treat'].std()
        es['Treat'] = (es['Treat'] - mu) / sigma

    event_years = sorted([y for y in es.year_int.unique() if y != base_year])
    for y in event_years:
        es[f'D{y}'] = es['Treat'] * (es['year_int'] == y).astype(float)

    d_vars = ' + '.join([f'D{y}' for y in event_years])
    es['Stkcd_fe'] = es['Stkcd_str']
    es['year_fe'] = es['year_int'].astype(str)

    if fe_type == 'firm_indyear':
        fe_formula = 'Stkcd_fe + IndYear'
    else:
        fe_formula = 'Stkcd_fe + year_fe'

    m = pf.feols(f"PriceDelay ~ {d_vars} + {ctrl_str} | {fe_formula}",
                 data=es, vcov={"CRV1": "IndYear"})

    rows = []
    for y in event_years:
        k = f'D{y}'
        c = m.coef()[k]
        t_val = m.tstat()[k]
        p_val = m.pvalue()[k]
        se = c / t_val if abs(t_val) > 1e-6 else np.nan
        rows.append({
            'Year': y, 'Coef': c, 'SE': abs(se),
            'CI_lo': c - 1.96 * abs(se), 'CI_hi': c + 1.96 * abs(se),
            't': t_val, 'p': p_val, 'N': m._N
        })
    # 基期
    rows.append({
        'Year': base_year, 'Coef': 0, 'SE': 0,
        'CI_lo': 0, 'CI_hi': 0,
        't': np.nan, 'p': np.nan, 'N': m._N
    })
    return pd.DataFrame(rows).sort_values('Year').reset_index(drop=True)

# ================================================================
# 4. 跑8版: 4原版(Firm+Year) + 4新版(Firm+Ind×Year)
# ================================================================
print(f"\n3. Event Study ({YEAR_LO}-{YEAR_HI})")

base_specs = {
    'binary_b2023': ('binary', 2023),
    'binary_b2022': ('binary', 2022),
    'cont_b2023': ('continuous', 2023),
    'cont_b2022': ('continuous', 2022),
}

results = {}

print("\n  --- Firm + Year FE ---")
for key, (ttype, base) in base_specs.items():
    print(f"  {key}...")
    results[key] = run_event_study(reg_df, ttype, base, YEAR_LO, YEAR_HI, ctrl_str,
                                   fe_type='firm_year')

print("\n  --- Firm + Industry×Year FE ---")
for key, (ttype, base) in base_specs.items():
    iy_key = f"{key}_indyear"
    print(f"  {iy_key}...")
    results[iy_key] = run_event_study(reg_df, ttype, base, YEAR_LO, YEAR_HI, ctrl_str,
                                      fe_type='firm_indyear')

# 打印对比
def print_results(res_dict, label):
    for key, df in res_dict.items():
        print(f"\n  --- {label}: {key} ---")
        for _, r in df.iterrows():
            if np.isnan(r.t):
                print(f"    {int(r.Year)}: BASE")
            else:
                sig = '***' if r.p < 0.01 else '**' if r.p < 0.05 else '*' if r.p < 0.1 else ''
                post = ' <POST>' if r.Year >= 2024 else ''
                print(f"    {int(r.Year)}: {r.Coef:+.5f} (se={r.SE:.5f}, t={r.t:+.3f}) {sig}{post}")

orig_results = {k: v for k, v in results.items() if 'indyear' not in k}
iy_results = {k: v for k, v in results.items() if 'indyear' in k}
print_results(orig_results, "Firm+Year")
print_results(iy_results, "Firm+Ind×Year")

# 对比表
print("\n" + "=" * 80)
print("FE对比: Binary base=2023")
print("=" * 80)
print(f"{'Year':<6} {'Firm+Year':>30} {'Firm+Ind×Year':>30}")
print(f"{'':6} {'Coef':>10} {'t':>8} {'sig':>5}    {'Coef':>10} {'t':>8} {'sig':>5}")
print("-" * 80)
df_fy = results['binary_b2023']
df_iy = results['binary_b2023_indyear']
for y in range(YEAR_LO, YEAR_HI + 1):
    r1 = df_fy[df_fy.Year == y].iloc[0]
    r2 = df_iy[df_iy.Year == y].iloc[0]
    if np.isnan(r1.t):
        print(f"  {y}   {'BASE':>10} {'':>8} {'':>5}    {'BASE':>10} {'':>8} {'':>5}")
    else:
        s1 = '***' if r1.p < 0.01 else '**' if r1.p < 0.05 else '*' if r1.p < 0.1 else ''
        s2 = '***' if r2.p < 0.01 else '**' if r2.p < 0.05 else '*' if r2.p < 0.1 else ''
        print(f"  {y}   {r1.Coef:>+10.5f} {r1.t:>+8.3f} {s1:>5}    {r2.Coef:>+10.5f} {r2.t:>+8.3f} {s2:>5}")

# ================================================================
# 5. 绘图
# ================================================================
print("\n4. 绘图...")

def plot_es(ax, df, title, base_year, policy_year=2024, show_announce=True):
    years = df['Year'].values
    coefs = df['Coef'].values
    ci_lo = df['CI_lo'].values
    ci_hi = df['CI_hi'].values

    ax.fill_between(years, ci_lo, ci_hi, alpha=0.15, color='#2166AC')
    ax.plot(years, coefs, 'o-', color='#2166AC', markersize=6, linewidth=1.8, zorder=5)
    ax.axhline(y=0, color='black', linewidth=0.8)
    # 政策实施线
    ax.axvline(x=policy_year - 0.5, color='#B2182B', linewidth=1.8, linestyle='--', alpha=0.8)

    # 显著性标注
    for _, r in df.iterrows():
        if np.isnan(r.p):
            continue
        if r.p < 0.01:
            marker = '***'
        elif r.p < 0.05:
            marker = '**'
        elif r.p < 0.1:
            marker = '*'
        else:
            continue
        offset_y = 8 if r.Coef >= 0 else -14
        ax.annotate(marker, (r.Year, r.Coef), textcoords="offset points",
                    xytext=(0, offset_y), ha='center', fontsize=9,
                    fontweight='bold', color='#B2182B')

    ax.set_title(title, fontsize=12, pad=8)
    ax.set_xlabel('')
    ax.set_ylabel('系数估计值', fontsize=11)
    ax.set_xticks(range(YEAR_LO, YEAR_HI + 1))
    ax.set_xticklabels([str(y) for y in range(YEAR_LO, YEAR_HI + 1)],
                       rotation=45, fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


# --- 主图: Firm+Ind×Year FE版本 (更可信) ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

plot_es(axes[0], results['binary_b2023_indyear'],
        '(a) Binary处理, Firm+Ind×Year FE (基期=2023)', 2023)
plot_es(axes[1], results['cont_b2023_indyear'],
        '(b) Continuous处理, Firm+Ind×Year FE (基期=2023)', 2023)

fig.suptitle(f'事件研究法: 2024年入表新规对股价延迟的动态效应 ({YEAR_LO}-{YEAR_HI})',
             fontsize=14, fontweight='bold', y=1.02)
fig.tight_layout()
fig.savefig(f"{RES_DIR}/event_study_main.png", dpi=200, bbox_inches='tight')
print(f"  主图(Ind×Year FE): {RES_DIR}/event_study_main.png")

# --- FE对比图: 2x2, binary_b2023 Firm+Year vs Firm+Ind×Year ---
fig_cmp, axes_cmp = plt.subplots(2, 2, figsize=(14, 10))

plot_es(axes_cmp[0, 0], results['binary_b2023'],
        '(a) Binary, Firm+Year FE', 2023)
plot_es(axes_cmp[0, 1], results['binary_b2023_indyear'],
        '(b) Binary, Firm+Ind×Year FE', 2023)
plot_es(axes_cmp[1, 0], results['cont_b2023'],
        '(c) Continuous, Firm+Year FE', 2023)
plot_es(axes_cmp[1, 1], results['cont_b2023_indyear'],
        '(d) Continuous, Firm+Ind×Year FE', 2023)

fig_cmp.suptitle(f'FE对比: Firm+Year vs Firm+Ind×Year ({YEAR_LO}-{YEAR_HI}, 基期=2023)',
                 fontsize=14, fontweight='bold', y=1.01)
fig_cmp.tight_layout()
fig_cmp.savefig(f"{RES_DIR}/event_study_fe_compare.png", dpi=200, bbox_inches='tight')
print(f"  FE对比图: {RES_DIR}/event_study_fe_compare.png")

# --- 4-panel 全版 (Ind×Year FE) ---
fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))

plot_es(axes2[0, 0], results['binary_b2023_indyear'],
        '(a) Binary, 基期=2023', 2023)
plot_es(axes2[0, 1], results['binary_b2022_indyear'],
        '(b) Binary, 基期=2022', 2022)
plot_es(axes2[1, 0], results['cont_b2023_indyear'],
        '(c) Continuous, 基期=2023', 2023)
plot_es(axes2[1, 1], results['cont_b2022_indyear'],
        '(d) Continuous, 基期=2022', 2022)

fig2.suptitle(f'事件研究法 (Firm+Ind×Year FE): 不同设定下的动态效应 ({YEAR_LO}-{YEAR_HI})',
              fontsize=14, fontweight='bold', y=1.01)
fig2.tight_layout()
fig2.savefig(f"{RES_DIR}/event_study_4panel.png", dpi=200, bbox_inches='tight')
print(f"  四图(Ind×Year FE): {RES_DIR}/event_study_4panel.png")

plt.close('all')

# ================================================================
# 6. 保存数据
# ================================================================
print("\n5. 保存CSV...")

all_dfs = []
for key, df in results.items():
    df_out = df.copy()
    df_out['Specification'] = key
    all_dfs.append(df_out)

df_all = pd.concat(all_dfs, ignore_index=True)
df_all.to_csv(f"{RES_DIR}/event_study_coefficients.csv", index=False)
print(f"  {RES_DIR}/event_study_coefficients.csv ({len(df_all)} rows)")

# ================================================================
# 7. 汇总: 推荐Firm+Ind×Year版本
# ================================================================
print("\n" + "=" * 60)
print(f"推荐主报告: Binary, base=2023, Firm+Ind×Year FE")
print("=" * 60)
main = results['binary_b2023_indyear']
for _, r in main.iterrows():
    y = int(r.Year)
    if np.isnan(r.t):
        print(f"  {y}: BASE")
    else:
        sig = '***' if r.p < 0.01 else '**' if r.p < 0.05 else '*' if r.p < 0.1 else ''
        print(f"  {y}: coef={r.Coef:+.5f}  se={r.SE:.5f}  t={r.t:+.3f}  p={r.p:.4f}  {sig}")

print(f"\n推荐主报告: Continuous, base=2023, Firm+Ind×Year FE")
print("=" * 60)
main2 = results['cont_b2023_indyear']
for _, r in main2.iterrows():
    y = int(r.Year)
    if np.isnan(r.t):
        print(f"  {y}: BASE")
    else:
        sig = '***' if r.p < 0.01 else '**' if r.p < 0.05 else '*' if r.p < 0.1 else ''
        print(f"  {y}: coef={r.Coef:+.5f}  se={r.SE:.5f}  t={r.t:+.3f}  p={r.p:.4f}  {sig}")

print("\n完成！")
