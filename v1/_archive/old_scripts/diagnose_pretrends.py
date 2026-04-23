"""
Pre-trends Diagnostic: Test multiple treatment definitions for parallel pre-trends.

For the DID on the 2024 data-asset accounting reform, we need to check whether
the treatment and control groups had parallel trends in PriceDelay before the
policy shock. This script tests 5 different treatment definitions.
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

DATA = "/Users/mac/computerscience/15会计研究/data_parquet"

# ============================================================
# 1. DATA PREPARATION (mirrors run_event_study.py)
# ============================================================
print("=" * 80)
print("PRE-TRENDS DIAGNOSTIC: Multiple Treatment Definitions")
print("=" * 80)

print("\n[1] Loading data...")
panel = pd.read_parquet(f"{DATA}/panel.parquet")
ar_feat = pd.read_parquet(f"{DATA}/annual_report_features.parquet")
fi = pd.read_parquet(f"{DATA}/firm_info.parquet",
                     columns=['Symbol', 'EndDate', 'IndustryCodeC', 'LISTINGSTATE'])

# Merge kw_per10k
panel = panel.merge(
    ar_feat[['Stkcd', 'year', 'kw_per10k', 'kw_total', 'substantive_ratio',
             'substantive_count']],
    on=['Stkcd', 'year'], how='left'
)

# Merge firm info
fi = fi.rename(columns={'Symbol': 'Stkcd'})
fi['EndDate'] = pd.to_datetime(fi['EndDate'])
fi['year'] = fi['EndDate'].dt.year
fi = fi.sort_values(['Stkcd', 'year', 'EndDate']).drop_duplicates(
    subset=['Stkcd', 'year'], keep='last')
panel = panel.merge(fi[['Stkcd', 'year', 'IndustryCodeC', 'LISTINGSTATE']],
                    on=['Stkcd', 'year'], how='left')

# Exclusions: financial, ST, new firms
mask_fin = panel['IndustryCodeC'].str.startswith('J', na=False)
mask_st = panel['LISTINGSTATE'].isin(['ST', '*ST'])
mask_new = panel['Age'] <= 0
panel = panel[~mask_fin & ~mask_st & ~mask_new].copy()
print(f"  After exclusions: {len(panel):,} obs")

# Construct DU_kw with 1-year lag
panel['DU_kw'] = panel['kw_per10k']
panel = panel.sort_values(['Stkcd', 'year'])
panel['DU_kw'] = panel.groupby('Stkcd')['DU_kw'].shift(1)

# Industry code
panel['Ind2'] = panel['IndustryCodeC'].str[:3]

# Winsorize PriceDelay and DU_kw
def winsorize(s, lo=0.01, hi=0.99):
    q = s.quantile([lo, hi])
    return s.clip(q.iloc[0], q.iloc[1])

for v in ['PriceDelay', 'DU_kw']:
    if v in panel.columns:
        panel[v] = winsorize(panel[v])

# Working sample: need PriceDelay and DU_kw
df = panel.dropna(subset=['PriceDelay', 'DU_kw']).copy()
print(f"  Working sample: {len(df):,} obs, years {df.year.min()}-{df.year.max()}")
print(f"  Unique firms: {df['Stkcd'].nunique()}")

# ============================================================
# 2. TREATMENT DEFINITIONS
# ============================================================
print("\n[2] Constructing treatment definitions...")

# Pre-treatment window for defining treatment: 2017-2021 (same as event study)
pre_df = df[(df.year >= 2017) & (df.year <= 2021)]
firm_kw_mean = pre_df.groupby('Stkcd')['DU_kw'].mean()

# (a) Current: median split
med = firm_kw_mean.median()
treat_a = set(firm_kw_mean[firm_kw_mean >= med].index)
ctrl_a = set(firm_kw_mean[firm_kw_mean < med].index)

# (b) Top quartile vs bottom quartile
q25, q75 = firm_kw_mean.quantile(0.25), firm_kw_mean.quantile(0.75)
treat_b = set(firm_kw_mean[firm_kw_mean >= q75].index)
ctrl_b = set(firm_kw_mean[firm_kw_mean <= q25].index)

# (c) Top tercile vs bottom tercile
t33, t67 = firm_kw_mean.quantile(1/3), firm_kw_mean.quantile(2/3)
treat_c = set(firm_kw_mean[firm_kw_mean >= t67].index)
ctrl_c = set(firm_kw_mean[firm_kw_mean <= t33].index)

# (d) Any mention vs none
treat_d = set(firm_kw_mean[firm_kw_mean > 0].index)
ctrl_d = set(firm_kw_mean[firm_kw_mean == 0].index)

# (e) Industry-level: top-3 vs bottom-3 industries by mean DU_kw
# First get industry for each firm (most common industry in pre-period)
firm_ind = pre_df.groupby('Stkcd')['Ind2'].agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0])
ind_kw = pre_df.groupby('Ind2')['DU_kw'].mean().sort_values(ascending=False)
# Need industries with enough firms
ind_counts = pre_df.groupby('Ind2')['Stkcd'].nunique()
big_inds = ind_counts[ind_counts >= 30].index
ind_kw_big = ind_kw[ind_kw.index.isin(big_inds)]

top3_inds = set(ind_kw_big.head(3).index)
bot3_inds = set(ind_kw_big.tail(3).index)
firms_with_ind = firm_ind.to_dict()
treat_e = set(s for s, i in firms_with_ind.items() if i in top3_inds)
ctrl_e = set(s for s, i in firms_with_ind.items() if i in bot3_inds)

treatments = {
    'A: Median split': (treat_a, ctrl_a),
    'B: Q75 vs Q25': (treat_b, ctrl_b),
    'C: Tercile T3 vs T1': (treat_c, ctrl_c),
    'D: Any mention vs None': (treat_d, ctrl_d),
    'E: Top-3 vs Bot-3 industry': (treat_e, ctrl_e),
}

print(f"\n  Firm-level DU_kw (2017-2021 mean) distribution:")
print(f"    Mean: {firm_kw_mean.mean():.4f}")
print(f"    Median: {med:.4f}")
print(f"    Q25: {q25:.4f}, Q75: {q75:.4f}")
print(f"    Firms with DU_kw=0: {(firm_kw_mean == 0).sum()}")
print(f"    Firms with DU_kw>0: {(firm_kw_mean > 0).sum()}")

print(f"\n  Industry-level DU_kw (top/bottom with >=30 firms):")
print(f"    Top 3: {list(top3_inds)} (mean DU_kw: {[f'{ind_kw_big[i]:.3f}' for i in top3_inds]})")
print(f"    Bot 3: {list(bot3_inds)} (mean DU_kw: {[f'{ind_kw_big[i]:.3f}' for i in bot3_inds]})")

for name, (t, c) in treatments.items():
    n_t = len(t)
    n_c = len(c)
    print(f"  {name}: Treat={n_t}, Control={n_c}, Total={n_t+n_c}")

# ============================================================
# 3. PRE-TREND TESTS FOR EACH DEFINITION
# ============================================================
print("\n[3] Pre-trend analysis for each treatment definition...")
print("    (Policy year: 2024; pre-treatment: 2015-2023)")

all_years = sorted(df.year.unique())
all_results = {}

for name, (treat_set, ctrl_set) in treatments.items():
    print(f"\n{'='*70}")
    print(f"  Treatment: {name}")
    print(f"  Treat firms: {len(treat_set)}, Control firms: {len(ctrl_set)}")
    print(f"{'='*70}")
    
    sub = df[df['Stkcd'].isin(treat_set | ctrl_set)].copy()
    sub['Treat'] = sub['Stkcd'].isin(treat_set).astype(int)
    
    year_results = []
    
    print(f"\n  {'Year':>6} {'N_T':>6} {'N_C':>6} {'Mean_T':>10} {'Mean_C':>10} {'Diff':>10} {'t-stat':>8} {'p-val':>8} {'Sig':>5}")
    print(f"  {'-'*75}")
    
    for y in all_years:
        ydata = sub[sub.year == y]
        treat_vals = ydata[ydata.Treat == 1]['PriceDelay']
        ctrl_vals = ydata[ydata.Treat == 0]['PriceDelay']
        
        if len(treat_vals) < 5 or len(ctrl_vals) < 5:
            continue
        
        mean_t = treat_vals.mean()
        mean_c = ctrl_vals.mean()
        diff = mean_t - mean_c
        
        # Welch's t-test
        t_stat, p_val = stats.ttest_ind(treat_vals, ctrl_vals, equal_var=False)
        
        sig = '***' if p_val < 0.01 else '**' if p_val < 0.05 else '*' if p_val < 0.1 else ''
        marker = ' <-- POST' if y >= 2024 else ''
        
        print(f"  {y:>6} {len(treat_vals):>6} {len(ctrl_vals):>6} {mean_t:>10.5f} {mean_c:>10.5f} {diff:>+10.5f} {t_stat:>+8.3f} {p_val:>8.4f} {sig:>5}{marker}")
        
        year_results.append({
            'Year': y, 'N_T': len(treat_vals), 'N_C': len(ctrl_vals),
            'Mean_T': mean_t, 'Mean_C': mean_c, 'Diff': diff,
            't_stat': t_stat, 'p_val': p_val
        })
    
    yr_df = pd.DataFrame(year_results)
    all_results[name] = yr_df
    
    # Joint F-test for pre-trends (2015-2023): are all differences = 0?
    pre_years = yr_df[yr_df.Year < 2024]
    if len(pre_years) > 0:
        # Simple approach: test if pre-treatment differences are jointly zero
        # Using the individual t-stats, compute a chi-squared-like test
        # More properly: run a regression Diff = sum(year_dummies) and F-test
        
        # Actually, let's do a proper pooled regression approach
        pre_sub = sub[sub.year < 2024].copy()
        pre_sub['TreatYear'] = pre_sub['Treat'] * pre_sub['year']
        
        # Simple: regress PriceDelay on Treat * year_dummies in pre-period
        # and test if all Treat*year coefficients are jointly zero
        from scipy.stats import f as f_dist
        
        # Collect pre-treatment differences and their standard errors
        pre_diffs = pre_years['Diff'].values
        pre_n_t = pre_years['N_T'].values
        pre_n_c = pre_years['N_C'].values
        
        # Variance of each difference
        pre_vars = []
        for _, row in pre_years.iterrows():
            ydata = sub[sub.year == row['Year']]
            var_t = ydata[ydata.Treat == 1]['PriceDelay'].var()
            var_c = ydata[ydata.Treat == 0]['PriceDelay'].var()
            pre_vars.append(var_t / row['N_T'] + var_c / row['N_C'])
        pre_vars = np.array(pre_vars)
        
        # Wald test: chi2 = sum(diff_i^2 / var_i)
        chi2_stat = np.sum(pre_diffs**2 / pre_vars)
        dof = len(pre_diffs)
        f_stat = chi2_stat / dof
        f_pval = 1 - f_dist.cdf(f_stat, dof, 1000)  # approximate
        
        # Also: simple test -- count how many pre-years are significant
        n_sig_01 = (pre_years['p_val'] < 0.01).sum()
        n_sig_05 = (pre_years['p_val'] < 0.05).sum()
        n_sig_10 = (pre_years['p_val'] < 0.1).sum()
        
        # Max absolute t-stat in pre-period
        max_abs_t = pre_years['t_stat'].abs().max()
        max_t_year = pre_years.loc[pre_years['t_stat'].abs().idxmax(), 'Year']
        
        # Average absolute difference in pre-period
        avg_abs_diff = pre_years['Diff'].abs().mean()
        
        # Trend test: is there a linear trend in the differences?
        if len(pre_years) >= 3:
            slope, intercept, r_val, trend_p, std_err = stats.linregress(
                pre_years['Year'].values, pre_years['Diff'].values
            )
            trend_info = f"slope={slope:+.6f}, p={trend_p:.4f}"
        else:
            trend_info = "N/A (too few years)"
        
        print(f"\n  --- Pre-trend Summary (2015-2023) ---")
        print(f"  Joint F-test (Wald): F={f_stat:.3f}, p={f_pval:.4f}")
        print(f"  Significant pre-years: {n_sig_01}@1%, {n_sig_05}@5%, {n_sig_10}@10% (out of {len(pre_years)})")
        print(f"  Max |t-stat| in pre-period: {max_abs_t:.3f} (year {int(max_t_year)})")
        print(f"  Average |Diff| in pre-period: {avg_abs_diff:.5f}")
        print(f"  Linear trend in Diff: {trend_info}")

# ============================================================
# 4. COMPARISON TABLE
# ============================================================
print("\n\n" + "=" * 80)
print("COMPARISON TABLE: Which treatment definition has the cleanest pre-trends?")
print("=" * 80)

summary_rows = []
for name, yr_df in all_results.items():
    pre = yr_df[yr_df.Year < 2024]
    post = yr_df[yr_df.Year >= 2024]
    
    n_sig_05 = (pre['p_val'] < 0.05).sum()
    n_sig_10 = (pre['p_val'] < 0.1).sum()
    max_abs_t = pre['t_stat'].abs().max()
    avg_abs_diff = pre['Diff'].abs().mean()
    
    # Trend
    if len(pre) >= 3:
        slope, _, _, trend_p, _ = stats.linregress(pre['Year'].values, pre['Diff'].values)
    else:
        slope, trend_p = np.nan, np.nan
    
    # Post-treatment effect (if 2024 exists)
    post_diff = post['Diff'].mean() if len(post) > 0 else np.nan
    post_t = post['t_stat'].mean() if len(post) > 0 else np.nan
    
    # Treatment and control sizes
    treat_set, ctrl_set = treatments[name]
    
    summary_rows.append({
        'Treatment': name,
        'N_treat': len(treat_set),
        'N_ctrl': len(ctrl_set),
        'Pre_sig@5%': n_sig_05,
        'Pre_sig@10%': n_sig_10,
        'Pre_max|t|': max_abs_t,
        'Pre_avg|Diff|': avg_abs_diff,
        'Trend_slope': slope,
        'Trend_p': trend_p,
        'Post_Diff': post_diff,
        'Post_t': post_t,
    })

summary = pd.DataFrame(summary_rows)
print(summary.to_string(index=False, float_format='%.4f'))

# Score each definition
print("\n\nSCORING (lower = better pre-trends):")
print("-" * 60)
for _, row in summary.iterrows():
    score = 0
    # Penalize significant pre-years
    score += row['Pre_sig@5%'] * 3
    score += row['Pre_sig@10%'] * 1
    # Penalize large max t-stat
    if row['Pre_max|t|'] > 2:
        score += 3
    elif row['Pre_max|t|'] > 1.65:
        score += 1
    # Penalize significant trend
    if not np.isnan(row['Trend_p']) and row['Trend_p'] < 0.05:
        score += 3
    elif not np.isnan(row['Trend_p']) and row['Trend_p'] < 0.1:
        score += 1
    # Bonus for larger sample
    if row['N_treat'] + row['N_ctrl'] > 3000:
        score -= 1
    
    print(f"  {row['Treatment']:<35} Score: {score:>3}  (sig@5%={int(row['Pre_sig@5%'])}, max|t|={row['Pre_max|t|']:.2f}, trend_p={row['Trend_p']:.3f})")

# ============================================================
# 5. RAW MEANS TABLE (for current definition A)
# ============================================================
print("\n\n" + "=" * 80)
print("RAW MEAN PriceDelay: Treatment A (Median Split)")
print("=" * 80)

sub_a = df[df['Stkcd'].isin(treat_a | ctrl_a)].copy()
sub_a['Treat'] = sub_a['Stkcd'].isin(treat_a).astype(int)

print(f"\n  {'Year':>6} {'N_Treat':>8} {'Mean_Treat':>12} {'N_Ctrl':>8} {'Mean_Ctrl':>12} {'Diff':>10} {'Ratio':>8}")
print(f"  {'-'*70}")

for y in sorted(sub_a.year.unique()):
    ydata = sub_a[sub_a.year == y]
    t_vals = ydata[ydata.Treat == 1]['PriceDelay']
    c_vals = ydata[ydata.Treat == 0]['PriceDelay']
    if len(t_vals) == 0 or len(c_vals) == 0:
        continue
    ratio = t_vals.mean() / c_vals.mean() if c_vals.mean() != 0 else np.nan
    marker = ' <-- POST' if y >= 2024 else ''
    print(f"  {y:>6} {len(t_vals):>8} {t_vals.mean():>12.5f} {len(c_vals):>8} {c_vals.mean():>12.5f} {t_vals.mean()-c_vals.mean():>+10.5f} {ratio:>8.3f}{marker}")

# ============================================================
# 6. ADDITIONAL: DIFF-IN-DIFF PRE-TREND (year-over-year changes)
# ============================================================
print("\n\n" + "=" * 80)
print("DIFF-IN-DIFF: Year-over-Year Changes in PriceDelay")
print("(True parallel trends = constant difference, i.e., similar YoY changes)")
print("=" * 80)

for name, (treat_set, ctrl_set) in treatments.items():
    sub = df[df['Stkcd'].isin(treat_set | ctrl_set)].copy()
    sub['Treat'] = sub['Stkcd'].isin(treat_set).astype(int)
    
    yearly_means = sub.groupby(['year', 'Treat'])['PriceDelay'].mean().unstack()
    if 0 not in yearly_means.columns or 1 not in yearly_means.columns:
        continue
    
    yearly_means.columns = ['Control', 'Treat']
    yearly_means['Diff'] = yearly_means['Treat'] - yearly_means['Control']
    yearly_means['Diff_change'] = yearly_means['Diff'].diff()
    
    pre_changes = yearly_means.loc[yearly_means.index < 2024, 'Diff_change'].dropna()
    
    if len(pre_changes) > 0:
        mean_change = pre_changes.mean()
        std_change = pre_changes.std()
        t_test_change = mean_change / (std_change / np.sqrt(len(pre_changes))) if std_change > 0 else np.nan
        
        print(f"\n  {name}:")
        print(f"    Mean YoY change in Diff (pre-2024): {mean_change:+.5f}")
        print(f"    Std of YoY change: {std_change:.5f}")
        print(f"    t-test (H0: mean change=0): t={t_test_change:+.3f}")

# ============================================================
# 7. FINAL ASSESSMENT
# ============================================================
print("\n\n" + "=" * 80)
print("FINAL ASSESSMENT")
print("=" * 80)

print("""
Key observations:

1. The treatment is defined based on 2017-2021 DU_kw (data keyword density in 
   annual reports), which is a FIRM CHARACTERISTIC, not an exogenous shock.
   Firms with higher data usage tend to be systematically different.

2. For parallel pre-trends to hold, we need that the TREND (not level) of 
   PriceDelay is similar across treatment and control groups before 2024.
   Level differences are absorbed by firm fixed effects in the actual regression.

3. The comparison above shows whether any treatment definition gives cleaner
   pre-trends in terms of:
   - Fewer individually significant year differences
   - Smaller maximum t-statistic
   - No significant linear trend in the differences
   
4. Even with significant level differences, the DID is valid IF those differences
   are CONSTANT over time (parallel trends). Look at the Diff_change analysis
   above for the truest test.
""")

print("Done. Results printed above.")
