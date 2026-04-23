"""
扩展机制检验和异质性分析 (v10 sample)
渠道: Analyst, Disp, InstHold, FinAsset (同期 + 滞后一期)
异质性: SOE/Amihud交互 + SOE/Analyst/Size/HighTech分组
"""

import pandas as pd
import numpy as np
import pyfixest as pf
import json, warnings, os
warnings.filterwarnings('ignore')

BASE = "/Users/mac/computerscience/15会计研究"

# ============================================================
# 1. Data loading - exact v10 sample construction
# ============================================================
print("Loading data...")
panel = pd.read_parquet(f"{BASE}/data_parquet/panel.parquet")
ar_feat = pd.read_parquet(f"{BASE}/data_parquet/annual_report_features.parquet")

panel = panel.merge(
    ar_feat[['Stkcd','year','kw_total','kw_per10k','substantive_count',
             'kw_data_stock','kw_data_dev','kw_data_app','kw_data_value','kw_data_gov']],
    on=['Stkcd','year'], how='left'
)

fi = pd.read_parquet(f"{BASE}/data_parquet/firm_info.parquet",
                     columns=['Symbol','EndDate','IndustryCodeC','LISTINGSTATE'])
fi = fi.rename(columns={'Symbol':'Stkcd'})
fi['EndDate'] = pd.to_datetime(fi['EndDate'])
fi['year'] = fi['EndDate'].dt.year
fi = fi.sort_values(['Stkcd','year','EndDate']).drop_duplicates(subset=['Stkcd','year'], keep='last')

ind_mode = fi.groupby('Stkcd')['IndustryCodeC'].agg(
    lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else np.nan)
panel = panel.merge(ind_mode.rename('IndCode_mode').reset_index(), on='Stkcd', how='left')
panel = panel.merge(fi[['Stkcd','year','IndustryCodeC','LISTINGSTATE']], on=['Stkcd','year'], how='left')
panel['IndustryCodeC'] = panel['IndustryCodeC'].fillna(panel['IndCode_mode'])

mask_fin = panel['IndustryCodeC'].str.startswith('J', na=False)
mask_st = panel['LISTINGSTATE'].isin(['ST','*ST'])
mask_new = panel['Age'] <= 0
panel = panel[~mask_fin & ~mask_st & ~mask_new]

panel['Ind2'] = panel['IndustryCodeC'].str[:3]
panel['IndYear'] = panel['Ind2'].astype(str) + '_' + panel['year'].astype(str)
panel['DU_kw'] = panel['kw_per10k']
panel['DU_kw_ln'] = np.log1p(panel['kw_total'])
panel['DU_sub_ln'] = np.log1p(panel['substantive_count'])
panel['MainBoard'] = panel['Stkcd'].astype(str).str.match(r'^(00[01]|6)')

# ============================================================
# 2. Construct Disp (analyst forecast dispersion)
# ============================================================
print("Constructing Disp from analyst forecasts...")
af = pd.read_parquet(f"{BASE}/data_parquet/analyst_forecast.parquet")
af['Rptdt'] = pd.to_datetime(af['Rptdt'])
af['Fenddt'] = pd.to_datetime(af['Fenddt'])
# Keep forecasts where forecast target year = report year
af['rpt_year'] = af['Rptdt'].dt.year
af['fend_year'] = af['Fenddt'].dt.year
# Use current-year forecasts: forecast issued in year t for fiscal year t
af_curr = af[af['rpt_year'] == af['fend_year']].copy()
af_curr = af_curr.dropna(subset=['Feps'])

# Compute dispersion: SD of EPS forecasts per stock-year (need >=2 analysts)
disp = af_curr.groupby(['Stkcd','fend_year'])['Feps'].agg(['std','count']).reset_index()
disp.columns = ['Stkcd','year','Disp','n_forecasts']
disp = disp[disp['n_forecasts'] >= 2]
disp = disp[['Stkcd','year','Disp']]

panel = panel.merge(disp, on=['Stkcd','year'], how='left')
print(f"  Disp available: {panel['Disp'].notna().sum():,} obs")

# ============================================================
# 3. Winsorize & prepare regression sample
# ============================================================
def winsorize(s):
    lo, hi = s.quantile([0.01, 0.99])
    return s.clip(lo, hi)

controls = ['Size','Lev','ROA','TobinQ','Age','Growth','BoardSize','IndepRatio',
            'Dual','Top1Share','SOE','InstHold','Amihud','Analyst','AuditType']

cont_vars = ['PriceDelay','DU_kw','DU_kw_ln','DU_sub_ln','FinAsset',
             'Lev','ROA','Growth','Size','TobinQ','Age','BoardSize','IndepRatio',
             'Top1Share','InstHold','Amihud','Analyst','Disp']
for v in cont_vars:
    if v in panel.columns and panel[v].notna().any():
        panel[v] = winsorize(panel[v])

reg = panel.dropna(subset=['PriceDelay','DU_kw'] + controls).copy()
reg['Stkcd_str'] = reg['Stkcd'].astype(str)
reg['year_str'] = reg['year'].astype(str)
print(f"Regression sample: N={len(reg):,}, firms={reg.Stkcd.nunique():,}")

ctrl_str = " + ".join(controls)

def sig_stars(p):
    if p is None or pd.isna(p): return ""
    if p < 0.01: return "***"
    if p < 0.05: return "**"
    if p < 0.1:  return "*"
    return ""

# ============================================================
# 4. Mechanism: concurrent and lagged (江艇 two-step)
# ============================================================
print("\n=== MECHANISM (expanded) ===")

mech_vars = {
    'Analyst': 'Analyst',
    'Disp': 'Disp',
    'InstHold': 'InstHold',
    'FinAsset': 'FinAsset',
}

results_mech = {}

for label, dv in mech_vars.items():
    print(f"\n--- {label} ---")

    # Exclude DV from controls to avoid multicollinearity
    mech_controls = [c for c in controls if c != dv]
    mech_ctrl_str = " + ".join(mech_controls)

    # Panel A: concurrent X_t -> M_t
    sub = reg.dropna(subset=[dv]).copy()
    fml = f"{dv} ~ DU_kw + {mech_ctrl_str} | Stkcd_str + year_str"
    m = pf.feols(fml, data=sub, vcov={"CRV1": "IndYear"})
    coef = float(m.coef()['DU_kw'])
    se = float(m.se()['DU_kw'])
    pval = float(m.pvalue()['DU_kw'])
    stars = sig_stars(pval)
    n = m._N
    r2 = m._r2
    print(f"  Concurrent: coef={coef:.4f}, se={se:.4f}, t={coef/se:.2f}, p={pval:.4f}{stars}, N={n}, R2={r2:.3f}")

    results_mech[f'{label}_concurrent'] = {
        'coef': coef, 'se': se, 'sig': stars, 'N': n, 'R2': r2, 'p': pval
    }

    # Panel B: lagged X_t -> M_{t+1}
    lag_df = reg[['Stkcd','year','DU_kw'] + controls + ['Stkcd_str','year_str','IndYear']].copy()
    future_m = panel[['Stkcd','year',dv]].copy()
    future_m['year'] = future_m['year'] - 1  # shift: year t's M becomes year t-1's "future M"
    future_m = future_m.rename(columns={dv: f'{dv}_lead'})
    lag_df = lag_df.merge(future_m, on=['Stkcd','year'], how='inner')
    lag_df = lag_df.dropna(subset=[f'{dv}_lead'])

    fml_lag = f"{dv}_lead ~ DU_kw + {mech_ctrl_str} | Stkcd_str + year_str"
    m_lag = pf.feols(fml_lag, data=lag_df, vcov={"CRV1": "IndYear"})
    coef_l = float(m_lag.coef()['DU_kw'])
    se_l = float(m_lag.se()['DU_kw'])
    pval_l = float(m_lag.pvalue()['DU_kw'])
    stars_l = sig_stars(pval_l)
    n_l = m_lag._N
    r2_l = m_lag._r2
    print(f"  Lagged:     coef={coef_l:.4f}, se={se_l:.4f}, t={coef_l/se_l:.2f}, p={pval_l:.4f}{stars_l}, N={n_l}, R2={r2_l:.3f}")

    results_mech[f'{label}_lagged'] = {
        'coef': coef_l, 'se': se_l, 'sig': stars_l, 'N': n_l, 'R2': r2_l, 'p': pval_l
    }

# ============================================================
# 5. Heterogeneity: interaction + subgroup
# ============================================================
print("\n=== HETEROGENEITY (expanded) ===")

base_fml = f"PriceDelay ~ DU_kw + {{interactions}} + {ctrl_str} | Stkcd_str + year_str"

# --- Panel A: Interaction regressions ---
print("\n--- Panel A: Interactions ---")

# Center Amihud
reg['Amihud_c'] = reg['Amihud'] - reg['Amihud'].mean()

# (1) DU_kw × SOE
reg['DU_SOE'] = reg['DU_kw'] * reg['SOE']
fml1 = f"PriceDelay ~ DU_kw + DU_SOE + {ctrl_str} | Stkcd_str + year_str"
m1 = pf.feols(fml1, data=reg, vcov={"CRV1": "IndYear"})
print(f"  (1) DU_kw×SOE: coef={float(m1.coef()['DU_SOE']):.4f}, se={float(m1.se()['DU_SOE']):.4f}, sig={sig_stars(float(m1.pvalue()['DU_SOE']))}")

# (2) DU_kw × Amihud_c
reg['DU_Amihud'] = reg['DU_kw'] * reg['Amihud_c']
fml2 = f"PriceDelay ~ DU_kw + DU_Amihud + {ctrl_str} | Stkcd_str + year_str"
m2 = pf.feols(fml2, data=reg, vcov={"CRV1": "IndYear"})
print(f"  (2) DU_kw×Amihud_c: coef={float(m2.coef()['DU_Amihud']):.4f}, se={float(m2.se()['DU_Amihud']):.4f}, sig={sig_stars(float(m2.pvalue()['DU_Amihud']))}")

# (3) Both
fml3 = f"PriceDelay ~ DU_kw + DU_SOE + DU_Amihud + {ctrl_str} | Stkcd_str + year_str"
m3 = pf.feols(fml3, data=reg, vcov={"CRV1": "IndYear"})
print(f"  (3) Both: SOE coef={float(m3.coef()['DU_SOE']):.4f} sig={sig_stars(float(m3.pvalue()['DU_SOE']))}, Amihud coef={float(m3.coef()['DU_Amihud']):.4f} sig={sig_stars(float(m3.pvalue()['DU_Amihud']))}")

# (4) DU_kw × HighAnalyst
med_analyst = reg['Analyst'].median()
reg['HighAnalyst'] = (reg['Analyst'] >= med_analyst).astype(int)
reg['DU_HighAnalyst'] = reg['DU_kw'] * reg['HighAnalyst']
fml4 = f"PriceDelay ~ DU_kw + DU_HighAnalyst + {ctrl_str} | Stkcd_str + year_str"
m4 = pf.feols(fml4, data=reg, vcov={"CRV1": "IndYear"})
print(f"  (4) DU_kw×HighAnalyst: coef={float(m4.coef()['DU_HighAnalyst']):.4f}, se={float(m4.se()['DU_HighAnalyst']):.4f}, sig={sig_stars(float(m4.pvalue()['DU_HighAnalyst']))}")

interact_results = {}
for label, model, var in [
    ('col1', m1, 'DU_SOE'), ('col2', m2, 'DU_Amihud'),
    ('col3', m3, 'DU_SOE'), ('col3b', m3, 'DU_Amihud'),
    ('col4', m4, 'DU_HighAnalyst')
]:
    interact_results[label] = {
        'DU_kw': {'coef': float(model.coef()['DU_kw']), 'se': float(model.se()['DU_kw']),
                  'sig': sig_stars(float(model.pvalue()['DU_kw']))},
        var: {'coef': float(model.coef()[var]), 'se': float(model.se()[var]),
              'sig': sig_stars(float(model.pvalue()[var]))},
        'N': model._N, 'R2': model._r2
    }

# --- Panel B: Subgroup regressions ---
print("\n--- Panel B: Subgroup regressions ---")

subgroup_results = {}

def run_subgroup(data, label):
    fml = f"PriceDelay ~ DU_kw + {ctrl_str} | Stkcd_str + year_str"
    m = pf.feols(fml, data=data, vcov={"CRV1": "IndYear"})
    coef = float(m.coef()['DU_kw'])
    se = float(m.se()['DU_kw'])
    pval = float(m.pvalue()['DU_kw'])
    stars = sig_stars(pval)
    print(f"  {label}: coef={coef:.4f}, se={se:.4f}, t={coef/se:.2f}, p={pval:.4f}{stars}, N={m._N}")
    subgroup_results[label] = {
        'coef': coef, 'se': se, 'sig': stars, 'N': m._N, 'R2': m._r2, 'p': pval
    }

# SOE vs Non-SOE
run_subgroup(reg[reg['SOE'] == 1], '国有')
run_subgroup(reg[reg['SOE'] == 0], '非国有')

# High vs Low Analyst
run_subgroup(reg[reg['HighAnalyst'] == 1], '高分析师覆盖')
run_subgroup(reg[reg['HighAnalyst'] == 0], '低分析师覆盖')

# Size: median split
med_size = reg['Size'].median()
run_subgroup(reg[reg['Size'] >= med_size], '大企业')
run_subgroup(reg[reg['Size'] < med_size], '小企业')

# HighTech: ICT industries (I类信息传输+软件, C39计算机通信电子)
reg['HighTech'] = reg['IndustryCodeC'].str.startswith('I', na=False) | \
                  reg['IndustryCodeC'].str.startswith('C39', na=False)
run_subgroup(reg[reg['HighTech'] == True], '高科技行业')
run_subgroup(reg[reg['HighTech'] == False], '传统行业')

# Fisher test for group differences
print("\n--- Fisher z-tests ---")
from scipy import stats as sp_stats

def fisher_z_test(b1, se1, b2, se2):
    z = (b1 - b2) / np.sqrt(se1**2 + se2**2)
    p = 2 * (1 - sp_stats.norm.cdf(abs(z)))
    return z, p

fisher_results = {}
for dim, g1, g2 in [
    ('产权性质', '国有', '非国有'),
    ('分析师覆盖', '高分析师覆盖', '低分析师覆盖'),
    ('企业规模', '大企业', '小企业'),
    ('行业属性', '高科技行业', '传统行业'),
]:
    r1, r2 = subgroup_results[g1], subgroup_results[g2]
    z, p = fisher_z_test(r1['coef'], r1['se'], r2['coef'], r2['se'])
    print(f"  {dim}: z={z:.2f}, p={p:.3f}")
    fisher_results[dim] = {'z': z, 'p': p}

# ============================================================
# 6. Save all results
# ============================================================
output = {
    'mechanism': results_mech,
    'interact': interact_results,
    'subgroup': subgroup_results,
    'fisher': fisher_results,
}

os.makedirs(f"{BASE}/results/v11_expanded", exist_ok=True)
with open(f"{BASE}/results/v11_expanded/expanded_results.json", 'w') as f:
    json.dump(output, f, indent=2, default=str)

print(f"\nResults saved to results/v11_expanded/expanded_results.json")
print("\n=== SUMMARY ===")
print(f"Mechanism channels: {len(mech_vars)} × 2 (concurrent + lagged)")
print(f"Interaction models: 4")
print(f"Subgroup comparisons: {len(subgroup_results)} groups, {len(fisher_results)} Fisher tests")
