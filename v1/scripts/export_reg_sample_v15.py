"""
Export regression sample to Stata .dta format with 27 control variables.
Output: data_stata/reg_sample_v5.dta

Follows the same preprocessing as generate_tables_v13.py.
27 controls (Opinion removed, identical to AuditType):
  Size, Lev, ROA, TobinQ, Age, Growth, BoardSize, IndepRatio, Dual, Top1Share,
  SOE, InstHold, Amihud, Analyst, AuditType, CFO, RetVol, Turnover, Intangible,
  PPE, BM, Employee, ShareholderNum, Manhold, Balance, Separation, Market
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

BASE = "/Users/mac/computerscience/15会计研究"

# ============================================================
# 1. Load panel_dml (base data with all controls)
# ============================================================
print("Loading data...")
panel = pd.read_parquet(f"{BASE}/data_parquet/panel_dml.parquet")
print(f"  panel_dml: {panel.shape}")

# ============================================================
# 2. Merge annual_report_features for kw variables
# ============================================================
ar_feat = pd.read_parquet(f"{BASE}/data_parquet/annual_report_features.parquet")
panel = panel.merge(
    ar_feat[['Stkcd','year','kw_total','kw_per10k','substantive_count',
             'kw_data_stock','kw_data_dev','kw_data_app','kw_data_value','kw_data_gov']],
    on=['Stkcd','year'], how='left'
)
print(f"  After merge ar_feat: {panel.shape}")

# ============================================================
# 3. Merge firm_info for IndustryCodeC, LISTINGSTATE, PROVINCE
# ============================================================
fi = pd.read_parquet(f"{BASE}/data_parquet/firm_info.parquet",
                     columns=['Symbol','EndDate','IndustryCodeC','LISTINGSTATE','PROVINCE'])
fi = fi.rename(columns={'Symbol':'Stkcd'})
fi['EndDate'] = pd.to_datetime(fi['EndDate'])
fi['year'] = fi['EndDate'].dt.year
fi = fi.sort_values(['Stkcd','year','EndDate']).drop_duplicates(subset=['Stkcd','year'], keep='last')

# ============================================================
# 4. Fill missing IndustryCodeC with company mode
# ============================================================
ind_mode = fi.groupby('Stkcd')['IndustryCodeC'].agg(
    lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else np.nan)
panel = panel.merge(ind_mode.rename('IndCode_mode').reset_index(), on='Stkcd', how='left')
panel = panel.merge(fi[['Stkcd','year','IndustryCodeC','LISTINGSTATE','PROVINCE']],
                    on=['Stkcd','year'], how='left')
panel['IndustryCodeC'] = panel['IndustryCodeC'].fillna(panel['IndCode_mode'])

# Fill Province with company mode too
prov_mode = fi.groupby('Stkcd')['PROVINCE'].agg(
    lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else np.nan)
panel = panel.merge(prov_mode.rename('Prov_mode').reset_index(), on='Stkcd', how='left')
panel['PROVINCE'] = panel['PROVINCE'].fillna(panel['Prov_mode'])

# ============================================================
# 5. Filter: remove finance (J), ST, Age<=0
# ============================================================
mask_fin = panel['IndustryCodeC'].str.startswith('J', na=False)
mask_st = panel['LISTINGSTATE'].isin(['ST','*ST'])
mask_new = panel['Age'] <= 0
print(f"  Filtering: finance={mask_fin.sum()}, ST={mask_st.sum()}, Age<=0={mask_new.sum()}")
panel = panel[~mask_fin & ~mask_st & ~mask_new]
print(f"  After filter: {panel.shape}")

# ============================================================
# 6. Construct variables
# ============================================================
panel['Ind2'] = panel['IndustryCodeC'].str[:3]
panel['IndYear'] = panel['Ind2'].astype(str) + '_' + panel['year'].astype(str)
panel['DU_kw'] = panel['kw_per10k']
panel['DU_kw_ln'] = np.log1p(panel['kw_total'])
panel['DU_sub_ln'] = np.log1p(panel['substantive_count'])
panel['Province'] = panel['PROVINCE']
panel['ProvYear'] = panel['Province'].astype(str) + '_' + panel['year'].astype(str)

# MainBoard
panel['MainBoard'] = panel['Stkcd'].astype(str).str.match(r'^(00[01]|6)').astype(int)

# ============================================================
# 7. Merge SYNCH
# ============================================================
synch = pd.read_parquet(f"{BASE}/data_parquet/price_synchronicity.parquet")
panel = panel.merge(synch[['Stkcd','year','SYNCH']], on=['Stkcd','year'], how='left')

# ============================================================
# 7b. Construct Inv (存货比例) and Hhi (行业竞争度)
# ============================================================
bs = pd.read_parquet(f"{BASE}/data_parquet/balance_sheet.parquet")
bs['EndDate'] = pd.to_datetime(bs['Accper'])
bs['year'] = bs['EndDate'].dt.year
bs = bs[bs['Typrep'] == 'A'].sort_values(['Stkcd','year','EndDate']).drop_duplicates(['Stkcd','year'], keep='last')
bs['Inv'] = bs['A001218000'] / bs['A001000000']
panel = panel.merge(bs[['Stkcd','year','Inv']], on=['Stkcd','year'], how='left')

inc = pd.read_parquet(f"{BASE}/data_parquet/income_stmt.parquet")
inc['EndDate'] = pd.to_datetime(inc['Accper'])
inc['year'] = inc['EndDate'].dt.year
inc = inc[inc['Typrep'] == 'A'].sort_values(['Stkcd','year','EndDate']).drop_duplicates(['Stkcd','year'], keep='last')
inc_ind = panel[['Stkcd','year','Ind2']].merge(inc[['Stkcd','year','B001101000']], on=['Stkcd','year'], how='left')
import numpy as np
hhi = inc_ind.groupby(['Ind2','year']).apply(
    lambda g: ((g['B001101000'] / g['B001101000'].sum()) ** 2).sum() if g['B001101000'].sum() > 0 else np.nan
).reset_index(name='Hhi')
panel = panel.merge(hhi, on=['Ind2','year'], how='left')
print(f"  Inv non-null: {panel['Inv'].notna().sum()}, Hhi non-null: {panel['Hhi'].notna().sum()}")

# ============================================================
# 8. Construct DU_kw_lead (t+1) and DU_kw_indadj
# ============================================================
panel = panel.sort_values(['Stkcd','year'])
panel['DU_kw_lead'] = panel.groupby('Stkcd')['DU_kw'].shift(-1)

# Industry-year mean adjusted
ind_year_mean = panel.groupby(['Ind2','year'])['DU_kw'].transform('mean')
panel['DU_kw_indadj'] = panel['DU_kw'] - ind_year_mean

# DU_kw_high (above median, for PSM)
med_dukw = panel['DU_kw'].median()
panel['DU_kw_high'] = (panel['DU_kw'] > med_dukw).astype(int)

# ============================================================
# 9. Winsorize continuous vars at 1%/99%
# ============================================================
def winsorize(s):
    lo, hi = s.quantile([0.01, 0.99])
    return s.clip(lo, hi)

controls = ['Size','Lev','ROA','TobinQ','Age','Growth','IndepRatio',
            'Dual','Top1Share','SOE','CFO','Inv','Hhi']

cont_vars = ['PriceDelay','DU_kw','DU_kw_ln','DU_sub_ln','SYNCH','FinAsset',
             'Lev','ROA','Growth','Size','TobinQ','Age','IndepRatio',
             'Top1Share','CFO','Inv','Hhi']
for v in cont_vars:
    if v in panel.columns and panel[v].notna().any():
        panel[v] = winsorize(panel[v])

# ============================================================
# 10. Create regression sample (drop missing on key vars + 27 controls)
# ============================================================
reg = panel.dropna(subset=['PriceDelay','DU_kw'] + controls).copy()
print(f"  Regression sample: N={len(reg):,}, firms={reg.Stkcd.nunique():,}")

# ============================================================
# 11. Create numeric IDs for Stata
# ============================================================
reg['Stkcd_num'] = reg['Stkcd'].astype('category').cat.codes + 1
reg['year_num'] = reg['year'].astype(int)

# Ind2 numeric
ind2_cats = sorted(reg['Ind2'].dropna().unique())
ind2_map = {v: i+1 for i, v in enumerate(ind2_cats)}
reg['Ind2_num'] = reg['Ind2'].map(ind2_map)

# IndYear numeric
iy_cats = sorted(reg['IndYear'].dropna().unique())
iy_map = {v: i+1 for i, v in enumerate(iy_cats)}
reg['IndYear_num'] = reg['IndYear'].map(iy_map)

# Province numeric
prov_cats = sorted(reg['Province'].dropna().unique())
prov_map = {v: i+1 for i, v in enumerate(prov_cats)}
reg['Prov_num'] = reg['Province'].map(prov_map)

# ProvYear numeric
py_cats = sorted(reg['ProvYear'].dropna().unique())
py_map = {v: i+1 for i, v in enumerate(py_cats)}
reg['ProvYear_num'] = reg['ProvYear'].map(py_map)

# ============================================================
# 12. Select columns for export
# ============================================================
export_cols = (
    # Core variables
    ['Stkcd', 'year', 'PriceDelay', 'SYNCH', 'DU_kw', 'DU_kw_ln', 'DU_sub_ln', 'FinAsset',
     'DU_kw_lead', 'DU_kw_indadj', 'DU_kw_high']
    # 27 controls
    + controls
    # FE and grouping
    + ['Ind2', 'IndYear', 'Province', 'ProvYear', 'MainBoard']
    # Numeric IDs
    + ['Stkcd_num', 'year_num', 'Ind2_num', 'IndYear_num', 'Prov_num', 'ProvYear_num']
)

# Only keep columns that exist
export_cols = [c for c in export_cols if c in reg.columns]
reg_export = reg[export_cols].copy()

# Verify
print(f"\n  Export columns ({len(export_cols)}):")
for c in export_cols:
    nn = reg_export[c].notna().sum()
    print(f"    {c}: {nn:,} non-null ({nn/len(reg_export)*100:.1f}%)")

# ============================================================
# 13. Save to Stata .dta
# ============================================================
os.makedirs(f"{BASE}/data_stata", exist_ok=True)
output_path = f"{BASE}/data_stata/reg_sample_v5.dta"
reg_export.to_stata(output_path, write_index=False, version=118)
print(f"\nSaved to: {output_path}")
print(f"  N={len(reg_export):,}, Cols={len(export_cols)}")
