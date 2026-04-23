"""
Export reg_sample_v3.dta — mirrors generate_tables_v9_1.py preprocessing
plus new variables for Table 3 robustness:
  - Province, Prov_num, ProvYear_num
  - MainBoard
  - DU_kw_lead (t+1)
  - DU_kw_indadj (industry-year mean adjusted)
"""

import pandas as pd
import numpy as np

BASE = "/Users/mac/computerscience/15会计研究"

# --- Load data (same as v9_1) ---
panel = pd.read_parquet(f"{BASE}/data_parquet/panel.parquet")
ar_feat = pd.read_parquet(f"{BASE}/data_parquet/annual_report_features.parquet")

panel = panel.merge(
    ar_feat[['Stkcd','year','kw_total','kw_per10k','substantive_count']],
    on=['Stkcd','year'], how='left'
)

fi = pd.read_parquet(f"{BASE}/data_parquet/firm_info.parquet",
                     columns=['Symbol','EndDate','IndustryCodeC','LISTINGSTATE','PROVINCECODE'])
fi = fi.rename(columns={'Symbol':'Stkcd'})
fi['EndDate'] = pd.to_datetime(fi['EndDate'])
fi['year'] = fi['EndDate'].dt.year
fi = fi.sort_values(['Stkcd','year','EndDate']).drop_duplicates(subset=['Stkcd','year'], keep='last')

# Industry code mode fill (v9.1 fix)
ind_mode = fi.groupby('Stkcd')['IndustryCodeC'].agg(
    lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else np.nan)
panel = panel.merge(ind_mode.rename('IndCode_mode').reset_index(), on='Stkcd', how='left')
panel = panel.merge(fi[['Stkcd','year','IndustryCodeC','LISTINGSTATE','PROVINCECODE']],
                    on=['Stkcd','year'], how='left')
panel['IndustryCodeC'] = panel['IndustryCodeC'].fillna(panel['IndCode_mode'])

# Province mode fill (same logic as industry)
prov_mode = fi.groupby('Stkcd')['PROVINCECODE'].agg(
    lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else np.nan)
panel = panel.merge(prov_mode.rename('ProvCode_mode').reset_index(), on='Stkcd', how='left')
panel['PROVINCECODE'] = panel['PROVINCECODE'].fillna(panel['ProvCode_mode'])
panel['Province'] = panel['PROVINCECODE']

# --- Sample filters (same as v9_1) ---
mask_fin = panel['IndustryCodeC'].str.startswith('J', na=False)
mask_st = panel['LISTINGSTATE'].isin(['ST', '*ST'])
mask_new = panel['Age'] <= 0
panel = panel[~mask_fin & ~mask_st & ~mask_new]
print(f"After filters: {len(panel)} obs")

# --- Construct variables ---
panel['Ind2'] = panel['IndustryCodeC'].str[:3]
panel['IndYear'] = panel['Ind2'].astype(str) + '_' + panel['year'].astype(str)
panel['DU_kw'] = panel['kw_per10k']
panel['DU_kw_ln'] = np.log1p(panel['kw_total'])
panel['DU_sub_ln'] = np.log1p(panel['substantive_count'])

# SYNCH from separate file
synch = pd.read_parquet(f"{BASE}/data_parquet/price_synchronicity.parquet")
panel = panel.merge(synch[['Stkcd','year','SYNCH']], on=['Stkcd','year'], how='left')

# MainBoard (same as v9_1)
panel['MainBoard'] = panel['Stkcd'].astype(str).str.match(r'^(00[01]|6)').astype(int)

# ProvYear
panel['ProvYear'] = panel['Province'].astype(str) + '_' + panel['year'].astype(str)

# --- Winsorize ---
def winsorize(s):
    lo, hi = s.quantile([0.01, 0.99])
    return s.clip(lo, hi)

controls = ['Size','Lev','ROA','TobinQ','Age','Growth','BoardSize','IndepRatio',
            'Dual','Top1Share','SOE','InstHold','Amihud','Analyst','AuditType']

cont_vars = ['PriceDelay','DU_kw','DU_kw_ln','DU_sub_ln','SYNCH','FinAsset',
             'Lev','ROA','Growth','Size','TobinQ','Age','BoardSize','IndepRatio',
             'Top1Share','InstHold','Amihud','Analyst']
for v in cont_vars:
    if v in panel.columns and panel[v].notna().any():
        panel[v] = winsorize(panel[v])

# --- Regression sample ---
reg = panel.dropna(subset=['PriceDelay','DU_kw'] + controls).copy()
print(f"Regression sample: N={len(reg):,}, firms={reg['Stkcd'].nunique():,}")

# --- New variables for Table 3 ---
# DU_kw_lead: forward one period
reg = reg.sort_values(['Stkcd', 'year'])
reg['DU_kw_lead'] = reg.groupby('Stkcd')['DU_kw'].shift(-1)
print(f"DU_kw_lead non-null: {reg['DU_kw_lead'].notna().sum()}")

# DU_kw_indadj: subtract industry-year mean
ind_year_mean = reg.groupby(['Ind2', 'year'])['DU_kw'].transform('mean')
reg['DU_kw_indadj'] = reg['DU_kw'] - ind_year_mean
print(f"DU_kw_indadj non-null: {reg['DU_kw_indadj'].notna().sum()}")

# --- Numeric encodings for Stata absorb() ---
for col, num_col in [('Stkcd', 'Stkcd_num'), ('Ind2', 'Ind2_num'),
                      ('IndYear', 'IndYear_num'), ('year', 'year_num'),
                      ('Province', 'Prov_num'), ('ProvYear', 'ProvYear_num')]:
    if col in reg.columns:
        reg[num_col] = pd.Categorical(reg[col]).codes

# --- Export ---
export_cols = ['Stkcd', 'year', 'PriceDelay', 'DU_kw', 'DU_kw_ln', 'DU_sub_ln',
               'DU_kw_lead', 'DU_kw_indadj',
               'SYNCH', 'FinAsset', 'Ind2', 'IndYear', 'Province', 'ProvYear',
               'MainBoard',
               'Size', 'Lev', 'ROA', 'TobinQ', 'Age', 'Growth',
               'BoardSize', 'IndepRatio', 'Dual', 'Top1Share', 'SOE',
               'InstHold', 'Amihud', 'Analyst', 'AuditType',
               'Stkcd_num', 'Ind2_num', 'IndYear_num', 'year_num',
               'Prov_num', 'ProvYear_num']

out_path = f"{BASE}/data_stata/reg_sample_v3.dta"
reg[export_cols].to_stata(out_path, write_index=False, version=118)

print(f"\nExported to {out_path}")
print(f"N={len(reg):,}, Firms={reg['Stkcd'].nunique():,}")
print(f"Year range: {reg['year'].min()}-{reg['year'].max()}")
print(f"Province missing: {reg['Province'].isna().sum()}")
print(f"Ind2 missing: {reg['Ind2'].isna().sum()}")
print(f"MainBoard: {reg['MainBoard'].value_counts().to_dict()}")
