"""
Export regression sample with IV variables to Stata format.
Constructs: DU_kw_peer, bartik_iv, DU_kw_lag, peer_lag
Exactly replicates run_endogeneity_v4.py logic.
"""
import os
import pandas as pd
import numpy as np

BASE = "/Users/mac/computerscience/15会计研究"
DATA = f"{BASE}/data_parquet"

# ── Load data: identical to run_endogeneity_v4.py ──
panel = pd.read_parquet(f"{DATA}/panel.parquet")
ar_feat = pd.read_parquet(f"{DATA}/annual_report_features.parquet")
panel = panel.merge(
    ar_feat[['Stkcd','year','kw_total','kw_per10k','substantive_count']],
    on=['Stkcd','year'], how='left')

fi = pd.read_parquet(f"{DATA}/firm_info.parquet",
                     columns=['Symbol','EndDate','IndustryCodeC','LISTINGSTATE'])
fi = fi.rename(columns={'Symbol':'Stkcd'})
fi['EndDate'] = pd.to_datetime(fi['EndDate'])
fi['year'] = fi['EndDate'].dt.year
fi = fi.sort_values(['Stkcd','year','EndDate']).drop_duplicates(
    subset=['Stkcd','year'], keep='last')
panel = panel.merge(fi[['Stkcd','year','IndustryCodeC','LISTINGSTATE']],
                    on=['Stkcd','year'], how='left')

# ── Sample filter: identical to v3/v4 ──
mask_fin = panel['IndustryCodeC'].str.startswith('J', na=False)
mask_st  = panel['LISTINGSTATE'].isin(['ST','*ST'])
mask_new = panel['Age'] <= 0
panel = panel[~mask_fin & ~mask_st & ~mask_new]

# ── Variables ──
panel['DU_kw'] = panel['kw_per10k']
panel['Ind2'] = panel['IndustryCodeC'].str[:3]
panel['IndYear'] = panel['Ind2'].astype(str) + '_' + panel['year'].astype(str)

controls = ['Size','Lev','ROA','TobinQ','Age','Growth','BoardSize',
            'IndepRatio','Dual','Top1Share','SOE','InstHold','Amihud',
            'Analyst','AuditType']

def winsorize(s, lo=0.01, hi=0.99):
    q = s.quantile([lo, hi])
    return s.clip(q.iloc[0], q.iloc[1])

for v in ['PriceDelay','DU_kw','Lev','ROA','Growth','Size','TobinQ','Age',
          'BoardSize','IndepRatio','Top1Share','InstHold','Amihud','Analyst']:
    if v in panel.columns:
        panel[v] = winsorize(panel[v])

df = panel.dropna(subset=['PriceDelay','DU_kw'] + controls).copy()
print(f"Base sample: {len(df):,} obs")

# ── 1. Peer IV: leave-one-out industry-year mean ──
iy = df.groupby(['Ind2','year'])['DU_kw'].agg(['sum','count'])
iy.columns = ['iy_sum','iy_count']
df = df.merge(iy, on=['Ind2','year'], how='left')
df['DU_kw_peer'] = (df['iy_sum'] - df['DU_kw']) / (df['iy_count'] - 1)
df.drop(columns=['iy_sum','iy_count'], inplace=True)
print(f"Peer IV non-missing: {df['DU_kw_peer'].notna().sum():,}")

# ── 2. Bartik IV: base_exposure * leave-out-industry growth ──
base_year = df['year'].min()
ind_base = df[df['year']==base_year].groupby('Ind2')['DU_kw'].mean()

yearly_stats = []
for yr in df['year'].unique():
    yr_data = df[df['year']==yr]
    national_sum = yr_data['DU_kw'].sum()
    national_n = len(yr_data)
    for ind in yr_data['Ind2'].unique():
        ind_data = yr_data[yr_data['Ind2']==ind]
        ind_sum = ind_data['DU_kw'].sum()
        ind_n = len(ind_data)
        if national_n - ind_n > 0:
            excl_mean = (national_sum - ind_sum) / (national_n - ind_n)
        else:
            excl_mean = np.nan
        yearly_stats.append({'Ind2':ind, 'year':yr, 'DU_excl_mean':excl_mean})

excl_df = pd.DataFrame(yearly_stats)
excl_base = excl_df[excl_df['year']==base_year].set_index('Ind2')['DU_excl_mean']
excl_df['excl_base'] = excl_df['Ind2'].map(excl_base)
excl_df['growth_excl'] = excl_df['DU_excl_mean'] / excl_df['excl_base'].clip(lower=0.001)
excl_df['base_exposure'] = excl_df['Ind2'].map(ind_base)
excl_df['bartik_iv'] = excl_df['base_exposure'] * excl_df['growth_excl']

df = df.merge(excl_df[['Ind2','year','bartik_iv']], on=['Ind2','year'], how='left')
print(f"Bartik IV non-missing: {df['bartik_iv'].notna().sum():,}")

# ── 3. Lag variables ──
df = df.sort_values(['Stkcd','year'])
df['DU_kw_lag'] = df.groupby('Stkcd')['DU_kw'].shift(1)

# Peer lag: leave-one-out of DU_kw_lag
df_has_lag = df.dropna(subset=['DU_kw_lag']).copy()
iy_lag = df_has_lag.groupby(['Ind2','year'])['DU_kw_lag'].agg(['sum','count'])
iy_lag.columns = ['iyl_sum','iyl_count']
df = df.merge(iy_lag, on=['Ind2','year'], how='left')
df['peer_lag'] = (df['iyl_sum'] - df['DU_kw_lag']) / (df['iyl_count'] - 1)
df.drop(columns=['iyl_sum','iyl_count'], inplace=True)
print(f"DU_kw_lag non-missing: {df['DU_kw_lag'].notna().sum():,}")
print(f"peer_lag non-missing: {df['peer_lag'].notna().sum():,}")

# ── Encode for Stata ──
# Numeric firm/year identifiers
df['Stkcd_num'] = df['Stkcd'].astype(int)
df['year_num'] = df['year'].astype(int)

# Encode IndYear as numeric
indyear_map = {v: i+1 for i, v in enumerate(sorted(df['IndYear'].unique()))}
df['IndYear_num'] = df['IndYear'].map(indyear_map)

# Select columns for export
export_cols = ['Stkcd_num', 'year_num', 'PriceDelay', 'DU_kw',
               'DU_kw_peer', 'bartik_iv', 'DU_kw_lag', 'peer_lag',
               'IndYear_num'] + controls
out = df[export_cols].copy()

# Fill string-type Ind2/IndYear not needed; all export cols are numeric
out_path = f"{BASE}/data_stata/reg_sample_iv.dta"
out.to_stata(out_path, write_index=False, version=118)
print(f"\nSaved: {out_path}")
print(f"Shape: {out.shape}")
print(f"Columns: {list(out.columns)}")
