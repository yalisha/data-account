"""
Construct new mechanism channels for v18:
1. TFP: Total Factor Productivity (Levinsohn-Petrin method, simplified)
2. CrashRisk: Stock price crash risk (NCSKEW)
3. CashFlowVol: Cash flow volatility (3-year rolling)
4. DUVOL: Down-to-up volatility ratio (alternative crash risk)

Merge to reg_sample_v18.dta and re-export.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

DATA = Path('/Users/mac/computerscience/15会计研究/data_parquet')
STATA = Path('/Users/mac/computerscience/15会计研究/data_stata')

def winsorize(s, lo=0.01, hi=0.99):
    q = s.quantile([lo, hi])
    return s.clip(q.iloc[0], q.iloc[1])

# ── Load base ──
print("Loading reg_sample_v18.dta...")
base = pd.read_stata(STATA / 'reg_sample_v18.dta')
base['Stkcd'] = base['Stkcd'].astype(int)
base['year'] = base['year'].astype(int)
print(f"  {len(base)} obs, {base.columns.shape[0]} cols")

# ── Industry codes ──
fi = pd.read_parquet(DATA / 'firm_info.parquet')
fi['Stkcd'] = pd.to_numeric(fi['Symbol'], errors='coerce')
fi_ind = fi[fi['IndustryCodeC'].notna()][['Stkcd', 'IndustryCodeC']].drop_duplicates(
    subset=['Stkcd'], keep='last')
fi_ind['Stkcd'] = fi_ind['Stkcd'].astype(int)
fi_ind['Ind2'] = fi_ind['IndustryCodeC'].str[:3]

# ════════════════════════════════════════════
# 1. TFP (Levinsohn-Petrin, simplified)
# ════════════════════════════════════════════
print("\n[1] TFP (LP method)...")
# Y = Revenue, L = Employees, K = Fixed Assets, M = intermediate inputs (COGS - depreciation approx)
# Simplified: use OLS residual from Cobb-Douglas by industry-year as TFP proxy
# ln(Y) = a + b1*ln(L) + b2*ln(K) + b3*ln(M) + epsilon
# TFP = epsilon (residual)

bs = pd.read_parquet(DATA / 'balance_sheet.parquet')
inc = pd.read_parquet(DATA / 'income_stmt.parquet')

for df in [bs, inc]:
    df['Stkcd'] = pd.to_numeric(df['Stkcd'], errors='coerce')
    df['Accper'] = pd.to_datetime(df['Accper'], errors='coerce')

def get_annual(df, cols):
    d = df[(df['Accper'].dt.month == 12) & (df['Typrep'] == 'A')].copy()
    d['year'] = d['Accper'].dt.year
    d = d.drop_duplicates(subset=['Stkcd', 'year'], keep='last')
    return d[['Stkcd', 'year'] + [c for c in cols if c in d.columns]]

# B001101000 = Revenue, B001300000 = Operating cost (COGS)
# A001202000 = Fixed assets, A001000000 = Total assets
inc_a = get_annual(inc, ['B001101000', 'B001300000', 'B002000000'])
inc_a.columns = ['Stkcd', 'year', 'Revenue', 'COGS', 'NI']

bs_a = get_annual(bs, ['A001202000', 'A001000000'])
bs_a.columns = ['Stkcd', 'year', 'FA', 'TA']

# Employee data from panel_dml
dml = pd.read_parquet(DATA / 'panel_dml.parquet')
emp = dml[['Stkcd', 'year', 'Employee']].dropna()
emp['Stkcd'] = emp['Stkcd'].astype(int)

tfp = inc_a.merge(bs_a, on=['Stkcd', 'year'], how='inner')
tfp = tfp.merge(emp, on=['Stkcd', 'year'], how='inner')

# Intermediate inputs proxy: COGS (operating cost)
tfp['M'] = tfp['COGS']

# Filter valid
for col in ['Revenue', 'FA', 'Employee', 'M']:
    tfp = tfp[tfp[col] > 0]

# Log transform
tfp['lnY'] = np.log(tfp['Revenue'])
tfp['lnL'] = np.log(tfp['Employee'])
tfp['lnK'] = np.log(tfp['FA'])
tfp['lnM'] = np.log(tfp['M'])

# Add industry
tfp['Stkcd'] = tfp['Stkcd'].astype(int)
tfp = tfp.merge(fi_ind[['Stkcd', 'Ind2']], on='Stkcd', how='left')
tfp = tfp[tfp['Ind2'].notna()].copy()
tfp['ind_year'] = tfp['Ind2'] + '_' + tfp['year'].astype(str)

# Industry-year production function regression
xvars = ['lnL', 'lnK', 'lnM']

def run_prod_func(group):
    if len(group) < 15:
        return pd.Series(np.nan, index=group.index)
    X = sm.add_constant(group[xvars])
    y = group['lnY']
    mask = X.notna().all(axis=1) & y.notna()
    if mask.sum() < 15:
        return pd.Series(np.nan, index=group.index)
    try:
        resid = pd.Series(np.nan, index=group.index)
        resid[mask] = sm.OLS(y[mask], X[mask]).fit().resid
        return resid
    except:
        return pd.Series(np.nan, index=group.index)

tfp['TFP'] = tfp.groupby('ind_year', group_keys=False).apply(run_prod_func)
tfp_out = tfp[['Stkcd', 'year', 'TFP']].dropna()
print(f"  TFP: {len(tfp_out)} obs")

# ════════════════════════════════════════════
# 2. CrashRisk: NCSKEW and DUVOL
# ════════════════════════════════════════════
print("\n[2] CrashRisk (NCSKEW + DUVOL)...")
dr = pd.read_parquet(DATA / 'daily_return.parquet')
dr['Stkcd'] = pd.to_numeric(dr['Stkcd'], errors='coerce')
dr['Trddt'] = pd.to_datetime(dr['Trddt'], errors='coerce')
dr['year'] = dr['Trddt'].dt.year
dr['Dretwd'] = pd.to_numeric(dr['Dretwd'], errors='coerce')

# Market return
mkt = pd.read_parquet(DATA / 'market_index.parquet')
mkt['Trddt'] = pd.to_datetime(mkt['Trddt'], errors='coerce')
mkt['Retindex'] = pd.to_numeric(mkt['Retindex'], errors='coerce')
mkt_sh = mkt[mkt['Indexcd'] == 1][['Trddt', 'Retindex']].rename(columns={'Retindex': 'MktRet'})
mkt_sh = mkt_sh.drop_duplicates(subset=['Trddt'], keep='first')
dr = dr.merge(mkt_sh, on='Trddt', how='left')

# Firm-specific weekly return (use daily, get residuals from market model)
# Then compute NCSKEW and DUVOL from residuals
def calc_crash_risk(group):
    valid = group[['Dretwd', 'MktRet']].dropna()
    n = len(valid)
    result = {'NCSKEW': np.nan, 'DUVOL': np.nan}
    if n < 30:
        return pd.Series(result)

    try:
        # Market model residuals (firm-specific return)
        X = sm.add_constant(valid['MktRet'])
        resid = sm.OLS(valid['Dretwd'], X).fit().resid
        W = resid.values
        n = len(W)

        # NCSKEW = -[n(n-1)^(3/2) * sum(W^3)] / [(n-1)(n-2) * (sum(W^2))^(3/2)]
        sum_w2 = np.sum(W**2)
        sum_w3 = np.sum(W**3)
        if sum_w2 > 0:
            ncskew = -(n * (n-1)**1.5 * sum_w3) / ((n-1) * (n-2) * sum_w2**1.5)
            result['NCSKEW'] = ncskew

        # DUVOL = ln[(n_u - 1) * sum(W_d^2) / ((n_d - 1) * sum(W_u^2))]
        mean_w = np.mean(W)
        W_up = W[W > mean_w]
        W_down = W[W <= mean_w]
        n_u = len(W_up)
        n_d = len(W_down)
        if n_u > 1 and n_d > 1:
            sum_wu2 = np.sum(W_up**2)
            sum_wd2 = np.sum(W_down**2)
            if sum_wu2 > 0:
                duvol = np.log(((n_u - 1) * sum_wd2) / ((n_d - 1) * sum_wu2))
                result['DUVOL'] = duvol
    except:
        pass

    return pd.Series(result)

print("  Computing crash risk (this takes a minute)...")
crash = dr.groupby(['Stkcd', 'year']).apply(calc_crash_risk).reset_index()
print(f"  NCSKEW: {crash['NCSKEW'].notna().sum()} obs")
print(f"  DUVOL: {crash['DUVOL'].notna().sum()} obs")

# ════════════════════════════════════════════
# 3. CashFlowVol (3-year rolling std of CFO/TA)
# ════════════════════════════════════════════
print("\n[3] CashFlowVol...")
cfl = pd.read_parquet(DATA / 'cashflow.parquet')
cfl['Stkcd'] = pd.to_numeric(cfl['Stkcd'], errors='coerce')
cfl['Accper'] = pd.to_datetime(cfl['Accper'], errors='coerce')
cfl_a = cfl[(cfl['Accper'].dt.month == 12) & (cfl['Typrep'] == 'A')].copy()
cfl_a['year'] = cfl_a['Accper'].dt.year
cfl_a = cfl_a.drop_duplicates(subset=['Stkcd', 'year'], keep='last')

bs_full = pd.read_parquet(DATA / 'balance_sheet.parquet')
bs_full['Stkcd'] = pd.to_numeric(bs_full['Stkcd'], errors='coerce')
bs_full['Accper'] = pd.to_datetime(bs_full['Accper'], errors='coerce')
bs_full_a = bs_full[(bs_full['Accper'].dt.month == 12) & (bs_full['Typrep'] == 'A')].copy()
bs_full_a['year'] = bs_full_a['Accper'].dt.year
bs_full_a = bs_full_a.drop_duplicates(subset=['Stkcd', 'year'], keep='last')

cfv = cfl_a[['Stkcd', 'year', 'C001000000']].rename(columns={'C001000000': 'CFO'})
cfv = cfv.merge(bs_full_a[['Stkcd', 'year', 'A001000000']].rename(columns={'A001000000': 'TA'}),
                on=['Stkcd', 'year'], how='inner')
cfv = cfv[(cfv['TA'] > 0) & cfv['CFO'].notna()].copy()
cfv['CFO_TA'] = cfv['CFO'] / cfv['TA']
cfv = cfv.sort_values(['Stkcd', 'year'])

# 3-year rolling std (t-2, t-1, t)
cfv['CashFlowVol'] = cfv.groupby('Stkcd')['CFO_TA'].transform(
    lambda x: x.rolling(window=3, min_periods=2).std()
)
cfv_out = cfv[['Stkcd', 'year', 'CashFlowVol']].dropna()
print(f"  CashFlowVol: {len(cfv_out)} obs")

# ════════════════════════════════════════════
# MERGE ALL TO BASE
# ════════════════════════════════════════════
print("\n[4] Merging...")
result = base.copy()

for name, df in [('TFP', tfp_out), ('CashFlowVol', cfv_out)]:
    df = df.copy()
    df['Stkcd'] = df['Stkcd'].astype(int)
    df['year'] = df['year'].astype(int)
    result = result.merge(df, on=['Stkcd', 'year'], how='left')
    n = result[name].notna().sum()
    print(f"  {name:20s}: {n:6d} ({n/len(result)*100:.1f}%)")

# Crash risk
crash['Stkcd'] = crash['Stkcd'].astype(int)
crash['year'] = crash['year'].astype(int)
result = result.merge(crash[['Stkcd', 'year', 'NCSKEW', 'DUVOL']], on=['Stkcd', 'year'], how='left')
for c in ['NCSKEW', 'DUVOL']:
    n = result[c].notna().sum()
    print(f"  {c:20s}: {n:6d} ({n/len(result)*100:.1f}%)")

# Winsorize
print("\nWinsorizing...")
for v in ['TFP', 'CashFlowVol', 'NCSKEW', 'DUVOL']:
    if result[v].notna().sum() > 100:
        result[v] = winsorize(result[v])

# Save
out = STATA / 'reg_sample_v18.dta'
print(f"\nSaving to {out}...")
for col in result.select_dtypes(include=['object']).columns:
    result[col] = result[col].astype(str)
result.to_stata(out, write_index=False, version=118)
print(f"  Saved: {len(result)} obs, {result.columns.shape[0]} cols")
print(f"  New: TFP, NCSKEW, DUVOL, CashFlowVol")
