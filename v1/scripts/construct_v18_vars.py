"""
Construct all mechanism + heterogeneity variables for v18.
Export as reg_sample_v18.dta.
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

# ── Load base sample ──
print("Loading base sample...")
base = pd.read_stata(STATA / 'reg_sample_iv_v16.dta')
base['Stkcd'] = base['Stkcd'].astype(int)
base['year'] = base['year'].astype(int)
print(f"  Base: {len(base)} obs")

# ── Industry codes (from firm_info, filter NaN first) ──
fi = pd.read_parquet(DATA / 'firm_info.parquet')
fi['Stkcd'] = pd.to_numeric(fi['Symbol'], errors='coerce')
fi_ind = fi[fi['IndustryCodeC'].notna()][['Stkcd', 'IndustryCodeC']].drop_duplicates(
    subset=['Stkcd'], keep='last')
fi_ind['Stkcd'] = fi_ind['Stkcd'].astype(int)
fi_ind['Ind2'] = fi_ind['IndustryCodeC'].str[:3]
print(f"  Industry codes: {len(fi_ind)} firms")

# ── Helper: load annual financial statements ──
def load_annual(name, cols):
    df = pd.read_parquet(DATA / f'{name}.parquet')
    df['Stkcd'] = pd.to_numeric(df['Stkcd'], errors='coerce')
    df['Accper'] = pd.to_datetime(df['Accper'], errors='coerce')
    df = df[(df['Accper'].dt.month == 12) & (df['Typrep'] == 'A')].copy()
    df['year'] = df['Accper'].dt.year
    df = df.drop_duplicates(subset=['Stkcd', 'year'], keep='last')
    return df[['Stkcd', 'year'] + [c for c in cols if c in df.columns]]

bs_a = load_annual('balance_sheet', ['A001000000', 'A001107000', 'A001202000'])
inc_a = load_annual('income_stmt', ['B001101000', 'B002000000'])
cfl_a = load_annual('cashflow', ['C001000000', 'C002000000'])

# ════════════════════════════════════════════
# 1. ForecastDisp + FcstAcc + ReportFreq
# ════════════════════════════════════════════
print("\n[1] Analyst forecast variables...")
af = pd.read_parquet(DATA / 'analyst_forecast.parquet')
af['Stkcd'] = pd.to_numeric(af['Stkcd'], errors='coerce')
af['Fenddt'] = pd.to_datetime(af['Fenddt'], errors='coerce')
af['fyear'] = af['Fenddt'].dt.year
af['Rptdt'] = pd.to_datetime(af['Rptdt'], errors='coerce')
af['rpt_year'] = af['Rptdt'].dt.year
af_sy = af[(af['rpt_year'] == af['fyear']) & af['Feps'].notna()].copy()

# ForecastDisp
disp = af_sy.groupby(['Stkcd', 'fyear'])['Feps'].agg(['std', 'count']).reset_index()
disp.columns = ['Stkcd', 'year', 'ForecastDisp', 'n_fcst']
disp.loc[disp['n_fcst'] < 3, 'ForecastDisp'] = np.nan
disp = disp[['Stkcd', 'year', 'ForecastDisp']]

# FcstAcc
ps = pd.read_parquet(DATA / 'per_share.parquet')
ps['Stkcd'] = pd.to_numeric(ps['Stkcd'], errors='coerce')
if 'Accper' in ps.columns:
    ps['Accper'] = pd.to_datetime(ps['Accper'], errors='coerce')
    ps_a = ps[ps['Accper'].dt.month == 12].copy()
    ps_a['year'] = ps_a['Accper'].dt.year
else:
    ps_a = ps.copy()

eps_col = 'F090101B' if 'F090101B' in ps_a.columns else None
if eps_col is None:
    for c in ps_a.columns:
        if 'eps' in c.lower():
            eps_col = c
            break

if eps_col:
    ps_a = ps_a[['Stkcd', 'year', eps_col]].rename(columns={eps_col: 'ActualEPS'})
    ps_a = ps_a.drop_duplicates(subset=['Stkcd', 'year'], keep='last')
    af_acc = af_sy.merge(ps_a, left_on=['Stkcd', 'fyear'], right_on=['Stkcd', 'year'], how='inner')
    af_acc['FE_norm'] = (af_acc['Feps'] - af_acc['ActualEPS']).abs() / np.maximum(af_acc['ActualEPS'].abs(), 0.01)
    fcst_acc = af_acc.groupby(['Stkcd', 'fyear']).agg(
        FcstAcc=('FE_norm', 'mean'), n_acc=('FE_norm', 'count')
    ).reset_index().rename(columns={'fyear': 'year'})
    fcst_acc.loc[fcst_acc['n_acc'] < 2, 'FcstAcc'] = np.nan
    fcst_acc = fcst_acc[['Stkcd', 'year', 'FcstAcc']]
else:
    fcst_acc = pd.DataFrame(columns=['Stkcd', 'year', 'FcstAcc'])

# ReportFreq
rpt_count = af.groupby(['Stkcd', 'rpt_year']).size().reset_index(name='n_reports')
rpt_count.columns = ['Stkcd', 'year', 'n_reports']
rpt_count['ReportFreq'] = np.log1p(rpt_count['n_reports'])
rpt_count = rpt_count[['Stkcd', 'year', 'ReportFreq']]

print(f"  ForecastDisp: {disp['ForecastDisp'].notna().sum()}")
print(f"  FcstAcc: {fcst_acc['FcstAcc'].notna().sum() if len(fcst_acc) else 0}")
print(f"  ReportFreq: {len(rpt_count)}")

# ════════════════════════════════════════════
# 2. RatingDisp
# ════════════════════════════════════════════
print("\n[2] Rating dispersion...")
ar = pd.read_parquet(DATA / 'analyst_rating.parquet')
ar['Stkcd'] = pd.to_numeric(ar['Stkcd'], errors='coerce')
ar['Rptdt'] = pd.to_datetime(ar['Rptdt'], errors='coerce')
ar['rpt_year'] = ar['Rptdt'].dt.year
ar['rank_num'] = pd.to_numeric(ar.get('Stdrank', pd.Series(dtype=float)), errors='coerce')
if ar['rank_num'].notna().sum() < 1000 and 'Investrank' in ar.columns:
    rank_map = {'买入': 5, '增持': 4, '推荐': 4, '中性': 3, '持有': 3,
                '观望': 3, '减持': 2, '回避': 2, '卖出': 1}
    mapped = ar['Investrank'].map(rank_map)
    ar['rank_num'] = ar['rank_num'].fillna(mapped)

rating_disp = ar[ar['rank_num'].notna()].groupby(['Stkcd', 'rpt_year']).agg(
    RatingDisp=('rank_num', 'std'), n_rating=('rank_num', 'count')
).reset_index().rename(columns={'rpt_year': 'year'})
rating_disp.loc[rating_disp['n_rating'] < 3, 'RatingDisp'] = np.nan
rating_disp = rating_disp[['Stkcd', 'year', 'RatingDisp']]
print(f"  RatingDisp: {rating_disp['RatingDisp'].notna().sum()}")

# ════════════════════════════════════════════
# 3. absDA (Kothari 2005)
# ════════════════════════════════════════════
print("\n[3] absDA...")
acc = bs_a.rename(columns={'A001000000': 'TA', 'A001107000': 'Recv', 'A001202000': 'FA'})
acc = acc.merge(inc_a.rename(columns={'B001101000': 'Rev', 'B002000000': 'NI'}),
                on=['Stkcd', 'year'], how='inner')
acc = acc.merge(cfl_a.rename(columns={'C001000000': 'CFO'})[['Stkcd', 'year', 'CFO']],
                on=['Stkcd', 'year'], how='inner')
acc['Recv'] = acc['Recv'].fillna(0)
acc = acc.sort_values(['Stkcd', 'year'])
for col in ['TA', 'Rev', 'Recv']:
    acc[f'lag_{col}'] = acc.groupby('Stkcd')[col].shift(1)
acc = acc.dropna(subset=['lag_TA', 'NI', 'CFO', 'Rev', 'FA'])
acc = acc[acc['lag_TA'] > 0].copy()

acc['TA_scaled'] = (acc['NI'] - acc['CFO']) / acc['lag_TA']
acc['inv_lagTA'] = 1.0 / acc['lag_TA']
acc['dRev_adj'] = ((acc['Rev'] - acc['lag_Rev']) - (acc['Recv'] - acc['lag_Recv'])) / acc['lag_TA']
acc['PPE_scaled'] = acc['FA'] / acc['lag_TA']
acc['ROA_r'] = acc['NI'] / acc['lag_TA']

# Use fi_ind (filtered for valid IndustryCodeC)
acc['Stkcd'] = acc['Stkcd'].astype(int)
acc = acc.merge(fi_ind[['Stkcd', 'Ind2']], on='Stkcd', how='left')
acc = acc[acc['Ind2'].notna()].copy()
acc['ind_year'] = acc['Ind2'] + '_' + acc['year'].astype(str)
print(f"  Accruals panel: {len(acc)} obs, {acc['ind_year'].nunique()} groups")
gs = acc.groupby('ind_year').size()
print(f"  Groups >= 10: {(gs >= 10).sum()}, median size: {gs.median():.0f}")

xvars = ['inv_lagTA', 'dRev_adj', 'PPE_scaled', 'ROA_r']
def run_jones(group):
    if len(group) < 10:
        return pd.Series(np.nan, index=group.index)
    X = sm.add_constant(group[xvars])
    y = group['TA_scaled']
    mask = X.notna().all(axis=1) & y.notna()
    if mask.sum() < 10:
        return pd.Series(np.nan, index=group.index)
    try:
        resid = pd.Series(np.nan, index=group.index)
        resid[mask] = sm.OLS(y[mask], X[mask]).fit().resid.abs()
        return resid
    except:
        return pd.Series(np.nan, index=group.index)

acc['absDA'] = acc.groupby('ind_year', group_keys=False).apply(run_jones)
absda = acc[['Stkcd', 'year', 'absDA']].dropna()
print(f"  absDA: {len(absda)} obs")

# ════════════════════════════════════════════
# 4. InvestIneff (Richardson 2006)
# ════════════════════════════════════════════
print("\n[4] InvestIneff...")
inv = bs_a.rename(columns={'A001000000': 'TA'}).copy()
inv = inv.merge(inc_a.rename(columns={'B001101000': 'Rev', 'B002000000': 'NI'}),
                on=['Stkcd', 'year'], how='inner')
inv = inv.merge(cfl_a.rename(columns={'C001000000': 'CFO', 'C002000000': 'CFI'}),
                on=['Stkcd', 'year'], how='inner')
inv = inv.sort_values(['Stkcd', 'year'])
inv['lagTA'] = inv.groupby('Stkcd')['TA'].shift(1)
inv = inv[inv['lagTA'] > 0].copy()

inv['Invest'] = -inv['CFI'].fillna(0) / inv['lagTA']
inv['lagRev'] = inv.groupby('Stkcd')['Rev'].shift(1)
inv['Growth_r'] = (inv['Rev'] / inv['lagRev']) - 1
inv['ROA_r'] = inv['NI'] / inv['lagTA']
inv['Cash_r'] = inv['CFO'].fillna(0) / inv['TA']
inv['Size_r'] = np.log(inv['TA'])

for col in ['Invest', 'Growth_r', 'ROA_r', 'Cash_r', 'Size_r']:
    inv[f'lag_{col}'] = inv.groupby('Stkcd')[col].shift(1)

inv['Stkcd'] = inv['Stkcd'].astype(int)
inv = inv.merge(fi_ind[['Stkcd', 'Ind2']], on='Stkcd', how='left')
inv = inv[inv['Ind2'].notna()].copy()
inv['ind_year'] = inv['Ind2'] + '_' + inv['year'].astype(str)

reg_cols = ['lag_Invest', 'lag_Growth_r', 'lag_ROA_r', 'lag_Cash_r', 'lag_Size_r']
inv_clean = inv.dropna(subset=['Invest'] + reg_cols)
print(f"  Investment panel: {len(inv_clean)} obs, {inv_clean['ind_year'].nunique()} groups")

def run_richardson(group):
    if len(group) < 15:
        return pd.Series(np.nan, index=group.index)
    X = sm.add_constant(group[reg_cols])
    y = group['Invest']
    try:
        resid = pd.Series(np.nan, index=group.index)
        resid.loc[group.index] = sm.OLS(y, X).fit().resid.abs()
        return resid
    except:
        return pd.Series(np.nan, index=group.index)

inv_clean['InvestIneff'] = inv_clean.groupby('ind_year', group_keys=False).apply(run_richardson)
investineff = inv_clean[['Stkcd', 'year', 'InvestIneff']].dropna()
print(f"  InvestIneff: {len(investineff)} obs")

# ════════════════════════════════════════════
# 5. AuditFee
# ════════════════════════════════════════════
print("\n[5] AuditFee...")
aud = pd.read_parquet(DATA / 'audit.parquet')
aud['Stkcd'] = pd.to_numeric(aud['Stkcd'], errors='coerce')
aud['Accper'] = pd.to_datetime(aud['Accper'], errors='coerce')
aud = aud[aud['Accper'].dt.month == 12].copy()
aud['year'] = aud['Accper'].dt.year
aud['fee'] = pd.to_numeric(aud['Tcost'], errors='coerce').fillna(
    pd.to_numeric(aud['Dcost'], errors='coerce'))
aud = aud[aud['fee'] > 0].copy()
aud['AuditFee'] = np.log(aud['fee'])
aud = aud.drop_duplicates(subset=['Stkcd', 'year'], keep='last')
auditfee = aud[['Stkcd', 'year', 'AuditFee']]
print(f"  AuditFee: {len(auditfee)} obs")

# ════════════════════════════════════════════
# 6. Daily return vars (RetAutoCorr, IdioVol, TurnoverVol)
# ════════════════════════════════════════════
print("\n[6] Daily return variables...")
dr = pd.read_parquet(DATA / 'daily_return.parquet')
dr['Stkcd'] = pd.to_numeric(dr['Stkcd'], errors='coerce')
dr['Trddt'] = pd.to_datetime(dr['Trddt'], errors='coerce')
dr['year'] = dr['Trddt'].dt.year
dr['Dretwd'] = pd.to_numeric(dr['Dretwd'], errors='coerce')

# Market return: use Shanghai Composite (Indexcd=1), column Retindex
mkt = pd.read_parquet(DATA / 'market_index.parquet')
mkt['Trddt'] = pd.to_datetime(mkt['Trddt'], errors='coerce')
mkt['Retindex'] = pd.to_numeric(mkt['Retindex'], errors='coerce')
mkt_sh = mkt[mkt['Indexcd'] == 1][['Trddt', 'Retindex']].rename(columns={'Retindex': 'MktRet'})
mkt_sh = mkt_sh.drop_duplicates(subset=['Trddt'], keep='first')
print(f"  Market return obs: {mkt_sh['MktRet'].notna().sum()}")

dr = dr.merge(mkt_sh, on='Trddt', how='left')

# Vectorized aggregation for speed
print("  Computing daily aggregates...")

def calc_daily_vars(group):
    ret = group['Dretwd'].dropna()
    n = len(ret)
    result = {}
    # RetAutoCorr
    result['RetAutoCorr'] = ret.autocorr(lag=1) if n >= 60 else np.nan
    # IdioVol
    valid = group[['Dretwd', 'MktRet']].dropna()
    if len(valid) >= 60:
        try:
            X = sm.add_constant(valid['MktRet'])
            result['IdioVol'] = np.std(sm.OLS(valid['Dretwd'], X).fit().resid, ddof=1)
        except:
            result['IdioVol'] = np.nan
    else:
        result['IdioVol'] = np.nan
    # TurnoverVol
    if 'Dnvaltrd' in group.columns and 'Dsmvosd' in group.columns:
        tv = pd.to_numeric(group['Dnvaltrd'], errors='coerce') / pd.to_numeric(group['Dsmvosd'], errors='coerce')
        tv = tv.replace([np.inf, -np.inf], np.nan).dropna()
        result['TurnoverVol'] = tv.std() if len(tv) >= 60 else np.nan
    else:
        result['TurnoverVol'] = np.nan
    return pd.Series(result)

daily_vars = dr.groupby(['Stkcd', 'year']).apply(calc_daily_vars).reset_index()
for c in ['RetAutoCorr', 'IdioVol', 'TurnoverVol']:
    print(f"  {c}: {daily_vars[c].notna().sum()}")

# ════════════════════════════════════════════
# 7. InstStable
# ════════════════════════════════════════════
print("\n[7] InstStable...")
ih = pd.read_parquet(DATA / 'inst_holding.parquet')
ih['Stkcd'] = pd.to_numeric(ih['Symbol'], errors='coerce')
ih['EndDate'] = pd.to_datetime(ih['EndDate'], errors='coerce')
ih_annual = ih[ih['EndDate'].dt.month == 12].copy()
ih_annual['year'] = ih_annual['EndDate'].dt.year
for col in ['FundHoldProportion', 'InsuranceHoldProportion', 'QFIIHoldProportion']:
    ih_annual[col] = pd.to_numeric(ih_annual[col], errors='coerce').fillna(0)
ih_annual['InstStable'] = (ih_annual['FundHoldProportion'] +
                           ih_annual['InsuranceHoldProportion'] +
                           ih_annual['QFIIHoldProportion'])
ih_annual = ih_annual.drop_duplicates(subset=['Stkcd', 'year'], keep='last')
inststable = ih_annual[['Stkcd', 'year', 'InstStable']]
print(f"  InstStable: {len(inststable)} obs")

# ════════════════════════════════════════════
# 8. SA index
# ════════════════════════════════════════════
print("\n[8] SA index...")
sa = base[['Stkcd', 'year', 'Size', 'Age']].copy()
sa['SA'] = -0.737 * sa['Size'] + 0.043 * sa['Size']**2 - 0.040 * sa['Age']
sa = sa[['Stkcd', 'year', 'SA']]
print(f"  SA: {sa['SA'].notna().sum()}")

# ════════════════════════════════════════════
# 9. HighTech
# ════════════════════════════════════════════
print("\n[9] HighTech...")
hitech_prefixes = ['C25', 'C26', 'C27', 'C28', 'C32', 'C34', 'C35', 'C36',
                   'C37', 'C38', 'C39', 'C40', 'C41',
                   'I63', 'I64', 'I65', 'M73', 'M74', 'M75']
# Use base sample's own Ind2 (already in data)
base_ind2 = base[['Stkcd', 'Ind2']].drop_duplicates(subset=['Stkcd'], keep='last')
base_ind2['HighTech'] = base_ind2['Ind2'].isin(hitech_prefixes).astype(int)
hitech = base_ind2[['Stkcd', 'HighTech']]
print(f"  HighTech=1: {hitech['HighTech'].sum()} firms")

# ════════════════════════════════════════════
# MERGE ALL
# ════════════════════════════════════════════
print("\n[10] Merging...")
result = base.copy()
result['Post2020'] = (result['year'] >= 2020).astype(int)

merge_list = [
    ('ForecastDisp', disp), ('FcstAcc', fcst_acc), ('ReportFreq', rpt_count),
    ('RatingDisp', rating_disp), ('absDA', absda), ('InvestIneff', investineff),
    ('AuditFee', auditfee), ('InstStable', inststable), ('SA', sa),
]
for name, df in merge_list:
    df = df.copy()
    df['Stkcd'] = df['Stkcd'].astype(int)
    df['year'] = df['year'].astype(int)
    new_cols = [c for c in df.columns if c not in ['Stkcd', 'year']]
    result = result.merge(df, on=['Stkcd', 'year'], how='left')
    for c in new_cols:
        n = result[c].notna().sum()
        print(f"  {c:20s}: {n:6d} ({n/len(result)*100:.1f}%)")

# Daily vars
daily_vars['Stkcd'] = daily_vars['Stkcd'].astype(int)
daily_vars['year'] = daily_vars['year'].astype(int)
result = result.merge(daily_vars[['Stkcd', 'year', 'RetAutoCorr', 'IdioVol', 'TurnoverVol']],
                      on=['Stkcd', 'year'], how='left')
for c in ['RetAutoCorr', 'IdioVol', 'TurnoverVol']:
    n = result[c].notna().sum()
    print(f"  {c:20s}: {n:6d} ({n/len(result)*100:.1f}%)")

# HighTech
hitech['Stkcd'] = hitech['Stkcd'].astype(int)
result = result.merge(hitech, on='Stkcd', how='left')
result['HighTech'] = result['HighTech'].fillna(0).astype(int)
print(f"  {'HighTech':20s}: {result['HighTech'].sum():6d} obs=1")

# ── Winsorize ──
print("\nWinsorizing...")
win_vars = ['ForecastDisp', 'FcstAcc', 'absDA', 'InvestIneff', 'AuditFee',
            'RetAutoCorr', 'IdioVol', 'TurnoverVol', 'InstStable',
            'RatingDisp', 'ReportFreq', 'SA']
for v in win_vars:
    if v in result.columns and result[v].notna().sum() > 100:
        result[v] = winsorize(result[v])

# ── Export ──
out = STATA / 'reg_sample_v18.dta'
print(f"\nExporting to {out}...")
for col in result.select_dtypes(include=['object']).columns:
    result[col] = result[col].astype(str)
result.to_stata(out, write_index=False, version=118)

# ── Summary ──
print(f"\n{'='*60}")
print(f"SAVED: {len(result)} obs, {len(result.columns)} cols")
print(f"{'='*60}")

mech = ['Analyst', 'Amihud', 'RetVol', 'InstHold', 'Turnover',
        'ForecastDisp', 'FcstAcc', 'absDA', 'InvestIneff', 'AuditFee',
        'RetAutoCorr', 'IdioVol', 'TurnoverVol', 'InstStable', 'RatingDisp', 'ReportFreq']
het = ['SOE', 'Size', 'Analyst', 'ShareholderNum', 'Hhi', 'InstHold',
       'ProvDigital2016', 'Market', 'MainBoard', 'Big4', 'BM', 'Intangible',
       'SA', 'HighTech', 'Post2020']

print("\nMechanism vars:")
for v in mech:
    if v in result.columns:
        n = result[v].notna().sum()
        print(f"  {v:20s}: {n:6d} ({n/len(result)*100:.1f}%)")

print("\nHeterogeneity vars:")
for v in het:
    if v in result.columns:
        n = result[v].notna().sum()
        print(f"  {v:20s}: {n:6d} ({n/len(result)*100:.1f}%)")
