"""
v14 机制回归: 4个渠道变量 x (同期+滞后) = 8个回归, 27个控制变量
渠道: absDA, AuditFee, RetAutoCorr, InvestIneff
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import pyfixest as pf
import json, os, warnings
warnings.filterwarnings('ignore')

BASE = "/Users/mac/computerscience/15会计研究"

# ============================================================
# 0. 加载基础面板 (与 generate_tables_v13.py 一致)
# ============================================================
print("=" * 70)
print("0. 加载基础面板 (panel_dml + annual_report_features)")
print("=" * 70)

panel = pd.read_parquet(f"{BASE}/data_parquet/panel_dml.parquet")
ar_feat = pd.read_parquet(f"{BASE}/data_parquet/annual_report_features.parquet")

panel = panel.merge(
    ar_feat[['Stkcd', 'year', 'kw_total', 'kw_per10k', 'substantive_count',
             'kw_data_stock', 'kw_data_dev', 'kw_data_app', 'kw_data_value', 'kw_data_gov']],
    on=['Stkcd', 'year'], how='left'
)

fi = pd.read_parquet(f"{BASE}/data_parquet/firm_info.parquet",
                     columns=['Symbol', 'EndDate', 'IndustryCodeC', 'LISTINGSTATE'])
fi = fi.rename(columns={'Symbol': 'Stkcd'})
fi['EndDate'] = pd.to_datetime(fi['EndDate'])
fi['year'] = fi['EndDate'].dt.year
fi = fi.sort_values(['Stkcd', 'year', 'EndDate']).drop_duplicates(
    subset=['Stkcd', 'year'], keep='last')

ind_mode = fi.groupby('Stkcd')['IndustryCodeC'].agg(
    lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else np.nan)
panel = panel.merge(ind_mode.rename('IndCode_mode').reset_index(), on='Stkcd', how='left')
panel = panel.merge(fi[['Stkcd', 'year', 'IndustryCodeC', 'LISTINGSTATE']],
                    on=['Stkcd', 'year'], how='left')
panel['IndustryCodeC'] = panel['IndustryCodeC'].fillna(panel['IndCode_mode'])

# 样本筛选
mask_fin = panel['IndustryCodeC'].str.startswith('J', na=False)
mask_st = panel['LISTINGSTATE'].isin(['ST', '*ST'])
mask_new = panel['Age'] <= 0
panel = panel[~mask_fin & ~mask_st & ~mask_new].copy()

panel['Ind2'] = panel['IndustryCodeC'].str[:3]
panel['IndYear'] = panel['Ind2'].astype(str) + '_' + panel['year'].astype(str)
panel['DU_kw'] = panel['kw_per10k']

print(f"基础面板: {len(panel):,} obs, {panel['Stkcd'].nunique():,} firms")

# ============================================================
# 1. 构造 absDA (Kothari 2005)
# ============================================================
print("\n" + "=" * 70)
print("1. 构造 absDA (Kothari 2005 业绩调整Jones模型)")
print("=" * 70)

def annual_report_filter(df):
    df = df.copy()
    df['Accper'] = df['Accper'].astype(str)
    mask = df['Accper'].str.endswith('12-31') & (df['Typrep'] == 'A')
    df = df[mask].copy()
    df['year'] = pd.to_datetime(df['Accper']).dt.year
    df['Stkcd'] = df['Stkcd'].astype(int)
    return df.drop_duplicates(['Stkcd', 'year'], keep='last')

bs = annual_report_filter(pd.read_parquet(f"{BASE}/data_parquet/balance_sheet.parquet"))
inc = annual_report_filter(pd.read_parquet(f"{BASE}/data_parquet/income_stmt.parquet"))
cfl = annual_report_filter(pd.read_parquet(f"{BASE}/data_parquet/cashflow.parquet"))

fin = bs[['Stkcd', 'year', 'A001000000', 'A001107000', 'A001202000']].merge(
    inc[['Stkcd', 'year', 'B001101000', 'B002000000']], on=['Stkcd', 'year'], how='inner'
).merge(
    cfl[['Stkcd', 'year', 'C001000000']], on=['Stkcd', 'year'], how='inner'
)

fin = fin.rename(columns={
    'A001000000': 'TotalAssets', 'A001107000': 'Receivables',
    'A001202000': 'FixedAssets',
    'B001101000': 'Revenue_j', 'B002000000': 'NetIncome_j',
    'C001000000': 'CFO_j',
})
for c in fin.columns[2:]:
    fin[c] = pd.to_numeric(fin[c], errors='coerce')
fin['Receivables'] = fin['Receivables'].fillna(0)
fin = fin.sort_values(['Stkcd', 'year'])

fin['lagTA_j'] = fin.groupby('Stkcd')['TotalAssets'].shift(1)
fin['dRev'] = fin['Revenue_j'] - fin.groupby('Stkcd')['Revenue_j'].shift(1)
fin['dRec'] = fin['Receivables'] - fin.groupby('Stkcd')['Receivables'].shift(1)

# TA = NI - CFO (total accruals)
fin['TA_scaled'] = (fin['NetIncome_j'] - fin['CFO_j']) / fin['lagTA_j']
fin['inv_lagTA'] = 1.0 / fin['lagTA_j']
fin['dRev_adj'] = (fin['dRev'] - fin['dRec']) / fin['lagTA_j']
fin['PPE_scaled'] = fin['FixedAssets'] / fin['lagTA_j']
fin['ROA_j'] = fin['NetIncome_j'] / fin['lagTA_j']

# Add industry
fin = fin.merge(fi[['Stkcd', 'year', 'IndustryCodeC']], on=['Stkcd', 'year'], how='left')
fin_ind_mode = fi.groupby('Stkcd')['IndustryCodeC'].agg(
    lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else np.nan)
fin = fin.merge(fin_ind_mode.rename('Ind2_mode').reset_index(), on='Stkcd', how='left')
fin['IndustryCodeC'] = fin['IndustryCodeC'].fillna(fin['Ind2_mode'])
fin['Ind2_j'] = fin['IndustryCodeC'].str[:3]
fin = fin[~fin['Ind2_j'].str.startswith('J', na=False)]

# Kothari model: TA/A = a(1/A) + b1(dRev-dRec)/A + b2(PPE/A) + b3(ROA) + e
jones_vars = ['TA_scaled', 'inv_lagTA', 'dRev_adj', 'PPE_scaled', 'ROA_j']
fin_reg = fin.dropna(subset=jones_vars + ['Ind2_j']).copy()
for v in jones_vars:
    lo, hi = fin_reg[v].quantile([0.01, 0.99])
    fin_reg[v] = fin_reg[v].clip(lo, hi)

da_results = []
for (ind, yr), grp in fin_reg.groupby(['Ind2_j', 'year']):
    if len(grp) < 10:
        continue
    y = grp['TA_scaled'].values
    X = grp[['inv_lagTA', 'dRev_adj', 'PPE_scaled', 'ROA_j']].values
    X = sm.add_constant(X)
    try:
        model = sm.OLS(y, X).fit()
        for idx, r in zip(grp.index, model.resid):
            da_results.append({
                'Stkcd': grp.loc[idx, 'Stkcd'],
                'year': grp.loc[idx, 'year'],
                'absDA': abs(r)
            })
    except Exception:
        pass

da_df = pd.DataFrame(da_results)
panel = panel.merge(da_df, on=['Stkcd', 'year'], how='left')
print(f"  absDA: {panel['absDA'].notna().sum():,} obs, mean={panel['absDA'].mean():.4f}")


# ============================================================
# 2. 构造 AuditFee (审计费用对数)
# ============================================================
print("\n" + "=" * 70)
print("2. 构造 AuditFee (ln审计总费用)")
print("=" * 70)

audit = pd.read_parquet(f"{BASE}/data_parquet/audit.parquet")
audit['Accper'] = audit['Accper'].astype(str)
audit_annual = audit[audit['Accper'].str.endswith('12-31')].copy()
audit_annual['year'] = pd.to_datetime(audit_annual['Accper']).dt.year
audit_annual['Stkcd'] = audit_annual['Stkcd'].astype(int)
audit_annual['Tcost'] = pd.to_numeric(audit_annual['Tcost'], errors='coerce')
audit_annual['Dcost'] = pd.to_numeric(audit_annual['Dcost'], errors='coerce')
audit_annual['fee'] = audit_annual['Tcost'].fillna(audit_annual['Dcost'])
audit_annual = audit_annual.dropna(subset=['fee'])
audit_annual = audit_annual[audit_annual['fee'] > 0]
audit_annual['AuditFee'] = np.log(audit_annual['fee'])
audit_annual = audit_annual.drop_duplicates(['Stkcd', 'year'], keep='last')

panel = panel.merge(audit_annual[['Stkcd', 'year', 'AuditFee']],
                    on=['Stkcd', 'year'], how='left')
print(f"  AuditFee: {panel['AuditFee'].notna().sum():,} obs, mean={panel['AuditFee'].mean():.2f}")


# ============================================================
# 3. 构造 RetAutoCorr (收益率一阶自相关)
# ============================================================
print("\n" + "=" * 70)
print("3. 构造 RetAutoCorr (日收益一阶自相关)")
print("=" * 70)

daily = pd.read_parquet(f"{BASE}/data_parquet/daily_return.parquet")
daily['Trddt'] = pd.to_datetime(daily['Trddt'])
daily['year'] = daily['Trddt'].dt.year

def calc_autocorr(g):
    ret = g['Dretwd'].dropna()
    if len(ret) < 60:
        return np.nan
    return ret.autocorr(lag=1)

autocorr = daily.groupby(['Stkcd', 'year']).apply(calc_autocorr).reset_index()
autocorr.columns = ['Stkcd', 'year', 'RetAutoCorr']

panel = panel.merge(autocorr, on=['Stkcd', 'year'], how='left')
print(f"  RetAutoCorr: {panel['RetAutoCorr'].notna().sum():,} obs, "
      f"mean={panel['RetAutoCorr'].mean():.4f}")

del daily  # free memory


# ============================================================
# 4. 构造 InvestIneff (Richardson 2006)
# ============================================================
print("\n" + "=" * 70)
print("4. 构造 InvestIneff (Richardson 2006 投资偏离)")
print("=" * 70)

inv_data = bs[['Stkcd', 'year', 'A001000000', 'A001100000', 'A002000000',
               'A001202000']].merge(
    inc[['Stkcd', 'year', 'B001101000', 'B002000000']], on=['Stkcd', 'year'], how='inner'
).merge(
    cfl[['Stkcd', 'year', 'C001000000', 'C002000000']], on=['Stkcd', 'year'], how='inner'
)

inv_data = inv_data.rename(columns={
    'A001000000': 'TA_i', 'A001100000': 'CA_i', 'A002000000': 'TL_i',
    'A001202000': 'FixedAssets_i',
    'B001101000': 'Revenue_i', 'B002000000': 'NI_i',
    'C001000000': 'CFO_i', 'C002000000': 'CFI_i',
})

for c in inv_data.columns[2:]:
    inv_data[c] = pd.to_numeric(inv_data[c], errors='coerce')

inv_data = inv_data.sort_values(['Stkcd', 'year'])
inv_data['lagTA_i'] = inv_data.groupby('Stkcd')['TA_i'].shift(1)

# Invest = -CFI / lagTA (CFI usually negative for investment)
inv_data['Invest'] = -inv_data['CFI_i'] / inv_data['lagTA_i']

# Richardson model controls
inv_data['Lev_r'] = inv_data['TL_i'] / inv_data['TA_i']
inv_data['Cash_r'] = inv_data['CA_i'] / inv_data['TA_i']
inv_data['Size_r'] = np.log(inv_data['TA_i'].clip(lower=1))
inv_data['Growth_r'] = inv_data['Revenue_i'] / inv_data.groupby('Stkcd')['Revenue_i'].shift(1) - 1
inv_data['ROA_r'] = inv_data['NI_i'] / inv_data['lagTA_i']

# Lagged regressors
for v in ['Invest', 'Lev_r', 'Cash_r', 'Size_r', 'Growth_r', 'ROA_r']:
    inv_data[f'lag_{v}'] = inv_data.groupby('Stkcd')[v].shift(1)

# Add industry
inv_data = inv_data.merge(fi[['Stkcd', 'year', 'IndustryCodeC']], on=['Stkcd', 'year'], how='left')
inv_data_ind_mode = fi.groupby('Stkcd')['IndustryCodeC'].agg(
    lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else np.nan)
inv_data = inv_data.merge(inv_data_ind_mode.rename('Ind2_mode_i').reset_index(), on='Stkcd', how='left')
inv_data['IndustryCodeC'] = inv_data['IndustryCodeC'].fillna(inv_data['Ind2_mode_i'])
inv_data['Ind2_i'] = inv_data['IndustryCodeC'].str[:3]
inv_data = inv_data[~inv_data['Ind2_i'].str.startswith('J', na=False)]

rich_vars = ['Invest', 'lag_Invest', 'lag_Lev_r', 'lag_Cash_r', 'lag_Size_r',
             'lag_Growth_r', 'lag_ROA_r']
inv_reg = inv_data.dropna(subset=rich_vars + ['Ind2_i']).copy()

for v in rich_vars:
    lo, hi = inv_reg[v].quantile([0.01, 0.99])
    inv_reg[v] = inv_reg[v].clip(lo, hi)

invest_results = []
for (ind, yr), grp in inv_reg.groupby(['Ind2_i', 'year']):
    if len(grp) < 15:
        continue
    y = grp['Invest'].values
    X = grp[['lag_Invest', 'lag_Lev_r', 'lag_Cash_r', 'lag_Size_r',
             'lag_Growth_r', 'lag_ROA_r']].values
    X = sm.add_constant(X)
    try:
        model = sm.OLS(y, X).fit()
        for idx, r in zip(grp.index, model.resid):
            invest_results.append({
                'Stkcd': grp.loc[idx, 'Stkcd'],
                'year': grp.loc[idx, 'year'],
                'InvestIneff': abs(r)
            })
    except Exception:
        pass

invest_df = pd.DataFrame(invest_results)
panel = panel.merge(invest_df, on=['Stkcd', 'year'], how='left')
print(f"  InvestIneff: {panel['InvestIneff'].notna().sum():,} obs, "
      f"mean={panel['InvestIneff'].mean():.4f}")


# ============================================================
# 5. Winsorize + 准备回归样本
# ============================================================
print("\n" + "=" * 70)
print("5. Winsorize 并准备回归样本 (27个控制变量)")
print("=" * 70)

def winsorize(s):
    lo, hi = s.quantile([0.01, 0.99])
    return s.clip(lo, hi)

controls_27 = ['Size', 'Lev', 'ROA', 'TobinQ', 'Age', 'Growth', 'BoardSize', 'IndepRatio',
               'Dual', 'Top1Share', 'SOE', 'InstHold', 'Amihud', 'Analyst', 'AuditType',
               'CFO', 'RetVol', 'Turnover', 'Intangible', 'PPE', 'BM',
               'Employee', 'ShareholderNum', 'Manhold', 'Balance', 'Separation', 'Market']

mech_vars = ['absDA', 'AuditFee', 'RetAutoCorr', 'InvestIneff']

# Winsorize continuous variables
cont_vars = ['PriceDelay', 'DU_kw'] + mech_vars + [
    v for v in controls_27 if v not in ['Dual', 'SOE', 'AuditType']]
for v in cont_vars:
    if v in panel.columns and panel[v].notna().any():
        mask = panel[v].notna()
        if mask.sum() > 100:
            panel.loc[mask, v] = winsorize(panel.loc[mask, v])

# Build regression sample
reg = panel.dropna(subset=['PriceDelay', 'DU_kw'] + controls_27).copy()
reg['Stkcd_str'] = reg['Stkcd'].astype(str)
reg['year_str'] = reg['year'].astype(str)
print(f"回归样本: N={len(reg):,}, firms={reg['Stkcd'].nunique():,}")

for v in mech_vars:
    n = reg[v].notna().sum()
    print(f"  {v}: {n:,} obs ({n/len(reg)*100:.1f}%)")


# ============================================================
# 6. 江艇两步法回归 (27控制变量)
# ============================================================
print("\n" + "=" * 70)
print("6. 江艇两步法机制回归 (v14: 27控制变量)")
print("=" * 70)

def sig_stars(p):
    if p is None or pd.isna(p):
        return ""
    if p < 0.01: return "***"
    if p < 0.05: return "**"
    if p < 0.1: return "*"
    return ""

mech_map = {
    'absDA':       ('盈余管理程度', 'Kothari et al. (2005)'),
    'AuditFee':    ('审计费用(对数)', 'DeFond and Zhang (2014)'),
    'RetAutoCorr': ('收益率自相关', 'Chordia et al. (2008)'),
    'InvestIneff': ('投资效率偏离', 'Richardson (2006)'),
}

results_all = {}

for var, (cn_name, source) in mech_map.items():
    print(f"\n--- {var} ({cn_name}) [{source}] ---")

    # For AuditFee, exclude AuditType from controls (overlap)
    if var == 'AuditFee':
        mech_controls = [c for c in controls_27 if c != 'AuditType']
    else:
        mech_controls = controls_27[:]

    mech_ctrl_str = " + ".join(mech_controls)

    # ---- Panel A: Concurrent DU_kw_t -> M_t ----
    sub = reg.dropna(subset=[var]).copy()
    if len(sub) < 500:
        print(f"  同期跳过: 仅{len(sub)}个obs")
        continue

    fml = f"{var} ~ DU_kw + {mech_ctrl_str} | Stkcd_str + year_str"
    try:
        m = pf.feols(fml, data=sub, vcov={"CRV1": "IndYear"})
        coef = float(m.coef()['DU_kw'])
        se = float(m.se()['DU_kw'])
        pval = float(m.pvalue()['DU_kw'])
        stars = sig_stars(pval)
        n = m._N
        r2 = m._r2
        t_stat = coef / se
        print(f"  同期:   coef={coef:.6f}, se={se:.6f}, t={t_stat:.2f}, "
              f"p={pval:.4f}{stars}, N={n:,}, R2={r2:.3f}")
        results_all[f'{var}_concurrent'] = {
            'var': var, 'cn_name': cn_name, 'source': source,
            'type': 'concurrent',
            'coef': round(coef, 6), 'se': round(se, 6), 't': round(t_stat, 2),
            'p': round(pval, 4), 'sig': stars,
            'N': n, 'R2': round(r2, 3),
            'n_controls': len(mech_controls),
        }
    except Exception as e:
        print(f"  同期回归失败: {e}")

    # ---- Panel B: Lagged DU_kw_t -> M_{t+1} ----
    lag_df = reg[['Stkcd', 'year', 'DU_kw'] + mech_controls +
                 ['Stkcd_str', 'year_str', 'IndYear']].copy()
    future_m = panel[['Stkcd', 'year', var]].copy()
    future_m['year'] = future_m['year'] - 1  # shift M_{t+1} to align with X_t
    future_m = future_m.rename(columns={var: f'{var}_lead'})
    lag_df = lag_df.merge(future_m, on=['Stkcd', 'year'], how='inner')
    lag_df = lag_df.dropna(subset=[f'{var}_lead'])

    if len(lag_df) < 500:
        print(f"  滞后跳过: 仅{len(lag_df)}个obs")
        continue

    fml_lag = f"{var}_lead ~ DU_kw + {mech_ctrl_str} | Stkcd_str + year_str"
    try:
        m_lag = pf.feols(fml_lag, data=lag_df, vcov={"CRV1": "IndYear"})
        coef_l = float(m_lag.coef()['DU_kw'])
        se_l = float(m_lag.se()['DU_kw'])
        pval_l = float(m_lag.pvalue()['DU_kw'])
        stars_l = sig_stars(pval_l)
        n_l = m_lag._N
        r2_l = m_lag._r2
        t_l = coef_l / se_l
        print(f"  滞后:   coef={coef_l:.6f}, se={se_l:.6f}, t={t_l:.2f}, "
              f"p={pval_l:.4f}{stars_l}, N={n_l:,}, R2={r2_l:.3f}")
        results_all[f'{var}_lagged'] = {
            'var': var, 'cn_name': cn_name, 'source': source,
            'type': 'lagged',
            'coef': round(coef_l, 6), 'se': round(se_l, 6), 't': round(t_l, 2),
            'p': round(pval_l, 4), 'sig': stars_l,
            'N': n_l, 'R2': round(r2_l, 3),
            'n_controls': len(mech_controls),
        }
    except Exception as e:
        print(f"  滞后回归失败: {e}")


# ============================================================
# 7. 汇总输出
# ============================================================
print("\n" + "=" * 70)
print("7. 汇总 (v14: 27控制变量)")
print("=" * 70)

print(f"\n{'变量':<16} {'类型':<12} {'系数':>12} {'t值':>8} {'显著性':>6} {'N':>8} {'R2':>8}")
print("-" * 78)
for key in ['absDA_concurrent', 'absDA_lagged',
            'AuditFee_concurrent', 'AuditFee_lagged',
            'RetAutoCorr_concurrent', 'RetAutoCorr_lagged',
            'InvestIneff_concurrent', 'InvestIneff_lagged']:
    if key in results_all:
        r = results_all[key]
        print(f"{r['var']:<16} {r['type']:<12} "
              f"{r['coef']:>12.6f} {r['t']:>8.2f} {r['sig']:>6} {r['N']:>8,} {r['R2']:>8.3f}")

# Save JSON
os.makedirs(f"{BASE}/results/v14_tables", exist_ok=True)
out_path = f"{BASE}/results/v14_tables/mechanism_v14.json"
with open(out_path, 'w') as f:
    json.dump(results_all, f, indent=2, default=str, ensure_ascii=False)

print(f"\n结果已保存至 {out_path}")

# Also generate summary for easy copy into generate_tables_v13.py
print("\n" + "=" * 70)
print("生成可直接粘贴到 generate_tables_v13.py 的结果字典")
print("=" * 70)

print("\n# Panel A: 同期 X_t -> M_t")
print("mech_results = {")
for var in ['absDA', 'AuditFee', 'RetAutoCorr', 'InvestIneff']:
    key = f'{var}_concurrent'
    if key in results_all:
        r = results_all[key]
        print(f"    '{var}': {{\"coef\": {r['coef']}, \"se\": {r['se']}, "
              f"\"p\": {r['p']}, \"sig\": \"{r['sig']}\", \"N\": {r['N']}, \"R2\": {r['R2']}}},")
print("}")

print("\n# Panel B: 滞后 X_t -> M_{t+1}")
print("mech_lagged = {")
for var in ['absDA', 'AuditFee', 'RetAutoCorr', 'InvestIneff']:
    key = f'{var}_lagged'
    if key in results_all:
        r = results_all[key]
        print(f"    '{var}': {{\"coef\": {r['coef']}, \"se\": {r['se']}, "
              f"\"p\": {r['p']}, \"sig\": \"{r['sig']}\", \"N\": {r['N']}, \"R2\": {r['R2']}}},")
print("}")
