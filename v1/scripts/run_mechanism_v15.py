"""
全渠道机制检验: 10个渠道 × OLS/DML × 同期/滞后 = 40个回归
渠道:
  1. absDA         盈余管理程度 (Kothari 2005)
  2. AuditFee      审计费用(ln) (DeFond & Zhang 2014)
  3. RetAutoCorr   收益率自相关 (Chordia et al. 2008)
  4. InvestIneff   投资效率偏离 (Richardson 2006)
  5. Analyst       分析师覆盖 (Zhang 2006)
  6. Amihud        非流动性 (Amihud 2002)
  7. InstHold      机构持股比例
  8. Turnover      换手率
  9. RetVol        收益波动率
  10. ForecastDisp  分析师预测分歧

方法:
  OLS: pyfixest Firm+Year FE, 行业×年份聚类SE, 27控制变量
  DML: DoubleML PLR, CRE (Mundlak), Lasso+RF, 5fold×3rep
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import pyfixest as pf
import json, os, time, warnings
warnings.filterwarnings('ignore')

BASE = "/Users/mac/computerscience/15会计研究"
DATA = f"{BASE}/data_parquet"
OUT  = f"{BASE}/results/v15_tables"
os.makedirs(OUT, exist_ok=True)

# ================================================================
# 0. 加载基础面板
# ================================================================
print("=" * 70)
print("0. 加载基础面板")
print("=" * 70)

panel = pd.read_parquet(f"{DATA}/panel_dml.parquet")
ar_feat = pd.read_parquet(f"{DATA}/annual_report_features.parquet")

panel = panel.merge(
    ar_feat[['Stkcd', 'year', 'kw_total', 'kw_per10k', 'substantive_count',
             'kw_data_stock', 'kw_data_dev', 'kw_data_app', 'kw_data_value', 'kw_data_gov']],
    on=['Stkcd', 'year'], how='left'
)

fi = pd.read_parquet(f"{DATA}/firm_info.parquet",
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

# ================================================================
# 1. 构造需要计算的渠道变量
# ================================================================
print("\n" + "=" * 70)
print("1. 构造渠道变量")
print("=" * 70)

# --- 1a. absDA (Kothari 2005) ---
print("\n1a. absDA (Kothari 2005)")
def annual_report_filter(df):
    df = df.copy()
    df['Accper'] = df['Accper'].astype(str)
    mask = df['Accper'].str.endswith('12-31') & (df['Typrep'] == 'A')
    df = df[mask].copy()
    df['year'] = pd.to_datetime(df['Accper']).dt.year
    df['Stkcd'] = df['Stkcd'].astype(int)
    return df.drop_duplicates(['Stkcd', 'year'], keep='last')

bs = annual_report_filter(pd.read_parquet(f"{DATA}/balance_sheet.parquet"))
inc = annual_report_filter(pd.read_parquet(f"{DATA}/income_stmt.parquet"))
cfl = annual_report_filter(pd.read_parquet(f"{DATA}/cashflow.parquet"))

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
fin['TA_scaled'] = (fin['NetIncome_j'] - fin['CFO_j']) / fin['lagTA_j']
fin['inv_lagTA'] = 1.0 / fin['lagTA_j']
fin['dRev_adj'] = (fin['dRev'] - fin['dRec']) / fin['lagTA_j']
fin['PPE_scaled'] = fin['FixedAssets'] / fin['lagTA_j']
fin['ROA_j'] = fin['NetIncome_j'] / fin['lagTA_j']

fin = fin.merge(fi[['Stkcd', 'year', 'IndustryCodeC']], on=['Stkcd', 'year'], how='left')
fin_ind_mode = fi.groupby('Stkcd')['IndustryCodeC'].agg(
    lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else np.nan)
fin = fin.merge(fin_ind_mode.rename('Ind2_mode').reset_index(), on='Stkcd', how='left')
fin['IndustryCodeC'] = fin['IndustryCodeC'].fillna(fin['Ind2_mode'])
fin['Ind2_j'] = fin['IndustryCodeC'].str[:3]
fin = fin[~fin['Ind2_j'].str.startswith('J', na=False)]

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
            da_results.append({'Stkcd': grp.loc[idx, 'Stkcd'],
                             'year': grp.loc[idx, 'year'], 'absDA': abs(r)})
    except Exception:
        pass

da_df = pd.DataFrame(da_results)
panel = panel.merge(da_df, on=['Stkcd', 'year'], how='left')
print(f"  absDA: {panel['absDA'].notna().sum():,} obs")

# --- 1b. AuditFee ---
print("1b. AuditFee (ln审计费用)")
audit = pd.read_parquet(f"{DATA}/audit.parquet")
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
print(f"  AuditFee: {panel['AuditFee'].notna().sum():,} obs")

# --- 1c. RetAutoCorr ---
print("1c. RetAutoCorr (日收益一阶自相关)")
daily = pd.read_parquet(f"{DATA}/daily_return.parquet")
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
print(f"  RetAutoCorr: {panel['RetAutoCorr'].notna().sum():,} obs")
del daily

# --- 1d. InvestIneff (Richardson 2006) ---
print("1d. InvestIneff (Richardson 2006)")
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
inv_data['Invest'] = -inv_data['CFI_i'] / inv_data['lagTA_i']
inv_data['Lev_r'] = inv_data['TL_i'] / inv_data['TA_i']
inv_data['Cash_r'] = inv_data['CA_i'] / inv_data['TA_i']
inv_data['Size_r'] = np.log(inv_data['TA_i'].clip(lower=1))
inv_data['Growth_r'] = inv_data['Revenue_i'] / inv_data.groupby('Stkcd')['Revenue_i'].shift(1) - 1
inv_data['ROA_r'] = inv_data['NI_i'] / inv_data['lagTA_i']
for v in ['Invest', 'Lev_r', 'Cash_r', 'Size_r', 'Growth_r', 'ROA_r']:
    inv_data[f'lag_{v}'] = inv_data.groupby('Stkcd')[v].shift(1)

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
            invest_results.append({'Stkcd': grp.loc[idx, 'Stkcd'],
                                 'year': grp.loc[idx, 'year'], 'InvestIneff': abs(r)})
    except Exception:
        pass

invest_df = pd.DataFrame(invest_results)
panel = panel.merge(invest_df, on=['Stkcd', 'year'], how='left')
print(f"  InvestIneff: {panel['InvestIneff'].notna().sum():,} obs")

# --- 1e. ForecastDisp (分析师预测分歧) ---
print("1e. ForecastDisp (分析师预测分歧)")
try:
    af = pd.read_parquet(f"{DATA}/analyst_forecast.parquet")
    af['Stkcd'] = af['Stkcd'].astype(int)
    # Feps = forecasted EPS
    af['Feps'] = pd.to_numeric(af.get('Feps', af.columns[af.columns.str.contains('eps', case=False)].tolist()[0] if any(af.columns.str.contains('eps', case=False)) else 'Feps'), errors='coerce')
    af['Rptdt'] = pd.to_datetime(af['Rptdt'])
    af['rpt_year'] = af['Rptdt'].dt.year
    # ForecastDisp = std of EPS forecasts within year for same stock
    disp = af.groupby(['Stkcd', 'rpt_year']).agg(
        ForecastDisp=('Feps', 'std'),
        ForecastNum=('Feps', 'count')
    ).reset_index()
    disp = disp.rename(columns={'rpt_year': 'year'})
    # Need at least 3 forecasts for meaningful dispersion
    disp.loc[disp['ForecastNum'] < 3, 'ForecastDisp'] = np.nan
    panel = panel.merge(disp[['Stkcd', 'year', 'ForecastDisp']], on=['Stkcd', 'year'], how='left')
    print(f"  ForecastDisp: {panel['ForecastDisp'].notna().sum():,} obs")
except Exception as e:
    print(f"  ForecastDisp构造失败: {e}")
    panel['ForecastDisp'] = np.nan

# Construct Inv (存货比例) from bs (already filtered to annual)
bs_inv = bs[['Stkcd','year','A001218000','A001000000']].copy()
bs_inv['Inv'] = bs_inv['A001218000'] / bs_inv['A001000000']
panel = panel.merge(bs_inv[['Stkcd','year','Inv']], on=['Stkcd','year'], how='left')

# Construct Hhi (行业竞争度) from inc (already filtered to annual)
inc_hhi = panel[['Stkcd','year','Ind2']].merge(inc[['Stkcd','year','B001101000']], on=['Stkcd','year'], how='left')
hhi_df = inc_hhi.groupby(['Ind2','year']).apply(
    lambda g: ((g['B001101000'] / g['B001101000'].sum()) ** 2).sum() if g['B001101000'].sum() > 0 else np.nan
).reset_index(name='Hhi')
panel = panel.merge(hhi_df, on=['Ind2','year'], how='left')
print(f"  Inv: {panel['Inv'].notna().sum():,} obs, Hhi: {panel['Hhi'].notna().sum():,} obs")

# Free memory
del bs, inc, cfl, fin, fin_reg, inv_data, inv_reg, audit

# ================================================================
# 2. Winsorize + 准备回归样本
# ================================================================
print("\n" + "=" * 70)
print("2. Winsorize + 准备回归样本")
print("=" * 70)

def winsorize(s):
    lo, hi = s.quantile([0.01, 0.99])
    return s.clip(lo, hi)

controls_27 = ['Size', 'Lev', 'ROA', 'TobinQ', 'Age', 'Growth', 'IndepRatio',
               'Dual', 'Top1Share', 'SOE', 'CFO', 'Inv', 'Hhi']

# All mechanism variables
all_mech_vars = ['absDA', 'AuditFee', 'RetAutoCorr', 'InvestIneff',
                 'Analyst', 'Amihud', 'InstHold', 'Turnover', 'RetVol', 'ForecastDisp']

# Winsorize continuous variables
cont_vars = ['PriceDelay', 'DU_kw'] + [v for v in all_mech_vars if v not in ['Dual', 'SOE', 'AuditType']] + \
    [v for v in controls_27 if v not in ['Dual', 'SOE', 'AuditType']]
cont_vars = list(set(cont_vars))

for v in cont_vars:
    if v in panel.columns and panel[v].notna().any():
        mask = panel[v].notna()
        if mask.sum() > 100:
            panel.loc[mask, v] = winsorize(panel.loc[mask, v])

# Build regression sample (base: need DU_kw + 27 controls)
reg = panel.dropna(subset=['PriceDelay', 'DU_kw'] + controls_27).copy()
reg['Stkcd_str'] = reg['Stkcd'].astype(str)
reg['year_str'] = reg['year'].astype(str)
print(f"回归样本: N={len(reg):,}, firms={reg['Stkcd'].nunique():,}")

for v in all_mech_vars:
    n = reg[v].notna().sum()
    print(f"  {v}: {n:,} obs ({n/len(reg)*100:.1f}%)")

# ================================================================
# 3. 渠道定义
# ================================================================

# channel_name -> (cn_name, source, is_control_var, expected_sign)
# expected_sign: direction of DU_kw effect on this variable that improves pricing efficiency
CHANNELS = {
    'absDA':        ('盈余管理程度',   'Kothari et al. (2005)',       False, '-'),
    'AuditFee':     ('审计费用(对数)', 'DeFond and Zhang (2014)',    False, '+'),
    'RetAutoCorr':  ('收益率自相关',   'Chordia et al. (2008)',      False, '-'),
    'InvestIneff':  ('投资效率偏离',   'Richardson (2006)',          False, '-'),
    'Analyst':      ('分析师覆盖',     'Zhang (2006)',               False, '+'),
    'Amihud':       ('非流动性',       'Amihud (2002)',              False, '-'),
    'InstHold':     ('机构持股比例',   'Buss and Sundaresan (2023)', False, '+'),
    'Turnover':     ('换手率',         'Chordia et al. (2008)',      False, '+'),
    'RetVol':       ('收益波动率',     'Zhang (2006)',               False, '-'),
    'ForecastDisp': ('预测分歧',       'Diether et al. (2002)',      False, '-'),
}

# ================================================================
# 4. OLS机制回归 (pyfixest)
# ================================================================
print("\n" + "=" * 70)
print("4. OLS机制回归 (Firm+Year FE, IndYear cluster)")
print("=" * 70)

def sig_stars(p):
    if p is None or pd.isna(p):
        return ""
    if p < 0.01: return "***"
    if p < 0.05: return "**"
    if p < 0.1: return "*"
    return ""

ols_results = {}

for var, (cn_name, source, is_ctrl, exp_sign) in CHANNELS.items():
    print(f"\n--- {var} ({cn_name}) ---")

    # If this var is a control, remove from control list
    if is_ctrl:
        mech_controls = [c for c in controls_27 if c != var]
    elif var == 'AuditFee':
        mech_controls = [c for c in controls_27 if c != 'AuditType']
    else:
        mech_controls = controls_27[:]

    mech_ctrl_str = " + ".join(mech_controls)

    # ---- A: Concurrent DU_kw_t -> M_t ----
    sub = reg.dropna(subset=[var]).copy()
    if len(sub) < 500:
        print(f"  [OLS] 同期跳过: 仅{len(sub)}个obs")
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
        print(f"  [OLS] 同期: coef={coef:.6f}, t={t_stat:.2f}{stars}, N={n:,}, R2={r2:.3f}")
        ols_results[f'{var}_concurrent'] = {
            'var': var, 'cn_name': cn_name, 'source': source, 'method': 'OLS',
            'type': 'concurrent', 'expected_sign': exp_sign,
            'coef': round(coef, 6), 'se': round(se, 6), 't': round(t_stat, 2),
            'p': round(pval, 4), 'sig': stars, 'N': n, 'R2': round(r2, 3),
            'n_controls': len(mech_controls),
        }
    except Exception as e:
        print(f"  [OLS] 同期失败: {e}")

    # ---- B: Lagged DU_kw_t -> M_{t+1} ----
    lag_df = reg[['Stkcd', 'year', 'DU_kw'] + mech_controls +
                 ['Stkcd_str', 'year_str', 'IndYear']].copy()
    future_m = panel[['Stkcd', 'year', var]].copy()
    future_m['year'] = future_m['year'] - 1
    future_m = future_m.rename(columns={var: f'{var}_lead'})
    lag_df = lag_df.merge(future_m, on=['Stkcd', 'year'], how='inner')
    lag_df = lag_df.dropna(subset=[f'{var}_lead'])

    if len(lag_df) < 500:
        print(f"  [OLS] 滞后跳过: 仅{len(lag_df)}个obs")
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
        print(f"  [OLS] 滞后: coef={coef_l:.6f}, t={t_l:.2f}{stars_l}, N={n_l:,}, R2={r2_l:.3f}")
        ols_results[f'{var}_lagged'] = {
            'var': var, 'cn_name': cn_name, 'source': source, 'method': 'OLS',
            'type': 'lagged', 'expected_sign': exp_sign,
            'coef': round(coef_l, 6), 'se': round(se_l, 6), 't': round(t_l, 2),
            'p': round(pval_l, 4), 'sig': stars_l, 'N': n_l, 'R2': round(r2_l, 3),
            'n_controls': len(mech_controls),
        }
    except Exception as e:
        print(f"  [OLS] 滞后失败: {e}")

# Save OLS results
with open(f"{OUT}/mechanism_all_ols.json", 'w') as f:
    json.dump(ols_results, f, indent=2, default=str, ensure_ascii=False)
print(f"\nOLS结果已保存: {OUT}/mechanism_all_ols.json")


# ================================================================
# 5. DML机制回归 (DoubleML PLR + CRE)
# ================================================================
print("\n" + "=" * 70)
print("5. DML-CRE机制回归")
print("=" * 70)

from doubleml import DoubleMLData, DoubleMLPLR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor

def make_learners():
    ml_l = Pipeline([("scaler", StandardScaler()),
                     ("model", LassoCV(cv=5, max_iter=3000, n_jobs=-1))])
    ml_m = RandomForestRegressor(n_estimators=500, max_depth=6,
                                  min_samples_leaf=10, n_jobs=-1, random_state=42)
    return ml_l, ml_m


def build_cre(df_in, y_var, d_var, ctrl_list, id_col="Stkcd", year_col="year"):
    """CRE (Mundlak): firm means + year dummies"""
    req = [y_var, d_var] + ctrl_list + [id_col, year_col]
    df_c = df_in[req].dropna().copy()

    for v in [y_var, d_var] + ctrl_list:
        lo = df_c[v].quantile(0.01)
        hi = df_c[v].quantile(0.99)
        df_c[v] = df_c[v].clip(lo, hi)

    fm_cols = []
    for v in ctrl_list:
        col = f"fm_{v}"
        df_c[col] = df_c.groupby(id_col)[v].transform("mean")
        fm_cols.append(col)

    yr_dummies = pd.get_dummies(df_c[year_col], prefix="yr", drop_first=True)
    yr_cols = yr_dummies.columns.tolist()
    df_c = pd.concat([df_c, yr_dummies], axis=1)

    X_cols = ctrl_list + fm_cols + yr_cols
    X_mat  = df_c[X_cols].values.astype(float)
    Y_vec  = df_c[y_var].values
    D_vec  = df_c[d_var].values

    return X_mat, Y_vec, D_vec, len(df_c)


def run_dml_spec(X_mat, Y_vec, D_vec, label="DML"):
    np.random.seed(42)
    dml_data = DoubleMLData.from_arrays(x=X_mat, y=Y_vec, d=D_vec)
    ml_l, ml_m = make_learners()
    plr = DoubleMLPLR(
        obj_dml_data=dml_data, ml_l=ml_l, ml_m=ml_m,
        n_folds=5, n_rep=3, score="partialling out",
    )
    t0 = time.time()
    plr.fit()
    elapsed = time.time() - t0

    res = {
        "coef": float(plr.coef[0]),
        "se":   float(plr.se[0]),
        "t":    float(plr.t_stat[0]),
        "p":    float(plr.pval[0]),
    }
    print(f"  [{label}]: coef={res['coef']:.6f}, t={res['t']:.2f}, p={res['p']:.4f}  [{elapsed:.0f}s]")
    return res


dml_results = {}

for var, (cn_name, source, is_ctrl, exp_sign) in CHANNELS.items():
    print(f"\n--- DML: {var} ({cn_name}) ---")

    if is_ctrl:
        mech_controls = [c for c in controls_27 if c != var]
    elif var == 'AuditFee':
        mech_controls = [c for c in controls_27 if c != 'AuditType']
    else:
        mech_controls = controls_27[:]

    # ---- A: Concurrent ----
    sub = reg.dropna(subset=[var]).copy()
    if len(sub) < 500:
        print(f"  同期跳过: 仅{len(sub)}个obs")
        continue

    try:
        X_mat, Y_vec, D_vec, n = build_cre(sub, var, 'DU_kw', mech_controls)
        print(f"  同期样本: {n:,}")
        res = run_dml_spec(X_mat, Y_vec, D_vec, label=f"DML-同期")
        stars = sig_stars(res['p'])
        dml_results[f'{var}_concurrent'] = {
            'var': var, 'cn_name': cn_name, 'source': source, 'method': 'DML',
            'type': 'concurrent', 'expected_sign': exp_sign,
            'coef': round(res['coef'], 6), 'se': round(res['se'], 6),
            't': round(res['t'], 2), 'p': round(res['p'], 4), 'sig': stars,
            'N': n, 'n_controls': len(mech_controls),
        }
    except Exception as e:
        print(f"  DML同期失败: {e}")

    # ---- B: Lagged ----
    lag_df = reg[['Stkcd', 'year', 'DU_kw'] + mech_controls].copy()
    future_m = panel[['Stkcd', 'year', var]].copy()
    future_m['year'] = future_m['year'] - 1
    future_m = future_m.rename(columns={var: f'{var}_lead'})
    lag_df = lag_df.merge(future_m, on=['Stkcd', 'year'], how='inner')
    lag_df = lag_df.dropna(subset=[f'{var}_lead'])

    if len(lag_df) < 500:
        print(f"  滞后跳过: 仅{len(lag_df)}个obs")
        continue

    try:
        X_mat_l, Y_vec_l, D_vec_l, n_l = build_cre(lag_df, f'{var}_lead', 'DU_kw', mech_controls)
        print(f"  滞后样本: {n_l:,}")
        res_l = run_dml_spec(X_mat_l, Y_vec_l, D_vec_l, label=f"DML-滞后")
        stars_l = sig_stars(res_l['p'])
        dml_results[f'{var}_lagged'] = {
            'var': var, 'cn_name': cn_name, 'source': source, 'method': 'DML',
            'type': 'lagged', 'expected_sign': exp_sign,
            'coef': round(res_l['coef'], 6), 'se': round(res_l['se'], 6),
            't': round(res_l['t'], 2), 'p': round(res_l['p'], 4), 'sig': stars_l,
            'N': n_l, 'n_controls': len(mech_controls),
        }
    except Exception as e:
        print(f"  DML滞后失败: {e}")

# Save DML results
with open(f"{OUT}/mechanism_all_dml.json", 'w') as f:
    json.dump(dml_results, f, indent=2, default=str, ensure_ascii=False)
print(f"\nDML结果已保存: {OUT}/mechanism_all_dml.json")


# ================================================================
# 6. 汇总对比
# ================================================================
print("\n" + "=" * 70)
print("6. 全渠道 OLS vs DML 汇总")
print("=" * 70)

header = f"{'渠道':<16} {'类型':<6} {'方法':<5} {'系数':>12} {'t值':>8} {'显著性':>6} {'N':>8} {'预期':>4}"
print(header)
print("-" * 78)

for var in CHANNELS:
    for timing in ['concurrent', 'lagged']:
        key = f'{var}_{timing}'
        timing_cn = '同期' if timing == 'concurrent' else '滞后'
        for method, results in [('OLS', ols_results), ('DML', dml_results)]:
            if key in results:
                r = results[key]
                match = '✓' if (r['expected_sign'] == '+' and r['coef'] > 0) or \
                               (r['expected_sign'] == '-' and r['coef'] < 0) else '✗'
                print(f"{r['var']:<16} {timing_cn:<6} {method:<5} "
                      f"{r['coef']:>12.6f} {r['t']:>8.2f} {r['sig']:>6} {r['N']:>8,} {match:>4}")

# Save combined results
combined = {'ols': ols_results, 'dml': dml_results}
with open(f"{OUT}/mechanism_all_combined.json", 'w') as f:
    json.dump(combined, f, indent=2, default=str, ensure_ascii=False)
print(f"\n全部结果已保存: {OUT}/mechanism_all_combined.json")

# Summary table
print("\n" + "=" * 70)
print("简明汇总: 哪些渠道在OLS和DML下都显著?")
print("=" * 70)
for var in CHANNELS:
    ols_c = ols_results.get(f'{var}_concurrent', {})
    ols_l = ols_results.get(f'{var}_lagged', {})
    dml_c = dml_results.get(f'{var}_concurrent', {})
    dml_l = dml_results.get(f'{var}_lagged', {})

    ols_sig = (ols_c.get('sig', '') != '') or (ols_l.get('sig', '') != '')
    dml_sig = (dml_c.get('sig', '') != '') or (dml_l.get('sig', '') != '')
    both = ols_sig and dml_sig

    ols_c_t = f"t={ols_c.get('t','N/A')}{ols_c.get('sig','')}" if ols_c else "N/A"
    dml_c_t = f"t={dml_c.get('t','N/A')}{dml_c.get('sig','')}" if dml_c else "N/A"

    status = "OLS+DML" if both else ("OLS only" if ols_sig else ("DML only" if dml_sig else "n.s."))
    print(f"  {var:<16} OLS同期={ols_c_t:<18} DML同期={dml_c_t:<18} [{status}]")

print("\nDone!")
