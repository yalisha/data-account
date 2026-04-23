"""
9个新机制变量构造 + 江艇两步法回归
对应顶刊理论来源:
1. FcstAccuracy  - 分析师预测绝对误差 (McClure et al. RAS 2025)
2. FcstAcc_ST    - 短期预测准确度 (Dessaint et al. JF 2024)
3. FcstAcc_LT    - 长期预测准确度 (同上, horizon effect)
4. InvestIneff   - Richardson投资效率偏离 (Li et al. JFS 2024)
5. absDA         - Kothari盈余管理 (盈余质量)
6. ARLength      - 年报篇幅万字 (信息处理成本)
7. IdioVol       - 特质波动率 (Davila & Parlatore JFE 2023)
8. TurnoverVol   - 换手率波动率 (IREF 2024 噪声交易)
9. InstStable    - 稳定型机构持股 (基金+保险+QFII)
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import pyfixest as pf
import json, os, warnings
warnings.filterwarnings('ignore')

BASE = "/Users/mac/computerscience/15会计研究"

# ============================================================
# 0. 加载基础面板 (与 run_expanded_mechanism_het.py 一致)
# ============================================================
print("=" * 70)
print("0. 加载基础面板")
print("=" * 70)

panel = pd.read_parquet(f"{BASE}/data_parquet/panel.parquet")
ar_feat = pd.read_parquet(f"{BASE}/data_parquet/annual_report_features.parquet")

panel = panel.merge(
    ar_feat[['Stkcd', 'year', 'kw_total', 'kw_per10k', 'substantive_count',
             'total_chars']],
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

mask_fin = panel['IndustryCodeC'].str.startswith('J', na=False)
mask_st = panel['LISTINGSTATE'].isin(['ST', '*ST'])
mask_new = panel['Age'] <= 0
panel = panel[~mask_fin & ~mask_st & ~mask_new].copy()

panel['Ind2'] = panel['IndustryCodeC'].str[:3]
panel['IndYear'] = panel['Ind2'].astype(str) + '_' + panel['year'].astype(str)
panel['DU_kw'] = panel['kw_per10k']

print(f"基础面板: {len(panel):,} obs, {panel['Stkcd'].nunique():,} firms")


# ============================================================
# 1. FcstAccuracy / FcstAcc_ST / FcstAcc_LT
# ============================================================
print("\n" + "=" * 70)
print("1. 构造分析师预测准确度 (FcstAccuracy, FcstAcc_ST, FcstAcc_LT)")
print("=" * 70)

af = pd.read_parquet(f"{BASE}/data_parquet/analyst_forecast.parquet")
af['Rptdt'] = pd.to_datetime(af['Rptdt'])
af['Fenddt'] = pd.to_datetime(af['Fenddt'])
af['rpt_year'] = af['Rptdt'].dt.year
af['fend_year'] = af['Fenddt'].dt.year
af = af.dropna(subset=['Feps'])

# 实际EPS
ps = pd.read_parquet(f"{BASE}/data_parquet/per_share.parquet",
                     columns=['Stkcd', 'Accper', 'Typrep', 'F090101B'])
ps['Accper'] = ps['Accper'].astype(str)
ps_annual = ps[ps['Accper'].str.endswith('12-31') & (ps['Typrep'] == 'A')].copy()
ps_annual['year'] = pd.to_datetime(ps_annual['Accper']).dt.year
ps_annual['Stkcd'] = ps_annual['Stkcd'].astype(int)
ps_annual = ps_annual.drop_duplicates(['Stkcd', 'year'], keep='last')
ps_annual = ps_annual.rename(columns={'F090101B': 'ActualEPS'})
ps_annual['ActualEPS'] = pd.to_numeric(ps_annual['ActualEPS'], errors='coerce')

# 合并预测与实际
af = af.merge(ps_annual[['Stkcd', 'year', 'ActualEPS']],
              left_on=['Stkcd', 'fend_year'], right_on=['Stkcd', 'year'],
              how='inner', suffixes=('', '_actual'))
af['FE'] = np.abs(af['Feps'] - af['ActualEPS'])
# 标准化: 除以|ActualEPS|, 避免除零
af['FE_norm'] = af['FE'] / np.maximum(np.abs(af['ActualEPS']), 0.01)

# 短期: rpt_year == fend_year (当年预测当年)
af['horizon'] = af['fend_year'] - af['rpt_year']
af_st = af[af['horizon'] == 0]
af_lt = af[af['horizon'] >= 1]

# 按stock-报告年汇总
def agg_accuracy(df, suffix=''):
    g = df.groupby(['Stkcd', 'rpt_year']).agg(
        fe_mean=('FE_norm', 'mean'),
        fe_count=('FE_norm', 'count')
    ).reset_index()
    g = g[g['fe_count'] >= 2]
    g = g.rename(columns={'rpt_year': 'year', 'fe_mean': f'FcstAcc{suffix}'})
    return g[['Stkcd', 'year', f'FcstAcc{suffix}']]

fcst_all = agg_accuracy(af, '')
fcst_st = agg_accuracy(af_st, '_ST')
fcst_lt = agg_accuracy(af_lt, '_LT')

panel = panel.merge(fcst_all, on=['Stkcd', 'year'], how='left')
panel = panel.merge(fcst_st, on=['Stkcd', 'year'], how='left')
panel = panel.merge(fcst_lt, on=['Stkcd', 'year'], how='left')

for v in ['FcstAcc', 'FcstAcc_ST', 'FcstAcc_LT']:
    n = panel[v].notna().sum()
    print(f"  {v}: {n:,} obs, mean={panel[v].mean():.4f}" if n > 0 else f"  {v}: 0 obs")


# ============================================================
# 2. InvestIneff - Richardson (2006) 投资效率
# ============================================================
print("\n" + "=" * 70)
print("2. 构造投资效率偏离 (InvestIneff)")
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

# 合并
inv_data = bs[['Stkcd', 'year', 'A001000000', 'A001100000', 'A002000000',
               'A001202000']].merge(
    inc[['Stkcd', 'year', 'B001101000', 'B002000000']], on=['Stkcd', 'year'], how='inner'
).merge(
    cfl[['Stkcd', 'year', 'C001000000', 'C002000000']], on=['Stkcd', 'year'], how='inner'
)

inv_data = inv_data.rename(columns={
    'A001000000': 'TA', 'A001100000': 'CA', 'A002000000': 'TL',
    'A001202000': 'FixedAssets',
    'B001101000': 'Revenue', 'B002000000': 'NI',
    'C001000000': 'CFO', 'C002000000': 'CFI',
})

for c in ['TA', 'CA', 'TL', 'FixedAssets', 'Revenue', 'NI', 'CFO', 'CFI']:
    inv_data[c] = pd.to_numeric(inv_data[c], errors='coerce')

inv_data = inv_data.sort_values(['Stkcd', 'year'])
inv_data['lagTA'] = inv_data.groupby('Stkcd')['TA'].shift(1)

# Investment = (固定资产净额_t - 固定资产净额_{t-1} + 折旧) / lagTA
# 简化: 用CFI / lagTA 作为投资率 (Richardson 2006 用资本支出)
inv_data['Invest'] = -inv_data['CFI'] / inv_data['lagTA']  # CFI通常为负

# Richardson模型: Invest_t = f(Growth_{t-1}, Lev_{t-1}, Cash_{t-1}, Age, Size_{t-1}, Return_{t-1})
inv_data['Lev_r'] = inv_data['TL'] / inv_data['TA']
inv_data['Cash_r'] = inv_data['CA'] / inv_data['TA']
inv_data['Size_r'] = np.log(inv_data['TA'])
inv_data['Growth_r'] = inv_data['Revenue'] / inv_data.groupby('Stkcd')['Revenue'].shift(1) - 1
inv_data['ROA_r'] = inv_data['NI'] / inv_data['lagTA']

# 滞后解释变量
for v in ['Invest', 'Lev_r', 'Cash_r', 'Size_r', 'Growth_r', 'ROA_r']:
    inv_data[f'lag_{v}'] = inv_data.groupby('Stkcd')[v].shift(1)

# 加行业
inv_data = inv_data.merge(fi[['Stkcd', 'year', 'IndustryCodeC']], on=['Stkcd', 'year'], how='left')
inv_data['Ind2'] = inv_data['IndustryCodeC'].str[:3]
inv_data = inv_data[~inv_data['Ind2'].str.startswith('J', na=False)]

rich_vars = ['Invest', 'lag_Invest', 'lag_Lev_r', 'lag_Cash_r', 'lag_Size_r',
             'lag_Growth_r', 'lag_ROA_r']
inv_reg = inv_data.dropna(subset=rich_vars + ['Ind2']).copy()

# Winsorize
for v in rich_vars:
    lo, hi = inv_reg[v].quantile([0.01, 0.99])
    inv_reg[v] = inv_reg[v].clip(lo, hi)

# 分行业-年度回归
invest_results = []
for (ind, yr), grp in inv_reg.groupby(['Ind2', 'year']):
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
# 3. absDA - Kothari (2005) 业绩调整Jones模型
# ============================================================
print("\n" + "=" * 70)
print("3. 构造盈余管理 (absDA)")
print("=" * 70)

fin = bs[['Stkcd', 'year', 'A001000000', 'A001107000']].merge(
    inc[['Stkcd', 'year', 'B001101000', 'B002000000']], on=['Stkcd', 'year'], how='inner'
).merge(
    cfl[['Stkcd', 'year', 'C001000000']], on=['Stkcd', 'year'], how='inner'
)
fin = fin.rename(columns={
    'A001000000': 'TotalAssets', 'A001107000': 'Receivables',
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
fin['ROA_j'] = fin['NetIncome_j'] / fin['lagTA_j']

fin = fin.merge(fi[['Stkcd', 'year', 'IndustryCodeC']], on=['Stkcd', 'year'], how='left')
fin['Ind2_j'] = fin['IndustryCodeC'].str[:3]
fin = fin[~fin['Ind2_j'].str.startswith('J', na=False)]

jones_vars = ['TA_scaled', 'inv_lagTA', 'dRev_adj', 'ROA_j']
fin_reg = fin.dropna(subset=jones_vars + ['Ind2_j']).copy()
for v in jones_vars:
    lo, hi = fin_reg[v].quantile([0.01, 0.99])
    fin_reg[v] = fin_reg[v].clip(lo, hi)

da_results = []
for (ind, yr), grp in fin_reg.groupby(['Ind2_j', 'year']):
    if len(grp) < 10:
        continue
    y = grp['TA_scaled'].values
    X = grp[['inv_lagTA', 'dRev_adj', 'ROA_j']].values
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
# 4. ARLength - 年报篇幅 (信息处理成本代理)
# ============================================================
print("\n" + "=" * 70)
print("4. 构造年报篇幅 (ARLength)")
print("=" * 70)

# total_chars 已经在 panel 中 (from annual_report_features)
panel['ARLength'] = panel['total_chars'] / 10000  # 万字
print(f"  ARLength: {panel['ARLength'].notna().sum():,} obs, "
      f"mean={panel['ARLength'].mean():.1f}万字")


# ============================================================
# 5. IdioVol - 特质波动率
# ============================================================
print("\n" + "=" * 70)
print("5. 构造特质波动率 (IdioVol)")
print("=" * 70)

daily = pd.read_parquet(f"{BASE}/data_parquet/daily_return.parquet")
daily['Trddt'] = pd.to_datetime(daily['Trddt'])
daily['year'] = daily['Trddt'].dt.year

# 市场收益率
mkt = pd.read_parquet(f"{BASE}/data_parquet/market_index.parquet")
if 'Trddt' in mkt.columns:
    mkt['Trddt'] = pd.to_datetime(mkt['Trddt'])
else:
    # 尝试不同列名
    date_col = [c for c in mkt.columns if 'date' in c.lower() or 'trd' in c.lower()]
    if date_col:
        mkt = mkt.rename(columns={date_col[0]: 'Trddt'})
        mkt['Trddt'] = pd.to_datetime(mkt['Trddt'])

print(f"  Market index columns: {list(mkt.columns)}")

# 用沪深300或全A等权作为市场收益
ret_col = [c for c in mkt.columns if 'ret' in c.lower() or 'return' in c.lower() or 'Cdretwdos' in c]
print(f"  Return columns found: {ret_col}")

# 如果没有现成市场收益，用个股收益的截面均值
if not ret_col:
    print("  No market return column found, computing from daily cross-sectional mean")
    mkt_ret = daily.groupby('Trddt')['Dretwd'].mean().reset_index()
    mkt_ret = mkt_ret.rename(columns={'Dretwd': 'MktRet'})
else:
    mkt_ret = mkt[['Trddt', ret_col[0]]].copy()
    mkt_ret = mkt_ret.rename(columns={ret_col[0]: 'MktRet'})
    mkt_ret['MktRet'] = pd.to_numeric(mkt_ret['MktRet'], errors='coerce')

daily = daily.merge(mkt_ret, on='Trddt', how='left')

# 分stock-year计算特质波动率
def calc_idiovol(g):
    g = g.dropna(subset=['Dretwd', 'MktRet'])
    if len(g) < 60:
        return np.nan
    y = g['Dretwd'].values
    X = sm.add_constant(g['MktRet'].values)
    try:
        resid = sm.OLS(y, X).fit().resid
        return np.std(resid, ddof=1)
    except Exception:
        return np.nan

idiovol = daily.groupby(['Stkcd', 'year']).apply(calc_idiovol).reset_index()
idiovol.columns = ['Stkcd', 'year', 'IdioVol']
panel = panel.merge(idiovol, on=['Stkcd', 'year'], how='left')
print(f"  IdioVol: {panel['IdioVol'].notna().sum():,} obs, mean={panel['IdioVol'].mean():.4f}")


# ============================================================
# 6. TurnoverVol - 换手率波动率 (噪声交易代理)
# ============================================================
print("\n" + "=" * 70)
print("6. 构造换手率波动率 (TurnoverVol)")
print("=" * 70)

# Turnover = 成交金额 / 流通市值
daily['Turnover'] = daily['Dnvaltrd'] / daily['Dsmvosd']

def calc_turnover_vol(g):
    t = g['Turnover'].dropna()
    if len(t) < 60:
        return np.nan
    return np.std(t, ddof=1)

tvol = daily.groupby(['Stkcd', 'year']).apply(calc_turnover_vol).reset_index()
tvol.columns = ['Stkcd', 'year', 'TurnoverVol']
panel = panel.merge(tvol, on=['Stkcd', 'year'], how='left')
print(f"  TurnoverVol: {panel['TurnoverVol'].notna().sum():,} obs, "
      f"mean={panel['TurnoverVol'].mean():.6f}")


# ============================================================
# 7. InstStable - 稳定型机构持股 (基金+保险+QFII)
# ============================================================
print("\n" + "=" * 70)
print("7. 构造稳定型机构持股 (InstStable)")
print("=" * 70)

ih = pd.read_parquet(f"{BASE}/data_parquet/inst_holding.parquet")
ih = ih.rename(columns={'Symbol': 'Stkcd'})
ih['Stkcd'] = ih['Stkcd'].astype(int)
ih['EndDate'] = ih['EndDate'].astype(str)
# 取年末数据
ih_annual = ih[ih['EndDate'].str.endswith('12-31')].copy()
ih_annual['year'] = pd.to_datetime(ih_annual['EndDate']).dt.year

for c in ['FundHoldProportion', 'InsuranceHoldProportion', 'QFIIHoldProportion']:
    ih_annual[c] = pd.to_numeric(ih_annual[c], errors='coerce').fillna(0)

ih_annual['InstStable'] = (ih_annual['FundHoldProportion'] +
                           ih_annual['InsuranceHoldProportion'] +
                           ih_annual['QFIIHoldProportion'])

ih_annual = ih_annual.drop_duplicates(['Stkcd', 'year'], keep='last')
panel = panel.merge(ih_annual[['Stkcd', 'year', 'InstStable']],
                    on=['Stkcd', 'year'], how='left')
print(f"  InstStable: {panel['InstStable'].notna().sum():,} obs, "
      f"mean={panel['InstStable'].mean():.4f}")


# ============================================================
# 8. Winsorize 所有新变量
# ============================================================
print("\n" + "=" * 70)
print("8. Winsorize 新变量")
print("=" * 70)

def winsorize(s):
    lo, hi = s.quantile([0.01, 0.99])
    return s.clip(lo, hi)

new_mech_vars = ['FcstAcc', 'FcstAcc_ST', 'FcstAcc_LT', 'InvestIneff', 'absDA',
                 'ARLength', 'IdioVol', 'TurnoverVol', 'InstStable']

controls = ['Size', 'Lev', 'ROA', 'TobinQ', 'Age', 'Growth', 'BoardSize',
            'IndepRatio', 'Dual', 'Top1Share', 'SOE', 'InstHold', 'Amihud',
            'Analyst', 'AuditType']

# Winsorize
cont_vars = ['PriceDelay', 'DU_kw'] + new_mech_vars
for v in cont_vars + controls:
    if v in panel.columns and panel[v].notna().any() and panel[v].dtype in ['float64', 'float32', 'int64']:
        mask = panel[v].notna()
        if mask.sum() > 100:
            panel.loc[mask, v] = winsorize(panel.loc[mask, v])

# 回归样本
reg = panel.dropna(subset=['PriceDelay', 'DU_kw'] + controls).copy()
reg['Stkcd_str'] = reg['Stkcd'].astype(str)
reg['year_str'] = reg['year'].astype(str)
print(f"回归样本: N={len(reg):,}, firms={reg['Stkcd'].nunique():,}")

for v in new_mech_vars:
    n = reg[v].notna().sum()
    if n > 0:
        print(f"  {v}: {n:,} obs ({n/len(reg)*100:.1f}%)")


# ============================================================
# 9. 江艇两步法回归: DU_kw -> M (同期 + 滞后一期)
# ============================================================
print("\n" + "=" * 70)
print("9. 江艇两步法机制回归")
print("=" * 70)

ctrl_str = " + ".join(controls)

def sig_stars(p):
    if p is None or pd.isna(p):
        return ""
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.1:
        return "*"
    return ""

mech_map = {
    'FcstAcc':     ('分析师预测误差',     'McClure et al. (RAS 2025)'),
    'FcstAcc_ST':  ('短期预测误差',       'Dessaint et al. (JF 2024)'),
    'FcstAcc_LT':  ('长期预测误差',       'Dessaint et al. (JF 2024)'),
    'InvestIneff': ('投资效率偏离',       'Li et al. (JFS 2024)'),
    'absDA':       ('盈余管理程度',       'Kothari (2005)'),
    'ARLength':    ('年报篇幅(万字)',     '信息处理成本'),
    'IdioVol':     ('特质波动率',         'Davila & Parlatore (JFE 2023)'),
    'TurnoverVol': ('换手率波动率',       'IREF 2024'),
    'InstStable':  ('稳定型机构持股',     '基金+保险+QFII'),
}

results_all = {}

for var, (cn_name, source) in mech_map.items():
    print(f"\n--- {var} ({cn_name}) [{source}] ---")

    # 从控制变量中排除自身(避免多重共线性)
    mech_controls = [c for c in controls if c != var]
    mech_ctrl_str = " + ".join(mech_controls)

    # Panel A: 同期 DU_kw_t -> M_t
    sub = reg.dropna(subset=[var]).copy()
    if len(sub) < 500:
        print(f"  跳过: 仅{len(sub)}个obs")
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
            'coef': coef, 'se': se, 't': t_stat, 'p': pval, 'sig': stars,
            'N': n, 'R2': r2
        }
    except Exception as e:
        print(f"  同期回归失败: {e}")

    # Panel B: 滞后 DU_kw_t -> M_{t+1}
    lag_df = reg[['Stkcd', 'year', 'DU_kw'] + mech_controls +
                 ['Stkcd_str', 'year_str', 'IndYear']].copy()
    future_m = panel[['Stkcd', 'year', var]].copy()
    future_m['year'] = future_m['year'] - 1
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
            'coef': coef_l, 'se': se_l, 't': t_l, 'p': pval_l, 'sig': stars_l,
            'N': n_l, 'R2': r2_l
        }
    except Exception as e:
        print(f"  滞后回归失败: {e}")


# ============================================================
# 10. 汇总输出
# ============================================================
print("\n" + "=" * 70)
print("10. 汇总")
print("=" * 70)

print(f"\n{'变量':<16} {'中文名':<14} {'类型':<6} {'系数':>12} {'t值':>8} {'显著性':>6} {'N':>8}")
print("-" * 78)
for key, r in sorted(results_all.items()):
    print(f"{r['var']:<16} {r['cn_name']:<14} {r['type']:<6} "
          f"{r['coef']:>12.6f} {r['t']:>8.2f} {r['sig']:>6} {r['N']:>8,}")

# 保存
os.makedirs(f"{BASE}/results/new_mechanisms_9", exist_ok=True)
with open(f"{BASE}/results/new_mechanisms_9/results.json", 'w') as f:
    json.dump(results_all, f, indent=2, default=str, ensure_ascii=False)

# 保存CSV汇总
rows = []
for key, r in sorted(results_all.items()):
    rows.append(r)
pd.DataFrame(rows).to_csv(f"{BASE}/results/new_mechanisms_9/summary.csv", index=False)

print(f"\n结果已保存至 results/new_mechanisms_9/")
