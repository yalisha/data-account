"""
第二轮新机制变量 (6个) + 江艇两步法回归
10. AuditFee     - 审计费用对数 (审计质量/可验证性渠道)
11. RatingDisp   - 分析师评级分歧度 (投资建议共识渠道)
12. ReportFreq   - 分析师研报频次 (信息生产频率渠道)
13. RetAutoCorr  - 收益率自相关 (价格调整速度渠道)
14. CashFlowVol  - 现金流波动率 (经营不确定性渠道)
15. MgmtHold     - 管理层持股比例 (代理成本渠道)
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import pyfixest as pf
import json, os, warnings
warnings.filterwarnings('ignore')

BASE = "/Users/mac/computerscience/15会计研究"

# ============================================================
# 0. 加载基础面板 (同 round 1)
# ============================================================
print("=" * 70)
print("加载基础面板")
print("=" * 70)

panel = pd.read_parquet(f"{BASE}/data_parquet/panel.parquet")
ar_feat = pd.read_parquet(f"{BASE}/data_parquet/annual_report_features.parquet")
panel = panel.merge(
    ar_feat[['Stkcd', 'year', 'kw_total', 'kw_per10k', 'substantive_count']],
    on=['Stkcd', 'year'], how='left')

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
# 10. AuditFee - 审计费用对数
# ============================================================
print("\n" + "=" * 70)
print("10. 构造审计费用 (AuditFee)")
print("=" * 70)

audit = pd.read_parquet(f"{BASE}/data_parquet/audit.parquet")
audit['Accper'] = audit['Accper'].astype(str)
audit_annual = audit[audit['Accper'].str.endswith('12-31')].copy()
audit_annual['year'] = pd.to_datetime(audit_annual['Accper']).dt.year
audit_annual['Stkcd'] = audit_annual['Stkcd'].astype(int)
audit_annual['Tcost'] = pd.to_numeric(audit_annual['Tcost'], errors='coerce')
# 用Dcost补Tcost
audit_annual['Dcost'] = pd.to_numeric(audit_annual['Dcost'], errors='coerce')
audit_annual['fee'] = audit_annual['Tcost'].fillna(audit_annual['Dcost'])
audit_annual = audit_annual.dropna(subset=['fee'])
audit_annual = audit_annual[audit_annual['fee'] > 0]
audit_annual['AuditFee'] = np.log(audit_annual['fee'])
audit_annual = audit_annual.drop_duplicates(['Stkcd', 'year'], keep='last')

panel = panel.merge(audit_annual[['Stkcd', 'year', 'AuditFee']],
                    on=['Stkcd', 'year'], how='left')
print(f"  AuditFee: {panel['AuditFee'].notna().sum():,} obs, "
      f"mean={panel['AuditFee'].mean():.2f}")

# ============================================================
# 11. RatingDisp - 分析师评级分歧度
# ============================================================
print("\n" + "=" * 70)
print("11. 构造分析师评级分歧度 (RatingDisp)")
print("=" * 70)

ar = pd.read_parquet(f"{BASE}/data_parquet/analyst_rating.parquet")
ar['Stkcd'] = pd.to_numeric(ar['Stkcd'], errors='coerce')
ar = ar.dropna(subset=['Stkcd'])
ar['Stkcd'] = ar['Stkcd'].astype(int)
ar['Rptdt'] = pd.to_datetime(ar['Rptdt'])
ar['rpt_year'] = ar['Rptdt'].dt.year

# Stdrank编码: 买入=5, 增持=4, 中性=3, 减持=2, 卖出=1
rank_map = {'买入': 5, '增持': 4, '中性': 3, '减持': 2, '卖出': 1}
ar['rank_num'] = ar['Stdrank'].map(rank_map)
ar = ar.dropna(subset=['rank_num'])

rating_disp = ar.groupby(['Stkcd', 'rpt_year']).agg(
    rd_std=('rank_num', 'std'),
    rd_count=('rank_num', 'count')
).reset_index()
rating_disp = rating_disp[rating_disp['rd_count'] >= 3]
rating_disp = rating_disp.rename(columns={'rpt_year': 'year', 'rd_std': 'RatingDisp'})

panel = panel.merge(rating_disp[['Stkcd', 'year', 'RatingDisp']],
                    on=['Stkcd', 'year'], how='left')
print(f"  RatingDisp: {panel['RatingDisp'].notna().sum():,} obs, "
      f"mean={panel['RatingDisp'].mean():.4f}")

# ============================================================
# 12. ReportFreq - 分析师研报频次
# ============================================================
print("\n" + "=" * 70)
print("12. 构造分析师研报频次 (ReportFreq)")
print("=" * 70)

report_freq = ar.groupby(['Stkcd', 'rpt_year'])['rank_num'].count().reset_index()
report_freq.columns = ['Stkcd', 'year', 'ReportFreq']
report_freq['ReportFreq'] = np.log1p(report_freq['ReportFreq'])

panel = panel.merge(report_freq, on=['Stkcd', 'year'], how='left')
print(f"  ReportFreq: {panel['ReportFreq'].notna().sum():,} obs, "
      f"mean={panel['ReportFreq'].mean():.2f}")

# ============================================================
# 13. RetAutoCorr - 收益率自相关系数
# ============================================================
print("\n" + "=" * 70)
print("13. 构造收益率自相关 (RetAutoCorr)")
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

# ============================================================
# 14. CashFlowVol - 现金流波动率 (t, t+1, t+2)
# ============================================================
print("\n" + "=" * 70)
print("14. 构造现金流波动率 (CashFlowVol)")
print("=" * 70)

cfl = pd.read_parquet(f"{BASE}/data_parquet/cashflow.parquet")
cfl['Accper'] = cfl['Accper'].astype(str)
cfl_annual = cfl[cfl['Accper'].str.endswith('12-31') & (cfl['Typrep'] == 'A')].copy()
cfl_annual['year'] = pd.to_datetime(cfl_annual['Accper']).dt.year
cfl_annual['Stkcd'] = cfl_annual['Stkcd'].astype(int)
cfl_annual['CFO'] = pd.to_numeric(cfl_annual['C001000000'], errors='coerce')
cfl_annual = cfl_annual.drop_duplicates(['Stkcd', 'year'], keep='last')

bs = pd.read_parquet(f"{BASE}/data_parquet/balance_sheet.parquet")
bs['Accper'] = bs['Accper'].astype(str)
bs_annual = bs[bs['Accper'].str.endswith('12-31') & (bs['Typrep'] == 'A')].copy()
bs_annual['year'] = pd.to_datetime(bs_annual['Accper']).dt.year
bs_annual['Stkcd'] = bs_annual['Stkcd'].astype(int)
bs_annual['TA'] = pd.to_numeric(bs_annual['A001000000'], errors='coerce')
bs_annual = bs_annual.drop_duplicates(['Stkcd', 'year'], keep='last')

cf_data = cfl_annual[['Stkcd', 'year', 'CFO']].merge(
    bs_annual[['Stkcd', 'year', 'TA']], on=['Stkcd', 'year'], how='inner')
cf_data['CFO_TA'] = cf_data['CFO'] / cf_data['TA']
cf_data = cf_data.sort_values(['Stkcd', 'year'])

# 滚动3年标准差 (t, t+1, t+2) -> 赋给t年
results_cf = []
for stkcd, g in cf_data.groupby('Stkcd'):
    g = g.sort_values('year')
    for i in range(len(g)):
        vals = g['CFO_TA'].iloc[i:i+3]
        if len(vals) >= 2:
            results_cf.append({'Stkcd': stkcd,
                              'year': g['year'].iloc[i],
                              'CashFlowVol': vals.std()})

cfvol = pd.DataFrame(results_cf)
panel = panel.merge(cfvol, on=['Stkcd', 'year'], how='left')
print(f"  CashFlowVol: {panel['CashFlowVol'].notna().sum():,} obs, "
      f"mean={panel['CashFlowVol'].mean():.4f}")

# ============================================================
# 15. MgmtHold - 管理层持股比例
# ============================================================
print("\n" + "=" * 70)
print("15. 构造管理层持股 (MgmtHold)")
print("=" * 70)

gov = pd.read_parquet(f"{BASE}/data_parquet/governance.parquet")
gov['Reptdt'] = gov['Reptdt'].astype(str)
gov_annual = gov[gov['Reptdt'].str.endswith('12-31')].copy()
gov_annual['year'] = pd.to_datetime(gov_annual['Reptdt']).dt.year
gov_annual['Stkcd'] = gov_annual['Stkcd'].astype(int)
gov_annual['MgmtHold'] = pd.to_numeric(gov_annual['ManagerHoldsharesRatio'],
                                        errors='coerce')
gov_annual = gov_annual.drop_duplicates(['Stkcd', 'year'], keep='last')

panel = panel.merge(gov_annual[['Stkcd', 'year', 'MgmtHold']],
                    on=['Stkcd', 'year'], how='left')
print(f"  MgmtHold: {panel['MgmtHold'].notna().sum():,} obs, "
      f"mean={panel['MgmtHold'].mean():.4f}")

# ============================================================
# Winsorize + 回归
# ============================================================
print("\n" + "=" * 70)
print("Winsorize 并准备回归")
print("=" * 70)

def winsorize(s):
    lo, hi = s.quantile([0.01, 0.99])
    return s.clip(lo, hi)

new_vars = ['AuditFee', 'RatingDisp', 'ReportFreq', 'RetAutoCorr',
            'CashFlowVol', 'MgmtHold']

controls = ['Size', 'Lev', 'ROA', 'TobinQ', 'Age', 'Growth', 'BoardSize',
            'IndepRatio', 'Dual', 'Top1Share', 'SOE', 'InstHold', 'Amihud',
            'Analyst', 'AuditType']

for v in ['PriceDelay', 'DU_kw'] + new_vars + controls:
    if v in panel.columns and panel[v].notna().any() and panel[v].dtype in ['float64', 'float32', 'int64']:
        mask = panel[v].notna()
        if mask.sum() > 100:
            panel.loc[mask, v] = winsorize(panel.loc[mask, v])

reg = panel.dropna(subset=['PriceDelay', 'DU_kw'] + controls).copy()
reg['Stkcd_str'] = reg['Stkcd'].astype(str)
reg['year_str'] = reg['year'].astype(str)
print(f"回归样本: N={len(reg):,}")

for v in new_vars:
    n = reg[v].notna().sum()
    print(f"  {v}: {n:,} obs ({n/len(reg)*100:.1f}%)")

# ============================================================
# 江艇两步法
# ============================================================
print("\n" + "=" * 70)
print("江艇两步法机制回归 (Round 2)")
print("=" * 70)

ctrl_str = " + ".join(controls)

def sig_stars(p):
    if p is None or pd.isna(p): return ""
    if p < 0.01: return "***"
    if p < 0.05: return "**"
    if p < 0.1: return "*"
    return ""

mech_map = {
    'AuditFee':    ('审计费用(对数)',     '审计质量/可验证性'),
    'RatingDisp':  ('评级分歧度',        '投资建议共识'),
    'ReportFreq':  ('研报频次(对数)',     '信息生产频率'),
    'RetAutoCorr': ('收益率自相关',       '价格调整速度'),
    'CashFlowVol': ('现金流波动率',       '经营不确定性'),
    'MgmtHold':    ('管理层持股比例',     '代理成本'),
}

results_all = {}

for var, (cn_name, source) in mech_map.items():
    print(f"\n--- {var} ({cn_name}) [{source}] ---")
    mech_controls = [c for c in controls if c != var]
    mech_ctrl_str = " + ".join(mech_controls)

    # 同期
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
        t_stat = coef / se
        print(f"  同期:   coef={coef:.6f}, se={se:.6f}, t={t_stat:.2f}, "
              f"p={pval:.4f}{stars}, N={m._N:,}, R2={m._r2:.3f}")
        results_all[f'{var}_concurrent'] = {
            'var': var, 'cn_name': cn_name, 'source': source,
            'type': 'concurrent',
            'coef': coef, 'se': se, 't': t_stat, 'p': pval, 'sig': stars,
            'N': m._N, 'R2': m._r2
        }
    except Exception as e:
        print(f"  同期回归失败: {e}")

    # 滞后
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
        t_l = coef_l / se_l
        print(f"  滞后:   coef={coef_l:.6f}, se={se_l:.6f}, t={t_l:.2f}, "
              f"p={pval_l:.4f}{stars_l}, N={m_lag._N:,}, R2={m_lag._r2:.3f}")
        results_all[f'{var}_lagged'] = {
            'var': var, 'cn_name': cn_name, 'source': source,
            'type': 'lagged',
            'coef': coef_l, 'se': se_l, 't': t_l, 'p': pval_l, 'sig': stars_l,
            'N': m_lag._N, 'R2': m_lag._r2
        }
    except Exception as e:
        print(f"  滞后回归失败: {e}")

# ============================================================
# 汇总
# ============================================================
print("\n" + "=" * 70)
print("汇总 (Round 2)")
print("=" * 70)

print(f"\n{'变量':<14} {'中文名':<14} {'类型':<12} {'系数':>12} {'t值':>8} {'显著性':>6} {'N':>8}")
print("-" * 78)
for key, r in sorted(results_all.items()):
    print(f"{r['var']:<14} {r['cn_name']:<14} {r['type']:<12} "
          f"{r['coef']:>12.6f} {r['t']:>8.2f} {r['sig']:>6} {r['N']:>8,}")

os.makedirs(f"{BASE}/results/new_mechanisms_9", exist_ok=True)
with open(f"{BASE}/results/new_mechanisms_9/results_round2.json", 'w') as f:
    json.dump(results_all, f, indent=2, default=str, ensure_ascii=False)

print(f"\n结果已保存至 results/new_mechanisms_9/results_round2.json")
