#!/usr/bin/env python3
"""
内生性检验 v4 — 修复Bartik退化 + N用feols._N + lag-IV补诊断
=============================================================
修复记录:
  v3→v4: Bartik构造改为 base_j * growth_excl_j (leave-out行业增长)
         N统一用feols._N而非len(df)
         lag-IV补AR/DWH诊断
口径: 同期DU_kw, 样本筛选与v3_concurrent完全一致
"""

import os, json
import pandas as pd
import numpy as np
import pyfixest as pf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data_parquet")
RES_DIR = os.path.join(BASE_DIR, "results/endogeneity_v4")
os.makedirs(RES_DIR, exist_ok=True)

# ── 数据加载: 完全复制 v3_concurrent ──
panel = pd.read_parquet(f"{DATA_DIR}/panel.parquet")
ar_feat = pd.read_parquet(f"{DATA_DIR}/annual_report_features.parquet")
panel = panel.merge(
    ar_feat[['Stkcd','year','kw_total','kw_per10k','substantive_count']],
    on=['Stkcd','year'], how='left')

fi = pd.read_parquet(f"{DATA_DIR}/firm_info.parquet",
                     columns=['Symbol','EndDate','IndustryCodeC','LISTINGSTATE'])
fi = fi.rename(columns={'Symbol':'Stkcd'})
fi['EndDate'] = pd.to_datetime(fi['EndDate'])
fi['year'] = fi['EndDate'].dt.year
fi = fi.sort_values(['Stkcd','year','EndDate']).drop_duplicates(
    subset=['Stkcd','year'], keep='last')
panel = panel.merge(fi[['Stkcd','year','IndustryCodeC','LISTINGSTATE']],
                    on=['Stkcd','year'], how='left')

# ── 样本筛选: 与v3完全一致 ──
mask_fin = panel['IndustryCodeC'].str.startswith('J', na=False)
mask_st  = panel['LISTINGSTATE'].isin(['ST','*ST'])
mask_new = panel['Age'] <= 0
panel = panel[~mask_fin & ~mask_st & ~mask_new]

# ── 变量构造 ──
panel['DU_kw'] = panel['kw_per10k']
panel['Ind2'] = panel['IndustryCodeC'].str[:3]
panel['IndYear'] = panel['Ind2'].astype(str) + '_' + panel['year'].astype(str)

controls = ['Size','Lev','ROA','TobinQ','Age','Growth','BoardSize',
            'IndepRatio','Dual','Top1Share','SOE','InstHold','Amihud',
            'Analyst','AuditType']
ctrl_str = ' + '.join(controls)

def winsorize(s, lo=0.01, hi=0.99):
    q = s.quantile([lo, hi])
    return s.clip(q.iloc[0], q.iloc[1])

for v in ['PriceDelay','DU_kw','Lev','ROA','Growth','Size','TobinQ','Age',
          'BoardSize','IndepRatio','Top1Share','InstHold','Amihud','Analyst']:
    if v in panel.columns:
        panel[v] = winsorize(panel[v])

df = panel.dropna(subset=['PriceDelay','DU_kw'] + controls).copy()
df['Stkcd_str'] = df['Stkcd'].astype(str)
df['year_str']  = df['year'].astype(str)

def sig(p):
    if p < 0.01: return '***'
    if p < 0.05: return '**'
    if p < 0.1:  return '*'
    return ''

def get_N(model):
    """从feols对象取实际回归样本量"""
    return getattr(model, '_N', None) or len(model.resid())

def do_dwh(df_sub, first_model, endog='DU_kw', iv_name='DU_kw_peer'):
    """DWH内生性检验: 一阶段残差加入OLS, 检验残差系数"""
    try:
        resid = first_model.resid()
        if len(resid) != len(df_sub):
            # fallback: 手动OLS
            from sklearn.linear_model import LinearRegression
            X = df_sub[[iv_name] + controls].values
            y = df_sub[endog].values
            reg = LinearRegression().fit(X, y)
            df_sub = df_sub.copy()
            df_sub['_resid1'] = y - reg.predict(X)
        else:
            df_sub = df_sub.copy()
            df_sub['_resid1'] = resid
        m = pf.feols(f"PriceDelay ~ {endog} + _resid1 + {ctrl_str} | Stkcd_str + year_str",
                     data=df_sub, vcov={'CRV1': 'IndYear'})
        t = m.coef()['_resid1'] / m.se()['_resid1']
        p = m.pvalue()['_resid1']
        return t**2, p
    except:
        return np.nan, np.nan

print(f"预回归样本: {len(df)}")
results = {}

# ============================================================
#  1. OLS基准
# ============================================================
print("="*60)
print("1. OLS基准")
print("="*60)

ols = pf.feols(f"PriceDelay ~ DU_kw + {ctrl_str} | Stkcd_str + year_str",
               data=df, vcov={'CRV1': 'IndYear'})
N_ols = get_N(ols)
ols_coef = ols.coef()['DU_kw']
ols_se   = ols.se()['DU_kw']
ols_t    = ols_coef / ols_se
ols_p    = ols.pvalue()['DU_kw']
ols_r2   = ols._r2
print(f"  coef={ols_coef:.6f}, t={ols_t:.2f}, p={ols_p:.6f}, N={N_ols}")
results['ols'] = {'coef':ols_coef, 'se':ols_se, 't':ols_t, 'p':ols_p,
                  'N':N_ols, 'R2':ols_r2}
print()

# ============================================================
#  2. 同行业均值IV
# ============================================================
print("="*60)
print("2. 同行业均值IV")
print("="*60)

iy = df.groupby(['Ind2','year'])['DU_kw'].agg(['sum','count'])
iy.columns = ['iy_sum','iy_count']
df = df.merge(iy, on=['Ind2','year'], how='left')
df['DU_kw_peer'] = (df['iy_sum'] - df['DU_kw']) / (df['iy_count'] - 1)
df_iv = df.dropna(subset=['DU_kw_peer']).copy()

# 一阶段
first = pf.feols(f"DU_kw ~ DU_kw_peer + {ctrl_str} | Stkcd_str + year_str",
                 data=df_iv, vcov={'CRV1': 'IndYear'})
fs_t = first.coef()['DU_kw_peer'] / first.se()['DU_kw_peer']
kp_F = fs_t**2

# 2SLS
iv_peer = pf.feols(f"PriceDelay ~ 1 + {ctrl_str} | Stkcd_str + year_str | DU_kw ~ DU_kw_peer",
                   data=df_iv, vcov={'CRV1': 'IndYear'})
N_peer = get_N(iv_peer)
iv_c = iv_peer.coef()['DU_kw']
iv_s = iv_peer.se()['DU_kw']
iv_t = iv_c / iv_s
iv_p = iv_peer.pvalue()['DU_kw']

# AR
rf = pf.feols(f"PriceDelay ~ DU_kw_peer + {ctrl_str} | Stkcd_str + year_str",
              data=df_iv, vcov={'CRV1': 'IndYear'})
rf_t = rf.coef()['DU_kw_peer'] / rf.se()['DU_kw_peer']
ar_F = rf_t**2; ar_p = rf.pvalue()['DU_kw_peer']

# DWH
dwh_F, dwh_p = do_dwh(df_iv, first, 'DU_kw', 'DU_kw_peer')

print(f"  2SLS: coef={iv_c:.6f}, t={iv_t:.2f}, p={iv_p:.4f}, N={N_peer}")
print(f"  KP F={kp_F:.2f}, AR F={ar_F:.2f}(p={ar_p:.4f}), DWH F={dwh_F:.2f}(p={dwh_p:.4f})")

results['peer_iv'] = {
    'coef':iv_c, 'se':iv_s, 't':iv_t, 'p':iv_p, 'N':N_peer,
    'KP_F':kp_F, 'AR_F':ar_F, 'AR_p':ar_p, 'DWH_F':dwh_F, 'DWH_p':dwh_p}
print()

# ============================================================
#  3. Bartik IV (修复: leave-out行业增长)
# ============================================================
print("="*60)
print("3. Bartik IV (修正构造)")
print("="*60)

# 正确的Bartik: base_j_2011 * growth_excl_j_t
# growth_excl_j_t = (全国DU均值_t - 行业j均值_t*行业j占比) / (1-行业j占比)
# 简化: 用全国排除行业j的DU均值增长率

base_year = df['year'].min()

# 基期行业DU均值 (份额/暴露度)
ind_base = df[df['year']==base_year].groupby('Ind2')['DU_kw'].mean()

# 每年每行业的DU均值 + 全国排除本行业的DU均值
yearly_stats = []
for yr in df['year'].unique():
    yr_data = df[df['year']==yr]
    national_mean = yr_data['DU_kw'].mean()
    national_sum  = yr_data['DU_kw'].sum()
    national_n    = len(yr_data)
    for ind in yr_data['Ind2'].unique():
        ind_data = yr_data[yr_data['Ind2']==ind]
        ind_sum = ind_data['DU_kw'].sum()
        ind_n = len(ind_data)
        # leave-out均值: 全国排除本行业
        if national_n - ind_n > 0:
            excl_mean = (national_sum - ind_sum) / (national_n - ind_n)
        else:
            excl_mean = np.nan
        yearly_stats.append({'Ind2':ind, 'year':yr, 'DU_excl_mean':excl_mean})

excl_df = pd.DataFrame(yearly_stats)

# 基期的排除均值
excl_base = excl_df[excl_df['year']==base_year].set_index('Ind2')['DU_excl_mean']

# 增长率: excl_mean_t / excl_mean_base
excl_df['excl_base'] = excl_df['Ind2'].map(excl_base)
excl_df['growth_excl'] = excl_df['DU_excl_mean'] / excl_df['excl_base'].clip(lower=0.001)

# Bartik IV = 基期行业暴露度 * 排除本行业的全国增长率
excl_df['base_exposure'] = excl_df['Ind2'].map(ind_base)
excl_df['bartik_iv'] = excl_df['base_exposure'] * excl_df['growth_excl']

df_bk = df.merge(excl_df[['Ind2','year','bartik_iv']], on=['Ind2','year'], how='left')
df_bk = df_bk.dropna(subset=['bartik_iv']).copy()

corr_bk = df_bk[['DU_kw','bartik_iv']].corr().iloc[0,1]
print(f"  IV-X corr: {corr_bk:.4f}, N_pre={len(df_bk)}")

# 验证Bartik不再等于行业均值
corr_peer = df_bk[['DU_kw_peer','bartik_iv']].corr().iloc[0,1]
print(f"  Bartik-Peer corr: {corr_peer:.4f} (应远低于1.0)")

# 一阶段
first_b = pf.feols(f"DU_kw ~ bartik_iv + {ctrl_str} | Stkcd_str + year_str",
                   data=df_bk, vcov={'CRV1': 'IndYear'})
fb_t = first_b.coef()['bartik_iv'] / first_b.se()['bartik_iv']
fb_F = fb_t**2

# 2SLS
iv_bk = pf.feols(f"PriceDelay ~ 1 + {ctrl_str} | Stkcd_str + year_str | DU_kw ~ bartik_iv",
                 data=df_bk, vcov={'CRV1': 'IndYear'})
N_bk = get_N(iv_bk)
bk_c = iv_bk.coef()['DU_kw']
bk_s = iv_bk.se()['DU_kw']
bk_t = bk_c / bk_s
bk_p = iv_bk.pvalue()['DU_kw']

# AR
rf_b = pf.feols(f"PriceDelay ~ bartik_iv + {ctrl_str} | Stkcd_str + year_str",
                data=df_bk, vcov={'CRV1': 'IndYear'})
rfb_t = rf_b.coef()['bartik_iv'] / rf_b.se()['bartik_iv']
ar_b_F = rfb_t**2; ar_b_p = rf_b.pvalue()['bartik_iv']

# DWH
dwh_b_F, dwh_b_p = do_dwh(df_bk, first_b, 'DU_kw', 'bartik_iv')

print(f"  2SLS: coef={bk_c:.6f}, t={bk_t:.2f}, p={bk_p:.4f}, N={N_bk}")
print(f"  KP F={fb_F:.2f}, AR F={ar_b_F:.2f}(p={ar_b_p:.4f}), DWH F={dwh_b_F:.2f}(p={dwh_b_p:.4f})")

results['bartik_iv'] = {
    'coef':bk_c, 'se':bk_s, 't':bk_t, 'p':bk_p, 'N':N_bk,
    'KP_F':fb_F, 'AR_F':ar_b_F, 'AR_p':ar_b_p, 'DWH_F':dwh_b_F, 'DWH_p':dwh_b_p}
print()

# ============================================================
#  4. DU_{t-1} + 完整诊断
# ============================================================
print("="*60)
print("4. DU_{t-1} 滞后一期")
print("="*60)

df_lag = df.sort_values(['Stkcd','year']).copy()
df_lag['DU_kw_lag'] = df_lag.groupby('Stkcd')['DU_kw'].shift(1)
df_lag = df_lag.dropna(subset=['DU_kw_lag'])

# OLS lag
ols_lag = pf.feols(f"PriceDelay ~ DU_kw_lag + {ctrl_str} | Stkcd_str + year_str",
                   data=df_lag, vcov={'CRV1': 'IndYear'})
N_lag_ols = get_N(ols_lag)
lc = ols_lag.coef()['DU_kw_lag']; ls = ols_lag.se()['DU_kw_lag']
lt = lc/ls; lp = ols_lag.pvalue()['DU_kw_lag']
print(f"  OLS(t-1): coef={lc:.6f}, t={lt:.2f}, p={lp:.6f}, N={N_lag_ols}")
results['lag_ols'] = {'coef':lc, 'se':ls, 't':lt, 'p':lp, 'N':N_lag_ols}

# IV lag: peer lag
iy_lag = df_lag.groupby(['Ind2','year'])['DU_kw_lag'].agg(['sum','count'])
iy_lag.columns = ['iyl_sum','iyl_count']
df_lag = df_lag.merge(iy_lag, on=['Ind2','year'], how='left')
df_lag['peer_lag'] = (df_lag['iyl_sum'] - df_lag['DU_kw_lag']) / (df_lag['iyl_count'] - 1)
df_lag_iv = df_lag.dropna(subset=['peer_lag']).copy()

try:
    # 一阶段
    first_l = pf.feols(f"DU_kw_lag ~ peer_lag + {ctrl_str} | Stkcd_str + year_str",
                       data=df_lag_iv, vcov={'CRV1': 'IndYear'})
    fl_t = first_l.coef()['peer_lag'] / first_l.se()['peer_lag']
    fl_F = fl_t**2

    # 2SLS
    iv_l = pf.feols(f"PriceDelay ~ 1 + {ctrl_str} | Stkcd_str + year_str | DU_kw_lag ~ peer_lag",
                    data=df_lag_iv, vcov={'CRV1': 'IndYear'})
    N_lag_iv = get_N(iv_l)
    ilc = iv_l.coef()['DU_kw_lag']; ils = iv_l.se()['DU_kw_lag']
    ilt = ilc/ils; ilp = iv_l.pvalue()['DU_kw_lag']

    # AR
    rf_l = pf.feols(f"PriceDelay ~ peer_lag + {ctrl_str} | Stkcd_str + year_str",
                    data=df_lag_iv, vcov={'CRV1': 'IndYear'})
    rfl_t = rf_l.coef()['peer_lag'] / rf_l.se()['peer_lag']
    ar_l_F = rfl_t**2; ar_l_p = rf_l.pvalue()['peer_lag']

    # DWH
    dwh_l_F, dwh_l_p = do_dwh(df_lag_iv, first_l, 'DU_kw_lag', 'peer_lag')

    print(f"  2SLS(t-1): coef={ilc:.6f}, t={ilt:.2f}, p={ilp:.4f}, N={N_lag_iv}")
    print(f"  KP F={fl_F:.2f}, AR F={ar_l_F:.2f}(p={ar_l_p:.4f}), DWH F={dwh_l_F:.2f}(p={dwh_l_p:.4f})")

    results['lag_iv'] = {
        'coef':ilc, 'se':ils, 't':ilt, 'p':ilp, 'N':N_lag_iv,
        'KP_F':fl_F, 'AR_F':ar_l_F, 'AR_p':ar_l_p, 'DWH_F':dwh_l_F, 'DWH_p':dwh_l_p}
except Exception as e:
    print(f"  IV(lag) error: {e}")
print()

# ============================================================
#  5. Oster (2019)
# ============================================================
print("="*60)
print("5. Oster (2019)")
print("="*60)

ols_short = pf.feols(f"PriceDelay ~ DU_kw | Stkcd_str + year_str",
                     data=df, vcov={'CRV1': 'IndYear'})
bs = ols_short.coef()['DU_kw']; rs = ols_short._r2
bf = ols_coef; rf_ = ols_r2

r_13 = min(1.0, 1.3*rf_)
r_con = 2*rf_ - rs

def oster_d(bs,rs,bf,rf,rm):
    d = (bf*(rm-rf)) / ((bs-bf)*(rf-rs)) if abs((bs-bf)*(rf-rs))>1e-15 else float('inf')
    return d
def oster_b(bs,rs,bf,rf,rm):
    return bf - (bs-bf)*(rm-rf)/(rf-rs) if abs(rf-rs)>1e-15 else bf

d13  = oster_d(bs,rs,bf,rf_,r_13);  ba13  = oster_b(bs,rs,bf,rf_,r_13)
dcon = oster_d(bs,rs,bf,rf_,r_con); bacon = oster_b(bs,rs,bf,rf_,r_con)

print(f"  β_short={bs:.6f}, R2_short={rs:.4f}")
print(f"  β_full={bf:.6f}, R2_full={rf_:.4f}")
print(f"  1.3R: δ*={d13:.1f}, β_adj={ba13:.6f} {'(同号)' if ba13*bf>0 else '(变号)'}")
print(f"  保守:  δ*={dcon:.1f}, β_adj={bacon:.6f} {'(同号)' if bacon*bf>0 else '(变号)'}")

results['oster'] = {
    'beta_short':bs, 'r2_short':rs, 'beta_full':bf, 'r2_full':rf_,
    'r2_max_13':r_13, 'delta_13':d13, 'beta_adj_13':ba13,
    'r2_max_con':r_con, 'delta_con':dcon, 'beta_adj_con':bacon}
print()

# ============================================================
#  汇总
# ============================================================
print("="*60)
print("汇总")
print("="*60)
rows = []
for label, key in [('OLS基准','ols'),('Peer IV','peer_iv'),('Bartik IV','bartik_iv'),
                    ('OLS(t-1)','lag_ols'),('IV(t-1)','lag_iv')]:
    if key in results:
        r = results[key]
        fs = f", KP={r['KP_F']:.0f}" if 'KP_F' in r else ''
        ar = f", AR_p={r['AR_p']:.3f}" if 'AR_p' in r else ''
        dw = f", DWH_p={r['DWH_p']:.3f}" if 'DWH_p' in r else ''
        print(f"  {label:12s}: β={r['coef']:.6f}, t={r['t']:.2f}{sig(r['p'])}, N={r['N']}{fs}{ar}{dw}")
        rows.append({**{'方法':label}, **{k:v for k,v in r.items()}})

pd.DataFrame(rows).to_csv(f"{RES_DIR}/endogeneity_v4_summary.csv", index=False, encoding='utf-8-sig')

def conv(o):
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.integer,)):  return int(o)
    return o
with open(f"{RES_DIR}/endogeneity_v4_results.json",'w') as f:
    json.dump({k:{kk:conv(vv) for kk,vv in v.items()} for k,v in results.items()},
              f, indent=2, ensure_ascii=False)
print(f"\n保存至 {RES_DIR}/")
