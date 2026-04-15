#!/usr/bin/env python3
"""
内生性检验 v3 — 严格对齐v3_concurrent基准回归口径
=============================================
数据处理: 完全复制 run_regression_v3_concurrent.py 的加载、合并、筛选逻辑
目标样本: N=43,847 (与基准回归一致)
"""

import os, sys, json
import pandas as pd
import numpy as np
import pyfixest as pf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data_parquet")
RES_DIR = os.path.join(BASE_DIR, "results/endogeneity_v3")
os.makedirs(RES_DIR, exist_ok=True)

# ============================================================
#  数据加载 — 完全复制 v3_concurrent 逻辑
# ============================================================
panel = pd.read_parquet(f"{DATA_DIR}/panel.parquet")
ar_feat = pd.read_parquet(f"{DATA_DIR}/annual_report_features.parquet")

panel = panel.merge(
    ar_feat[['Stkcd', 'year', 'kw_total', 'kw_per10k', 'substantive_count']],
    on=['Stkcd', 'year'], how='left'
)

# 行业信息: 按(Stkcd, year)匹配，取最新EndDate (与v3一致)
fi = pd.read_parquet(f"{DATA_DIR}/firm_info.parquet",
                     columns=['Symbol', 'EndDate', 'IndustryCodeC', 'LISTINGSTATE'])
fi = fi.rename(columns={'Symbol': 'Stkcd'})
fi['EndDate'] = pd.to_datetime(fi['EndDate'])
fi['year'] = fi['EndDate'].dt.year
fi = fi.sort_values(['Stkcd', 'year', 'EndDate']).drop_duplicates(
    subset=['Stkcd', 'year'], keep='last')
panel = panel.merge(fi[['Stkcd', 'year', 'IndustryCodeC', 'LISTINGSTATE']],
                    on=['Stkcd', 'year'], how='left')

# ============================================================
#  样本筛选 — 与v3完全一致
# ============================================================
mask_fin = panel['IndustryCodeC'].str.startswith('J', na=False)
mask_st = panel['LISTINGSTATE'].isin(['ST', '*ST'])
mask_new = panel['Age'] <= 0
panel = panel[~mask_fin & ~mask_st & ~mask_new]

# ============================================================
#  变量构造 — 同期，不做滞后
# ============================================================
panel['DU_kw'] = panel['kw_per10k']
panel['DU_kw_ln'] = np.log1p(panel['kw_total'])
panel['Ind2'] = panel['IndustryCodeC'].str[:3]
panel['IndYear'] = panel['Ind2'].astype(str) + '_' + panel['year'].astype(str)

controls = ['Size', 'Lev', 'ROA', 'TobinQ', 'Age', 'Growth', 'BoardSize',
            'IndepRatio', 'Dual', 'Top1Share', 'SOE', 'InstHold', 'Amihud',
            'Analyst', 'AuditType']
ctrl_str = ' + '.join(controls)

# Winsorize (与v3一致)
def winsorize(s, lower=0.01, upper=0.99):
    q_lo, q_hi = s.quantile([lower, upper])
    return s.clip(q_lo, q_hi)

cont_vars = ['PriceDelay', 'DU_kw', 'DU_kw_ln',
             'Lev', 'ROA', 'Growth', 'Size', 'TobinQ', 'Age',
             'BoardSize', 'IndepRatio', 'Top1Share', 'InstHold', 'Amihud', 'Analyst']
for v in cont_vars:
    if v in panel.columns:
        panel[v] = winsorize(panel[v])

# 筛选有效样本
df = panel.dropna(subset=['PriceDelay', 'DU_kw'] + controls).copy()
df['Stkcd_str'] = df['Stkcd'].astype(str)
df['year_str'] = df['year'].astype(str)

print(f"样本: N={len(df)}, Firms={df['Stkcd'].nunique()}, Years={df['year'].nunique()}")
print(f"  (基准回归v3应为 N=43,847)")
print()

results = {}

# ============================================================
#  1. OLS基准
# ============================================================
print("=" * 60)
print("1. OLS基准")
print("=" * 60)

ols = pf.feols(f"PriceDelay ~ DU_kw + {ctrl_str} | Stkcd_str + year_str",
               data=df, vcov={'CRV1': 'IndYear'})
ols_coef = ols.coef()['DU_kw']
ols_se = ols.se()['DU_kw']
ols_t = ols_coef / ols_se
ols_p = ols.pvalue()['DU_kw']
ols_r2 = ols._r2
print(f"  coef={ols_coef:.6f}, t={ols_t:.2f}, p={ols_p:.6f}, N={len(df)}")
print(f"  R2={ols_r2:.4f}")
results['ols'] = {'coef': ols_coef, 'se': ols_se, 't': ols_t, 'p': ols_p,
                  'N': len(df), 'R2': ols_r2}
print()

# ============================================================
#  2. 同行业均值IV
# ============================================================
print("=" * 60)
print("2. 同行业均值IV")
print("=" * 60)

# 用Ind2 (证监会二级) 构造peer mean
iy = df.groupby(['Ind2', 'year'])['DU_kw'].agg(['sum', 'count'])
iy.columns = ['iy_sum', 'iy_count']
df = df.merge(iy, on=['Ind2', 'year'], how='left')
df['DU_kw_peer'] = (df['iy_sum'] - df['DU_kw']) / (df['iy_count'] - 1)
# 单企业行业年份组peer为NaN，剔除
df_iv = df.dropna(subset=['DU_kw_peer']).copy()

print(f"  IV样本: N={len(df_iv)} (剔除singleton行业年份后)")
corr = df_iv[['DU_kw', 'DU_kw_peer']].corr().iloc[0, 1]
print(f"  IV-X correlation: {corr:.4f}")

# 一阶段
first = pf.feols(f"DU_kw ~ DU_kw_peer + {ctrl_str} | Stkcd_str + year_str",
                 data=df_iv, vcov={'CRV1': 'IndYear'})
fs_coef = first.coef()['DU_kw_peer']
fs_t = fs_coef / first.se()['DU_kw_peer']
kp_F = fs_t ** 2  # 恰好识别: KP rk F = t^2
print(f"  一阶段: coef={fs_coef:.4f}, t={fs_t:.2f}")
print(f"  KP rk Wald F: {kp_F:.2f}")

# 2SLS
iv_peer = pf.feols(f"PriceDelay ~ 1 + {ctrl_str} | Stkcd_str + year_str | DU_kw ~ DU_kw_peer",
                   data=df_iv, vcov={'CRV1': 'IndYear'})
iv_coef = iv_peer.coef()['DU_kw']
iv_se = iv_peer.se()['DU_kw']
iv_t = iv_coef / iv_se
iv_p = iv_peer.pvalue()['DU_kw']
print(f"  2SLS: coef={iv_coef:.6f}, t={iv_t:.2f}, p={iv_p:.4f}")

# Anderson-Rubin (reduced form)
rf = pf.feols(f"PriceDelay ~ DU_kw_peer + {ctrl_str} | Stkcd_str + year_str",
              data=df_iv, vcov={'CRV1': 'IndYear'})
rf_t = rf.coef()['DU_kw_peer'] / rf.se()['DU_kw_peer']
rf_p = rf.pvalue()['DU_kw_peer']
ar_F = rf_t ** 2
print(f"  Anderson-Rubin F: {ar_F:.2f}, p={rf_p:.4f}")

# DWH: 用predict对齐样本量
try:
    resid_arr = first.resid()
    # 对齐: pyfixest可能丢singleton FE obs
    if len(resid_arr) == len(df_iv):
        df_iv['resid_first'] = resid_arr
    else:
        # 用fitted_values对齐
        fitted = first.predict()
        if len(fitted) == len(df_iv):
            df_iv['resid_first'] = df_iv['DU_kw'].values - fitted
        else:
            # 最后手段: 手动OLS拟合残差
            from sklearn.linear_model import LinearRegression
            X_tmp = df_iv[['DU_kw_peer'] + controls].values
            y_tmp = df_iv['DU_kw'].values
            mask_ok = ~np.isnan(X_tmp).any(axis=1) & ~np.isnan(y_tmp)
            reg_tmp = LinearRegression().fit(X_tmp[mask_ok], y_tmp[mask_ok])
            df_iv['resid_first'] = y_tmp - reg_tmp.predict(X_tmp)
    dwh_reg = pf.feols(f"PriceDelay ~ DU_kw + resid_first + {ctrl_str} | Stkcd_str + year_str",
                       data=df_iv, vcov={'CRV1': 'IndYear'})
    dwh_t = dwh_reg.coef()['resid_first'] / dwh_reg.se()['resid_first']
    dwh_p = dwh_reg.pvalue()['resid_first']
    dwh_F = dwh_t ** 2
    print(f"  DWH F: {dwh_F:.2f}, p={dwh_p:.4f}")
except Exception as e:
    print(f"  DWH error: {e}")
    dwh_F, dwh_p = np.nan, np.nan

results['peer_iv'] = {
    'coef': iv_coef, 'se': iv_se, 't': iv_t, 'p': iv_p,
    'N': len(df_iv), 'KP_F': kp_F, 'AR_F': ar_F, 'AR_p': rf_p,
    'DWH_F': dwh_F, 'DWH_p': dwh_p,
    'first_stage_coef': fs_coef, 'first_stage_t': fs_t
}
print()

# ============================================================
#  3. Bartik IV
# ============================================================
print("=" * 60)
print("3. Bartik IV")
print("=" * 60)

base_year = df['year'].min()
ind_base = df[df['year'] == base_year].groupby('Ind2')['DU_kw'].mean()
ind_year_mean = df.groupby(['Ind2', 'year'])['DU_kw'].mean()

# 构造增长率
bartik_data = []
for (ind, yr), mean_val in ind_year_mean.items():
    if ind in ind_base.index and ind_base[ind] > 0.001:
        bartik_data.append({'Ind2': ind, 'year': yr,
                           'bartik_iv': ind_base[ind] * (mean_val / ind_base[ind])})
bartik_df = pd.DataFrame(bartik_data)
df_bk = df.merge(bartik_df, on=['Ind2', 'year'], how='left')
df_bk = df_bk.dropna(subset=['bartik_iv']).copy()

print(f"  Bartik样本: N={len(df_bk)}")
corr_b = df_bk[['DU_kw', 'bartik_iv']].corr().iloc[0, 1]
print(f"  IV-X correlation: {corr_b:.4f}")

# 一阶段
first_b = pf.feols(f"DU_kw ~ bartik_iv + {ctrl_str} | Stkcd_str + year_str",
                   data=df_bk, vcov={'CRV1': 'IndYear'})
fb_t = first_b.coef()['bartik_iv'] / first_b.se()['bartik_iv']
fb_F = fb_t ** 2
print(f"  KP rk Wald F: {fb_F:.2f}")

# 2SLS
iv_bk = pf.feols(f"PriceDelay ~ 1 + {ctrl_str} | Stkcd_str + year_str | DU_kw ~ bartik_iv",
                 data=df_bk, vcov={'CRV1': 'IndYear'})
bk_coef = iv_bk.coef()['DU_kw']
bk_se = iv_bk.se()['DU_kw']
bk_t = bk_coef / bk_se
bk_p = iv_bk.pvalue()['DU_kw']
print(f"  2SLS: coef={bk_coef:.6f}, t={bk_t:.2f}, p={bk_p:.4f}")

# AR
rf_b = pf.feols(f"PriceDelay ~ bartik_iv + {ctrl_str} | Stkcd_str + year_str",
                data=df_bk, vcov={'CRV1': 'IndYear'})
rfb_t = rf_b.coef()['bartik_iv'] / rf_b.se()['bartik_iv']
rfb_p = rf_b.pvalue()['bartik_iv']
ar_b_F = rfb_t ** 2
print(f"  Anderson-Rubin F: {ar_b_F:.2f}, p={rfb_p:.4f}")

# DWH
try:
    resid_b_arr = first_b.resid()
    if len(resid_b_arr) == len(df_bk):
        df_bk['resid_b'] = resid_b_arr
    else:
        fitted_b = first_b.predict()
        if len(fitted_b) == len(df_bk):
            df_bk['resid_b'] = df_bk['DU_kw'].values - fitted_b
        else:
            from sklearn.linear_model import LinearRegression
            X_b = df_bk[['bartik_iv'] + controls].values
            y_b = df_bk['DU_kw'].values
            reg_b = LinearRegression().fit(X_b, y_b)
            df_bk['resid_b'] = y_b - reg_b.predict(X_b)
    dwh_b = pf.feols(f"PriceDelay ~ DU_kw + resid_b + {ctrl_str} | Stkcd_str + year_str",
                     data=df_bk, vcov={'CRV1': 'IndYear'})
    dwhb_t = dwh_b.coef()['resid_b'] / dwh_b.se()['resid_b']
    dwhb_p = dwh_b.pvalue()['resid_b']
    dwhb_F = dwhb_t ** 2
    print(f"  DWH F: {dwhb_F:.2f}, p={dwhb_p:.4f}")
except Exception as e:
    print(f"  DWH error: {e}")
    dwhb_F, dwhb_p = np.nan, np.nan

results['bartik_iv'] = {
    'coef': bk_coef, 'se': bk_se, 't': bk_t, 'p': bk_p,
    'N': len(df_bk), 'KP_F': fb_F, 'AR_F': ar_b_F, 'AR_p': rfb_p,
    'DWH_F': dwhb_F, 'DWH_p': dwhb_p
}
print()

# ============================================================
#  4. 滞后一期
# ============================================================
print("=" * 60)
print("4. DU_{t-1} 滞后一期")
print("=" * 60)

df_lag = df.sort_values(['Stkcd', 'year']).copy()
df_lag['DU_kw_lag'] = df_lag.groupby('Stkcd')['DU_kw'].shift(1)
df_lag = df_lag.dropna(subset=['DU_kw_lag'])

# OLS
ols_lag = pf.feols(f"PriceDelay ~ DU_kw_lag + {ctrl_str} | Stkcd_str + year_str",
                   data=df_lag, vcov={'CRV1': 'IndYear'})
lag_coef = ols_lag.coef()['DU_kw_lag']
lag_se = ols_lag.se()['DU_kw_lag']
lag_t = lag_coef / lag_se
lag_p = ols_lag.pvalue()['DU_kw_lag']
n_lag_ols = len(df_lag)
print(f"  OLS(DU_t-1): coef={lag_coef:.6f}, t={lag_t:.2f}, p={lag_p:.6f}, N={n_lag_ols}")
results['lag_ols'] = {'coef': lag_coef, 'se': lag_se, 't': lag_t, 'p': lag_p, 'N': n_lag_ols}

# IV with lag
# 构造同行业peer均值的lag
iy_lag = df_lag.groupby(['Ind2', 'year'])['DU_kw_lag'].agg(['sum', 'count'])
iy_lag.columns = ['iy_lag_sum', 'iy_lag_count']
df_lag = df_lag.merge(iy_lag, on=['Ind2', 'year'], how='left')
df_lag['DU_kw_peer_lag'] = (df_lag['iy_lag_sum'] - df_lag['DU_kw_lag']) / (df_lag['iy_lag_count'] - 1)
df_lag_iv = df_lag.dropna(subset=['DU_kw_peer_lag']).copy()
n_lag_iv = len(df_lag_iv)

try:
    first_lag = pf.feols(f"DU_kw_lag ~ DU_kw_peer_lag + {ctrl_str} | Stkcd_str + year_str",
                         data=df_lag_iv, vcov={'CRV1': 'IndYear'})
    fl_t = first_lag.coef()['DU_kw_peer_lag'] / first_lag.se()['DU_kw_peer_lag']
    fl_F = fl_t ** 2

    iv_lag = pf.feols(f"PriceDelay ~ 1 + {ctrl_str} | Stkcd_str + year_str | DU_kw_lag ~ DU_kw_peer_lag",
                      data=df_lag_iv, vcov={'CRV1': 'IndYear'})
    ivl_coef = iv_lag.coef()['DU_kw_lag']
    ivl_se = iv_lag.se()['DU_kw_lag']
    ivl_t = ivl_coef / ivl_se
    ivl_p = iv_lag.pvalue()['DU_kw_lag']
    print(f"  2SLS(DU_t-1): coef={ivl_coef:.6f}, t={ivl_t:.2f}, p={ivl_p:.4f}, N={n_lag_iv}")
    print(f"  一阶段F: {fl_F:.2f}")
    results['lag_iv'] = {'coef': ivl_coef, 'se': ivl_se, 't': ivl_t, 'p': ivl_p,
                         'N': n_lag_iv, 'KP_F': fl_F}
except Exception as e:
    print(f"  IV(lag) error: {e}")
print()

# ============================================================
#  5. Oster (2019)
# ============================================================
print("=" * 60)
print("5. Oster (2019) δ* bounds")
print("=" * 60)

ols_short = pf.feols(f"PriceDelay ~ DU_kw | Stkcd_str + year_str",
                     data=df, vcov={'CRV1': 'IndYear'})
beta_s = ols_short.coef()['DU_kw']
r2_s = ols_short._r2
beta_f = ols_coef
r2_f = ols_r2

# 两种R_max设定
r2_max_13 = min(1.0, 1.3 * r2_f)
r2_max_con = 2 * r2_f - r2_s  # 保守设定

def oster_delta(beta_s, r2_s, beta_f, r2_f, r2_max):
    num = beta_f * (r2_max - r2_f)
    den = (beta_s - beta_f) * (r2_f - r2_s)
    if abs(den) < 1e-15:
        return float('inf')
    return num / den

def oster_beta_adj(beta_s, r2_s, beta_f, r2_f, r2_max):
    den = r2_f - r2_s
    if abs(den) < 1e-15:
        return beta_f
    return beta_f - (beta_s - beta_f) * (r2_max - r2_f) / den

d13 = oster_delta(beta_s, r2_s, beta_f, r2_f, r2_max_13)
dcon = oster_delta(beta_s, r2_s, beta_f, r2_f, r2_max_con)
ba13 = oster_beta_adj(beta_s, r2_s, beta_f, r2_f, r2_max_13)
bacon = oster_beta_adj(beta_s, r2_s, beta_f, r2_f, r2_max_con)

print(f"  β_short={beta_s:.6f}, R2_short={r2_s:.4f}")
print(f"  β_full={beta_f:.6f}, R2_full={r2_f:.4f}")
print(f"  --- R_max = 1.3*R_full = {r2_max_13:.4f} ---")
print(f"      δ* = {d13:.2f}")
print(f"      β_adj(δ=1) = {ba13:.6f} {'(同号)' if ba13*beta_f>0 else '(变号!)'}")
print(f"  --- R_max = 2R_full - R_short = {r2_max_con:.4f} (保守) ---")
print(f"      δ* = {dcon:.2f}")
print(f"      β_adj(δ=1) = {bacon:.6f} {'(同号)' if bacon*beta_f>0 else '(变号!)'}")

results['oster'] = {
    'beta_short': beta_s, 'r2_short': r2_s,
    'beta_full': beta_f, 'r2_full': r2_f,
    'r2_max_13': r2_max_13, 'delta_star_13': d13, 'beta_adj_13': ba13,
    'r2_max_con': r2_max_con, 'delta_star_con': dcon, 'beta_adj_con': bacon
}
print()

# ============================================================
#  汇总
# ============================================================
print("=" * 60)
print("汇总")
print("=" * 60)

def sig(p):
    if p < 0.01: return '***'
    if p < 0.05: return '**'
    if p < 0.1: return '*'
    return ''

rows = []
for label, key in [('OLS基准', 'ols'), ('同行业均值IV', 'peer_iv'),
                    ('Bartik IV', 'bartik_iv'), ('OLS(DU_t-1)', 'lag_ols'),
                    ('IV(DU_t-1)', 'lag_iv')]:
    if key in results:
        r = results[key]
        fs = f", KP_F={r['KP_F']:.1f}" if 'KP_F' in r else ''
        print(f"  {label:20s}: β={r['coef']:.6f}, t={r['t']:.2f}{sig(r['p'])}, N={r['N']}{fs}")
        rows.append({
            '方法': label, '系数': r['coef'], '标准误': r['se'],
            't值': r['t'], 'p值': r['p'], '显著性': sig(r['p']),
            'N': r['N'],
            'KP_F': r.get('KP_F', ''),
            'AR_F': r.get('AR_F', ''),
            'AR_p': r.get('AR_p', ''),
            'DWH_F': r.get('DWH_F', ''),
            'DWH_p': r.get('DWH_p', '')
        })

pd.DataFrame(rows).to_csv(os.path.join(RES_DIR, "endogeneity_v3_summary.csv"),
                           index=False, encoding='utf-8-sig')

# JSON
def conv(o):
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.integer,)): return int(o)
    return o

with open(os.path.join(RES_DIR, "endogeneity_v3_results.json"), 'w') as f:
    json.dump({k: {kk: conv(vv) for kk, vv in v.items()} for k, v in results.items()},
              f, indent=2, ensure_ascii=False)

print(f"\n保存至 {RES_DIR}/")
