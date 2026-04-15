#!/usr/bin/env python3
"""
内生性检验 v2 — 统一同期DU_kw口径 + 完整IV诊断统计量
==========================================
口径: 同期DU_kw (与基准回归v3_concurrent一致)
固定效应: 企业+年份
聚类标准误: 行业×年份

检验方法:
1. OLS基准
2. 同行业均值IV (2SLS) + KP F / AR / DWH
3. Bartik移位份额IV (2SLS) + KP F / AR / DWH
4. DU_{t-1}滞后一期 (稳健性)
5. Oster (2019) δ* bounds
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
RES_DIR = os.path.join(BASE_DIR, "results/endogeneity_v2")
os.makedirs(RES_DIR, exist_ok=True)

# ── 加载数据 (与v3_concurrent完全一致) ──
panel = pd.read_parquet(os.path.join(DATA_DIR, "panel.parquet"))
af = pd.read_parquet(os.path.join(DATA_DIR, "annual_report_features.parquet"))
fi = pd.read_parquet(os.path.join(DATA_DIR, "firm_info.parquet"))

# 合并DU_kw (同期，不做滞后)
af_merge = af[['Stkcd','year','kw_per10k','substantive_count','kw_total']].copy()
af_merge.columns = ['Stkcd','year','DU_kw','DU_sub','kw_total']
af_merge['DU_kw_ln'] = np.log1p(af_merge['kw_total'])

df = panel.merge(af_merge[['Stkcd','year','DU_kw','DU_sub','DU_kw_ln']], on=['Stkcd','year'], how='left')

# 获取行业代码
fi_latest = fi.sort_values('EndDate').groupby('Symbol').last().reset_index()
fi_latest = fi_latest[['Symbol','IndustryCodeC']].rename(
    columns={'Symbol':'Stkcd','IndustryCodeC':'Indcd'}
)
df = df.merge(fi_latest, on='Stkcd', how='left')

# 控制变量
controls = ['Size','Lev','ROA','TobinQ','Age','Growth','BoardSize','IndepRatio',
            'Dual','Top1Share','SOE','InstHold','Amihud','Analyst','AuditType']
ctrl_str = ' + '.join(controls)

# 筛选有效样本
df = df.dropna(subset=['PriceDelay','DU_kw','Indcd'] + controls)

# Winsorize
def winsorize(s, lo=0.01, hi=0.99):
    q = s.quantile([lo, hi])
    return s.clip(q.iloc[0], q.iloc[1])

cont_vars = ['PriceDelay','DU_kw','DU_sub','DU_kw_ln'] + [c for c in controls if df[c].dtype in ['float64','float32','int64']]
for v in cont_vars:
    if v in df.columns and v not in ['SOE','AuditType','Dual']:
        df[v] = winsorize(df[v])

# 固定效应标识
df['Stkcd_str'] = df['Stkcd'].astype(str)
df['year_str'] = df['year'].astype(str)
df['IndYear'] = df['Indcd'].astype(str) + '_' + df['year'].astype(str)

print(f"样本: N={len(df)}, Firms={df['Stkcd'].nunique()}, Years={df['year'].nunique()}")
print(f"行业: {df['Indcd'].nunique()} unique")
print(f"DU_kw: mean={df['DU_kw'].mean():.4f}, sd={df['DU_kw'].std():.4f}")
print()

results = {}

# ============================================================
#  1. OLS基准 (同期DU_kw)
# ============================================================
print("="*60)
print("1. OLS基准 (同期DU_kw, 与基准回归v3一致)")
print("="*60)

ols = pf.feols(f"PriceDelay ~ DU_kw + {ctrl_str} | Stkcd_str + year_str",
               data=df, vcov={'CRV1': 'IndYear'})
ols_coef = ols.coef()['DU_kw']
ols_se = ols.se()['DU_kw']
ols_t = ols_coef / ols_se
ols_p = ols.pvalue()['DU_kw']
ols_r2 = ols._r2
ols_r2_within = ols._r2_within if hasattr(ols, '_r2_within') else None
print(f"  coef={ols_coef:.6f}, se={ols_se:.6f}, t={ols_t:.2f}, p={ols_p:.6f}")
print(f"  R2={ols_r2:.4f}")
results['ols'] = {
    'coef': ols_coef, 'se': ols_se, 't': ols_t, 'p': ols_p,
    'N': len(df), 'R2': ols_r2
}
print()

# ============================================================
#  2. 同行业均值IV
# ============================================================
print("="*60)
print("2. 同行业均值IV (2SLS + 完整诊断)")
print("="*60)

# 构造同年同行业排除自身的DU_kw均值
iy = df.groupby(['Indcd','year'])['DU_kw'].agg(['sum','count'])
iy.columns = ['iy_sum','iy_count']
df = df.merge(iy, on=['Indcd','year'], how='left', suffixes=('','_dup'))
# 清理可能的重复列
for c in df.columns:
    if c.endswith('_dup'):
        df.drop(columns=c, inplace=True)

df['DU_kw_peer'] = (df['iy_sum'] - df['DU_kw']) / (df['iy_count'] - 1)

# 检查相关性
corr = df[['DU_kw','DU_kw_peer']].corr().iloc[0,1]
print(f"  IV-X correlation: {corr:.4f}")

# --- 一阶段 ---
first = pf.feols(f"DU_kw ~ DU_kw_peer + {ctrl_str} | Stkcd_str + year_str",
                 data=df, vcov={'CRV1': 'IndYear'})
fs_coef = first.coef()['DU_kw_peer']
fs_se = first.se()['DU_kw_peer']
fs_t = fs_coef / fs_se
fs_F = fs_t**2  # 恰好识别时 KP rk F = t^2
print(f"  一阶段: coef={fs_coef:.4f}, t={fs_t:.2f}")
print(f"  Kleibergen-Paap rk F (=t^2 for just-identified): {fs_F:.2f}")

# --- 2SLS ---
iv_peer = pf.feols(f"PriceDelay ~ 1 + {ctrl_str} | Stkcd_str + year_str | DU_kw ~ DU_kw_peer",
                   data=df, vcov={'CRV1': 'IndYear'})
iv_coef = iv_peer.coef()['DU_kw']
iv_se = iv_peer.se()['DU_kw']
iv_t = iv_coef / iv_se
iv_p = iv_peer.pvalue()['DU_kw']
print(f"  2SLS: coef={iv_coef:.6f}, se={iv_se:.6f}, t={iv_t:.2f}, p={iv_p:.4f}")

# --- Anderson-Rubin检验 (Reduced Form) ---
# AR: 检验reduced form系数是否为零
rf = pf.feols(f"PriceDelay ~ DU_kw_peer + {ctrl_str} | Stkcd_str + year_str",
              data=df, vcov={'CRV1': 'IndYear'})
rf_coef = rf.coef()['DU_kw_peer']
rf_se = rf.se()['DU_kw_peer']
rf_t = rf_coef / rf_se
rf_p = rf.pvalue()['DU_kw_peer']
ar_F = rf_t**2
print(f"  Reduced Form: coef={rf_coef:.6f}, t={rf_t:.2f}, p={rf_p:.4f}")
print(f"  Anderson-Rubin F: {ar_F:.2f}, p={rf_p:.4f}")

# --- DWH内生性检验 ---
# Durbin-Wu-Hausman: 将一阶段残差加入OLS，检验残差系数是否显著
# pyfixest可能因singleton FE丢弃部分obs，用predict代替resid
try:
    df['DU_kw_hat'] = first.predict()
    df['resid_first'] = df['DU_kw'] - df['DU_kw_hat']
except:
    # fallback: 手动OLS获取残差
    resid_vals = first.resid()
    # 需要对齐index
    used_idx = first._data.index if hasattr(first, '_data') else df.index[:len(resid_vals)]
    df['resid_first'] = np.nan
    df.loc[df.index[:len(resid_vals)], 'resid_first'] = resid_vals

df_dwh = df.dropna(subset=['resid_first'])
dwh = pf.feols(f"PriceDelay ~ DU_kw + resid_first + {ctrl_str} | Stkcd_str + year_str",
               data=df_dwh, vcov={'CRV1': 'IndYear'})
dwh_coef = dwh.coef()['resid_first']
dwh_se = dwh.se()['resid_first']
dwh_t = dwh_coef / dwh_se
dwh_p = dwh.pvalue()['resid_first']
dwh_F = dwh_t**2
print(f"  DWH内生性检验: resid_coef={dwh_coef:.6f}, t={dwh_t:.2f}, p={dwh_p:.4f}")
print(f"  DWH F={dwh_F:.2f} {'→ 拒绝外生性，IV是必要的' if dwh_p < 0.1 else '→ 不拒绝外生性'}")

results['peer_iv'] = {
    'coef': iv_coef, 'se': iv_se, 't': iv_t, 'p': iv_p,
    'N': len(df),
    'first_stage_F': fs_F,
    'first_stage_coef': fs_coef, 'first_stage_t': fs_t,
    'AR_F': ar_F, 'AR_p': rf_p,
    'DWH_F': dwh_F, 'DWH_p': dwh_p,
    'rf_coef': rf_coef, 'rf_t': rf_t
}
print()

# ============================================================
#  3. Bartik移位份额IV
# ============================================================
print("="*60)
print("3. Bartik/Shift-Share IV (2SLS + 完整诊断)")
print("="*60)

base_year = df['year'].min()  # 2011
ind_base = df[df['year']==base_year].groupby('Indcd')['DU_kw'].mean().reset_index()
ind_base.columns = ['Indcd','DU_kw_base']

ind_year_mean = df.groupby(['Indcd','year'])['DU_kw'].mean().reset_index()
ind_year_mean.columns = ['Indcd','year','DU_kw_ind_mean']

# 基期行业均值
ind_base_map = ind_base.set_index('Indcd')['DU_kw_base']
ind_year_mean['DU_kw_ind_base'] = ind_year_mean['Indcd'].map(ind_base_map)
ind_year_mean['ind_growth'] = ind_year_mean['DU_kw_ind_mean'] / ind_year_mean['DU_kw_ind_base'].clip(lower=0.001)

df = df.merge(ind_base[['Indcd','DU_kw_base']], on='Indcd', how='left', suffixes=('','_dup2'))
df = df.merge(ind_year_mean[['Indcd','year','ind_growth']], on=['Indcd','year'], how='left', suffixes=('','_dup3'))
# 清理重复列
for c in list(df.columns):
    if c.endswith('_dup2') or c.endswith('_dup3'):
        df.drop(columns=c, inplace=True)

if 'DU_kw_base' not in df.columns:
    df['DU_kw_base'] = df['Indcd'].map(ind_base_map)
if 'ind_growth' not in df.columns:
    ig_map = ind_year_mean.set_index(['Indcd','year'])['ind_growth']
    df['ind_growth'] = df.set_index(['Indcd','year']).index.map(ig_map.get)

df['bartik_iv'] = df['DU_kw_base'] * df['ind_growth']
df['bartik_iv'] = df['bartik_iv'].fillna(0)

corr_b = df[['DU_kw','bartik_iv']].corr().iloc[0,1]
print(f"  IV-X correlation: {corr_b:.4f}")

# 一阶段
first_b = pf.feols(f"DU_kw ~ bartik_iv + {ctrl_str} | Stkcd_str + year_str",
                   data=df, vcov={'CRV1': 'IndYear'})
fb_coef = first_b.coef()['bartik_iv']
fb_se = first_b.se()['bartik_iv']
fb_t = fb_coef / fb_se
fb_F = fb_t**2
print(f"  一阶段: coef={fb_coef:.4f}, t={fb_t:.2f}")
print(f"  Kleibergen-Paap rk F: {fb_F:.2f}")

# 2SLS
iv_bartik = pf.feols(f"PriceDelay ~ 1 + {ctrl_str} | Stkcd_str + year_str | DU_kw ~ bartik_iv",
                     data=df, vcov={'CRV1': 'IndYear'})
bk_coef = iv_bartik.coef()['DU_kw']
bk_se = iv_bartik.se()['DU_kw']
bk_t = bk_coef / bk_se
bk_p = iv_bartik.pvalue()['DU_kw']
print(f"  2SLS: coef={bk_coef:.6f}, se={bk_se:.6f}, t={bk_t:.2f}, p={bk_p:.4f}")

# Anderson-Rubin
rf_b = pf.feols(f"PriceDelay ~ bartik_iv + {ctrl_str} | Stkcd_str + year_str",
                data=df, vcov={'CRV1': 'IndYear'})
rf_b_coef = rf_b.coef()['bartik_iv']
rf_b_t = rf_b_coef / rf_b.se()['bartik_iv']
rf_b_p = rf_b.pvalue()['bartik_iv']
ar_b_F = rf_b_t**2
print(f"  Anderson-Rubin F: {ar_b_F:.2f}, p={rf_b_p:.4f}")

# DWH
try:
    df['DU_kw_hat_b'] = first_b.predict()
    df['resid_first_b'] = df['DU_kw'] - df['DU_kw_hat_b']
except:
    resid_b = first_b.resid()
    df['resid_first_b'] = np.nan
    df.loc[df.index[:len(resid_b)], 'resid_first_b'] = resid_b

df_dwh_b = df.dropna(subset=['resid_first_b'])
dwh_b = pf.feols(f"PriceDelay ~ DU_kw + resid_first_b + {ctrl_str} | Stkcd_str + year_str",
                 data=df_dwh_b, vcov={'CRV1': 'IndYear'})
dwh_b_t = dwh_b.coef()['resid_first_b'] / dwh_b.se()['resid_first_b']
dwh_b_p = dwh_b.pvalue()['resid_first_b']
dwh_b_F = dwh_b_t**2
print(f"  DWH F={dwh_b_F:.2f}, p={dwh_b_p:.4f}")

results['bartik_iv'] = {
    'coef': bk_coef, 'se': bk_se, 't': bk_t, 'p': bk_p,
    'N': len(df),
    'first_stage_F': fb_F,
    'first_stage_coef': fb_coef, 'first_stage_t': fb_t,
    'AR_F': ar_b_F, 'AR_p': rf_b_p,
    'DWH_F': dwh_b_F, 'DWH_p': dwh_b_p
}
print()

# ============================================================
#  4. DU_{t-1} 滞后一期稳健性
# ============================================================
print("="*60)
print("4. DU_{t-1} 滞后一期 (稳健性)")
print("="*60)

df_lag = df.copy()
df_lag = df_lag.sort_values(['Stkcd','year'])
df_lag['DU_kw_lag'] = df_lag.groupby('Stkcd')['DU_kw'].shift(1)
df_lag = df_lag.dropna(subset=['DU_kw_lag'])

ols_lag = pf.feols(f"PriceDelay ~ DU_kw_lag + {ctrl_str} | Stkcd_str + year_str",
                   data=df_lag, vcov={'CRV1': 'IndYear'})
lag_coef = ols_lag.coef()['DU_kw_lag']
lag_se = ols_lag.se()['DU_kw_lag']
lag_t = lag_coef / lag_se
lag_p = ols_lag.pvalue()['DU_kw_lag']
print(f"  OLS(DU_t-1): coef={lag_coef:.6f}, t={lag_t:.2f}, p={lag_p:.6f}")
print(f"  N={len(df_lag)}")

# 同行业均值IV with lagged DU
df_lag['DU_kw_peer_lag'] = df_lag.groupby('Stkcd')['DU_kw_peer'].shift(1)
df_lag_iv = df_lag.dropna(subset=['DU_kw_peer_lag'])

try:
    iv_lag = pf.feols(f"PriceDelay ~ 1 + {ctrl_str} | Stkcd_str + year_str | DU_kw_lag ~ DU_kw_peer_lag",
                      data=df_lag_iv, vcov={'CRV1': 'IndYear'})
    ivl_coef = iv_lag.coef()['DU_kw_lag']
    ivl_se = iv_lag.se()['DU_kw_lag']
    ivl_t = ivl_coef / ivl_se
    ivl_p = iv_lag.pvalue()['DU_kw_lag']

    # 一阶段F
    first_lag = pf.feols(f"DU_kw_lag ~ DU_kw_peer_lag + {ctrl_str} | Stkcd_str + year_str",
                         data=df_lag_iv, vcov={'CRV1': 'IndYear'})
    fl_t = first_lag.coef()['DU_kw_peer_lag'] / first_lag.se()['DU_kw_peer_lag']
    fl_F = fl_t**2

    print(f"  2SLS(DU_t-1): coef={ivl_coef:.6f}, t={ivl_t:.2f}, p={ivl_p:.4f}")
    print(f"  一阶段F: {fl_F:.2f}")

    results['lag_ols'] = {'coef': lag_coef, 'se': lag_se, 't': lag_t, 'p': lag_p, 'N': len(df_lag)}
    results['lag_iv'] = {'coef': ivl_coef, 'se': ivl_se, 't': ivl_t, 'p': ivl_p,
                         'N': len(df_lag_iv), 'first_stage_F': fl_F}
except Exception as e:
    print(f"  IV with lag error: {e}")
    results['lag_ols'] = {'coef': lag_coef, 'se': lag_se, 't': lag_t, 'p': lag_p, 'N': len(df_lag)}

print()

# ============================================================
#  5. Oster (2019) δ* bounds
# ============================================================
print("="*60)
print("5. Oster (2019) δ* bounds")
print("="*60)

# Short regression: DU_kw only (no controls) with FE
ols_short = pf.feols(f"PriceDelay ~ DU_kw | Stkcd_str + year_str",
                     data=df, vcov={'CRV1': 'IndYear'})
beta_short = ols_short.coef()['DU_kw']
r2_short = ols_short._r2

# Full regression
beta_full = ols_coef
r2_full = ols._r2

# Oster bound: δ* = (β_full * (R_max - R_full)) / ((β_short - β_full) * (R_full - R_short))
# R_max = min(1, 1.3 * R_full) per Oster suggestion
r2_max = min(1.0, 1.3 * r2_full)

numerator = beta_full * (r2_max - r2_full)
denominator = (beta_short - beta_full) * (r2_full - r2_short)

if abs(denominator) > 1e-12:
    delta_star = numerator / denominator
else:
    delta_star = float('inf')

print(f"  β_short (no controls) = {beta_short:.6f}, R2_short = {r2_short:.4f}")
print(f"  β_full  (with controls) = {beta_full:.6f}, R2_full  = {r2_full:.4f}")
print(f"  R2_max (1.3*R2_full) = {r2_max:.4f}")
print(f"  Oster δ* = {delta_star:.2f}")
print(f"  解读: 遗漏变量需要是可观测控制变量的 {abs(delta_star):.1f} 倍才能推翻结果")
if abs(delta_star) > 1:
    print(f"  → δ*>1, 结果对遗漏变量偏误稳健")

# Bias-adjusted β (at δ=1)
if abs(denominator) > 1e-12:
    beta_adj = beta_full - (beta_short - beta_full) * (r2_max - r2_full) / (r2_full - r2_short)
    print(f"  Bias-adjusted β (δ=1) = {beta_adj:.6f} {'(同号→稳健)' if beta_adj * beta_full > 0 else '(变号→不稳健)'}")
else:
    beta_adj = beta_full

results['oster'] = {
    'beta_short': beta_short, 'r2_short': r2_short,
    'beta_full': beta_full, 'r2_full': r2_full,
    'r2_max': r2_max, 'delta_star': delta_star,
    'beta_adjusted': beta_adj
}
print()

# ============================================================
#  汇总表
# ============================================================
print("="*60)
print("汇总")
print("="*60)

summary_rows = []

def sig_stars(p):
    if p < 0.01: return '***'
    if p < 0.05: return '**'
    if p < 0.1: return '*'
    return ''

for label, key in [('OLS基准(同期)', 'ols'),
                    ('同行业均值IV', 'peer_iv'),
                    ('Bartik IV', 'bartik_iv'),
                    ('OLS(DU_t-1)', 'lag_ols'),
                    ('同行业均值IV(DU_t-1)', 'lag_iv')]:
    if key in results:
        r = results[key]
        row = {
            '方法': label,
            '系数': r['coef'],
            '标准误': r['se'],
            't值': r['t'],
            'p值': r['p'],
            '显著性': sig_stars(r['p']),
            'N': r.get('N', ''),
        }
        if 'first_stage_F' in r:
            row['一阶段F'] = r['first_stage_F']
        if 'AR_F' in r:
            row['AR_F'] = r['AR_F']
            row['AR_p'] = r['AR_p']
        if 'DWH_F' in r:
            row['DWH_F'] = r['DWH_F']
            row['DWH_p'] = r['DWH_p']
        summary_rows.append(row)
        stars = sig_stars(r['p'])
        fs_info = f", F1={r['first_stage_F']:.1f}" if 'first_stage_F' in r else ''
        print(f"  {label:25s}: β={r['coef']:.6f}, t={r['t']:.2f}{stars}{fs_info}")

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(os.path.join(RES_DIR, "endogeneity_v2_summary.csv"), index=False, encoding='utf-8-sig')

# 保存完整结果
def convert(obj):
    if isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    return obj

results_clean = {k: {kk: convert(vv) for kk, vv in v.items()} for k, v in results.items()}
with open(os.path.join(RES_DIR, "endogeneity_v2_results.json"), 'w') as f:
    json.dump(results_clean, f, indent=2, ensure_ascii=False)

print(f"\n结果已保存至 {RES_DIR}/")
