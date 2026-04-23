"""
稳健性检验：披露后窗口 PriceDelay

问题：DU_kw(t) 来自 t 年年报，在 t+1 年4月底前披露。
      基准回归用 t 年日历年收益率构造 PriceDelay(t)，存在时序倒挂。

解决：构造 PriceDelay_post，用 t+1 年 5-12 月的日度收益率。
      此时 t 年年报已公开，时序完全对齐。
      回归 PriceDelay_post(t) ~ DU_kw(t) + controls(t)
"""

import pandas as pd
import numpy as np
import pyfixest as pf
import os, time, warnings, json
warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = f"{BASE}/data_parquet"
N_LAGS = 5
MIN_OBS = 80  # 8个月窗口，降低阈值（原120对应12个月）

# ============================================================
# 1. 构造 PriceDelay_post
# ============================================================
print("=" * 60)
print("1. 构造披露后窗口 PriceDelay_post")
print("   窗口：t+1 年 5月1日 至 12月31日")
print("=" * 60)

t0 = time.time()

# 加载日度收益率
dr = pd.read_parquet(f"{OUT_DIR}/daily_return.parquet",
                     columns=['Stkcd', 'Trddt', 'Dretwd', 'Markettype'])
dr = dr[dr['Markettype'].isin([1, 4, 16, 32])].copy()
dr['Trddt'] = pd.to_datetime(dr['Trddt'])
dr['Dretwd'] = pd.to_numeric(dr['Dretwd'], errors='coerce')
dr = dr.dropna(subset=['Dretwd'])

# 加载市场收益率
ff3 = pd.read_parquet(f"{OUT_DIR}/ff3_daily.parquet")
mkt = ff3[ff3['MarkettypeID'] == 'P9714'][['TradingDate', 'RiskPremium1']].copy()
mkt.columns = ['Trddt', 'Rm']
mkt['Trddt'] = pd.to_datetime(mkt['Trddt'])
mkt = mkt.sort_values('Trddt').reset_index(drop=True)

for lag in range(1, N_LAGS + 1):
    mkt[f'Rm_lag{lag}'] = mkt['Rm'].shift(lag)
mkt = mkt.dropna()

# 合并
df = dr.merge(mkt, on='Trddt', how='inner')

# 关键：筛选 5-12 月数据，年份标记为 t = calendar_year - 1
# 因为 5-12 月属于 t+1 年，对应的 DU_kw 是 t 年
df['month'] = df['Trddt'].dt.month
df['cal_year'] = df['Trddt'].dt.year
df_post = df[(df['month'] >= 5) & (df['month'] <= 12)].copy()
df_post['year'] = df_post['cal_year'] - 1  # 映射到 DU_kw 的年份

print(f"  日度观测: {len(df_post):,}")
print(f"  年份范围 (DU_kw年): {df_post['year'].min()} ~ {df_post['year'].max()}")
print(f"  加载耗时: {time.time()-t0:.1f}s")


# 核心函数
def compute_delay(ri, rm, rm_lags):
    T = len(ri)
    ones = np.ones((T, 1))
    ss_tot = np.sum((ri - ri.mean()) ** 2)
    if ss_tot == 0:
        return np.nan
    X_r = np.column_stack([ones, rm])
    beta_r = np.linalg.lstsq(X_r, ri, rcond=None)[0]
    ss_res_r = np.sum((ri - X_r @ beta_r) ** 2)
    r2_r = 1.0 - ss_res_r / ss_tot
    X_u = np.column_stack([ones, rm, rm_lags])
    beta_u = np.linalg.lstsq(X_u, ri, rcond=None)[0]
    ss_res_u = np.sum((ri - X_u @ beta_u) ** 2)
    r2_u = 1.0 - ss_res_u / ss_tot
    if r2_u <= 0:
        return np.nan
    return np.clip(1.0 - r2_r / r2_u, 0.0, 1.0)


# 按企业-年度计算
print(f"\n  计算 PriceDelay_post (MIN_OBS={MIN_OBS})...")
t1 = time.time()
lag_cols = [f'Rm_lag{i}' for i in range(1, N_LAGS + 1)]
results = []

for (stkcd, year), g in df_post.groupby(['Stkcd', 'year']):
    if len(g) < MIN_OBS:
        continue
    delay = compute_delay(g['Dretwd'].values, g['Rm'].values, g[lag_cols].values)
    results.append({'Stkcd': stkcd, 'year': year, 'PriceDelay_post': delay, 'n_obs_post': len(g)})

delay_post = pd.DataFrame(results).dropna(subset=['PriceDelay_post'])
print(f"  完成: {len(delay_post):,} 个企业-年度观测")
print(f"  PriceDelay_post: mean={delay_post['PriceDelay_post'].mean():.4f}, "
      f"sd={delay_post['PriceDelay_post'].std():.4f}")
print(f"  耗时: {time.time()-t1:.1f}s")

# ============================================================
# 2. 合并面板数据并跑回归
# ============================================================
print("\n" + "=" * 60)
print("2. 合并面板数据并跑稳健性回归")
print("=" * 60)

# 加载面板数据（复用 generate_tables_v8.py 的构造逻辑）
panel = pd.read_parquet(f"{BASE}/data_parquet/panel.parquet")
ar_feat = pd.read_parquet(f"{BASE}/data_parquet/annual_report_features.parquet")
panel = panel.merge(
    ar_feat[['Stkcd','year','kw_total','kw_per10k','substantive_count']],
    on=['Stkcd','year'], how='left'
)

fi = pd.read_parquet(f"{BASE}/data_parquet/firm_info.parquet",
                     columns=['Symbol','EndDate','IndustryCodeC','LISTINGSTATE'])
fi = fi.rename(columns={'Symbol':'Stkcd'})
fi['EndDate'] = pd.to_datetime(fi['EndDate'])
fi['year'] = fi['EndDate'].dt.year
fi = fi.sort_values(['Stkcd','year','EndDate']).drop_duplicates(subset=['Stkcd','year'], keep='last')
panel = panel.merge(fi[['Stkcd','year','IndustryCodeC','LISTINGSTATE']], on=['Stkcd','year'], how='left')
panel = panel.sort_values(['Stkcd', 'year'])
panel[['IndustryCodeC', 'LISTINGSTATE']] = panel.groupby('Stkcd')[['IndustryCodeC', 'LISTINGSTATE']].transform(
    lambda x: x.ffill().bfill()
)

mask_fin = panel['IndustryCodeC'].str.startswith('J', na=False)
mask_st = panel['LISTINGSTATE'].isin(['ST','*ST'])
mask_new = panel['Age'] <= 0
panel = panel[~mask_fin & ~mask_st & ~mask_new]

panel['Ind2'] = panel['IndustryCodeC'].str[:3]
panel['IndYear'] = panel['Ind2'].astype(str) + '_' + panel['year'].astype(str)
panel['DU_kw'] = panel['kw_per10k']

def winsorize(s):
    lo, hi = s.quantile([0.01, 0.99])
    return s.clip(lo, hi)

controls = ['Size','Lev','ROA','TobinQ','Age','Growth','BoardSize','IndepRatio',
            'Dual','Top1Share','SOE','InstHold','Amihud','Analyst','AuditType']
cont_vars = ['DU_kw','Lev','ROA','Growth','Size','TobinQ','Age','BoardSize','IndepRatio',
             'Top1Share','InstHold','Amihud','Analyst']
for v in cont_vars:
    if v in panel.columns and panel[v].notna().any():
        panel[v] = winsorize(panel[v])

# 合并 PriceDelay_post
panel = panel.merge(delay_post[['Stkcd','year','PriceDelay_post']], on=['Stkcd','year'], how='left')

# Winsorize PriceDelay_post
panel['PriceDelay_post'] = winsorize(panel['PriceDelay_post'].dropna()).reindex(panel.index)

# 回归样本
reg = panel.dropna(subset=['PriceDelay_post','DU_kw'] + controls + ['IndYear']).copy()
reg['Stkcd_str'] = reg['Stkcd'].astype(str)
reg['year_str'] = reg['year'].astype(str)
ctrl_str = " + ".join(controls)

print(f"  回归样本: N={len(reg):,}, firms={reg.Stkcd.nunique():,}")
print(f"  PriceDelay_post: mean={reg['PriceDelay_post'].mean():.4f}, sd={reg['PriceDelay_post'].std():.4f}")

# 跑回归：与基准模型(2)对应
m = pf.feols(f"PriceDelay_post ~ DU_kw + {ctrl_str} | Stkcd_str + year_str",
             data=reg, vcov={"CRV1": "IndYear"})

coef = m.coef()['DU_kw']
se = m.se()['DU_kw']
tval = m.tstat()['DU_kw']
pval = m.pvalue()['DU_kw']
n_obs = m._N

print(f"\n  结果: DU_kw → PriceDelay_post")
print(f"  coef = {coef:.6f}")
print(f"  SE   = {se:.6f}")
print(f"  t    = {tval:.3f}")
print(f"  p    = {pval:.4f}")
print(f"  N    = {n_obs:,}")

stars = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
print(f"  显著性: {stars}")

# 保存结果
result = {
    "test": "post_disclosure_window",
    "window": "May(t+1) to Dec(t+1)",
    "min_obs": MIN_OBS,
    "coef": round(coef, 6),
    "se": round(se, 6),
    "tval": round(tval, 3),
    "pval": round(pval, 4),
    "N": int(n_obs),
    "stars": stars,
    "PriceDelay_post_mean": round(reg['PriceDelay_post'].mean(), 4),
    "PriceDelay_post_sd": round(reg['PriceDelay_post'].std(), 4),
}

os.makedirs(f"{BASE}/results/v11_expanded", exist_ok=True)
with open(f"{BASE}/results/v11_expanded/timing_robustness.json", 'w') as f:
    json.dump(result, f, indent=2, ensure_ascii=False)
print(f"\n  结果已保存: results/v11_expanded/timing_robustness.json")
