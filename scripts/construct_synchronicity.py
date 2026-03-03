"""
构造替代因变量：股价同步性 (Price Synchronicity)

SYNCH = log(R² / (1 - R²))

其中 R² 来自:
  r_it = α + β₁·r_mt + β₂·r_ind,t + ε_it

r_mt: 市场收益率 (FF3 综合A股市场超额收益 P9714)
r_ind,t: 行业等权收益率 (CSRC二级行业, 剔除本企业)

参考: Morck, Yeung & Yu (2000); Gul, Kim & Qiu (2010)
按企业-年度计算，要求每年至少120个交易日
"""

import pandas as pd
import numpy as np
import os, time, warnings
warnings.filterwarnings('ignore')

OUT_DIR = "/Users/mac/computerscience/15会计研究/data_parquet"
MIN_OBS = 120

# ============================================================
# 1. 加载日收益率
# ============================================================
print("=" * 60)
print("构造股价同步性 SYNCH")
print("=" * 60)
t0 = time.time()

print("\n1. 加载日收益率...")
dr = pd.read_parquet(f"{OUT_DIR}/daily_return.parquet",
                     columns=['Stkcd', 'Trddt', 'Dretwd', 'Dsmvosd', 'Markettype'])
dr = dr[dr['Markettype'].isin([1, 4, 16, 32])].copy()
dr['Trddt'] = pd.to_datetime(dr['Trddt'])
dr['Dretwd'] = pd.to_numeric(dr['Dretwd'], errors='coerce')
dr = dr.dropna(subset=['Dretwd'])
dr['year'] = dr['Trddt'].dt.year
print(f"  {len(dr):,} obs, {dr.Stkcd.nunique():,} stocks")

# ============================================================
# 2. 加载市场收益率
# ============================================================
print("\n2. 加载市场收益率 (FF3 P9714)...")
ff3 = pd.read_parquet(f"{OUT_DIR}/ff3_daily.parquet")
mkt = ff3[ff3['MarkettypeID'] == 'P9714'][['TradingDate', 'RiskPremium1']].copy()
mkt.columns = ['Trddt', 'Rm']
mkt['Trddt'] = pd.to_datetime(mkt['Trddt'])
mkt = mkt.sort_values('Trddt').drop_duplicates(subset=['Trddt'], keep='last')
print(f"  {len(mkt):,} days, {mkt.Trddt.min().date()} ~ {mkt.Trddt.max().date()}")

# ============================================================
# 3. 加载行业分类
# ============================================================
print("\n3. 加载行业分类 (CSRC二级)...")
fi = pd.read_parquet(f"{OUT_DIR}/firm_info.parquet",
                     columns=['Symbol', 'EndDate', 'IndustryCodeC'])
fi = fi.rename(columns={'Symbol': 'Stkcd'})
fi['EndDate'] = pd.to_datetime(fi['EndDate'])
fi['year'] = fi['EndDate'].dt.year
fi = fi.sort_values(['Stkcd', 'year', 'EndDate']).drop_duplicates(
    subset=['Stkcd', 'year'], keep='last')
fi['Ind2'] = fi['IndustryCodeC'].str[:3]

# 每只股票最新的行业分类 (用于填补缺失年份)
fi_latest = fi.sort_values(['Stkcd', 'year']).drop_duplicates(
    subset=['Stkcd'], keep='last')[['Stkcd', 'Ind2']].rename(
    columns={'Ind2': 'Ind2_latest'})
print(f"  {fi.Stkcd.nunique():,} firms, {fi.Ind2.nunique()} industries")

# ============================================================
# 4. 合并行业分类到日收益率
# ============================================================
print("\n4. 合并行业分类...")
dr_ind = dr.merge(fi[['Stkcd', 'year', 'Ind2']], on=['Stkcd', 'year'], how='left')
dr_ind = dr_ind.merge(fi_latest, on='Stkcd', how='left')
dr_ind['Ind2'] = dr_ind['Ind2'].fillna(dr_ind['Ind2_latest'])
dr_ind = dr_ind.drop(columns=['Ind2_latest'])
dr_ind = dr_ind.dropna(subset=['Ind2'])
print(f"  有行业分类: {len(dr_ind):,} obs, {dr_ind.Stkcd.nunique():,} stocks")

# ============================================================
# 5. 计算行业等权日收益率 (用于后续剔除本企业)
# ============================================================
print("\n5. 计算行业日收益率...")
t1 = time.time()
ind_daily = dr_ind.groupby(['Ind2', 'Trddt']).agg(
    ind_ret_sum=('Dretwd', 'sum'),
    ind_count=('Dretwd', 'count')
).reset_index()
print(f"  行业-日: {len(ind_daily):,} obs, 耗时 {time.time()-t1:.1f}s")

# ============================================================
# 6. 合并市场收益 & 行业汇总
# ============================================================
print("\n6. 合并市场收益与行业汇总...")
dr_full = dr_ind.merge(mkt, on='Trddt', how='inner')
dr_full = dr_full.merge(ind_daily, on=['Ind2', 'Trddt'], how='left')

# 行业收益剔除本企业: R_ind = (sum - r_i) / (N - 1)
dr_full['Rind'] = (dr_full['ind_ret_sum'] - dr_full['Dretwd']) / \
                  (dr_full['ind_count'] - 1)
# 行业内仅1家企业: 用行业均值代替 (自身收益)
mask_single = dr_full['ind_count'] <= 1
dr_full.loc[mask_single, 'Rind'] = np.nan
dr_full = dr_full.dropna(subset=['Rm', 'Rind'])
print(f"  完整样本: {len(dr_full):,} obs, {dr_full.Stkcd.nunique():,} stocks")

# ============================================================
# 7. 按企业-年度计算 SYNCH
# ============================================================
print(f"\n7. 计算 SYNCH (每年 >= {MIN_OBS} 交易日)...")
t2 = time.time()
results = []
groups = dr_full.groupby(['Stkcd', 'year'])
total = len(groups)
done = 0

for (stkcd, year), g in groups:
    if len(g) < MIN_OBS:
        continue

    ri = g['Dretwd'].values
    rm = g['Rm'].values
    rind = g['Rind'].values
    T = len(ri)

    ss_tot = np.sum((ri - ri.mean()) ** 2)
    if ss_tot == 0:
        continue

    # OLS: r_it = α + β₁·r_mt + β₂·r_ind,t + ε
    X = np.column_stack([np.ones(T), rm, rind])
    beta = np.linalg.lstsq(X, ri, rcond=None)[0]
    ss_res = np.sum((ri - X @ beta) ** 2)
    r2 = 1.0 - ss_res / ss_tot

    # SYNCH = log(R² / (1 - R²))
    r2_clipped = np.clip(r2, 1e-6, 1 - 1e-6)
    synch = np.log(r2_clipped / (1 - r2_clipped))

    results.append({
        'Stkcd': stkcd,
        'year': year,
        'SYNCH': synch,
        'R2_synch': r2,
        'n_obs_synch': T,
    })

    done += 1
    if done % 5000 == 0:
        print(f"  进度: {done}/{total} ({done/total*100:.1f}%)")

synch_df = pd.DataFrame(results)
print(f"\n  计算完成: {len(synch_df):,} firm-year obs")
print(f"  企业数: {synch_df.Stkcd.nunique():,}")
print(f"  年份: {synch_df.year.min()} ~ {synch_df.year.max()}")
print(f"\n  SYNCH 描述统计:")
print(synch_df['SYNCH'].describe().to_string())
print(f"\n  R² 描述统计:")
print(synch_df['R2_synch'].describe().to_string())
print(f"\n  耗时: {time.time()-t2:.1f}s")

# ============================================================
# 8. 保存
# ============================================================
outpath = f"{OUT_DIR}/price_synchronicity.parquet"
synch_df.to_parquet(outpath, index=False)
print(f"\n已保存: {outpath}")
print(f"总耗时: {time.time()-t0:.0f}s")
