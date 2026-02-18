"""
构造因变量：股价延迟 Hou-Moskowitz (2005)

Delay = 1 - R²_restricted / R²_unrestricted

受限模型:   r_i,t = α + β₀ · r_m,t + ε
无限制模型: r_i,t = α + β₀ · r_m,t + β₁ · r_m,t-1 + ... + β₅ · r_m,t-5 + ε

按企业-年度计算，要求每年至少120个交易日
"""

import pandas as pd
import numpy as np
import os, time, warnings
warnings.filterwarnings('ignore')

OUT_DIR = "/Users/mac/computerscience/15会计研究/data_parquet"
N_LAGS = 5       # 滞后阶数
MIN_OBS = 120    # 每个企业-年度最少交易日

# ============ 1. 加载数据 ============
print("加载数据...")
t0 = time.time()

# 个股日回报率 (只要沪深A股: Markettype 1,4,16,32)
dr = pd.read_parquet(f"{OUT_DIR}/daily_return.parquet",
                     columns=['Stkcd', 'Trddt', 'Dretwd', 'Markettype'])
dr = dr[dr['Markettype'].isin([1, 4, 16, 32])].copy()
dr['Trddt'] = pd.to_datetime(dr['Trddt'])
dr['Dretwd'] = pd.to_numeric(dr['Dretwd'], errors='coerce')
dr = dr.dropna(subset=['Dretwd'])

# 市场回报率: 用FF3的综合A股市场超额收益 (P9714)
# R² 不受常数项影响，所以用超额收益和原始收益算出的delay一样
ff3 = pd.read_parquet(f"{OUT_DIR}/ff3_daily.parquet")
mkt = ff3[ff3['MarkettypeID'] == 'P9714'][['TradingDate', 'RiskPremium1']].copy()
mkt.columns = ['Trddt', 'Rm']
mkt['Trddt'] = pd.to_datetime(mkt['Trddt'])
mkt = mkt.sort_values('Trddt').reset_index(drop=True)

# 生成市场回报率的滞后项
for lag in range(1, N_LAGS + 1):
    mkt[f'Rm_lag{lag}'] = mkt['Rm'].shift(lag)
mkt = mkt.dropna()

print(f"  个股日回报率: {len(dr):,} obs, {dr.Stkcd.nunique():,} stocks")
print(f"  市场回报率: {len(mkt):,} days, {mkt.Trddt.min().date()} ~ {mkt.Trddt.max().date()}")
print(f"  加载耗时: {time.time()-t0:.1f}s")

# ============ 2. 合并个股与市场 ============
print("\n合并个股与市场数据...")
df = dr.merge(mkt, on='Trddt', how='inner')
df['year'] = df['Trddt'].dt.year
print(f"  合并后: {len(df):,} obs")

# ============ 3. 核心函数: 计算单个企业-年度的delay ============

def compute_delay(ri, rm, rm_lags):
    """
    计算单个企业-年度的 Hou-Moskowitz 股价延迟
    Delay = 1 - R²_restricted / R²_unrestricted
    """
    T = len(ri)
    ones = np.ones((T, 1))
    ss_tot = np.sum((ri - ri.mean()) ** 2)
    if ss_tot == 0:
        return np.nan

    # 受限模型: ri ~ 1 + rm
    X_r = np.column_stack([ones, rm])
    beta_r = np.linalg.lstsq(X_r, ri, rcond=None)[0]
    ss_res_r = np.sum((ri - X_r @ beta_r) ** 2)
    r2_r = 1.0 - ss_res_r / ss_tot

    # 无限制模型: ri ~ 1 + rm + rm_lags
    X_u = np.column_stack([ones, rm, rm_lags])
    beta_u = np.linalg.lstsq(X_u, ri, rcond=None)[0]
    ss_res_u = np.sum((ri - X_u @ beta_u) ** 2)
    r2_u = 1.0 - ss_res_u / ss_tot

    if r2_u <= 0:
        return np.nan
    delay = 1.0 - r2_r / r2_u
    return np.clip(delay, 0.0, 1.0)


# ============ 4. 按企业-年度循环计算 ============
print(f"\n开始计算 Delay (要求每年 >= {MIN_OBS} 个交易日)...")
t1 = time.time()

lag_cols = [f'Rm_lag{i}' for i in range(1, N_LAGS + 1)]
results = []

groups = df.groupby(['Stkcd', 'year'])
total = len(groups)
done = 0

for (stkcd, year), g in groups:
    if len(g) < MIN_OBS:
        continue

    ri = g['Dretwd'].values
    rm = g['Rm'].values
    rm_lags = g[lag_cols].values

    delay = compute_delay(ri, rm, rm_lags)
    results.append({'Stkcd': stkcd, 'year': year, 'PriceDelay': delay, 'n_obs': len(g)})

    done += 1
    if done % 5000 == 0:
        print(f"  进度: {done}/{total} ({done/total*100:.1f}%)")

delay_df = pd.DataFrame(results)
delay_df = delay_df.dropna(subset=['PriceDelay'])

print(f"\n计算完成: {len(delay_df):,} 个企业-年度观测")
print(f"  年份范围: {delay_df.year.min()} ~ {delay_df.year.max()}")
print(f"  Delay 描述统计:")
print(delay_df['PriceDelay'].describe().to_string())
print(f"  耗时: {time.time()-t1:.1f}s")

# ============ 5. 保存 ============
delay_df.to_parquet(f"{OUT_DIR}/price_delay.parquet", index=False)
print(f"\n已保存: {OUT_DIR}/price_delay.parquet")
