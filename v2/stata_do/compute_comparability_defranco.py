"""
构造 De Franco-Kothari-Verdi (2011, TAR) 会计信息可比性 (Comparability)
公式：
  Step 1: 对每家企业 i, rolling 5-year window 估计 Earnings_it = α_i + β_i·Return_it + ε
  Step 2: 对每对 (i, j) 在 Ind2 内、时间 t：
          E_{jj,t} = α_j + β_j·Return_{jt}（j 用自己系数）
          E_{ij,t} = α_i + β_i·Return_{jt}（j 的回报套 i 的系数）
          Comp_{ij,t} = -|E_{jj,t} - E_{ij,t}|
  Step 3: Firm j 的 Comparability = 同行业内 (所有 peer i) 的 Comp_{ij,t} 中位数
解读：Comp 值越大（接近 0）= 可比性越高
作机制变量时预期：披露 → Comp ↑（可比性提升）

Earnings: 用 ROA (NI/TA)
Return: 从 monthly_return 构造年度复合回报
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

ROOT = '/Users/mac/computerscience/0做完了/15会计研究/v1/data_parquet'

print("=" * 60)
print("De Franco-Kothari-Verdi (2011) 会计信息可比性构造")
print("=" * 60)

# 1. 年度回报
mr = pd.read_parquet(f'{ROOT}/monthly_return.parquet',
                     columns=['Stkcd','Trdmnt','Mretwd','Markettype'])
mr = mr[mr['Markettype'].isin([1,4,16,32])].copy()
mr['year'] = mr['Trdmnt'].str[:4].astype(int)
mr['Mretwd'] = pd.to_numeric(mr['Mretwd'], errors='coerce')
mr = mr.dropna(subset=['Mretwd'])
# 年度复合回报
mr['1pr'] = 1 + mr['Mretwd']
ann = mr.groupby(['Stkcd','year'])['1pr'].apply(lambda s: s.prod() - 1 if len(s) >= 10 else np.nan).reset_index()
ann.columns = ['Stkcd','year','AnnRet']
ann = ann.dropna()
print(f"年度回报 N = {len(ann):,}")

# 2. 年度 ROA
panel = pd.read_parquet(f'{ROOT}/panel_dml.parquet', columns=['Stkcd','year','ROA'])
panel = panel.dropna()

# 3. 合并 ROA + AnnRet + 行业码
fi = pd.read_parquet(f'{ROOT}/firm_info.parquet',
                    columns=['Symbol','EndDate','IndustryCodeC','IndustryCodeD'])
fi = fi.rename(columns={'Symbol':'Stkcd'})
fi['EndDate'] = pd.to_datetime(fi['EndDate'])
fi['year'] = fi['EndDate'].dt.year
fi['Ind2'] = fi['IndustryCodeC'].fillna(fi['IndustryCodeD']).str[:3]
fi = fi.sort_values(['Stkcd','year','EndDate']).drop_duplicates(['Stkcd','year'], keep='last')
fi_latest = fi.sort_values(['Stkcd','year']).drop_duplicates(['Stkcd'], keep='last')[['Stkcd','Ind2']].rename(columns={'Ind2':'Ind2_latest'})

data = panel.merge(ann, on=['Stkcd','year'], how='inner')
data = data.merge(fi[['Stkcd','year','Ind2']], on=['Stkcd','year'], how='left')
data = data.merge(fi_latest, on='Stkcd', how='left')
data['Ind2'] = data['Ind2'].fillna(data['Ind2_latest']).str[:3]
data = data.drop(columns=['Ind2_latest']).dropna(subset=['Ind2'])

# Winsorize ROA 和 AnnRet
for c in ['ROA','AnnRet']:
    p1, p99 = data[c].quantile([0.01, 0.99])
    data[c] = data[c].clip(p1, p99)

print(f"\n样本 merged N = {len(data):,}; {data['Ind2'].nunique()} industries")

# 4. 对每家企业 rolling 5-year 估计 (α_i, β_i)
data = data.sort_values(['Stkcd','year']).reset_index(drop=True)
MIN_OBS = 5  # 至少 5 年

def rolling_ols(sub):
    """对一家企业按年份排序，对每个 t 使用 t-4..t 5年窗口估计 (α, β)"""
    n = len(sub)
    alpha = np.full(n, np.nan)
    beta = np.full(n, np.nan)
    for i in range(MIN_OBS-1, n):
        window = sub.iloc[max(0, i-MIN_OBS+1):i+1]
        if len(window) < MIN_OBS:
            continue
        x = window['AnnRet'].values
        y = window['ROA'].values
        if np.std(x) == 0:
            continue
        b = np.cov(x, y, ddof=0)[0,1] / np.var(x, ddof=0)
        a = y.mean() - b * x.mean()
        alpha[i] = a
        beta[i] = b
    sub = sub.copy()
    sub['alpha_i'] = alpha
    sub['beta_i'] = beta
    return sub

print("\n估计每家企业的 rolling (α_i, β_i)...")
data = data.groupby('Stkcd', group_keys=False).apply(rolling_ols).reset_index(drop=True)
n_with_coefs = data['alpha_i'].notna().sum()
print(f"有 (α, β) 的观测数: {n_with_coefs:,} / {len(data):,}")

# 5. 对每个 firm-year j: Comp_{j,t} = Ind2 内其他企业 i 的 Comp_{ij,t} 中位数
print("\n计算可比性...")
comp_results = []
for (ind2, year), g in data.groupby(['Ind2','year']):
    g = g[g['alpha_i'].notna()].reset_index(drop=True)
    if len(g) < 5:
        continue
    # 每对 (i, j)
    alphas = g['alpha_i'].values
    betas = g['beta_i'].values
    rets = g['AnnRet'].values
    stkcds = g['Stkcd'].values
    n = len(g)
    for j_idx in range(n):
        E_jj = alphas[j_idx] + betas[j_idx] * rets[j_idx]
        comps = []
        for i_idx in range(n):
            if i_idx == j_idx:
                continue
            E_ij = alphas[i_idx] + betas[i_idx] * rets[j_idx]
            comps.append(-abs(E_jj - E_ij))
        if len(comps) == 0:
            continue
        # 全行业中位数 + Top-4 均值
        comp_ind_median = np.median(comps)
        comp_top4 = np.mean(sorted(comps, reverse=True)[:4]) if len(comps) >= 4 else np.nan
        comp_results.append({
            'Stkcd': stkcds[j_idx],
            'year': year,
            'Comparability_med': comp_ind_median,
            'Comparability_top4': comp_top4,
        })

comp_df = pd.DataFrame(comp_results)
print(f"\n可比性 N = {len(comp_df):,}")

# 6. winsorize
for c in ['Comparability_med','Comparability_top4']:
    mask = comp_df[c].notna()
    p1, p99 = comp_df.loc[mask, c].quantile([0.01, 0.99])
    comp_df.loc[mask, c] = comp_df.loc[mask, c].clip(p1, p99)

print("\nComparability_med 描述 (中位数口径):")
print(comp_df['Comparability_med'].describe())
print("\nComparability_top4 描述 (top-4 均值):")
print(comp_df['Comparability_top4'].describe())

# 7. 保存
comp_df.to_parquet(f'{ROOT}/accounting_comparability.parquet', index=False)
comp_df.to_csv(f'{ROOT}/accounting_comparability.csv', index=False)
print(f"\n已保存: {ROOT}/accounting_comparability.parquet (+.csv)")
