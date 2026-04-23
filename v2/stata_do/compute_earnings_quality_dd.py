"""
构造 Dechow-Dichev (2002) 盈余质量 (EarnQual) 机制变量
公式：Accruals_t = α + β1·CFO_{t-1} + β2·CFO_t + β3·CFO_{t+1} + ε
EarnQual_{i,t} = rolling 5-year σ(ε_{i,t'}) over t'=t-4..t
解读：EarnQual 越大 = 质量越低
用作机制变量时预期：披露 → EarnQual ↓（质量 ↑）
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

ROOT = '/Users/mac/computerscience/0做完了/15会计研究/v1/data_parquet'

print("=" * 60)
print("DD 2002 盈余质量构造")
print("=" * 60)

# 1. 读取 panel_dml 取 ROA, CFO（均已按期末总资产 scale）
panel = pd.read_parquet(f'{ROOT}/panel_dml.parquet', columns=['Stkcd','year','ROA','CFO'])
panel = panel.dropna(subset=['ROA','CFO']).copy()
panel['Accruals'] = panel['ROA'] - panel['CFO']
print(f"Panel N = {len(panel):,}; Accruals describe:")
print(panel['Accruals'].describe())

# 2. 合并行业代码
fi = pd.read_parquet(f'{ROOT}/firm_info.parquet',
                    columns=['Symbol','EndDate','IndustryCodeC','IndustryCodeD'])
fi = fi.rename(columns={'Symbol': 'Stkcd'})
fi['EndDate'] = pd.to_datetime(fi['EndDate'])
fi['year'] = fi['EndDate'].dt.year
fi['Ind2'] = fi['IndustryCodeC'].fillna(fi['IndustryCodeD']).str[:3]
fi = fi.sort_values(['Stkcd','year','EndDate']).drop_duplicates(['Stkcd','year'], keep='last')
# 用最新行业码填缺失年度
fi_latest = fi.sort_values(['Stkcd','year']).drop_duplicates(['Stkcd'], keep='last')[['Stkcd','Ind2']].rename(columns={'Ind2':'Ind2_latest'})

panel = panel.merge(fi[['Stkcd','year','Ind2']], on=['Stkcd','year'], how='left')
panel = panel.merge(fi_latest, on='Stkcd', how='left')
panel['Ind2'] = panel['Ind2'].fillna(panel['Ind2_latest'])
panel = panel.drop(columns=['Ind2_latest']).dropna(subset=['Ind2'])
print(f"Merged industry N = {len(panel):,}; {panel['Ind2'].nunique()} industries")

# 3. 构造 CFO_lag, CFO_lead
panel = panel.sort_values(['Stkcd','year']).reset_index(drop=True)
panel['CFO_lag'] = panel.groupby('Stkcd')['CFO'].shift(1)
panel['CFO_lead'] = panel.groupby('Stkcd')['CFO'].shift(-1)

# 4. winsorize 1%/99% (每个变量全样本)
for c in ['Accruals','CFO','CFO_lag','CFO_lead']:
    p1, p99 = panel[c].quantile([0.01, 0.99])
    panel[c] = panel[c].clip(p1, p99)

# 5. 按 (Ind2, year) 跑 DD 回归，收集残差（要求该 cell >= 10 个有效观测）
from sklearn.linear_model import LinearRegression
results = []
for (ind2, year), g in panel.groupby(['Ind2','year']):
    g_valid = g.dropna(subset=['Accruals','CFO_lag','CFO','CFO_lead'])
    if len(g_valid) < 10:
        continue
    X = g_valid[['CFO_lag','CFO','CFO_lead']].values
    y = g_valid['Accruals'].values
    lr = LinearRegression().fit(X, y)
    resid = y - lr.predict(X)
    for i, idx in enumerate(g_valid.index):
        results.append({
            'Stkcd': g_valid.loc[idx, 'Stkcd'],
            'year': g_valid.loc[idx, 'year'],
            'DD_resid': resid[i],
        })
resid_df = pd.DataFrame(results)
print(f"\nDD 残差 N = {len(resid_df):,}")

# 6. 按企业 rolling 5-year σ(ε)
resid_df = resid_df.sort_values(['Stkcd','year']).reset_index(drop=True)

def rolling_std(s, window=5, min_periods=3):
    return s.rolling(window=window, min_periods=min_periods).std()

resid_df['EarnQual'] = resid_df.groupby('Stkcd')['DD_resid'].transform(
    lambda s: rolling_std(s, window=5, min_periods=3))

# 7. 也给出 |DD_resid| 作为简化版本 (firm-year level, all-year coverage)
resid_df['EarnQualAbs'] = resid_df['DD_resid'].abs()

# 8. Winsorize 1%/99%
for c in ['EarnQual','EarnQualAbs']:
    mask = resid_df[c].notna()
    if mask.sum() > 0:
        p1, p99 = resid_df.loc[mask, c].quantile([0.01, 0.99])
        resid_df.loc[mask, c] = resid_df.loc[mask, c].clip(p1, p99)

print(f"\nEarnQual (rolling σ) 描述:")
print(resid_df['EarnQual'].describe())
print(f"\nEarnQualAbs (|ε|) 描述:")
print(resid_df['EarnQualAbs'].describe())

# 9. 保存
out = resid_df[['Stkcd','year','EarnQual','EarnQualAbs','DD_resid']].copy()
out.to_parquet(f'{ROOT}/earnings_quality_dd.parquet', index=False)
print(f"\n已保存: {ROOT}/earnings_quality_dd.parquet")
print(f"Final N = {len(out):,}; EarnQual 非缺失 = {out['EarnQual'].notna().sum():,}")
