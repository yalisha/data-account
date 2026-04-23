"""
构造机制检验新变量：
1. Zeros: Lesmond et al. (1999) 零收益日占比 (信息不对称代理)
2. absDA: Kothari et al. (2005) 业绩调整Jones模型可操纵应计绝对值

输出: 合并至 reg_sample_v4.dta
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

BASE = "/Users/mac/computerscience/15会计研究"

# ============================================================
# 1. Zeros: 零收益日占比 (Lesmond et al. 1999)
# ============================================================
print("=" * 60)
print("1. 构造 Zeros (零收益日占比)")
print("=" * 60)

daily = pd.read_parquet(f"{BASE}/data_parquet/daily_return.parquet")
daily['Trddt'] = pd.to_datetime(daily['Trddt'])
daily['year'] = daily['Trddt'].dt.year

# 每只股票每年: 零收益日数 / 总交易日数
def calc_zeros(g):
    ret = g['Dretwd'].dropna()
    if len(ret) < 30:
        return np.nan
    return (ret == 0).sum() / len(ret)

zeros = daily.groupby(['Stkcd', 'year']).apply(calc_zeros).reset_index()
zeros.columns = ['Stkcd', 'year', 'Zeros']
print(f"Zeros: {zeros['Zeros'].notna().sum()} obs")
print(f"  Mean={zeros['Zeros'].mean():.4f}, "
      f"Median={zeros['Zeros'].median():.4f}, "
      f"SD={zeros['Zeros'].std():.4f}")

# ============================================================
# 2. Kothari et al. (2005) 业绩调整Jones模型 absDA
# ============================================================
print("\n" + "=" * 60)
print("2. 构造 absDA (Kothari 2005)")
print("=" * 60)


def annual_report(df):
    df = df.copy()
    df['Accper'] = df['Accper'].astype(str)
    mask = df['Accper'].str.endswith('12-31') & (df['Typrep'] == 'A')
    df = df[mask].copy()
    df['year'] = pd.to_datetime(df['Accper']).dt.year
    df['Stkcd'] = df['Stkcd'].astype(int)
    return df.drop_duplicates(['Stkcd', 'year'], keep='last')


bs = annual_report(pd.read_parquet(f"{BASE}/data_parquet/balance_sheet.parquet"))
inc = annual_report(pd.read_parquet(f"{BASE}/data_parquet/income_stmt.parquet"))
cf = annual_report(pd.read_parquet(f"{BASE}/data_parquet/cashflow.parquet"))

# 合并 (不需要PPE)
fin = bs[['Stkcd', 'year', 'A001000000', 'A001107000']].merge(
    inc[['Stkcd', 'year', 'B001101000', 'B002000000']], on=['Stkcd', 'year'], how='inner'
).merge(
    cf[['Stkcd', 'year', 'C001000000']], on=['Stkcd', 'year'], how='inner'
)

fin = fin.rename(columns={
    'A001000000': 'TotalAssets',
    'A001107000': 'Receivables',
    'B001101000': 'Revenue',
    'B002000000': 'NetIncome',
    'C001000000': 'CFO',
})

for c in ['TotalAssets', 'Receivables', 'Revenue', 'NetIncome', 'CFO']:
    fin[c] = pd.to_numeric(fin[c], errors='coerce')

# 应收账款缺失填0
fin['Receivables'] = fin['Receivables'].fillna(0)

fin = fin.sort_values(['Stkcd', 'year'])

# 构造变量
fin['lagTA'] = fin.groupby('Stkcd')['TotalAssets'].shift(1)
fin['dRev'] = fin['Revenue'] - fin.groupby('Stkcd')['Revenue'].shift(1)
fin['dRec'] = fin['Receivables'] - fin.groupby('Stkcd')['Receivables'].shift(1)

# 总应计 = 净利润 - 经营现金流
fin['TA_accrual'] = fin['NetIncome'] - fin['CFO']

# ROA = NetIncome / lagTA (用于Kothari业绩调整)
fin['ROA_jones'] = fin['NetIncome'] / fin['lagTA']

# 标准化
fin['TA_scaled'] = fin['TA_accrual'] / fin['lagTA']
fin['inv_lagTA'] = 1.0 / fin['lagTA']
fin['dRev_adj_scaled'] = (fin['dRev'] - fin['dRec']) / fin['lagTA']

# 加载行业信息
fi = pd.read_parquet(f"{BASE}/data_parquet/firm_info.parquet",
                     columns=['Symbol', 'EndDate', 'IndustryCodeC'])
fi = fi.rename(columns={'Symbol': 'Stkcd'})
fi['Stkcd'] = fi['Stkcd'].astype(int)
fi['EndDate'] = pd.to_datetime(fi['EndDate'])
fi['year'] = fi['EndDate'].dt.year
fi = fi.sort_values(['Stkcd', 'year', 'EndDate']).drop_duplicates(
    subset=['Stkcd', 'year'], keep='last')
fi['Ind2'] = fi['IndustryCodeC'].str[:3]

fin = fin.merge(fi[['Stkcd', 'year', 'Ind2']], on=['Stkcd', 'year'], how='left')
ind_mode = fi.groupby('Stkcd')['Ind2'].agg(
    lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else np.nan)
fin = fin.merge(ind_mode.rename('Ind2_mode').reset_index(), on='Stkcd', how='left')
fin['Ind2'] = fin['Ind2'].fillna(fin['Ind2_mode'])
fin = fin[~fin['Ind2'].str.startswith('J', na=False)]

# Kothari模型回归变量: TA/A = α(1/A) + β1(ΔRev-ΔRec)/A + β2(ROA) + ε
jones_vars = ['TA_scaled', 'inv_lagTA', 'dRev_adj_scaled', 'ROA_jones']
fin_reg = fin.dropna(subset=jones_vars + ['Ind2']).copy()

# 剔除极端值 (仅winsorize, 不删除)
for v in jones_vars:
    lo, hi = fin_reg[v].quantile([0.01, 0.99])
    fin_reg[v] = fin_reg[v].clip(lo, hi)

print(f"Kothari模型回归样本: {len(fin_reg)} obs")

# 分行业-年度回归
results = []
skipped = 0
for (ind, yr), grp in fin_reg.groupby(['Ind2', 'year']):
    if len(grp) < 10:
        skipped += len(grp)
        continue
    y = grp['TA_scaled'].values
    X = grp[['inv_lagTA', 'dRev_adj_scaled', 'ROA_jones']].values
    X = sm.add_constant(X)
    try:
        model = sm.OLS(y, X).fit()
        for idx, r in zip(grp.index, model.resid):
            results.append({
                'Stkcd': grp.loc[idx, 'Stkcd'],
                'year': grp.loc[idx, 'year'],
                'absDA': abs(r)
            })
    except Exception:
        skipped += len(grp)

da_df = pd.DataFrame(results)
print(f"absDA: {len(da_df)} obs (跳过 {skipped} obs因行业-年度组太小)")
print(f"  Mean={da_df['absDA'].mean():.4f}, "
      f"Median={da_df['absDA'].median():.4f}, "
      f"SD={da_df['absDA'].std():.4f}")

# ============================================================
# 3. 合并至回归样本
# ============================================================
print("\n" + "=" * 60)
print("3. 合并至回归样本")
print("=" * 60)

reg = pd.read_stata(f"{BASE}/data_stata/reg_sample_v3.dta")
print(f"原始样本: {len(reg)} obs")

reg = reg.merge(zeros, on=['Stkcd', 'year'], how='left')
print(f"Zeros匹配率: {reg['Zeros'].notna().mean():.1%}")

reg = reg.merge(da_df[['Stkcd', 'year', 'absDA']], on=['Stkcd', 'year'], how='left')
print(f"absDA匹配率: {reg['absDA'].notna().mean():.1%}")

# Winsorize新变量
def winsorize(s):
    lo, hi = s.quantile([0.01, 0.99])
    return s.clip(lo, hi)

for v in ['Zeros', 'absDA']:
    mask = reg[v].notna()
    reg.loc[mask, v] = winsorize(reg.loc[mask, v])

print("\n新变量描述性统计:")
for v in ['Zeros', 'absDA']:
    s = reg[v].dropna()
    print(f"  {v}: N={len(s)}, Mean={s.mean():.4f}, SD={s.std():.4f}, "
          f"Min={s.min():.4f}, Median={s.median():.4f}, Max={s.max():.4f}")

# 导出
out_path = f"{BASE}/data_stata/reg_sample_v4.dta"
reg.to_stata(out_path, write_index=False, version=118)
print(f"\n导出至 {out_path}")
print(f"N={len(reg):,}, 新增变量: Zeros, absDA")
