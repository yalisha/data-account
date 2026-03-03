"""
主检验二：投资组合回测

设计:
- 每年6月按DataUsage_KW五分组
- 计算7月至次年6月月度等权组合收益
- Q5-Q1多空组合alpha检验: CAPM / FF3 / FF5+Momentum
- 构建DAT因子 = Q5 - Q1
- GRS检验
- 子期对比: 入表前 vs 入表后
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings, os
warnings.filterwarnings('ignore')

OUT_DIR = "/Users/mac/computerscience/15会计研究/data_parquet"
RES_DIR = "/Users/mac/computerscience/15会计研究/results"
os.makedirs(RES_DIR, exist_ok=True)

# ================================================================
# 1. 加载数据
# ================================================================
print("=" * 60)
print("1. 加载数据")
print("=" * 60)

# 月度个股回报率 (沪深A股)
mr = pd.read_parquet(f"{OUT_DIR}/monthly_return.parquet")
mr = mr[mr['Markettype'].isin([1, 4, 16, 32])].copy()
mr['Mretwd'] = pd.to_numeric(mr['Mretwd'], errors='coerce')
mr['Msmvttl'] = pd.to_numeric(mr['Msmvttl'], errors='coerce')
mr = mr.dropna(subset=['Mretwd'])
# Parse Trdmnt -> year, month
mr['ym'] = pd.to_datetime(mr['Trdmnt'] + '-01')
mr['year'] = mr['ym'].dt.year
mr['month'] = mr['ym'].dt.month
print(f"  月度回报: {len(mr):,} obs, {mr.Stkcd.nunique():,} stocks")

# 年报关键词指标
kw = pd.read_parquet(f"{OUT_DIR}/annual_report_kw_scores.parquet")
kw = kw[['Stkcd', 'year', 'DataUsage_KW']].copy()
print(f"  关键词指标: {len(kw):,} obs")

# FF3 日频 -> 月频 (P9714=综合A股)
ff3 = pd.read_parquet(f"{OUT_DIR}/ff3_daily.parquet")
ff3 = ff3[ff3['MarkettypeID'] == 'P9714'].copy()
ff3['TradingDate'] = pd.to_datetime(ff3['TradingDate'])
ff3['ym'] = ff3['TradingDate'].dt.to_period('M')
for col in ['RiskPremium1', 'SMB1', 'HML1']:
    ff3[col] = pd.to_numeric(ff3[col], errors='coerce')

# 日频 -> 月频: 复合收益
ff3_m = ff3.groupby('ym').agg(
    MKT=('RiskPremium1', lambda x: (1 + x).prod() - 1),
    SMB=('SMB1', lambda x: (1 + x).prod() - 1),
    HML=('HML1', lambda x: (1 + x).prod() - 1),
    n_days=('RiskPremium1', 'count')
).reset_index()
ff3_m['ym_dt'] = ff3_m['ym'].dt.to_timestamp()
print(f"  FF3月频: {len(ff3_m)} months, {ff3_m.ym.min()} ~ {ff3_m.ym.max()}")

# FF5 日频 -> 月频
ff5 = pd.read_parquet(f"{OUT_DIR}/ff5_daily.parquet")
ff5 = ff5[ff5['MarkettypeID'] == 'P9714'].copy()
ff5['TradingDate'] = pd.to_datetime(ff5['TradingDate'])
ff5['ym'] = ff5['TradingDate'].dt.to_period('M')
for col in ['RiskPremium1', 'SMB1', 'HML1', 'RMW1', 'CMA1']:
    ff5[col] = pd.to_numeric(ff5[col], errors='coerce')

ff5_m = ff5.groupby('ym').agg(
    MKT5=('RiskPremium1', lambda x: (1 + x).prod() - 1),
    SMB5=('SMB1', lambda x: (1 + x).prod() - 1),
    HML5=('HML1', lambda x: (1 + x).prod() - 1),
    RMW=('RMW1', lambda x: (1 + x).prod() - 1),
    CMA=('CMA1', lambda x: (1 + x).prod() - 1),
).reset_index()
ff5_m['ym_dt'] = ff5_m['ym'].dt.to_timestamp()
print(f"  FF5月频: {len(ff5_m)} months")

# Momentum: P9714, FormationPeriod=1 (2-12月), StockClass=0, Quantile=30%
mom = pd.read_parquet(f"{OUT_DIR}/momentum.parquet")
mom_f = mom[(mom.MarkettypeID == 'P9714') &
            (mom.FormationPeriod == 1) &
            (mom.StockClass == 0) &
            (mom.Quantile == '30%')].copy()
mom_f['ym_dt'] = pd.to_datetime(mom_f['TradingMonth'] + '-01')
mom_f['UMD'] = pd.to_numeric(mom_f['MomRe1'], errors='coerce')
mom_f = mom_f[['ym_dt', 'UMD']].dropna()
print(f"  UMD月频: {len(mom_f)} months")

# ================================================================
# 2. 组合构建
# ================================================================
print("\n" + "=" * 60)
print("2. 组合构建: 每年6月按DataUsage_KW五分组")
print("=" * 60)

# 排序年: 用t-1年的DataUsage_KW, 在t年6月排序, 持有t年7月到t+1年6月
# 例: 2011年年报 -> 2012年6月排序 -> 2012年7月~2013年6月持有
kw['sort_year'] = kw['year'] + 1  # 报告年+1 = 排序年

# 合并行业信息 (排除金融)
fi = pd.read_parquet(f"{OUT_DIR}/firm_info.parquet",
                     columns=['Symbol', 'EndDate', 'IndustryCodeC', 'LISTINGSTATE'])
fi = fi.rename(columns={'Symbol': 'Stkcd'})
fi['EndDate'] = pd.to_datetime(fi['EndDate'])
fi['year'] = fi['EndDate'].dt.year
fi = fi.sort_values(['Stkcd', 'year', 'EndDate']).drop_duplicates(
    subset=['Stkcd', 'year'], keep='last')

kw = kw.merge(fi[['Stkcd', 'year', 'IndustryCodeC', 'LISTINGSTATE']],
              on=['Stkcd', 'year'], how='left')
# 排除金融和ST
kw = kw[~kw['IndustryCodeC'].str.startswith('J', na=False)]
kw = kw[~kw['LISTINGSTATE'].isin(['ST', '*ST'])]

portfolio_returns = []

for sort_year in range(2012, 2025):  # 2011年报 -> 2012排序 -> 2012.7~2013.6
    # 获取t-1年DataUsage_KW
    scores = kw[kw['sort_year'] == sort_year][['Stkcd', 'DataUsage_KW']].dropna()
    if len(scores) < 100:
        continue

    # 五分位排序
    scores['quintile'] = pd.qcut(scores['DataUsage_KW'], 5, labels=[1, 2, 3, 4, 5],
                                  duplicates='drop')
    if scores['quintile'].isna().all():
        continue

    # 持有期: sort_year年7月到sort_year+1年6月
    hold_start = f"{sort_year}-07-01"
    hold_end = f"{sort_year + 1}-06-30"

    # 月度回报
    hold_mr = mr[(mr.ym >= hold_start) & (mr.ym <= hold_end)].copy()
    hold_mr = hold_mr.merge(scores[['Stkcd', 'quintile']], on='Stkcd', how='inner')

    # 等权组合月度回报
    for ym_val, grp in hold_mr.groupby('ym'):
        for q in range(1, 6):
            q_ret = grp[grp.quintile == q]['Mretwd']
            if len(q_ret) >= 5:
                portfolio_returns.append({
                    'ym': ym_val,
                    'quintile': q,
                    'ret': q_ret.mean(),
                    'n_stocks': len(q_ret),
                    'sort_year': sort_year,
                })

    n_per_q = scores.groupby('quintile').size()
    if sort_year % 3 == 0:
        print(f"  {sort_year}: {len(scores)} stocks, Q1={n_per_q.get(1,0)}, Q5={n_per_q.get(5,0)}")

port_df = pd.DataFrame(portfolio_returns)
port_df['ym_dt'] = pd.to_datetime(port_df['ym'])
print(f"\n  组合月度: {len(port_df):,} obs, {port_df.ym.nunique()} months")
print(f"  持有期: {port_df.ym.min()} ~ {port_df.ym.max()}")

# ================================================================
# 3. 多空组合 (Q5 - Q1) = DAT因子
# ================================================================
print("\n" + "=" * 60)
print("3. DAT因子 = Q5 - Q1")
print("=" * 60)

# Pivot: quintile -> columns
port_wide = port_df.pivot_table(index='ym_dt', columns='quintile', values='ret')
port_wide.columns = [f'Q{int(c)}' for c in port_wide.columns]
port_wide['DAT'] = port_wide['Q5'] - port_wide['Q1']
port_wide['Q5_Q1'] = port_wide['DAT']

print(f"  DAT因子: {len(port_wide)} months")
print(f"\n  各组合月均收益率:")
for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'DAT']:
    if q in port_wide.columns:
        m = port_wide[q].mean() * 100
        t = port_wide[q].mean() / (port_wide[q].std() / np.sqrt(len(port_wide)))
        print(f"    {q}: {m:.3f}% (t={t:.3f})")

# ================================================================
# 4. Alpha检验
# ================================================================
print("\n" + "=" * 60)
print("4. Alpha检验: CAPM / FF3 / FF5+MOM")
print("=" * 60)

# 合并因子
port_wide = port_wide.reset_index()
port_wide = port_wide.merge(ff3_m[['ym_dt', 'MKT', 'SMB', 'HML']], on='ym_dt', how='inner')
port_wide = port_wide.merge(ff5_m[['ym_dt', 'MKT5', 'SMB5', 'HML5', 'RMW', 'CMA']],
                            on='ym_dt', how='left')
port_wide = port_wide.merge(mom_f[['ym_dt', 'UMD']], on='ym_dt', how='left')
port_wide = port_wide.dropna(subset=['MKT'])

print(f"  合并因子后: {len(port_wide)} months")

def run_alpha(y, X_cols, label):
    """OLS回归计算alpha"""
    data = port_wide.dropna(subset=[y] + X_cols)
    Y = data[y].values
    X = np.column_stack([np.ones(len(data))] + [data[c].values for c in X_cols])
    try:
        beta = np.linalg.lstsq(X, Y, rcond=None)[0]
        resid = Y - X @ beta
        n, k = len(Y), len(beta)
        se = np.sqrt(np.sum(resid**2) / (n - k) * np.linalg.inv(X.T @ X).diagonal())
        t_stats = beta / se
        alpha = beta[0]
        alpha_se = se[0]
        alpha_t = t_stats[0]
        p_val = 2 * (1 - stats.t.cdf(abs(alpha_t), n - k))
        print(f"  {label}: alpha={alpha*100:.3f}% (t={alpha_t:.3f}, p={p_val:.4f}), N={n}")
        return alpha, alpha_t, p_val, n
    except Exception as e:
        print(f"  {label}: ERROR - {e}")
        return np.nan, np.nan, np.nan, 0

# 对各组合和DAT因子做alpha检验
results = []
for port_name in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'DAT']:
    if port_name not in port_wide.columns:
        continue
    print(f"\n--- {port_name} ---")
    # CAPM
    a, t, p, n = run_alpha(port_name, ['MKT'], 'CAPM')
    results.append({'Portfolio': port_name, 'Model': 'CAPM', 'alpha': a, 't': t, 'p': p, 'N': n})
    # FF3
    a, t, p, n = run_alpha(port_name, ['MKT', 'SMB', 'HML'], 'FF3')
    results.append({'Portfolio': port_name, 'Model': 'FF3', 'alpha': a, 't': t, 'p': p, 'N': n})
    # FF5
    cols5 = ['MKT5', 'SMB5', 'HML5', 'RMW', 'CMA']
    valid5 = [c for c in cols5 if c in port_wide.columns and port_wide[c].notna().sum() > 10]
    if len(valid5) >= 3:
        a, t, p, n = run_alpha(port_name, valid5, 'FF5')
        results.append({'Portfolio': port_name, 'Model': 'FF5', 'alpha': a, 't': t, 'p': p, 'N': n})
    # FF5+MOM
    cols5m = valid5 + ['UMD']
    valid5m = [c for c in cols5m if c in port_wide.columns and port_wide[c].notna().sum() > 10]
    if len(valid5m) >= 4:
        a, t, p, n = run_alpha(port_name, valid5m, 'FF5+MOM')
        results.append({'Portfolio': port_name, 'Model': 'FF5+MOM', 'alpha': a, 't': t, 'p': p, 'N': n})

# ================================================================
# 5. GRS检验
# ================================================================
print("\n" + "=" * 60)
print("5. GRS检验 (FF3)")
print("=" * 60)

# GRS = (T-N-K)/N * [1/(1+mean_f'*Omega_f^{-1}*mean_f)] * alpha' * Sigma^{-1} * alpha
# 简化版: F-test
try:
    T = len(port_wide)
    N_port = 5
    K = 3  # FF3 factors

    # Collect residuals from FF3 regressions for Q1-Q5
    alphas = []
    resid_mat = []
    factor_data = port_wide.dropna(subset=['MKT', 'SMB', 'HML'] + [f'Q{i}' for i in range(1,6)])
    X = np.column_stack([np.ones(len(factor_data)),
                         factor_data['MKT'].values,
                         factor_data['SMB'].values,
                         factor_data['HML'].values])

    for q in range(1, 6):
        Y = factor_data[f'Q{q}'].values
        beta = np.linalg.lstsq(X, Y, rcond=None)[0]
        alphas.append(beta[0])
        resid_mat.append(Y - X @ beta)

    alpha_vec = np.array(alphas)
    resid_mat = np.array(resid_mat).T  # T x N
    Sigma = resid_mat.T @ resid_mat / (T - K - 1)

    # Factor mean and covariance
    F = factor_data[['MKT', 'SMB', 'HML']].values
    f_mean = F.mean(axis=0)
    Omega_f = np.cov(F.T)

    # GRS statistic
    grs_num = alpha_vec @ np.linalg.inv(Sigma) @ alpha_vec
    grs_denom = 1 + f_mean @ np.linalg.inv(Omega_f) @ f_mean
    grs_stat = (T - N_port - K) / N_port * grs_num / grs_denom

    # F distribution p-value
    grs_p = 1 - stats.f.cdf(grs_stat, N_port, T - N_port - K)
    print(f"  GRS F-stat: {grs_stat:.4f}")
    print(f"  GRS p-value: {grs_p:.4f}")
    print(f"  解释: {'拒绝' if grs_p < 0.05 else '不拒绝'}所有alpha联合为零 (5%水平)")
except Exception as e:
    print(f"  GRS计算错误: {e}")

# ================================================================
# 6. 子期对比
# ================================================================
print("\n" + "=" * 60)
print("6. 子期对比: 入表前 vs 入表后")
print("=" * 60)

# 入表前: 2012.7 ~ 2024.6
pre = port_wide[port_wide.ym_dt < '2024-07-01']
# 入表后: 2024.7 ~ (如果有数据)
post = port_wide[port_wide.ym_dt >= '2024-07-01']

for period, data, label in [(pre, 'pre', '入表前(2012-2023)'), (post, 'post', '入表后(2024-)')]:
    if len(period) < 6:
        print(f"\n  {label}: 样本不足 ({len(period)} months)")
        continue
    print(f"\n--- {label} ({len(period)} months) ---")
    for q in ['Q1', 'Q5', 'DAT']:
        if q in period.columns:
            m = period[q].mean() * 100
            t_val = period[q].mean() / (period[q].std() / np.sqrt(len(period)))
            print(f"  {q}: {m:.3f}% (t={t_val:.3f})")
    # DAT alpha (FF3)
    valid = period.dropna(subset=['DAT', 'MKT', 'SMB', 'HML'])
    if len(valid) >= 6:
        Y = valid['DAT'].values
        X = np.column_stack([np.ones(len(valid)), valid['MKT'].values,
                             valid['SMB'].values, valid['HML'].values])
        beta = np.linalg.lstsq(X, Y, rcond=None)[0]
        resid = Y - X @ beta
        se = np.sqrt(np.sum(resid**2) / (len(Y)-4) * np.linalg.inv(X.T @ X).diagonal())
        alpha_t = beta[0] / se[0]
        print(f"  DAT FF3 alpha: {beta[0]*100:.3f}% (t={alpha_t:.3f})")

# ================================================================
# 7. 保存
# ================================================================
print("\n" + "=" * 60)
print("7. 保存结果")
print("=" * 60)

res_df = pd.DataFrame(results)
print(res_df.to_string(index=False))
res_df.to_csv(f"{RES_DIR}/portfolio_alpha.csv", index=False)

# 保存组合月度收益
port_wide.to_csv(f"{RES_DIR}/portfolio_monthly_returns.csv", index=False)
print(f"\n已保存: {RES_DIR}/portfolio_alpha.csv")
print(f"已保存: {RES_DIR}/portfolio_monthly_returns.csv")
print("完成！")
