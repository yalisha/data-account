"""
异质性分析 v3: 调整分组标准

变更:
  1. 企业规模: 三分位（前1/3 vs 后1/3），去掉中间组
  2. 新增融资约束维度: SA index (Hadlock & Pierce 2010) 中位数分组
  3. 保留: 产权性质、分析师覆盖、行业属性
  4. 新增: 组间差异 Fisher 检验

SA index = -0.737 × Size + 0.043 × Size² - 0.040 × Age
其中 Size = ln(总资产/1e6), Age = 上市年限(年)
SA 绝对值越大 → 融资约束越高
"""

import pandas as pd
import numpy as np
import pyfixest as pf
import warnings, os
from scipy import stats
warnings.filterwarnings('ignore')

OUT_DIR = "/Users/mac/computerscience/15会计研究/data_parquet"
RES_DIR = "/Users/mac/computerscience/15会计研究/results"
os.makedirs(RES_DIR, exist_ok=True)

# ================================================================
# 1. 复用 run_regression_v2.py 的数据加载和预处理
# ================================================================
print("=" * 60)
print("加载数据（复用 v2 流程）")
print("=" * 60)

panel = pd.read_parquet(f"{OUT_DIR}/panel.parquet")
ar_feat = pd.read_parquet(f"{OUT_DIR}/annual_report_features.parquet")

panel = panel.merge(
    ar_feat[['Stkcd', 'year', 'kw_total', 'kw_per10k', 'mda_kw_total',
             'mda_kw_per10k', 'substantive_ratio', 'substantive_count',
             'kw_data_stock', 'kw_data_dev', 'kw_data_app', 'kw_data_value',
             'kw_data_gov', 'has_mda']],
    on=['Stkcd', 'year'], how='left'
)

fi = pd.read_parquet(f"{OUT_DIR}/firm_info.parquet",
                     columns=['Symbol', 'EndDate', 'ShortName', 'IndustryCodeC', 'LISTINGSTATE'])
fi = fi.rename(columns={'Symbol': 'Stkcd'})
fi['EndDate'] = pd.to_datetime(fi['EndDate'])
fi['year'] = fi['EndDate'].dt.year
fi = fi.sort_values(['Stkcd', 'year', 'EndDate']).drop_duplicates(
    subset=['Stkcd', 'year'], keep='last')
panel = panel.merge(fi[['Stkcd', 'year', 'IndustryCodeC', 'LISTINGSTATE']],
                    on=['Stkcd', 'year'], how='left')

# 样本筛选
mask_fin = panel['IndustryCodeC'].str.startswith('J', na=False)
mask_st = panel['LISTINGSTATE'].isin(['ST', '*ST'])
mask_new = panel['Age'] <= 0
panel = panel[~mask_fin & ~mask_st & ~mask_new]

# 构造变量
panel['Ind2'] = panel['IndustryCodeC'].str[:3]
panel['IndYear'] = panel['Ind2'].astype(str) + '_' + panel['year'].astype(str)
panel['DU_kw'] = panel['kw_per10k']
panel['DU_kw_ln'] = np.log1p(panel['kw_total'])
panel['DU_sub_ln'] = np.log1p(panel['substantive_count'])
panel['DU_ratio'] = panel['substantive_ratio']

# t-1 年年报
panel = panel.sort_values(['Stkcd', 'year'])
for v in ['DU_kw', 'DU_kw_ln', 'DU_sub_ln', 'DU_ratio']:
    panel[v] = panel.groupby('Stkcd')[v].shift(1)

# Winsorize
def winsorize(s, lower=0.01, upper=0.99):
    q_lo, q_hi = s.quantile([lower, upper])
    return s.clip(q_lo, q_hi)

cont_vars = ['PriceDelay', 'DU_kw', 'DU_kw_ln', 'DU_sub_ln',
             'Lev', 'FinAsset', 'DataAsset', 'ROA', 'Growth', 'Size',
             'TobinQ', 'Age', 'BoardSize', 'IndepRatio', 'Top1Share',
             'InstHold', 'Amihud', 'Analyst']
for v in cont_vars:
    if v in panel.columns and panel[v].notna().any():
        panel[v] = winsorize(panel[v])

controls = ['Size', 'Lev', 'ROA', 'TobinQ', 'Age', 'Growth',
            'BoardSize', 'IndepRatio', 'Dual', 'Top1Share',
            'SOE', 'InstHold', 'Amihud', 'Analyst', 'AuditType']

reg_fe = panel.dropna(subset=['PriceDelay', 'DU_kw'] + controls).copy()
reg_fe['Stkcd'] = reg_fe['Stkcd'].astype(str)
reg_fe['year'] = reg_fe['year'].astype(str)

print(f"回归样本: {len(reg_fe):,} obs")

# ================================================================
# 2. 构造 SA 融资约束指数
# ================================================================
print("\n" + "=" * 60)
print("构造 SA 融资约束指数")
print("=" * 60)

# SA = -0.737 × Size_adj + 0.043 × Size_adj² - 0.040 × Age_years
# Size_adj = ln(总资产/1e6) = Size_raw - ln(1e6)
# Age_years = exp(Age) 因为 Age = ln(上市年限)
reg_fe['Size_adj'] = reg_fe['Size'] - np.log(1e6)  # ~8.8
reg_fe['Age_years'] = np.exp(reg_fe['Age'])  # 还原为实际年数
reg_fe['SA'] = -0.737 * reg_fe['Size_adj'] + 0.043 * reg_fe['Size_adj']**2 - 0.040 * reg_fe['Age_years']

# SA 绝对值越大 → 融资约束越高
# 但注意 SA 的符号：对于中国A股，SA 通常为负值
# |SA| 越大 = 约束越强
reg_fe['SA_abs'] = reg_fe['SA'].abs()

print(f"SA 描述性统计:")
print(f"  Mean: {reg_fe['SA'].mean():.3f}")
print(f"  Std:  {reg_fe['SA'].std():.3f}")
print(f"  Min:  {reg_fe['SA'].min():.3f}")
print(f"  Max:  {reg_fe['SA'].max():.3f}")
print(f"  |SA| Mean: {reg_fe['SA_abs'].mean():.3f}")

sa_median = reg_fe['SA_abs'].median()
print(f"  |SA| Median: {sa_median:.3f}")

# ================================================================
# 3. 异质性回归
# ================================================================
print("\n" + "=" * 60)
print("异质性回归")
print("=" * 60)

base_fml = f"PriceDelay ~ DU_kw + {' + '.join(controls)} | Stkcd + year"
hightech = {'C39', 'I65', 'C35', 'C27', 'C40'}

# 规模三分位
size_q33 = reg_fe['Size'].quantile(1/3)
size_q67 = reg_fe['Size'].quantile(2/3)
print(f"\n规模三分位: Q33={size_q33:.3f}, Q67={size_q67:.3f}")
print(f"  小: {(reg_fe.Size < size_q33).sum():,}")
print(f"  中: {((reg_fe.Size >= size_q33) & (reg_fe.Size < size_q67)).sum():,}")
print(f"  大: {(reg_fe.Size >= size_q67).sum():,}")

# SA 中位数分组
print(f"\nSA 中位数: |SA|={sa_median:.3f}")
print(f"  高约束 (|SA|>=median): {(reg_fe.SA_abs >= sa_median).sum():,}")
print(f"  低约束 (|SA|<median):  {(reg_fe.SA_abs < sa_median).sum():,}")

def run_het(label, sub_df):
    """运行异质性子样本回归"""
    try:
        m = pf.feols(base_fml, data=sub_df, vcov={"CRV1": "IndYear"})
        c = m.coef().get('DU_kw', np.nan)
        t_val = m.tstat().get('DU_kw', np.nan)
        p_val = m.pvalue().get('DU_kw', np.nan)
        se = m.se().get('DU_kw', np.nan)
        sig = '***' if p_val < 0.01 else '**' if p_val < 0.05 else '*' if p_val < 0.1 else ''
        print(f"  {label:12s} (N={m._N:>6,}): coef={c:>10.6f}, se={se:.6f}, t={t_val:>7.3f}, p={p_val:.4f} {sig}")
        return {'Group': label, 'Coef': c, 'SE': se, 't': t_val, 'p': p_val,
                'Sig': sig, 'N': m._N, 'R2': m._r2}
    except Exception as e:
        print(f"  {label}: ERROR {e}")
        return None

def fisher_test(coef1, se1, coef2, se2):
    """组间系数差异 Fisher 检验"""
    z = (coef1 - coef2) / np.sqrt(se1**2 + se2**2)
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return z, p

results = []
fisher_pairs = []

# (1) 产权性质
print("\n--- (1) 产权性质 ---")
r1 = run_het('国企', reg_fe[reg_fe.SOE == 1])
r2 = run_het('民企', reg_fe[reg_fe.SOE == 0])
results.extend([r for r in [r1, r2] if r])
if r1 and r2:
    z, p = fisher_test(r1['Coef'], r1['SE'], r2['Coef'], r2['SE'])
    print(f"  Fisher 组间差异: z={z:.3f}, p={p:.4f}")
    fisher_pairs.append(('产权性质', z, p))

# (2) 企业规模（三分位）
print("\n--- (2) 企业规模（三分位：前1/3 vs 后1/3） ---")
r1 = run_het('大企业(T)', reg_fe[reg_fe.Size >= size_q67])
r2 = run_het('小企业(T)', reg_fe[reg_fe.Size < size_q33])
results.extend([r for r in [r1, r2] if r])
if r1 and r2:
    z, p = fisher_test(r1['Coef'], r1['SE'], r2['Coef'], r2['SE'])
    print(f"  Fisher 组间差异: z={z:.3f}, p={p:.4f}")
    fisher_pairs.append(('企业规模(三分位)', z, p))

# (2b) 企业规模（中位数，保留对照）
print("\n--- (2b) 企业规模（中位数，对照） ---")
r1b = run_het('大企业(M)', reg_fe[reg_fe.Size >= reg_fe.Size.median()])
r2b = run_het('小企业(M)', reg_fe[reg_fe.Size < reg_fe.Size.median()])
results.extend([r for r in [r1b, r2b] if r])
if r1b and r2b:
    z, p = fisher_test(r1b['Coef'], r1b['SE'], r2b['Coef'], r2b['SE'])
    print(f"  Fisher 组间差异: z={z:.3f}, p={p:.4f}")
    fisher_pairs.append(('企业规模(中位数)', z, p))

# (3) 分析师覆盖
print("\n--- (3) 信息中介活跃度（分析师覆盖） ---")
r1 = run_het('高分析师', reg_fe[reg_fe.Analyst >= reg_fe.Analyst.median()])
r2 = run_het('低分析师', reg_fe[reg_fe.Analyst < reg_fe.Analyst.median()])
results.extend([r for r in [r1, r2] if r])
if r1 and r2:
    z, p = fisher_test(r1['Coef'], r1['SE'], r2['Coef'], r2['SE'])
    print(f"  Fisher 组间差异: z={z:.3f}, p={p:.4f}")
    fisher_pairs.append(('信息中介活跃度', z, p))

# (4) 融资约束（SA index）
print("\n--- (4) 融资约束（SA index） ---")
r1 = run_het('高约束', reg_fe[reg_fe.SA_abs >= sa_median])
r2 = run_het('低约束', reg_fe[reg_fe.SA_abs < sa_median])
results.extend([r for r in [r1, r2] if r])
if r1 and r2:
    z, p = fisher_test(r1['Coef'], r1['SE'], r2['Coef'], r2['SE'])
    print(f"  Fisher 组间差异: z={z:.3f}, p={p:.4f}")
    fisher_pairs.append(('融资约束(SA)', z, p))

# (5) 行业属性
print("\n--- (5) 行业属性 ---")
r1 = run_het('高科技', reg_fe[reg_fe.Ind2.isin(hightech)])
r2 = run_het('传统', reg_fe[~reg_fe.Ind2.isin(hightech)])
results.extend([r for r in [r1, r2] if r])
if r1 and r2:
    z, p = fisher_test(r1['Coef'], r1['SE'], r2['Coef'], r2['SE'])
    print(f"  Fisher 组间差异: z={z:.3f}, p={p:.4f}")
    fisher_pairs.append(('行业属性', z, p))

# ================================================================
# 4. 保存结果
# ================================================================
print("\n" + "=" * 60)
print("保存结果")
print("=" * 60)

df_res = pd.DataFrame(results)
df_res.to_csv(f"{RES_DIR}/heterogeneity_v3.csv", index=False)
print(f"已保存: heterogeneity_v3.csv")

df_fisher = pd.DataFrame(fisher_pairs, columns=['Dimension', 'z', 'p'])
df_fisher.to_csv(f"{RES_DIR}/heterogeneity_v3_fisher.csv", index=False)
print(f"已保存: heterogeneity_v3_fisher.csv")

print("\n" + "=" * 60)
print("Fisher 组间差异汇总")
print("=" * 60)
for _, row in df_fisher.iterrows():
    sig = '***' if row.p < 0.01 else '**' if row.p < 0.05 else '*' if row.p < 0.1 else ''
    print(f"  {row.Dimension:20s}: z={row.z:>7.3f}, p={row.p:.4f} {sig}")

print("\n完成！")
