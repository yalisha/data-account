"""
阶段四 v2: 用年报关键词特征 (2010-2024) 替换CNRDS指标重跑全部回归

自变量:
  - DU_kw: 上一年年报全文数据关键词频率 (每万字)
  - DU_kw_ln: 上一年 ln(1+关键词总数)
  - DU_sub_ln: 上一年 ln(1+实质利用次数)

模型:
  1. 基准回归: PriceDelay ~ DataUsage + Controls + FE(firm+year)
  2. DID: 2024入表新规
  3. 机制检验
  4. 异质性分析
"""

import pandas as pd
import numpy as np
import pyfixest as pf
import warnings, os
warnings.filterwarnings('ignore')

OUT_DIR = "/Users/mac/computerscience/15会计研究/data_parquet"
RES_DIR = "/Users/mac/computerscience/15会计研究/results"
os.makedirs(RES_DIR, exist_ok=True)

# ================================================================
# 1. 加载 & 合并数据
# ================================================================
print("=" * 60)
print("1. 加载数据")
print("=" * 60)

panel = pd.read_parquet(f"{OUT_DIR}/panel.parquet")
ar_feat = pd.read_parquet(f"{OUT_DIR}/annual_report_features.parquet")
print(f"  原始面板: {len(panel):,} obs")
print(f"  年报特征: {len(ar_feat):,} obs, {ar_feat.Stkcd.nunique()} firms")

# 合并年报特征
panel = panel.merge(
    ar_feat[['Stkcd', 'year', 'kw_total', 'kw_per10k', 'mda_kw_total',
             'mda_kw_per10k', 'substantive_ratio', 'substantive_count',
             'kw_data_stock', 'kw_data_dev', 'kw_data_app', 'kw_data_value',
             'kw_data_gov', 'has_mda']],
    on=['Stkcd', 'year'], how='left'
)
print(f"  合并后 kw_total 非空: {panel.kw_total.notna().sum():,}")

# 合并行业
fi = pd.read_parquet(f"{OUT_DIR}/firm_info.parquet",
                     columns=['Symbol', 'EndDate', 'ShortName', 'IndustryCodeC', 'LISTINGSTATE'])
fi = fi.rename(columns={'Symbol': 'Stkcd'})
fi['EndDate'] = pd.to_datetime(fi['EndDate'])
fi['year'] = fi['EndDate'].dt.year
fi = fi.sort_values(['Stkcd', 'year', 'EndDate']).drop_duplicates(
    subset=['Stkcd', 'year'], keep='last')
panel = panel.merge(fi[['Stkcd', 'year', 'IndustryCodeC', 'LISTINGSTATE']],
                    on=['Stkcd', 'year'], how='left')

# ================================================================
# 2. 样本筛选
# ================================================================
print("\n" + "=" * 60)
print("2. 样本筛选")
print("=" * 60)

n0 = len(panel)
mask_fin = panel['IndustryCodeC'].str.startswith('J', na=False)
mask_st = panel['LISTINGSTATE'].isin(['ST', '*ST'])
mask_new = panel['Age'] <= 0
print(f"  金融: {mask_fin.sum():,}, ST: {mask_st.sum():,}, 上市<1年: {mask_new.sum():,}")
panel = panel[~mask_fin & ~mask_st & ~mask_new]
print(f"  筛选后: {len(panel):,} obs")

# ================================================================
# 3. 构造回归变量
# ================================================================
print("\n" + "=" * 60)
print("3. 构造回归变量")
print("=" * 60)

panel['Ind2'] = panel['IndustryCodeC'].str[:3]
panel['IndYear'] = panel['Ind2'].astype(str) + '_' + panel['year'].astype(str)

# 主自变量
panel['DU_kw'] = panel['kw_per10k']
panel['DU_mda'] = panel['mda_kw_per10k']
panel['DU_sub'] = panel['substantive_count']
panel['DU_ratio'] = panel['substantive_ratio']
panel['DU_kw_ln'] = np.log1p(panel['kw_total'])
panel['DU_sub_ln'] = np.log1p(panel['substantive_count'])

# 时点对齐: 用t-1年年报解释t年PriceDelay, 避免前视偏差
panel = panel.sort_values(['Stkcd', 'year'])
for v in ['DU_kw', 'DU_mda', 'DU_sub', 'DU_ratio', 'DU_kw_ln', 'DU_sub_ln']:
    panel[v] = panel.groupby('Stkcd')[v].shift(1)

# Winsorize
def winsorize(s, lower=0.01, upper=0.99):
    q_lo, q_hi = s.quantile([lower, upper])
    return s.clip(q_lo, q_hi)

cont_vars = ['PriceDelay', 'DU_kw', 'DU_mda', 'DU_sub', 'DU_ratio',
             'DU_kw_ln', 'DU_sub_ln',
             'Lev', 'FinAsset', 'DataAsset', 'ROA', 'Growth', 'Size',
             'TobinQ', 'Age', 'BoardSize', 'IndepRatio', 'Top1Share',
             'InstHold', 'Amihud', 'Analyst']
for v in cont_vars:
    if v in panel.columns and panel[v].notna().any():
        panel[v] = winsorize(panel[v])

controls = ['Size', 'Lev', 'ROA', 'TobinQ', 'Age', 'Growth',
            'BoardSize', 'IndepRatio', 'Dual', 'Top1Share',
            'SOE', 'InstHold', 'Amihud', 'Analyst', 'AuditType']

# 回归样本
reg_df = panel.dropna(subset=['PriceDelay', 'DU_kw'] + controls).copy()
reg_df['Stkcd_str'] = reg_df['Stkcd'].astype(str)
reg_df['year_int'] = reg_df['year']
reg_df['year_str'] = reg_df['year'].astype(str)

print(f"  回归样本: {len(reg_df):,} obs, {reg_df.Stkcd.nunique():,} firms")
print(f"  年份: {reg_df.year.min()}-{reg_df.year.max()}")

# ================================================================
# 4. 描述性统计
# ================================================================
print("\n" + "=" * 60)
print("4. 描述性统计")
print("=" * 60)

desc_vars = ['PriceDelay', 'DU_kw', 'DU_kw_ln', 'DU_sub', 'DU_ratio'] + controls
desc = reg_df[desc_vars].describe(percentiles=[0.25, 0.5, 0.75]).T
desc = desc[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']]
desc.columns = ['N', 'Mean', 'Std', 'Min', 'P25', 'Median', 'P75', 'Max']
print(desc.round(4).to_string())
desc.round(4).to_csv(f"{RES_DIR}/descriptive_stats_v2.csv")

# ================================================================
# 5. 基准回归
# ================================================================
print("\n" + "=" * 60)
print("5. 基准回归")
print("=" * 60)

ctrl_str = ' + '.join(controls)

reg_fe = reg_df.copy()
reg_fe['Stkcd'] = reg_fe['Stkcd_str']
reg_fe['year'] = reg_fe['year_str']

models = {}

# M1: kw频率, 仅FE
models['(1)'] = pf.feols(f"PriceDelay ~ DU_kw | Stkcd + year", data=reg_fe, vcov={"CRV1": "IndYear"})
# M2: kw频率 + Controls
models['(2)'] = pf.feols(f"PriceDelay ~ DU_kw + {ctrl_str} | Stkcd + year", data=reg_fe, vcov={"CRV1": "IndYear"})
# M3: ln(1+kw) + Controls
models['(3)'] = pf.feols(f"PriceDelay ~ DU_kw_ln + {ctrl_str} | Stkcd + year", data=reg_fe, vcov={"CRV1": "IndYear"})
# M4: 实质利用次数(对数)
models['(4)'] = pf.feols(f"PriceDelay ~ DU_sub_ln + {ctrl_str} | Stkcd + year", data=reg_fe, vcov={"CRV1": "IndYear"})
# M5: Industry+Year FE
models['(5)'] = pf.feols(f"PriceDelay ~ DU_kw + {ctrl_str} | Ind2 + year", data=reg_fe, vcov={"CRV1": "IndYear"})
# M6: Industry+Year FE + ln
models['(6)'] = pf.feols(f"PriceDelay ~ DU_kw_ln + {ctrl_str} | Ind2 + year", data=reg_fe, vcov={"CRV1": "IndYear"})

for name, m in models.items():
    key_var = [v for v in m.coef().keys() if v.startswith('DU_')][0]
    coef = m.coef()[key_var]
    t = m.tstat()[key_var]
    p = m.pvalue()[key_var]
    sig = '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.1 else ''
    fe_type = 'Firm+Yr' if 'Stkcd' in str(m._fml) else 'Ind+Yr'
    print(f"  {name} [{fe_type}] {key_var}: coef={coef:.6f}, t={t:.3f}, p={p:.4f} {sig}")

# 打印关键模型
print("\n--- Model (2): Firm+Year FE ---")
print(models['(2)'].summary())
print("\n--- Model (5): Industry+Year FE ---")
print(models['(5)'].summary())

# ================================================================
# 6. DID: 2024入表新规
# ================================================================
print("\n" + "=" * 60)
print("6. DID: 2024入表新规")
print("=" * 60)

pre = reg_df[(reg_df.year_int >= 2019) & (reg_df.year_int <= 2023)]
firm_kw = pre.groupby('Stkcd')['DU_kw'].mean()
treat_th = firm_kw.median()
treat_firms = set(firm_kw[firm_kw >= treat_th].index)
print(f"  处理组: DU_kw均值 >= {treat_th:.2f} ({len(treat_firms):,} firms)")

did_df = reg_df.copy()
did_df['Treat'] = did_df['Stkcd'].isin(treat_firms).astype(int)
did_df['Post'] = (did_df['year_int'] >= 2024).astype(int)
did_df['TreatPost'] = did_df['Treat'] * did_df['Post']
did_df['Stkcd'] = did_df['Stkcd_str']
did_df['year'] = did_df['year_str']

# DID: 2021-2024
did_sub = did_df[(did_df.year_int >= 2021) & (did_df.year_int <= 2024)]
print(f"  DID样本: {len(did_sub):,} obs")
m_did = pf.feols(f"PriceDelay ~ TreatPost + {ctrl_str} | Stkcd + year", data=did_sub, vcov={"CRV1": "IndYear"})
print(m_did.summary())

# 平行趋势
did_full = did_df[(did_df.year_int >= 2019) & (did_df.year_int <= 2024)].copy()
for y in [2019, 2020, 2021, 2022, 2024]:
    did_full[f'Treat_Y{y}'] = did_full['Treat'] * (did_full['year_int'] == y).astype(int)
trend_vars = ' + '.join([f'Treat_Y{y}' for y in [2019, 2020, 2021, 2022, 2024]])
m_trend = pf.feols(f"PriceDelay ~ {trend_vars} + {ctrl_str} | Stkcd + year", data=did_full, vcov={"CRV1": "IndYear"})
print("\n--- 平行趋势 (基期=2023) ---")
for k in sorted([k for k in m_trend.coef().keys() if k.startswith('Treat_Y')]):
    c, t, p = m_trend.coef()[k], m_trend.tstat()[k], m_trend.pvalue()[k]
    sig = '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.1 else ''
    print(f"  {k}: coef={c:.6f}, t={t:.3f}, p={p:.4f} {sig}")

# ================================================================
# 7. 机制检验
# ================================================================
print("\n" + "=" * 60)
print("7. 机制检验")
print("=" * 60)

for dep, fml in [
    ('FinAsset', f"FinAsset ~ DU_kw + {ctrl_str} | Stkcd + year"),
    ('DataAsset', f"DataAsset ~ DU_kw + {ctrl_str} | Stkcd + year"),
    ('Analyst', f"Analyst ~ DU_kw + {ctrl_str.replace(' + Analyst', '')} | Stkcd + year"),
    ('InstHold', f"InstHold ~ DU_kw + {ctrl_str.replace(' + InstHold', '')} | Stkcd + year"),
]:
    sub = reg_fe.dropna(subset=[dep])
    m = pf.feols(fml, data=sub, vcov={"CRV1": "IndYear"})
    c = m.coef().get('DU_kw', np.nan)
    t = m.tstat().get('DU_kw', np.nan)
    p = m.pvalue().get('DU_kw', np.nan)
    sig = '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.1 else ''
    print(f"  DU_kw -> {dep}: coef={c:.6f}, t={t:.3f}, p={p:.4f} {sig} (N={m._N})")

# ================================================================
# 8. 异质性
# ================================================================
print("\n" + "=" * 60)
print("8. 异质性分析")
print("=" * 60)

base = f"PriceDelay ~ DU_kw + {ctrl_str} | Stkcd + year"
hightech = {'C39', 'I65', 'C35', 'C27', 'C40'}

for label, sub in [
    ('国企', reg_fe[reg_fe.SOE == 1]),
    ('民企', reg_fe[reg_fe.SOE == 0]),
    ('大企业', reg_fe[reg_fe.Size >= reg_fe.Size.median()]),
    ('小企业', reg_fe[reg_fe.Size < reg_fe.Size.median()]),
    ('高分析师', reg_fe[reg_fe.Analyst >= reg_fe.Analyst.median()]),
    ('低分析师', reg_fe[reg_fe.Analyst < reg_fe.Analyst.median()]),
    ('高科技', reg_fe[reg_fe.Ind2.isin(hightech)]),
    ('传统', reg_fe[~reg_fe.Ind2.isin(hightech)]),
]:
    try:
        m = pf.feols(base, data=sub, vcov={"CRV1": "IndYear"})
        c = m.coef().get('DU_kw', np.nan)
        t = m.tstat().get('DU_kw', np.nan)
        p = m.pvalue().get('DU_kw', np.nan)
        sig = '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.1 else ''
        print(f"  {label} (N={m._N:,}): coef={c:.6f}, t={t:.3f}, p={p:.4f} {sig}")
    except Exception as e:
        print(f"  {label}: ERROR {e}")

# ================================================================
# 9. 汇总
# ================================================================
print("\n" + "=" * 60)
print("9. 汇总保存")
print("=" * 60)

rows = []
for name, m in models.items():
    kv = [v for v in m.coef().keys() if v.startswith('DU_')][0]
    fe = 'Firm+Year' if 'Stkcd' in str(m._fml) else 'Ind+Year'
    rows.append({'Model': name, 'FE': fe, 'Var': kv,
                 'Coef': m.coef()[kv], 't': m.tstat()[kv], 'p': m.pvalue()[kv], 'N': m._N})
rows.append({'Model': 'DID', 'FE': 'Firm+Year', 'Var': 'TreatPost',
             'Coef': m_did.coef()['TreatPost'], 't': m_did.tstat()['TreatPost'],
             'p': m_did.pvalue()['TreatPost'], 'N': m_did._N})

pd.DataFrame(rows).to_csv(f"{RES_DIR}/regression_summary_v2.csv", index=False)
print(pd.DataFrame(rows).to_string(index=False))
print(f"\n已保存: {RES_DIR}/regression_summary_v2.csv")
print("完成！")
