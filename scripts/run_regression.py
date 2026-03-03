"""
阶段四：实证分析 — 基准回归 + DID + 机制 + 异质性

模型: PriceDelay = α + β·DataUsage + γ·Controls + FE(firm+year) + ε
聚类: 行业-年份 clustered standard errors
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
# 1. 加载数据 & 合并行业
# ================================================================
print("=" * 60)
print("1. 加载数据")
print("=" * 60)

panel = pd.read_parquet(f"{OUT_DIR}/panel.parquet")
print(f"  原始面板: {len(panel):,} obs, {panel.Stkcd.nunique():,} firms")

# 行业分类: 取每个firm-year最近的EndDate记录
fi = pd.read_parquet(f"{OUT_DIR}/firm_info.parquet",
                     columns=['Symbol', 'EndDate', 'ShortName', 'IndustryCodeC', 'LISTINGSTATE'])
fi = fi.rename(columns={'Symbol': 'Stkcd'})
fi['EndDate'] = pd.to_datetime(fi['EndDate'])
fi['year'] = fi['EndDate'].dt.year

# 每个firm-year保留一条 (取最近日期的记录)
fi = fi.sort_values(['Stkcd', 'year', 'EndDate']).drop_duplicates(
    subset=['Stkcd', 'year'], keep='last')

panel = panel.merge(fi[['Stkcd', 'year', 'IndustryCodeC', 'LISTINGSTATE', 'ShortName']],
                    on=['Stkcd', 'year'], how='left')
print(f"  合并行业后: IndustryCodeC非空 {panel.IndustryCodeC.notna().sum():,}")

# ================================================================
# 2. 样本筛选
# ================================================================
print("\n" + "=" * 60)
print("2. 样本筛选")
print("=" * 60)

n0 = len(panel)

# (a) 排除金融行业 (J开头)
mask_fin = panel['IndustryCodeC'].str.startswith('J', na=False)
print(f"  金融行业: {mask_fin.sum():,} obs")
panel = panel[~mask_fin]

# (b) 排除ST/*ST
mask_st = panel['LISTINGSTATE'].isin(['ST', '*ST'])
print(f"  ST/*ST: {mask_st.sum():,} obs")
panel = panel[~mask_st]

# (c) 排除上市不满1年 (Age=0 means listing year = current year)
mask_new = panel['Age'] <= 0
print(f"  上市不满1年: {mask_new.sum():,} obs")
panel = panel[~mask_new]

print(f"  筛选后: {len(panel):,} obs (排除 {n0 - len(panel):,})")

# ================================================================
# 3. 构造回归变量
# ================================================================
print("\n" + "=" * 60)
print("3. 构造回归变量")
print("=" * 60)

# 行业代码2位 (用于行业固定效应和聚类)
panel['Ind2'] = panel['IndustryCodeC'].str[:3]  # e.g. C39, K70, I65
panel['IndYear'] = panel['Ind2'].astype(str) + '_' + panel['year'].astype(str)

# 标准化 DataUsage (除以标准差, 便于解读系数)
du_std = panel['DataUsage'].std()
panel['DataUsage_std'] = panel['DataUsage'] / du_std if du_std > 0 else panel['DataUsage']
print(f"  DataUsage std = {du_std:.2f}, 标准化后1单位 = 1个标准差")

# Winsorize 连续变量 at 1%/99%
cont_vars = ['PriceDelay', 'DataUsage', 'Lev', 'FinAsset', 'DataAsset',
             'ROA', 'Growth', 'Size', 'TobinQ', 'Age', 'BoardSize',
             'IndepRatio', 'Top1Share', 'InstHold', 'Amihud', 'Analyst']

def winsorize(s, lower=0.01, upper=0.99):
    q_lo, q_hi = s.quantile([lower, upper])
    return s.clip(q_lo, q_hi)

for v in cont_vars:
    if v in panel.columns:
        panel[v] = winsorize(panel[v])

print(f"  连续变量 Winsorize (1%, 99%) 完成")

# 控制变量列表
controls = ['Size', 'Lev', 'ROA', 'TobinQ', 'Age', 'Growth',
            'BoardSize', 'IndepRatio', 'Dual', 'Top1Share',
            'SOE', 'InstHold', 'Amihud', 'Analyst', 'AuditType']

# 回归样本: 需要 DataUsage + PriceDelay + 所有控制变量非空
reg_vars = ['PriceDelay', 'DataUsage'] + controls + ['Stkcd', 'year', 'Ind2', 'IndYear']
reg_df = panel.dropna(subset=['PriceDelay', 'DataUsage'] + controls).copy()

# 确保 FE 变量为分类
reg_df['Stkcd'] = reg_df['Stkcd'].astype(str)
reg_df['year'] = reg_df['year'].astype(str)

print(f"  回归样本: {len(reg_df):,} obs, {reg_df.Stkcd.nunique():,} firms")
print(f"  年份: {sorted(reg_df.year.unique())}")

# ================================================================
# 4. 描述性统计
# ================================================================
print("\n" + "=" * 60)
print("4. 描述性统计")
print("=" * 60)

desc_vars = ['PriceDelay', 'DataUsage'] + controls
desc = reg_df[desc_vars].describe(percentiles=[0.25, 0.5, 0.75]).T
desc = desc[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']]
desc.columns = ['N', 'Mean', 'Std', 'Min', 'P25', 'Median', 'P75', 'Max']
print(desc.round(4).to_string())
desc.round(4).to_csv(f"{RES_DIR}/descriptive_stats.csv")
print(f"\n  已保存: {RES_DIR}/descriptive_stats.csv")

# ================================================================
# 5. 相关系数矩阵
# ================================================================
print("\n" + "=" * 60)
print("5. Pearson 相关系数 (关键变量)")
print("=" * 60)

corr_vars = ['PriceDelay', 'DataUsage', 'Size', 'Lev', 'ROA', 'TobinQ',
             'FinAsset', 'DataAsset', 'SOE', 'Analyst', 'InstHold', 'Amihud']
corr = reg_df[corr_vars].corr()
print(corr.round(3).to_string())
corr.round(4).to_csv(f"{RES_DIR}/correlation_matrix.csv")

# ================================================================
# 6. 基准回归
# ================================================================
print("\n" + "=" * 60)
print("6. 基准回归: PriceDelay ~ DataUsage + Controls + FE")
print("=" * 60)

ctrl_str = ' + '.join(controls)

# Model 1: 仅 DataUsage, 企业+年份FE, 行业-年聚类
fml1 = f"PriceDelay ~ DataUsage | Stkcd + year"
# Model 2: DataUsage + 全部控制变量
fml2 = f"PriceDelay ~ DataUsage + {ctrl_str} | Stkcd + year"
# Model 3: 标准化 DataUsage
fml3 = f"PriceDelay ~ DataUsage_std + {ctrl_str} | Stkcd + year"

print("\n--- Model 1: 仅DataUsage ---")
m1 = pf.feols(fml1, data=reg_df, vcov={"CRV1": "IndYear"})
print(m1.summary())

print("\n--- Model 2: DataUsage + 全部Controls ---")
m2 = pf.feols(fml2, data=reg_df, vcov={"CRV1": "IndYear"})
print(m2.summary())

print("\n--- Model 3: 标准化DataUsage + 全部Controls ---")
m3 = pf.feols(fml3, data=reg_df, vcov={"CRV1": "IndYear"})
print(m3.summary())

# 汇总表
print("\n--- 基准回归汇总 ---")
etable = pf.etable([m1, m2, m3],
                    labels={'DataUsage': '数据要素利用',
                            'DataUsage_std': '数据要素利用(标准化)'},
                    type='md')
print(etable)

# ================================================================
# 7. DID: 2024入表新规
# ================================================================
print("\n" + "=" * 60)
print("7. DID: 2024入表新规 × 数据利用强度")
print("=" * 60)

# 用完整面板 (不仅限DataUsage非空期间)
did_df = panel.dropna(subset=['PriceDelay'] + controls).copy()
did_df['Stkcd_str'] = did_df['Stkcd'].astype(str)
did_df['year_str'] = did_df['year'].astype(str)

# 处理组: DataUsage在2018-2023期间高于中位数的企业
du_pre = panel[(panel.year >= 2018) & (panel.year <= 2023) & panel.DataUsage.notna()]
du_median = du_pre.groupby('Stkcd')['DataUsage'].mean()
treat_threshold = du_median.median()
treat_firms = set(du_median[du_median >= treat_threshold].index)
print(f"  处理组定义: 2018-2023 DataUsage均值 >= 中位数({treat_threshold:.1f})")
print(f"  处理组: {len(treat_firms):,} firms")

did_df['Treat'] = did_df['Stkcd'].isin(treat_firms).astype(int)
did_df['Post'] = (did_df['year'] >= 2024).astype(int)
did_df['TreatPost'] = did_df['Treat'] * did_df['Post']

# 限制DID样本: 2021-2024 (前后各3年)
did_sub = did_df[(did_df.year >= 2021) & (did_df.year <= 2024)].copy()
did_sub['Stkcd'] = did_sub['Stkcd_str']
did_sub['year'] = did_sub['year_str']
print(f"  DID样本 (2021-2024): {len(did_sub):,} obs")

did_fml = f"PriceDelay ~ TreatPost + {ctrl_str} | Stkcd + year"
m_did = pf.feols(did_fml, data=did_sub, vcov={"CRV1": "IndYear"})
print("\n--- DID回归 ---")
print(m_did.summary())

# 平行趋势: 加年份交互项
did_full = did_df[(did_df.year >= 2019) & (did_df.year <= 2024)].copy()
did_full['Stkcd'] = did_full['Stkcd_str']
# 生成年份虚拟变量交互
for y in [2019, 2020, 2021, 2022, 2023, 2024]:
    did_full[f'Treat_Y{y}'] = did_full['Treat'] * (did_full['year'].astype(int) == y).astype(int)
did_full['year'] = did_full['year_str']

# 以2023为基期
trend_vars = ' + '.join([f'Treat_Y{y}' for y in [2019, 2020, 2021, 2022, 2024]])
trend_fml = f"PriceDelay ~ {trend_vars} + {ctrl_str} | Stkcd + year"
m_trend = pf.feols(trend_fml, data=did_full, vcov={"CRV1": "IndYear"})
print("\n--- 平行趋势检验 (基期=2023) ---")
print(m_trend.summary())

# ================================================================
# 8. 机制检验
# ================================================================
print("\n" + "=" * 60)
print("8. 机制检验")
print("=" * 60)

# 8a. 资产配置渠道: DataUsage → FinAsset
print("\n--- 8a. DataUsage → FinAsset (金融资产配置) ---")
mech_df = reg_df.dropna(subset=['FinAsset']).copy()
fml_fa = f"FinAsset ~ DataUsage + {ctrl_str} | Stkcd + year"
m_fa = pf.feols(fml_fa, data=mech_df, vcov={"CRV1": "IndYear"})
print(m_fa.summary())

# 8b. DataUsage → DataAsset (数据资产)
print("\n--- 8b. DataUsage → DataAsset (数据资产配置) ---")
mech_df2 = reg_df.dropna(subset=['DataAsset']).copy()
fml_da = f"DataAsset ~ DataUsage + {ctrl_str} | Stkcd + year"
m_da = pf.feols(fml_da, data=mech_df2, vcov={"CRV1": "IndYear"})
print(m_da.summary())

# 8c. 信息环境渠道: DataUsage → Analyst
print("\n--- 8c. DataUsage → Analyst (分析师覆盖) ---")
fml_an = f"Analyst ~ DataUsage + {ctrl_str.replace(' + Analyst', '')} | Stkcd + year"
m_an = pf.feols(fml_an, data=reg_df, vcov={"CRV1": "IndYear"})
print(m_an.summary())

# 8d. DataUsage → InstHold (机构持股)
print("\n--- 8d. DataUsage → InstHold (机构持股) ---")
ctrl_no_inst = ctrl_str.replace(' + InstHold', '')
fml_ih = f"InstHold ~ DataUsage + {ctrl_no_inst} | Stkcd + year"
m_ih = pf.feols(fml_ih, data=reg_df, vcov={"CRV1": "IndYear"})
print(m_ih.summary())

# ================================================================
# 9. 异质性分析
# ================================================================
print("\n" + "=" * 60)
print("9. 异质性分析")
print("=" * 60)

base_fml = f"PriceDelay ~ DataUsage + {ctrl_str} | Stkcd + year"

# 9a. 产权性质: 国企 vs 民企
print("\n--- 9a. 产权性质 ---")
soe_df = reg_df[reg_df['SOE'] == 1].copy()
nsoe_df = reg_df[reg_df['SOE'] == 0].copy()
m_soe = pf.feols(base_fml, data=soe_df, vcov={"CRV1": "IndYear"})
m_nsoe = pf.feols(base_fml, data=nsoe_df, vcov={"CRV1": "IndYear"})
print(f"  国企 (N={len(soe_df):,}): DataUsage coef={m_soe.coef().get('DataUsage', np.nan):.6f}, "
      f"t={m_soe.tstat().get('DataUsage', np.nan):.3f}")
print(f"  民企 (N={len(nsoe_df):,}): DataUsage coef={m_nsoe.coef().get('DataUsage', np.nan):.6f}, "
      f"t={m_nsoe.tstat().get('DataUsage', np.nan):.3f}")

# 9b. 企业规模: 大 vs 小 (以中位数分)
print("\n--- 9b. 企业规模 ---")
size_med = reg_df['Size'].median()
big_df = reg_df[reg_df['Size'] >= size_med].copy()
small_df = reg_df[reg_df['Size'] < size_med].copy()
m_big = pf.feols(base_fml, data=big_df, vcov={"CRV1": "IndYear"})
m_small = pf.feols(base_fml, data=small_df, vcov={"CRV1": "IndYear"})
print(f"  大企业 (N={len(big_df):,}): DataUsage coef={m_big.coef().get('DataUsage', np.nan):.6f}, "
      f"t={m_big.tstat().get('DataUsage', np.nan):.3f}")
print(f"  小企业 (N={len(small_df):,}): DataUsage coef={m_small.coef().get('DataUsage', np.nan):.6f}, "
      f"t={m_small.tstat().get('DataUsage', np.nan):.3f}")

# 9c. 分析师覆盖: 高 vs 低
print("\n--- 9c. 分析师覆盖 ---")
an_med = reg_df['Analyst'].median()
hi_an = reg_df[reg_df['Analyst'] >= an_med].copy()
lo_an = reg_df[reg_df['Analyst'] < an_med].copy()
m_hi_an = pf.feols(base_fml, data=hi_an, vcov={"CRV1": "IndYear"})
m_lo_an = pf.feols(base_fml, data=lo_an, vcov={"CRV1": "IndYear"})
print(f"  高分析师覆盖 (N={len(hi_an):,}): DataUsage coef={m_hi_an.coef().get('DataUsage', np.nan):.6f}, "
      f"t={m_hi_an.tstat().get('DataUsage', np.nan):.3f}")
print(f"  低分析师覆盖 (N={len(lo_an):,}): DataUsage coef={m_lo_an.coef().get('DataUsage', np.nan):.6f}, "
      f"t={m_lo_an.tstat().get('DataUsage', np.nan):.3f}")

# 9d. 行业: 高科技 vs 传统 (高科技: C39电子, I65软件, C35专用设备, C27医药)
print("\n--- 9d. 行业属性 ---")
hightech_inds = {'C39', 'I65', 'C35', 'C27', 'C40'}
reg_df['HighTech'] = reg_df['Ind2'].isin(hightech_inds).astype(int)
ht_df = reg_df[reg_df['HighTech'] == 1].copy()
trad_df = reg_df[reg_df['HighTech'] == 0].copy()
m_ht = pf.feols(base_fml, data=ht_df, vcov={"CRV1": "IndYear"})
m_trad = pf.feols(base_fml, data=trad_df, vcov={"CRV1": "IndYear"})
print(f"  高科技 (N={len(ht_df):,}): DataUsage coef={m_ht.coef().get('DataUsage', np.nan):.6f}, "
      f"t={m_ht.tstat().get('DataUsage', np.nan):.3f}")
print(f"  传统 (N={len(trad_df):,}): DataUsage coef={m_trad.coef().get('DataUsage', np.nan):.6f}, "
      f"t={m_trad.tstat().get('DataUsage', np.nan):.3f}")

# ================================================================
# 10. 保存回归结果汇总
# ================================================================
print("\n" + "=" * 60)
print("10. 结果汇总")
print("=" * 60)

def get_nobs(model):
    try:
        return model._N
    except AttributeError:
        try:
            return model.nobs
        except AttributeError:
            return len(model._Y)

def get_r2(model):
    try:
        return model._r2
    except AttributeError:
        try:
            return model.r2
        except (AttributeError, TypeError):
            return np.nan

results = {
    '基准M1_仅DU': {'coef': m1.coef().get('DataUsage', np.nan),
                    'se': m1.se().get('DataUsage', np.nan),
                    't': m1.tstat().get('DataUsage', np.nan),
                    'N': get_nobs(m1),
                    'R2': get_r2(m1)},
    '基准M2_全Controls': {'coef': m2.coef().get('DataUsage', np.nan),
                          'se': m2.se().get('DataUsage', np.nan),
                          't': m2.tstat().get('DataUsage', np.nan),
                          'N': get_nobs(m2),
                          'R2': get_r2(m2)},
    'DID_TreatPost': {'coef': m_did.coef().get('TreatPost', np.nan),
                      'se': m_did.se().get('TreatPost', np.nan),
                      't': m_did.tstat().get('TreatPost', np.nan),
                      'N': get_nobs(m_did),
                      'R2': get_r2(m_did)},
    '机制_FinAsset': {'coef': m_fa.coef().get('DataUsage', np.nan),
                      't': m_fa.tstat().get('DataUsage', np.nan),
                      'N': get_nobs(m_fa)},
    '机制_DataAsset': {'coef': m_da.coef().get('DataUsage', np.nan),
                       't': m_da.tstat().get('DataUsage', np.nan),
                       'N': get_nobs(m_da)},
    '机制_Analyst': {'coef': m_an.coef().get('DataUsage', np.nan),
                     't': m_an.tstat().get('DataUsage', np.nan),
                     'N': get_nobs(m_an)},
    '机制_InstHold': {'coef': m_ih.coef().get('DataUsage', np.nan),
                      't': m_ih.tstat().get('DataUsage', np.nan),
                      'N': get_nobs(m_ih)},
}

res_df = pd.DataFrame(results).T
print(res_df.round(6).to_string())
res_df.round(6).to_csv(f"{RES_DIR}/regression_summary.csv")

# ================================================================
# 11. 补充: 行业FE替代企业FE (仅3年数据，firm FE可能过度吸收)
# ================================================================
print("\n" + "=" * 60)
print("11. 补充: 行业FE替代企业FE")
print("=" * 60)

fml_ind = f"PriceDelay ~ DataUsage + {ctrl_str} | Ind2 + year"
m_ind = pf.feols(fml_ind, data=reg_df, vcov={"CRV1": "IndYear"})
print("\n--- Model: Industry+Year FE ---")
print(m_ind.summary())

# Pooled OLS with industry dummies
fml_ols = f"PriceDelay ~ DataUsage + {ctrl_str} + C(Ind2) + C(year)"
m_ols = pf.feols(fml_ols, data=reg_df, vcov={"CRV1": "IndYear"})
print("\n--- Pooled OLS with Industry + Year dummies ---")
print(m_ols.summary())

print(f"\n已保存: {RES_DIR}/regression_summary.csv")
print("完成！")
