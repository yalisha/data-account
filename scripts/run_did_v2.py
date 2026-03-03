"""
DID v2: 重新设计双重差分

核心修复:
1. 处理组用 2017-2021 年 DU_kw 均值定义 (完全在政策公告前)
2. 多种事件时点: 2024实施 / 2023公告
3. 连续处理 DID
4. 完整 event study
5. 安慰剂检验 (修正后处理组)
6. 所有结果保存 CSV
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
# 1. 数据准备 (与 run_regression_v2.py 一致)
# ================================================================
print("=" * 60)
print("1. 加载数据")
print("=" * 60)

panel = pd.read_parquet(f"{OUT_DIR}/panel.parquet")
ar_feat = pd.read_parquet(f"{OUT_DIR}/annual_report_features.parquet")

panel = panel.merge(
    ar_feat[['Stkcd', 'year', 'kw_total', 'kw_per10k', 'substantive_ratio',
             'substantive_count', 'kw_data_stock', 'kw_data_dev', 'kw_data_app',
             'kw_data_value', 'kw_data_gov']],
    on=['Stkcd', 'year'], how='left'
)

fi = pd.read_parquet(f"{OUT_DIR}/firm_info.parquet",
                     columns=['Symbol', 'EndDate', 'IndustryCodeC', 'LISTINGSTATE'])
fi = fi.rename(columns={'Symbol': 'Stkcd'})
fi['EndDate'] = pd.to_datetime(fi['EndDate'])
fi['year'] = fi['EndDate'].dt.year
fi = fi.sort_values(['Stkcd', 'year', 'EndDate']).drop_duplicates(
    subset=['Stkcd', 'year'], keep='last')
panel = panel.merge(fi[['Stkcd', 'year', 'IndustryCodeC', 'LISTINGSTATE']],
                    on=['Stkcd', 'year'], how='left')

# 筛选
mask_fin = panel['IndustryCodeC'].str.startswith('J', na=False)
mask_st = panel['LISTINGSTATE'].isin(['ST', '*ST'])
mask_new = panel['Age'] <= 0
panel = panel[~mask_fin & ~mask_st & ~mask_new]

# 变量
panel['Ind2'] = panel['IndustryCodeC'].str[:3]
panel['IndYear'] = panel['Ind2'].astype(str) + '_' + panel['year'].astype(str)
panel['DU_kw'] = panel['kw_per10k']
panel['DU_kw_ln'] = np.log1p(panel['kw_total'])
panel['DU_sub_ln'] = np.log1p(panel['substantive_count'])

# 时点对齐: t-1年年报 -> t年
panel = panel.sort_values(['Stkcd', 'year'])
for v in ['DU_kw', 'DU_kw_ln', 'DU_sub_ln']:
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
ctrl_str = ' + '.join(controls)

reg_df = panel.dropna(subset=['PriceDelay', 'DU_kw'] + controls).copy()
reg_df['Stkcd_str'] = reg_df['Stkcd'].astype(str)
reg_df['year_int'] = reg_df['year']
reg_df['year_str'] = reg_df['year'].astype(str)
print(f"  回归样本: {len(reg_df):,} obs, {reg_df.Stkcd.nunique():,} firms")
print(f"  年份: {reg_df.year.min()}-{reg_df.year.max()}")

# ================================================================
# 2. 处理组定义 (核心修复: 只用 pre-2022 数据)
# ================================================================
print("\n" + "=" * 60)
print("2. 处理组定义")
print("=" * 60)

# 方案A: 2017-2021年均值 (近5年, 完全在政策公告前)
pre_firms_a = reg_df[(reg_df.year_int >= 2017) & (reg_df.year_int <= 2021)]
firm_kw_a = pre_firms_a.groupby('Stkcd')['DU_kw'].mean()
th_a = firm_kw_a.median()
treat_a = set(firm_kw_a[firm_kw_a >= th_a].index)
print(f"  方案A (2017-2021 median): 阈值={th_a:.4f}, 处理组={len(treat_a)} firms")

# 方案B: 2011-2021年均值 (全部pre-policy数据)
pre_firms_b = reg_df[(reg_df.year_int >= 2011) & (reg_df.year_int <= 2021)]
firm_kw_b = pre_firms_b.groupby('Stkcd')['DU_kw'].mean()
th_b = firm_kw_b.median()
treat_b = set(firm_kw_b[firm_kw_b >= th_b].index)
print(f"  方案B (2011-2021 median): 阈值={th_b:.4f}, 处理组={len(treat_b)} firms")

# 方案C: 连续处理 (用2017-2021 DU_kw均值作为处理强度)
firm_kw_cont = firm_kw_a.to_dict()

# 对照: 旧方案 (2019-2023, 有内生性问题)
pre_old = reg_df[(reg_df.year_int >= 2019) & (reg_df.year_int <= 2023)]
firm_kw_old = pre_old.groupby('Stkcd')['DU_kw'].mean()
th_old = firm_kw_old.median()
treat_old = set(firm_kw_old[firm_kw_old >= th_old].index)
print(f"  旧方案 (2019-2023 median): 阈值={th_old:.4f}, 处理组={len(treat_old)} firms")
print(f"  新旧处理组重叠: {len(treat_a & treat_old)} firms")

# ================================================================
# 3. Design A: Binary DID, 2024事件, 窗口 2021-2024
# ================================================================
print("\n" + "=" * 60)
print("3. Design A: Binary DID (2024事件, 2021-2024窗口)")
print("=" * 60)

all_results = []

for treat_name, treat_set in [('2017-2021', treat_a), ('2011-2021', treat_b), ('旧2019-2023', treat_old)]:
    did_df = reg_df.copy()
    did_df['Treat'] = did_df['Stkcd'].isin(treat_set).astype(int)
    did_df['Post'] = (did_df['year_int'] >= 2024).astype(int)
    did_df['TreatPost'] = did_df['Treat'] * did_df['Post']
    did_df['Stkcd'] = did_df['Stkcd_str']
    did_df['year'] = did_df['year_str']

    did_sub = did_df[(did_df.year_int >= 2021) & (did_df.year_int <= 2024)]
    m = pf.feols(f"PriceDelay ~ TreatPost + {ctrl_str} | Stkcd + year",
                 data=did_sub, vcov={"CRV1": "IndYear"})
    c, t, p = m.coef()['TreatPost'], m.tstat()['TreatPost'], m.pvalue()['TreatPost']
    sig = '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.1 else ''
    print(f"  处理组={treat_name}: coef={c:.6f}, t={t:.3f}, p={p:.4f} {sig} (N={m._N})")
    all_results.append({
        'Design': f'BinaryDID_2024_w2124', 'Treatment': treat_name,
        'Var': 'TreatPost', 'Coef': c, 't': t, 'p': p, 'N': m._N
    })

# ================================================================
# 4. Design B: Binary DID, 2024事件, 更长窗口 2019-2024
# ================================================================
print("\n" + "=" * 60)
print("4. Design B: Binary DID (2024事件, 2019-2024窗口)")
print("=" * 60)

for treat_name, treat_set in [('2017-2021', treat_a), ('2011-2021', treat_b)]:
    did_df = reg_df.copy()
    did_df['Treat'] = did_df['Stkcd'].isin(treat_set).astype(int)
    did_df['Post'] = (did_df['year_int'] >= 2024).astype(int)
    did_df['TreatPost'] = did_df['Treat'] * did_df['Post']
    did_df['Stkcd'] = did_df['Stkcd_str']
    did_df['year'] = did_df['year_str']

    did_sub = did_df[(did_df.year_int >= 2019) & (did_df.year_int <= 2024)]
    m = pf.feols(f"PriceDelay ~ TreatPost + {ctrl_str} | Stkcd + year",
                 data=did_sub, vcov={"CRV1": "IndYear"})
    c, t, p = m.coef()['TreatPost'], m.tstat()['TreatPost'], m.pvalue()['TreatPost']
    sig = '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.1 else ''
    print(f"  处理组={treat_name}: coef={c:.6f}, t={t:.3f}, p={p:.4f} {sig} (N={m._N})")
    all_results.append({
        'Design': f'BinaryDID_2024_w1924', 'Treatment': treat_name,
        'Var': 'TreatPost', 'Coef': c, 't': t, 'p': p, 'N': m._N
    })

# ================================================================
# 5. Design C: Binary DID, 2023公告事件
# ================================================================
print("\n" + "=" * 60)
print("5. Design C: Binary DID (2023公告事件)")
print("=" * 60)
print("  入表新规2023年8月发布, 2024年1月生效")
print("  Post定义: year >= 2023 (公告年) 或 year >= 2024 (实施年)")

for post_year, post_label in [(2023, '2023公告'), (2024, '2024实施')]:
    did_df = reg_df.copy()
    did_df['Treat'] = did_df['Stkcd'].isin(treat_a).astype(int)
    did_df['Post'] = (did_df['year_int'] >= post_year).astype(int)
    did_df['TreatPost'] = did_df['Treat'] * did_df['Post']
    did_df['Stkcd'] = did_df['Stkcd_str']
    did_df['year'] = did_df['year_str']

    did_sub = did_df[(did_df.year_int >= 2019) & (did_df.year_int <= 2024)]
    m = pf.feols(f"PriceDelay ~ TreatPost + {ctrl_str} | Stkcd + year",
                 data=did_sub, vcov={"CRV1": "IndYear"})
    c, t, p = m.coef()['TreatPost'], m.tstat()['TreatPost'], m.pvalue()['TreatPost']
    sig = '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.1 else ''
    print(f"  Post={post_label}, 处理组=2017-2021: coef={c:.6f}, t={t:.3f}, p={p:.4f} {sig} (N={m._N})")
    all_results.append({
        'Design': f'BinaryDID_{post_label}_w1924', 'Treatment': '2017-2021',
        'Var': 'TreatPost', 'Coef': c, 't': t, 'p': p, 'N': m._N
    })

# ================================================================
# 6. Design D: 连续处理 DID
# ================================================================
print("\n" + "=" * 60)
print("6. Design D: 连续处理 DID (DU_kw_pre x Post)")
print("=" * 60)

did_df = reg_df.copy()
did_df['DU_kw_pre'] = did_df['Stkcd'].map(firm_kw_cont).fillna(0)
# 标准化
did_df['DU_kw_pre_std'] = (did_df['DU_kw_pre'] - did_df['DU_kw_pre'].mean()) / did_df['DU_kw_pre'].std()

for post_year, post_label in [(2024, '2024实施'), (2023, '2023公告')]:
    did_df['Post'] = (did_df['year_int'] >= post_year).astype(int)
    did_df['IntensityPost'] = did_df['DU_kw_pre_std'] * did_df['Post']

    did_df_fe = did_df.copy()
    did_df_fe['Stkcd'] = did_df_fe['Stkcd_str']
    did_df_fe['year'] = did_df_fe['year_str']

    for window_label, year_lo, year_hi in [('2019-2024', 2019, 2024), ('2021-2024', 2021, 2024)]:
        sub = did_df_fe[(did_df_fe.year_int >= year_lo) & (did_df_fe.year_int <= year_hi)]
        m = pf.feols(f"PriceDelay ~ IntensityPost + {ctrl_str} | Stkcd + year",
                     data=sub, vcov={"CRV1": "IndYear"})
        c, t, p = m.coef()['IntensityPost'], m.tstat()['IntensityPost'], m.pvalue()['IntensityPost']
        sig = '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.1 else ''
        print(f"  Post={post_label}, 窗口={window_label}: coef={c:.6f}, t={t:.3f}, p={p:.4f} {sig} (N={m._N})")
        all_results.append({
            'Design': f'ContDID_{post_label}_{window_label}', 'Treatment': 'DU_kw_pre_std',
            'Var': 'IntensityPost', 'Coef': c, 't': t, 'p': p, 'N': m._N
        })

# ================================================================
# 7. Design E: Event Study (完整动态效应)
# ================================================================
print("\n" + "=" * 60)
print("7. Design E: Event Study")
print("=" * 60)

event_results = []

for treat_name, treat_set, base_year in [
    ('Binary_2017-2021', treat_a, 2022),
    ('Continuous_2017-2021', None, 2022),
]:
    es_df = reg_df[(reg_df.year_int >= 2017) & (reg_df.year_int <= 2024)].copy()

    if treat_set is not None:
        # Binary event study
        es_df['Treat'] = es_df['Stkcd'].isin(treat_set).astype(float)
        event_years = [y for y in range(2017, 2025) if y != base_year]
        for y in event_years:
            es_df[f'Treat_Y{y}'] = es_df['Treat'] * (es_df['year_int'] == y).astype(float)
        trend_vars = ' + '.join([f'Treat_Y{y}' for y in event_years])
    else:
        # Continuous event study
        es_df['DU_kw_pre'] = es_df['Stkcd'].map(firm_kw_cont).fillna(0)
        es_df['DU_kw_pre_std'] = (es_df['DU_kw_pre'] - es_df['DU_kw_pre'].mean()) / es_df['DU_kw_pre'].std()
        event_years = [y for y in range(2017, 2025) if y != base_year]
        for y in event_years:
            es_df[f'Treat_Y{y}'] = es_df['DU_kw_pre_std'] * (es_df['year_int'] == y).astype(float)
        trend_vars = ' + '.join([f'Treat_Y{y}' for y in event_years])

    es_df['Stkcd'] = es_df['Stkcd_str']
    es_df['year'] = es_df['year_str']

    m = pf.feols(f"PriceDelay ~ {trend_vars} + {ctrl_str} | Stkcd + year",
                 data=es_df, vcov={"CRV1": "IndYear"})

    print(f"\n  --- Event Study: {treat_name} (基期={base_year}) ---")
    for y in event_years:
        k = f'Treat_Y{y}'
        c, t, p = m.coef()[k], m.tstat()[k], m.pvalue()[k]
        se = c / t if abs(t) > 0.001 else np.nan
        sig = '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.1 else ''
        marker = ' <-- POST' if y >= 2023 else ''
        print(f"    {y}: coef={c:.6f}, se={se:.6f}, t={t:.3f}, p={p:.4f} {sig}{marker}")
        event_results.append({
            'Design': treat_name, 'BaseYear': base_year,
            'Year': y, 'Coef': c, 'SE': se, 't': t, 'p': p, 'N': m._N
        })

# ================================================================
# 8. 安慰剂检验 (修正后处理组)
# ================================================================
print("\n" + "=" * 60)
print("8. 安慰剂检验 (修正后处理组: 2017-2021)")
print("=" * 60)

placebo_results = []

for fake_year in [2019, 2020, 2021, 2022]:
    did_df = reg_df.copy()
    did_df['Treat'] = did_df['Stkcd'].isin(treat_a).astype(int)
    did_df['Post'] = (did_df['year_int'] >= fake_year).astype(int)
    did_df['TreatPost'] = did_df['Treat'] * did_df['Post']
    did_df['Stkcd'] = did_df['Stkcd_str']
    did_df['year'] = did_df['year_str']

    # 窗口: fake_year前2年 到 fake_year后1年 (类似2021-2024对应2024)
    lo = fake_year - 3
    hi = fake_year + 1
    sub = did_df[(did_df.year_int >= lo) & (did_df.year_int <= hi)]

    if len(sub) < 100:
        print(f"  安慰剂{fake_year}: 样本太小, 跳过")
        continue

    m = pf.feols(f"PriceDelay ~ TreatPost + {ctrl_str} | Stkcd + year",
                 data=sub, vcov={"CRV1": "IndYear"})
    c, t, p = m.coef()['TreatPost'], m.tstat()['TreatPost'], m.pvalue()['TreatPost']
    sig = '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.1 else ''
    expected = 'n.s. ✓' if p >= 0.1 else f'显著 ⚠ {sig}'
    print(f"  安慰剂{fake_year} (窗口{lo}-{hi}): coef={c:.6f}, t={t:.3f}, p={p:.4f} {expected} (N={m._N})")
    placebo_results.append({
        'FakeYear': fake_year, 'Window': f'{lo}-{hi}',
        'Coef': c, 't': t, 'p': p, 'N': m._N
    })

# 真实政策年对照
did_df = reg_df.copy()
did_df['Treat'] = did_df['Stkcd'].isin(treat_a).astype(int)
did_df['Post'] = (did_df['year_int'] >= 2024).astype(int)
did_df['TreatPost'] = did_df['Treat'] * did_df['Post']
did_df['Stkcd'] = did_df['Stkcd_str']
did_df['year'] = did_df['year_str']
sub = did_df[(did_df.year_int >= 2021) & (did_df.year_int <= 2024)]
m = pf.feols(f"PriceDelay ~ TreatPost + {ctrl_str} | Stkcd + year",
             data=sub, vcov={"CRV1": "IndYear"})
c, t, p = m.coef()['TreatPost'], m.tstat()['TreatPost'], m.pvalue()['TreatPost']
sig = '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.1 else ''
print(f"  真实2024 (窗口2021-2024): coef={c:.6f}, t={t:.3f}, p={p:.4f} {sig} (N={m._N})")
placebo_results.append({
    'FakeYear': '2024_real', 'Window': '2021-2024',
    'Coef': c, 't': t, 'p': p, 'N': m._N
})

# ================================================================
# 9. 不同处理组阈值的敏感性
# ================================================================
print("\n" + "=" * 60)
print("9. 处理组阈值敏感性 (P33 / P50 / P67)")
print("=" * 60)

sensitivity_results = []

for pct, pct_label in [(0.33, 'P33'), (0.50, 'P50'), (0.67, 'P67')]:
    th = firm_kw_a.quantile(pct)
    treat_set = set(firm_kw_a[firm_kw_a >= th].index)

    did_df = reg_df.copy()
    did_df['Treat'] = did_df['Stkcd'].isin(treat_set).astype(int)
    did_df['Post'] = (did_df['year_int'] >= 2024).astype(int)
    did_df['TreatPost'] = did_df['Treat'] * did_df['Post']
    did_df['Stkcd'] = did_df['Stkcd_str']
    did_df['year'] = did_df['year_str']

    sub = did_df[(did_df.year_int >= 2021) & (did_df.year_int <= 2024)]
    m = pf.feols(f"PriceDelay ~ TreatPost + {ctrl_str} | Stkcd + year",
                 data=sub, vcov={"CRV1": "IndYear"})
    c, t, p = m.coef()['TreatPost'], m.tstat()['TreatPost'], m.pvalue()['TreatPost']
    sig = '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.1 else ''
    print(f"  阈值={pct_label} (th={th:.4f}, 处理组={len(treat_set)}): coef={c:.6f}, t={t:.3f}, p={p:.4f} {sig} (N={m._N})")
    sensitivity_results.append({
        'Threshold': pct_label, 'ThresholdValue': th,
        'TreatN': len(treat_set), 'Coef': c, 't': t, 'p': p, 'N': m._N
    })

# ================================================================
# 10. 保存所有结果
# ================================================================
print("\n" + "=" * 60)
print("10. 保存结果")
print("=" * 60)

# 主DID结果
df_main = pd.DataFrame(all_results)
df_main.to_csv(f"{RES_DIR}/did_v2_main.csv", index=False)
print(f"  DID主结果: {RES_DIR}/did_v2_main.csv ({len(df_main)} rows)")

# Event study
df_event = pd.DataFrame(event_results)
df_event.to_csv(f"{RES_DIR}/did_v2_event_study.csv", index=False)
print(f"  Event study: {RES_DIR}/did_v2_event_study.csv ({len(df_event)} rows)")

# 安慰剂
df_placebo = pd.DataFrame(placebo_results)
df_placebo.to_csv(f"{RES_DIR}/did_v2_placebo.csv", index=False)
print(f"  安慰剂: {RES_DIR}/did_v2_placebo.csv ({len(df_placebo)} rows)")

# 阈值敏感性
df_sens = pd.DataFrame(sensitivity_results)
df_sens.to_csv(f"{RES_DIR}/did_v2_sensitivity.csv", index=False)
print(f"  阈值敏感性: {RES_DIR}/did_v2_sensitivity.csv ({len(df_sens)} rows)")

# 汇总打印
print("\n" + "=" * 60)
print("汇总")
print("=" * 60)
print("\n--- 主DID ---")
print(df_main.to_string(index=False))
print("\n--- Event Study (Binary, base=2022) ---")
es_bin = df_event[df_event.Design == 'Binary_2017-2021']
print(es_bin[['Year', 'Coef', 't', 'p']].to_string(index=False))
print("\n--- 安慰剂 ---")
print(df_placebo.to_string(index=False))
print("\n--- 阈值敏感性 ---")
print(df_sens.to_string(index=False))

print("\n完成！")
