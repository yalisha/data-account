"""
v3: 同期回归（参照朱康2025范文模型设定）
DU_{i,t} 与 PriceDelay_{i,t} 同期，不做滞后处理
结果保存到 results/v3_concurrent/ 目录，不覆盖v2结果
"""

import pandas as pd
import numpy as np
import pyfixest as pf
import warnings, os
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE_DIR, "data_parquet")
RES_DIR = os.path.join(BASE_DIR, "results/v3_concurrent")
os.makedirs(RES_DIR, exist_ok=True)

# ================================================================
# 1. 加载 & 合并数据（同v2）
# ================================================================
print("=" * 60)
print("v3 同期回归: 加载数据")
print("=" * 60)

panel = pd.read_parquet(f"{OUT_DIR}/panel.parquet")
ar_feat = pd.read_parquet(f"{OUT_DIR}/annual_report_features.parquet")

panel = panel.merge(
    ar_feat[['Stkcd', 'year', 'kw_total', 'kw_per10k', 'mda_kw_per10k',
             'substantive_ratio', 'substantive_count',
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

# ================================================================
# 2. 样本筛选
# ================================================================
mask_fin = panel['IndustryCodeC'].str.startswith('J', na=False)
mask_st = panel['LISTINGSTATE'].isin(['ST', '*ST'])
mask_new = panel['Age'] <= 0
panel = panel[~mask_fin & ~mask_st & ~mask_new]

# ================================================================
# 3. 构造回归变量 —— 关键区别：不做shift
# ================================================================
print("构造变量（同期，无滞后）")

panel['Ind2'] = panel['IndustryCodeC'].str[:3]
panel['IndYear'] = panel['Ind2'].astype(str) + '_' + panel['year'].astype(str)

panel['DU_kw'] = panel['kw_per10k']
panel['DU_kw_ln'] = np.log1p(panel['kw_total'])
panel['DU_sub_ln'] = np.log1p(panel['substantive_count'])

# ★ 不做 shift —— 同期变量 ★

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

reg_df = panel.dropna(subset=['PriceDelay', 'DU_kw'] + controls).copy()
reg_df['Stkcd_str'] = reg_df['Stkcd'].astype(str)
reg_df['year_int'] = reg_df['year']
reg_df['year_str'] = reg_df['year'].astype(str)

print(f"  回归样本: {len(reg_df):,} obs, {reg_df.Stkcd.nunique():,} firms")
print(f"  年份: {reg_df.year.min()}-{reg_df.year.max()}")

# ================================================================
# 4. 基准回归
# ================================================================
print("\n" + "=" * 60)
print("基准回归（同期DU）")
print("=" * 60)

ctrl_str = ' + '.join(controls)
reg_fe = reg_df.copy()
reg_fe['Stkcd'] = reg_fe['Stkcd_str']
reg_fe['year'] = reg_fe['year_str']

models = {}
models['(1)'] = pf.feols(f"PriceDelay ~ DU_kw | Stkcd + year", data=reg_fe, vcov={"CRV1": "IndYear"})
models['(2)'] = pf.feols(f"PriceDelay ~ DU_kw + {ctrl_str} | Stkcd + year", data=reg_fe, vcov={"CRV1": "IndYear"})
models['(3)'] = pf.feols(f"PriceDelay ~ DU_kw_ln + {ctrl_str} | Stkcd + year", data=reg_fe, vcov={"CRV1": "IndYear"})
models['(4)'] = pf.feols(f"PriceDelay ~ DU_sub_ln + {ctrl_str} | Stkcd + year", data=reg_fe, vcov={"CRV1": "IndYear"})
models['(5)'] = pf.feols(f"PriceDelay ~ DU_kw + {ctrl_str} | Ind2 + year", data=reg_fe, vcov={"CRV1": "IndYear"})
models['(6)'] = pf.feols(f"PriceDelay ~ DU_kw_ln + {ctrl_str} | Ind2 + year", data=reg_fe, vcov={"CRV1": "IndYear"})

print(f"\n{'Model':<8} {'FE':<12} {'Var':<12} {'Coef':>10} {'SE':>10} {'t':>8} {'p':>8} {'Sig':<4} {'N':>8} {'R2':>8}")
print("-" * 90)

rows = []
for name, m in models.items():
    kv = [v for v in m.coef().keys() if v.startswith('DU_')][0]
    c = m.coef()[kv]
    se = m.se()[kv]
    t = m.tstat()[kv]
    p = m.pvalue()[kv]
    r2 = m._r2
    sig = '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.1 else ''
    fe = 'Firm+Year' if 'Stkcd' in str(m._fml) else 'Ind+Year'
    print(f"  {name:<6} {fe:<12} {kv:<12} {c:>10.6f} {se:>10.6f} {t:>8.3f} {p:>8.4f} {sig:<4} {m._N:>8} {r2:>8.4f}")
    rows.append({'Model': name, 'FE': fe, 'Var': kv, 'Coef': c, 'SE': se,
                 't': t, 'p': p, 'Sig': sig, 'N': m._N, 'R2': r2})

# 打印完整模型(2)
print("\n--- Model (2) 完整结果 ---")
print(models['(2)'].summary())

pd.DataFrame(rows).to_csv(f"{RES_DIR}/baseline_v3.csv", index=False)

# ================================================================
# 5. 机制检验（同期）
# ================================================================
print("\n" + "=" * 60)
print("机制检验（同期DU）")
print("=" * 60)

mech_rows = []
for dep, fml in [
    ('Analyst', f"Analyst ~ DU_kw + {ctrl_str.replace(' + Analyst', '')} | Stkcd + year"),
    ('FinAsset', f"FinAsset ~ DU_kw + {ctrl_str} | Stkcd + year"),
    ('InstHold', f"InstHold ~ DU_kw + {ctrl_str.replace(' + InstHold', '')} | Stkcd + year"),
]:
    sub = reg_fe.dropna(subset=[dep])
    m = pf.feols(fml, data=sub, vcov={"CRV1": "IndYear"})
    c = m.coef().get('DU_kw', np.nan)
    se = m.se().get('DU_kw', np.nan)
    t = m.tstat().get('DU_kw', np.nan)
    p = m.pvalue().get('DU_kw', np.nan)
    sig = '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.1 else ''
    print(f"  DU -> {dep}: coef={c:.6f}, se=({se:.6f}), t={t:.3f}, p={p:.4f} {sig} (N={m._N})")
    mech_rows.append({'DV': dep, 'Coef': c, 'SE': se, 't': t, 'p': p, 'Sig': sig, 'N': m._N, 'R2': m._r2})

pd.DataFrame(mech_rows).to_csv(f"{RES_DIR}/mechanism_v3.csv", index=False)

# ================================================================
# 6. 异质性（同期）
# ================================================================
print("\n" + "=" * 60)
print("异质性分析（同期DU）")
print("=" * 60)

base = f"PriceDelay ~ DU_kw + {ctrl_str} | Stkcd + year"
hightech = {'C39', 'I65', 'C35', 'C27', 'C40'}

het_rows = []
for label, sub in [
    ('国企', reg_fe[reg_fe.SOE == 1]),
    ('民企', reg_fe[reg_fe.SOE == 0]),
    ('高分析师', reg_fe[reg_fe.Analyst >= reg_fe.Analyst.median()]),
    ('低分析师', reg_fe[reg_fe.Analyst < reg_fe.Analyst.median()]),
    ('高科技', reg_fe[reg_fe.Ind2.isin(hightech)]),
    ('传统', reg_fe[~reg_fe.Ind2.isin(hightech)]),
]:
    try:
        m = pf.feols(base, data=sub, vcov={"CRV1": "IndYear"})
        c = m.coef().get('DU_kw', np.nan)
        se = m.se().get('DU_kw', np.nan)
        t = m.tstat().get('DU_kw', np.nan)
        p = m.pvalue().get('DU_kw', np.nan)
        sig = '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.1 else ''
        print(f"  {label} (N={m._N:,}): coef={c:.6f}, se=({se:.6f}), t={t:.3f}, p={p:.4f} {sig}")
        het_rows.append({'Group': label, 'Coef': c, 'SE': se, 't': t, 'p': p, 'Sig': sig, 'N': m._N})
    except Exception as e:
        print(f"  {label}: ERROR {e}")

pd.DataFrame(het_rows).to_csv(f"{RES_DIR}/heterogeneity_v3.csv", index=False)

print("\n" + "=" * 60)
print("全部完成，结果保存在 results/v3_concurrent/")
print("=" * 60)
