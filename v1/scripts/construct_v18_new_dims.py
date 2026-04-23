"""
Construct new heterogeneity dimensions for v18:
1. StrategicEmerging: 战略性新兴产业 dummy (narrower definition)
2. DigEconCore: 数字经济核心产业 dummy (NBS 2021, narrow 4-industry)
3. IndustryCluster: 产业集群 proxy (Location Quotient based)

Merge to reg_sample_v18.dta and re-export.
"""

import pandas as pd
import numpy as np
from pathlib import Path

STATA = Path('/Users/mac/computerscience/15会计研究/data_stata')

print("Loading reg_sample_v18.dta...")
df = pd.read_stata(STATA / 'reg_sample_v18.dta')
df['Stkcd'] = df['Stkcd'].astype(int)
df['year'] = df['year'].astype(int)
print(f"  {len(df)} obs, {df.columns.shape[0]} cols")

# ════════════════════════════════════════════
# 1. 战略性新兴产业 (Strategic Emerging Industries)
# ════════════════════════════════════════════
# Narrow definition: high-end manufacturing + IT + biomedical + new energy
# Excludes commodity industries (C25 petrochemical, C28 chemical fiber, C29 rubber,
# C31 ferrous metals, C32 non-ferrous metals, C41 other manufacturing)
strategic_codes = [
    'C27',  # 医药制造 (biomedical)
    'C34',  # 通用设备 (general equipment)
    'C35',  # 专用设备 (specialized equipment)
    'C36',  # 汽车 (auto, inc. new energy vehicles)
    'C37',  # 铁路船舶航空航天 (rail/ship/aerospace)
    'C38',  # 电气机械 (electrical machinery)
    'C39',  # 计算机通信电子 (ICT manufacturing)
    'C40',  # 仪器仪表 (instruments)
    'D44',  # 电力热力 (power, inc. new energy)
    'I63',  # 电信广播卫星 (telecom)
    'I64',  # 互联网 (internet)
    'I65',  # 软件信息技术 (software/IT)
    'M73',  # 研究和试验发展 (R&D)
    'M74',  # 专业技术服务 (technical services)
    'N77',  # 生态保护环境治理 (environmental)
]

df['StrategicEmerging'] = df['Ind2'].isin(strategic_codes).astype(int)
n_se = df['StrategicEmerging'].sum()
pct_se = n_se / len(df) * 100
print(f"\n[1] StrategicEmerging: {n_se} obs ({pct_se:.1f}%)")
print(f"    Unique firms: {df[df['StrategicEmerging']==1]['Stkcd'].nunique()}")

# ════════════════════════════════════════════
# 2. 数字经济核心产业 (Digital Economy Core Industries)
# ════════════════════════════════════════════
# NBS 2021 narrow definition (数字产业化, 4 industries)
digital_core_codes = [
    'C39',  # 数字产品制造 (digital product manufacturing)
    'I63',  # 数字技术应用 (digital tech application, telecom)
    'I64',  # 数字产品服务 (digital product services, internet)
    'I65',  # 数字产品服务 (software/IT services)
]

df['DigEconCore'] = df['Ind2'].isin(digital_core_codes).astype(int)
n_dc = df['DigEconCore'].sum()
pct_dc = n_dc / len(df) * 100
print(f"\n[2] DigEconCore: {n_dc} obs ({pct_dc:.1f}%)")
print(f"    Unique firms: {df[df['DigEconCore']==1]['Stkcd'].nunique()}")

# ════════════════════════════════════════════
# 3. 产业集群 (Industrial Cluster proxy via Location Quotient)
# ════════════════════════════════════════════
# LQ = (n_ij / n_j) / (n_i / n)
# n_ij: firms in industry i, province j
# n_j: total firms in province j
# n_i: total firms in industry i
# n: total firms
# High LQ (>= median) indicates clustered

print("\n[3] IndustryCluster (Location Quotient)...")

# Compute LQ per province-industry-year
def compute_lq(group):
    """Compute location quotient for each province-industry pair within a year."""
    n_total = len(group)
    # Firms per province
    n_prov = group.groupby('Prov_short').size()
    # Firms per industry
    n_ind = group.groupby('Ind2').size()
    # Firms per province-industry
    n_pi = group.groupby(['Prov_short', 'Ind2']).size()

    lq_dict = {}
    for (prov, ind), n_ij in n_pi.items():
        n_j = n_prov[prov]
        n_i = n_ind[ind]
        lq = (n_ij / n_j) / (n_i / n_total)
        lq_dict[(prov, ind)] = lq

    return lq_dict

lq_all = {}
for yr, grp in df.groupby('year'):
    lq_yr = compute_lq(grp)
    for (prov, ind), lq in lq_yr.items():
        lq_all[(prov, ind, yr)] = lq

# Map LQ back to each observation
df['LQ'] = df.apply(lambda r: lq_all.get((r['Prov_short'], r['Ind2'], r['year']), np.nan), axis=1)

# Cluster dummy: LQ >= year-specific median
lq_med = df.groupby('year')['LQ'].transform('median')
df['IndustryCluster'] = (df['LQ'] >= lq_med).astype(int)

# Also keep continuous LQ for potential use
n_cl = df['IndustryCluster'].sum()
pct_cl = n_cl / len(df) * 100
print(f"  LQ range: {df['LQ'].min():.2f} - {df['LQ'].max():.2f}, median: {df['LQ'].median():.2f}")
print(f"  IndustryCluster=1: {n_cl} obs ({pct_cl:.1f}%)")

# ════════════════════════════════════════════
# Cross-tabulation
# ════════════════════════════════════════════
print("\n── Cross-tabulation ──")
print(f"StrategicEmerging & DigEconCore overlap: {((df['StrategicEmerging']==1) & (df['DigEconCore']==1)).sum()}")
print(f"StrategicEmerging & IndustryCluster overlap: {((df['StrategicEmerging']==1) & (df['IndustryCluster']==1)).sum()}")
print(f"DigEconCore & IndustryCluster overlap: {((df['DigEconCore']==1) & (df['IndustryCluster']==1)).sum()}")

# ════════════════════════════════════════════
# Save
# ════════════════════════════════════════════
out = STATA / 'reg_sample_v18.dta'
print(f"\nSaving to {out}...")
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].astype(str)
df.to_stata(out, write_index=False, version=118)
print(f"  Saved: {len(df)} obs, {df.columns.shape[0]} cols")
print(f"  New cols: StrategicEmerging, DigEconCore, LQ, IndustryCluster")
