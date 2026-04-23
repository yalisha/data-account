"""
向 reg_sample_iv_v16.dta 追加异质性分析变量:
  - Big4: 四大审计二值变量
  - Intangible: 无形资产/总资产
  - Market: 樊纲市场化指数
输出: data_stata/reg_sample_het.dta
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

BASE = "/Users/mac/computerscience/15会计研究"

# ============================================================
# 1. Load base Stata dataset
# ============================================================
print("Loading reg_sample_iv_v16.dta...")
reg = pd.read_stata(f"{BASE}/data_stata/reg_sample_iv_v16.dta")
print(f"  Base: N={len(reg):,}, cols={reg.shape[1]}")
print(f"  Stkcd dtype={reg['Stkcd'].dtype}, year dtype={reg['year'].dtype}")

# Ensure merge keys are compatible
reg['Stkcd_int'] = reg['Stkcd'].astype(int)
reg['year_int'] = reg['year'].astype(int)

# ============================================================
# 2. Merge Intangible and Market from panel_dml
# ============================================================
print("\nMerging Intangible & Market from panel_dml.parquet...")
dml = pd.read_parquet(f"{BASE}/data_parquet/panel_dml.parquet",
                      columns=['Stkcd', 'year', 'Intangible', 'Market'])
dml['Stkcd'] = dml['Stkcd'].astype(int)
dml['year'] = dml['year'].astype(int)
dml = dml.drop_duplicates(subset=['Stkcd', 'year'], keep='last')

n_before = len(reg)
reg = reg.merge(dml, left_on=['Stkcd_int', 'year_int'],
                right_on=['Stkcd', 'year'], how='left',
                suffixes=('', '_dml'))
# Drop duplicate merge keys
for c in ['Stkcd_dml', 'year_dml']:
    if c in reg.columns:
        reg.drop(columns=[c], inplace=True)

intang_rate = reg['Intangible'].notna().mean()
market_rate = reg['Market'].notna().mean()
print(f"  Intangible merge rate: {intang_rate:.1%} ({reg['Intangible'].notna().sum():,}/{len(reg):,})")
print(f"  Market merge rate: {market_rate:.1%} ({reg['Market'].notna().sum():,}/{len(reg):,})")

# ============================================================
# 3. Construct Big4 from audit.parquet
# ============================================================
print("\nConstructing Big4 from audit.parquet...")
audit = pd.read_parquet(f"{BASE}/data_parquet/audit.parquet")
print(f"  Raw audit: {audit.shape}")

# Parse year from Accper
audit['Accper_dt'] = pd.to_datetime(audit['Accper'], errors='coerce')
audit['audit_year'] = audit['Accper_dt'].dt.year
audit['audit_month'] = audit['Accper_dt'].dt.month

# Keep annual reports (month == 12)
audit = audit[audit['audit_month'] == 12].copy()
audit['Stkcd'] = audit['Stkcd'].astype(int)

# Big4 identification via domestic auditor firm name
big4_keywords = ['普华永道中天', '安永华明', '德勤华永', '毕马威华振']
audit['Big4'] = audit['Dadtunit'].apply(
    lambda x: int(any(kw in str(x) for kw in big4_keywords)) if pd.notna(x) else 0
)

# Also check international auditor field as backup
int_big4 = ['PricewaterhouseCoopers', 'PwC', 'Ernst & Young', 'EY',
            'Deloitte', 'KPMG']
audit['Big4_int'] = audit['Iadtunit'].apply(
    lambda x: int(any(kw.lower() in str(x).lower() for kw in int_big4)) if pd.notna(x) else 0
)
audit['Big4'] = ((audit['Big4'] == 1) | (audit['Big4_int'] == 1)).astype(int)

# Deduplicate: keep last per Stkcd+year
audit = audit.sort_values(['Stkcd', 'audit_year', 'Accper_dt'])
audit = audit.drop_duplicates(subset=['Stkcd', 'audit_year'], keep='last')

big4_rate = audit['Big4'].mean()
print(f"  Big4 rate in audit data: {big4_rate:.1%} ({audit['Big4'].sum():,}/{len(audit):,})")

# Merge
reg = reg.merge(audit[['Stkcd', 'audit_year', 'Big4']],
                left_on=['Stkcd_int', 'year_int'],
                right_on=['Stkcd', 'audit_year'], how='left',
                suffixes=('', '_audit'))
for c in ['Stkcd_audit', 'audit_year']:
    if c in reg.columns:
        reg.drop(columns=[c], inplace=True)

reg['Big4'] = reg['Big4'].fillna(0).astype(int)
big4_in_sample = reg['Big4'].mean()
print(f"  Big4 in regression sample: {big4_in_sample:.1%} ({reg['Big4'].sum():,}/{len(reg):,})")

# ============================================================
# 4. Winsorize Intangible and Market at 1%/99%
# ============================================================
print("\nWinsorizing...")
def winsorize(s):
    lo, hi = s.quantile([0.01, 0.99])
    return s.clip(lo, hi)

for v in ['Intangible', 'Market']:
    if v in reg.columns and reg[v].notna().any():
        before_std = reg[v].std()
        reg[v] = winsorize(reg[v])
        after_std = reg[v].std()
        print(f"  {v}: mean={reg[v].mean():.4f}, sd={after_std:.4f}, "
              f"min={reg[v].min():.4f}, max={reg[v].max():.4f}")

# ============================================================
# 5. Clean up and export
# ============================================================
# Drop temp columns
reg.drop(columns=['Stkcd_int', 'year_int'], inplace=True, errors='ignore')

# Summary
print(f"\n{'='*60}")
print(f"Final dataset: N={len(reg):,}, cols={reg.shape[1]}")
print(f"  New variables: Intangible, Market, Big4")
print(f"  Intangible non-null: {reg['Intangible'].notna().sum():,}")
print(f"  Market non-null: {reg['Market'].notna().sum():,}")
print(f"  Big4 == 1: {reg['Big4'].sum():,} ({reg['Big4'].mean():.1%})")

# Save
outpath = f"{BASE}/data_stata/reg_sample_het.dta"
reg.to_stata(outpath, write_index=False, version=118)
print(f"\nSaved to {outpath}")
