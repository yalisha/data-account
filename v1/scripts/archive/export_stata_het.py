"""
Export regression sample to Stata format with heterogeneity grouping variables.
Prepares data for Fisher permutation test in Stata.
"""
import pandas as pd
import numpy as np

BASE = "/Users/mac/computerscience/15会计研究"

# Load existing v3 Stata sample as base
reg = pd.read_stata(f"{BASE}/data_stata/reg_sample_v3.dta")
print(f"Base sample: {len(reg):,} obs")

# Load additional data
panel = pd.read_parquet(f"{BASE}/data_parquet/panel.parquet")
fi = pd.read_parquet(f"{BASE}/data_parquet/firm_info.parquet",
                     columns=['Symbol','EndDate','IndustryCodeC'])
fi = fi.rename(columns={'Symbol':'Stkcd'})
fi['EndDate'] = pd.to_datetime(fi['EndDate'])
fi['year'] = fi['EndDate'].dt.year
fi = fi.sort_values(['Stkcd','year','EndDate']).drop_duplicates(
    subset=['Stkcd','year'], keep='last')

# Merge IndustryCodeC for industry-level groupings
reg = reg.merge(fi[['Stkcd','year','IndustryCodeC']], on=['Stkcd','year'], how='left')
# Fill missing with mode
ind_mode = fi.groupby('Stkcd')['IndustryCodeC'].agg(
    lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else np.nan)
reg = reg.merge(ind_mode.rename('IndCode_mode').reset_index(), on='Stkcd', how='left')
reg['IndustryCodeC'] = reg['IndustryCodeC'].fillna(reg['IndCode_mode'])
reg.drop(columns=['IndCode_mode'], inplace=True)

# ============================================================
# Construct grouping variables
# ============================================================

# 1. SOE (already exists)
print(f"SOE: {reg['SOE'].value_counts().to_dict()}")

# 2. 信息环境: 分析师覆盖按行业年度中位数分组 (like 朱康)
reg['Ind1'] = reg['IndustryCodeC'].str[0]  # 1-digit industry
reg['analyst_indyr_med'] = reg.groupby(['Ind1','year'])['Analyst'].transform('median')
reg['HighInfoEnv'] = (reg['Analyst'] >= reg['analyst_indyr_med']).astype(int)
print(f"HighInfoEnv (industry-year median): {reg['HighInfoEnv'].value_counts().to_dict()}")

# 3. 融资约束 SA index (Hadlock & Pierce, 2010)
# SA = -0.737*Size + 0.043*Size^2 - 0.040*Age
# Higher SA (less negative) = more constrained
reg['SA'] = -0.737 * reg['Size'] + 0.043 * reg['Size']**2 - 0.040 * reg['Age']
sa_med = reg['SA'].median()
reg['HighSA'] = (reg['SA'] >= sa_med).astype(int)  # 1 = more constrained
print(f"HighSA (financing constraints): {reg['HighSA'].value_counts().to_dict()}")

# 4. 机构持股 (institutional ownership) median split
insthold_med = reg['InstHold'].median()
reg['HighInstHold'] = (reg['InstHold'] >= insthold_med).astype(int)
print(f"HighInstHold: {reg['HighInstHold'].value_counts().to_dict()}")

# 5. 审计质量 (Big4 audit)
reg['Big4'] = reg['AuditType'].astype(int)
print(f"Big4: {reg['Big4'].value_counts().to_dict()}")

# 6. 高科技行业 (keep for reference)
hitech_codes = ['I', 'M']
reg['HighTech'] = reg['IndustryCodeC'].str[0].isin(hitech_codes).astype(int)
print(f"HighTech: {reg['HighTech'].value_counts().to_dict()}")

# 7. Size median (keep for reference)
size_med = reg['Size'].median()
reg['HighSize'] = (reg['Size'] >= size_med).astype(int)
print(f"HighSize: {reg['HighSize'].value_counts().to_dict()}")

# 8. Analyst overall median (keep for reference)
analyst_med = reg['Analyst'].median()
reg['HighAnalyst'] = (reg['Analyst'] >= analyst_med).astype(int)
print(f"HighAnalyst (overall median): {reg['HighAnalyst'].value_counts().to_dict()}")

# Clean up string columns for Stata
reg['IndustryCodeC'] = reg['IndustryCodeC'].fillna('')
reg['Ind1'] = reg['Ind1'].fillna('')

# Drop temporary columns
reg.drop(columns=['analyst_indyr_med'], inplace=True)

# Save
out_path = f"{BASE}/data_stata/reg_sample_het.dta"
reg.to_stata(out_path, write_index=False, version=118)
print(f"\nSaved: {out_path}")
print(f"Final shape: {reg.shape}")
print(f"Columns: {list(reg.columns)}")
