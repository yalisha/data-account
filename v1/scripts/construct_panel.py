"""
阶段三：构造全部变量并合并为企业-年度面板
输出: data_parquet/panel.parquet
"""
import pandas as pd
import numpy as np
import os, warnings
warnings.filterwarnings('ignore')

OUT = "/Users/mac/computerscience/15会计研究/data_parquet"


def load(name, cols=None):
    path = f"{OUT}/{name}.parquet"
    return pd.read_parquet(path, columns=cols)


# ============================================================
# 0. 加载因变量 (Price Delay)
# ============================================================
print("0. 加载因变量 PriceDelay...")
panel = load('price_delay')  # Stkcd, year, PriceDelay, n_obs
panel['Stkcd'] = panel['Stkcd'].astype(int)
print(f"   {len(panel):,} obs, {panel.year.nunique()} years")

# ============================================================
# 1. 自变量: CNRDS 数据要素开发利用指数
# ============================================================
print("\n1. 自变量: CNRDS 数据要素指数...")
cnrds = load('cnrds_data_element_index')
# 跳过表头行 (Scode='股票代码')
cnrds = cnrds[cnrds['Scode'] != '股票代码'].copy()
cnrds = cnrds[cnrds['Texttype'] == '上市公司年报'].copy()
cnrds['Stkcd'] = cnrds['Scode'].astype(int)
cnrds['year'] = cnrds['Year'].astype(int)
cnrds['DataUsage'] = pd.to_numeric(cnrds['TWFre_count'], errors='coerce')
cnrds['DataUsage_termonly'] = pd.to_numeric(cnrds['Term_Only'], errors='coerce')
# 标准化: 词频/年报总词数 * 10000
cnrds['DataUsage_norm'] = cnrds['DataUsage'] / cnrds['DataUsage_termonly'] * 10000
cnrds = cnrds[['Stkcd', 'year', 'DataUsage', 'DataUsage_norm']].drop_duplicates(['Stkcd', 'year'])
panel = panel.merge(cnrds, on=['Stkcd', 'year'], how='left')
print(f"   匹配率: {panel.DataUsage.notna().mean():.1%}")

# ============================================================
# 2. 控制变量: 从各parquet表构造
# ============================================================

# --- 辅助: 筛选年报合并报表 ---
def annual_report(df, date_col='Accper', type_col='Typrep'):
    """筛选年报(12-31)合并报表(A)"""
    df = df.copy()
    df[date_col] = df[date_col].astype(str)
    mask = df[date_col].str.endswith('12-31')
    if type_col and type_col in df.columns:
        mask = mask & (df[type_col] == 'A')
    df = df[mask].copy()
    df['year'] = pd.to_datetime(df[date_col]).dt.year
    return df


# 2a. Size, Lev, ROA, FinAsset, DataAsset from balance sheet + income stmt
print("\n2a. 资产负债表 + 利润表...")
bs = annual_report(load('balance_sheet'))
is_ = annual_report(load('income_stmt'))

# Size = ln(总市值) → 从月度回报率取年末市值
# 先从资产负债表取 Lev, FinAsset, DataAsset
bs['Stkcd'] = bs['Stkcd'].astype(int)
bs['Lev'] = bs['A002000000'] / bs['A001000000']  # 负债/资产
bs['FinAsset'] = (bs['A001107000'].fillna(0) + bs['A001202000'].fillna(0) +
                  bs['A001211000'].fillna(0) + bs['A001229000'].fillna(0)) / bs['A001000000']
bs['DataAsset'] = (bs['A001123101'].fillna(0) + bs['A001218201'].fillna(0) +
                   bs['A001219101'].fillna(0)) / bs['A001000000']
bs_vars = bs[['Stkcd', 'year', 'Lev', 'FinAsset', 'DataAsset', 'A001000000']].copy()
bs_vars = bs_vars.rename(columns={'A001000000': 'TotalAsset'})
bs_vars = bs_vars.drop_duplicates(['Stkcd', 'year'], keep='last')

# ROA, Growth from income stmt
is_['Stkcd'] = is_['Stkcd'].astype(int)
is_ = is_.drop_duplicates(['Stkcd', 'year'], keep='last')
is_ = is_.merge(bs_vars[['Stkcd', 'year', 'TotalAsset']], on=['Stkcd', 'year'], how='left')
is_['ROA'] = is_['B002000000'] / is_['TotalAsset']  # 净利润/总资产

# Growth = 营业收入增长率 (需要lag)
is_ = is_.sort_values(['Stkcd', 'year'])
is_['Revenue_lag'] = is_.groupby('Stkcd')['B001101000'].shift(1)
is_['Growth'] = (is_['B001101000'] - is_['Revenue_lag']) / is_['Revenue_lag'].abs()
is_['Growth'] = is_['Growth'].replace([np.inf, -np.inf], np.nan)
is_vars = is_[['Stkcd', 'year', 'ROA', 'Growth']].copy()

panel = panel.merge(bs_vars[['Stkcd', 'year', 'Lev', 'FinAsset', 'DataAsset']], on=['Stkcd', 'year'], how='left')
panel = panel.merge(is_vars, on=['Stkcd', 'year'], how='left')
print(f"   Lev匹配: {panel.Lev.notna().mean():.1%}, ROA: {panel.ROA.notna().mean():.1%}")

# 2b. Size = ln(总市值) from monthly return (年末12月)
print("2b. Size (总市值)...")
mr = load('monthly_return', cols=['Stkcd', 'Trdmnt', 'Msmvttl'])
mr['Stkcd'] = mr['Stkcd'].astype(int)
mr['Trdmnt'] = mr['Trdmnt'].astype(str)
mr_dec = mr[mr['Trdmnt'].str.endswith('-12')].copy()
mr_dec['year'] = pd.to_datetime(mr_dec['Trdmnt']).dt.year
mr_dec['Size'] = np.log(pd.to_numeric(mr_dec['Msmvttl'], errors='coerce') * 1000)  # CSMAR市值单位千元
mr_dec = mr_dec[['Stkcd', 'year', 'Size']].drop_duplicates(['Stkcd', 'year'], keep='last')
panel = panel.merge(mr_dec, on=['Stkcd', 'year'], how='left')
print(f"   Size匹配: {panel.Size.notna().mean():.1%}")

# 2c. TobinQ from relative value
print("2c. TobinQ...")
rv = load('relative_value', cols=['Stkcd', 'Accper', 'Source', 'F100901A'])
rv = rv.copy()  # Source=0 in this dataset, no filter needed
rv['Accper'] = rv['Accper'].astype(str)
rv = rv[rv['Accper'].str.endswith('12-31')].copy()
rv['Stkcd'] = rv['Stkcd'].astype(int)
rv['year'] = pd.to_datetime(rv['Accper']).dt.year
rv['TobinQ'] = pd.to_numeric(rv['F100901A'], errors='coerce')
rv = rv[['Stkcd', 'year', 'TobinQ']].drop_duplicates(['Stkcd', 'year'], keep='last')
panel = panel.merge(rv, on=['Stkcd', 'year'], how='left')
print(f"   TobinQ匹配: {panel.TobinQ.notna().mean():.1%}")

# 2d. Age = ln(上市年限) from firm_info
print("2d. Age...")
fi = load('firm_info', cols=['Symbol', 'EndDate'])
fi = fi.rename(columns={'Symbol': 'Stkcd'})
fi['Stkcd'] = fi['Stkcd'].astype(int)
fi['EndDate'] = fi['EndDate'].astype(str)
fi = fi[fi['EndDate'].str.endswith('12-31')].copy()
fi['year'] = pd.to_datetime(fi['EndDate']).dt.year
# 取每个公司最早的年份作为上市年
listing_year = fi.groupby('Stkcd')['year'].min().reset_index()
listing_year.columns = ['Stkcd', 'ListYear']
panel = panel.merge(listing_year, on='Stkcd', how='left')
panel['Age'] = np.log(panel['year'] - panel['ListYear'] + 1)
panel.loc[panel['Age'] < 0, 'Age'] = np.nan
panel = panel.drop(columns=['ListYear'])
print(f"   Age匹配: {panel.Age.notna().mean():.1%}")

# 2e. BoardSize, IndepRatio from manager_salary
print("2e. BoardSize, IndepRatio...")
ms = load('manager_salary')
ms = ms.rename(columns={'Symbol': 'Stkcd', 'Enddate': 'EndDate'})
ms['Stkcd'] = ms['Stkcd'].astype(int)
ms['EndDate'] = ms['EndDate'].astype(str)
ms = ms[ms['EndDate'].str.endswith('12-31')].copy()
ms['year'] = pd.to_datetime(ms['EndDate']).dt.year
# StatisticalCaliber = 1 现任
ms_now = ms[ms['StatisticalCaliber'] == 1].copy() if 1 in ms['StatisticalCaliber'].values else ms.copy()
ms_now['DirectorNumber'] = pd.to_numeric(ms_now['DirectorNumber'], errors='coerce')
ms_now['IndependentDirectorNumber'] = pd.to_numeric(ms_now['IndependentDirectorNumber'], errors='coerce')
ms_now['BoardSize'] = np.log(ms_now['DirectorNumber'].clip(lower=1))
ms_now['IndepRatio'] = ms_now['IndependentDirectorNumber'] / ms_now['DirectorNumber']
ms_now = ms_now[['Stkcd', 'year', 'BoardSize', 'IndepRatio']].drop_duplicates(['Stkcd', 'year'], keep='last')
panel = panel.merge(ms_now, on=['Stkcd', 'year'], how='left')
print(f"   BoardSize匹配: {panel.BoardSize.notna().mean():.1%}")

# 2f. Dual (两职合一) from governance
print("2f. Dual (两职合一)...")
gov = load('governance', cols=['Stkcd', 'Reptdt', 'Y1001b'])
gov['Stkcd'] = gov['Stkcd'].astype(int)
gov['Reptdt'] = gov['Reptdt'].astype(str)
gov = gov[gov['Reptdt'].str.endswith('12-31')].copy()
gov['year'] = pd.to_datetime(gov['Reptdt']).dt.year
gov['Dual'] = (gov['Y1001b'] == 1).astype(float)  # 1=两职合一, 2=不合一
gov = gov[['Stkcd', 'year', 'Dual']].drop_duplicates(['Stkcd', 'year'], keep='last')
panel = panel.merge(gov, on=['Stkcd', 'year'], how='left')
print(f"   Dual匹配: {panel.Dual.notna().mean():.1%}")

# 2g. Top1Share, SOE from equity_nature
print("2g. Top1Share, SOE...")
en = load('equity_nature', cols=['Symbol', 'EndDate', 'LargestHolderRate', 'EquityNatureID'])
en = en.rename(columns={'Symbol': 'Stkcd'})
en['Stkcd'] = en['Stkcd'].astype(int)
en['EndDate'] = en['EndDate'].astype(str)
en = en[en['EndDate'].str.endswith('12-31')].copy()
en['year'] = pd.to_datetime(en['EndDate']).dt.year
en['Top1Share'] = pd.to_numeric(en['LargestHolderRate'], errors='coerce')
en['SOE'] = en['EquityNatureID'].astype(str).str.contains('1').astype(float)  # '1'=国企
en = en[['Stkcd', 'year', 'Top1Share', 'SOE']].drop_duplicates(['Stkcd', 'year'], keep='last')
panel = panel.merge(en, on=['Stkcd', 'year'], how='left')
print(f"   Top1Share匹配: {panel.Top1Share.notna().mean():.1%}")

# 2h. InstHold from inst_holding
print("2h. InstHold...")
ih = load('inst_holding', cols=['Symbol', 'EndDate', 'InsInvestorProp'])
ih = ih.rename(columns={'Symbol': 'Stkcd'})
ih['Stkcd'] = ih['Stkcd'].astype(int)
ih['EndDate'] = ih['EndDate'].astype(str)
ih = ih[ih['EndDate'].str.endswith('12-31')].copy()
ih['year'] = pd.to_datetime(ih['EndDate']).dt.year
ih['InstHold'] = pd.to_numeric(ih['InsInvestorProp'], errors='coerce')
ih = ih[['Stkcd', 'year', 'InstHold']].drop_duplicates(['Stkcd', 'year'], keep='last')
panel = panel.merge(ih, on=['Stkcd', 'year'], how='left')
print(f"   InstHold匹配: {panel.InstHold.notna().mean():.1%}")

# 2i. Amihud = 年度均值
print("2i. Amihud...")
ami = load('amihud_daily', cols=['Stkcd', 'Trddt', 'ILLIQ'])
ami['Stkcd'] = ami['Stkcd'].astype(int)
ami['year'] = pd.to_datetime(ami['Trddt'], errors='coerce').dt.year
ami_yr = ami.groupby(['Stkcd', 'year'])['ILLIQ'].mean().reset_index()
ami_yr.columns = ['Stkcd', 'year', 'Amihud']
panel = panel.merge(ami_yr, on=['Stkcd', 'year'], how='left')
print(f"   Amihud匹配: {panel.Amihud.notna().mean():.1%}")

# 2j. Analyst = ln(1 + 分析师覆盖人数)
print("2j. Analyst...")
af = load('analyst_forecast', cols=['Stkcd', 'Fenddt', 'ReportID'])
af['Stkcd'] = af['Stkcd'].astype(int)
af['Fenddt'] = af['Fenddt'].astype(str)
af = af[af['Fenddt'].str.endswith('12-31')].copy()
af['year'] = pd.to_datetime(af['Fenddt']).dt.year
# 每个企业-年有多少不同分析师(用ReportID去重)
analyst_count = af.groupby(['Stkcd', 'year'])['ReportID'].nunique().reset_index()
analyst_count.columns = ['Stkcd', 'year', 'AnalystCount']
analyst_count['Analyst'] = np.log(1 + analyst_count['AnalystCount'])
panel = panel.merge(analyst_count[['Stkcd', 'year', 'Analyst']], on=['Stkcd', 'year'], how='left')
panel['Analyst'] = panel['Analyst'].fillna(0)  # 无覆盖=0
print(f"   Analyst有值: {(panel.Analyst > 0).mean():.1%}")

# 2k. AuditType from audit
print("2k. AuditType...")
aud = load('audit', cols=['Stkcd', 'Accper', 'Audittyp'])
aud['Stkcd'] = aud['Stkcd'].astype(int)
aud['Accper'] = aud['Accper'].astype(str)
aud = aud[aud['Accper'].str.endswith('12-31')].copy()
aud['year'] = pd.to_datetime(aud['Accper']).dt.year
aud['AuditType'] = (aud['Audittyp'] == '标准无保留意见').astype(float)  # 标准无保留=1
aud = aud[['Stkcd', 'year', 'AuditType']].drop_duplicates(['Stkcd', 'year'], keep='last')
panel = panel.merge(aud, on=['Stkcd', 'year'], how='left')
print(f"   AuditType匹配: {panel.AuditType.notna().mean():.1%}")

# ============================================================
# 3. 样本筛选
# ============================================================
print("\n3. 样本筛选...")
print(f"   合并前: {len(panel):,} obs")

# 保留2010-2024
panel = panel[(panel.year >= 2010) & (panel.year <= 2024)]
print(f"   2010-2024: {len(panel):,} obs")

# ============================================================
# 4. 保存
# ============================================================
panel.to_parquet(f"{OUT}/panel.parquet", index=False)
print(f"\n已保存: {OUT}/panel.parquet")
print(f"最终面板: {len(panel):,} obs, {panel.Stkcd.nunique():,} firms, {panel.year.nunique()} years")
print(f"\n变量覆盖率:")
for col in panel.columns:
    pct = panel[col].notna().mean()
    print(f"  {col:<20} {pct:.1%}")

print(f"\n描述性统计:")
desc_cols = ['PriceDelay', 'DataUsage', 'DataUsage_norm', 'Size', 'Lev', 'ROA',
             'TobinQ', 'Age', 'Growth', 'BoardSize', 'IndepRatio', 'Dual',
             'Top1Share', 'SOE', 'InstHold', 'Amihud', 'Analyst', 'AuditType',
             'FinAsset', 'DataAsset']
avail_desc = [c for c in desc_cols if c in panel.columns]
print(panel[avail_desc].describe().round(4).to_string())
