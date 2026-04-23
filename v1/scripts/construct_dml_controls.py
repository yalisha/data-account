"""
构造DML额外控制变量
将新变量合并到 panel_dml.parquet

新增变量:
  1. CFO         经营现金流/总资产
  2. RetVol      年度收益率波动率 (日收益率标准差)
  3. Turnover    年均换手率 (日成交金额/流通市值)
  4. Intangible  无形资产净额/总资产
  5. PPE         固定资产净额/总资产
  6. BM          账面市值比 (所有者权益/总市值)
  7. Roastd      前3年ROA标准差
  8. Employee    ln(员工人数)
  9. ShareholderNum  ln(股东总数)
  10. Manhold    管理层持股比例
  11. Opinion    审计意见 (标准无保留=1)
  12. Balance    股权制衡度 (第2大/第1大股东持股)
  13. Separation 两权分离度 (控制权-所有权)
  14. Market     省份市场化指数
"""

import pandas as pd
import numpy as np
import os, time, warnings
warnings.filterwarnings('ignore')

BASE = "/Users/mac/computerscience/15会计研究"
DATA = f"{BASE}/data_parquet"
RAW = "/Users/mac/computerscience/第三方资料/第三方数据资源"
SH_BASE = f"{RAW}/上市公司股东"

t0 = time.time()
print("=" * 60)
print("构造DML额外控制变量")
print("=" * 60)

# ================================================================
# 加载主面板
# ================================================================
print("\n1. 加载主面板...")
panel = pd.read_parquet(f"{DATA}/panel.parquet")
print(f"   panel: {len(panel):,} obs, {panel.Stkcd.nunique():,} firms")

# ================================================================
# 1. CFO = 经营现金流 / 总资产
# ================================================================
print("\n2. CFO (经营现金流/总资产)...")
cf = pd.read_parquet(f"{DATA}/cashflow.parquet")
cf = cf[cf['Typrep'] == 'A'].copy()
cf['Accper'] = pd.to_datetime(cf['Accper'])
cf = cf[cf['Accper'].dt.month == 12]
cf['year'] = cf['Accper'].dt.year
cf = cf.sort_values(['Stkcd', 'year', 'Accper']).drop_duplicates(
    subset=['Stkcd', 'year'], keep='last')

bs = pd.read_parquet(f"{DATA}/balance_sheet.parquet")
bs = bs[bs['Typrep'] == 'A'].copy()
bs['Accper'] = pd.to_datetime(bs['Accper'])
bs = bs[bs['Accper'].dt.month == 12]
bs['year'] = bs['Accper'].dt.year
bs = bs.sort_values(['Stkcd', 'year', 'Accper']).drop_duplicates(
    subset=['Stkcd', 'year'], keep='last')

cfo_df = cf[['Stkcd', 'year', 'C001000000']].merge(
    bs[['Stkcd', 'year', 'A001000000', 'A001218000', 'A003100000']],
    on=['Stkcd', 'year'], how='inner')
cfo_df['CFO'] = cfo_df['C001000000'] / cfo_df['A001000000']
cfo_df['Intangible'] = cfo_df['A001218000'] / cfo_df['A001000000']
cfo_df['CFO'] = cfo_df['CFO'].replace([np.inf, -np.inf], np.nan)
cfo_df['Intangible'] = cfo_df['Intangible'].replace([np.inf, -np.inf], np.nan)
print(f"   CFO: {cfo_df['CFO'].notna().sum():,} obs")
print(f"   Intangible: {cfo_df['Intangible'].notna().sum():,} obs")

# ================================================================
# 2. RetVol & Turnover (从日收益率数据)
# ================================================================
print("\n3. RetVol & Turnover (日度数据)...")
t1 = time.time()
dr = pd.read_parquet(f"{DATA}/daily_return.parquet",
                     columns=['Stkcd', 'Trddt', 'Dretwd', 'Dnvaltrd', 'Dsmvosd', 'Markettype'])
dr = dr[dr['Markettype'].isin([1, 4, 16, 32])].copy()
dr['Trddt'] = pd.to_datetime(dr['Trddt'])
dr['year'] = dr['Trddt'].dt.year
dr['Dretwd'] = pd.to_numeric(dr['Dretwd'], errors='coerce')
dr['Dnvaltrd'] = pd.to_numeric(dr['Dnvaltrd'], errors='coerce')
dr['Dsmvosd'] = pd.to_numeric(dr['Dsmvosd'], errors='coerce')

# 换手率 = 成交金额 / 流通市值
dr['daily_turnover'] = dr['Dnvaltrd'] / dr['Dsmvosd']
dr['daily_turnover'] = dr['daily_turnover'].replace([np.inf, -np.inf], np.nan)

def calc_market_vars(g):
    ret = g['Dretwd'].dropna()
    turn = g['daily_turnover'].dropna()
    if len(ret) < 60:
        return pd.Series({'RetVol': np.nan, 'Turnover': np.nan, 'AvgMktCap': np.nan})
    return pd.Series({
        'RetVol': ret.std(),
        'Turnover': turn.mean(),
        'AvgMktCap': g['Dsmvosd'].mean(),
    })

mkt_vars = dr.groupby(['Stkcd', 'year']).apply(calc_market_vars).reset_index()
print(f"   RetVol: {mkt_vars['RetVol'].notna().sum():,} obs")
print(f"   Turnover: {mkt_vars['Turnover'].notna().sum():,} obs")
print(f"   耗时: {time.time()-t1:.1f}s")

# ================================================================
# 3. BM (账面市值比) = 所有者权益 / 年末总市值
# ================================================================
print("\n4. BM (账面市值比)...")
# 用月度数据取年末总市值
mr = pd.read_parquet(f"{DATA}/monthly_return.parquet",
                     columns=['Stkcd', 'Trdmnt', 'Msmvttl', 'Markettype'])
mr = mr[mr['Markettype'].isin([1, 4, 16, 32])].copy()
mr['Trdmnt'] = pd.to_datetime(mr['Trdmnt'])
mr['year'] = mr['Trdmnt'].dt.year
mr['month'] = mr['Trdmnt'].dt.month
# 取12月数据作为年末市值
yr_end_cap = mr[mr['month'] == 12][['Stkcd', 'year', 'Msmvttl']].copy()
yr_end_cap = yr_end_cap.sort_values(['Stkcd', 'year']).drop_duplicates(
    subset=['Stkcd', 'year'], keep='last')
yr_end_cap.columns = ['Stkcd', 'year', 'YearEndMktCap']
# Msmvttl 单位是千元, A003100000(所有者权益) 单位是元
bm_df = bs[['Stkcd', 'year', 'A003100000']].merge(yr_end_cap, on=['Stkcd', 'year'], how='inner')
bm_df['BM'] = bm_df['A003100000'] / (bm_df['YearEndMktCap'] * 1000)
bm_df['BM'] = bm_df['BM'].replace([np.inf, -np.inf], np.nan)
bm_med = bm_df['BM'].median()
print(f"   BM median: {bm_med:.4f}")
if bm_med < 0.01 or bm_med > 100:
    bm_df['BM'] = bm_df['A003100000'] / bm_df['YearEndMktCap']
    bm_med2 = bm_df['BM'].median()
    print(f"   调整后 BM median: {bm_med2:.4f}")
print(f"   BM: {bm_df['BM'].notna().sum():,} obs")

# ================================================================
# 4. Roastd (前3年ROA标准差)
# ================================================================
print("\n5. Roastd (前3年ROA波动)...")
roa_panel = panel[['Stkcd', 'year', 'ROA']].dropna().sort_values(['Stkcd', 'year'])
roa_panel['Roastd'] = roa_panel.groupby('Stkcd')['ROA'].transform(
    lambda x: x.rolling(window=3, min_periods=3).std())
print(f"   Roastd: {roa_panel['Roastd'].notna().sum():,} obs")

# ================================================================
# 5. Employee & ShareholderNum & Manhold (从治理数据)
# ================================================================
print("\n6. Employee, ShareholderNum, Manhold (治理数据)...")
gov = pd.read_parquet(f"{DATA}/governance.parquet")
gov['Reptdt'] = pd.to_datetime(gov['Reptdt'])
gov['year'] = gov['Reptdt'].dt.year
gov = gov.sort_values(['Stkcd', 'year', 'Reptdt']).drop_duplicates(
    subset=['Stkcd', 'year'], keep='last')

gov['Employee'] = np.log(gov['Y0601b'].clip(lower=1))
gov['ShareholderNum'] = np.log(gov['Y0401b'].clip(lower=1))
gov['Manhold'] = gov['ManagerHoldsharesRatio']
gov_vars = gov[['Stkcd', 'year', 'Employee', 'ShareholderNum', 'Manhold']].copy()
print(f"   Employee: {gov_vars['Employee'].notna().sum():,} obs")
print(f"   ShareholderNum: {gov_vars['ShareholderNum'].notna().sum():,} obs")
print(f"   Manhold: {gov_vars['Manhold'].notna().sum():,} obs")

# ================================================================
# 6. Opinion (审计意见)
# ================================================================
print("\n7. Opinion (审计意见)...")
audit = pd.read_parquet(f"{DATA}/audit.parquet")
audit['Accper'] = pd.to_datetime(audit['Accper'])
audit['year'] = audit['Accper'].dt.year
audit = audit.sort_values(['Stkcd', 'year', 'Accper']).drop_duplicates(
    subset=['Stkcd', 'year'], keep='last')
audit['Opinion'] = (audit['Audittyp'] == '标准无保留意见').astype(int)
opinion_df = audit[['Stkcd', 'year', 'Opinion']].copy()
print(f"   Opinion: {opinion_df['Opinion'].notna().sum():,} obs")
print(f"   标准无保留占比: {opinion_df['Opinion'].mean():.2%}")

# ================================================================
# 7. Balance (股权制衡度 = 第2大/第1大)
# ================================================================
print("\n8. Balance (股权制衡度)...")
t10 = pd.read_parquet(f"{DATA}/top10_shareholders.parquet")
t10['Reptdt'] = pd.to_datetime(t10['Reptdt'])
t10['year'] = t10['Reptdt'].dt.year
# S0501b = 排名, S0301b = 持股比例
t10['S0501b'] = pd.to_numeric(t10['S0501b'], errors='coerce')
t10['S0301b'] = pd.to_numeric(t10['S0301b'], errors='coerce')

# 取年末数据
t10_yr = t10[t10['Reptdt'].dt.month == 12].copy()
top1 = t10_yr[t10_yr['S0501b'] == 1][['Stkcd', 'year', 'S0301b']].rename(
    columns={'S0301b': 'Top1'})
top2 = t10_yr[t10_yr['S0501b'] == 2][['Stkcd', 'year', 'S0301b']].rename(
    columns={'S0301b': 'Top2'})
top1 = top1.drop_duplicates(subset=['Stkcd', 'year'], keep='last')
top2 = top2.drop_duplicates(subset=['Stkcd', 'year'], keep='last')
balance_df = top1.merge(top2, on=['Stkcd', 'year'], how='inner')
balance_df['Balance'] = balance_df['Top2'] / balance_df['Top1']
balance_df['Balance'] = balance_df['Balance'].replace([np.inf, -np.inf], np.nan)
balance_df = balance_df[['Stkcd', 'year', 'Balance']]
print(f"   Balance: {balance_df['Balance'].notna().sum():,} obs")

# ================================================================
# 8. Separation (两权分离度, 新下载数据)
# ================================================================
print("\n9. Separation (两权分离度)...")
contrshr_path = f"{SH_BASE}/上市公司控制人文件162114431(仅供四川大学使用)/HLD_Contrshr.xlsx"
contrshr = pd.read_excel(contrshr_path, header=0, skiprows=[1, 2],
                         usecols=['Stkcd', 'Reptdt', 'S0701a', 'S0704b', 'S0704c', 'Seperation'])
contrshr['Reptdt'] = pd.to_datetime(contrshr['Reptdt'])
contrshr['year'] = contrshr['Reptdt'].dt.year
# S0701a=1 年报披露, S0701a=2 计算所得; 取判断标准=1的行
contrshr_yr = contrshr[contrshr['S0701a'] == 1].copy()
contrshr_yr = contrshr_yr.sort_values(['Stkcd', 'year', 'Reptdt']).drop_duplicates(
    subset=['Stkcd', 'year'], keep='last')
contrshr_yr = contrshr_yr[['Stkcd', 'year', 'Seperation', 'S0704b', 'S0704c']].copy()
contrshr_yr['Separation'] = contrshr_yr['Seperation'].fillna(
    contrshr_yr['S0704b'] - contrshr_yr['S0704c'])
sep_df = contrshr_yr[['Stkcd', 'year', 'Separation']].copy()
print(f"   Separation: {sep_df['Separation'].notna().sum():,} obs")

# ================================================================
# 9. PPE (固定资产净额/总资产, 新下载资产负债表)
# ================================================================
print("\n10. PPE (固定资产/总资产)...")
bs_new_path = f"{SH_BASE}/资产负债表165211499(仅供四川大学使用)/FS_Combas.xlsx"
# 只读需要的列
bs_new = pd.read_excel(bs_new_path, header=0, skiprows=[1, 2],
                       usecols=['Stkcd', 'Accper', 'Typrep', 'A001212000', 'A001000000'])
bs_new = bs_new[bs_new['Typrep'] == 'A'].copy()
bs_new['Accper'] = pd.to_datetime(bs_new['Accper'])
bs_new = bs_new[bs_new['Accper'].dt.month == 12]
bs_new['year'] = bs_new['Accper'].dt.year
bs_new = bs_new.sort_values(['Stkcd', 'year', 'Accper']).drop_duplicates(
    subset=['Stkcd', 'year'], keep='last')
bs_new['PPE'] = bs_new['A001212000'] / bs_new['A001000000']
bs_new['PPE'] = bs_new['PPE'].replace([np.inf, -np.inf], np.nan)
ppe_df = bs_new[['Stkcd', 'year', 'PPE']].copy()
print(f"   PPE: {ppe_df['PPE'].notna().sum():,} obs")

# ================================================================
# 10. Market (市场化指数, 需要省份匹配)
# ================================================================
print("\n11. Market (市场化指数)...")
mkt_idx = pd.read_excel(
    "/Users/mac/computerscience/第三方资料/1997-2024年樊纲中国分省份市场化指数&各分项指数（附最新计算代码，匹配公司数据）_/1997-2024年市场化指数和各分项指数.xlsx")
mkt_idx = mkt_idx[['省份', 'year', 'market']].copy()
mkt_idx = mkt_idx.rename(columns={'省份': 'Province_short', 'market': 'Market'})

# 匹配省份: firm_info用全称, 市场化指数用简称, 需要映射
fi = pd.read_parquet(f"{DATA}/firm_info.parquet",
                     columns=['Symbol', 'EndDate', 'PROVINCE'])
fi = fi.rename(columns={'Symbol': 'Stkcd'})
fi['EndDate'] = pd.to_datetime(fi['EndDate'])
fi['year'] = fi['EndDate'].dt.year
fi = fi.sort_values(['Stkcd', 'year', 'EndDate']).drop_duplicates(
    subset=['Stkcd', 'year'], keep='last')
fi = fi[['Stkcd', 'year', 'PROVINCE']].copy()

# 全称 -> 简称映射
def province_to_short(name):
    if pd.isna(name):
        return np.nan
    # 直辖市: 去掉"市"
    for city in ['北京', '上海', '天津', '重庆']:
        if name.startswith(city):
            return city
    # 自治区: 取前几个字
    mapping = {
        '内蒙古自治区': '内蒙古', '广西壮族自治区': '广西',
        '西藏自治区': '西藏', '宁夏回族自治区': '宁夏',
        '新疆维吾尔自治区': '新疆', '香港特别行政区': '香港',
    }
    if name in mapping:
        return mapping[name]
    # 省: 去掉"省"
    return name.replace('省', '')

fi['Province_short'] = fi['PROVINCE'].apply(province_to_short)

# 合并
fi_market = fi.merge(mkt_idx, on=['Province_short', 'year'], how='left')
fi_market = fi_market[['Stkcd', 'year', 'Market']].copy()
print(f"   Market: {fi_market['Market'].notna().sum():,} obs")

# ================================================================
# 合并所有新变量到面板
# ================================================================
print("\n" + "=" * 60)
print("合并所有新变量...")
print("=" * 60)

# 逐步合并
panel = panel.merge(cfo_df[['Stkcd', 'year', 'CFO', 'Intangible']],
                    on=['Stkcd', 'year'], how='left')
panel = panel.merge(mkt_vars[['Stkcd', 'year', 'RetVol', 'Turnover']],
                    on=['Stkcd', 'year'], how='left')
panel = panel.merge(bm_df[['Stkcd', 'year', 'BM']],
                    on=['Stkcd', 'year'], how='left')
panel = panel.merge(roa_panel[['Stkcd', 'year', 'Roastd']],
                    on=['Stkcd', 'year'], how='left')
panel = panel.merge(gov_vars, on=['Stkcd', 'year'], how='left')
panel = panel.merge(opinion_df, on=['Stkcd', 'year'], how='left')
panel = panel.merge(balance_df, on=['Stkcd', 'year'], how='left')
panel = panel.merge(sep_df, on=['Stkcd', 'year'], how='left')
panel = panel.merge(ppe_df, on=['Stkcd', 'year'], how='left')
panel = panel.merge(fi_market, on=['Stkcd', 'year'], how='left')

# 汇总覆盖率
new_vars = ['CFO', 'RetVol', 'Turnover', 'Intangible', 'PPE', 'BM',
            'Roastd', 'Employee', 'ShareholderNum', 'Manhold',
            'Opinion', 'Balance', 'Separation', 'Market']
print(f"\n{'变量':<20} {'非空数':>8} {'覆盖率':>8}")
print("-" * 40)
for v in new_vars:
    n = panel[v].notna().sum()
    pct = n / len(panel) * 100
    print(f"{v:<20} {n:>8,} {pct:>7.1f}%")

# 保存
out_path = f"{DATA}/panel_dml.parquet"
panel.to_parquet(out_path, index=False)
print(f"\n保存: {out_path}")
print(f"Shape: {panel.shape}")
print(f"总耗时: {time.time()-t0:.0f}s")
