"""
Step 2: 变量构造 + 面板合并 + 基准回归
依赖：先运行 preprocess_all.py 生成 data_parquet/ 下的所有parquet文件

运行方式：
    cd /Users/mac/computerscience/15会计研究
    python step2_build_panel.py

输出：
    data_parquet/panel_final.parquet  -- 最终面板数据
    results/descriptive_stats.csv     -- 描述性统计
    results/correlation_matrix.csv    -- 相关系数矩阵
    results/baseline_regression.txt   -- 基准回归结果
"""

import pandas as pd
import numpy as np
import os, gc, warnings, sys
from datetime import datetime

warnings.filterwarnings('ignore')

# ============================================================
# 路径配置
# ============================================================
PROJECT = os.path.dirname(os.path.abspath(__file__))
PQ = os.path.join(PROJECT, "data_parquet")
RESULTS = os.path.join(PROJECT, "results")
os.makedirs(RESULTS, exist_ok=True)

def load(name):
    path = os.path.join(PQ, f"{name}.parquet")
    if not os.path.exists(path):
        print(f"  WARNING: {name}.parquet not found, skipping")
        return None
    df = pd.read_parquet(path)
    print(f"  Loaded {name}: {df.shape}")
    return df

print(f"Project: {PROJECT}")
print(f"Parquet dir: {PQ}")
print(f"Results dir: {RESULTS}")
print(f"Time: {datetime.now()}\n")

# ============================================================
# PART 1: 构造因变量 -- 股价延迟 Hou-Moskowitz (2005)
# ============================================================
print("="*60)
print("PART 1: 构造股价延迟")
print("="*60)

daily = load("daily_return")
idx_daily = load("index_daily")

if daily is not None:
    # 准备日收益率
    daily['Trddt'] = pd.to_datetime(daily['Trddt'], errors='coerce')
    daily['year'] = daily['Trddt'].dt.year
    # Dretwd = 考虑现金红利再投资的日个股回报率
    daily['Stkcd'] = daily['Stkcd'].astype(int)

    # 准备市场收益率（用综合指数日行情或FF3的RiskPremium）
    # 方案A：用FF3因子数据中的市场超额收益
    ff3 = load("ff3_daily")
    if ff3 is not None:
        # P9714 = 全部A股等权, 用RiskPremium2(流通市值加权)
        mkt = ff3[ff3['MarkettypeID'] == 'P9714'][['TradingDate','RiskPremium2']].copy()
        mkt['TradingDate'] = pd.to_datetime(mkt['TradingDate'], errors='coerce')
        mkt.rename(columns={'TradingDate': 'Trddt', 'RiskPremium2': 'MktRet'}, inplace=True)
        mkt = mkt.drop_duplicates('Trddt')
    else:
        # 方案B: 用指数数据
        if idx_daily is not None:
            # 000001 = 上证综指
            mkt = idx_daily[idx_daily['Indexcd'] == '000001'][['Trddt','Retindex']].copy()
            mkt['Trddt'] = pd.to_datetime(mkt['Trddt'], errors='coerce')
            mkt.rename(columns={'Retindex': 'MktRet'}, inplace=True)
        else:
            print("ERROR: 无市场收益率数据，无法计算股价延迟")
            sys.exit(1)

    # 合并市场收益率到个股数据
    daily = daily.merge(mkt[['Trddt','MktRet']], on='Trddt', how='left')

    # 构造滞后市场收益率（1-4期）
    # 先按日期排序
    daily = daily.sort_values(['Stkcd','Trddt'])

    # 对每只股票，构造滞后市场收益率
    print("  Constructing lagged market returns...")
    daily['MktRet_L1'] = daily.groupby('Stkcd')['MktRet'].shift(1)
    daily['MktRet_L2'] = daily.groupby('Stkcd')['MktRet'].shift(2)
    daily['MktRet_L3'] = daily.groupby('Stkcd')['MktRet'].shift(3)
    daily['MktRet_L4'] = daily.groupby('Stkcd')['MktRet'].shift(4)

    # 按企业-年度计算股价延迟
    print("  Computing Hou-Moskowitz delay by firm-year...")

    def compute_delay(group):
        """对一个企业-年度组计算Hou-Moskowitz股价延迟"""
        g = group.dropna(subset=['Dretwd','MktRet','MktRet_L1','MktRet_L2','MktRet_L3','MktRet_L4'])
        if len(g) < 120:  # 至少120个交易日
            return pd.Series({'PriceDelay': np.nan, 'nobs_delay': len(g)})

        y = g['Dretwd'].values

        # 无限制模型: r_it = a + b0*r_mt + b1*r_m,t-1 + ... + b4*r_m,t-4 + e
        X_full = np.column_stack([
            np.ones(len(g)),
            g['MktRet'].values,
            g['MktRet_L1'].values,
            g['MktRet_L2'].values,
            g['MktRet_L3'].values,
            g['MktRet_L4'].values
        ])

        # 受限模型: r_it = a + b0*r_mt + e
        X_rest = np.column_stack([
            np.ones(len(g)),
            g['MktRet'].values
        ])

        try:
            # OLS
            beta_full = np.linalg.lstsq(X_full, y, rcond=None)
            resid_full = y - X_full @ beta_full[0]
            ss_res_full = np.sum(resid_full**2)
            ss_tot = np.sum((y - np.mean(y))**2)
            r2_full = 1 - ss_res_full / ss_tot if ss_tot > 0 else 0

            beta_rest = np.linalg.lstsq(X_rest, y, rcond=None)
            resid_rest = y - X_rest @ beta_rest[0]
            ss_res_rest = np.sum(resid_rest**2)
            r2_rest = 1 - ss_res_rest / ss_tot if ss_tot > 0 else 0

            # Delay = 1 - R2_rest / R2_full
            if r2_full > 0.001:
                delay = 1 - r2_rest / r2_full
            else:
                delay = np.nan

            return pd.Series({'PriceDelay': delay, 'nobs_delay': len(g)})
        except:
            return pd.Series({'PriceDelay': np.nan, 'nobs_delay': len(g)})

    delay_df = daily.groupby(['Stkcd','year']).apply(compute_delay).reset_index()
    print(f"  Price Delay computed: {delay_df.shape}")
    print(f"  Non-null: {delay_df['PriceDelay'].notna().sum():,}")
    print(f"  Mean: {delay_df['PriceDelay'].mean():.4f}, Median: {delay_df['PriceDelay'].median():.4f}")

    delay_df.to_parquet(os.path.join(PQ, "price_delay.parquet"), index=False)
    del daily; gc.collect()
else:
    print("  WARNING: daily_return.parquet not found")
    print("  Trying to load pre-computed price_delay...")
    delay_df = load("price_delay")

# ============================================================
# PART 2: 构造控制变量面板
# ============================================================
print("\n" + "="*60)
print("PART 2: 构造控制变量面板")
print("="*60)

# --- 面板骨架 ---
print("\n--- 面板骨架 ---")
fi = load("firm_info")
fi['EndDate'] = pd.to_datetime(fi['EndDate'], errors='coerce')
fi['year'] = fi['EndDate'].dt.year.astype(int)
fi['Stkcd'] = fi['Symbol'].astype(int)

# 过滤：正常上市、非金融、2010-2024
fi = fi[fi['LISTINGSTATE'] == '正常上市'].copy()
fi['ind1'] = fi['IndustryCode'].astype(str).str[0]
fi = fi[fi['ind1'] != 'J'].copy()
fi = fi[(fi['year'] >= 2010) & (fi['year'] <= 2024)].copy()

# Age
fi['LISTINGDATE'] = pd.to_datetime(fi['LISTINGDATE'], errors='coerce')
fi['Age'] = np.log((fi['EndDate'] - fi['LISTINGDATE']).dt.days / 365.25 + 1)

panel = fi[['Stkcd','year','ShortName','IndustryCode','IndustryCodeC','Age']].copy()
print(f"  骨架: {panel.shape}")

# --- CNRDS自变量 ---
print("\n--- CNRDS数据要素指数 ---")
cnrds = load("cnrds_index")
if cnrds is not None:
    # 跳过中文描述行（如有）
    if cnrds.iloc[0]['Scode'] == '股票代码':
        cnrds = cnrds.iloc[1:].copy()
        for col in ['count_sde','count_ddc','count_dba','count_mvd','TWFre_count','Term_Only','Year']:
            cnrds[col] = pd.to_numeric(cnrds[col], errors='coerce')

    cnrds = cnrds[cnrds['Texttype'] == '上市公司年报'].copy()
    cnrds['Stkcd'] = pd.to_numeric(cnrds['Scode'].astype(str).str.lstrip('0'), errors='coerce').astype('Int64')
    cnrds['year'] = cnrds['Year'].astype(int)
    cnrds.rename(columns={'TWFre_count': 'DataUsage', 'Term_Only': 'DataUsage_term'}, inplace=True)
    panel = panel.merge(
        cnrds[['Stkcd','year','DataUsage','DataUsage_term','count_sde','count_ddc','count_dba','count_mvd']],
        on=['Stkcd','year'], how='left')
    print(f"  DataUsage non-null: {panel['DataUsage'].notna().sum():,}")

# --- Size ---
print("\n--- Size ---")
mr = load("monthly_return")
if mr is not None:
    mr['Trdmnt'] = pd.to_datetime(mr['Trdmnt'].astype(str), format='%Y-%m', errors='coerce')
    mr['year'] = mr['Trdmnt'].dt.year.astype(int)
    mr['month'] = mr['Trdmnt'].dt.month
    size = mr[(mr['month'] == 12) & (mr['Markettype'].isin([1,4,16]))].copy()
    size = size.groupby(['Stkcd','year'])['Msmvttl'].last().reset_index()
    size['Size'] = np.log(size['Msmvttl'].clip(lower=1))
    panel = panel.merge(size[['Stkcd','year','Size']], on=['Stkcd','year'], how='left')
    print(f"  Size non-null: {panel['Size'].notna().sum():,}")
    del mr, size; gc.collect()

# --- Lev & ROA (need balance_sheet + income_stmt) ---
print("\n--- Lev & ROA ---")
bs = load("balance_sheet")
ins = load("income_stmt")
if bs is not None and ins is not None and len(bs.columns) > 3:
    # 资产负债表
    bs['Accper'] = pd.to_datetime(bs['Accper'], errors='coerce')
    bs['year'] = bs['Accper'].dt.year.astype(int)
    bs['Stkcd'] = bs['Stkcd'].astype(int)
    # Typrep=A 合并报表
    bs_a = bs[bs['Typrep'] == 'A'].copy()
    bs_a = bs_a.groupby(['Stkcd','year']).last().reset_index()

    # Lev = 负债合计 / 资产总计
    bs_a['Lev'] = bs_a['A002000000'] / bs_a['A001000000'].replace(0, np.nan)

    # 金融资产配置比例
    fin_cols = ['A001107000','A001202000','A001211000','A001229000']
    avail_fin = [c for c in fin_cols if c in bs_a.columns]
    bs_a['FinAsset'] = bs_a[avail_fin].sum(axis=1) / bs_a['A001000000'].replace(0, np.nan)

    # 数据资产配置比例 (2024年新入表字段)
    data_cols = ['A001123101','A001218201','A001219101']
    avail_data = [c for c in data_cols if c in bs_a.columns]
    if avail_data:
        bs_a['DataAsset'] = bs_a[avail_data].sum(axis=1) / bs_a['A001000000'].replace(0, np.nan)

    panel = panel.merge(bs_a[['Stkcd','year','Lev','FinAsset'] +
                              (['DataAsset'] if 'DataAsset' in bs_a.columns else [])],
                        on=['Stkcd','year'], how='left')
    print(f"  Lev non-null: {panel['Lev'].notna().sum():,}")

    # 利润表
    ins['Accper'] = pd.to_datetime(ins['Accper'], errors='coerce')
    ins['year'] = ins['Accper'].dt.year.astype(int)
    ins['Stkcd'] = ins['Stkcd'].astype(int)
    ins_a = ins[ins['Typrep'] == 'A'].copy()
    ins_a = ins_a.groupby(['Stkcd','year']).last().reset_index()

    # 合并资产总计来计算ROA
    ins_a = ins_a.merge(bs_a[['Stkcd','year','A001000000']], on=['Stkcd','year'], how='left')
    ins_a['ROA'] = ins_a['B002000000'] / ins_a['A001000000'].replace(0, np.nan)

    # Growth = 营业收入增长率
    ins_a = ins_a.sort_values(['Stkcd','year'])
    ins_a['Revenue_lag'] = ins_a.groupby('Stkcd')['B001101000'].shift(1)
    ins_a['Growth'] = (ins_a['B001101000'] - ins_a['Revenue_lag']) / ins_a['Revenue_lag'].replace(0, np.nan)

    panel = panel.merge(ins_a[['Stkcd','year','ROA','Growth']], on=['Stkcd','year'], how='left')
    print(f"  ROA non-null: {panel['ROA'].notna().sum():,}")
    del bs, ins, bs_a, ins_a; gc.collect()
else:
    print("  WARNING: balance_sheet/income_stmt not available or incomplete, skipping Lev/ROA/Growth")

# --- TobinQ ---
print("\n--- TobinQ ---")
rv = load("relative_value")
if rv is not None:
    rv['Accper'] = pd.to_datetime(rv['Accper'], errors='coerce')
    rv['year'] = rv['Accper'].dt.year.astype(int)
    rv0 = rv[rv['Source'] == 0].groupby(['Stkcd','year'])['F100901A'].last().reset_index()
    rv0.rename(columns={'F100901A': 'TobinQ'}, inplace=True)
    panel = panel.merge(rv0[['Stkcd','year','TobinQ']], on=['Stkcd','year'], how='left')
    print(f"  TobinQ non-null: {panel['TobinQ'].notna().sum():,}")
    del rv, rv0; gc.collect()

# --- BoardSize & IndepRatio ---
print("\n--- BoardSize & IndepRatio ---")
ms = load("manager_salary")
if ms is not None:
    ms['Enddate'] = pd.to_datetime(ms['Enddate'], errors='coerce')
    ms['year'] = ms['Enddate'].dt.year.astype(int)
    ms['Stkcd'] = ms['Symbol'].astype(int)
    ms1 = ms[ms['StatisticalCaliber'] == 1].groupby(['Stkcd','year']).agg(
        DirectorNumber=('DirectorNumber','last'),
        IndependentDirectorNumber=('IndependentDirectorNumber','last')
    ).reset_index()
    ms1['BoardSize'] = np.log(ms1['DirectorNumber'].clip(lower=1))
    ms1['IndepRatio'] = ms1['IndependentDirectorNumber'] / ms1['DirectorNumber'].clip(lower=1)
    panel = panel.merge(ms1[['Stkcd','year','BoardSize','IndepRatio']], on=['Stkcd','year'], how='left')
    print(f"  BoardSize non-null: {panel['BoardSize'].notna().sum():,}")
    del ms, ms1; gc.collect()

# --- Dual ---
print("\n--- Dual ---")
gov = load("governance")
if gov is not None:
    gov['Reptdt'] = pd.to_datetime(gov['Reptdt'], errors='coerce')
    gov['year'] = gov['Reptdt'].dt.year.astype(int)
    gov = gov.groupby(['Stkcd','year'])['Y1001b'].last().reset_index()
    gov['Dual'] = (gov['Y1001b'] == 2).astype(int)  # Y1001b=2 表示两职合一
    panel = panel.merge(gov[['Stkcd','year','Dual']], on=['Stkcd','year'], how='left')
    print(f"  Dual non-null: {panel['Dual'].notna().sum():,}")
    del gov; gc.collect()

# --- Top1Share & SOE ---
print("\n--- Top1Share & SOE ---")
eq = load("equity_nature")
if eq is not None:
    eq['EndDate'] = pd.to_datetime(eq['EndDate'], errors='coerce')
    eq['year'] = eq['EndDate'].dt.year.astype(int)
    eq['Stkcd'] = eq['Symbol'].astype(int)
    eq = eq.groupby(['Stkcd','year']).agg(
        Top1Share=('LargestHolderRate','last'),
        EquityNatureID=('EquityNatureID','last')
    ).reset_index()
    eq['SOE'] = (eq['EquityNatureID'] == 1).astype(int)
    panel = panel.merge(eq[['Stkcd','year','Top1Share','SOE']], on=['Stkcd','year'], how='left')
    print(f"  Top1Share non-null: {panel['Top1Share'].notna().sum():,}")
    del eq; gc.collect()

# --- InstHold ---
print("\n--- InstHold ---")
ih = load("inst_holding")
if ih is not None:
    ih['EndDate'] = pd.to_datetime(ih['EndDate'], errors='coerce')
    ih['year'] = ih['EndDate'].dt.year.astype(int)
    ih['Stkcd'] = ih['Symbol'].astype(int)
    ih12 = ih[ih['EndDate'].dt.month == 12].groupby(['Stkcd','year'])['InsInvestorProp'].last().reset_index()
    ih12.rename(columns={'InsInvestorProp': 'InstHold'}, inplace=True)
    panel = panel.merge(ih12[['Stkcd','year','InstHold']], on=['Stkcd','year'], how='left')
    print(f"  InstHold non-null: {panel['InstHold'].notna().sum():,}")
    del ih, ih12; gc.collect()

# --- Analyst ---
print("\n--- Analyst ---")
analyst = load("analyst_forecast")
if analyst is not None:
    # 按企业-年度计算分析师覆盖人数
    analyst['Stkcd'] = analyst['Stkcd'].astype(int) if 'Stkcd' in analyst.columns else analyst['Symbol'].astype(int)
    date_col = 'Fenddt' if 'Fenddt' in analyst.columns else 'EndDate'
    analyst[date_col] = pd.to_datetime(analyst[date_col], errors='coerce')
    analyst['year'] = analyst[date_col].dt.year.astype(int)

    # 按分析师姓名去重
    name_col = 'Analyst' if 'Analyst' in analyst.columns else analyst.columns[2]
    ana_count = analyst.groupby(['Stkcd','year'])[name_col].nunique().reset_index()
    ana_count.rename(columns={name_col: 'AnalystCount'}, inplace=True)
    ana_count['Analyst'] = np.log(1 + ana_count['AnalystCount'])
    panel = panel.merge(ana_count[['Stkcd','year','Analyst','AnalystCount']], on=['Stkcd','year'], how='left')
    panel['Analyst'] = panel['Analyst'].fillna(0)
    panel['AnalystCount'] = panel['AnalystCount'].fillna(0)
    print(f"  Analyst non-null: {panel['Analyst'].notna().sum():,}")
    del analyst, ana_count; gc.collect()

# --- Amihud ---
print("\n--- Amihud ---")
amihud = load("amihud_daily")
if amihud is not None:
    amihud['TradingDate'] = pd.to_datetime(amihud['TradingDate'], errors='coerce')
    amihud['year'] = amihud['TradingDate'].dt.year.astype(int)
    amihud['Stkcd'] = amihud['Stkcd'].astype(int)
    ami_year = amihud.groupby(['Stkcd','year'])['Amihud'].mean().reset_index()
    ami_year.rename(columns={'Amihud': 'Amihud_mean'}, inplace=True)
    panel = panel.merge(ami_year[['Stkcd','year','Amihud_mean']], on=['Stkcd','year'], how='left')
    print(f"  Amihud non-null: {panel['Amihud_mean'].notna().sum():,}")
    del amihud, ami_year; gc.collect()

# --- AuditType ---
print("\n--- AuditType ---")
au = load("audit")
if au is not None:
    au['Accper'] = pd.to_datetime(au['Accper'], errors='coerce')
    au['year'] = au['Accper'].dt.year.astype(int)
    au = au.groupby(['Stkcd','year'])['Audittyp'].last().reset_index()
    au['AuditClean'] = (au['Audittyp'] == 1).astype(int)
    panel = panel.merge(au[['Stkcd','year','AuditClean']], on=['Stkcd','year'], how='left')
    print(f"  AuditClean non-null: {panel['AuditClean'].notna().sum():,}")
    del au; gc.collect()

# ============================================================
# PART 3: 合并因变量
# ============================================================
print("\n" + "="*60)
print("PART 3: 合并因变量")
print("="*60)

if delay_df is not None:
    panel = panel.merge(delay_df[['Stkcd','year','PriceDelay','nobs_delay']], on=['Stkcd','year'], how='left')
    print(f"  PriceDelay non-null: {panel['PriceDelay'].notna().sum():,}")

# ============================================================
# PART 4: 数据清洗
# ============================================================
print("\n" + "="*60)
print("PART 4: 数据清洗")
print("="*60)

# Winsorize连续变量在1%和99%分位
from scipy.stats import mstats

continuous_vars = ['PriceDelay','DataUsage','Size','Lev','ROA','TobinQ','Age','Growth',
                   'BoardSize','IndepRatio','Top1Share','InstHold','Amihud_mean','Analyst']
continuous_vars = [v for v in continuous_vars if v in panel.columns]

print(f"  Winsorizing {len(continuous_vars)} continuous variables at 1%/99%...")
for var in continuous_vars:
    if panel[var].notna().sum() > 100:
        vals = panel[var].dropna()
        q01, q99 = vals.quantile(0.01), vals.quantile(0.99)
        panel[var] = panel[var].clip(lower=q01, upper=q99)

# 保存
panel.to_parquet(os.path.join(PQ, "panel_final.parquet"), index=False)
print(f"\n  Saved panel_final.parquet: {panel.shape}")

# ============================================================
# PART 5: 描述性统计
# ============================================================
print("\n" + "="*60)
print("PART 5: 描述性统计")
print("="*60)

# 变量覆盖率
print("\n变量覆盖率:")
for col in panel.columns:
    n = panel[col].notna().sum()
    pct = n / len(panel) * 100
    print(f"  {col:25s}: {n:>7,} / {len(panel):,} ({pct:.1f}%)")

# 描述性统计表
stat_vars = [v for v in ['PriceDelay','DataUsage','Size','Lev','ROA','TobinQ','Age','Growth',
             'BoardSize','IndepRatio','Dual','Top1Share','SOE','InstHold','Amihud_mean',
             'Analyst','AuditClean','FinAsset','DataAsset'] if v in panel.columns]

desc = panel[stat_vars].describe(percentiles=[.01,.05,.25,.5,.75,.95,.99]).T
desc = desc[['count','mean','std','1%','25%','50%','75%','99%']]
desc.to_csv(os.path.join(RESULTS, "descriptive_stats.csv"))
print("\n描述性统计:")
print(desc.round(4).to_string())

# 相关系数矩阵
corr = panel[stat_vars].corr()
corr.to_csv(os.path.join(RESULTS, "correlation_matrix.csv"))
print(f"\n相关系数矩阵已保存至 {RESULTS}/correlation_matrix.csv")

# ============================================================
# PART 6: 基准回归
# ============================================================
print("\n" + "="*60)
print("PART 6: 基准回归")
print("="*60)

try:
    import linearmodels
    from linearmodels.panel import PanelOLS
    HAS_LINEARMODELS = True
except ImportError:
    HAS_LINEARMODELS = False
    print("  linearmodels not installed, trying statsmodels...")

try:
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

if 'PriceDelay' not in panel.columns or panel['PriceDelay'].notna().sum() < 100:
    print("  WARNING: PriceDelay not available, skipping regression")
    print("  Please run preprocess_all.py first to generate daily_return.parquet")
else:
    # 回归样本
    reg_vars = ['PriceDelay','DataUsage','Size','TobinQ','Age','BoardSize','IndepRatio',
                'Top1Share','InstHold','AuditClean']
    # 加入可选变量
    for v in ['Lev','ROA','Growth','Dual','Amihud_mean','Analyst','FinAsset']:
        if v in panel.columns and panel[v].notna().sum() > 1000:
            reg_vars.append(v)

    reg_df = panel[['Stkcd','year','IndustryCode'] + reg_vars].dropna(subset=reg_vars)
    print(f"  回归样本: {reg_df.shape}")

    if len(reg_df) < 100:
        print("  WARNING: 回归样本不足100，跳过回归")
    else:
        # DataUsage标准化
        reg_df['DataUsage_std'] = (reg_df['DataUsage'] - reg_df['DataUsage'].mean()) / reg_df['DataUsage'].std()

        results_text = []

        if HAS_LINEARMODELS:
            # 用linearmodels做面板固定效应
            reg_df = reg_df.set_index(['Stkcd','year'])

            # 模型1: 无控制变量
            dep = reg_df['PriceDelay']
            exog1 = sm.add_constant(reg_df[['DataUsage_std']])
            mod1 = PanelOLS(dep, exog1, entity_effects=True, time_effects=True)
            res1 = mod1.fit(cov_type='clustered', cluster_entity=True)

            # 模型2: 全控制变量
            controls = [v for v in reg_vars if v not in ['PriceDelay','DataUsage']]
            exog2 = sm.add_constant(reg_df[['DataUsage_std'] + controls])
            mod2 = PanelOLS(dep, exog2, entity_effects=True, time_effects=True)
            res2 = mod2.fit(cov_type='clustered', cluster_entity=True)

            results_text.append("=" * 80)
            results_text.append("基准回归: PriceDelay = α + β·DataUsage + γ·Controls + FE(firm+year) + ε")
            results_text.append("=" * 80)
            results_text.append(f"\n模型1 (无控制变量):\n{res1.summary}")
            results_text.append(f"\n模型2 (全控制变量):\n{res2.summary}")

        elif HAS_STATSMODELS:
            # 用statsmodels做FE (demeaned)
            # 简化版: 加入企业和年份虚拟变量
            reg_df['firm_id'] = pd.Categorical(reg_df['Stkcd']).codes
            reg_df['year_id'] = pd.Categorical(reg_df['year']).codes

            controls = [v for v in reg_vars if v not in ['PriceDelay','DataUsage']]

            # 组内去均值做FE
            for col in ['PriceDelay','DataUsage_std'] + controls:
                firm_mean = reg_df.groupby('Stkcd')[col].transform('mean')
                year_mean = reg_df.groupby('year')[col].transform('mean')
                grand_mean = reg_df[col].mean()
                reg_df[f'{col}_dm'] = reg_df[col] - firm_mean - year_mean + grand_mean

            # 模型1
            X1 = sm.add_constant(reg_df['DataUsage_std_dm'])
            mod1 = sm.OLS(reg_df['PriceDelay_dm'], X1).fit(cov_type='cluster',
                cov_kwds={'groups': reg_df['Stkcd']})

            # 模型2
            ctrl_dm = [f'{c}_dm' for c in controls]
            X2 = sm.add_constant(reg_df[['DataUsage_std_dm'] + ctrl_dm])
            mod2 = sm.OLS(reg_df['PriceDelay_dm'], X2).fit(cov_type='cluster',
                cov_kwds={'groups': reg_df['Stkcd']})

            results_text.append("=" * 80)
            results_text.append("基准回归 (FE via demeaning): PriceDelay = α + β·DataUsage + γ·Controls + FE + ε")
            results_text.append("=" * 80)
            results_text.append(f"\n模型1:\n{mod1.summary()}")
            results_text.append(f"\n模型2:\n{mod2.summary()}")

        # 保存结果
        if results_text:
            output = "\n".join(results_text)
            with open(os.path.join(RESULTS, "baseline_regression.txt"), 'w') as f:
                f.write(output)
            print(output[:3000])
            print(f"\n  完整结果已保存至 {RESULTS}/baseline_regression.txt")

print("\n" + "="*60)
print("ALL DONE")
print("="*60)
