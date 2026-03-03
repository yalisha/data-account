"""
数据要素与资本定价效率 - 全量数据预处理脚本
在本机用 Claude Code 或直接 python 运行
输出: parquet/ 目录下的所有清洗后数据

使用前先安装依赖:
  pip install pandas openpyxl pyarrow
"""
import pandas as pd
import os, gc, sys, warnings, time
warnings.filterwarnings('ignore')

# ============ 路径配置 ============
DATA_DIR = "/Users/mac/computerscience/第三方资料/第三方数据资源/上市公司财务信息"
ANNUAL_DIR = "/Users/mac/computerscience/第三方资料/第三方数据资源/2001~2024年年报"
CNRDS_DIR = DATA_DIR  # CNRDS文件也在这个目录

# 输出目录
OUT_DIR = "/Users/mac/computerscience/15会计研究/data_parquet"
os.makedirs(OUT_DIR, exist_ok=True)

# ============ 辅助函数 ============

def find_file(base_dir, keyword, ext='.xlsx'):
    """在目录下找包含keyword的文件"""
    for root, dirs, files in os.walk(base_dir):
        for f in files:
            if keyword in f and f.endswith(ext):
                return os.path.join(root, f)
    return None

def find_all_files(base_dir, keyword, ext='.xlsx'):
    """找所有匹配的文件"""
    results = []
    for root, dirs, files in os.walk(base_dir):
        for f in files:
            if keyword in f and f.endswith(ext):
                results.append(os.path.join(root, f))
    return sorted(results)

def read_csmar(path, usecols=None):
    """
    统一读取CSMAR xlsx
    CSMAR格式: row0=英文header, row1=中文说明, row2=单位, row3+=数据
    """
    df = pd.read_excel(path, header=0, skiprows=[1, 2], usecols=usecols)
    return df

def save_parquet(df, name):
    """保存为parquet并报告"""
    outpath = os.path.join(OUT_DIR, f"{name}.parquet")
    df.to_parquet(outpath, index=False)
    sz = os.path.getsize(outpath) / 1024 / 1024
    print(f"  ✓ {name}: {len(df):,} rows × {len(df.columns)} cols = {sz:.1f} MB")
    return outpath

def timer(msg):
    """简单计时"""
    t = time.time()
    print(f"\n{'='*60}\n{msg}\n{'='*60}")
    return t

# ============ 1. 财务报表 ============

t = timer("1. 资产负债表")
path = find_file(DATA_DIR, 'FS_Combas')
if path:
    bs_cols = ['Stkcd', 'Accper', 'Typrep',
        'A001107000',  # 交易性金融资产
        'A001202000',  # 可供出售金融资产
        'A001211000',  # 投资性房地产
        'A001229000',  # 其他非流动金融资产
        'A001123101',  # 数据资源(存货)
        'A001218000',  # 无形资产净额
        'A001218201',  # 数据资源(无形资产)
        'A001219000',  # 开发支出
        'A001219101',  # 数据资源(开发支出)
        'A001000000',  # 资产总计
        'A002000000',  # 负债合计
        'A001100000',  # 流动资产合计
        'A002100000',  # 流动负债合计
        'A003100000',  # 归属母公司所有者权益合计
        'A003000000',  # 所有者权益合计
    ]
    df = read_csmar(path, usecols=bs_cols)
    for col in df.columns:
        if col.startswith('A'):
            df[col] = pd.to_numeric(df[col], errors='coerce')
    # 只保留合并报表(Typrep=A)和年报(Accper以12-31结尾)
    df['Accper'] = df['Accper'].astype(str)
    save_parquet(df, 'balance_sheet')
    del df; gc.collect()
    print(f"  耗时: {time.time()-t:.1f}s")
else:
    print("  ✗ 未找到资产负债表文件")

t = timer("2. 利润表")
path = find_file(DATA_DIR, 'FS_Comins')
if path:
    is_cols = ['Stkcd', 'Accper', 'Typrep',
        'B001101000',  # 营业收入
        'B001100000',  # 营业总收入
        'B001300000',  # 营业利润
        'B001000000',  # 利润总额
        'B002000000',  # 净利润
        'B002000101',  # 归属母公司净利润
        'B001216000',  # 研发费用
    ]
    df = read_csmar(path, usecols=is_cols)
    for col in df.columns:
        if col.startswith('B'):
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df['Accper'] = df['Accper'].astype(str)
    save_parquet(df, 'income_stmt')
    del df; gc.collect()
    print(f"  耗时: {time.time()-t:.1f}s")

t = timer("3. 现金流量表")
path = find_file(DATA_DIR, 'FS_Comscfd')
if path:
    cf_cols = ['Stkcd', 'Accper', 'Typrep',
        'C001000000',  # 经营活动现金流量净额
        'C002000000',  # 投资活动现金流量净额
        'C003000000',  # 筹资活动现金流量净额
    ]
    df = read_csmar(path, usecols=cf_cols)
    for col in df.columns:
        if col.startswith('C'):
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df['Accper'] = df['Accper'].astype(str)
    save_parquet(df, 'cashflow')
    del df; gc.collect()
    print(f"  耗时: {time.time()-t:.1f}s")

# ============ 2. 交易数据 ============

t = timer("4. 日个股回报率 (多批合并)")
daily_files = find_all_files(DATA_DIR, 'TRD_Dalyr')
if daily_files:
    daily_cols = ['Stkcd', 'Trddt', 'Clsprc', 'Dnvaltrd', 'Dsmvosd', 'Dsmvtll',
                  'Dretwd', 'Dretnd', 'Markettype']
    chunks = []
    for f in daily_files:
        print(f"  读取: {os.path.basename(f)}...")
        try:
            df_chunk = read_csmar(f, usecols=daily_cols)
            chunks.append(df_chunk)
        except Exception as e:
            # 有些列可能不在某些批次
            df_chunk = read_csmar(f)
            avail = [c for c in daily_cols if c in df_chunk.columns]
            chunks.append(df_chunk[avail])
    df = pd.concat(chunks, ignore_index=True)
    # 去重(重叠的年份)
    df = df.drop_duplicates(subset=['Stkcd', 'Trddt'], keep='first')
    for col in ['Dretwd', 'Dretnd', 'Dnvaltrd', 'Dsmvosd', 'Dsmvtll', 'Clsprc']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    save_parquet(df, 'daily_return')
    del df, chunks; gc.collect()
    print(f"  耗时: {time.time()-t:.1f}s")

t = timer("5. 月个股回报率")
path = find_file(DATA_DIR, 'TRD_Mnth')
if path:
    m_cols = ['Stkcd', 'Trdmnt', 'Msmvosd', 'Msmvttl', 'Mretwd', 'Mretnd', 'Markettype']
    df = read_csmar(path, usecols=m_cols)
    for col in ['Msmvosd', 'Msmvttl', 'Mretwd', 'Mretnd']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    save_parquet(df, 'monthly_return')
    del df; gc.collect()
    print(f"  耗时: {time.time()-t:.1f}s")

t = timer("6. 市场指数")
# 原始综合指数
idx_files = find_all_files(DATA_DIR, 'TRD_Index')
if idx_files:
    df = read_csmar(idx_files[0])
    save_parquet(df, 'market_index')
    del df; gc.collect()

# 行业指数日行情 (新下载的)
idx_daily_files = find_all_files(DATA_DIR, 'IDX_Idxtrd')
if idx_daily_files:
    chunks = []
    for f in idx_daily_files:
        print(f"  读取行业指数: {os.path.basename(f)}...")
        df_c = read_csmar(f)
        chunks.append(df_c)
    df = pd.concat(chunks, ignore_index=True)
    df = df.drop_duplicates(subset=['Indexcd', 'Idxtrd01'], keep='first')
    save_parquet(df, 'industry_index_daily')
    del df, chunks; gc.collect()
print(f"  耗时: {time.time()-t:.1f}s")

# ============ 3. 因子数据 ============

t = timer("7. Fama-French因子 + 动量")
for keyword, name in [('THRFACDAY', 'ff3_daily'), ('FIVEFACDAY', 'ff5_daily'), ('MOMENTUM', 'momentum')]:
    path = find_file(DATA_DIR, keyword)
    if path:
        df = read_csmar(path)
        save_parquet(df, name)
        del df; gc.collect()
print(f"  耗时: {time.time()-t:.1f}s")

# ============ 4. 公司特征 ============

t = timer("8. 上市公司基本信息")
path = find_file(DATA_DIR, 'STK_LISTEDCOINFOANL')
if path:
    df = read_csmar(path)
    save_parquet(df, 'firm_info')
    del df; gc.collect()

t = timer("9. 相对价值指标 (TobinQ, PB, PE)")
path = find_file(DATA_DIR, 'FI_T10')
if path:
    df = read_csmar(path)
    save_parquet(df, 'relative_value')
    del df; gc.collect()

t = timer("10. 每股指标")
path = find_file(DATA_DIR, 'FI_T9')
if path:
    df = read_csmar(path)
    save_parquet(df, 'per_share')
    del df; gc.collect()

# ============ 5. 治理与股权 ============

t = timer("11. 治理综合信息")
path = find_file(DATA_DIR, 'CG_Ybasic')
if path:
    df = read_csmar(path)
    save_parquet(df, 'governance')
    del df; gc.collect()

t = timer("12. 高管人数持股薪酬")
path = find_file(DATA_DIR, 'CG_ManagerShareSalary')
if path:
    df = read_csmar(path, usecols=['Symbol', 'Enddate', 'StatisticalCaliber',
        'DirectorNumber', 'IndependentDirectorNumber', 'SupervisorNumber', 'ManagerNumber'])
    save_parquet(df, 'manager_salary')
    del df; gc.collect()

t = timer("13. 股权性质")
path = find_file(DATA_DIR, 'EN_EquityNatureAll')
if path:
    df = read_csmar(path)
    save_parquet(df, 'equity_nature')
    del df; gc.collect()

t = timer("14. 十大股东 (多批合并)")
sh_files = find_all_files(DATA_DIR, 'CG_Sharehold')
if sh_files:
    chunks = [read_csmar(f) for f in sh_files]
    df = pd.concat(chunks, ignore_index=True)
    df = df.drop_duplicates()
    save_parquet(df, 'top10_shareholders')
    del df, chunks; gc.collect()

t = timer("15. 股本结构")
path = find_file(DATA_DIR, 'CG_Capchg')
if path:
    df = read_csmar(path)
    save_parquet(df, 'capital_structure')
    del df; gc.collect()

t = timer("16. 机构持股")
path = find_file(DATA_DIR, 'INI_HolderSystematics')
if path:
    df = read_csmar(path, usecols=['Symbol', 'EndDate', 'InsInvestorProp',
        'FundHoldProportion', 'QFIIHoldProportion', 'InsuranceHoldProportion',
        'SecurityFundHoldProportion', 'TotalHoldShares'])
    save_parquet(df, 'inst_holding')
    del df; gc.collect()

# ============ 6. 分析师 ============

t = timer("17. 分析师预测 (多批合并)")
af_files = find_all_files(DATA_DIR, 'AF_Forecast')
if af_files:
    chunks = [read_csmar(f) for f in af_files]
    df = pd.concat(chunks, ignore_index=True)
    df = df.drop_duplicates()
    save_parquet(df, 'analyst_forecast')
    del df, chunks; gc.collect()

t = timer("18. 分析师评级 (多批合并)")
ab_files = find_all_files(DATA_DIR, 'AF_Bench')
if ab_files:
    chunks = [read_csmar(f) for f in ab_files]
    df = pd.concat(chunks, ignore_index=True)
    df = df.drop_duplicates()
    save_parquet(df, 'analyst_rating')
    del df, chunks; gc.collect()

# ============ 7. 审计 ============

t = timer("19. 审计意见")
path = find_file(DATA_DIR, 'FIN_Audit')
if path:
    df = read_csmar(path)
    save_parquet(df, 'audit')
    del df; gc.collect()

# ============ 8. Amihud流动性 ============

t = timer("20. Amihud指标 (多批合并)")
ami_files = find_all_files(DATA_DIR, 'LIQ_AMIHUD')
if ami_files:
    chunks = []
    for f in ami_files:
        print(f"  读取: {os.path.basename(f)}...")
        chunks.append(read_csmar(f))
    df = pd.concat(chunks, ignore_index=True)
    df = df.drop_duplicates(subset=['Stkcd', 'Trddt'], keep='first')
    df['ILLIQ'] = pd.to_numeric(df['ILLIQ'], errors='coerce')
    save_parquet(df, 'amihud_daily')
    del df, chunks; gc.collect()
    print(f"  耗时: {time.time()-t:.1f}s")

# ============ 9. CNRDS数据要素 ============

t = timer("21. CNRDS数据要素指数")
# CNRDS的7z文件如果已解压
cnrds_path = find_file(DATA_DIR, '企业数据要素开发利用指数', ext='.xlsx')
if not cnrds_path:
    # 尝试在7z解压目录找
    for root, dirs, files in os.walk(DATA_DIR):
        for f in files:
            if '企业数据要素开发利用指数' in f and f.endswith('.xlsx'):
                cnrds_path = os.path.join(root, f)
                break
if cnrds_path:
    df = pd.read_excel(cnrds_path)
    # CNRDS不一定有CSMAR那种header格式，直接读
    save_parquet(df, 'cnrds_data_element_index')
    del df; gc.collect()
else:
    print("  ✗ 未找到CNRDS数据要素指数文件 (可能需要先解压7z)")

cnrds_path2 = find_file(DATA_DIR, '企业数据要素开发利用情况', ext='.xlsx')
if cnrds_path2:
    # 可能有多个分片
    files = find_all_files(DATA_DIR, '企业数据要素开发利用情况', ext='.xlsx')
    chunks = [pd.read_excel(f) for f in files]
    df = pd.concat(chunks, ignore_index=True)
    save_parquet(df, 'cnrds_data_element_detail')
    del df; gc.collect()

t = timer("22. 各省份大数据发展指数")
prov_path = find_file(DATA_DIR, '各省份大数据发展指数', ext='.xlsx')
if prov_path:
    df = pd.read_excel(prov_path)
    save_parquet(df, 'province_bigdata_index')
    del df; gc.collect()

# ============ 汇总 ============

print(f"\n{'='*60}")
print(f"全部完成! 输出目录: {OUT_DIR}")
print(f"{'='*60}")
for f in sorted(os.listdir(OUT_DIR)):
    if f.endswith('.parquet'):
        sz = os.path.getsize(os.path.join(OUT_DIR, f)) / 1024 / 1024
        print(f"  {f}: {sz:.1f} MB")

total_sz = sum(os.path.getsize(os.path.join(OUT_DIR, f))
    for f in os.listdir(OUT_DIR) if f.endswith('.parquet')) / 1024 / 1024
print(f"\n总计: {total_sz:.1f} MB")
