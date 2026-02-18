"""
修复多批次数据覆盖问题
从独立子目录读取各批次xlsx，合并去重后保存parquet
"""
import pandas as pd
import os, gc, time, warnings
warnings.filterwarnings('ignore')

DATA_DIR = "/Users/mac/computerscience/第三方资料/第三方数据资源/上市公司财务信息"
OUT_DIR = "/Users/mac/computerscience/15会计研究/data_parquet"


def read_csmar(path, usecols=None):
    return pd.read_excel(path, header=0, skiprows=[1, 2], usecols=usecols)


def find_xlsx(base_dir):
    results = []
    for root, dirs, files in os.walk(base_dir):
        for f in files:
            if f.endswith('.xlsx'):
                results.append(os.path.join(root, f))
    return sorted(results)


def save_parquet(df, name):
    outpath = os.path.join(OUT_DIR, f"{name}.parquet")
    df.to_parquet(outpath, index=False)
    sz = os.path.getsize(outpath) / 1024 / 1024
    print(f"  -> {name}: {len(df):,} rows x {len(df.columns)} cols = {sz:.1f} MB")


# ============ 1. 日个股回报率 ============
print("\n=== 日个股回报率 (3 batches) ===")
t = time.time()
daily_cols = ['Stkcd', 'Trddt', 'Clsprc', 'Dnvaltrd', 'Dsmvosd', 'Dsmvtll',
              'Dretwd', 'Dretnd', 'Markettype']
batches = ['_daily_2011_2016', '_daily_2016_2021', '_daily_2021_2026']
chunks = []
for batch in batches:
    batch_dir = os.path.join(DATA_DIR, batch)
    files = find_xlsx(batch_dir)
    for f in files:
        print(f"  读取: {batch}/{os.path.basename(f)}")
        try:
            df_c = read_csmar(f, usecols=daily_cols)
        except Exception:
            df_c = read_csmar(f)
            avail = [c for c in daily_cols if c in df_c.columns]
            df_c = df_c[avail]
        chunks.append(df_c)

df = pd.concat(chunks, ignore_index=True)
for col in ['Dretwd', 'Dretnd', 'Dnvaltrd', 'Dsmvosd', 'Dsmvtll', 'Clsprc']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
df = df.drop_duplicates(subset=['Stkcd', 'Trddt'], keep='first')
df = df.sort_values(['Stkcd', 'Trddt']).reset_index(drop=True)
save_parquet(df, 'daily_return')
print(f"  日期范围: {df.Trddt.min()} ~ {df.Trddt.max()}")
print(f"  耗时: {time.time()-t:.0f}s")
del df, chunks; gc.collect()

# ============ 2. Amihud ============
print("\n=== Amihud (3 batches) ===")
t = time.time()
batches = ['_amihud_batch1', '_amihud_batch2', '_amihud_batch3']
chunks = []
for batch in batches:
    batch_dir = os.path.join(DATA_DIR, batch)
    files = find_xlsx(batch_dir)
    for f in files:
        print(f"  读取: {batch}/{os.path.basename(f)}")
        chunks.append(read_csmar(f))

df = pd.concat(chunks, ignore_index=True)
df['ILLIQ'] = pd.to_numeric(df['ILLIQ'], errors='coerce')
df = df.drop_duplicates(subset=['Stkcd', 'Trddt'], keep='first')
df = df.sort_values(['Stkcd', 'Trddt']).reset_index(drop=True)
save_parquet(df, 'amihud_daily')
print(f"  日期范围: {df.Trddt.min()} ~ {df.Trddt.max()}")
print(f"  耗时: {time.time()-t:.0f}s")
del df, chunks; gc.collect()

# ============ 3. 分析师预测 ============
print("\n=== 分析师预测 (2 batches) ===")
t = time.time()
batches = ['_forecast_batch1', '_forecast_batch2']
chunks = []
for batch in batches:
    batch_dir = os.path.join(DATA_DIR, batch)
    files = find_xlsx(batch_dir)
    for f in files:
        print(f"  读取: {batch}/{os.path.basename(f)}")
        chunks.append(read_csmar(f))

df = pd.concat(chunks, ignore_index=True)
df = df.drop_duplicates()
save_parquet(df, 'analyst_forecast')
print(f"  耗时: {time.time()-t:.0f}s")
del df, chunks; gc.collect()

# ============ 4. 分析师评级 ============
print("\n=== 分析师评级 (2 batches) ===")
t = time.time()
batches = ['_rating_batch1', '_rating_batch2']
chunks = []
for batch in batches:
    batch_dir = os.path.join(DATA_DIR, batch)
    files = find_xlsx(batch_dir)
    for f in files:
        print(f"  读取: {batch}/{os.path.basename(f)}")
        chunks.append(read_csmar(f))

df = pd.concat(chunks, ignore_index=True)
df = df.drop_duplicates()
save_parquet(df, 'analyst_rating')
print(f"  耗时: {time.time()-t:.0f}s")
del df, chunks; gc.collect()

# ============ 5. 行业指数日行情 ============
print("\n=== 行业指数日行情 (4 batches) ===")
t = time.time()
batches = ['_idx_batch1', '_idx_batch2', '_idx_batch3', '_idx_batch4']
chunks = []
for batch in batches:
    batch_dir = os.path.join(DATA_DIR, batch)
    files = find_xlsx(batch_dir)
    for f in files:
        print(f"  读取: {batch}/{os.path.basename(f)}")
        chunks.append(read_csmar(f))

df = pd.concat(chunks, ignore_index=True)
df = df.drop_duplicates(subset=['Indexcd', 'Idxtrd01'], keep='first')
df = df.sort_values(['Indexcd', 'Idxtrd01']).reset_index(drop=True)
save_parquet(df, 'industry_index_daily')
print(f"  日期范围: {df.Idxtrd01.min()} ~ {df.Idxtrd01.max()}")
print(f"  耗时: {time.time()-t:.0f}s")
del df, chunks; gc.collect()

print("\n=== 修复完成 ===")
