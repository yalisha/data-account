"""
从年报TXT中构造数据要素利用指标 (2010-2024)

Stage 1: 关键词词频统计 (对标CNRDS方法论, 扩展覆盖至2010-2024)
  - 使用CNRDS同一套56个关键词
  - 统计总词频 (TWFre) 和总年报词数 (Term_Only)
  - 计算归一化指标 DataUsage = TWFre / Term_Only * 10000

Stage 2 (TODO): LLM语义评分, 区分实质利用vs概念提及

数据来源: 年报TXT格式zip压缩包 (2010-2024, 共~53,000份)
"""

import zipfile, os, re, time
import pandas as pd
import numpy as np
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# ============ 配置 ============
BASE_DIR = "/Users/mac/computerscience/第三方资料/第三方数据资源/2001~2024年年报/2001-2024年A股年报TXT格式"
OUT_DIR = "/Users/mac/computerscience/15会计研究/data_parquet"
YEARS = range(2010, 2025)

# CNRDS 56个数据要素关键词
KEYWORDS = [
    '3D工具', '3D打印', '3D技术', '5G', 'AI', 'B2B', 'B2C', 'C2B', 'C2C',
    'O2O', 'P2P', '云服务', '云端', '云计算', '互联网', '信息化', '信息技术',
    '信息时代', '信息通信', '信息集成', '区块链', '大数据', '数字体系', '数字供应链',
    '数字化', '数字技术', '数字科技', '数字终端', '数字经济', '数字营销', '数字货币',
    '数字贸易', '数字运营', '数据信息', '数据管理', '数据融合', '数据资产', '数据集成',
    '智慧业务', '智慧建设', '智慧时代', '智能', '机器人', '机器学习', '物联网',
    '电商平台', '电子商务', '电子技术', '电子科技', '线上', '线上线下', '网络',
    '自动化', '计算机技术', '跨境电商', '边缘计算'
]

# 按长度降序排列 (优先匹配长关键词, 避免"线上"匹配到"线上线下")
KEYWORDS_SORTED = sorted(KEYWORDS, key=len, reverse=True)


def count_keywords(text):
    """统计关键词频率, 返回 {keyword: count} 和总词数"""
    counts = {}
    for kw in KEYWORDS_SORTED:
        c = text.count(kw)
        if c > 0:
            counts[kw] = c
    total_words = len(text)  # 字符数近似词数 (中文)
    return counts, total_words


def extract_mda(text):
    """提取管理层讨论与分析 (MD&A) 部分"""
    # 尝试匹配 "第三节 管理层讨论与分析" 或 "第四节 管理层讨论与分析"
    patterns = [
        r'(第[三四]节\s*管理层讨论与分析)',
        r'(第[三四]节\s*管理层讨论与分析.*?(?=第[四五六七八]节))',
    ]
    # 简单方法: 找到MD&A起始位置和下一节的起始位置
    mda_start = None
    for pat in [r'第[三四]节\s*管理层讨论与分析']:
        m = re.search(pat, text)
        if m:
            mda_start = m.start()
            break

    if mda_start is None:
        return text  # 找不到就用全文

    # 找下一节
    next_section = re.search(r'第[四五六七八九十]+节\s', text[mda_start + 20:])
    if next_section:
        mda_end = mda_start + 20 + next_section.start()
    else:
        mda_end = len(text)

    return text[mda_start:mda_end]


def process_one_file(args):
    """处理单个年报文件"""
    zip_path, entry_name, year = args
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            raw = zf.read(entry_name)
            # 尝试多种编码
            for enc in ['utf-8', 'gbk', 'gb2312', 'gb18030']:
                try:
                    text = raw.decode(enc)
                    break
                except (UnicodeDecodeError, LookupError):
                    continue
            else:
                text = raw.decode('utf-8', errors='replace')

        # 解码文件名获取股票代码
        try:
            fname = entry_name.encode('cp437').decode('gbk')
        except (UnicodeDecodeError, UnicodeEncodeError):
            fname = entry_name
        stkcd = fname[:6]

        # 提取MD&A
        mda = extract_mda(text)

        # 统计关键词
        kw_counts, total_words = count_keywords(text)  # 全文统计
        mda_kw_counts, mda_words = count_keywords(mda)  # MD&A统计

        tw_fre = sum(kw_counts.values())
        mda_tw_fre = sum(mda_kw_counts.values())

        return {
            'Stkcd': stkcd,
            'year': year,
            'TWFre_count': tw_fre,
            'Term_Only': total_words,
            'MDA_TWFre': mda_tw_fre,
            'MDA_Words': mda_words,
            'n_keywords_hit': len(kw_counts),
            'top_keywords': ', '.join(f'{k}:{v}' for k, v in
                                      sorted(kw_counts.items(), key=lambda x: -x[1])[:5]),
        }
    except Exception as e:
        return {'Stkcd': 'ERROR', 'year': year, 'error': str(e)}


def process_year(year):
    """处理一年的年报"""
    # 找到对应的zip文件
    zip_files = [f for f in os.listdir(BASE_DIR) if f.startswith(str(year)) and f.endswith('.zip')]
    if not zip_files:
        print(f"  {year}: 未找到zip文件")
        return []

    zip_path = os.path.join(BASE_DIR, zip_files[0])
    results = []

    with zipfile.ZipFile(zip_path, 'r') as zf:
        entries = zf.namelist()
        print(f"  {year}: {len(entries)} 份年报 ({zip_files[0]})")

        for i, entry in enumerate(entries):
            result = process_one_file((zip_path, entry, year))
            results.append(result)
            if (i + 1) % 500 == 0:
                print(f"    {year}: {i+1}/{len(entries)}")

    return results


# ============ 主程序 ============
if __name__ == '__main__':
    print("=" * 60)
    print("年报数据要素关键词统计 (2010-2024)")
    print("=" * 60)
    t0 = time.time()

    all_results = []
    for year in YEARS:
        t1 = time.time()
        results = process_year(year)
        all_results.extend(results)
        n_valid = sum(1 for r in results if r.get('Stkcd', 'ERROR') != 'ERROR')
        print(f"    -> {n_valid} 有效, 耗时 {time.time()-t1:.0f}s")

    # 转为DataFrame
    df = pd.DataFrame(all_results)
    df = df[df['Stkcd'] != 'ERROR'].copy()

    # 股票代码标准化为整数
    df['Stkcd'] = pd.to_numeric(df['Stkcd'], errors='coerce')
    df = df.dropna(subset=['Stkcd'])
    df['Stkcd'] = df['Stkcd'].astype(int)

    # 计算归一化指标
    df['DataUsage_KW'] = df['TWFre_count'] / df['Term_Only'] * 10000
    df['DataUsage_MDA'] = df['MDA_TWFre'] / df['MDA_Words'] * 10000

    # 去重 (同一企业同一年可能有多份报告, 取最后一份)
    df = df.sort_values(['Stkcd', 'year']).drop_duplicates(
        subset=['Stkcd', 'year'], keep='last')

    print(f"\n{'='*60}")
    print(f"结果汇总")
    print(f"{'='*60}")
    print(f"  总观测: {len(df):,}")
    print(f"  企业数: {df.Stkcd.nunique():,}")
    print(f"  年份: {df.year.min()} ~ {df.year.max()}")
    print(f"\n  DataUsage_KW 描述统计:")
    print(df['DataUsage_KW'].describe().to_string())
    print(f"\n  各年份均值:")
    print(df.groupby('year')['DataUsage_KW'].agg(['count', 'mean', 'median']).to_string())
    print(f"\n  总耗时: {time.time()-t0:.0f}s")

    # 保存
    outpath = f"{OUT_DIR}/annual_report_kw_scores.parquet"
    df.to_parquet(outpath, index=False)
    print(f"\n  已保存: {outpath}")

    # 与CNRDS对比验证 (2018-2020)
    print(f"\n{'='*60}")
    print(f"CNRDS对比验证 (2018-2020)")
    print(f"{'='*60}")
    cnrds = pd.read_parquet(f"{OUT_DIR}/cnrds_data_element_index.parquet")
    cnrds = cnrds[cnrds['Year'] != '会计年度'].copy()
    cnrds['Stkcd'] = pd.to_numeric(cnrds['Scode'], errors='coerce').astype('Int64')
    cnrds['year'] = cnrds['Year'].astype(int)
    cnrds['TWFre_cnrds'] = pd.to_numeric(cnrds['TWFre_count'], errors='coerce')
    cnrds['Term_cnrds'] = pd.to_numeric(cnrds['Term_Only'], errors='coerce')

    compare = df[df.year.isin([2018, 2019, 2020])].merge(
        cnrds[['Stkcd', 'year', 'TWFre_cnrds', 'Term_cnrds']],
        on=['Stkcd', 'year'], how='inner'
    )
    if len(compare) > 0:
        corr_tw = compare['TWFre_count'].corr(compare['TWFre_cnrds'])
        corr_term = compare['Term_Only'].corr(compare['Term_cnrds'])
        print(f"  匹配观测: {len(compare):,}")
        print(f"  TWFre相关系数: {corr_tw:.4f}")
        print(f"  Term_Only相关系数: {corr_term:.4f}")
    else:
        print("  无匹配观测")
