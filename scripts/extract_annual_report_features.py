"""
从年报TXT提取数据要素利用特征 (直接从zip读取, 不解压到磁盘)
覆盖2010-2024, 输出firm-year级别特征

两步策略:
  Phase 1: 关键词频率 + 上下文特征 (本脚本, 立即可用)
  Phase 2: LLM语义评分 (后续, 区分实质利用vs概念提及)
"""

import zipfile
import os
import re
import pandas as pd
import numpy as np
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# ================================================================
# 配置
# ================================================================
ZIP_DIR = "/Users/mac/computerscience/第三方资料/第三方数据资源/2001~2024年年报/2001-2024年A股年报TXT格式"
OUT_DIR = "/Users/mac/computerscience/15会计研究/data_parquet"
YEARS = range(2010, 2025)  # 2010-2024

# 数据要素关键词体系 (参考CNRDS + 朱康2025)
KEYWORDS = {
    # 维度一: 数据存量 (data stock)
    'data_stock': [
        '大数据', '数据库', '数据中心', '数据仓库', '数据湖',
        '数据集', '数据存储', '数据采集', '数据资源', '数据积累',
        '用户数据', '客户数据', '行为数据', '交易数据', '运营数据',
    ],
    # 维度二: 数据开发能力 (data development capability)
    'data_dev': [
        '数据挖掘', '数据分析', '数据处理', '数据清洗', '数据建模',
        '机器学习', '深度学习', '人工智能', '算法', '自然语言处理',
        '数据科学', '数据工程', '数据平台', '数据中台', '数据架构',
        '数字化转型', '数字化', '智能化', '信息化',
    ],
    # 维度三: 数据驱动商业应用 (data-driven business application)
    'data_app': [
        '数据驱动', '精准营销', '个性化推荐', '智能推荐', '用户画像',
        '风险控制', '风控模型', '智能决策', '智能客服', '智能制造',
        '预测模型', '需求预测', '供应链优化', '智慧物流', '智慧城市',
        '数字营销', '程序化', '数据赋能', '数据服务',
    ],
    # 维度四: 数据价值变现 (data value monetization)
    'data_value': [
        '数据资产', '数据要素', '数据交易', '数据产品', '数据变现',
        '数据确权', '数据定价', '数据流通', '数据市场',
        '数据入表', '数据资源入表',
    ],
    # 维度五: 数据治理 (data governance)
    'data_gov': [
        '数据治理', '数据安全', '数据隐私', '数据合规', '数据质量',
        '数据标准', '数据脱敏', '个人信息保护', '数据分类分级',
    ],
}

# 所有关键词的flat list
ALL_KEYWORDS = []
for v in KEYWORDS.values():
    ALL_KEYWORDS.extend(v)
ALL_KEYWORDS = sorted(set(ALL_KEYWORDS), key=len, reverse=True)  # 长词优先匹配

# 章节定位正则
SECTION_PATTERNS = {
    'mda': re.compile(r'(管理层讨论与分析|经营情况讨论与分析|董事会报告)'),
    'business': re.compile(r'(公司业务概要|主营业务分析|业务概述)'),
    'risk': re.compile(r'(风险因素|面临的风险|风险管理)'),
}


# ================================================================
# 核心函数
# ================================================================
def decode_zipname(raw_name):
    """解码zip内文件名 (GBK编码)"""
    try:
        return raw_name.encode('cp437').decode('gbk')
    except (UnicodeDecodeError, UnicodeEncodeError):
        return raw_name


def extract_sections(text):
    """提取关键章节文本"""
    sections = {}
    # 找到所有章节标题的位置
    headers = list(re.finditer(
        r'^第[一二三四五六七八九十]+[节章]\s*.+',
        text, re.MULTILINE
    ))

    for i, h in enumerate(headers):
        title = h.group().strip()
        start = h.end()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        content = text[start:end]

        for sec_name, pat in SECTION_PATTERNS.items():
            if pat.search(title):
                if sec_name not in sections:
                    sections[sec_name] = content
                else:
                    sections[sec_name] += '\n' + content

    return sections


def count_keywords_in_text(text, keywords_dict):
    """统计各维度关键词频率"""
    counts = {}
    total = 0
    for dim, kws in keywords_dict.items():
        dim_count = 0
        for kw in kws:
            c = text.count(kw)
            dim_count += c
        counts[dim] = dim_count
        total += dim_count
    counts['total'] = total
    return counts


def extract_keyword_contexts(text, window=150):
    """提取关键词周围的上下文 (用于后续LLM评分)"""
    contexts = []
    for kw in ALL_KEYWORDS:
        idx = 0
        while True:
            idx = text.find(kw, idx)
            if idx == -1:
                break
            start = max(0, idx - window)
            end = min(len(text), idx + len(kw) + window)
            contexts.append(text[start:end])
            idx += len(kw)
    return contexts


def compute_substantive_ratio(text):
    """
    计算"实质利用"vs"概念提及"的近似比例
    实质利用: 关键词出现在具体业务描述中 (含数字/百分比/具体产品名)
    概念提及: 关键词出现在泛泛的趋势描述中
    """
    substantive = 0
    mention_only = 0

    for kw in ALL_KEYWORDS:
        idx = 0
        while True:
            idx = text.find(kw, idx)
            if idx == -1:
                break
            # 取前后100字的窗口
            window = text[max(0, idx - 100):min(len(text), idx + len(kw) + 100)]

            # 实质利用的信号: 包含数字、百分比、具体动作词
            has_numbers = bool(re.search(r'\d+\.?\d*[%万亿元]', window))
            has_action = bool(re.search(
                r'(实现|完成|建设|搭建|部署|应用|推出|上线|落地|开发|构建|运营|服务|处理|分析)',
                window))
            has_specific = bool(re.search(
                r'(平台|系统|模型|工具|产品|方案|项目|业务|客户|用户)',
                window))

            if has_numbers or (has_action and has_specific):
                substantive += 1
            else:
                mention_only += 1
            idx += len(kw)

    total = substantive + mention_only
    if total == 0:
        return 0.0, 0, 0
    return substantive / total, substantive, mention_only


def process_single_report(args):
    """处理单个年报文件, 返回特征字典"""
    zip_path, raw_name = args
    decoded = decode_zipname(raw_name)

    # 从文件名提取信息: {Stkcd}_{year}_{name}_{title}_{date}.txt
    parts = decoded.split('_')
    if len(parts) < 3:
        return None
    stkcd = parts[0]
    year = parts[1]

    try:
        stkcd_int = int(stkcd)
        year_int = int(year)
    except ValueError:
        return None

    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            data = zf.read(raw_name)
    except Exception:
        return None

    # 尝试多种编码
    text = None
    for enc in ['utf-8', 'gbk', 'gb18030', 'gb2312']:
        try:
            text = data.decode(enc)
            break
        except (UnicodeDecodeError, LookupError):
            continue
    if text is None:
        text = data.decode('utf-8', errors='replace')

    total_chars = len(text)
    total_words = len(text.split())

    # 提取章节
    sections = extract_sections(text)
    mda_text = sections.get('mda', '')
    biz_text = sections.get('business', '')

    # 全文关键词统计
    full_counts = count_keywords_in_text(text, KEYWORDS)

    # MD&A章节关键词统计
    mda_counts = count_keywords_in_text(mda_text, KEYWORDS) if mda_text else {k: 0 for k in list(KEYWORDS.keys()) + ['total']}

    # 实质利用比例 (基于MD&A或全文)
    analysis_text = mda_text if mda_text else text
    sub_ratio, sub_count, mention_count = compute_substantive_ratio(analysis_text)

    result = {
        'Stkcd': stkcd_int,
        'year': year_int,
        'total_chars': total_chars,
        # 全文关键词
        'kw_total': full_counts['total'],
        'kw_data_stock': full_counts['data_stock'],
        'kw_data_dev': full_counts['data_dev'],
        'kw_data_app': full_counts['data_app'],
        'kw_data_value': full_counts['data_value'],
        'kw_data_gov': full_counts['data_gov'],
        # 归一化 (每万字)
        'kw_per10k': full_counts['total'] / max(total_chars, 1) * 10000,
        # MD&A关键词
        'mda_kw_total': mda_counts['total'],
        'mda_kw_per10k': mda_counts['total'] / max(len(mda_text), 1) * 10000 if mda_text else 0,
        # 实质利用
        'substantive_ratio': sub_ratio,
        'substantive_count': sub_count,
        'mention_count': mention_count,
        # MD&A长度
        'mda_chars': len(mda_text),
        'has_mda': 1 if mda_text else 0,
    }
    return result


# ================================================================
# 主流程
# ================================================================
def main():
    print("=" * 60)
    print("年报数据要素特征提取 (2010-2024)")
    print("=" * 60)

    # 收集所有zip文件和内部文件
    tasks = []
    for year in YEARS:
        # 找到对应年份的zip
        zips = [f for f in os.listdir(ZIP_DIR) if f.startswith(f'{year}_') and f.endswith('.zip')]
        if not zips:
            print(f"  {year}: 无zip文件")
            continue
        zip_path = os.path.join(ZIP_DIR, zips[0])
        with zipfile.ZipFile(zip_path, 'r') as zf:
            names = zf.namelist()
        n_files = len(names)
        print(f"  {year}: {n_files} files in {zips[0]}")
        for name in names:
            tasks.append((zip_path, name))

    print(f"\n总计: {len(tasks)} 个年报文件")
    print("开始提取...")

    # 串行处理 (zip内文件不适合多进程打开同一zip)
    # 按zip分组处理
    results = []
    from collections import defaultdict
    zip_tasks = defaultdict(list)
    for zip_path, name in tasks:
        zip_tasks[zip_path].append(name)

    processed = 0
    for zip_path, names in zip_tasks.items():
        year_label = os.path.basename(zip_path).split('_')[0]
        batch_results = []

        for name in names:
            r = process_single_report((zip_path, name))
            if r is not None:
                batch_results.append(r)
            processed += 1

        results.extend(batch_results)
        print(f"  {year_label}: {len(batch_results)} reports extracted ({processed}/{len(tasks)})")

    # 汇总
    df = pd.DataFrame(results)
    print(f"\n提取完成: {len(df)} obs")
    print(f"年份覆盖: {sorted(df.year.unique())}")
    print(f"企业数: {df.Stkcd.nunique()}")

    # 去重: 同一firm-year保留关键词最多的版本
    df = df.sort_values('kw_total', ascending=False).drop_duplicates(
        subset=['Stkcd', 'year'], keep='first')
    print(f"去重后: {len(df)} obs")

    # 描述性统计
    print("\n--- 关键词统计 ---")
    for col in ['kw_total', 'kw_per10k', 'mda_kw_total', 'substantive_ratio']:
        print(f"  {col}: mean={df[col].mean():.2f}, median={df[col].median():.2f}, "
              f"std={df[col].std():.2f}, min={df[col].min():.0f}, max={df[col].max():.0f}")

    # 按年份统计
    print("\n--- 按年份统计 ---")
    by_year = df.groupby('year').agg(
        n=('Stkcd', 'count'),
        kw_mean=('kw_total', 'mean'),
        kw_median=('kw_total', 'median'),
        sub_ratio_mean=('substantive_ratio', 'mean'),
    ).round(2)
    print(by_year.to_string())

    # 保存
    outpath = os.path.join(OUT_DIR, 'annual_report_features.parquet')
    df.to_parquet(outpath, index=False)
    print(f"\n已保存: {outpath}")

    # 同时保存CSV方便查看
    df.to_csv(os.path.join(OUT_DIR, 'annual_report_features.csv'), index=False)
    return df


if __name__ == '__main__':
    main()
