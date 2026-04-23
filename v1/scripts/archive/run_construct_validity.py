"""
构念效度检验：DU_kw 测到的是数据要素利用，还是叙事风格/热词？

Part 1: Horse-race 区分性检验（控制年报长度）
Part 2: Placebo 安慰剂 + DU_kw_strict（去泛化词）
Part 3: CNRDS 外部收敛效度
Part 4: DataAsset 对照验证
"""

import pandas as pd
import numpy as np
import pyfixest as pf
import os, time, json, warnings, zipfile, re
from scipy import stats
from concurrent.futures import ProcessPoolExecutor, as_completed
warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = f"{BASE}/data_parquet"
ZIP_DIR = "/Users/mac/computerscience/15会计研究/第三方资料/第三方数据资源/2001~2024年年报/2001-2024年A股年报TXT格式"
YEARS = range(2010, 2025)

results = {}

# ============================================================
# 准备面板数据（复用 generate_tables 的构造逻辑）
# ============================================================
print("=" * 60)
print("加载面板数据...")
print("=" * 60)

panel = pd.read_parquet(f"{OUT_DIR}/panel.parquet")
ar_feat = pd.read_parquet(f"{OUT_DIR}/annual_report_features.parquet")

# 合并年报特征
panel = panel.merge(
    ar_feat[['Stkcd','year','kw_total','kw_per10k','substantive_count',
             'total_chars','mda_chars',
             'kw_data_stock','kw_data_dev','kw_data_app','kw_data_value','kw_data_gov']],
    on=['Stkcd','year'], how='left'
)

# 行业信息
fi = pd.read_parquet(f"{OUT_DIR}/firm_info.parquet",
                     columns=['Symbol','EndDate','IndustryCodeC','LISTINGSTATE'])
fi = fi.rename(columns={'Symbol':'Stkcd'})
fi['EndDate'] = pd.to_datetime(fi['EndDate'])
fi['year'] = fi['EndDate'].dt.year
fi = fi.sort_values(['Stkcd','year','EndDate']).drop_duplicates(subset=['Stkcd','year'], keep='last')
panel = panel.merge(fi[['Stkcd','year','IndustryCodeC','LISTINGSTATE']], on=['Stkcd','year'], how='left')
panel = panel.sort_values(['Stkcd', 'year'])
panel[['IndustryCodeC', 'LISTINGSTATE']] = panel.groupby('Stkcd')[['IndustryCodeC', 'LISTINGSTATE']].transform(
    lambda x: x.ffill().bfill()
)

mask_fin = panel['IndustryCodeC'].str.startswith('J', na=False)
mask_st = panel['LISTINGSTATE'].isin(['ST','*ST'])
mask_new = panel['Age'] <= 0
panel = panel[~mask_fin & ~mask_st & ~mask_new]

panel['Ind2'] = panel['IndustryCodeC'].str[:3]
panel['IndYear'] = panel['Ind2'].astype(str) + '_' + panel['year'].astype(str)
panel['DU_kw'] = panel['kw_per10k']

def winsorize(s):
    lo, hi = s.quantile([0.01, 0.99])
    return s.clip(lo, hi)

controls = ['Size','Lev','ROA','TobinQ','Age','Growth','BoardSize','IndepRatio',
            'Dual','Top1Share','SOE','InstHold','Amihud','Analyst','AuditType']
cont_vars = ['DU_kw','Lev','ROA','Growth','Size','TobinQ','Age','BoardSize','IndepRatio',
             'Top1Share','InstHold','Amihud','Analyst']
for v in cont_vars:
    if v in panel.columns and panel[v].notna().any():
        panel[v] = winsorize(panel[v])

# 构造 ln 变量
panel['ln_total_chars'] = np.log1p(panel['total_chars'])
panel['ln_mda_chars'] = np.log1p(panel['mda_chars'])
panel['Stkcd_str'] = panel['Stkcd'].astype(str)
panel['year_str'] = panel['year'].astype(str)
ctrl_str = " + ".join(controls)

print(f"  面板: {len(panel):,} obs, {panel.Stkcd.nunique():,} firms")

# ============================================================
# Part 1: Horse-race 区分性检验
# ============================================================
print("\n" + "=" * 60)
print("Part 1: Horse-race 区分性检验")
print("=" * 60)

# 基准样本
reg_base = panel.dropna(subset=['PriceDelay','DU_kw'] + controls + ['IndYear']).copy()
print(f"  基准样本: N={len(reg_base):,}")

# 模型 1: DU_kw + ln_total_chars
reg1 = reg_base.dropna(subset=['ln_total_chars']).copy()
reg1['ln_total_chars'] = winsorize(reg1['ln_total_chars'])
m1 = pf.feols(f"PriceDelay ~ DU_kw + ln_total_chars + {ctrl_str} | Stkcd_str + year_str",
              data=reg1, vcov={"CRV1": "IndYear"})
print(f"\n  模型1: DU_kw + ln(总字数)")
print(f"    DU_kw:          coef={m1.coef()['DU_kw']:.6f}, t={m1.tstat()['DU_kw']:.3f}, p={m1.pvalue()['DU_kw']:.4f}")
print(f"    ln_total_chars: coef={m1.coef()['ln_total_chars']:.6f}, t={m1.tstat()['ln_total_chars']:.3f}")
print(f"    N={m1._N:,}")

results['horserace_total_chars'] = {
    'DU_kw_coef': round(m1.coef()['DU_kw'], 6),
    'DU_kw_t': round(m1.tstat()['DU_kw'], 3),
    'DU_kw_p': round(m1.pvalue()['DU_kw'], 4),
    'ln_total_chars_coef': round(m1.coef()['ln_total_chars'], 6),
    'ln_total_chars_t': round(m1.tstat()['ln_total_chars'], 3),
    'N': int(m1._N),
}

# 模型 2: DU_kw + ln_mda_chars
reg2 = reg_base.dropna(subset=['ln_mda_chars']).copy()
reg2 = reg2[reg2['mda_chars'] > 0]  # 排除无MD&A的
reg2['ln_mda_chars'] = winsorize(reg2['ln_mda_chars'])
m2 = pf.feols(f"PriceDelay ~ DU_kw + ln_mda_chars + {ctrl_str} | Stkcd_str + year_str",
              data=reg2, vcov={"CRV1": "IndYear"})
print(f"\n  模型2: DU_kw + ln(MD&A字数)")
print(f"    DU_kw:        coef={m2.coef()['DU_kw']:.6f}, t={m2.tstat()['DU_kw']:.3f}, p={m2.pvalue()['DU_kw']:.4f}")
print(f"    ln_mda_chars: coef={m2.coef()['ln_mda_chars']:.6f}, t={m2.tstat()['ln_mda_chars']:.3f}")
print(f"    N={m2._N:,}")

results['horserace_mda_chars'] = {
    'DU_kw_coef': round(m2.coef()['DU_kw'], 6),
    'DU_kw_t': round(m2.tstat()['DU_kw'], 3),
    'DU_kw_p': round(m2.pvalue()['DU_kw'], 4),
    'ln_mda_chars_coef': round(m2.coef()['ln_mda_chars'], 6),
    'ln_mda_chars_t': round(m2.tstat()['ln_mda_chars'], 3),
    'N': int(m2._N),
}

# 模型 3: DU_kw + 两个字数变量
reg3 = reg_base.dropna(subset=['ln_total_chars', 'ln_mda_chars']).copy()
reg3 = reg3[reg3['mda_chars'] > 0]
reg3['ln_total_chars'] = winsorize(reg3['ln_total_chars'])
reg3['ln_mda_chars'] = winsorize(reg3['ln_mda_chars'])
m3 = pf.feols(f"PriceDelay ~ DU_kw + ln_total_chars + ln_mda_chars + {ctrl_str} | Stkcd_str + year_str",
              data=reg3, vcov={"CRV1": "IndYear"})
print(f"\n  模型3: DU_kw + ln(总字数) + ln(MD&A字数)")
print(f"    DU_kw:          coef={m3.coef()['DU_kw']:.6f}, t={m3.tstat()['DU_kw']:.3f}, p={m3.pvalue()['DU_kw']:.4f}")
print(f"    ln_total_chars: coef={m3.coef()['ln_total_chars']:.6f}, t={m3.tstat()['ln_total_chars']:.3f}")
print(f"    ln_mda_chars:   coef={m3.coef()['ln_mda_chars']:.6f}, t={m3.tstat()['ln_mda_chars']:.3f}")
print(f"    N={m3._N:,}")

results['horserace_both'] = {
    'DU_kw_coef': round(m3.coef()['DU_kw'], 6),
    'DU_kw_t': round(m3.tstat()['DU_kw'], 3),
    'DU_kw_p': round(m3.pvalue()['DU_kw'], 4),
    'N': int(m3._N),
}

# ============================================================
# Part 2: Placebo + DU_kw_strict（扫描原文）
# ============================================================
print("\n" + "=" * 60)
print("Part 2: Placebo 安慰剂 + DU_kw_strict")
print("=" * 60)

PLACEBO_KW = sorted([
    '高质量发展', '创新驱动', '绿色低碳', '低碳', '协同发展',
    '供给侧改革', '供给侧结构性改革', '新质生产力', '产业升级', '转型升级',
    '提质增效', '降本增效', '可持续发展', '碳中和', '碳达峰',
    '乡村振兴', '一带一路', '双循环', '新发展格局', '高水平开放',
], key=len, reverse=True)

BUZZWORDS = sorted([
    '数字化转型', '数字化', '智能化', '信息化', '人工智能', '大数据', '智能制造', '智慧城市'
], key=len, reverse=True)

def decode_zipname(raw_name):
    try:
        return raw_name.encode('cp437').decode('gbk')
    except (UnicodeDecodeError, UnicodeEncodeError):
        return raw_name

def count_keywords_simple(text, keywords):
    total = 0
    for kw in keywords:
        total += text.count(kw)
    return total

def process_report_placebo(args):
    zip_path, raw_name = args
    decoded = decode_zipname(raw_name)
    parts = decoded.split('_')
    if len(parts) < 3:
        return None
    try:
        stkcd_int = int(parts[0])
        year_int = int(parts[1])
    except ValueError:
        return None

    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            data = zf.read(raw_name)
    except Exception:
        return None

    text = None
    for enc in ['utf-8', 'gbk', 'gb18030', 'gb2312']:
        try:
            text = data.decode(enc)
            break
        except (UnicodeDecodeError, LookupError):
            continue
    if text is None:
        text = data.decode('utf-8', errors='replace')

    placebo_count = count_keywords_simple(text, PLACEBO_KW)
    buzzword_count = count_keywords_simple(text, BUZZWORDS)
    total_chars = len(text)

    return {
        'Stkcd': stkcd_int,
        'year': year_int,
        'placebo_count': placebo_count,
        'buzzword_count': buzzword_count,
        'total_chars_check': total_chars,
    }

# 收集所有待处理的 (zip_path, raw_name) 对
tasks = []
for yr in YEARS:
    # 查找对应年份的zip文件
    for fn in os.listdir(ZIP_DIR):
        if fn.startswith(str(yr)) and fn.endswith('.zip'):
            zip_path = os.path.join(ZIP_DIR, fn)
            try:
                with zipfile.ZipFile(zip_path, 'r') as zf:
                    for name in zf.namelist():
                        if name.endswith('.txt'):
                            tasks.append((zip_path, name))
            except Exception as e:
                print(f"  Warning: cannot read {fn}: {e}")

print(f"  待处理年报: {len(tasks):,} 份")
t0 = time.time()

placebo_results = []
for i, t in enumerate(tasks):
    result = process_report_placebo(t)
    if result is not None:
        placebo_results.append(result)
    if (i + 1) % 10000 == 0:
        print(f"  进度: {i+1}/{len(tasks)} ({(i+1)/len(tasks)*100:.1f}%)")

placebo_df = pd.DataFrame(placebo_results)
# 去重（同一企业同一年可能有多份年报，取最长的）
placebo_df = placebo_df.sort_values('total_chars_check', ascending=False).drop_duplicates(
    subset=['Stkcd', 'year'], keep='first'
)
print(f"  完成: {len(placebo_df):,} 份，耗时 {time.time()-t0:.1f}s")

# 保存
placebo_df.to_parquet(f"{OUT_DIR}/placebo_features.parquet", index=False)
print(f"  已保存: placebo_features.parquet")

# 合并到面板
panel = panel.merge(placebo_df[['Stkcd','year','placebo_count','buzzword_count']],
                    on=['Stkcd','year'], how='left')

# 构造变量
panel['Placebo_kw'] = panel['placebo_count'] / panel['total_chars'].clip(lower=1) * 10000
panel['DU_kw_strict'] = (panel['kw_total'] - panel['buzzword_count'].fillna(0)) / panel['total_chars'].clip(lower=1) * 10000
panel['DU_kw_strict'] = panel['DU_kw_strict'].clip(lower=0)

# Winsorize
for v in ['Placebo_kw', 'DU_kw_strict']:
    if panel[v].notna().any():
        panel[v] = winsorize(panel[v])

# Placebo 回归
reg_p = panel.dropna(subset=['PriceDelay','DU_kw','Placebo_kw'] + controls + ['IndYear']).copy()
print(f"\n  Placebo 回归样本: N={len(reg_p):,}")

# 模型 P1: 仅 Placebo
mp1 = pf.feols(f"PriceDelay ~ Placebo_kw + {ctrl_str} | Stkcd_str + year_str",
               data=reg_p, vcov={"CRV1": "IndYear"})
print(f"\n  模型P1: 仅 Placebo_kw")
print(f"    Placebo_kw: coef={mp1.coef()['Placebo_kw']:.6f}, t={mp1.tstat()['Placebo_kw']:.3f}, p={mp1.pvalue()['Placebo_kw']:.4f}")

results['placebo_only'] = {
    'Placebo_kw_coef': round(mp1.coef()['Placebo_kw'], 6),
    'Placebo_kw_t': round(mp1.tstat()['Placebo_kw'], 3),
    'Placebo_kw_p': round(mp1.pvalue()['Placebo_kw'], 4),
    'N': int(mp1._N),
}

# 模型 P2: DU_kw + Placebo_kw
mp2 = pf.feols(f"PriceDelay ~ DU_kw + Placebo_kw + {ctrl_str} | Stkcd_str + year_str",
               data=reg_p, vcov={"CRV1": "IndYear"})
print(f"\n  模型P2: DU_kw + Placebo_kw")
print(f"    DU_kw:      coef={mp2.coef()['DU_kw']:.6f}, t={mp2.tstat()['DU_kw']:.3f}, p={mp2.pvalue()['DU_kw']:.4f}")
print(f"    Placebo_kw: coef={mp2.coef()['Placebo_kw']:.6f}, t={mp2.tstat()['Placebo_kw']:.3f}, p={mp2.pvalue()['Placebo_kw']:.4f}")

results['horserace_placebo'] = {
    'DU_kw_coef': round(mp2.coef()['DU_kw'], 6),
    'DU_kw_t': round(mp2.tstat()['DU_kw'], 3),
    'DU_kw_p': round(mp2.pvalue()['DU_kw'], 4),
    'Placebo_kw_coef': round(mp2.coef()['Placebo_kw'], 6),
    'Placebo_kw_t': round(mp2.tstat()['Placebo_kw'], 3),
    'Placebo_kw_p': round(mp2.pvalue()['Placebo_kw'], 4),
    'N': int(mp2._N),
}

# 模型 S1: DU_kw_strict
reg_s = panel.dropna(subset=['PriceDelay','DU_kw_strict'] + controls + ['IndYear']).copy()
ms1 = pf.feols(f"PriceDelay ~ DU_kw_strict + {ctrl_str} | Stkcd_str + year_str",
               data=reg_s, vcov={"CRV1": "IndYear"})
print(f"\n  模型S1: DU_kw_strict（去泛化词）")
print(f"    DU_kw_strict: coef={ms1.coef()['DU_kw_strict']:.6f}, t={ms1.tstat()['DU_kw_strict']:.3f}, p={ms1.pvalue()['DU_kw_strict']:.4f}")
print(f"    N={ms1._N:,}")

results['du_kw_strict'] = {
    'coef': round(ms1.coef()['DU_kw_strict'], 6),
    't': round(ms1.tstat()['DU_kw_strict'], 3),
    'p': round(ms1.pvalue()['DU_kw_strict'], 4),
    'N': int(ms1._N),
}

# 描述统计
print(f"\n  描述统计:")
print(f"    DU_kw:        mean={panel['DU_kw'].mean():.4f}, sd={panel['DU_kw'].std():.4f}")
print(f"    Placebo_kw:   mean={panel['Placebo_kw'].mean():.4f}, sd={panel['Placebo_kw'].std():.4f}")
print(f"    DU_kw_strict: mean={panel['DU_kw_strict'].mean():.4f}, sd={panel['DU_kw_strict'].std():.4f}")
print(f"    buzzword占比:  {(panel['buzzword_count'].fillna(0) / panel['kw_total'].clip(lower=1)).mean():.2%}")

# ============================================================
# Part 3: CNRDS 外部收敛效度
# ============================================================
print("\n" + "=" * 60)
print("Part 3: CNRDS 外部收敛效度")
print("=" * 60)

cnrds = pd.read_parquet(f"{OUT_DIR}/cnrds_data_element_index.parquet")
cnrds = cnrds.iloc[1:]  # 跳过中文说明行
cnrds['Stkcd'] = pd.to_numeric(cnrds['Scode'], errors='coerce')
cnrds['year'] = pd.to_numeric(cnrds['Year'], errors='coerce')
cnrds['CNRDS_count'] = pd.to_numeric(cnrds['TWFre_count'], errors='coerce')
cnrds['CNRDS_words'] = pd.to_numeric(cnrds['Term_Only'], errors='coerce')
cnrds = cnrds.dropna(subset=['Stkcd','year','CNRDS_count','CNRDS_words'])
cnrds['Stkcd'] = cnrds['Stkcd'].astype(int)
cnrds['year'] = cnrds['year'].astype(int)
cnrds['CNRDS_per10k'] = cnrds['CNRDS_count'] / cnrds['CNRDS_words'].clip(lower=1) * 10000

# 合并
merged = ar_feat.merge(cnrds[['Stkcd','year','CNRDS_per10k','CNRDS_count']],
                       on=['Stkcd','year'], how='inner')
print(f"  匹配样本: {len(merged):,} obs ({merged.year.min()}-{merged.year.max()})")

# 相关系数
pearson_r, pearson_p = stats.pearsonr(merged['kw_per10k'], merged['CNRDS_per10k'])
spearman_r, spearman_p = stats.spearmanr(merged['kw_per10k'], merged['CNRDS_per10k'])
print(f"  Pearson r  = {pearson_r:.4f} (p={pearson_p:.4e})")
print(f"  Spearman ρ = {spearman_r:.4f} (p={spearman_p:.4e})")

# 也看 raw count 的相关
pearson_r2, _ = stats.pearsonr(merged['kw_total'], merged['CNRDS_count'])
print(f"  Raw count Pearson r = {pearson_r2:.4f}")

results['cnrds_validity'] = {
    'N': int(len(merged)),
    'pearson_r': round(pearson_r, 4),
    'pearson_p': round(pearson_p, 6),
    'spearman_r': round(spearman_r, 4),
    'spearman_p': round(spearman_p, 6),
    'raw_count_pearson': round(pearson_r2, 4),
}

# ============================================================
# Part 4: DataAsset 对照
# ============================================================
print("\n" + "=" * 60)
print("Part 4: DataAsset 对照验证")
print("=" * 60)

da_valid = panel.dropna(subset=['DataAsset', 'DU_kw']).copy()
print(f"  DataAsset 非缺失: {len(da_valid):,}")
print(f"  DataAsset > 0: {(da_valid['DataAsset'] > 0).sum():,} ({(da_valid['DataAsset'] > 0).mean():.2%})")
print(f"  DataAsset 年份分布:")
print(da_valid[da_valid['DataAsset'] > 0].groupby('year').size().to_string())

if (da_valid['DataAsset'] > 0).sum() > 100:
    da_valid['has_DataAsset'] = (da_valid['DataAsset'] > 0).astype(int)
    reg_da = da_valid.dropna(subset=['has_DataAsset','DU_kw'] + controls + ['IndYear']).copy()
    
    # OLS: has_DataAsset ~ DU_kw + controls | year FE
    mda = pf.feols(f"has_DataAsset ~ DU_kw + {ctrl_str} | year_str",
                   data=reg_da, vcov={"CRV1": "IndYear"})
    print(f"\n  OLS: has_DataAsset ~ DU_kw + controls | year FE")
    print(f"    DU_kw: coef={mda.coef()['DU_kw']:.6f}, t={mda.tstat()['DU_kw']:.3f}, p={mda.pvalue()['DU_kw']:.4f}")
    print(f"    N={mda._N:,}")
    
    results['data_asset'] = {
        'coef': round(mda.coef()['DU_kw'], 6),
        't': round(mda.tstat()['DU_kw'], 3),
        'p': round(mda.pvalue()['DU_kw'], 4),
        'N': int(mda._N),
        'has_DataAsset_pct': round((da_valid['DataAsset'] > 0).mean(), 4),
    }
else:
    print("  DataAsset > 0 样本量不足，跳过回归")
    results['data_asset'] = {'note': 'insufficient sample'}

# ============================================================
# 保存所有结果
# ============================================================
os.makedirs(f"{BASE}/results/v11_expanded", exist_ok=True)
with open(f"{BASE}/results/v11_expanded/construct_validity.json", 'w') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\n结果已保存: results/v11_expanded/construct_validity.json")

# 汇总
print("\n" + "=" * 60)
print("汇总")
print("=" * 60)
print(f"  Horse-race (控制总字数):   DU_kw t={results['horserace_total_chars']['DU_kw_t']}")
print(f"  Horse-race (控制MD&A):     DU_kw t={results['horserace_mda_chars']['DU_kw_t']}")
print(f"  Horse-race (两者都控制):   DU_kw t={results['horserace_both']['DU_kw_t']}")
print(f"  Placebo单独:               t={results['placebo_only']['Placebo_kw_t']}")
print(f"  DU_kw + Placebo:           DU_kw t={results['horserace_placebo']['DU_kw_t']}, Placebo t={results['horserace_placebo']['Placebo_kw_t']}")
print(f"  DU_kw_strict:              t={results['du_kw_strict']['t']}")
print(f"  CNRDS Pearson r:           {results['cnrds_validity']['pearson_r']}")
print(f"  CNRDS Spearman ρ:          {results['cnrds_validity']['spearman_r']}")
