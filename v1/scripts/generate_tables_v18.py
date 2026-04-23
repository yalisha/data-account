"""
v18表格PDF: 机制+异质性全部结果，朱康(2025)会计研究格式
"""
import os, warnings
warnings.filterwarnings('ignore')

from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (SimpleDocTemplate, Table, TableStyle, Paragraph,
                                Spacer, PageBreak, KeepTogether)
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase.pdfmetrics import registerFontFamily

BASE = "/Users/mac/computerscience/15会计研究"
OUT = f"{BASE}/results/v18"
os.makedirs(OUT, exist_ok=True)

# ── Font ──
CN = "Helvetica"
SONGTI = "/System/Library/Fonts/Supplemental/Songti.ttc"
if os.path.exists(SONGTI):
    try:
        pdfmetrics.registerFont(TTFont('SongtiSC', SONGTI, subfontIndex=6))
        pdfmetrics.registerFont(TTFont('SongtiSC-Bold', SONGTI, subfontIndex=1))
        registerFontFamily('SongtiSC', normal='SongtiSC', bold='SongtiSC-Bold')
        CN = 'SongtiSC'
    except:
        pass

# ── Styles ──
s_title = ParagraphStyle('T', fontName=CN, fontSize=10.5, alignment=TA_CENTER,
                         spaceAfter=4, spaceBefore=6, leading=14)
s_note = ParagraphStyle('N', fontName=CN, fontSize=7.5, alignment=TA_LEFT, leading=10)
s_c = ParagraphStyle('C', fontName=CN, fontSize=8, alignment=TA_CENTER, leading=10)
s_l = ParagraphStyle('L', fontName=CN, fontSize=8, alignment=TA_LEFT, leading=10)
s_c_sm = ParagraphStyle('CS', fontName=CN, fontSize=7, alignment=TA_CENTER, leading=9)
s_l_sm = ParagraphStyle('LS', fontName=CN, fontSize=7, alignment=TA_LEFT, leading=9)

P = lambda t, style=s_c: Paragraph(str(t), style)
L = lambda t, style=s_l: Paragraph(str(t), style)
Ps = lambda t: Paragraph(str(t), s_c_sm)
Ls = lambda t: Paragraph(str(t), s_l_sm)


def three_line(n_rows, hdr=1, spans=None):
    sty = [
        ('FONTNAME', (0, 0), (-1, -1), CN),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ('ALIGN', (0, 0), (0, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LINEABOVE', (0, 0), (-1, 0), 1.0, colors.black),
        ('LINEBELOW', (0, hdr - 1), (-1, hdr - 1), 0.5, colors.black),
        ('LINEBELOW', (0, n_rows - 1), (-1, n_rows - 1), 1.0, colors.black),
        ('TOPPADDING', (0, 0), (-1, -1), 1.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 1.5),
    ]
    if spans:
        sty.extend(spans)
    return TableStyle(sty)


def stars(t):
    at = abs(t)
    if at >= 2.576: return '***'
    if at >= 1.960: return '**'
    if at >= 1.645: return '*'
    return ''


def fmt_coef(coef, t, digits=4):
    s = stars(t)
    sign = '\u2212' if coef < 0 else ''
    return f'{sign}{abs(coef):.{digits}f}{s}'


def fmt_se(se, digits=4):
    return f'({se:.{digits}f})'


def fmt_t(t):
    sign = '\u2212' if t < 0 else ''
    return f'[{sign}{abs(t):.2f}]'


# ================================================================
# Load data
# ================================================================
import pandas as pd

mech = pd.read_csv(f"{OUT}/mechanism_v18.csv")
mech_new = pd.read_csv(f"{OUT}/mechanism_new_v18.csv")
mech_all = pd.concat([mech, mech_new], ignore_index=True)

het = pd.read_csv(f"{OUT}/heterogeneity_v18.csv")

# Supply chain mechanism (manually from log)
sc_mech = pd.DataFrame([
    {'channel': 'CustConc', 'treatment': 'DU_kw', 'coef': -0.0380, 'se': 0.0776, 'tstat': -0.49, 'pval': 0.624, 'N': 41975, 'r2': 0.8165},
    {'channel': 'CustConc', 'treatment': 'DU_llm', 'coef': -0.3260, 'se': 0.0780, 'tstat': -4.18, 'pval': 0.00003, 'N': 41975, 'r2': 0.8166},
    {'channel': 'SuppConc', 'treatment': 'DU_kw', 'coef': -0.4705, 'se': 0.0771, 'tstat': -6.10, 'pval': 1e-9, 'N': 39023, 'r2': 0.7542},
    {'channel': 'SuppConc', 'treatment': 'DU_llm', 'coef': -0.3776, 'se': 0.0837, 'tstat': -4.51, 'pval': 7e-6, 'N': 39023, 'r2': 0.7541},
    {'channel': 'SCConc', 'treatment': 'DU_kw', 'coef': -0.2767, 'se': 0.0692, 'tstat': -4.00, 'pval': 0.00007, 'N': 42332, 'r2': 0.7646},
    {'channel': 'SCConc', 'treatment': 'DU_llm', 'coef': -0.4602, 'se': 0.0680, 'tstat': -6.77, 'pval': 2e-11, 'N': 42332, 'r2': 0.7647},
    {'channel': 'CustHHI', 'treatment': 'DU_kw', 'coef': 0.1558, 'se': 0.0484, 'tstat': 3.22, 'pval': 0.001, 'N': 31188, 'r2': 0.7677},
    {'channel': 'CustHHI', 'treatment': 'DU_llm', 'coef': -0.0009, 'se': 0.045, 'tstat': -0.02, 'pval': 0.984, 'N': 31188, 'r2': 0.7675},
])
mech_all = pd.concat([mech_all, sc_mech], ignore_index=True)

# Supply chain heterogeneity (manually from log)
sc_het = pd.DataFrame([
    {'dimension': 'CustConc', 'treatment': 'DU_kw', 'coef_high': -0.0037, 'se_high': 0.0012, 't_high': -3.11,
     'N_high': 20684, 'r2_high': 0.43, 'coef_low': -0.0058, 'se_low': 0.0011, 't_low': -5.27,
     'N_low': 20678, 'r2_low': 0.44, 'fisher_p': 0.100},
    {'dimension': 'CustConc', 'treatment': 'DU_llm', 'coef_high': -0.0013, 'se_high': 0.0011, 't_high': -1.22,
     'N_high': 20684, 'r2_high': 0.43, 'coef_low': -0.0055, 'se_low': 0.0010, 't_low': -5.41,
     'N_low': 20678, 'r2_low': 0.44, 'fisher_p': 0.014},
    {'dimension': 'SCConc', 'treatment': 'DU_kw', 'coef_high': -0.0044, 'se_high': 0.0014, 't_high': -3.20,
     'N_high': 20811, 'r2_high': 0.43, 'coef_low': -0.0048, 'se_low': 0.0010, 't_low': -4.68,
     'N_low': 20725, 'r2_low': 0.44, 'fisher_p': 0.780},
    {'dimension': 'SCConc', 'treatment': 'DU_llm', 'coef_high': -0.0015, 'se_high': 0.0012, 't_high': -1.30,
     'N_high': 20811, 'r2_high': 0.43, 'coef_low': -0.0049, 'se_low': 0.0010, 't_low': -4.79,
     'N_low': 20725, 'r2_low': 0.44, 'fisher_p': 0.034},
])
het_all = pd.concat([het, sc_het], ignore_index=True)


# ================================================================
# 表1: 预期机制检验 (ForecastDisp, CashFlowVol, SCConc)
# ================================================================
print("表1: 预期机制检验")

key_mech = [
    ('ForecastDisp', '分析师预测分歧'),
    ('CashFlowVol', '现金流波动'),
    ('SCConc', '综合供应链集中度'),
]

d1 = [
    [L(''), P('(1)'), P('(2)'), P('(3)'), P('(4)'), P('(5)'), P('(6)')],
    [L(''),
     P('ForecastDisp'), P('ForecastDisp'),
     P('CashFlowVol'), P('CashFlowVol'),
     P('SCConc'), P('SCConc')],
]

# DU_kw row
row_kw = [L('DU<sub>kw</sub>')]
row_kw_se = [L('')]
for ch, _ in key_mech:
    r = mech_all[(mech_all['channel'] == ch) & (mech_all['treatment'] == 'DU_kw')]
    if len(r) > 0:
        r = r.iloc[0]
        row_kw.append(P(fmt_coef(r['coef'], r['tstat'])))
        row_kw_se.append(P(fmt_se(r['se'])))
    else:
        row_kw.extend([P(''), P('')])
        row_kw_se.extend([P(''), P('')])
    row_kw.append(P(''))
    row_kw_se.append(P(''))
# Fix: remove trailing empty
row_kw = row_kw[:7]
row_kw_se = row_kw_se[:7]
d1.append(row_kw)
d1.append(row_kw_se)

# DU_llm row
row_llm = [L('DU<sub>llm</sub>')]
row_llm_se = [L('')]
for ch, _ in key_mech:
    r = mech_all[(mech_all['channel'] == ch) & (mech_all['treatment'] == 'DU_llm')]
    if len(r) > 0:
        r = r.iloc[0]
        row_llm.append(P(''))
        row_llm_se.append(P(''))
        row_llm.append(P(fmt_coef(r['coef'], r['tstat'])))
        row_llm_se.append(P(fmt_se(r['se'])))
    else:
        row_llm.extend([P(''), P('')])
        row_llm_se.extend([P(''), P('')])
row_llm = row_llm[:7]
row_llm_se = row_llm_se[:7]
d1.append(row_llm)
d1.append(row_llm_se)

# Controls, FE, N, R2
d1.append([L('控制变量')] + [P('YES')] * 6)
d1.append([L('Firm/Year FE')] + [P('YES')] * 6)

# N row
row_n = [L('N')]
for ch, _ in key_mech:
    for treat in ['DU_kw', 'DU_llm']:
        r = mech_all[(mech_all['channel'] == ch) & (mech_all['treatment'] == treat)]
        if len(r) > 0:
            row_n.append(P(f'{int(r.iloc[0]["N"]):,}'))
        else:
            row_n.append(P(''))
d1.append(row_n)

# R2 row
row_r2 = [L('R\u00b2')]
for ch, _ in key_mech:
    for treat in ['DU_kw', 'DU_llm']:
        r = mech_all[(mech_all['channel'] == ch) & (mech_all['treatment'] == treat)]
        if len(r) > 0:
            row_r2.append(P(f'{r.iloc[0]["r2"]:.3f}'))
        else:
            row_r2.append(P(''))
d1.append(row_r2)

t1 = Table(d1, colWidths=[2.2*cm] + [2.2*cm]*6)
t1.setStyle(three_line(len(d1), hdr=2, spans=[('LINEBELOW', (0, 1), (-1, 1), 0.5, colors.black)]))
title1 = Paragraph("表1 传导机制检验", s_title)
note1 = Paragraph(
    "注：因变量分别为分析师预测分歧(ForecastDisp，预测EPS标准差)、现金流波动(CashFlowVol，CFO/TA三年滚动标准差)和"
    "综合供应链集中度(SCConc，前五大客户与供应商集中度均值)。所有模型控制企业与年份固定效应和11个控制变量，"
    "标准误在行业\u00d7年份水平聚类。", s_note)


# ================================================================
# 表2: 预期异质性 (DigEconCore, Post2020, SA, HighTech)
# ================================================================
print("表2: 预期异质性")

key_het = [
    ('DigEconCore', '数字经济核心', 'DU_kw'),
    ('Post2020', '2020年后', 'DU_kw'),
    ('SA', '融资约束', 'DU_llm'),
    ('HighTech', '高科技产业', 'DU_kw'),
]

d2 = [
    [L(''), P('(1)'), P('(2)'), P('(3)'), P('(4)'), P('(5)'), P('(6)'), P('(7)'), P('(8)')],
    [L(''), P('数字核心'), P('非数字核心'), P('2020后'), P('2020前'), P('高约束'), P('低约束'), P('高科技'), P('非高科技')],
    [L(''), P('PriceDelay'), P('PriceDelay'), P('PriceDelay'), P('PriceDelay'), P('PriceDelay'), P('PriceDelay'), P('PriceDelay'), P('PriceDelay')],
]

# Build rows for each dimension
dims_data = []
for dim, label, treat in key_het:
    r = het_all[(het_all['dimension'] == dim) & (het_all['treatment'] == treat)]
    if len(r) > 0:
        r = r.iloc[0]
        dims_data.append({
            'dim': dim, 'label': label, 'treat': treat,
            'coef_h': r['coef_high'], 'se_h': r['se_high'], 't_h': r['t_high'], 'n_h': int(r['N_high']), 'r2_h': r['r2_high'],
            'coef_l': r['coef_low'], 'se_l': r['se_low'], 't_l': r['t_low'], 'n_l': int(r['N_low']), 'r2_l': r['r2_low'],
            'fp': r['fisher_p']
        })

# Treatment variable row
treat_label = {'DU_kw': 'DU<sub>kw</sub>', 'DU_llm': 'DU<sub>llm</sub>'}
row_coef = [L('')]
row_se = [L('')]
for dd in dims_data:
    tl = treat_label[dd['treat']]
    row_coef.extend([P(fmt_coef(dd['coef_h'], dd['t_h'])), P(fmt_coef(dd['coef_l'], dd['t_l']))])
    row_se.extend([P(fmt_se(dd['se_h'])), P(fmt_se(dd['se_l']))])

# We need to handle that different dims use different treatment vars
# Row for DU_kw dims (DigEconCore, Post2020, HighTech)
row_kw = [L('DU<sub>kw</sub>')]
row_kw_se = [L('')]
for dd in dims_data:
    if dd['treat'] == 'DU_kw':
        row_kw.extend([P(fmt_coef(dd['coef_h'], dd['t_h'])), P(fmt_coef(dd['coef_l'], dd['t_l']))])
        row_kw_se.extend([P(fmt_se(dd['se_h'])), P(fmt_se(dd['se_l']))])
    else:
        row_kw.extend([P(''), P('')])
        row_kw_se.extend([P(''), P('')])
d2.append(row_kw)
d2.append(row_kw_se)

# Row for DU_llm dims (SA)
row_llm = [L('DU<sub>llm</sub>')]
row_llm_se = [L('')]
for dd in dims_data:
    if dd['treat'] == 'DU_llm':
        row_llm.extend([P(fmt_coef(dd['coef_h'], dd['t_h'])), P(fmt_coef(dd['coef_l'], dd['t_l']))])
        row_llm_se.extend([P(fmt_se(dd['se_h'])), P(fmt_se(dd['se_l']))])
    else:
        row_llm.extend([P(''), P('')])
        row_llm_se.extend([P(''), P('')])
d2.append(row_llm)
d2.append(row_llm_se)

d2.append([L('控制变量')] + [P('YES')] * 8)
d2.append([L('Firm/Year FE')] + [P('YES')] * 8)

row_n2 = [L('N')]
for dd in dims_data:
    row_n2.extend([P(f'{dd["n_h"]:,}'), P(f'{dd["n_l"]:,}')])
d2.append(row_n2)

row_r22 = [L('R\u00b2')]
for dd in dims_data:
    row_r22.extend([P(f'{dd["r2_h"]:.3f}'), P(f'{dd["r2_l"]:.3f}')])
d2.append(row_r22)

# Fisher P row
row_fp = [L('Fisher P')]
for dd in dims_data:
    fp_s = stars(2.576 if dd['fp'] < 0.01 else (1.96 if dd['fp'] < 0.05 else (1.645 if dd['fp'] < 0.1 else 0)))
    row_fp.extend([P(f'{dd["fp"]:.3f}{fp_s}'), P('')])
d2.append(row_fp)

t2 = Table(d2, colWidths=[1.8*cm] + [1.8*cm]*8)
t2.setStyle(three_line(len(d2), hdr=3, spans=[
    ('LINEBELOW', (0, 2), (-1, 2), 0.5, colors.black),
]))
title2 = Paragraph("表2 异质性检验", s_title)
note2 = Paragraph(
    "注：数字经济核心产业按国家统计局(2021)分类(C39/I63/I64/I65)；2020年后指year\u22652020；"
    "融资约束按SA指数中位数分组(Hadlock和Pierce，2010)；高科技按CSRC行业代码分类。"
    "Fisher P为费舍尔组合检验P值(500次抽样)。DigEconCore/Post2020/HighTech报告DU<sub>kw</sub>，SA报告DU<sub>llm</sub>。", s_note)


# ================================================================
# 表3: 全部机制检验结果
# ================================================================
print("表3: 全部机制")

mech_order = [
    ('Analyst', '分析师覆盖', '信息中介'),
    ('ForecastDisp', '预测分歧', '信息质量'),
    ('RatingDisp', '评级分歧', '信息质量'),
    ('ReportFreq', '研报频次', '信息中介'),
    ('FcstAcc', '预测准确性', '信息质量'),
    ('Amihud', '非流动性', '市场微观'),
    ('RetVol', '收益波动', '市场微观'),
    ('IdioVol', '特质波动', '市场微观'),
    ('RetAutoCorr', '收益自相关', '价格调整'),
    ('Turnover', '换手率', '市场微观'),
    ('TurnoverVol', '换手波动', '噪声交易'),
    ('InstHold', '机构持股', '投资者'),
    ('InstStable', '稳定机构', '投资者'),
    ('absDA', '盈余管理', '会计质量'),
    ('InvestIneff', '投资效率', '实体效率'),
    ('AuditFee', '审计费用', '审计质量'),
    ('CashFlowVol', '现金流波动', '经营风险'),
    ('NCSKEW', '崩盘风险', '风险'),
    ('DUVOL', '下行波动比', '风险'),
    ('TFP', '全要素生产率', '实体效率'),
    ('CustConc', '客户集中度', '供应链'),
    ('SuppConc', '供应商集中度', '供应链'),
    ('SCConc', '供应链集中度', '供应链'),
    ('CustHHI', '客户HHI', '供应链'),
]

d3 = [
    [Ls('渠道变量'), Ls('类别'), Ls('DU<sub>kw</sub>系数'), Ls('t值'), Ls('DU<sub>llm</sub>系数'), Ls('t值'), Ls('N')],
]

for ch, cn, cat in mech_order:
    kw = mech_all[(mech_all['channel'] == ch) & (mech_all['treatment'] == 'DU_kw')]
    llm = mech_all[(mech_all['channel'] == ch) & (mech_all['treatment'] == 'DU_llm')]
    if len(kw) == 0 and len(llm) == 0:
        continue
    row = [Ls(cn), Ls(cat)]
    if len(kw) > 0:
        k = kw.iloc[0]
        row.extend([Ps(fmt_coef(k['coef'], k['tstat'])), Ps(fmt_t(k['tstat']))])
    else:
        row.extend([Ps(''), Ps('')])
    if len(llm) > 0:
        l = llm.iloc[0]
        row.extend([Ps(fmt_coef(l['coef'], l['tstat'])), Ps(fmt_t(l['tstat']))])
    else:
        row.extend([Ps(''), Ps('')])
    n_val = int(kw.iloc[0]['N']) if len(kw) > 0 else (int(llm.iloc[0]['N']) if len(llm) > 0 else 0)
    row.append(Ps(f'{n_val:,}'))
    d3.append(row)

t3 = Table(d3, colWidths=[2.4*cm, 1.6*cm, 2.4*cm, 1.8*cm, 2.4*cm, 1.8*cm, 1.8*cm])
t3.setStyle(three_line(len(d3)))
title3 = Paragraph("表3 全部机制检验结果", s_title)
note3 = Paragraph(
    "注：因变量为各渠道变量，自变量为DU<sub>kw</sub>或DU<sub>llm</sub>加11个控制变量，"
    "控制企业与年份固定效应，标准误在行业\u00d7年份水平聚类。", s_note)


# ================================================================
# 表4: 全部异质性检验结果
# ================================================================
print("表4: 全部异质性")

het_order = [
    ('Analyst', '分析师覆盖'),
    ('SOE', '产权性质'),
    ('InstHold', '机构持股'),
    ('HighTech', '高科技产业'),
    ('Post2020', '政策时期'),
    ('DigEconCore', '数字经济核心'),
    ('Size', '企业规模'),
    ('SA', '融资约束'),
    ('CustConc', '客户集中度'),
    ('SCConc', '供应链集中度'),
    ('MainBoard', '主板'),
    ('Hhi', '行业集中度'),
    ('ProvDigital2016', '数字基础设施'),
    ('Big4', '四大审计'),
    ('IndustryCluster', '产业集群'),
    ('StrategicEmerging', '战略新兴产业'),
    ('LQ', '区位商'),
    ('ShareholderNum', '股东分散度'),
    ('BM', '账面市值比'),
    ('Intangible', '无形资产'),
]

d4 = [
    [Ls('维度'), Ls('测度'),
     Ls('高组系数'), Ls('t'), Ls('低组系数'), Ls('t'), Ls('Fisher P'),
     Ls('N<sub>高</sub>'), Ls('N<sub>低</sub>')],
]

for dim, dn in het_order:
    for treat in ['DU_kw', 'DU_llm']:
        r = het_all[(het_all['dimension'] == dim) & (het_all['treatment'] == treat)]
        if len(r) == 0:
            continue
        r = r.iloc[0]
        fp = r['fisher_p']
        fp_str = f'{fp:.3f}{stars(2.576 if fp < 0.01 else (1.96 if fp < 0.05 else (1.645 if fp < 0.1 else 0)))}'
        treat_short = 'kw' if treat == 'DU_kw' else 'llm'
        d4.append([
            Ls(dn if treat == 'DU_kw' else ''),
            Ls(treat_short),
            Ps(fmt_coef(r['coef_high'], r['t_high'])),
            Ps(fmt_t(r['t_high'])),
            Ps(fmt_coef(r['coef_low'], r['t_low'])),
            Ps(fmt_t(r['t_low'])),
            Ps(fp_str),
            Ps(f'{int(r["N_high"]):,}'),
            Ps(f'{int(r["N_low"]):,}'),
        ])

t4 = Table(d4, colWidths=[1.8*cm, 0.8*cm, 2.0*cm, 1.5*cm, 2.0*cm, 1.5*cm, 1.5*cm, 1.5*cm, 1.5*cm])
t4.setStyle(three_line(len(d4)))
title4 = Paragraph("表4 全部异质性检验结果", s_title)
note4 = Paragraph(
    "注：因变量为PriceDelay。连续变量按年度中位数分组，二值变量直接分组。"
    "Fisher P为费舍尔组合检验P值(500次抽样)。所有模型控制企业与年份固定效应和11个控制变量，"
    "标准误在行业\u00d7年份水平聚类。", s_note)


# ================================================================
# Build PDF
# ================================================================
print("Building PDF...")
outfile = f"{OUT}/v18_tables.pdf"
doc = SimpleDocTemplate(outfile, pagesize=A4,
                        topMargin=2*cm, bottomMargin=2*cm,
                        leftMargin=2*cm, rightMargin=2*cm)

story = [
    title1, Spacer(1, 4), t1, Spacer(1, 4), note1,
    PageBreak(),
    title2, Spacer(1, 4), t2, Spacer(1, 4), note2,
    PageBreak(),
    title3, Spacer(1, 4), t3, Spacer(1, 4), note3,
    PageBreak(),
    title4, Spacer(1, 4), t4, Spacer(1, 4), note4,
]

doc.build(story)
print(f"Saved: {outfile}")
