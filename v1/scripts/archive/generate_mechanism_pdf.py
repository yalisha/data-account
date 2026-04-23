"""
生成全部表格PDF: 稳健性+内生性+机制(全渠道OLS+OLS vs DML)
数据来源: Stata .tex + mechanism_v15_all_channels.json
"""

import json, os
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
OUT = f"{BASE}/results/v14_tables"

# ============================================================
# 字体
# ============================================================
CN_FONT = "Helvetica"
SONGTI_PATH = "/System/Library/Fonts/Supplemental/Songti.ttc"
if os.path.exists(SONGTI_PATH):
    try:
        pdfmetrics.registerFont(TTFont('SongtiSC', SONGTI_PATH, subfontIndex=6))
        pdfmetrics.registerFont(TTFont('SongtiSC-Bold', SONGTI_PATH, subfontIndex=1))
        registerFontFamily('SongtiSC', normal='SongtiSC', bold='SongtiSC-Bold',
                           italic='SongtiSC', boldItalic='SongtiSC-Bold')
        CN_FONT = 'SongtiSC'
    except:
        pass

# ============================================================
# 样式
# ============================================================
title_style = ParagraphStyle('TableTitle', fontName=CN_FONT, fontSize=10,
                             leading=14, alignment=TA_CENTER, spaceAfter=6)
note_style = ParagraphStyle('TableNote', fontName=CN_FONT, fontSize=7.5,
                            leading=10, alignment=TA_LEFT, spaceAfter=12)
cell_style = ParagraphStyle('CellText', fontName=CN_FONT, fontSize=8,
                            leading=10, alignment=TA_CENTER)
cell_left = ParagraphStyle('CellLeft', fontName=CN_FONT, fontSize=8,
                           leading=10, alignment=TA_LEFT)
header_style = ParagraphStyle('HeaderText', fontName=CN_FONT, fontSize=8,
                              leading=10, alignment=TA_CENTER)

def P(text, style=cell_style):
    return Paragraph(str(text), style)

def P_coef(c, sig, dec=4):
    txt = f"{c:.{dec}f}"
    if sig:
        txt += f"<super>{sig}</super>"
    return Paragraph(txt, ParagraphStyle('Coef', fontName='Times-Roman',
                     fontSize=8, leading=10, alignment=TA_CENTER))

def P_se(s, dec=4):
    return Paragraph(f"({s:.{dec}f})", ParagraphStyle('SE', fontName='Times-Roman',
                     fontSize=8, leading=10, alignment=TA_CENTER))

def P_t(t):
    return Paragraph(f"[{t:.2f}]", ParagraphStyle('T', fontName='Times-Roman',
                     fontSize=7.5, leading=9, alignment=TA_CENTER))

def P_int(n):
    return Paragraph(f"{int(n):,}", ParagraphStyle('Int', fontName='Times-Roman',
                     fontSize=8, leading=10, alignment=TA_CENTER))

def P_num(n, dec=3):
    return Paragraph(f"{n:.{dec}f}", ParagraphStyle('Num', fontName='Times-Roman',
                     fontSize=8, leading=10, alignment=TA_CENTER))

def P_var(text):
    return Paragraph(f"<i>{text}</i>", ParagraphStyle('Var', fontName='Times-Italic',
                     fontSize=8, leading=10, alignment=TA_LEFT))

def P_hdr(text):
    return Paragraph(text, ParagraphStyle('Hdr', fontName=CN_FONT,
                     fontSize=7.5, leading=9, alignment=TA_CENTER))

# ============================================================
# 加载数据
# ============================================================
with open(f"{OUT}/mechanism_v15_all_channels.json", 'r') as f:
    data = json.load(f)

ols_file = f"{OUT}/mechanism_all_ols.json"
dml_file = f"{OUT}/mechanism_all_dml.json"
with open(ols_file, 'r') as f:
    ols = json.load(f)
with open(dml_file, 'r') as f:
    dml = json.load(f)

# ============================================================
# PDF
# ============================================================
pdf_path = f"{OUT}/mechanism_v15_tables.pdf"
doc = SimpleDocTemplate(pdf_path, pagesize=landscape(A4),
                        leftMargin=1.5*cm, rightMargin=1.5*cm,
                        topMargin=1.5*cm, bottomMargin=1.5*cm)

story = []

# ============================================================
# 表1: OLS全渠道筛选 (10渠道 × 同期+滞后)
# ============================================================
print("  Table 1: OLS全渠道筛选...")

all_channels = [
    ('ForecastDisp', '预测分歧',   'Diether et al. (2002)'),
    ('Analyst',      '分析师覆盖', 'Zhang (2006)'),
    ('RetAutoCorr',  '收益率自相关', 'Chordia et al. (2008)'),
    ('InvestIneff',  '投资效率偏离', 'Richardson (2006)'),
    ('absDA',        '盈余管理',   'Kothari et al. (2005)'),
    ('Turnover',     '换手率',     'Chordia et al. (2008)'),
    ('Amihud',       '非流动性',   'Amihud (2002)'),
    ('InstHold',     '机构持股',   'Buss & Sundaresan (2023)'),
    ('RetVol',       '收益波动率', 'Zhang (2006)'),
    ('AuditFee',     '审计费用',   'DeFond & Zhang (2014)'),
]

# Header rows
t1 = []
# Row 0: column numbers
t1.append([P('', cell_left), P('', cell_left)] +
          [P(f'({i+1})', header_style) for i in range(4)])

# Row 1: sub-headers
t1.append([P('渠道变量', cell_left), P('参考文献', cell_left),
           P_hdr('系数'), P_hdr('标准误'), P_hdr('t值'), P_hdr('系数'), ])
# Actually let me redo this more clearly

t1 = []
# Spanning header: 同期 | 滞后
t1.append([P('', cell_left), P('', cell_left),
           P('<b>同期回归</b>', header_style), P('', header_style),
           P('<b>滞后一期</b>', header_style), P('', header_style)])

t1.append([P('渠道变量', cell_left), P('参考文献', cell_left),
           P_hdr('系数'), P_hdr('t值'),
           P_hdr('系数'), P_hdr('t值')])

# Data rows
for var, cn, source in all_channels:
    c_key = f'{var}_concurrent'
    l_key = f'{var}_lagged'
    c = ols.get(c_key, {})
    l = ols.get(l_key, {})

    row = [P(cn, cell_left), P(source, ParagraphStyle('src', fontName='Times-Roman',
           fontSize=7, leading=9, alignment=TA_LEFT))]

    if c:
        row.append(P_coef(c['coef'], c.get('sig', ''), 4))
        row.append(P_t(c['t']))
    else:
        row += [P('', cell_style)] * 2

    if l:
        row.append(P_coef(l['coef'], l.get('sig', ''), 4))
        row.append(P_t(l['t']))
    else:
        row += [P('', cell_style)] * 2

    t1.append(row)

# Bottom rows
t1.append([P('控制变量', cell_left), P('', cell_left)] + [P('27个', cell_style)] * 4)
t1.append([P('固定效应', cell_left), P('', cell_left)] + [P('Firm+Year', cell_style)] * 4)

col_widths = [2.8*cm, 4.5*cm, 3.0*cm, 2.2*cm, 3.0*cm, 2.2*cm]
table1 = Table(t1, colWidths=col_widths)
table1.setStyle(TableStyle([
    ('FONT', (0,0), (-1,-1), CN_FONT, 7.5),
    ('ALIGN', (0,0), (-1,-1), 'CENTER'),
    ('ALIGN', (0,0), (1,-1), 'LEFT'),
    ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ('TOPPADDING', (0,0), (-1,-1), 2),
    ('BOTTOMPADDING', (0,0), (-1,-1), 2),
    # Spanning header
    ('SPAN', (2,0), (3,0)),  # 同期
    ('SPAN', (4,0), (5,0)),  # 滞后
    # Lines
    ('LINEABOVE', (0,0), (-1,0), 1.2, colors.black),
    ('LINEBELOW', (0,1), (-1,1), 0.5, colors.black),
    # Separator after robust channels (after row 5 = index 6 = 4th data row)
    ('LINEBELOW', (0,5), (-1,5), 0.3, colors.grey),
    # Bottom
    ('LINEBELOW', (0,-1), (-1,-1), 1.2, colors.black),
    # Background for robust channels
    ('BACKGROUND', (0,2), (-1,5), colors.Color(0.95, 0.97, 1.0)),
]))

story.append(Paragraph("表A　　　　　　　　OLS全渠道机制筛选（江艇两步法）", title_style))
story.append(table1)
story.append(Spacer(1, 4))
story.append(Paragraph(
    "注：遵循江艇(2022)两步法，仅估计X→M。被解释变量为各渠道变量，核心解释变量为DU<sub>kw</sub>。"
    "前四个渠道（蓝色底色）为OLS和DML双方法稳健的渠道。"
    "固定效应为企业+年份，标准误按行业×年份聚类。控制变量27个（当渠道变量为控制变量时从控制列表中移除）。"
    "***、**、* 分别表示在1%、5%、10%水平显著。", note_style))
story.append(PageBreak())


# ============================================================
# 表2: 稳健渠道 OLS vs DML 对比 (4渠道 × 同期+滞后 × OLS+DML)
# ============================================================
print("  Table 2: OLS vs DML对比...")

robust_channels = [
    ('ForecastDisp', '预测分歧',    'Diether et al. (2002)'),
    ('Analyst',      '分析师覆盖',  'Zhang (2006)'),
    ('RetAutoCorr',  '收益率自相关', 'Chordia et al. (2008)'),
    ('InvestIneff',  '投资效率偏离', 'Richardson (2006)'),
]

t2 = []

# Header row 0: method spanning
t2.append([P('', cell_left)] +
          [P('<b>OLS</b>', header_style)] * 4 +
          [P('<b>DML-CRE</b>', header_style)] * 4)

# Header row 1: column numbers
t2.append([P('', cell_left)] +
          [P(f'({i+1})', header_style) for i in range(8)])

# Header row 2: channel names (repeated for OLS and DML)
ch_names = [P_hdr(cn) for _, cn, _ in robust_channels]
t2.append([P('', cell_left)] + ch_names + ch_names)

# --- Panel A: 同期 ---
t2.append([P('<b>Panel A: 同期</b>', cell_left)] + [P('', cell_style)] * 8)

# Coef row
cr = [P_var('DU<sub>kw</sub>')]
for var, _, _ in robust_channels:
    r = ols.get(f'{var}_concurrent', {})
    cr.append(P_coef(r.get('coef', 0), r.get('sig', ''), 4) if r else P('', cell_style))
for var, _, _ in robust_channels:
    r = dml.get(f'{var}_concurrent', {})
    cr.append(P_coef(r.get('coef', 0), r.get('sig', ''), 4) if r else P('', cell_style))
t2.append(cr)

# SE row
sr = [P('', cell_left)]
for var, _, _ in robust_channels:
    r = ols.get(f'{var}_concurrent', {})
    sr.append(P_se(r.get('se', 0), 4) if r else P('', cell_style))
for var, _, _ in robust_channels:
    r = dml.get(f'{var}_concurrent', {})
    sr.append(P_se(r.get('se', 0), 4) if r else P('', cell_style))
t2.append(sr)

# t row
tr = [P('', cell_left)]
for var, _, _ in robust_channels:
    r = ols.get(f'{var}_concurrent', {})
    tr.append(P_t(r.get('t', 0)) if r else P('', cell_style))
for var, _, _ in robust_channels:
    r = dml.get(f'{var}_concurrent', {})
    tr.append(P_t(r.get('t', 0)) if r else P('', cell_style))
t2.append(tr)

# N row
nr = [P_var('N')]
for var, _, _ in robust_channels:
    r = ols.get(f'{var}_concurrent', {})
    nr.append(P_int(r.get('N', 0)) if r else P('', cell_style))
for var, _, _ in robust_channels:
    r = dml.get(f'{var}_concurrent', {})
    nr.append(P_int(r.get('N', 0)) if r else P('', cell_style))
t2.append(nr)

# R2 row (OLS only, DML no R2)
r2r = [P_var('R<super>2</super>')]
for var, _, _ in robust_channels:
    r = ols.get(f'{var}_concurrent', {})
    r2r.append(P_num(r.get('R2', 0)) if r else P('', cell_style))
for _ in robust_channels:
    r2r.append(P('', cell_style))
t2.append(r2r)

# --- Panel B: 滞后 ---
t2.append([P('<b>Panel B: 滞后一期</b>', cell_left)] + [P('', cell_style)] * 8)

# Coef row
cr2 = [P_var('DU<sub>kw</sub>')]
for var, _, _ in robust_channels:
    r = ols.get(f'{var}_lagged', {})
    cr2.append(P_coef(r.get('coef', 0), r.get('sig', ''), 4) if r else P('', cell_style))
for var, _, _ in robust_channels:
    r = dml.get(f'{var}_lagged', {})
    cr2.append(P_coef(r.get('coef', 0), r.get('sig', ''), 4) if r else P('', cell_style))
t2.append(cr2)

# SE row
sr2 = [P('', cell_left)]
for var, _, _ in robust_channels:
    r = ols.get(f'{var}_lagged', {})
    sr2.append(P_se(r.get('se', 0), 4) if r else P('', cell_style))
for var, _, _ in robust_channels:
    r = dml.get(f'{var}_lagged', {})
    sr2.append(P_se(r.get('se', 0), 4) if r else P('', cell_style))
t2.append(sr2)

# t row
tr2 = [P('', cell_left)]
for var, _, _ in robust_channels:
    r = ols.get(f'{var}_lagged', {})
    tr2.append(P_t(r.get('t', 0)) if r else P('', cell_style))
for var, _, _ in robust_channels:
    r = dml.get(f'{var}_lagged', {})
    tr2.append(P_t(r.get('t', 0)) if r else P('', cell_style))
t2.append(tr2)

# N row
nr2 = [P_var('N')]
for var, _, _ in robust_channels:
    r = ols.get(f'{var}_lagged', {})
    nr2.append(P_int(r.get('N', 0)) if r else P('', cell_style))
for var, _, _ in robust_channels:
    r = dml.get(f'{var}_lagged', {})
    nr2.append(P_int(r.get('N', 0)) if r else P('', cell_style))
t2.append(nr2)

# R2 row
r2r2 = [P_var('R<super>2</super>')]
for var, _, _ in robust_channels:
    r = ols.get(f'{var}_lagged', {})
    r2r2.append(P_num(r.get('R2', 0)) if r else P('', cell_style))
for _ in robust_channels:
    r2r2.append(P('', cell_style))
t2.append(r2r2)

# Bottom info rows
t2.append([P('控制变量', cell_left)] + [P('27个', cell_style)] * 4 + [P('27个+CRE', cell_style)] * 4)
t2.append([P('固定效应', cell_left)] + [P('Firm+Year', cell_style)] * 4 + [P('Mundlak+年份', cell_style)] * 4)
t2.append([P('ML学习器', cell_left)] + [P('', cell_style)] * 4 + [P('Lasso+RF', cell_style)] * 4)

col_widths2 = [2.5*cm] + [2.6*cm] * 8
table2 = Table(t2, colWidths=col_widths2)
table2.setStyle(TableStyle([
    ('FONT', (0,0), (-1,-1), CN_FONT, 7.5),
    ('ALIGN', (0,0), (-1,-1), 'CENTER'),
    ('ALIGN', (0,0), (0,-1), 'LEFT'),
    ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ('TOPPADDING', (0,0), (-1,-1), 1.5),
    ('BOTTOMPADDING', (0,0), (-1,-1), 1.5),
    # Spanning: OLS(1-4), DML(5-8) in row 0
    ('SPAN', (1,0), (4,0)),
    ('SPAN', (5,0), (8,0)),
    # Lines
    ('LINEABOVE', (0,0), (-1,0), 1.2, colors.black),
    ('LINEBELOW', (0,0), (-1,0), 0.5, colors.black),  # method header
    ('LINEBELOW', (0,2), (-1,2), 0.5, colors.black),  # channel names
    ('LINEBELOW', (0,8), (-1,8), 0.3, colors.black),  # Panel A bottom
    ('LINEBELOW', (0,-1), (-1,-1), 1.2, colors.black),
    # Vertical separator between OLS and DML
    ('LINEBEFORE', (5,0), (5,-1), 0.5, colors.grey),
]))

story.append(Paragraph("表B　　　　稳健渠道: OLS与DML-CRE对比（江艇两步法）", title_style))
story.append(table2)
story.append(Spacer(1, 4))
story.append(Paragraph(
    "注：本表报告OLS和DML-CRE双方法下均方向一致且显著的四个稳健渠道。"
    "OLS列使用企业+年份固定效应，行业×年份聚类标准误。"
    "DML-CRE列使用Chernozhukov et al. (2018)部分线性模型，"
    "CRE=关联随机效应(Mundlak, 1978)控制个体固定效应，年份虚拟变量控制时间效应，"
    "学习器为Lasso(LassoCV, 5折)+随机森林(500棵, max_depth=6)，5折×3次重复交叉拟合。"
    "括号内为标准误，方括号内为t值。"
    "***、**、* 分别表示在1%、5%、10%水平显著。", note_style))

story.append(PageBreak())

# ============================================================
# 表C: 稳健性检验 (Stata reghdfe, 27 controls, 9列)
# ============================================================
print("  Table C: 稳健性检验...")

rob_stata = {
    'indyear':  (-0.0018, 0.0006, '***', 38729, 0.522),
    'provyear': (-0.0033, 0.0008, '***', 38786, 0.482),
    'indadj':   (-0.0015, 0.0005, '***', 38786, 0.473),
    'no2024':   (-0.0035, 0.0009, '***', 34406, 0.486),
    'noit':     (-0.0032, 0.0009, '***', 36106, 0.473),
    'mainboard':(-0.0024, 0.0013, '*',   16157, 0.467),
    'psm':      (-0.0031, 0.0009, '***', 24458, 0.496),
    'twoway':   (-0.0034, 0.0011, '***', 38786, 0.473),
    'lead':     (-0.0029, 0.0009, '***', 34108, 0.486),
}
rob_lead_detail = {'coef': -0.0006, 'se': 0.0008, 'sig': ''}

rob_labels = [
    "(1) Ind×Year<br/>FE",
    "(2) Prov×Year<br/>FE",
    "(3) 行业均值<br/>调整",
    "(4) 剔除<br/>2024年",
    "(5) 剔除信息<br/>技术业",
    "(6) 仅<br/>主板",
    "(7) 倾向得<br/>分匹配",
    "(8) 双向<br/>聚类SE",
    "(9) 前导项<br/>检验",
]
rob_keys = ['indyear','provyear','indadj','no2024','noit','mainboard','psm','twoway','lead']
rob_var_names = ['DU_kw']*9
rob_var_names[2] = 'DU_kw_adj'

t3 = []
t3.append([P('', cell_left)] + [P_hdr(l) for l in rob_labels])
t3.append([P('被解释变量=', cell_left)] + [P('<i>PriceDelay</i>',
           ParagraphStyle('dv_r', fontName='Times-Italic', fontSize=7, leading=9, alignment=TA_CENTER))] * 9)

# Variable name row
t3.append([P('', cell_left)] + [P_var(v) for v in rob_var_names])

# Coef row
cr = [P('', cell_left)]
for k in rob_keys:
    c, s, sig, n, r2 = rob_stata[k]
    cr.append(P_coef(c, sig, 4))
t3.append(cr)

# SE row
sr = [P('', cell_left)]
for k in rob_keys:
    c, s, sig, n, r2 = rob_stata[k]
    sr.append(P_se(s, 4))
t3.append(sr)

# Lead variable extra row (col 9 only)
t3.append([P('', cell_left)] + [P('', cell_style)]*8 + [P_var('DU_kw_lead')])
t3.append([P('', cell_left)] + [P('', cell_style)]*8 +
          [P_coef(rob_lead_detail['coef'], rob_lead_detail['sig'], 4)])
t3.append([P('', cell_left)] + [P('', cell_style)]*8 +
          [P_se(rob_lead_detail['se'], 4)])

# FE rows
rob_fe = {
    'Firm FE':      ['YES']*9,
    'Year FE':      ['NO','NO','YES','YES','YES','YES','YES','YES','YES'],
    'Ind×Year FE':  ['YES','NO','NO','NO','NO','NO','NO','NO','NO'],
    'Prov×Year FE': ['NO','YES','NO','NO','NO','NO','NO','NO','NO'],
}
for label, vals in rob_fe.items():
    t3.append([P(label, cell_left)] + [P(v, cell_style) for v in vals])

t3.append([P('Controls', cell_left)] + [P('27个', cell_style)]*9)
t3.append([P_var('N')] + [P_int(rob_stata[k][3]) for k in rob_keys])
t3.append([P_var('R<super>2</super>')] + [P_num(rob_stata[k][4]) for k in rob_keys])

table3 = Table(t3, colWidths=[2.2*cm] + [1.75*cm]*9)
table3.setStyle(TableStyle([
    ('FONT', (0,0), (-1,-1), CN_FONT, 7),
    ('ALIGN', (0,0), (-1,-1), 'CENTER'),
    ('ALIGN', (0,0), (0,-1), 'LEFT'),
    ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ('TOPPADDING', (0,0), (-1,-1), 1.5),
    ('BOTTOMPADDING', (0,0), (-1,-1), 1.5),
    ('LINEABOVE', (0,0), (-1,0), 1.2, colors.black),
    ('LINEBELOW', (0,0), (-1,0), 0.5, colors.black),
    ('LINEBELOW', (0,2), (-1,2), 0.5, colors.black),
    ('LINEBELOW', (0,-1), (-1,-1), 1.2, colors.black),
]))

story.append(Paragraph("表C　　　　　　　　　　　　稳健性检验", title_style))
story.append(table3)
story.append(Spacer(1, 4))
story.append(Paragraph(
    "注：全部模型来自Stata reghdfe，27个控制变量。"
    "模型(1)控制企业和行业×年份交互FE，模型(2)控制企业和省份×年份交互FE。"
    "模型(3)的DU_kw_adj为行业年度均值调整后的DU_kw。"
    "模型(7)以DU_kw中位数划分处理组，经Logit倾向得分1:1近邻匹配(caliper=0.05)后在匹配样本上回归。"
    "模型(8)使用企业和年份双向聚类标准误。"
    "模型(9)同时纳入DU_kw和DU_kw_lead(t+1期)，DU_kw_lead系数不显著，缓解反向因果担忧。"
    "括号内为聚类稳健标准误。***、**、* 分别表示在1%、5%、10%水平显著。", note_style))
story.append(PageBreak())


# ============================================================
# 表D: 内生性检验 (Stata ivreghdfe, 27 controls)
# ============================================================
print("  Table D: 内生性检验...")

# First stage
fs_data = {
    'peer':   ('DU_kw_peer', 0.5675, 0.0350, '***', 38730),
    'bartik': ('bartik_iv',  0.2848, 0.0351, '***', 38077),
    'lag':    ('peer_lag',   0.5740, 0.0426, '***', 31576),
}
# Second stage
iv_models = [
    ('OLS',        'DU_kw',       -0.0034, 0.0008, '***', 38786, 0.473),
    ('Peer IV',    'DU_kw',       -0.0114, 0.0047, '**',  38730, None),
    ('Bartik IV',  'DU_kw',       -0.0094, 0.0066, '',    38077, None),
    ('Lag OLS',    'DU_kw_lag',   -0.0026, 0.0010, '***', 31637, 0.460),
    ('Lag IV',     'DU_kw_lag',   -0.0131, 0.0054, '**',  31576, None),
]
iv_kp_f = {1: 263.1, 2: 65.8, 4: 181.3}
iv_dwh_p = {1: 0.050, 2: 0.290, 4: 0.026}

# Panel A: First Stage
t4a = []
t4a.append([P('<b>Panel A: First Stage</b>', cell_left)] + [P('', cell_style)]*2)
fs_labels = ["(2) Peer IV", "(3) Bartik IV", "(5) Lag IV"]
t4a.append([P('', cell_left)] + [P_hdr(l) for l in fs_labels])
t4a.append([P('被解释变量=', cell_left),
            P('<i>DU<sub>kw</sub></i>', ParagraphStyle('dv_fs1', fontName='Times-Italic', fontSize=8, leading=10, alignment=TA_CENTER)),
            P('<i>DU<sub>kw</sub></i>', ParagraphStyle('dv_fs2', fontName='Times-Italic', fontSize=8, leading=10, alignment=TA_CENTER)),
            P('<i>DU<sub>kw,lag</sub></i>', ParagraphStyle('dv_fs3', fontName='Times-Italic', fontSize=8, leading=10, alignment=TA_CENTER))])

fs_keys = ['peer', 'bartik', 'lag']
fs_var_display = ['DU_kw_peer', 'Bartik_IV', 'Peer_lag']
t4a.append([P('', cell_left)] + [P_var(v) for v in fs_var_display])

cr_fs = [P('', cell_left)]
sr_fs = [P('', cell_left)]
for k in fs_keys:
    _, c, s, sig, n = fs_data[k]
    cr_fs.append(P_coef(c, sig, 4))
    sr_fs.append(P_se(s, 4))
t4a.append(cr_fs)
t4a.append(sr_fs)

t4a.append([P('Controls', cell_left)] + [P('YES', cell_style)]*3)
t4a.append([P('Firm/Year FE', cell_left)] + [P('YES', cell_style)]*3)
t4a.append([P_var('N')] + [P_int(fs_data[k][4]) for k in fs_keys])

table4a = Table(t4a, colWidths=[2.5*cm] + [4.2*cm]*3)
table4a.setStyle(TableStyle([
    ('FONT', (0,0), (-1,-1), CN_FONT, 8),
    ('ALIGN', (0,0), (-1,-1), 'CENTER'),
    ('ALIGN', (0,0), (0,-1), 'LEFT'),
    ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ('TOPPADDING', (0,0), (-1,-1), 1.5),
    ('BOTTOMPADDING', (0,0), (-1,-1), 1.5),
    ('LINEABOVE', (0,0), (-1,0), 1.2, colors.black),
    ('LINEBELOW', (0,0), (-1,0), 0.5, colors.black),
    ('LINEBELOW', (0,2), (-1,2), 0.5, colors.black),
    ('LINEBELOW', (0,-1), (-1,-1), 0.5, colors.black),
]))

# Panel B: Second Stage
t4b = []
t4b.append([P('<b>Panel B: Second Stage</b>', cell_left)] + [P('', cell_style)]*4)
iv_col_labels = ["(1) OLS", "(2) Peer IV", "(3) Bartik IV", "(4) Lag OLS", "(5) Lag IV"]
t4b.append([P('', cell_left)] + [P_hdr(l) for l in iv_col_labels])
t4b.append([P('被解释变量=', cell_left)] +
           [P('<i>PriceDelay</i>', ParagraphStyle(f'dv_iv{i}', fontName='Times-Italic',
            fontSize=8, leading=10, alignment=TA_CENTER)) for i in range(5)])

cr_iv = [P_var('DU<sub>kw</sub>')]
sr_iv = [P('', cell_left)]
for label, var, c, s, sig, n, r2 in iv_models:
    cr_iv.append(P_coef(c, sig, 4))
    sr_iv.append(P_se(s, 4))
t4b.append(cr_iv)
t4b.append(sr_iv)

t4b.append([P('Controls', cell_left)] + [P('YES', cell_style)]*5)
t4b.append([P('Firm/Year FE', cell_left)] + [P('YES', cell_style)]*5)

kp_vals = [None, 263.1, 65.8, None, 181.3]
t4b.append([P('KP F', cell_left)] + [P(f"{v:.1f}" if v else '', cell_style) for v in kp_vals])

dwh_vals = [None, 0.050, 0.290, None, 0.026]
t4b.append([P('DWH p', cell_left)] + [P(f"{v:.3f}" if v else '', cell_style) for v in dwh_vals])

t4b.append([P_var('N')] + [P_int(n) for _, _, c, s, sig, n, r2 in iv_models])
t4b.append([P_var('R<super>2</super>')] + [P_num(r2) if r2 else P('', cell_style) for _, _, c, s, sig, n, r2 in iv_models])

table4b = Table(t4b, colWidths=[2.2*cm] + [2.8*cm]*5)
table4b.setStyle(TableStyle([
    ('FONT', (0,0), (-1,-1), CN_FONT, 8),
    ('ALIGN', (0,0), (-1,-1), 'CENTER'),
    ('ALIGN', (0,0), (0,-1), 'LEFT'),
    ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ('TOPPADDING', (0,0), (-1,-1), 1.5),
    ('BOTTOMPADDING', (0,0), (-1,-1), 1.5),
    ('LINEABOVE', (0,0), (-1,0), 0.5, colors.black),
    ('LINEBELOW', (0,0), (-1,0), 0.5, colors.black),
    ('LINEBELOW', (0,2), (-1,2), 0.5, colors.black),
    ('LINEBELOW', (0,-1), (-1,-1), 1.2, colors.black),
]))

story.append(Paragraph("表D　　　　　　　　　　　　内生性检验", title_style))
story.append(table4a)
story.append(Spacer(1, 2))
story.append(table4b)
story.append(Spacer(1, 4))
story.append(Paragraph(
    "注：全部模型来自Stata ivreghdfe，27个控制变量。"
    "工具变量: Peer IV为同省跨行业DU_kw均值(leave-own-industry-out)；"
    "Bartik IV为2016年省级大数据发展指数×时间趋势；"
    "Lag IV使用滞后一期Peer均值作为DU_kw_lag的工具变量。"
    "KP F为Kleibergen-Paap rk Wald F统计量。DWH p为Durbin-Wu-Hausman内生性检验p值。"
    "括号内为聚类稳健标准误。***、**、* 分别表示在1%、5%、10%水平显著。", note_style))


# ============================================================
# Build PDF
# ============================================================
doc.build(story)
print(f"\nPDF已生成: {pdf_path}")
