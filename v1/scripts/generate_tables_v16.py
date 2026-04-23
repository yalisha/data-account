"""
v16表格: 朱康(2025)会计研究格式
- 表2: 基准回归 (7列, 含逐步加控制变量, 列出控制变量系数)
- 表2b: LLM vs 传统词频对比 (DML结果)
"""
import pandas as pd
import numpy as np
import json, os, warnings
warnings.filterwarnings('ignore')

from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.units import cm, mm
from reportlab.lib import colors
from reportlab.platypus import (SimpleDocTemplate, Table, TableStyle, Paragraph,
                                Spacer, PageBreak, KeepTogether)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase.pdfmetrics import registerFontFamily

BASE = "/Users/mac/computerscience/15会计研究"
OUT = f"{BASE}/results/v16_tables"
os.makedirs(OUT, exist_ok=True)

# ============================================================
# Font
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
# Styles
# ============================================================
styles = getSampleStyleSheet()
title_style = ParagraphStyle('Title', fontName=CN_FONT, fontSize=11,
                             alignment=TA_CENTER, spaceAfter=4, spaceBefore=4)
note_style = ParagraphStyle('Note', fontName=CN_FONT, fontSize=7,
                            alignment=TA_LEFT, leading=9)
cell_c = ParagraphStyle('CellC', fontName=CN_FONT, fontSize=7.5,
                        alignment=TA_CENTER, leading=9)
cell_l = ParagraphStyle('CellL', fontName=CN_FONT, fontSize=7.5,
                        alignment=TA_LEFT, leading=9)

def P(text):
    return Paragraph(str(text), cell_c)

def PL(text):
    return Paragraph(str(text), cell_l)


def fmt_coef_se(info):
    """返回 (系数行, 标准误行) 的文本"""
    if not info or not isinstance(info, dict):
        return '', ''
    coef = info['coef']
    se = info['se']
    p = info['p']
    stars = '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.1 else ''
    return f"{coef:.4f}{stars}", f"({se:.4f})"


# ============================================================
# Load data
# ============================================================
with open(f"{OUT}/ols_full_results_v16.json", 'r') as f:
    ols = json.load(f)

dml = pd.read_csv(f"{OUT}/dml_main_results_v16.csv")
dml_res = {row['Spec']: row for _, row in dml.iterrows()}

# ============================================================
# Table 2: 基准回归 (OLS + FE, 朱康格式)
# ============================================================
print("生成 表2: 基准回归检验")

# 7 columns
col_specs = ['(1)', '(2)', '(3)', '(4)', '(5)', '(6)', '(7)', '(8)']
y_labels = {
    '(1)': 'PriceDelay', '(2)': 'PriceDelay', '(3)': 'PriceDelay',
    '(4)': 'PriceDelay', '(5)': 'PriceDelay',
    '(6)': 'SYNCH', '(7)': 'SYNCH', '(8)': 'SYNCH',
}
d_vars = {
    '(1)': 'DU_kw', '(2)': 'DU_kw', '(3)': 'llm_score',
    '(4)': 'llm_score', '(5)': 'llm_binary',
    '(6)': 'DU_kw', '(7)': 'llm_score', '(8)': 'llm_binary',
}

# Treatment variables: each gets its own row, blank where not used
treatment_vars = [
    ('DU_kw', 'DU_kw', {'(1)', '(2)', '(6)'}),
    ('llm_score', 'LLM_score', {'(3)', '(4)', '(7)'}),
    ('llm_binary', 'LLM_binary', {'(5)', '(8)'}),
]

# Control variables
control_display = [
    ('Size', 'Size'), ('Lev', 'Lev'), ('ROA', 'ROA'),
    ('TobinQ', 'TobinQ'), ('Age', 'Age'), ('Growth', 'Growth'),
    ('IndepRatio', 'IndepRatio'), ('Dual', 'Dual'),
    ('Top1Share', 'Top1Share'), ('SOE', 'SOE'), ('CFO', 'CFO'),
]

# Build table data
table_data = []

# Row 0: Y variable header
row_y = [PL('')]
for c in col_specs:
    row_y.append(P(y_labels[c]))
table_data.append(row_y)

# Row 1: column numbers
row_num = [PL('变量')]
for c in col_specs:
    row_num.append(P(c))
table_data.append(row_num)

# Treatment variable rows (each D gets its own coef + se row)
for var_key, var_label, active_cols in treatment_vars:
    coef_row = [PL(var_label)]
    se_row = [PL('')]
    for c in col_specs:
        if c in active_cols:
            col_data = ols.get(c, {})
            info = col_data.get(var_key, None)
            c_str, se_str = fmt_coef_se(info)
            coef_row.append(P(c_str))
            se_row.append(P(se_str))
        else:
            coef_row.append(P(''))
            se_row.append(P(''))
    table_data.append(coef_row)
    table_data.append(se_row)

# Control variable rows (only show for columns with controls)
cols_with_ctrl = {'(2)', '(4)', '(5)', '(6)', '(7)', '(8)'}
for var_key, var_label in control_display:
    coef_row = [PL(var_label)]
    se_row = [PL('')]
    for c in col_specs:
        if c in cols_with_ctrl:
            col_data = ols.get(c, {})
            info = col_data.get(var_key, None)
            c_str, se_str = fmt_coef_se(info)
            coef_row.append(P(c_str))
            se_row.append(P(se_str))
        else:
            coef_row.append(P(''))
            se_row.append(P(''))
    table_data.append(coef_row)
    table_data.append(se_row)

# Firm/Year FE row
fe_row = [PL('Firm/Year FE')]
for c in col_specs:
    fe_row.append(P('Yes'))
table_data.append(fe_row)

# N row
n_row = [PL('N')]
for c in col_specs:
    n_val = ols.get(c, {}).get('N', 0)
    n_row.append(P(f"{n_val:,}"))
table_data.append(n_row)

# R2 row
r2_row = [PL('R\u00b2')]
for c in col_specs:
    r2_val = ols.get(c, {}).get('R2', 0)
    r2_row.append(P(f"{r2_val:.4f}"))
table_data.append(r2_row)

# Build table
n_cols = 8
col_widths = [2.6*cm] + [2.2*cm] * 8
t1 = Table(table_data, colWidths=col_widths)

# Count rows
n_rows = len(table_data)
t1.setStyle(TableStyle([
    ('FONTNAME', (0, 0), (-1, -1), CN_FONT),
    ('FONTSIZE', (0, 0), (-1, -1), 7.5),
    ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
    ('ALIGN', (0, 0), (0, -1), 'LEFT'),
    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    # Top border
    ('LINEABOVE', (0, 0), (-1, 0), 1.2, colors.black),
    # Below Y header
    ('LINEBELOW', (0, 0), (-1, 0), 0.5, colors.black),
    # Below column numbers
    ('LINEBELOW', (0, 1), (-1, 1), 0.5, colors.black),
    # Above FE row
    ('LINEABOVE', (0, n_rows-3), (-1, n_rows-3), 0.5, colors.black),
    # Bottom border
    ('LINEBELOW', (0, n_rows-1), (-1, n_rows-1), 1.2, colors.black),
    ('TOPPADDING', (0, 0), (-1, -1), 1),
    ('BOTTOMPADDING', (0, 0), (-1, -1), 1),
]))

title1 = Paragraph("表2  基准回归检验", title_style)
note1 = Paragraph(
    "注: 因变量分别为PriceDelay(股价延迟)和SYNCH(股价同步性)。"
    "第(1)(3)列不含控制变量，第(2)(4)-(8)列含全部控制变量。"
    "DU_kw为关键词频率(每万字)，LLM_score为LLM语义评分(0-3连续变量)，LLM_binary为LLM实质性二值化(score>=2取1，否则取0)。"
    "所有回归包含企业和年份双向固定效应，括号内为行业-年份层面聚类稳健标准误。"
    "***、**、*分别表示在1%、5%、10%水平显著。",
    note_style
)

# ============================================================
# Table 3: DML结果 (LLM vs 传统度量对比)
# ============================================================
print("生成 表3: DML-PLR结果对比")

dml_specs = [
    ('DU_kw', 'DML-27X', 'DML-27X-SYNCH'),
    ('DU_kw_ln', 'DML-27X-ln', None),
    ('DU_sub_ln', 'DML-27X-sub', None),
    ('LLM_score', 'LLM-score', 'LLM-score-SYNCH'),
    ('LLM_binary', 'LLM-binary', 'LLM-binary-SYNCH'),
]

dml_data = []
# Header
dml_data.append([PL(''), P('Y = PriceDelay'), P(''), P(''), P('Y = SYNCH'), P(''), P('')])
dml_data.append([PL('处理变量D'), P('系数'), P('标准误'), P('t值'), P('系数'), P('标准误'), P('t值')])

for d_label, pd_spec, synch_spec in dml_specs:
    row = [PL(d_label)]
    # PriceDelay
    r = dml_res[pd_spec]
    stars = '***' if r['p'] < 0.01 else '**' if r['p'] < 0.05 else '*' if r['p'] < 0.1 else ''
    row.append(P(f"{r['coef']:.4f}{stars}"))
    row.append(P(f"({r['se']:.4f})"))
    row.append(P(f"[{r['t']:.2f}]"))
    # SYNCH
    if synch_spec and synch_spec in dml_res:
        r2 = dml_res[synch_spec]
        stars2 = '***' if r2['p'] < 0.01 else '**' if r2['p'] < 0.05 else '*' if r2['p'] < 0.1 else ''
        row.append(P(f"{r2['coef']:.4f}{stars2}"))
        row.append(P(f"({r2['se']:.4f})"))
        row.append(P(f"[{r2['t']:.2f}]"))
    else:
        row.extend([P(''), P(''), P('')])
    dml_data.append(row)

# N row
n_pd = int(dml_res['DML-27X']['N'])
n_sy = int(dml_res['DML-27X-SYNCH']['N'])
dml_data.append([PL('控制变量+CRE'), P('Yes'), P(''), P(''), P('Yes'), P(''), P('')])
dml_data.append([PL('N'), P(f"{n_pd:,}"), P(''), P(''), P(f"{n_sy:,}"), P(''), P('')])

col_widths3 = [3*cm] + [2.2*cm] * 6
t3 = Table(dml_data, colWidths=col_widths3)
n_dml_rows = len(dml_data)
t3.setStyle(TableStyle([
    ('FONTNAME', (0, 0), (-1, -1), CN_FONT),
    ('FONTSIZE', (0, 0), (-1, -1), 7.5),
    ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
    ('ALIGN', (0, 0), (0, -1), 'LEFT'),
    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ('LINEABOVE', (0, 0), (-1, 0), 1.2, colors.black),
    ('LINEBELOW', (0, 0), (-1, 0), 0.5, colors.black),
    ('LINEBELOW', (0, 1), (-1, 1), 0.5, colors.black),
    ('LINEBELOW', (0, n_dml_rows-1), (-1, n_dml_rows-1), 1.2, colors.black),
    ('TOPPADDING', (0, 0), (-1, -1), 2),
    ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
    ('SPAN', (1, 0), (3, 0)),
    ('SPAN', (4, 0), (6, 0)),
]))

title3 = Paragraph("表3  DML-PLR回归结果: LLM语义评分 vs 传统词频度量", title_style)
note3 = Paragraph(
    "注: 所有规格均采用DML-PLR with Mundlak CRE，控制变量含13个基础变量+CRE均值+年份虚拟变量。"
    "ML学习器: Lasso(Y) + RandomForest500(D)，5折交叉验证，3次重复。"
    "DU_kw为关键词频率(每万字)，DU_kw_ln为log(1+关键词总数)，DU_sub_ln为log(1+规则实质性关键词数)。"
    "LLM_score为两层文本分析评分(0-3): 第一层关键词匹配，第二层Haiku模型语义判断。"
    "LLM_binary为LLM评分>=2的二值化指标。"
    "括号内为标准误，方括号内为t统计量。***、**、*分别表示在1%、5%、10%水平显著。",
    note_style
)

# ============================================================
# Table 1: 描述性统计
# ============================================================
print("生成 表1: 描述性统计")
desc = pd.read_csv(f"{OUT}/descriptive_stats_v16.csv", index_col=0)

var_labels_cn = {
    'PriceDelay': 'PriceDelay', 'SYNCH': 'SYNCH',
    'DU_kw': 'DU_kw', 'llm_score': 'LLM_score', 'llm_binary': 'LLM_binary',
    'Size': 'Size', 'Lev': 'Lev', 'ROA': 'ROA', 'TobinQ': 'TobinQ',
    'Age': 'Age', 'Growth': 'Growth', 'IndepRatio': 'IndepRatio',
    'Dual': 'Dual', 'Top1Share': 'Top1Share', 'SOE': 'SOE', 'CFO': 'CFO',
}
desc_order = ['PriceDelay','SYNCH','DU_kw','llm_score','llm_binary',
              'Size','Lev','ROA','TobinQ','Age','Growth','IndepRatio',
              'Dual','Top1Share','SOE','CFO']

desc_data = []
desc_data.append([PL('变量'), P('N'), P('均值'), P('标准差'), P('最小值'), P('中位数'), P('最大值')])

for var in desc_order:
    if var not in desc.index:
        continue
    row = desc.loc[var]
    desc_data.append([
        PL(var_labels_cn.get(var, var)),
        P(f"{int(row['count']):,}"),
        P(f"{row['mean']:.4f}"),
        P(f"{row['std']:.4f}"),
        P(f"{row['min']:.4f}"),
        P(f"{row['50%']:.4f}"),
        P(f"{row['max']:.4f}"),
    ])

col_widths_desc = [2.8*cm] + [2.5*cm]*6
t_desc = Table(desc_data, colWidths=col_widths_desc)
n_desc = len(desc_data)
t_desc.setStyle(TableStyle([
    ('FONTNAME', (0,0), (-1,-1), CN_FONT),
    ('FONTSIZE', (0,0), (-1,-1), 7.5),
    ('ALIGN', (1,0), (-1,-1), 'CENTER'),
    ('ALIGN', (0,0), (0,-1), 'LEFT'),
    ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ('LINEABOVE', (0,0), (-1,0), 1.2, colors.black),
    ('LINEBELOW', (0,0), (-1,0), 0.5, colors.black),
    ('LINEBELOW', (0,n_desc-1), (-1,n_desc-1), 1.2, colors.black),
    ('TOPPADDING', (0,0), (-1,-1), 2),
    ('BOTTOMPADDING', (0,0), (-1,-1), 2),
]))
title_desc = Paragraph("表1  变量的描述性统计", title_style)
note_desc = Paragraph(
    "注: LLM_score为基于大语言模型的数据要素利用语义评分(0-3)，"
    "LLM_binary为LLM实质性二值化(LLM_score>=2取1，否则取0)。"
    "DU_kw为传统关键词频率(每万字)。"
    "所有连续变量在1%和99%分位数处进行缩尾处理。",
    note_style)

# ============================================================
# Table 4: 稳健性检验
# ============================================================
print("生成 表4: 稳健性检验")
with open(f"{OUT}/robustness_v16.json", 'r') as f:
    robust = json.load(f)

robust_specs_order = [
    ('IndYearFE', '行业×年份FE'),
    ('Drop2024', '剔除2024年'),
    ('DropIT', '剔除IT行业'),
    ('IndAdj', '行业年度均值调整'),
    ('TwoWayCluster', '双向聚类SE'),
]

rob_data = []
# Header
rob_data.append([PL(''), P('D = LLM_score'), P(''), P('D = DU_kw'), P('')])
rob_data.append([PL('稳健性检验'), P('系数'), P('t值'), P('系数'), P('t值')])

for spec_key, spec_label in robust_specs_order:
    row = [PL(spec_label)]
    for d_label in ['LLM', 'KW']:
        key = f"{spec_key}_{d_label}"
        if key in robust:
            r = robust[key]
            stars = '***' if r['p']<0.01 else '**' if r['p']<0.05 else '*' if r['p']<0.1 else ''
            row.append(P(f"{r['coef']:.4f}{stars}"))
            row.append(P(f"[{r['t']:.2f}]"))
        else:
            row.extend([P(''), P('')])
    rob_data.append(row)

# Add N and FE info
rob_data.append([PL('控制变量'), P('Yes'), P(''), P('Yes'), P('')])
rob_data.append([PL('Firm/Year FE'), P('Yes'), P(''), P('Yes'), P('')])

col_widths_rob = [4*cm, 2.5*cm, 2*cm, 2.5*cm, 2*cm]
t_rob = Table(rob_data, colWidths=col_widths_rob)
n_rob = len(rob_data)
t_rob.setStyle(TableStyle([
    ('FONTNAME', (0,0), (-1,-1), CN_FONT),
    ('FONTSIZE', (0,0), (-1,-1), 7.5),
    ('ALIGN', (1,0), (-1,-1), 'CENTER'),
    ('ALIGN', (0,0), (0,-1), 'LEFT'),
    ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ('LINEABOVE', (0,0), (-1,0), 1.2, colors.black),
    ('LINEBELOW', (0,0), (-1,0), 0.5, colors.black),
    ('LINEBELOW', (0,1), (-1,1), 0.5, colors.black),
    ('LINEBELOW', (0,n_rob-1), (-1,n_rob-1), 1.2, colors.black),
    ('TOPPADDING', (0,0), (-1,-1), 2),
    ('BOTTOMPADDING', (0,0), (-1,-1), 2),
    ('SPAN', (1,0), (2,0)),
    ('SPAN', (3,0), (4,0)),
]))
title_rob = Paragraph("表4  稳健性检验 (Y = PriceDelay)", title_style)
note_rob = Paragraph(
    "注: 因变量均为PriceDelay。第(1)行替换行业×年份交互固定效应，"
    "第(2)行剔除2024年样本，第(3)行剔除信息传输、软件与信息技术服务业，"
    "第(4)行使用行业年度均值调整后的处理变量，第(5)行使用企业+年份双向聚类标准误。"
    "其余设置与基准回归一致。方括号内为t统计量。***、**、*分别表示在1%、5%、10%水平显著。",
    note_style)

# ============================================================
# Table 5: 机制检验
# ============================================================
print("生成 表5: 作用机制检验")
with open(f"{OUT}/mechanism_v16.json", 'r') as f:
    mech_res = json.load(f)

mech_channels = ['Analyst', 'Amihud', 'RetVol', 'Turnover', 'InstHold']
mech_labels = {
    'Analyst': '分析师关注(Analyst)',
    'Amihud': '非流动性(Amihud)',
    'RetVol': '收益率波动(RetVol)',
    'Turnover': '换手率(Turnover)',
    'InstHold': '机构持股(InstHold)',
}

mech_data = []
mech_data.append([PL(''), P('D = LLM_score'), P(''), P(''), P('D = DU_kw'), P(''), P('')])
mech_data.append([PL('机制变量M'), P('系数'), P('t值'), P('N'), P('系数'), P('t值'), P('N')])

for ch in mech_channels:
    row = [PL(mech_labels.get(ch, ch))]
    for d_label in ['LLM', 'KW']:
        key = f"{ch}_{d_label}"
        if key in mech_res:
            r = mech_res[key]
            stars = '***' if r['p']<0.01 else '**' if r['p']<0.05 else '*' if r['p']<0.1 else ''
            row.append(P(f"{r['coef']:.4f}{stars}"))
            row.append(P(f"[{r['t']:.2f}]"))
            row.append(P(f"{r['N']:,}"))
        else:
            row.extend([P(''), P(''), P('')])
    mech_data.append(row)

mech_data.append([PL('控制变量'), P('Yes'), P(''), P(''), P('Yes'), P(''), P('')])
mech_data.append([PL('Firm/Year FE'), P('Yes'), P(''), P(''), P('Yes'), P(''), P('')])

col_widths_mech = [3.5*cm] + [2*cm]*6
t_mech = Table(mech_data, colWidths=col_widths_mech)
n_mech = len(mech_data)
t_mech.setStyle(TableStyle([
    ('FONTNAME', (0,0), (-1,-1), CN_FONT),
    ('FONTSIZE', (0,0), (-1,-1), 7.5),
    ('ALIGN', (1,0), (-1,-1), 'CENTER'),
    ('ALIGN', (0,0), (0,-1), 'LEFT'),
    ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ('LINEABOVE', (0,0), (-1,0), 1.2, colors.black),
    ('LINEBELOW', (0,0), (-1,0), 0.5, colors.black),
    ('LINEBELOW', (0,1), (-1,1), 0.5, colors.black),
    ('LINEBELOW', (0,n_mech-1), (-1,n_mech-1), 1.2, colors.black),
    ('TOPPADDING', (0,0), (-1,-1), 2),
    ('BOTTOMPADDING', (0,0), (-1,-1), 2),
    ('SPAN', (1,0), (3,0)),
    ('SPAN', (4,0), (6,0)),
]))
title_mech = Paragraph("表5  作用机制检验 (D → M)", title_style)
note_mech = Paragraph(
    "注: 每列回归的因变量为机制变量M，处理变量分别为LLM_score和DU_kw。"
    "Analyst为ln(1+分析师关注人数)，Amihud为Amihud非流动性指标，"
    "RetVol为年化日收益率标准差，Turnover为年均日换手率，InstHold为机构持股比例。"
    "所有回归包含控制变量、企业和年份双向固定效应，行业×年份聚类标准误。"
    "***、**、*分别表示在1%、5%、10%水平显著。",
    note_style)

# ============================================================
# Table 6: 异质性检验
# ============================================================
print("生成 表6: 异质性检验")
with open(f"{OUT}/heterogeneity_v16.json", 'r') as f:
    hetero = json.load(f)

# Panel A: Subgroup regressions
het_groups = [
    ('SOE', 'SOE=1', 'SOE=0', '国有企业', '非国有企业'),
    ('Size_median', 'Large', 'Small', '大企业', '小企业'),
    ('Industry', 'IT', 'Non-IT', 'IT行业', '非IT行业'),
]

het_data = []
het_data.append([PL(''), P('D = LLM_score'), P(''), P(''), P('D = DU_kw'), P(''), P('')])
het_data.append([PL('分组'), P('系数'), P('t值'), P('N'), P('系数'), P('t值'), P('N')])

for group_name, sub_h, sub_l, label_h, label_l in het_groups:
    for sub, label in [(sub_h, label_h), (sub_l, label_l)]:
        row = [PL(label)]
        for d_label in ['LLM', 'KW']:
            key = f"sub_{group_name}_{sub}_{d_label}"
            if key in hetero:
                r = hetero[key]
                stars = '***' if r['p']<0.01 else '**' if r['p']<0.05 else '*' if r['p']<0.1 else ''
                row.append(P(f"{r['coef']:.4f}{stars}"))
                row.append(P(f"[{r['t']:.2f}]"))
                row.append(P(f"{r['N']:,}"))
            else:
                row.extend([P(''), P(''), P('')])
        het_data.append(row)

# Panel B: Interaction terms
het_data.append([PL(''), P(''), P(''), P(''), P(''), P(''), P('')])
het_data.append([PL('交互项检验'), P('D = LLM_score'), P(''), P(''), P('D = DU_kw'), P(''), P('')])
het_data.append([PL('调节变量'), P('主效应'), P('交互项'), P('t(交互)'), P('主效应'), P('交互项'), P('t(交互)')])

for mod_label in ['SOE', 'Size_median']:
    mod_display = '国有(SOE)' if mod_label == 'SOE' else '规模(Size)'
    row = [PL(mod_display)]
    for d_label in ['LLM', 'KW']:
        key = f"interact_{mod_label}_{d_label}"
        if key in hetero:
            r = hetero[key]
            s1 = '***' if r['main_p']<0.01 else '**' if r['main_p']<0.05 else '*' if r['main_p']<0.1 else ''
            s2 = '***' if r['interact_p']<0.01 else '**' if r['interact_p']<0.05 else '*' if r['interact_p']<0.1 else ''
            row.append(P(f"{r['main_coef']:.4f}{s1}"))
            row.append(P(f"{r['interact_coef']:.4f}{s2}"))
            row.append(P(f"[{r['interact_t']:.2f}]"))
        else:
            row.extend([P(''), P(''), P('')])
    het_data.append(row)

het_data.append([PL('控制变量/FE'), P('Yes'), P(''), P(''), P('Yes'), P(''), P('')])

col_widths_het = [3*cm] + [2.2*cm]*6
t_het = Table(het_data, colWidths=col_widths_het)
n_het = len(het_data)
t_het.setStyle(TableStyle([
    ('FONTNAME', (0,0), (-1,-1), CN_FONT),
    ('FONTSIZE', (0,0), (-1,-1), 7.5),
    ('ALIGN', (1,0), (-1,-1), 'CENTER'),
    ('ALIGN', (0,0), (0,-1), 'LEFT'),
    ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ('LINEABOVE', (0,0), (-1,0), 1.2, colors.black),
    ('LINEBELOW', (0,0), (-1,0), 0.5, colors.black),
    ('LINEBELOW', (0,1), (-1,1), 0.5, colors.black),
    ('LINEBELOW', (0,n_het-1), (-1,n_het-1), 1.2, colors.black),
    ('TOPPADDING', (0,0), (-1,-1), 2),
    ('BOTTOMPADDING', (0,0), (-1,-1), 2),
    ('SPAN', (1,0), (3,0)),
    ('SPAN', (4,0), (6,0)),
]))
title_het = Paragraph("表6  异质性检验 (Y = PriceDelay)", title_style)
note_het = Paragraph(
    "注: 上半部分为分组回归，按国有/非国有、大/小企业(以Size中位数划分)、IT/非IT行业分组。"
    "下半部分为交互项检验，调节变量分别为SOE虚拟变量和Size中位数虚拟变量。"
    "所有回归包含控制变量、企业和年份双向固定效应。***、**、*分别表示在1%、5%、10%水平显著。",
    note_style)

# ============================================================
# Build PDF
# ============================================================
doc = SimpleDocTemplate(
    f"{OUT}/regression_tables_v16.pdf",
    pagesize=landscape(A4),
    topMargin=1.5*cm, bottomMargin=1.5*cm,
    leftMargin=1.5*cm, rightMargin=1.5*cm,
)

elements = [
    title_desc, Spacer(1, 4*mm), t_desc, Spacer(1, 3*mm), note_desc,
    PageBreak(),
    title1, Spacer(1, 4*mm), t1, Spacer(1, 3*mm), note1,
    PageBreak(),
    title3, Spacer(1, 4*mm), t3, Spacer(1, 3*mm), note3,
    PageBreak(),
    title_rob, Spacer(1, 4*mm), t_rob, Spacer(1, 3*mm), note_rob,
    PageBreak(),
    title_mech, Spacer(1, 4*mm), t_mech, Spacer(1, 3*mm), note_mech,
    PageBreak(),
    title_het, Spacer(1, 4*mm), t_het, Spacer(1, 3*mm), note_het,
]

doc.build(elements)
print(f"\nPDF: {OUT}/regression_tables_v16.pdf")
print("Done!")
