"""
v16表格 - 严格按朱康(2025)会计研究格式
v16.4: 全面对标朱康范文格式
"""
import pandas as pd
import numpy as np
import json, os, warnings
warnings.filterwarnings('ignore')

from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.units import cm, mm
from reportlab.lib import colors
from reportlab.platypus import (SimpleDocTemplate, Table, TableStyle, Paragraph,
                                Spacer, PageBreak)
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase.pdfmetrics import registerFontFamily

BASE = "/Users/mac/computerscience/15会计研究"
OUT = f"{BASE}/results/v16_tables"

CN_FONT = "Helvetica"
IT_FONT = "Helvetica-Oblique"  # italic for variable names
SONGTI_PATH = "/System/Library/Fonts/Supplemental/Songti.ttc"
if os.path.exists(SONGTI_PATH):
    try:
        pdfmetrics.registerFont(TTFont('SongtiSC', SONGTI_PATH, subfontIndex=6))
        pdfmetrics.registerFont(TTFont('SongtiSC-Bold', SONGTI_PATH, subfontIndex=1))
        registerFontFamily('SongtiSC', normal='SongtiSC', bold='SongtiSC-Bold',
                           italic='SongtiSC', boldItalic='SongtiSC-Bold')
        CN_FONT = 'SongtiSC'
    except: pass

# Try to register Times-Italic for variable names
try:
    IT_FONT = 'Times-Italic'
except: pass

title_s = ParagraphStyle('T', fontName=CN_FONT, fontSize=10.5, alignment=TA_CENTER, spaceAfter=4)
note_s = ParagraphStyle('N', fontName=CN_FONT, fontSize=7, alignment=TA_LEFT, leading=9)
cc = ParagraphStyle('CC', fontName=CN_FONT, fontSize=7.5, alignment=TA_CENTER, leading=9)
cl = ParagraphStyle('CL', fontName=CN_FONT, fontSize=7.5, alignment=TA_LEFT, leading=9)
# Italic style for variable names
ci = ParagraphStyle('CI', fontName=IT_FONT, fontSize=7.5, alignment=TA_LEFT, leading=9)

def P(t): return Paragraph(str(t), cc)
def PL(t): return Paragraph(str(t), cl)
def PI(t): return Paragraph(str(t), ci)  # italic variable name
def stars(p):
    if p < 0.01: return '<super>***</super>'
    if p < 0.05: return '<super>**</super>'
    if p < 0.1: return '<super>*</super>'
    return ''

def std_table_style(data, extra=None):
    n = len(data)
    s = [
        ('FONTNAME', (0,0), (-1,-1), CN_FONT), ('FONTSIZE', (0,0), (-1,-1), 7.5),
        ('ALIGN', (1,0), (-1,-1), 'CENTER'), ('ALIGN', (0,0), (0,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('LINEABOVE', (0,0), (-1,0), 1.2, colors.black),
        ('LINEBELOW', (0,0), (-1,0), 0.5, colors.black),
        ('LINEBELOW', (0,n-1), (-1,n-1), 1.2, colors.black),
        ('TOPPADDING', (0,0), (-1,-1), 1.5), ('BOTTOMPADDING', (0,0), (-1,-1), 1.5),
    ]
    if extra: s.extend(extra)
    return TableStyle(s)

# Load results
with open(f"{OUT}/ols_full_results_v16.json") as f: ols = json.load(f)
dml = pd.read_csv(f"{OUT}/dml_main_results_v16.csv")
dml_r = {r['Spec']: r for _,r in dml.iterrows()}
with open(f"{OUT}/robustness_v16.json") as f: robust = json.load(f)
with open(f"{OUT}/mechanism_v16.json") as f: mech = json.load(f)
with open(f"{OUT}/heterogeneity_v16.json") as f: hetero = json.load(f)
desc = pd.read_csv(f"{OUT}/descriptive_stats_v16.csv", index_col=0)
with open(f"{OUT}/extra_descriptive_v16.json") as f: extra_desc = json.load(f)

elements = []

# ============================================================
# 表1: 描述性统计
# ============================================================
print("表1")
desc_vars = ['PriceDelay','SYNCH','DU_kw','DU_llm',
             'Size','Lev','ROA','TobinQ','Age','Growth','IndepRatio','Dual','Top1Share','SOE','CFO']

d1 = [[PL('变量'), P('观测值'), P('均值'), P('标准差'), P('最小值'), P('中位数'), P('最大值')]]
for v in desc_vars:
    if v not in desc.index: continue
    r = desc.loc[v]
    d1.append([PI(v), P(f"{int(r['count'])}"), P(f"{r['mean']:.3f}"), P(f"{r['std']:.3f}"),
               P(f"{r['min']:.3f}"), P(f"{r['50%']:.3f}"), P(f"{r['max']:.3f}")])

t1 = Table(d1, colWidths=[2.8*cm]+[2.5*cm]*6)
t1.setStyle(std_table_style(d1))

corr = extra_desc.get('correlations', {})
elements += [Paragraph("表1\u3000变量的描述性统计", title_s), Spacer(1,4*mm), t1, Spacer(1,3*mm),
    Paragraph(f"注: DU_llm = log(1+关键词总数) \u00d7 LLM语义评分/3，为LLM质量加权的关键词强度指数。"
              f"DU_kw为关键词频率(每万字)。连续变量在1%和99%处缩尾。有效回归样本覆盖2011-2022年。"
              f"DU_kw与DU_llm的Pearson相关系数为{corr.get('DU_kw_DU_llm',0):.3f}。"
              f"括号内数值为经行业层面聚类调整后的稳健标准误;***、**、*分别表示在1%、5%、10%的水平上显著，下同。", note_s),
    PageBreak()]

# ============================================================
# 表2: 基准回归 (6列)
# ============================================================
print("表2")
ols_cols = ['(1)','(2)','(3)','(4)','(6)','(7)']
col_labels = ['(1)','(2)','(3)','(4)','(5)','(6)']
y_map = {'(1)':'PriceDelay','(2)':'PriceDelay','(3)':'PriceDelay','(4)':'PriceDelay','(6)':'SYNCH','(7)':'SYNCH'}

treat_rows = [
    ('DU_kw','DU_kw', {'(1)','(2)','(6)'}),
    ('DU_llm','DU_llm', {'(3)','(4)','(7)'}),
]
ctrl_vars_list = ['Size','Lev','ROA','TobinQ','Age','Growth','IndepRatio','Dual','Top1Share','SOE','CFO']
cols_ctrl = {'(2)','(4)','(6)','(7)'}

def fmt(info):
    if not info: return '',''
    return f"{info['coef']:.4f}{stars(info['p'])}", f"({info['se']:.4f})"

d2 = []
# Column numbers on top (朱康 format)
d2.append([PL('')]+[P(cl) for cl in col_labels])
# DV names below
d2.append([PL('变量')]+[P(y_map[c]) for c in ols_cols])

for vk,vl,ac in treat_rows:
    cr,sr = [PI(vl)],[PI('')]
    for c in ols_cols:
        if c in ac:
            cs,ss = fmt(ols.get(c,{}).get(vk))
            cr.append(P(cs)); sr.append(P(ss))
        else: cr.append(P('')); sr.append(P(''))
    d2.append(cr); d2.append(sr)

for vk in ctrl_vars_list:
    cr,sr = [PI(vk)],[PI('')]
    for c in ols_cols:
        if c in cols_ctrl:
            cs,ss = fmt(ols.get(c,{}).get(vk))
            cr.append(P(cs)); sr.append(P(ss))
        else: cr.append(P('')); sr.append(P(''))
    d2.append(cr); d2.append(sr)

# Footer: Firm/Year before 控制变量 (朱康 order)
d2.append([PL('Firm/Year')]+[P('YES')]*6)
d2.append([PL('N')]+[P(f"{ols.get(c,{}).get('N',0)}") for c in ols_cols])
d2.append([PL('R<super>2</super>')]+[P(f"{ols.get(c,{}).get('R2',0):.3f}") for c in ols_cols])

t2 = Table(d2, colWidths=[2.6*cm]+[2.8*cm]*6)
n2 = len(d2)
t2.setStyle(std_table_style(d2, [('LINEBELOW',(0,1),(-1,1),0.5,colors.black),
    ('LINEABOVE',(0,n2-3),(-1,n2-3),0.5,colors.black)]))

du_sd = desc.loc['DU_llm','std'] if 'DU_llm' in desc.index else 1.0
pd_mean = desc.loc['PriceDelay','mean'] if 'PriceDelay' in desc.index else 0.12
du_coef = ols.get('(4)',{}).get('DU_llm',{}).get('coef', -0.005)
econ_mag = abs(du_coef * du_sd / pd_mean * 100)

elements += [Paragraph("表2\u3000基准回归检验", title_s), Spacer(1,4*mm), t2, Spacer(1,3*mm),
    Paragraph(f"注: 第(1)(3)列不含控制变量。DU_llm每增加1个标准差({du_sd:.2f})，"
              f"PriceDelay降低约{abs(du_coef*du_sd):.4f}，相当于均值的{econ_mag:.1f}%。"
              f"R<super>2</super>为within-R<super>2</super>。", note_s),
    PageBreak()]

# ============================================================
# 表3: DML
# ============================================================
print("表3")
dml_specs = [
    ('DU_kw',     'DML-27X',     'DML-27X-SYNCH'),
    ('DU_kw_ln',  'DML-27X-ln',  None),
    ('DU_sub_ln', 'DML-27X-sub', None),
    ('DU_llm',    'DU-llm',      'DU-llm-SYNCH'),
]

d3 = [[PL(''), P('Y = PriceDelay'),P(''),P(''), P('Y = SYNCH'),P(''),P('')],
      [PL('处理变量D'), P('系数'),P('标准误'),P('t值'), P('系数'),P('标准误'),P('t值')]]
for dl,ps,ss in dml_specs:
    if ps not in dml_r:
        d3.append([PI(dl), P(''),P(''),P(''), P(''),P(''),P('')])
        continue
    r = dml_r[ps]
    row = [PI(dl), P(f"{r['coef']:.4f}{stars(r['p'])}"), P(f"({r['se']:.4f})"), P(f"[{r['t']:.2f}]")]
    if ss and ss in dml_r:
        r2=dml_r[ss]; row+=[P(f"{r2['coef']:.4f}{stars(r2['p'])}"), P(f"({r2['se']:.4f})"), P(f"[{r2['t']:.2f}]")]
    else:
        row+=[P(''),P(''),P('')]
    d3.append(row)
d3.append([PL('控制变量+CRE'),P('YES'),P(''),P(''),P('YES'),P(''),P('')])
n_pd_kw = int(dml_r.get('DML-27X',{}).get('N',0))
n_pd_llm = int(dml_r.get('DU-llm',{}).get('N',0))
n_synch_kw = int(dml_r.get('DML-27X-SYNCH',{}).get('N',0))
n_synch_llm = int(dml_r.get('DU-llm-SYNCH',{}).get('N',0))
d3.append([PL('N'),P(f"{n_pd_kw}/{n_pd_llm}"),P(''),P(''),
           P(f"{n_synch_kw}/{n_synch_llm}"),P(''),P('')])
t3 = Table(d3, colWidths=[3*cm]+[2.2*cm]*6)
t3.setStyle(std_table_style(d3, [('LINEBELOW',(0,1),(-1,1),0.5,colors.black),('SPAN',(1,0),(3,0)),('SPAN',(4,0),(6,0))]))

dml_du_coef = dml_r.get('DU-llm',{}).get('coef', 0)
dml_econ = abs(dml_du_coef * du_sd / pd_mean * 100) if dml_du_coef else 0

elements += [Paragraph("表3\u3000DML-PLR回归结果", title_s), Spacer(1,4*mm), t3, Spacer(1,3*mm),
    Paragraph(f"注: DML-PLR with Mundlak CRE。ML学习器: Lasso(Y)+RF500(D)，5折3重复。"
              f"括号内为标准误，方括号内为t值。N行分别报告DU_kw系列/DU_llm的样本量。"
              f"SYNCH列系数为正，与PriceDelay降低的方向看似相反，"
              f"前者反映个股价格发现加速，后者反映行业共性信息占比上升，二者捕捉信息效率的不同维度。", note_s),
    PageBreak()]

# ============================================================
# 表4: 稳健性检验
# ============================================================
print("表4")
rob_specs = [
    ('(1) 行业\u00d7\n年份FE', 'IndYearFE'),
    ('(2) 剔除\n2022年', 'DropLastYear'),
    ('(3) 剔除\nIT行业', 'DropIT'),
    ('(4) 滞后\n一期X', 'LaggedX'),
    ('(5) PSM\n匹配', 'PSM'),
    ('(6) 双向\n聚类SE', 'TwoWayCluster'),
]
n_rob = len(rob_specs)

d4 = []
d4.append([PL('')] + [P(label) for label, _ in rob_specs])
d4.append([PL('变量')] + [P('PriceDelay')]*n_rob)

# Panel A
d4.append([PL('Panel A: D = DU_llm')] + [P('')]*n_rob)
cr, sr = [PI('DU_llm')], [PI('')]
for _, key in rob_specs:
    r = robust.get(f'{key}_LLM', {})
    if r:
        cr.append(P(f"{r['coef']:.4f}{stars(r['p'])}"))
        sr.append(P(f"({r['se']:.4f})"))
    else: cr.append(P('')); sr.append(P(''))
d4.append(cr); d4.append(sr)

d4.append([PL('Firm/Year')] + [P('YES')]*n_rob)
d4.append([PL('控制变量')] + [P('YES')]*n_rob)
nr_a = [PL('N')]
rr_a = [PL('R<super>2</super>')]
for _, key in rob_specs:
    r = robust.get(f'{key}_LLM', {})
    nr_a.append(P(f"{r.get('N',0)}") if r else P(''))
    rr_a.append(P(f"{r.get('R2',0):.3f}") if r else P(''))
d4.append(nr_a); d4.append(rr_a)

# Panel B
d4.append([PL('Panel B: D = DU_kw')] + [P('')]*n_rob)
cr2, sr2 = [PI('DU_kw')], [PI('')]
for _, key in rob_specs:
    r = robust.get(f'{key}_KW', {})
    if r:
        cr2.append(P(f"{r['coef']:.4f}{stars(r['p'])}"))
        sr2.append(P(f"({r['se']:.4f})"))
    else: cr2.append(P('')); sr2.append(P(''))
d4.append(cr2); d4.append(sr2)

d4.append([PL('Firm/Year')] + [P('YES')]*n_rob)
d4.append([PL('控制变量')] + [P('YES')]*n_rob)
nr_b = [PL('N')]
rr_b = [PL('R<super>2</super>')]
for _, key in rob_specs:
    r = robust.get(f'{key}_KW', {})
    nr_b.append(P(f"{r.get('N',0)}") if r else P(''))
    rr_b.append(P(f"{r.get('R2',0):.3f}") if r else P(''))
d4.append(nr_b); d4.append(rr_b)

t4 = Table(d4, colWidths=[2.8*cm]+[2.6*cm]*n_rob)
n4 = len(d4)
panel_b4 = 9
t4.setStyle(std_table_style(d4, [
    ('LINEBELOW', (0,1), (-1,1), 0.5, colors.black),
    ('LINEABOVE', (0, panel_b4), (-1, panel_b4), 0.5, colors.black),
]))
elements += [Paragraph("表4\u3000稳健性检验", title_s), Spacer(1,4*mm), t4, Spacer(1,3*mm),
    Paragraph("注: 因变量均为PriceDelay。"
              "第(1)列替换行业\u00d7年份交互FE。第(2)列剔除样本最后一年(2022)。第(3)列剔除IT行业。"
              "第(4)列使用滞后一期处理变量X(t-1)。第(5)列为PSM匹配样本。第(6)列使用企业+年份双向聚类SE。", note_s),
    PageBreak()]

# ============================================================
# 表5: 机制检验
# ============================================================
print("表5")
mech_specs = [
    ('(1)', 'Analyst', '分析师关注\n(Analyst)'),
    ('(2)', 'Amihud', '非流动性\n(Amihud)'),
    ('(3)', 'RetVol', '收益率波动\n(RetVol)'),
    ('(4)', 'Turnover', '换手率\n(Turnover)'),
    ('(5)', 'InstHold', '机构持股\n(InstHold)'),
]
n_mech = len(mech_specs)

d5 = []
d5.append([PL('')] + [P(label) for _,_,label in mech_specs])
d5.append([PL('变量')] + [P(num) for num,_,_ in mech_specs])

d5.append([PL('Panel A: D = DU_llm')] + [P('')]*n_mech)
cr5, sr5 = [PI('DU_llm')], [PI('')]
for _, ch, _ in mech_specs:
    r = mech.get(f'{ch}_LLM', {})
    if r:
        cr5.append(P(f"{r['coef']:.4f}{stars(r['p'])}"))
        sr5.append(P(f"({r['se']:.4f})"))
    else: cr5.append(P('')); sr5.append(P(''))
d5.append(cr5); d5.append(sr5)

d5.append([PL('Firm/Year')] + [P('YES')]*n_mech)
d5.append([PL('控制变量')] + [P('YES')]*n_mech)
nr5a = [PL('N')]
rr5a = [PL('R<super>2</super>')]
for _, ch, _ in mech_specs:
    r = mech.get(f'{ch}_LLM', {})
    nr5a.append(P(f"{r.get('N',0)}") if r else P(''))
    rr5a.append(P(f"{r.get('R2',0):.3f}") if r else P(''))
d5.append(nr5a); d5.append(rr5a)

d5.append([PL('Panel B: D = DU_kw')] + [P('')]*n_mech)
cr5b, sr5b = [PI('DU_kw')], [PI('')]
for _, ch, _ in mech_specs:
    r = mech.get(f'{ch}_KW', {})
    if r:
        cr5b.append(P(f"{r['coef']:.4f}{stars(r['p'])}"))
        sr5b.append(P(f"({r['se']:.4f})"))
    else: cr5b.append(P('')); sr5b.append(P(''))
d5.append(cr5b); d5.append(sr5b)

d5.append([PL('Firm/Year')] + [P('YES')]*n_mech)
d5.append([PL('控制变量')] + [P('YES')]*n_mech)
nr5b = [PL('N')]
rr5b = [PL('R<super>2</super>')]
for _, ch, _ in mech_specs:
    r = mech.get(f'{ch}_KW', {})
    nr5b.append(P(f"{r.get('N',0)}") if r else P(''))
    rr5b.append(P(f"{r.get('R2',0):.3f}") if r else P(''))
d5.append(nr5b); d5.append(rr5b)

t5 = Table(d5, colWidths=[3.2*cm]+[3*cm]*n_mech)
n5 = len(d5)
panel_b5 = 9
t5.setStyle(std_table_style(d5, [
    ('LINEBELOW', (0,1), (-1,1), 0.5, colors.black),
    ('LINEABOVE', (0, panel_b5), (-1, panel_b5), 0.5, colors.black),
]))
elements += [Paragraph("表5\u3000作用机制检验", title_s), Spacer(1,4*mm), t5, Spacer(1,3*mm),
    Paragraph("注: 每列因变量为对应机制变量M。"
              "DU_llm在全部5个渠道显著: 分析师关注增加(+)、非流动性降低(-)、"
              "波动率降低(-)、换手率降低(-)、机构持股降低(-)。"
              "DU_kw在Amihud(p=0.166)和Turnover(p=0.143)上不显著。", note_s),
    PageBreak()]

# ============================================================
# 表6: 异质性检验 (组间差异P值移至底部)
# ============================================================
print("表6")
het_cols = [
    ('(1) 国有\n企业', 'SOE', 'SOE=1'),
    ('(2) 非国有\n企业', 'SOE', 'SOE=0'),
    ('(3) 大\n企业', 'Size_median', 'Large'),
    ('(4) 小\n企业', 'Size_median', 'Small'),
    ('(5) IT\n行业', 'Industry', 'IT'),
    ('(6) 非IT\n行业', 'Industry', 'Non-IT'),
]
n_het = len(het_cols)

d6 = []
d6.append([PL('')] + [P(label) for label,_,_ in het_cols])
d6.append([PL('变量')] + [P('PriceDelay')]*n_het)

# Panel A
d6.append([PL('Panel A: D = DU_llm')] + [P('')]*n_het)
cr6, sr6 = [PI('DU_llm')], [PI('')]
for _, grp, sub in het_cols:
    r = hetero.get(f'sub_{grp}_{sub}_LLM', {})
    if r:
        cr6.append(P(f"{r['coef']:.4f}{stars(r['p'])}"))
        sr6.append(P(f"({r['se']:.4f})"))
    else: cr6.append(P('')); sr6.append(P(''))
d6.append(cr6); d6.append(sr6)

d6.append([PL('Firm/Year')] + [P('YES')]*n_het)
d6.append([PL('控制变量')] + [P('YES')]*n_het)
nr6a = [PL('N')]
rr6a = [PL('R<super>2</super>')]
for _, grp, sub in het_cols:
    r = hetero.get(f'sub_{grp}_{sub}_LLM', {})
    nr6a.append(P(f"{r.get('N',0)}") if r else P(''))
    rr6a.append(P(f"{r.get('R2',0):.3f}") if r else P(''))
d6.append(nr6a); d6.append(rr6a)

# Panel A 组间差异检验P值 (at bottom of Panel A, after N/R2)
pval_a = [PL('组间差异检验P值')]
for i in range(0, n_het, 2):
    _, grp, _ = het_cols[i]
    ir = hetero.get(f'interact_{grp}_LLM', {})
    ip = ir.get('interact_p', '')
    if ip != '':
        pval_a.append(P(f"{ip:.3f}{stars(ip)}"))
        pval_a.append(P(''))
    else:
        pval_a.append(P('')); pval_a.append(P(''))
d6.append(pval_a)

# Panel B
d6.append([PL('Panel B: D = DU_kw')] + [P('')]*n_het)
cr6b, sr6b = [PI('DU_kw')], [PI('')]
for _, grp, sub in het_cols:
    r = hetero.get(f'sub_{grp}_{sub}_KW', {})
    if r:
        cr6b.append(P(f"{r['coef']:.4f}{stars(r['p'])}"))
        sr6b.append(P(f"({r['se']:.4f})"))
    else: cr6b.append(P('')); sr6b.append(P(''))
d6.append(cr6b); d6.append(sr6b)

d6.append([PL('Firm/Year')] + [P('YES')]*n_het)
d6.append([PL('控制变量')] + [P('YES')]*n_het)
nr6b = [PL('N')]
rr6b = [PL('R<super>2</super>')]
for _, grp, sub in het_cols:
    r = hetero.get(f'sub_{grp}_{sub}_KW', {})
    nr6b.append(P(f"{r.get('N',0)}") if r else P(''))
    rr6b.append(P(f"{r.get('R2',0):.3f}") if r else P(''))
d6.append(nr6b); d6.append(rr6b)

pval_b = [PL('组间差异检验P值')]
for i in range(0, n_het, 2):
    _, grp, _ = het_cols[i]
    ir = hetero.get(f'interact_{grp}_KW', {})
    ip = ir.get('interact_p', '')
    if ip != '':
        pval_b.append(P(f"{ip:.3f}{stars(ip)}"))
        pval_b.append(P(''))
    else:
        pval_b.append(P('')); pval_b.append(P(''))
d6.append(pval_b)

t6 = Table(d6, colWidths=[2.8*cm]+[2.6*cm]*n_het)
n6 = len(d6)
panel_b6 = 10  # Panel B header row
t6.setStyle(std_table_style(d6, [
    ('LINEBELOW', (0,1), (-1,1), 0.5, colors.black),
    ('LINEABOVE', (0, panel_b6), (-1, panel_b6), 0.5, colors.black),
]))
elements += [Paragraph("表6\u3000异质性检验", title_s), Spacer(1,4*mm), t6, Spacer(1,3*mm),
    Paragraph("注: 因变量均为PriceDelay。"
              "国有/非国有按产权性质划分，大/小企业按Size年度中位数划分，IT/非IT按证监会行业代码I类划分。"
              "组间差异检验P值基于全样本交互项回归。"
              "SOE组间差异不显著(p=0.956)，表明效应不受产权性质影响;"
              "企业规模组间差异显著(p=0.001)，大企业效应更强。", note_s)]

# Build PDF
doc = SimpleDocTemplate(f"{OUT}/regression_tables_v16.pdf", pagesize=landscape(A4),
    topMargin=1.5*cm, bottomMargin=1.5*cm, leftMargin=1.5*cm, rightMargin=1.5*cm)
doc.build(elements)
print(f"\nPDF: {OUT}/regression_tables_v16.pdf")
print("Done!")
