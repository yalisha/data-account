"""
v17表格PDF: 严格对标朱康(2025)会计研究格式
三线表, 递进式基准回归, 横排稳健性, 分组列异质性+Fisher P
"""
import os, warnings
warnings.filterwarnings('ignore')

from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.units import cm, mm
from reportlab.lib import colors
from reportlab.platypus import (SimpleDocTemplate, Table, TableStyle, Paragraph,
                                Spacer, PageBreak, Image, KeepTogether)
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase.pdfmetrics import registerFontFamily

BASE = "/Users/mac/computerscience/15会计研究"
OUT = f"{BASE}/results/v17_tables"
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

P = lambda t: Paragraph(str(t), s_c)
L = lambda t: Paragraph(str(t), s_l)


def three_line(n_rows, hdr=1, spans=None):
    """朱康格式三线表样式"""
    sty = [
        ('FONTNAME', (0,0), (-1,-1), CN),
        ('FONTSIZE', (0,0), (-1,-1), 8),
        ('ALIGN', (1,0), (-1,-1), 'CENTER'),
        ('ALIGN', (0,0), (0,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('LINEABOVE', (0,0), (-1,0), 1.0, colors.black),
        ('LINEBELOW', (0, hdr-1), (-1, hdr-1), 0.5, colors.black),
        ('LINEBELOW', (0, n_rows-1), (-1, n_rows-1), 1.0, colors.black),
        ('TOPPADDING', (0,0), (-1,-1), 1.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 1.5),
    ]
    if spans:
        sty.extend(spans)
    return TableStyle(sty)


# ================================================================
# 表1: 描述性统计
# ================================================================
print("表1")
rows1 = [
    ['变量', '观测值', '均值', '标准差', '最小值', '中位数', '最大值'],
    ['PriceDelay', '43,735', '0.112', '0.123', '0.005', '0.068', '0.665'],
    ['SYNCH', '43,677', '\u22120.505', '0.834', '\u22122.954', '\u22120.449', '1.229'],
    ['DU_kw', '43,735', '1.219', '1.861', '0.000', '0.572', '11.064'],
    ['DU_llm', '43,735', '1.321', '1.119', '0.000', '1.086', '6.799'],
    ['Size', '43,735', '22.597', '0.971', '20.948', '22.435', '25.608'],
    ['Lev', '43,735', '0.425', '0.207', '0.055', '0.417', '0.916'],
    ['ROA', '43,735', '0.030', '0.067', '\u22120.280', '0.033', '0.191'],
    ['TobinQ', '43,735', '2.005', '1.283', '0.829', '1.590', '8.451'],
    ['Age', '43,735', '2.159', '0.733', '0.693', '2.303', '3.219'],
    ['Growth', '43,735', '0.134', '0.369', '\u22120.591', '0.083', '2.157'],
    ['IndepRatio', '43,735', '0.378', '0.054', '0.333', '0.364', '0.571'],
    ['Dual', '43,735', '0.297', '0.457', '0.000', '0.000', '1.000'],
    ['Top1Share', '43,735', '33.420', '14.822', '8.126', '31.000', '74.000'],
    ['SOE', '43,735', '0.329', '0.470', '0.000', '0.000', '1.000'],
    ['CFO', '43,735', '0.046', '0.068', '\u22120.161', '0.045', '0.239'],
]
d1 = [[L(r[0])] + [P(c) for c in r[1:]] for r in rows1]
t1 = Table(d1, colWidths=[2.2*cm]+[2.3*cm]*6)
t1.setStyle(three_line(len(d1)))
title1 = Paragraph("表1 变量的描述性统计", s_title)
note1 = Paragraph("注：所有连续变量在1%和99%分位数处进行缩尾处理。DU<sub>kw</sub>为每万字关键词频率，DU<sub>llm</sub>=ln(1+kw_total)\u00d7(LLM<sub>score</sub>/3)。", s_note)

# ================================================================
# 表2: 基准回归 (朱康格式: 递进式, 含全部控制变量系数)
# ================================================================
print("表2")
import json as _json
with open(f"{BASE}/results/v17_tables/ols_stata_sample.json") as _f:
    _ols = _json.load(_f)

def _fc(col, var, digits=4):
    """Format coefficient with stars."""
    d = _ols[col].get(var)
    if not d: return ('', '')
    v = d['coef']; p = d['p']
    s = '***' if p<0.01 else ('**' if p<0.05 else ('*' if p<0.1 else ''))
    sign = '\u2212' if v < 0 else ''
    coef = f'{sign}{abs(v):.{digits}f}{s}'
    se = f'({abs(d["se"]):.{digits}f})'
    return (coef, se)

_ctrls = ['Size','Lev','ROA','TobinQ','Age','Growth','IndepRatio','Dual','Top1Share','SOE','CFO']
_ctrl_sub = {'Top1Share': 'Top1<sub>Share</sub>'}

d2 = [
    [L(''), P('(1)'), P('(2)'), P('(3)'), P('(4)'), P('(5)')],
    [L('变量'), P('PriceDelay'), P('PriceDelay'), P('PriceDelay'), P('PriceDelay'), P('PriceDelay')],
    [L('DU<sub>kw</sub>'), P('\u22120.0052***'), P('\u22120.0046***'), P(''), P(''), P('')],
    [L(''), P('(0.0010)'), P('(0.0009)'), P(''), P(''), P('')],
    [L('DU<sub>llm</sub>'), P(''), P(''), P('\u22120.0037***'), P('\u22120.0036***'), P('')],
    [L(''), P(''), P(''), P('(0.0008)'), P('(0.0008)'), P('')],
    [L('LLM<sub>binary</sub>'), P(''), P(''), P(''), P(''), P('\u22120.0033***')],
    [L(''), P(''), P(''), P(''), P(''), P('(0.0013)')],
]
# Add control variable rows
for cv in _ctrls:
    label = _ctrl_sub.get(cv, cv)
    coefs = []
    ses = []
    for col in ['(1)','(2)','(3)','(4)','(5)']:
        c, s = _fc(col, cv)
        coefs.append(P(c))
        ses.append(P(s))
    d2.append([L(label)] + coefs)
    d2.append([L('')] + ses)

d2 += [
    [L('Firm/Year'), P('YES'), P('YES'), P('YES'), P('YES'), P('YES')],
    [L('N'), P('43,721'), P('43,721'), P('43,721'), P('43,721'), P('43,721')],
    [L('R\u00b2'), P('0.409'), P('0.419'), P('0.408'), P('0.418'), P('0.418')],
]
t2 = Table(d2, colWidths=[2.5*cm]+[2.5*cm]*5)
t2.setStyle(three_line(len(d2), hdr=2, spans=[('LINEBELOW', (0,1), (-1,1), 0.5, colors.black)]))
title2 = Paragraph("表2 基准回归检验", s_title)
note2 = Paragraph(
    "注：括号内数值为行业\u00d7年份层面聚类调整后的稳健标准误；"
    "***、**、*分别表示在1%、5%、10%的水平上显著，下同。",
    s_note)

# ================================================================
# 表3: DML-PLR (8规格)
# ================================================================
print("表3")
d3 = [
    [L(''), P('Y=PriceDelay'), P(''), P(''), P('Y=SYNCH'), P(''), P('')],
    [L('处理变量D'), P('系数'), P('标准误'), P('t值'), P('系数'), P('标准误'), P('t值')],
    [L('DU<sub>kw</sub>'), P('\u22120.0029***'), P('(0.0004)'), P('[\u22126.50]'), P('+0.0277***'), P('(0.0029)'), P('[9.65]')],
    [L('DU<sub>kw_ln</sub>'), P('\u22120.0040***'), P('(0.0006)'), P('[\u22127.19]'), P(''), P(''), P('')],
    [L('DU<sub>sub_ln</sub>'), P('\u22120.0039***'), P('(0.0005)'), P('[\u22127.21]'), P(''), P(''), P('')],
    [L('DU<sub>llm</sub>'), P('\u22120.0042***'), P('(0.0007)'), P('[\u22126.35]'), P('+0.0326***'), P('(0.0041)'), P('[7.98]')],
    [L('LLM<sub>binary</sub>'), P('\u22120.0042***'), P('(0.0012)'), P('[\u22123.46]'), P('+0.0435***'), P('(0.0075)'), P('[5.80]')],
    [L('控制变量+CRE'), P('YES'), P(''), P(''), P('YES'), P(''), P('')],
    [L('N'), P('35,666'), P(''), P(''), P('35,614'), P(''), P('')],
]
t3 = Table(d3, colWidths=[2.5*cm]+[2*cm]*6)
t3.setStyle(three_line(len(d3), hdr=2, spans=[
    ('LINEBELOW', (0,1), (-1,1), 0.5, colors.black),
    ('SPAN', (1,0), (3,0)), ('SPAN', (4,0), (6,0)),
]))
title3 = Paragraph("表3\u3000DML-PLR回归结果", s_title)
note3 = Paragraph(
    "注：DML-PLR框架，LightGBM为第一阶段学习器，5折交叉拟合，Mundlak均值替代企业固定效应，27个控制变量。"
    "括号内为标准误，方括号内为t统计量。",
    s_note)

# ================================================================
# 表4: 内生性检验
# ================================================================
print("表4")
d4 = [
    [L(''), P('(1)'), P('(2)'), P('(3)'), P('(4)'), P('(5)')],
    [L('模型'), P('OLS基准'), P('IV1:跨省peer'), P('IV2:数字化'), P('滞后OLS'), P('滞后IV')],
    [L('DU<sub>kw</sub>'), P('\u22120.0046***'), P('\u22120.0143***'), P('\u22120.0352***'), P('\u22120.0039***'), P('\u22120.0163***')],
    [L(''), P('(0.0009)'), P('(0.0047)'), P('(0.0114)'), P('(0.0010)'), P('(0.0053)')],
    [L('First-stage F'), P(''), P('251.3'), P('72.7'), P(''), P('177.8')],
    [L('控制变量'), P('YES'), P('YES'), P('YES'), P('YES'), P('YES')],
    [L('Firm/Year'), P('YES'), P('YES'), P('YES'), P('YES'), P('YES')],
    [L('N'), P('43,695'), P('43,612'), P('43,695'), P('37,278'), P('37,198')],
    [L(''), P(''), P(''), P(''), P(''), P('')],
    [L('辅助检验'), P(''), P(''), P(''), P(''), P('')],
    [L('Oster \u03b4*(1.3R\u00b2)'), P('96.3'), P(''), P(''), P(''), P('')],
    [L('Oster \u03b4*(保守)'), P('7.5'), P(''), P(''), P(''), P('')],
    [L('Heckman: DU<sub>kw</sub>'), P('\u22120.0040***'), P('t=\u22124.39'), P(''), P(''), P('')],
    [L('Heckman: IMR'), P('\u22120.123***'), P('t=\u22126.23'), P(''), P(''), P('')],
    [L('安慰剂: DU\u1D62,\u1D57'), P('\u22120.0050***'), P('t=\u22124.64'), P(''), P(''), P('')],
    [L('安慰剂: DU\u1D62,\u1D57\u208A\u2081'), P('0.0001'), P('t=0.11'), P(''), P(''), P('')],
]
t4 = Table(d4, colWidths=[3*cm]+[2.5*cm]*5)
n4 = len(d4)
t4.setStyle(three_line(n4, hdr=2, spans=[
    ('LINEBELOW', (0,1), (-1,1), 0.5, colors.black),
    ('LINEABOVE', (0,9), (-1,9), 0.5, colors.black),
    ('LINEBELOW', (0,9), (-1,9), 0.5, colors.black),
]))
title4 = Paragraph("表4\u3000内生性检验", s_title)
note4 = Paragraph(
    "注：IV1为同行业跨省企业DU_kw留一均值，IV2为省级大数据发展指数(2016)\u00d7year。"
    "DWH检验均拒绝外生性(p<0.02)。括号内为聚类标准误。",
    s_note)

# ================================================================
# 表5: 稳健性检验 (朱康格式: 横排, 每种方法一列)
# ================================================================
print("表5")
d5 = [
    [L(''), P('(1)替换'), P('(2)剔除'), P('(3)剔除'), P('(4)滞后'), P('(5)倾向'), P('(6)双向'), P('(7)替换')],
    [L(''), P('固定效应'), P('末期年份'), P('IT行业'), P('控制变量'), P('得分匹配'), P('聚类SE'), P('自变量')],
    [L('变量'), P('PriceDelay'), P('PriceDelay'), P('PriceDelay'), P('PriceDelay'), P('PriceDelay'), P('PriceDelay'), P('PriceDelay')],
    [L('DU<sub>kw</sub>'), P('\u22120.0029***'), P('\u22120.0052***'), P('\u22120.0046***'), P('\u22120.0040***'), P('\u22120.0039***'), P('\u22120.0046***'), P('')],
    [L('(DU<sub>sub_ln</sub>)'), P(''), P(''), P(''), P(''), P(''), P(''), P('\u22120.0042***')],
    [L(''), P('(0.0006)'), P('(0.0010)'), P('(0.0009)'), P('(0.0009)'), P('(0.0008)'), P('(0.0012)'), P('(0.0008)')],
    [L('控制变量'), P('YES'), P('YES'), P('YES'), P('YES'), P('YES'), P('YES'), P('YES')],
    [L('Firm/Year'), P('YES'), P('YES'), P('YES'), P('YES'), P('YES'), P('YES'), P('YES')],
    [L('Ind\u00d7Year'), P('YES'), P('NO'), P('NO'), P('NO'), P('NO'), P('NO'), P('NO')],
    [L('N'), P('43,656'), P('38,812'), P('40,613'), P('37,294'), P('35,929'), P('43,721'), P('43,721')],
    [L('R\u00b2'), P('0.472'), P('0.428'), P('0.419'), P('0.416'), P('0.434'), P('0.419'), P('0.418')],
]
t5 = Table(d5, colWidths=[2.2*cm]+[1.9*cm]*7)
t5.setStyle(three_line(len(d5), hdr=3, spans=[
    ('LINEBELOW', (0,2), (-1,2), 0.5, colors.black),
]))
title5 = Paragraph("表5\u3000稳健性检验", s_title)
note5 = Paragraph(
    "注：第(1)列替换为行业\u00d7年份联合固定效应；第(5)列采用倾向得分匹配法(Rosenbaum和Rubin, 1983)，"
    "以DU_kw高于行业-年份中位数为处理组，Logit估计倾向得分，最近邻1:1匹配(卡尺0.05)；"
    "第(6)列在企业和年份层面双向聚类(Cameron等, 2011)；第(7)列以实质利用指标DU_sub_ln替代DU_kw。",
    s_note)

# ================================================================
# 图1: PSM核密度图
# ================================================================
psm_path = f"{BASE}/results/v16_tables/psm_density.png"

# ================================================================
# 表6: 传导机制检验 (朱康格式: 渠道变量为因变量)
# ================================================================
print("表6")
d6 = [
    [L(''), P('(1)'), P('(2)'), P('(3)'), P('(4)'), P('(5)'), P('(6)')],
    [L('变量'), P('Analyst'), P('Analyst'), P('Amihud'), P('Amihud'), P('RetVol'), P('RetVol')],
    [L('DU<sub>kw</sub>'), P('+0.0272***'), P(''), P('\u22120.0011***'), P(''), P('\u22120.0000'), P('')],
    [L(''), P('(0.0082)'), P(''), P('(0.0003)'), P(''), P('(0.0001)'), P('')],
    [L('DU<sub>llm</sub>'), P(''), P('+0.0635***'), P(''), P('\u22120.0006***'), P(''), P('\u22120.0002***')],
    [L(''), P(''), P('(0.0067)'), P(''), P('(0.0002)'), P(''), P('(0.0000)')],
    [L('控制变量'), P('YES'), P('YES'), P('YES'), P('YES'), P('YES'), P('YES')],
    [L('Firm/Year'), P('YES'), P('YES'), P('YES'), P('YES'), P('YES'), P('YES')],
    [L('N'), P('43,721'), P('43,721'), P('43,721'), P('43,721'), P('43,721'), P('43,721')],
    [L('R\u00b2'), P('0.793'), P('0.793'), P('0.564'), P('0.564'), P('0.660'), P('0.660')],
]
t6 = Table(d6, colWidths=[2.2*cm]+[2.1*cm]*6)
t6.setStyle(three_line(len(d6), hdr=2, spans=[('LINEBELOW', (0,1), (-1,1), 0.5, colors.black)]))
title6 = Paragraph("表6 传导机制检验", s_title)
note6 = Paragraph(
    "注：因变量分别为Analyst(ln(1+分析师人数))、Amihud(非流动性指标)和RetVol(日收益率年度标准差)。"
    "渠道变量经1%/99%缩尾处理。分析师覆盖和流动性在双测度下均显著，为稳健渠道。",
    s_note)

# ================================================================
# 表7: 异质性检验 (朱康格式: 分组列+Fisher P)
# ================================================================
print("表7")
d7 = [
    [L(''), P('(1)高'), P('(2)低'), P('(3)国有'), P('(4)非国有'), P('(5)高'), P('(6)低')],
    [L('变量'), P('分析师'), P('分析师'), P('企业'), P('企业'), P('股东分散'), P('股东分散')],
    [L(''), P('PriceDelay'), P('PriceDelay'), P('PriceDelay'), P('PriceDelay'), P('PriceDelay'), P('PriceDelay')],
    [L('DU<sub>kw</sub>'), P('\u22120.0062***'), P('\u22120.0033***'), P('\u22120.0064***'), P('\u22120.0039***'), P('\u22120.0053***'), P('\u22120.0016')],
    [L(''), P('(0.0011)'), P('(0.0010)'), P('(0.0012)'), P('(0.0010)'), P('(0.0010)'), P('(0.0010)')],
    [L('控制变量'), P('YES'), P('YES'), P('YES'), P('YES'), P('YES'), P('YES')],
    [L('Firm/Year'), P('YES'), P('YES'), P('YES'), P('YES'), P('YES'), P('YES')],
    [L('N'), P('21,861'), P('21,860'), P('14,393'), P('29,328'), P('21,861'), P('21,860')],
    [L('R\u00b2'), P('0.468'), P('0.419'), P('0.452'), P('0.416'), P('0.440'), P('0.437')],
    [L('组间差异检验P值'), P('0.018**'), P(''), P('0.040**'), P(''), P('0.024**'), P('')],
]
t7 = Table(d7, colWidths=[3*cm]+[2.1*cm]*6)
n7 = len(d7)
t7.setStyle(three_line(n7, hdr=3, spans=[
    ('LINEBELOW', (0,2), (-1,2), 0.5, colors.black),
    ('SPAN', (1, n7-1), (2, n7-1)),
    ('SPAN', (3, n7-1), (4, n7-1)),
    ('SPAN', (5, n7-1), (6, n7-1)),
]))
title7 = Paragraph("表7\u3000异质性检验", s_title)
note7 = Paragraph(
    "注：分析师覆盖和股东人数均以中位数分组。组间差异检验采用费舍尔组合检验(1000次抽样)。",
    s_note)

# ================================================================
# Build PDF
# ================================================================
print("PDF...")
doc = SimpleDocTemplate(
    f"{OUT}/regression_tables_v17.pdf",
    pagesize=A4,
    topMargin=2.5*cm, bottomMargin=2*cm,
    leftMargin=2.5*cm, rightMargin=2.5*cm,
)

elements = []
sp = Spacer(1, 3*mm)

elements += [title1, sp, t1, sp, note1, PageBreak()]
elements += [title2, sp, t2, sp, note2, PageBreak()]
elements += [title3, sp, t3, sp, note3, PageBreak()]
elements += [title4, sp, t4, sp, note4, PageBreak()]
elements += [title5, sp, t5, sp, note5]
if os.path.exists(psm_path):
    elements += [Spacer(1, 6*mm)]
    elements += [Paragraph("图1\u3000匹配前后核密度曲线", s_title)]
    elements += [Spacer(1, 2*mm)]
    elements += [Image(psm_path, width=14*cm, height=5.5*cm)]
    elements += [Spacer(1, 2*mm)]
    elements += [Paragraph("注：处理组定义为DU<sub>kw</sub>高于行业-年份中位数，Logit估计倾向得分，最近邻1:1匹配(卡尺0.05)。", s_note)]
elements += [PageBreak()]
elements += [title6, sp, t6, sp, note6, PageBreak()]
elements += [title7, sp, t7, sp, note7]

doc.build(elements)
print(f"\nDone: {OUT}/regression_tables_v17.pdf")
