import json
import os
import re
import warnings

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.pdfmetrics import registerFontFamily
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

warnings.filterwarnings("ignore")

BASE = "/Users/mac/computerscience/0做完了/15会计研究"
OUT = os.path.join(BASE, "results", "v15_current_tables")
os.makedirs(OUT, exist_ok=True)

CN = "Helvetica"
SONGTI = "/System/Library/Fonts/Supplemental/Songti.ttc"
if os.path.exists(SONGTI):
    try:
        pdfmetrics.registerFont(TTFont("SongtiSC", SONGTI, subfontIndex=6))
        pdfmetrics.registerFont(TTFont("SongtiSC-Bold", SONGTI, subfontIndex=1))
        registerFontFamily("SongtiSC", normal="SongtiSC", bold="SongtiSC-Bold")
        CN = "SongtiSC"
    except Exception:
        pass

s_title = ParagraphStyle(
    "title",
    fontName=CN,
    fontSize=10.5,
    alignment=TA_CENTER,
    spaceAfter=4,
    spaceBefore=6,
    leading=14,
)
s_note = ParagraphStyle("note", fontName=CN, fontSize=7.5, alignment=TA_LEFT, leading=10)
s_c = ParagraphStyle("center", fontName=CN, fontSize=8, alignment=TA_CENTER, leading=10)
s_l = ParagraphStyle("left", fontName=CN, fontSize=8, alignment=TA_LEFT, leading=10)

VAR_REPLACEMENTS = [
    ("DU_llm_lenstd_lag", "DU<sub>llm,lenstd</sub><sub>lag</sub>"),
    ("DU_kw_strict_lag", "DU<sub>kw,strict</sub><sub>lag</sub>"),
    ("DU_kw_resid_lag", "DU<sub>kw,resid</sub><sub>lag</sub>"),
    ("DUchain_count_lag", "DU<sub>chain,count</sub><sub>lag</sub>"),
    ("DUclosedloop_lag", "DU<sub>closedloop</sub><sub>lag</sub>"),
    ("DUkw_mda_lag", "DU<sub>kw,mda</sub><sub>lag</sub>"),
    ("IndPen_mean_z_lag", "IndPen<sub>mean,z</sub><sub>lag</sub>"),
    ("DU_kw_z_lag", "DU<sub>kw,z</sub><sub>lag</sub>"),
    ("BroadShallow_lag", "BroadShallow<sub>lag</sub>"),
    ("WashGap_lag", "WashGap<sub>lag</sub>"),
    ("GenericNarr_lag", "GenericNarr<sub>lag</sub>"),
    ("ln_total_chars_lag", "ln(TotalChars)<sub>lag</sub>"),
    ("ln_mda_chars_lag", "ln(MDAChars)<sub>lag</sub>"),
    ("DUevent_lag", "DU<sub>event</sub><sub>lag</sub>"),
    ("DUcore_lag", "DU<sub>core</sub><sub>lag</sub>"),
    ("DU_kw_lag", "DU<sub>kw</sub><sub>lag</sub>"),
    ("DU_llm_lag", "DU<sub>llm</sub><sub>lag</sub>"),
    ("DU_kw_z", "DU<sub>kw,z</sub>"),
    ("IndPen_mean_z", "IndPen<sub>mean,z</sub>"),
    ("DU_kw_ln", "DU<sub>kw,ln</sub>"),
    ("DU_sub_ln", "DU<sub>sub,ln</sub>"),
    ("llm_binary", "LLM<sub>binary</sub>"),
    ("LLMbinary", "LLM<sub>binary</sub>"),
    ("DU_llm", "DU<sub>llm</sub>"),
    ("DU_kw", "DU<sub>kw</sub>"),
    ("DUkw_mda", "DU<sub>kw,mda</sub>"),
    ("DUclosedloop", "DU<sub>closedloop</sub>"),
    ("DUchain_count", "DU<sub>chain,count</sub>"),
    ("DUevent", "DU<sub>event</sub>"),
    ("DUcore", "DU<sub>core</sub>"),
    ("DUkw", "DU<sub>kw</sub>"),
    ("DUllm", "DU<sub>llm</sub>"),
    ("data_stock", "data<sub>stock</sub>"),
    ("data_dev", "data<sub>dev</sub>"),
    ("data_app", "data<sub>app</sub>"),
    ("data_value", "data<sub>value</sub>"),
    ("data_gov", "data<sub>gov</sub>"),
]


def rich_text(text):
    text = str(text)
    for old, new in VAR_REPLACEMENTS:
        text = re.sub(rf"(?<![A-Za-z0-9]){re.escape(old)}(?![A-Za-z0-9])", new, text)
    text = text.replace("R²", "R<super>2</super>")
    text = text.replace("t-1", "t<super>-1</super>")
    return text


P = lambda t: Paragraph(rich_text(t), s_c)
L = lambda t: Paragraph(rich_text(t), s_l)


def stars(p):
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.1:
        return "*"
    return ""


def fmt_coef(coef, p=None, digits=4, force_sign=False):
    if coef is None or pd.isna(coef):
        return ""
    sign = "+" if force_sign and coef > 0 else ""
    return f"{sign}{coef:.{digits}f}{stars(p) if p is not None else ''}"


def fmt_se(se, digits=4):
    if se is None or pd.isna(se):
        return ""
    return f"({se:.{digits}f})"


def sig_from_t(t):
    at = abs(t)
    if at >= 2.58:
        return "***"
    if at >= 1.96:
        return "**"
    if at >= 1.645:
        return "*"
    return ""


def three_line(n_rows, hdr=1, extra=None):
    style = [
        ("FONTNAME", (0, 0), (-1, -1), CN),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("ALIGN", (0, 0), (0, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LINEABOVE", (0, 0), (-1, 0), 1.0, colors.black),
        ("LINEBELOW", (0, hdr - 1), (-1, hdr - 1), 0.5, colors.black),
        ("LINEBELOW", (0, n_rows - 1), (-1, n_rows - 1), 1.0, colors.black),
        ("TOPPADDING", (0, 0), (-1, -1), 1.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5),
    ]
    if extra:
        style.extend(extra)
    return TableStyle(style)


def make_table(rows, widths, hdr_rows, title, note, extra_style=None):
    parsed = [[L(r[0])] + [P(c) for c in r[1:]] for r in rows]
    tbl = Table(parsed, colWidths=widths, repeatRows=hdr_rows)
    tbl.setStyle(three_line(len(parsed), hdr=hdr_rows, extra=extra_style))
    return [
        Paragraph(rich_text(title), s_title),
        tbl,
        Spacer(1, 0.15 * cm),
        Paragraph(rich_text(note), s_note),
    ]


construct = json.load(open(os.path.join(BASE, "results", "v14", "construct_boundary_results.json")))
measure = json.load(open(os.path.join(BASE, "results", "v15_measurement", "measurement_upgrade_results.json")))
mech = pd.read_csv(os.path.join(BASE, "results", "v19", "table3_key_mechanisms_v19.csv"))
het = pd.read_csv(os.path.join(BASE, "results", "v19", "table4_heterogeneity_v19.csv"))
spill = pd.read_csv(os.path.join(BASE, "results", "v19", "table1_construct_boundary_v19.csv"))
wash = pd.read_csv(os.path.join(BASE, "results", "v19", "table2_measurement_washgap_v19.csv"))
het_appendix = json.load(open(os.path.join(BASE, "results", "v14", "heterogeneity_results.json")))

# 表1
rows1 = [
    ["维度", "关键词数", "示例"],
    ["数据存量 data_stock", "15", "大数据、数据库、数据中心"],
    ["数据开发能力 data_dev", "19", "数据挖掘、机器学习、数字化转型"],
    ["数据驱动应用 data_app", "19", "精准营销、智能推荐、风控模型"],
    ["数据价值变现 data_value", "11", "数据资产、数据交易、数据入表"],
    ["数据治理 data_gov", "9", "数据安全、数据隐私、数据合规"],
]

# 表2
rows2 = [
    ["变量", "定义"],
    ["Size", "总市值的自然对数"],
    ["Lev", "总负债/总资产"],
    ["ROA", "净利润/总资产"],
    ["TobinQ", "(股权市值+负债账面值)/总资产"],
    ["Age", "ln(当年-上市年份+1)"],
    ["Growth", "营收增长率"],
    ["IndepRatio", "独立董事人数/董事会总人数"],
    ["Dual", "董事长兼任CEO取1，否则为0"],
    ["Top1Share", "第一大股东持股比例"],
    ["SOE", "国有企业取1，否则为0"],
    ["CFO", "经营活动现金流/总资产"],
]

# 表3
rows3 = [
    ["渠道变量", "定义"],
    ["ForecastDisp 分析师预测分歧", "同一企业年度内至少3条盈利预测的标准差"],
    ["CashFlowVol 现金流波动率", "经营活动现金流/总资产的三年滚动标准差"],
    ["SCConc 综合供应链集中度", "前五大客户销售占比与前五大供应商采购占比的均值"],
]

# 表4
rows4 = [
    ["变量", "N", "均值", "标准差", "最小值", "中位数", "最大值"],
    ["PriceDelay", "43,735", "0.112", "0.123", "0.005", "0.068", "0.665"],
    ["SYNCH", "-43,677".replace("-", ""), "-0.505", "0.834", "-2.954", "-0.449", "1.229"],
    ["DU_kw", "43,735", "1.219", "1.861", "0.000", "0.572", "11.064"],
    ["DU_llm", "43,735", "1.321", "1.119", "0.000", "1.086", "6.799"],
    ["Size", "43,735", "22.597", "0.971", "20.948", "22.435", "25.608"],
    ["Lev", "43,735", "0.425", "0.207", "0.055", "0.417", "0.916"],
    ["ROA", "43,735", "0.030", "0.067", "-0.280", "0.033", "0.191"],
    ["TobinQ", "43,735", "2.005", "1.283", "0.829", "1.590", "8.451"],
    ["Age", "43,735", "2.159", "0.733", "0.693", "2.303", "3.219"],
    ["Growth", "43,735", "0.134", "0.369", "-0.591", "0.083", "2.157"],
    ["IndepRatio", "43,735", "0.378", "0.054", "0.333", "0.364", "0.571"],
    ["Dual", "43,735", "0.297", "0.457", "0.000", "0.000", "1.000"],
    ["Top1Share", "43,735", "33.420", "14.822", "8.126", "31.000", "74.000"],
    ["SOE", "43,735", "0.329", "0.470", "0.000", "0.000", "1.000"],
    ["CFO", "43,735", "0.046", "0.068", "-0.161", "0.045", "0.239"],
]

# 表5
rows5 = [
    ["", "(1)", "(2)", "(3)", "(4)", "(5)"],
    ["变量", "PriceDelay", "PriceDelay", "PriceDelay", "PriceDelay", "PriceDelay"],
    ["DU_kw", "-0.0052***", "-0.0046***", "", "", ""],
    ["", "(0.0010)", "(0.0009)", "", "", ""],
    ["DU_llm", "", "", "-0.0037***", "-0.0036***", ""],
    ["", "", "", "(0.0008)", "(0.0008)", ""],
    ["LLMbinary", "", "", "", "", "-0.0033***"],
    ["", "", "", "", "", "(0.0013)"],
    ["Size", "", "0.0063***", "", "0.0060**", "0.0055**"],
    ["", "", "(0.0024)", "", "(0.0024)", "(0.0024)"],
    ["Lev", "", "0.0225***", "", "0.0228***", "0.0225***"],
    ["", "", "(0.0063)", "", "(0.0063)", "(0.0063)"],
    ["ROA", "", "-0.0634***", "", "-0.0612***", "-0.0614***"],
    ["", "", "(0.0147)", "", "(0.0147)", "(0.0147)"],
    ["TobinQ", "", "0.0116***", "", "0.0117***", "0.0119***"],
    ["", "", "(0.0011)", "", "(0.0011)", "(0.0011)"],
    ["Age", "", "-0.0331***", "", "-0.0346***", "-0.0348***"],
    ["", "", "(0.0038)", "", "(0.0039)", "(0.0039)"],
    ["Growth", "", "0.0129***", "", "0.0130***", "0.0130***"],
    ["", "", "(0.0020)", "", "(0.0020)", "(0.0020)"],
    ["IndepRatio", "", "-0.0159", "", "-0.0149", "-0.0142"],
    ["", "", "(0.0155)", "", "(0.0155)", "(0.0155)"],
    ["Dual", "", "-0.0021", "", "-0.0021", "-0.0021"],
    ["", "", "(0.0016)", "", "(0.0016)", "(0.0016)"],
    ["Top1Share", "", "0.0000", "", "0.0000", "0.0000"],
    ["", "", "(0.0001)", "", "(0.0001)", "(0.0001)"],
    ["SOE", "", "0.0051", "", "0.0052", "0.0053"],
    ["", "", "(0.0040)", "", "(0.0040)", "(0.0040)"],
    ["CFO", "", "0.0047", "", "0.0051", "0.0055"],
    ["", "", "(0.0111)", "", "(0.0111)", "(0.0111)"],
    ["控制变量", "NO", "YES", "NO", "YES", "YES"],
    ["Firm FE", "YES", "YES", "YES", "YES", "YES"],
    ["Year FE", "YES", "YES", "YES", "YES", "YES"],
    ["N", "43,721", "43,721", "43,721", "43,721", "43,721"],
]

# 表6
rows6 = [
    ["", "(1)", "(2)", "(3)", "(4)", "(5)", "(6)", "(7)"],
    ["模型", "DUkw+Total", "DUkw+MDA", "DUllm+Total", "Lenstd+Total", "DUkw+Narr", "ResidDU", "StrictDU"],
    ["因变量", "PriceDelay", "PriceDelay", "PriceDelay", "PriceDelay", "PriceDelay", "PriceDelay", "PriceDelay"],
    [
        "DU_kw_lag",
        fmt_coef(construct["dukw_total_chars_lag"]["DU_kw_lag"]["coef"], construct["dukw_total_chars_lag"]["DU_kw_lag"]["p"]),
        fmt_coef(construct["dukw_mda_chars_lag"]["DU_kw_lag"]["coef"], construct["dukw_mda_chars_lag"]["DU_kw_lag"]["p"]),
        "",
        "",
        fmt_coef(construct["joint_lag"]["DU_kw_lag"]["coef"], construct["joint_lag"]["DU_kw_lag"]["p"]),
        "",
        "",
    ],
    [
        "",
        fmt_se(construct["dukw_total_chars_lag"]["DU_kw_lag"]["se"]),
        fmt_se(construct["dukw_mda_chars_lag"]["DU_kw_lag"]["se"]),
        "",
        "",
        fmt_se(construct["joint_lag"]["DU_kw_lag"]["se"]),
        "",
        "",
    ],
    [
        "DU_llm_lag",
        "",
        "",
        fmt_coef(construct["dullm_total_chars_lag"]["DU_llm_lag"]["coef"], construct["dullm_total_chars_lag"]["DU_llm_lag"]["p"]),
        "",
        "",
        "",
        "",
    ],
    ["", "", "", fmt_se(construct["dullm_total_chars_lag"]["DU_llm_lag"]["se"]), "", "", "", ""],
    [
        "DU_llm_lenstd_lag",
        "",
        "",
        "",
        fmt_coef(
            construct["dullm_lenstd_total_chars_lag"]["DU_llm_lenstd_lag"]["coef"],
            construct["dullm_lenstd_total_chars_lag"]["DU_llm_lenstd_lag"]["p"],
        ),
        "",
        "",
        "",
    ],
    ["", "", "", "", fmt_se(construct["dullm_lenstd_total_chars_lag"]["DU_llm_lenstd_lag"]["se"]), "", "", ""],
    [
        "ln_total_chars_lag",
        fmt_coef(
            construct["dukw_total_chars_lag"]["ln_total_chars_lag"]["coef"],
            construct["dukw_total_chars_lag"]["ln_total_chars_lag"]["p"],
        ),
        "",
        fmt_coef(
            construct["dullm_total_chars_lag"]["ln_total_chars_lag"]["coef"],
            construct["dullm_total_chars_lag"]["ln_total_chars_lag"]["p"],
        ),
        fmt_coef(
            construct["dullm_lenstd_total_chars_lag"]["ln_total_chars_lag"]["coef"],
            construct["dullm_lenstd_total_chars_lag"]["ln_total_chars_lag"]["p"],
        ),
        "",
        "",
        "",
    ],
    [
        "",
        fmt_se(construct["dukw_total_chars_lag"]["ln_total_chars_lag"]["se"]),
        "",
        fmt_se(construct["dullm_total_chars_lag"]["ln_total_chars_lag"]["se"]),
        fmt_se(construct["dullm_lenstd_total_chars_lag"]["ln_total_chars_lag"]["se"]),
        "",
        "",
        "",
    ],
    ["ln_mda_chars_lag", "", fmt_coef(construct["dukw_mda_chars_lag"]["ln_mda_chars_lag"]["coef"], construct["dukw_mda_chars_lag"]["ln_mda_chars_lag"]["p"]), "", "", "", "", ""],
    ["", "", fmt_se(construct["dukw_mda_chars_lag"]["ln_mda_chars_lag"]["se"]), "", "", "", "", ""],
    ["GenericNarr_lag", "", "", "", "", fmt_coef(construct["joint_lag"]["GenericNarr_lag"]["coef"], construct["joint_lag"]["GenericNarr_lag"]["p"]), "", ""],
    ["", "", "", "", "", fmt_se(construct["joint_lag"]["GenericNarr_lag"]["se"]), "", ""],
    ["DU_kw_resid_lag", "", "", "", "", "", fmt_coef(construct["dukw_resid_lag"]["DU_kw_resid_lag"]["coef"], construct["dukw_resid_lag"]["DU_kw_resid_lag"]["p"]), ""],
    ["", "", "", "", "", "", fmt_se(construct["dukw_resid_lag"]["DU_kw_resid_lag"]["se"]), ""],
    ["DU_kw_strict_lag", "", "", "", "", "", "", fmt_coef(construct["dukw_strict_lag"]["DU_kw_strict_lag"]["coef"], construct["dukw_strict_lag"]["DU_kw_strict_lag"]["p"])],
    ["", "", "", "", "", "", "", fmt_se(construct["dukw_strict_lag"]["DU_kw_strict_lag"]["se"])],
    ["控制变量", "YES", "YES", "YES", "YES", "YES", "YES", "YES"],
    ["Firm FE", "YES", "YES", "YES", "YES", "YES", "YES", "YES"],
    ["Year FE", "YES", "YES", "YES", "YES", "YES", "YES", "YES"],
    ["N", "37,294", "37,294", "37,294", "37,294", "37,294", "37,294", "37,294"],
]

# 表6A
m = measure["measurement_upgrade"]
rows6a = [
    ["", "(1)", "(2)", "(3)", "(4)", "(5)", "(6)"],
    ["模型", "DUevent", "DUchain", "DUclosed", "DUcore", "DUkw_mda", "DUclosed+Narr"],
    ["因变量", "PriceDelay", "PriceDelay", "PriceDelay", "PriceDelay", "PriceDelay", "PriceDelay"],
    ["DUevent_lag", fmt_coef(m["duevent_lag"]["DUevent_lag"]["coef"], m["duevent_lag"]["DUevent_lag"]["p"]), "", "", "", "", ""],
    ["", fmt_se(m["duevent_lag"]["DUevent_lag"]["se"]), "", "", "", "", ""],
    ["DUchain_count_lag", "", fmt_coef(m["duchain_count_lag"]["DUchain_count_lag"]["coef"], m["duchain_count_lag"]["DUchain_count_lag"]["p"]), "", "", "", ""],
    ["", "", fmt_se(m["duchain_count_lag"]["DUchain_count_lag"]["se"]), "", "", "", ""],
    [
        "DUclosedloop_lag",
        "",
        "",
        fmt_coef(m["duclosedloop_lag"]["DUclosedloop_lag"]["coef"], m["duclosedloop_lag"]["DUclosedloop_lag"]["p"]),
        "",
        "",
        fmt_coef(m["duclosedloop_joint_gn"]["DUclosedloop_lag"]["coef"], m["duclosedloop_joint_gn"]["DUclosedloop_lag"]["p"]),
    ],
    ["", "", "", fmt_se(m["duclosedloop_lag"]["DUclosedloop_lag"]["se"]), "", "", fmt_se(m["duclosedloop_joint_gn"]["DUclosedloop_lag"]["se"])],
    ["DUcore_lag", "", "", "", fmt_coef(m["ducore_lag"]["DUcore_lag"]["coef"], m["ducore_lag"]["DUcore_lag"]["p"]), "", ""],
    ["", "", "", "", fmt_se(m["ducore_lag"]["DUcore_lag"]["se"]), "", ""],
    ["DUkw_mda_lag", "", "", "", "", fmt_coef(m["dukw_mda_lag"]["DUkw_mda_lag"]["coef"], m["dukw_mda_lag"]["DUkw_mda_lag"]["p"]), ""],
    ["", "", "", "", "", fmt_se(m["dukw_mda_lag"]["DUkw_mda_lag"]["se"]), ""],
    ["GenericNarr_lag", "", "", "", "", "", fmt_coef(m["duclosedloop_joint_gn"]["GenericNarr_lag"]["coef"], m["duclosedloop_joint_gn"]["GenericNarr_lag"]["p"])],
    ["", "", "", "", "", "", fmt_se(m["duclosedloop_joint_gn"]["GenericNarr_lag"]["se"])],
    ["控制变量", "YES", "YES", "YES", "YES", "YES", "YES"],
    ["Firm FE", "YES", "YES", "YES", "YES", "YES", "YES"],
    ["Year FE", "YES", "YES", "YES", "YES", "YES", "YES"],
    ["N", "37,294", "37,294", "37,294", "37,294", "37,294", "37,294"],
]

# 表7
rows7 = [
    ["", "(1)", "(2)", "(3)", "(4)", "(5)"],
    ["模型", "OLS", "Lag OLS", "Lag IV", "IV1:Peer", "IV2:Dig×Year"],
    ["DU_kw", "-0.0046***", "-0.0039***", "-0.0163***", "-0.0143***", "-0.0352***"],
    ["", "(0.0009)", "(0.0010)", "(0.0053)", "(0.0047)", "(0.0114)"],
    ["First-stage F", "", "", "177.8", "251.3", "72.7"],
    ["控制变量", "YES", "YES", "YES", "YES", "YES"],
    ["Firm FE", "YES", "YES", "YES", "YES", "YES"],
    ["Year FE", "YES", "YES", "YES", "YES", "YES"],
    ["N", "43,695", "37,278", "37,198", "43,612", "43,695"],
]

# 表8
rows8 = [
    ["", "Y=PriceDelay", "", "", "Y=SYNCH", "", ""],
    ["处理变量D", "系数", "标准误", "t值", "系数", "标准误", "t值"],
    ["DU_kw", "-0.0029***", "(0.0004)", "[-6.50]", "+0.0277***", "(0.0029)", "[9.65]"],
    ["DU_kw_ln", "-0.0040***", "(0.0006)", "[-7.19]", "", "", ""],
    ["DU_sub_ln", "-0.0039***", "(0.0005)", "[-7.21]", "", "", ""],
    ["DU_llm", "-0.0042***", "(0.0007)", "[-6.35]", "+0.0326***", "(0.0041)", "[7.98]"],
    ["llm_binary", "-0.0042***", "(0.0012)", "[-3.46]", "+0.0435***", "(0.0075)", "[5.80]"],
    ["控制变量+CRE", "YES", "", "", "YES", "", ""],
    ["N", "35,666", "", "", "35,614", "", ""],
]

# 表9
rows9 = [
    ["", "(1)FE", "(2)尾年", "(3)IT", "(4)滞后", "(5)PSM", "(6)双聚类", "(7)替换"],
    ["", "行业×年FE", "剔除2024", "剔除IT", "滞后控制", "倾向匹配", "双向聚类", "DU_sub_ln"],
    ["变量", "PriceDelay", "PriceDelay", "PriceDelay", "PriceDelay", "PriceDelay", "PriceDelay", "PriceDelay"],
    ["DU_kw", "-0.0029***", "-0.0052***", "-0.0046***", "-0.0040***", "-0.0039***", "-0.0046***", ""],
    ["(DU_sub_ln)", "", "", "", "", "", "", "-0.0042***"],
    ["", "(0.0006)", "(0.0010)", "(0.0009)", "(0.0009)", "(0.0008)", "(0.0012)", "(0.0008)"],
    ["控制变量", "YES", "YES", "YES", "YES", "YES", "YES", "YES"],
    ["Firm FE", "YES", "YES", "YES", "YES", "YES", "YES", "YES"],
    ["Year FE", "NO", "YES", "YES", "YES", "YES", "YES", "YES"],
    ["Ind×Year FE", "YES", "NO", "NO", "NO", "NO", "NO", "NO"],
    ["N", "43,656", "38,812", "40,613", "37,294", "35,929", "43,721", "43,721"],
]

# 表10
rows10 = [
    ["", "(1)", "(2)", "(3)", "(4)", "(5)", "(6)"],
    ["变量", "ForecastDisp", "ForecastDisp", "CashFlowVol", "CashFlowVol", "SCConc", "SCConc"],
    ["DU_kw", "-0.0065***", "", "-0.0005**", "", "-0.277***", ""],
    ["", "(0.0016)", "", "(0.0002)", "", "(0.069)", ""],
    ["DU_llm", "", "+0.0010", "", "-0.0005**", "", "-0.460***"],
    ["", "", "(0.0017)", "", "(0.0002)", "", "(0.068)"],
    ["控制变量", "YES", "YES", "YES", "YES", "YES", "YES"],
    ["Firm FE", "YES", "YES", "YES", "YES", "YES", "YES"],
    ["Year FE", "YES", "YES", "YES", "YES", "YES", "YES"],
    ["N", "22,685", "22,685", "43,721", "43,721", "42,332", "42,332"],
]

# 表11
rows11 = [
    ["", "(1)", "(2)", "(3)"],
    ["模型", "WashGap", "BroadShallow", "联合回归"],
    ["因变量", "PriceDelay", "PriceDelay", "PriceDelay"],
    ["WashGap_lag", "+0.0035**", "", ""],
    ["", "(0.0017)", "", ""],
    ["BroadShallow_lag", "", "+0.0046", ""],
    ["", "", "(0.0036)", ""],
    ["DU_kw_lag", "", "", "-0.0016"],
    ["", "", "", "(0.0012)"],
    ["DU_llm_lenstd_lag", "", "", "-0.0146***"],
    ["", "", "", "(0.0039)"],
    ["控制变量", "YES", "YES", "YES"],
    ["Firm FE", "YES", "YES", "YES"],
    ["Year FE", "YES", "YES", "YES"],
    ["N", "37,294", "37,294", "37,294"],
]

# 表12
rows12 = [
    ["", "(1)", "(2)"],
    ["变量", "2011-2021", "2022-2024"],
    ["", "PriceDelay", "PriceDelay"],
    ["DU_kw", "-0.0066***", "-0.0018"],
    ["", "(0.0013)", "(0.0021)"],
    ["控制变量", "YES", "YES"],
    ["Firm FE", "YES", "YES"],
    ["Year FE", "YES", "YES"],
    ["N", "24,720", "12,240"],
    ["R²", "0.455", "0.557"],
    ["交互项P值", "0.031**", ""],
]

# 表13
rows13 = [
    ["", "(1)", "(2)", "(3)", "(4)"],
    ["维度", "高分析师关注", "低分析师关注", "主板公司", "非主板公司"],
    ["因变量", "PriceDelay", "PriceDelay", "PriceDelay", "PriceDelay"],
    ["DU_kw", "-0.0048***", "-0.0025*", "-0.0017", "-0.0042***"],
    ["", "(0.0011)", "(0.0014)", "(0.0017)", "(0.0010)"],
    ["控制变量", "YES", "YES", "YES", "YES"],
    ["Firm FE", "YES", "YES", "YES", "YES"],
    ["Year FE", "YES", "YES", "YES", "YES"],
    ["N", "18,912", "17,847", "15,900", "21,394"],
    ["交互项P值", "0.001***", "", "<0.001***", ""],
]

# 表14
rows14 = [
    ["", "(1)", "(2)", "(3)"],
    ["规格", "基准", "同期", "滞后"],
    ["因变量", "PriceDelay", "PriceDelay", "PriceDelay"],
    ["DU_kw_z", "-0.0086***", "-0.0067***", ""],
    ["", "(0.0017)", "(0.0013)", ""],
    ["DU_kw_z_lag", "", "", "-0.0070***"],
    ["", "", "", "(0.0020)"],
    ["IndPen_mean_z", "", "-0.0087**", ""],
    ["", "", "(0.0035)", ""],
    ["IndPen_mean_z_lag", "", "", "-0.0158***"],
    ["", "", "", "(0.0035)"],
    ["控制变量", "YES", "YES", "YES"],
    ["Firm FE", "YES", "YES", "YES"],
    ["Year FE", "YES", "YES", "YES"],
    ["N", "43,312", "43,312", "38,061"],
]

# 附录关键词长表
rows_kw = [
    ["维度", "序号", "关键词"],
    ["数据存量", "1", "大数据"],
    ["data_stock", "2", "数据库"],
    ["(15个)", "3", "数据中心"],
    ["", "4", "数据仓库"],
    ["", "5", "数据湖"],
    ["", "6", "数据集"],
    ["", "7", "数据存储"],
    ["", "8", "数据采集"],
    ["", "9", "数据资源"],
    ["", "10", "数据积累"],
    ["", "11", "用户数据"],
    ["", "12", "客户数据"],
    ["", "13", "行为数据"],
    ["", "14", "交易数据"],
    ["", "15", "运营数据"],
    ["数据开发能力", "16", "数据挖掘"],
    ["data_dev", "17", "数据分析"],
    ["(19个)", "18", "数据处理"],
    ["", "19", "数据清洗"],
    ["", "20", "数据建模"],
    ["", "21", "机器学习"],
    ["", "22", "深度学习"],
    ["", "23", "人工智能"],
    ["", "24", "算法"],
    ["", "25", "自然语言处理"],
    ["", "26", "数据科学"],
    ["", "27", "数据工程"],
    ["", "28", "数据平台"],
    ["", "29", "数据中台"],
    ["", "30", "数据架构"],
    ["", "31", "数字化转型"],
    ["", "32", "数字化"],
    ["", "33", "智能化"],
    ["", "34", "信息化"],
    ["数据驱动应用", "35", "数据驱动"],
    ["data_app", "36", "精准营销"],
    ["(19个)", "37", "个性化推荐"],
    ["", "38", "智能推荐"],
    ["", "39", "用户画像"],
    ["", "40", "风险控制"],
    ["", "41", "风控模型"],
    ["", "42", "智能决策"],
    ["", "43", "智能客服"],
    ["", "44", "智能制造"],
    ["", "45", "预测模型"],
    ["", "46", "需求预测"],
    ["", "47", "供应链优化"],
    ["", "48", "智慧物流"],
    ["", "49", "智慧城市"],
    ["", "50", "数字营销"],
    ["", "51", "程序化"],
    ["", "52", "数据赋能"],
    ["", "53", "数据服务"],
    ["数据价值变现", "54", "数据资产"],
    ["data_value", "55", "数据要素"],
    ["(11个)", "56", "数据交易"],
    ["", "57", "数据产品"],
    ["", "58", "数据变现"],
    ["", "59", "数据确权"],
    ["", "60", "数据定价"],
    ["", "61", "数据流通"],
    ["", "62", "数据市场"],
    ["", "63", "数据入表"],
    ["", "64", "数据资源入表"],
    ["数据治理", "65", "数据治理"],
    ["data_gov", "66", "数据安全"],
    ["(9个)", "67", "数据隐私"],
    ["", "68", "数据合规"],
    ["", "69", "数据质量"],
    ["", "70", "数据标准"],
    ["", "71", "数据脱敏"],
    ["", "72", "个人信息保护"],
    ["", "73", "数据分类分级"],
]

# 表A1
rowsA1 = [
    ["分组口径", "较早阶段/低组系数", "t值", "较晚阶段/高组系数", "t值", "交互项P值", "说明"],
    ["2011-2020 vs 2021-2024", "-0.0060***", "-5.22", "+0.0003", "0.18", "0.448", "2021切点不稳定"],
    ["2011-2019 vs 2021-2024（剔除2020）", "-0.0060***", "-5.06", "+0.0003", "0.18", "0.940", "剔除2020后差异消失"],
    ["非数字核心 vs 数字核心", "-0.0041***", "-3.40", "-0.0013", "-1.25", "0.507", "统一滞后规格下不显著"],
]

# 表A2
rowsA2 = [
    ["指标", "滞后系数", "t值", "与DU_kw联合", "与Narr联合", "定位"],
    [
        "DUevent",
        "-0.0023",
        "-1.51",
        "转正且显著",
        "转正且显著",
        "不稳",
    ],
    [
        "DUchain_count",
        "-0.0018**",
        "-2.13",
        "不显著",
        "不显著",
        "链条完整度",
    ],
    [
        "DUclosedloop",
        "-0.0083***",
        "-2.74",
        "不显著",
        "负向，10%显著",
        "闭环表述",
    ],
    [
        "DUcore",
        "-0.0095***",
        "-3.46",
        "不显著",
        "不显著",
        "窄口径支持",
    ],
    [
        "DUkw_mda",
        "-0.0006***",
        "-2.95",
        "不显著",
        "不显著",
        "MD&A口径支持",
    ],
]

story = []

# 按范文顺序优先放置：描述性统计、基准回归、图1、稳健性、机制、异质性
story += make_table(
    rows4,
    [2.5 * cm] + [2.0 * cm] * 6,
    1,
    "表1 变量的描述性统计",
    "注：所有连续变量在1%和99%分位数处进行缩尾处理。",
)
story.append(PageBreak())
story += make_table(
    rows5,
    [2.6 * cm] + [2.2 * cm] * 5,
    2,
    "表2 基准回归检验",
    "注：括号内为行业×年份层面聚类调整后的稳健标准误；***、**、*分别表示在1%、5%、10%的水平上显著。",
    extra_style=[("LINEBELOW", (0, 1), (-1, 1), 0.5, colors.black)],
)
story.append(PageBreak())

psm_path = os.path.join(BASE, "results", "v16_tables", "psm_density.png")
story.append(Paragraph(rich_text("图1 匹配前后核密度曲线"), s_title))
story.append(Spacer(1, 0.25 * cm))
psm_img = Image(psm_path, width=15.2 * cm, height=8.6 * cm)
story.append(psm_img)
story.append(Spacer(1, 0.2 * cm))
story.append(
    Paragraph(
        rich_text("注：图中展示倾向得分匹配前后处理组与对照组的倾向得分核密度分布，用于直观比较匹配后的样本可比性。"),
        s_note,
    )
)
story.append(PageBreak())

story += make_table(
    rows9,
    [2.25 * cm] + [1.8 * cm] * 7,
    3,
    "表3 稳健性检验",
    "注：第(1)列替换为行业×年份联合固定效应；第(5)列为倾向得分匹配结果；第(6)列为企业和年份双向聚类标准误；第(7)列以DU_sub_ln替代DU_kw。",
    extra_style=[("LINEBELOW", (0, 2), (-1, 2), 0.5, colors.black)],
)
story.append(PageBreak())
story += make_table(
    rows10,
    [2.5 * cm] + [1.9 * cm] * 6,
    2,
    "表4 数据要素利用与资产定价效率：作用机制检验",
    "注：渠道变量分别为分析师预测分歧、现金流波动率和综合供应链集中度。所有规格控制11个控制变量、企业固定效应和年份固定效应，标准误在行业×年份层面聚类。",
    extra_style=[("LINEBELOW", (0, 1), (-1, 1), 0.5, colors.black)],
)
story.append(PageBreak())
story += make_table(
    rows12,
    [2.6 * cm] + [3.0 * cm] * 2,
    3,
    "表5 数据要素利用与资产定价效率：异质性检验",
    "注：以2022年为制度阶段切点。交互项P值来自统一滞后规格下的交互项检验，正文同时报告费舍尔组合检验。",
    extra_style=[("LINEBELOW", (0, 2), (-1, 2), 0.5, colors.black)],
)
story.append(PageBreak())
story += make_table(
    rows13,
    [2.3 * cm] + [2.5 * cm] * 4,
    3,
    "表6 数据要素利用与资产定价效率：进一步异质性检验",
    "注：分别按分析师关注度和上市板块分组。交互项P值来自统一滞后规格下的交互项检验。",
    extra_style=[("LINEBELOW", (0, 2), (-1, 2), 0.5, colors.black)],
)
story.append(PageBreak())

# 范文中未出现的扩展结果，顺延至后
story += make_table(
    rows6,
    [2.8 * cm] + [1.7 * cm] * 7,
    3,
    "表7 数据要素利用的构念边界与替代测度检验",
    "注：所有规格均采用滞后一期解释变量，控制11个控制变量、企业固定效应和年份固定效应，标准误在行业×年份层面聚类。",
    extra_style=[("LINEBELOW", (0, 2), (-1, 2), 0.5, colors.black)],
)
story.append(PageBreak())
story += make_table(
    rows6a,
    [2.9 * cm] + [1.85 * cm] * 6,
    3,
    "表8 数据要素利用的增强测度与文本口径检验",
    "注：所有规格均为统一滞后设定。第(6)列在DUclosedloop与GenericNarr联合回归中同时报告两者系数。",
    extra_style=[("LINEBELOW", (0, 2), (-1, 2), 0.5, colors.black)],
)
story.append(PageBreak())
story += make_table(
    rows7,
    [2.6 * cm] + [2.15 * cm] * 5,
    2,
    "表9 数据要素利用、资产定价效率与内生性识别",
    "注：第(3)至(5)列为工具变量估计。First-stage F为第一阶段统计量；正文同时报告Oster界检验、Heckman两阶段法和安慰剂检验。",
    extra_style=[("LINEBELOW", (0, 1), (-1, 1), 0.5, colors.black)],
)
story.append(PageBreak())
story += make_table(
    rows8,
    [2.6 * cm] + [1.85 * cm] * 6,
    2,
    "表10 数据要素利用与资产定价效率：机器学习回归结果",
    "注：DML-PLR框架，LightGBM为第一阶段学习器，5折交叉拟合，Mundlak均值替代企业固定效应。括号内为标准误，方括号内为t统计量。",
    extra_style=[
        ("LINEBELOW", (0, 1), (-1, 1), 0.5, colors.black),
        ("SPAN", (1, 0), (3, 0)),
        ("SPAN", (4, 0), (6, 0)),
    ],
)
story.append(PageBreak())
story += make_table(
    rows11,
    [2.7 * cm] + [3.0 * cm] * 3,
    3,
    "表11 数据要素利用的广度—深度错配检验",
    "注：WashGap=z(DU_kw)-z(DU_llm_lenstd)；BroadShallow表示关键词广度高而语义深度低的企业。所有规格均采用滞后设定。",
    extra_style=[("LINEBELOW", (0, 2), (-1, 2), 0.5, colors.black)],
)
story.append(PageBreak())
story += make_table(
    rows14,
    [2.5 * cm] + [3.0 * cm] * 3,
    3,
    "表12 数据要素利用的同群溢出效应检验",
    "注：DU_kw_z与IndPen_mean_z均为标准化指标；IndPen_mean采用行业-年份留一均值构造。第(3)列为滞后规格。",
    extra_style=[("LINEBELOW", (0, 2), (-1, 2), 0.5, colors.black)],
)
story.append(PageBreak())

# 定义类与补充材料放在附表和附录
story += make_table(
    rows1,
    [4.2 * cm, 2.0 * cm, 8.0 * cm],
    1,
    "附表1 数据要素利用关键词体系说明",
    "注：关键词体系沿数据价值链构建，覆盖数据存量、开发能力、驱动应用、价值变现和治理五个环节，共73个关键词。",
)
story.append(PageBreak())
story += make_table(
    rows2,
    [3.4 * cm, 10.8 * cm],
    1,
    "附表2 控制变量说明",
    "注：控制变量参照股价延迟研究的通行做法选取，用于主回归、机制检验及异质性分析。",
)
story.append(PageBreak())
story += make_table(
    rows3,
    [5.0 * cm, 9.2 * cm],
    1,
    "附表3 机制变量说明",
    "注：渠道变量均经1%/99%分位数缩尾处理。",
)
story.append(PageBreak())
story += make_table(
    rows_kw,
    [2.8 * cm, 1.5 * cm, 8.9 * cm],
    1,
    "附录A 数据要素利用关键词清单",
    "注：关键词体系沿数据价值链构建，文本匹配采用长词优先策略，避免短词误匹配。",
)
story.append(PageBreak())
story += make_table(
    rowsA1,
    [4.8 * cm, 2.3 * cm, 1.6 * cm, 2.3 * cm, 1.6 * cm, 1.9 * cm, 3.1 * cm],
    1,
    "表A1 替代切点与补充异质性结果",
    "注：所有模型均采用t-1期DU解释t期PriceDelay的滞后规格，控制企业和年份固定效应及11个控制变量，标准误在行业×年份层面聚类。",
)
story.append(PageBreak())
story += make_table(
    rowsA2,
    [3.2 * cm, 2.2 * cm, 1.8 * cm, 2.8 * cm, 2.8 * cm, 2.5 * cm],
    1,
    "表A2 增强测度的补充检验结果",
    "注：与DU_kw联合、与Narr联合两列概括联合回归下增强测度的稳健性表现，Narr指GenericNarr。",
)

pdf_path = os.path.join(OUT, "regression_tables_current_v15.pdf")
doc = SimpleDocTemplate(
    pdf_path,
    pagesize=A4,
    leftMargin=2.1 * cm,
    rightMargin=2.1 * cm,
    topMargin=1.8 * cm,
    bottomMargin=1.8 * cm,
)
doc.build(story)
print(f"Done: {pdf_path}")
