"""
v2.1 回归表 PDF 生成器

约束：
- 仅复用已有 v17 数值或指定 CSV/DTA 数据，不重跑任何回归
- 仅使用 reportlab + pandas
"""
import json
import math
import os
import shutil
import tempfile
import warnings

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.pdfmetrics import registerFontFamily
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

warnings.filterwarnings("ignore")

BASE = "/Users/mac/computerscience/0做完了/15会计研究"
PRIMARY_OUT = f"{BASE}/v2/results/v2-1_tables"
OUTPUT_NAME = "regression_tables_v2-1.pdf"
PAGE_SIZE = (595.275, 807.874)
LEFT_MARGIN = 1.45 * cm
RIGHT_MARGIN = 1.45 * cm
TOP_MARGIN = 1.15 * cm
BOTTOM_MARGIN = 1.15 * cm
TEXT_WIDTH = PAGE_SIZE[0] - LEFT_MARGIN - RIGHT_MARGIN
OUTPUT_DIRS = [PRIMARY_OUT]
for out_dir in OUTPUT_DIRS:
    os.makedirs(out_dir, exist_ok=True)

# Font setup
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

# Styles
s_title = ParagraphStyle(
    "Title",
    fontName=CN,
    fontSize=11,
    alignment=TA_CENTER,
    leading=13,
    spaceBefore=0,
    spaceAfter=0,
)
s_caption_no = ParagraphStyle(
    "CaptionNo",
    fontName=CN,
    fontSize=10.5,
    alignment=TA_LEFT,
    leading=12,
)
s_caption_title = ParagraphStyle(
    "CaptionTitle",
    fontName=CN,
    fontSize=10.5,
    alignment=TA_CENTER,
    leading=12,
)
s_panel = ParagraphStyle(
    "Panel",
    fontName=CN,
    fontSize=9.2,
    alignment=TA_LEFT,
    leading=10.5,
    spaceBefore=1,
    spaceAfter=2,
)
s_section = ParagraphStyle(
    "Section",
    fontName=CN,
    fontSize=10.5,
    alignment=TA_CENTER,
    leading=12,
    spaceBefore=0,
    spaceAfter=2,
)
s_note = ParagraphStyle(
    "Note",
    fontName=CN,
    fontSize=8.2,
    alignment=TA_LEFT,
    leading=10.8,
    wordWrap="CJK",
)
s_cell_c = ParagraphStyle(
    "CellCenter",
    fontName=CN,
    fontSize=8.2,
    alignment=TA_CENTER,
    leading=9.6,
    wordWrap="CJK",
)
s_cell_l = ParagraphStyle(
    "CellLeft",
    fontName=CN,
    fontSize=8.2,
    alignment=TA_LEFT,
    leading=9.6,
    wordWrap="CJK",
)


def P(text):
    return Paragraph(str(text), s_cell_c)


def L(text):
    return Paragraph(str(text), s_cell_l)


def fmt_signed(value, digits):
    if value is None or pd.isna(value):
        return ""
    sign = "−" if float(value) < 0 else ""
    return f"{sign}{abs(float(value)):.{digits}f}"


def fmt_plain(value, digits):
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):.{digits}f}"


def fmt_int(value):
    if value is None or pd.isna(value):
        return ""
    return f"{int(round(float(value))):,}"


def starify(value, pval, digits=4):
    stars = ""
    if pval < 0.01:
        stars = "***"
    elif pval < 0.05:
        stars = "**"
    elif pval < 0.10:
        stars = "*"
    return f"{fmt_signed(value, digits)}{stars}"


def paren(value, digits=4):
    return f"({fmt_plain(value, digits)})"


def t_value(value):
    return fmt_signed(value, 2)


def fisher_p(value):
    stars = ""
    if value < 0.01:
        stars = "***"
    elif value < 0.05:
        stars = "**"
    elif value < 0.10:
        stars = "*"
    return f"{fmt_plain(value, 3)}{stars}"


def make_caption(number, title):
    table = Table(
        [[Paragraph(number, s_caption_no), Paragraph(title, s_caption_title), Paragraph("", s_caption_no)]],
        colWidths=[2.2 * cm, TEXT_WIDTH - 4.4 * cm, 2.2 * cm],
    )
    table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "BOTTOM"),
                ("ALIGN", (0, 0), (0, 0), "LEFT"),
                ("ALIGN", (1, 0), (1, 0), "CENTER"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
            ]
        )
    )
    return table


def three_line(n_rows, header_rows=1, spans=None, extra_lines=None):
    style = [
        ("FONTNAME", (0, 0), (-1, -1), CN),
        ("FONTSIZE", (0, 0), (-1, -1), 8.2),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("ALIGN", (0, 0), (0, -1), "LEFT"),
        ("LINEABOVE", (0, 0), (-1, 0), 1.5, colors.black),
        ("LINEBELOW", (0, header_rows - 1), (-1, header_rows - 1), 0.75, colors.black),
        ("LINEBELOW", (0, n_rows - 1), (-1, n_rows - 1), 1.5, colors.black),
        ("TOPPADDING", (0, 0), (-1, -1), 1.0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1.0),
        ("LEFTPADDING", (0, 0), (-1, -1), 2),
        ("RIGHTPADDING", (0, 0), (-1, -1), 2),
    ]
    if spans:
        style.extend(spans)
    if extra_lines:
        style.extend(extra_lines)
    return TableStyle(style)


ITALIC_LABELS = {
    "DU_kw": "<i>DU_kw</i>",
    "DU_llm": "<i>DU_llm</i>",
    "DU_stock": "<i>DU_stock</i>",
    "DU_dev": "<i>DU_dev</i>",
    "DU_app": "<i>DU_app</i>",
    "DU_value": "<i>DU_value</i>",
    "DU_gov": "<i>DU_gov</i>",
    "DU_llm_lenstd": "<i>DU_llm_lenstd</i>",
    "DUclosedloop": "<i>DU_closedloop</i>",
    "DUcore": "<i>DU_core</i>",
    "DUchain_count": "<i>DU_chain,count</i>",
    "DUkw_mda": "<i>DU_kw,mda</i>",
    "WashGap": "WashGap",
}


TABLE1_VARIABLES = [
    "PriceDelay",
    "SYNCH",
    "DU_kw",
    "DU_llm",
    "DU_stock",
    "DU_dev",
    "DU_app",
    "DU_value",
    "DU_gov",
    "DU_llm_lenstd",
    "DUclosedloop",
    "DUcore",
    "DUchain_count",
    "DUkw_mda",
    "WashGap",
    "Size",
    "Lev",
    "ROA",
    "TobinQ",
    "Age",
    "Growth",
    "IndepRatio",
    "Dual",
    "Top1Share",
    "SOE",
    "CFO",
]


def label_for(var_name, lag=False):
    base = var_name.replace("_lag", "")
    label = ITALIC_LABELS.get(base, base)
    if lag:
        return f"{label}(t−1)"
    return label


def make_table1():
    df = pd.read_stata(
        f"{BASE}/data_stata/reg_sample_v16_integrated.dta",
        convert_categoricals=False,
    )
    stats = df[TABLE1_VARIABLES].describe(percentiles=[0.5]).T
    rows = [["变量", "观测值", "均值", "标准差", "最小值", "中位数", "最大值"]]
    for var in TABLE1_VARIABLES:
        rows.append(
            [
                L(label_for(var)),
                P(fmt_int(stats.loc[var, "count"])),
                P(fmt_plain(stats.loc[var, "mean"], 3)),
                P(fmt_plain(stats.loc[var, "std"], 3)),
                P(fmt_signed(stats.loc[var, "min"], 3)),
                P(fmt_signed(stats.loc[var, "50%"], 3)),
                P(fmt_signed(stats.loc[var, "max"], 3)),
            ]
        )
    table = Table(rows, colWidths=[3.6 * cm] + [2.05 * cm] * 6)
    table.setStyle(three_line(len(rows)))
    note = Paragraph(
        "注：所有连续变量均在 1% 和 99% 分位数处缩尾处理，主要变量定义见正文。",
        s_note,
    )
    return None, table, note, len(rows)


def load_v17_ols():
    with open(f"{BASE}/results/v17_tables/ols_stata_sample.json", "r", encoding="utf-8") as fh:
        return json.load(fh)


def build_v17_table2():
    ols = load_v17_ols()
    control_order = [
        "Size",
        "Lev",
        "ROA",
        "TobinQ",
        "Age",
        "Growth",
        "IndepRatio",
        "Dual",
        "Top1Share",
        "SOE",
        "CFO",
    ]
    pretty = {"Top1Share": "Top1Share"}

    def fc(model, var):
        cell = ols[model].get(var)
        if not cell:
            return "", ""
        return starify(cell["coef"], cell["p"], 4), paren(cell["se"], 4)

    rows = [
        [L(""), P("(1)"), P("(2)"), P("(3)"), P("(4)"), P("(5)")],
        [L("变量"), P("PriceDelay"), P("PriceDelay"), P("PriceDelay"), P("PriceDelay"), P("PriceDelay")],
        [L(label_for("DU_kw")), P("−0.0052***"), P("−0.0046***"), P(""), P(""), P("")],
        [L(""), P("(0.0010)"), P("(0.0009)"), P(""), P(""), P("")],
        [L(label_for("DU_llm")), P(""), P(""), P("−0.0037***"), P("−0.0036***"), P("")],
        [L(""), P(""), P(""), P("(0.0008)"), P("(0.0008)"), P("")],
        [L("LLM_binary"), P(""), P(""), P(""), P(""), P("−0.0033***")],
        [L(""), P(""), P(""), P(""), P(""), P("(0.0013)")],
    ]
    for var in control_order:
        coef_row = [L(pretty.get(var, var))]
        se_row = [L("")]
        for model in ["(1)", "(2)", "(3)", "(4)", "(5)"]:
            coef, se = fc(model, var)
            coef_row.append(P(coef))
            se_row.append(P(se))
        rows.append(coef_row)
        rows.append(se_row)
    rows.extend(
        [
            [L("控制变量"), P("NO"), P("YES"), P("NO"), P("YES"), P("YES")],
            [L("Firm FE"), P("NO"), P("YES"), P("NO"), P("YES"), P("YES")],
            [L("Year FE"), P("NO"), P("YES"), P("NO"), P("YES"), P("YES")],
            [L("Cluster"), P("Ind×Year"), P("Ind×Year"), P("Ind×Year"), P("Ind×Year"), P("Ind×Year")],
            [L("N"), P("43,721"), P("43,721"), P("43,721"), P("43,721"), P("43,721")],
            [L("R²"), P("0.409"), P("0.419"), P("0.408"), P("0.418"), P("0.418")],
        ]
    )
    table = Table(rows, colWidths=[2.8 * cm] + [2.24 * cm] * 5)
    table.setStyle(
        three_line(
            len(rows),
            header_rows=2,
            extra_lines=[("LINEBELOW", (0, 1), (-1, 1), 0.75, colors.black)],
        )
    )
    note = Paragraph(
        "注：括号内数值为行业×年份层面聚类调整后的稳健标准误；"
        "***、**、* 分别表示在 1%、5%、10% 的水平上显著，下同。",
        s_note,
    )
    return None, table, note


def build_v17_table3():
    rows = [
        [L(""), P("Y=PriceDelay"), P(""), P(""), P("Y=SYNCH"), P(""), P("")],
        [L("处理变量 D"), P("系数"), P("标准误"), P("t 值"), P("系数"), P("标准误"), P("t 值")],
        [L(label_for("DU_kw")), P("−0.0029***"), P("(0.0004)"), P("[−6.50]"), P("+0.0277***"), P("(0.0029)"), P("[9.65]")],
        [L("<i>DU_kw_ln</i>"), P("−0.0040***"), P("(0.0006)"), P("[−7.19]"), P(""), P(""), P("")],
        [L("<i>DU_sub_ln</i>"), P("−0.0039***"), P("(0.0005)"), P("[−7.21]"), P(""), P(""), P("")],
        [L(label_for("DU_llm")), P("−0.0042***"), P("(0.0007)"), P("[−6.35]"), P("+0.0326***"), P("(0.0041)"), P("[7.98]")],
        [L("LLM_binary"), P("−0.0042***"), P("(0.0012)"), P("[−3.46]"), P("+0.0435***"), P("(0.0075)"), P("[5.80]")],
        [L("控制变量"), P("YES"), P(""), P(""), P("YES"), P(""), P("")],
        [L("Firm FE"), P("Mundlak"), P(""), P(""), P("Mundlak"), P(""), P("")],
        [L("Year FE"), P("YES"), P(""), P(""), P("YES"), P(""), P("")],
        [L("Cluster"), P("Ind×Year"), P(""), P(""), P("Ind×Year"), P(""), P("")],
        [L("N"), P("35,666"), P(""), P(""), P("35,614"), P(""), P("")],
    ]
    table = Table(rows, colWidths=[2.55 * cm] + [1.92 * cm] * 6)
    table.setStyle(
        three_line(
            len(rows),
            header_rows=2,
            spans=[("SPAN", (1, 0), (3, 0)), ("SPAN", (4, 0), (6, 0))],
            extra_lines=[("LINEBELOW", (0, 1), (-1, 1), 0.75, colors.black)],
        )
    )
    note = Paragraph(
        "注：DML-PLR 采用 5 折交叉拟合，LightGBM 为第一阶段学习器；括号内为标准误，方括号内为 t 统计量。",
        s_note,
    )
    return None, table, note


def build_v17_table4():
    rows = [
        [L(""), P("(1)"), P("(2)"), P("(3)"), P("(4)"), P("(5)")],
        [L("模型"), P("OLS 基准"), P("IV1: 跨省 peer"), P("IV2: 数字化"), P("滞后 OLS"), P("滞后 IV")],
        [L(label_for("DU_kw")), P("−0.0046***"), P("−0.0143***"), P("−0.0352***"), P("−0.0039***"), P("−0.0163***")],
        [L(""), P("(0.0009)"), P("(0.0047)"), P("(0.0114)"), P("(0.0010)"), P("(0.0053)")],
        [L("First-stage F"), P(""), P("251.3"), P("72.7"), P(""), P("177.8")],
        [L("控制变量"), P("YES"), P("YES"), P("YES"), P("YES"), P("YES")],
        [L("Firm FE"), P("YES"), P("YES"), P("YES"), P("YES"), P("YES")],
        [L("Year FE"), P("YES"), P("YES"), P("YES"), P("YES"), P("YES")],
        [L("Cluster"), P("Ind×Year"), P("Ind×Year"), P("Ind×Year"), P("Ind×Year"), P("Ind×Year")],
        [L("N"), P("43,695"), P("43,612"), P("43,695"), P("37,278"), P("37,198")],
        [L(""), P(""), P(""), P(""), P(""), P("")],
        [L("辅助检验"), P(""), P(""), P(""), P(""), P("")],
        [L("Oster δ*(1.3R²)"), P("96.3"), P(""), P(""), P(""), P("")],
        [L("Oster δ*(保守)"), P("7.5"), P(""), P(""), P(""), P("")],
        [L(f"Heckman: {label_for('DU_kw')}"), P("−0.0040***"), P("t=−4.39"), P(""), P(""), P("")],
        [L("Heckman: IMR"), P("−0.123***"), P("t=−6.23"), P(""), P(""), P("")],
        [L("安慰剂：当期披露"), P("−0.0050***"), P("t=−4.64"), P(""), P(""), P("")],
        [L("安慰剂：领先一期披露"), P("0.0001"), P("t=0.11"), P(""), P(""), P("")],
    ]
    table = Table(rows, colWidths=[3.25 * cm] + [2.15 * cm] * 5)
    table.setStyle(
        three_line(
            len(rows),
            header_rows=2,
            extra_lines=[
                ("LINEBELOW", (0, 1), (-1, 1), 0.75, colors.black),
                ("LINEABOVE", (0, 11), (-1, 11), 0.75, colors.black),
                ("LINEBELOW", (0, 11), (-1, 11), 0.75, colors.black),
            ],
        )
    )
    note = Paragraph(
        "注：IV1 为同行业跨省企业披露强度留一均值，IV2 为省级大数据发展指数（2016）× year；"
        "DWH 检验均拒绝外生性。",
        s_note,
    )
    return None, table, note


def build_v17_table5_and_figure():
    rows = [
        [L(""), P("(1)替换"), P("(2)剔除"), P("(3)剔除"), P("(4)滞后"), P("(5)倾向"), P("(6)双向"), P("(7)替换")],
        [L(""), P("固定效应"), P("末期年份"), P("IT 行业"), P("控制变量"), P("得分匹配"), P("聚类 SE"), P("自变量")],
        [L("变量"), P("PriceDelay"), P("PriceDelay"), P("PriceDelay"), P("PriceDelay"), P("PriceDelay"), P("PriceDelay"), P("PriceDelay")],
        [L(label_for("DU_kw")), P("−0.0029***"), P("−0.0052***"), P("−0.0046***"), P("−0.0040***"), P("−0.0039***"), P("−0.0046***"), P("")],
        [L("(<i>DU_sub_ln</i>)"), P(""), P(""), P(""), P(""), P(""), P(""), P("−0.0042***")],
        [L(""), P("(0.0006)"), P("(0.0010)"), P("(0.0009)"), P("(0.0009)"), P("(0.0008)"), P("(0.0012)"), P("(0.0008)")],
        [L("控制变量"), P("YES"), P("YES"), P("YES"), P("YES"), P("YES"), P("YES"), P("YES")],
        [L("Firm FE"), P("YES"), P("YES"), P("YES"), P("YES"), P("YES"), P("YES"), P("YES")],
        [L("Year FE"), P("YES"), P("YES"), P("YES"), P("YES"), P("YES"), P("YES"), P("YES")],
        [L("Cluster"), P("Ind×Year"), P("Ind×Year"), P("Ind×Year"), P("Ind×Year"), P("Ind×Year"), P("Firm+Year"), P("Ind×Year")],
        [L("Ind×Year FE"), P("YES"), P("NO"), P("NO"), P("NO"), P("NO"), P("NO"), P("NO")],
        [L("N"), P("43,656"), P("38,812"), P("40,613"), P("37,294"), P("35,929"), P("43,721"), P("43,721")],
        [L("R²"), P("0.472"), P("0.428"), P("0.419"), P("0.416"), P("0.434"), P("0.419"), P("0.418")],
    ]
    table = Table(rows, colWidths=[2.1 * cm] + [1.86 * cm] * 7)
    table.setStyle(
        three_line(
            len(rows),
            header_rows=3,
            extra_lines=[("LINEBELOW", (0, 2), (-1, 2), 0.75, colors.black)],
        )
    )
    note = Paragraph(
        "注：第（1）列替换为行业×年份联合固定效应；第（5）列采用倾向得分匹配；"
        "第（6）列采用企业与年份双向聚类；第（7）列以 <i>DU_sub_ln</i> 替代 <i>DU_kw</i>。",
        s_note,
    )
    fig_path = f"{BASE}/results/v16_tables/psm_density.png"
    fig_note = Paragraph(
        "注：处理组定义为 <i>DU_kw</i> 高于行业-年份中位数，采用 Logit 倾向得分和最近邻 1:1 匹配（卡尺 0.05）。",
        s_note,
    )
    image = Image(fig_path, width=14 * cm, height=5.2 * cm)
    return None, table, note, None, image, fig_note


def load_csv(path):
    return pd.read_csv(path)


def build_table6():
    df = load_csv(f"{BASE}/results/v16_integrated/h2a_value_chain.csv")
    order = ["DU_stock", "DU_dev", "DU_app", "DU_value", "DU_gov"]
    rows = [
        [L(""), P("(1)"), P("(2)"), P("(3)"), P("(4)"), P("(5)")],
        [L("变量"), P("PriceDelay"), P("PriceDelay"), P("PriceDelay"), P("PriceDelay"), P("PriceDelay")],
    ]
    for target in order:
        row = df.loc[df["model"] == target].iloc[0]
        coef_cells = [P("") for _ in order]
        se_cells = [P("") for _ in order]
        idx = order.index(target)
        coef_cells[idx] = P(starify(row["coef"], row["pval"], 4))
        se_cells[idx] = P(paren(row["se"], 4))
        rows.append([L(label_for(row["regressor"], lag=True))] + coef_cells)
        rows.append([L("")] + se_cells)
    rows.extend(
        [
            [L("控制变量"), P("YES"), P("YES"), P("YES"), P("YES"), P("YES")],
            [L("Firm FE"), P("YES"), P("YES"), P("YES"), P("YES"), P("YES")],
            [L("Year FE"), P("YES"), P("YES"), P("YES"), P("YES"), P("YES")],
            [L("Cluster"), P("Ind×Year"), P("Ind×Year"), P("Ind×Year"), P("Ind×Year"), P("Ind×Year")],
            [L("N")] + [P(fmt_int(df.loc[df["model"] == target, "N"].iloc[0])) for target in order],
            [L("R²")] + [P(fmt_plain(df.loc[df["model"] == target, "r2"].iloc[0], 3)) for target in order],
        ]
    )
    table = Table(rows, colWidths=[3.15 * cm] + [2.1 * cm] * 5)
    table.setStyle(
        three_line(
            len(rows),
            header_rows=2,
            extra_lines=[("LINEBELOW", (0, 1), (-1, 1), 0.75, colors.black)],
        )
    )
    note = Paragraph(
        "注：应用层、开发层和资源层披露显著降低 PriceDelay；价值化和治理层因披露稀疏而未达显著。",
        s_note,
    )
    return None, table, note


def build_table7_panel_a():
    df = load_csv(f"{BASE}/results/v16_integrated/h2b_quality_direct.csv")
    order = [
        "direct_DUclosedloop",
        "direct_DUcore",
        "direct_DUchain_count",
        "direct_DUkw_mda",
        "direct_DU_llm_lenstd",
    ]
    rows = [
        [L(""), P("(1)"), P("(2)"), P("(3)"), P("(4)"), P("(5)")],
        [L("变量"), P("PriceDelay"), P("PriceDelay"), P("PriceDelay"), P("PriceDelay"), P("PriceDelay")],
    ]
    for model in order:
        row = df.loc[df["model"] == model].iloc[0]
        cells = [P("") for _ in order]
        se_cells = [P("") for _ in order]
        idx = order.index(model)
        cells[idx] = P(starify(row["coef"], row["pval"], 4))
        se_cells[idx] = P(paren(row["se"], 4))
        rows.append([L(label_for(row["regressor"], lag=True))] + cells)
        rows.append([L("")] + se_cells)
    rows.extend(
        [
            [L("控制变量"), P("YES"), P("YES"), P("YES"), P("YES"), P("YES")],
            [L("Firm FE"), P("YES"), P("YES"), P("YES"), P("YES"), P("YES")],
            [L("Year FE"), P("YES"), P("YES"), P("YES"), P("YES"), P("YES")],
            [L("Cluster"), P("Ind×Year"), P("Ind×Year"), P("Ind×Year"), P("Ind×Year"), P("Ind×Year")],
            [L("N")] + [P(fmt_int(df.loc[df["model"] == model, "N"].iloc[0])) for model in order],
            [L("R²")] + [P(fmt_plain(df.loc[df["model"] == model, "r2"].iloc[0], 3)) for model in order],
        ]
    )
    table = Table(rows, colWidths=[3.2 * cm] + [2.08 * cm] * 5)
    table.setStyle(
        three_line(
            len(rows),
            header_rows=2,
            extra_lines=[("LINEBELOW", (0, 1), (-1, 1), 0.75, colors.black)],
        )
    )
    return table


def build_table7_panel_b():
    df = load_csv(f"{BASE}/results/v16_integrated/h2b_quality_joint.csv")
    df = df.loc[df["model"].str.startswith("joint_dukw_")].copy()
    order = [
        "joint_dukw_DUclosedloop",
        "joint_dukw_DUcore",
        "joint_dukw_DUchain_count",
        "joint_dukw_DUkw_mda",
        "joint_dukw_lenstd",
    ]
    rows = [
        [L(""), P("(1)"), P("(2)"), P("(3)"), P("(4)"), P("(5)")],
        [L("变量"), P("PriceDelay"), P("PriceDelay"), P("PriceDelay"), P("PriceDelay"), P("PriceDelay")],
    ]

    dkw_cells = [P("") for _ in order]
    dkw_se_cells = [P("") for _ in order]
    for idx, model in enumerate(order):
        base = df.loc[(df["model"] == model) & (df["regressor"] == "DU_kw_lag")].iloc[0]
        dkw_cells[idx] = P(starify(base["coef"], base["pval"], 4))
        dkw_se_cells[idx] = P(paren(base["se"], 4))
    rows.append([L(label_for("DU_kw", lag=True))] + dkw_cells)
    rows.append([L("")] + dkw_se_cells)

    second_var_order = {
        "joint_dukw_DUclosedloop": "DUclosedloop_lag",
        "joint_dukw_DUcore": "DUcore_lag",
        "joint_dukw_DUchain_count": "DUchain_count_lag",
        "joint_dukw_DUkw_mda": "DUkw_mda_lag",
        "joint_dukw_lenstd": "DU_llm_lenstd_lag",
    }
    for model in order:
        cells = [P("") for _ in order]
        se_cells = [P("") for _ in order]
        row = df.loc[(df["model"] == model) & (df["regressor"] == second_var_order[model])].iloc[0]
        idx = order.index(model)
        cells[idx] = P(starify(row["coef"], row["pval"], 4))
        se_cells[idx] = P(paren(row["se"], 4))
        rows.append([L(label_for(second_var_order[model], lag=True))] + cells)
        rows.append([L("")] + se_cells)
    rows.extend(
        [
            [L("控制变量"), P("YES"), P("YES"), P("YES"), P("YES"), P("YES")],
            [L("Firm FE"), P("YES"), P("YES"), P("YES"), P("YES"), P("YES")],
            [L("Year FE"), P("YES"), P("YES"), P("YES"), P("YES"), P("YES")],
            [L("Cluster"), P("Ind×Year"), P("Ind×Year"), P("Ind×Year"), P("Ind×Year"), P("Ind×Year")],
            [L("N")] + [P(fmt_int(df.loc[df["model"] == model, "N"].iloc[0])) for model in order],
            [L("R²")] + [P(fmt_plain(df.loc[df["model"] == model, "r2"].iloc[0], 3)) for model in order],
        ]
    )
    table = Table(rows, colWidths=[3.2 * cm] + [2.08 * cm] * 5)
    table.setStyle(
        three_line(
            len(rows),
            header_rows=2,
            extra_lines=[("LINEBELOW", (0, 1), (-1, 1), 0.75, colors.black)],
        )
    )
    note = Paragraph(
        "注：Panel A 为五项质量指标的单变量结果；Panel B 在控制 <i>DU_kw</i> 后检验其增量解释力。"
        "前四项质量指标在联合回归中转为不显著，<i>DU_llm_lenstd</i> 仍保持显著。",
        s_note,
    )
    return table, note


def build_table8():
    df = load_csv(f"{BASE}/results/v16_integrated/h2c_washgap.csv")
    direct = df.loc[df["model"] == "washgap_direct"].iloc[0]
    joint_kw_gap = df.loc[(df["model"] == "washgap_joint_kw") & (df["regressor"] == "WashGap_lag")].iloc[0]
    joint_kw_dkw = df.loc[(df["model"] == "washgap_joint_kw") & (df["regressor"] == "DU_kw_lag")].iloc[0]
    joint_llm_gap = df.loc[(df["model"] == "washgap_joint_llm") & (df["regressor"] == "WashGap_lag")].iloc[0]
    joint_llm_dllm = df.loc[(df["model"] == "washgap_joint_llm") & (df["regressor"] == "DU_llm_lag")].iloc[0]
    rows = [
        [L(""), P("(1)"), P("(2)"), P("(3)")],
        [L("变量"), P("PriceDelay"), P("PriceDelay"), P("PriceDelay")],
        [L("WashGap(t−1)"), P(starify(direct["coef"], direct["pval"], 4)), P(starify(joint_kw_gap["coef"], joint_kw_gap["pval"], 4)), P(starify(joint_llm_gap["coef"], joint_llm_gap["pval"], 4))],
        [L(""), P(paren(direct["se"], 4)), P(paren(joint_kw_gap["se"], 4)), P(paren(joint_llm_gap["se"], 4))],
        [L(label_for("DU_kw", lag=True)), P(""), P(starify(joint_kw_dkw["coef"], joint_kw_dkw["pval"], 4)), P("")],
        [L(""), P(""), P(paren(joint_kw_dkw["se"], 4)), P("")],
        [L(label_for("DU_llm", lag=True)), P(""), P(""), P(starify(joint_llm_dllm["coef"], joint_llm_dllm["pval"], 4))],
        [L(""), P(""), P(""), P(paren(joint_llm_dllm["se"], 4))],
        [L("控制变量"), P("YES"), P("YES"), P("YES")],
        [L("Firm FE"), P("YES"), P("YES"), P("YES")],
        [L("Year FE"), P("YES"), P("YES"), P("YES")],
        [L("Cluster"), P("Ind×Year"), P("Ind×Year"), P("Ind×Year")],
        [L("N"), P(fmt_int(direct["N"])), P(fmt_int(joint_kw_gap["N"])), P(fmt_int(joint_llm_gap["N"]))],
        [L("R²"), P(fmt_plain(direct["r2"], 3)), P(fmt_plain(joint_kw_gap["r2"], 3)), P(fmt_plain(joint_llm_gap["r2"], 3))],
    ]
    table = Table(rows, colWidths=[3.2 * cm] + [3.0 * cm] * 3)
    table.setStyle(
        three_line(
            len(rows),
            header_rows=2,
            extra_lines=[("LINEBELOW", (0, 1), (-1, 1), 0.75, colors.black)],
        )
    )
    note = Paragraph(
        "注：WashGap 度量披露“广而浅”的错配程度；第（2）列控制 <i>DU_kw</i>，第（3）列控制 <i>DU_llm</i>。",
        s_note,
    )
    return None, table, note


def build_table9():
    df = load_csv(f"{BASE}/results/v16_integrated/h3_downstream.csv")
    channels = ["ForecastDisp", "CashFlowVol", "SCConc"]
    rows = [
        [L(""), P("(1)"), P("(2)"), P("(3)"), P("(4)"), P("(5)"), P("(6)")],
        [L("变量"), P("ForecastDisp"), P("ForecastDisp"), P("CashFlowVol"), P("CashFlowVol"), P("SCConc"), P("SCConc")],
    ]
    kw_row = [L(label_for("DU_kw"))]
    llm_row = [L(label_for("DU_llm"))]
    kw_se_row = [L("")]
    llm_se_row = [L("")]
    for channel in channels:
        kw = df.loc[(df["channel"] == channel) & (df["treatment"] == "DU_kw")].iloc[0]
        llm = df.loc[(df["channel"] == channel) & (df["treatment"] == "DU_llm")].iloc[0]
        kw_row.extend([P(starify(kw["coef"], kw["pval"], 4)), P("")])
        kw_se_row.extend([P(paren(kw["se"], 4)), P("")])
        llm_row.extend([P(""), P(starify(llm["coef"], llm["pval"], 4))])
        llm_se_row.extend([P(""), P(paren(llm["se"], 4))])
    rows.extend([kw_row, kw_se_row, llm_row, llm_se_row])
    rows.extend(
        [
            [L("控制变量")] + [P("YES")] * 6,
            [L("Firm FE")] + [P("YES")] * 6,
            [L("Year FE")] + [P("YES")] * 6,
            [L("Cluster")] + [P("Ind×Year")] * 6,
            [
                L("N"),
                P(fmt_int(df.loc[(df["channel"] == "ForecastDisp") & (df["treatment"] == "DU_kw"), "N"].iloc[0])),
                P(fmt_int(df.loc[(df["channel"] == "ForecastDisp") & (df["treatment"] == "DU_llm"), "N"].iloc[0])),
                P(fmt_int(df.loc[(df["channel"] == "CashFlowVol") & (df["treatment"] == "DU_kw"), "N"].iloc[0])),
                P(fmt_int(df.loc[(df["channel"] == "CashFlowVol") & (df["treatment"] == "DU_llm"), "N"].iloc[0])),
                P(fmt_int(df.loc[(df["channel"] == "SCConc") & (df["treatment"] == "DU_kw"), "N"].iloc[0])),
                P(fmt_int(df.loc[(df["channel"] == "SCConc") & (df["treatment"] == "DU_llm"), "N"].iloc[0])),
            ],
            [
                L("R²"),
                P(fmt_plain(df.loc[(df["channel"] == "ForecastDisp") & (df["treatment"] == "DU_kw"), "r2"].iloc[0], 3)),
                P(fmt_plain(df.loc[(df["channel"] == "ForecastDisp") & (df["treatment"] == "DU_llm"), "r2"].iloc[0], 3)),
                P(fmt_plain(df.loc[(df["channel"] == "CashFlowVol") & (df["treatment"] == "DU_kw"), "r2"].iloc[0], 3)),
                P(fmt_plain(df.loc[(df["channel"] == "CashFlowVol") & (df["treatment"] == "DU_llm"), "r2"].iloc[0], 3)),
                P(fmt_plain(df.loc[(df["channel"] == "SCConc") & (df["treatment"] == "DU_kw"), "r2"].iloc[0], 3)),
                P(fmt_plain(df.loc[(df["channel"] == "SCConc") & (df["treatment"] == "DU_llm"), "r2"].iloc[0], 3)),
            ],
        ]
    )
    table = Table(rows, colWidths=[2.7 * cm] + [1.88 * cm] * 6)
    table.setStyle(
        three_line(
            len(rows),
            header_rows=2,
            extra_lines=[("LINEBELOW", (0, 1), (-1, 1), 0.75, colors.black)],
        )
    )
    note = Paragraph(
        "注：ForecastDisp 为分析师盈利预测分歧，CashFlowVol 为滚动五年经营现金流波动率，"
        "SCConc 为供应链综合集中度。三项结果共同指向估值不确定性下降的下游表现。",
        s_note,
    )
    return None, table, note


def heterogeneity_display_row(df, dimension, treatment):
    row = df.loc[(df["dimension"] == dimension) & (df["treatment"] == treatment)].iloc[0]
    if dimension == "Post2020":
        high_coef, high_t = row["coef_low"], row["t_low"]
        low_coef, low_t = row["coef_high"], row["t_high"]
        split = "2020 前 / 后"
    elif dimension == "DigEconCore":
        high_coef, high_t = row["coef_low"], row["t_low"]
        low_coef, low_t = row["coef_high"], row["t_high"]
        split = "非数字核心 / 数字核心"
    elif dimension == "HighTech":
        high_coef, high_t = row["coef_low"], row["t_low"]
        low_coef, low_t = row["coef_high"], row["t_high"]
        split = "非高科技 / 高科技"
    elif dimension == "SA":
        high_coef, high_t = row["coef_high"], row["t_high"]
        low_coef, low_t = row["coef_low"], row["t_low"]
        split = "高 SA / 低 SA"
    else:
        raise ValueError(f"Unexpected dimension: {dimension}")
    return {
        "split": split,
        "high_coef": high_coef,
        "high_t": high_t,
        "low_coef": low_coef,
        "low_t": low_t,
        "fisher_p": row["fisher_p"],
    }


def coef_from_t(coef, tstat):
    abs_t = abs(float(tstat))
    p_proxy = math.erfc(abs_t / math.sqrt(2))
    return starify(coef, p_proxy, 4)


def build_table10():
    df = load_csv(f"{BASE}/results/v18/heterogeneity_v18.csv")
    df = df.loc[df["dimension"].isin(["Post2020", "DigEconCore", "SA", "HighTech"])].copy()
    rows_a = [[L("维度"), L("组别划分"), P("高组系数"), P("高组 t"), P("低组系数"), P("低组 t"), P("Fisher P")]]
    rows_b = [[L("维度"), L("组别划分"), P("高组系数"), P("高组 t"), P("低组系数"), P("低组 t"), P("Fisher P")]]

    for dim in ["Post2020", "DigEconCore", "SA", "HighTech"]:
        disp = heterogeneity_display_row(df, dim, "DU_kw")
        rows_a.append(
            [
                L(dim),
                L(disp["split"]),
                P(coef_from_t(disp["high_coef"], disp["high_t"])),
                P(t_value(disp["high_t"])),
                P(coef_from_t(disp["low_coef"], disp["low_t"])),
                P(t_value(disp["low_t"])),
                P(fisher_p(disp["fisher_p"])),
            ]
        )
        disp = heterogeneity_display_row(df, dim, "DU_llm")
        rows_b.append(
            [
                L(dim),
                L(disp["split"]),
                P(coef_from_t(disp["high_coef"], disp["high_t"])),
                P(t_value(disp["high_t"])),
                P(coef_from_t(disp["low_coef"], disp["low_t"])),
                P(t_value(disp["low_t"])),
                P(fisher_p(disp["fisher_p"])),
            ]
        )

    table_a = Table(rows_a, colWidths=[2.45 * cm, 3.15 * cm, 2.05 * cm, 1.35 * cm, 2.05 * cm, 1.35 * cm, 2.0 * cm])
    table_b = Table(rows_b, colWidths=[2.45 * cm, 3.15 * cm, 2.05 * cm, 1.35 * cm, 2.05 * cm, 1.35 * cm, 2.0 * cm])
    table_a.setStyle(three_line(len(rows_a)))
    table_b.setStyle(three_line(len(rows_b)))
    note = Paragraph(
        "注：统一呈现 Post2020、DigEconCore、SA、HighTech 四个维度。"
        "高组/低组按照理论上信息摩擦或原始不确定性更高的一侧优先排列。"
        "Fisher P 基于 500 次随机置换检验，*/**/*** 分别对应 10%/5%/1% 显著性水平。",
        s_note,
    )
    return None, table_a, table_b, note


def build_document():
    tmp_pdf = os.path.join(tempfile.gettempdir(), OUTPUT_NAME)
    doc = SimpleDocTemplate(
        tmp_pdf,
        pagesize=PAGE_SIZE,
        topMargin=TOP_MARGIN,
        bottomMargin=BOTTOM_MARGIN,
        leftMargin=LEFT_MARGIN,
        rightMargin=RIGHT_MARGIN,
        pageCompression=0,
    )
    elements = []
    sp = Spacer(1, 2.2 * mm)

    _, t1, t1_note, _ = make_table1()
    elements += [make_caption("表 1", "变量的描述性统计"), sp, t1, sp, t1_note, PageBreak()]

    _, t2, t2_note = build_v17_table2()
    elements += [make_caption("表 2", "披露与股价定价效率：基准回归（H1）"), sp, t2, sp, t2_note, PageBreak()]

    _, t6, t6_note = build_table6()
    elements += [make_caption("表 3", "数据价值链五维分解（H2a）"), sp, t6, sp, t6_note, PageBreak()]

    t7a = build_table7_panel_a()
    t7b, t7_note = build_table7_panel_b()
    elements += [
        make_caption("表 4", "披露质量与增量解释力（H2b）"),
        Paragraph("Panel A：单变量回归", s_panel),
        sp,
        t7a,
        PageBreak(),
        make_caption("表 4（续）", "披露质量与增量解释力（H2b）"),
        Paragraph("Panel B：控制披露密度后的联合回归", s_panel),
        sp,
        t7b,
        sp,
        t7_note,
        PageBreak(),
    ]

    _, t8, t8_note = build_table8()
    elements += [make_caption("表 5", "广度-深度错配与 PriceDelay（H2c）"), sp, t8, sp, t8_note, PageBreak()]

    _, t9, t9_note = build_table9()
    elements += [make_caption("表 6", "与估值不确定性下降一致的下游 channels（H3）"), sp, t9, sp, t9_note, PageBreak()]

    _, t10a, t10b, t10_note = build_table10()
    elements += [
        make_caption("表 7", "情境异质性分析（H4）"),
        Paragraph(f"Panel A：{label_for('DU_kw')} 测度", s_panel),
        sp,
        t10a,
        PageBreak(),
        make_caption("表 7（续）", "情境异质性分析（H4）"),
        Paragraph(f"Panel B：{label_for('DU_llm')} 测度", s_panel),
        sp,
        t10b,
        sp,
        t10_note,
        PageBreak(),
    ]

    elements += [
        Paragraph("附录：H1 的补充识别与稳健性证据", s_section),
        sp,
    ]

    _, t3, t3_note = build_v17_table3()
    elements += [make_caption("附表 A1", "DML-PLR 补充结果"), sp, t3, sp, t3_note, PageBreak()]

    _, t4, t4_note = build_v17_table4()
    elements += [make_caption("附表 A2", "内生性检验"), sp, t4, sp, t4_note, PageBreak()]

    _, t5, t5_note, _, fig_img, fig_note = build_v17_table5_and_figure()
    elements += [make_caption("附表 A3", "稳健性检验"), sp, t5, sp, t5_note, PageBreak()]
    elements += [make_caption("附图 A1", "PSM 匹配前后核密度曲线"), sp, fig_img, sp, fig_note]

    def canvas_maker(*args, **kwargs):
        kwargs["pageCompression"] = 0
        return Canvas(*args, **kwargs)

    doc.build(elements, canvasmaker=canvas_maker)
    output_paths = []
    for out_dir in OUTPUT_DIRS:
        os.makedirs(out_dir, exist_ok=True)
        target = os.path.join(out_dir, OUTPUT_NAME)
        shutil.copyfile(tmp_pdf, target)
        output_paths.append(target)
    return output_paths


if __name__ == "__main__":
    paths = build_document()
    print(f"Done: {paths[0]}")
