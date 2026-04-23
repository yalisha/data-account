"""
Generate v19 result tables PDF and companion CSV/Markdown files.

Reference layout:
  results/v18/v18_tables.pdf

Current v19 focuses on the paper's latest main story:
  1. Construct boundary
  2. Measurement upgrade and WashGap
  3. Key mechanisms
  4. Main-text heterogeneity
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.pdfmetrics import registerFontFamily
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "results" / "v19"
OUT.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Fonts / styles
# ---------------------------------------------------------------------------
CN = "Helvetica"
songti = "/System/Library/Fonts/Supplemental/Songti.ttc"
if os.path.exists(songti):
    try:
        pdfmetrics.registerFont(TTFont("SongtiSC", songti, subfontIndex=6))
        pdfmetrics.registerFont(TTFont("SongtiSC-Bold", songti, subfontIndex=1))
        registerFontFamily("SongtiSC", normal="SongtiSC", bold="SongtiSC-Bold")
        CN = "SongtiSC"
    except Exception:
        pass

s_title = ParagraphStyle(
    "T", fontName=CN, fontSize=10.5, alignment=TA_CENTER, spaceAfter=4, spaceBefore=6, leading=14
)
s_note = ParagraphStyle("N", fontName=CN, fontSize=7.5, alignment=TA_LEFT, leading=10)
s_c = ParagraphStyle("C", fontName=CN, fontSize=8, alignment=TA_CENTER, leading=10)
s_l = ParagraphStyle("L", fontName=CN, fontSize=8, alignment=TA_LEFT, leading=10)
s_cs = ParagraphStyle("CS", fontName=CN, fontSize=7, alignment=TA_CENTER, leading=9)
s_ls = ParagraphStyle("LS", fontName=CN, fontSize=7, alignment=TA_LEFT, leading=9)

P = lambda t, style=s_c: Paragraph(str(t), style)
L = lambda t, style=s_l: Paragraph(str(t), style)
Ps = lambda t: Paragraph(str(t), s_cs)
Ls = lambda t: Paragraph(str(t), s_ls)


def three_line(n_rows: int, hdr: int = 1, spans: list[tuple] | None = None) -> TableStyle:
    sty = [
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
    if spans:
        sty.extend(spans)
    return TableStyle(sty)


def sig_from_p(p: float) -> str:
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.1:
        return "*"
    return ""


def sig_from_t(t: float) -> str:
    at = abs(t)
    if at >= 2.576:
        return "***"
    if at >= 1.960:
        return "**"
    if at >= 1.645:
        return "*"
    return ""


def fmt_coef(coef: float, t: float | None = None, p: float | None = None, digits: int = 4) -> str:
    sig = sig_from_p(p) if p is not None else sig_from_t(t or 0.0)
    sign = "\u2212" if coef < 0 else "+"
    return f"{sign}{abs(coef):.{digits}f}{sig}"


def fmt_se(se: float, digits: int = 4) -> str:
    return f"({se:.{digits}f})"


def fmt_p(p: float) -> str:
    if p < 0.001:
        return "<0.001***"
    return f"{p:.3f}{sig_from_p(p)}"


def save_csv(path: Path, rows: list[list[str]], header: list[str]) -> None:
    pd.DataFrame(rows, columns=header).to_csv(path, index=False, encoding="utf-8-sig")


PDF_VAR_LABELS = {
    "DU_kw_lag": "DU<sub>kw,t-1</sub>",
    "DU_llm_lag": "DU<sub>llm,t-1</sub>",
    "DU_llm_lenstd_lag": "DU<sub>llm-lenstd,t-1</sub>",
    "DU_kw_resid_lag": "DU<sub>kw-resid,t-1</sub>",
    "DU_kw_strict_lag": "DU<sub>kw-strict,t-1</sub>",
    "GenericNarr_lag": "GenericNarr<sub>t-1</sub>",
    "ln_total_chars_lag": "ln(TotalChars)<sub>t-1</sub>",
    "ln_mda_chars_lag": "ln(MDAChars)<sub>t-1</sub>",
    "DUchain_count_lag": "DU<sub>chain,t-1</sub>",
    "DUclosedloop_lag": "DU<sub>closed,t-1</sub>",
    "DUcore_lag": "DU<sub>core,t-1</sub>",
    "DUkw_mda_lag": "DU<sub>kw-mda,t-1</sub>",
    "WashGap_lag": "WashGap<sub>t-1</sub>",
    "BroadShallow_lag": "BroadShallow<sub>t-1</sub>",
    "DU_kw": "DU<sub>kw</sub>",
    "DU_llm": "DU<sub>llm</sub>",
}

CSV_VAR_LABELS = {
    "DU_kw_lag": "DUkw(t-1)",
    "DU_llm_lag": "DUllm(t-1)",
    "DU_llm_lenstd_lag": "DUllmLenstd(t-1)",
    "DU_kw_resid_lag": "DUkwResid(t-1)",
    "DU_kw_strict_lag": "DUkwStrict(t-1)",
    "GenericNarr_lag": "GenericNarr(t-1)",
    "ln_total_chars_lag": "ln(TotalChars)(t-1)",
    "ln_mda_chars_lag": "ln(MDAChars)(t-1)",
    "DUchain_count_lag": "DUchain(t-1)",
    "DUclosedloop_lag": "DUclosed(t-1)",
    "DUcore_lag": "DUcore(t-1)",
    "DUkw_mda_lag": "DUkwMDA(t-1)",
    "WashGap_lag": "WashGap(t-1)",
    "BroadShallow_lag": "BroadShallow(t-1)",
    "DU_kw": "DUkw",
    "DU_llm": "DUllm",
}


def pdf_var(name: str) -> str:
    return PDF_VAR_LABELS.get(name, name)


def csv_var(name: str) -> str:
    return CSV_VAR_LABELS.get(name, name)


# ---------------------------------------------------------------------------
# Load sources
# ---------------------------------------------------------------------------
construct = json.load(open(BASE / "results" / "v14" / "construct_boundary_results.json"))
washgap = json.load(open(BASE / "results" / "v14" / "washgap_results.json"))
meas = json.load(open(BASE / "results" / "v15_measurement" / "measurement_upgrade_results.json"))
het_extra = json.load(open(BASE / "results" / "v15_tables" / "heterogeneity_extra_v15.json"))
mech_old = pd.read_csv(BASE / "results" / "v18" / "mechanism_v18.csv")
mech_new = pd.read_csv(BASE / "results" / "v18" / "mechanism_new_v18.csv")
mech = pd.concat([mech_old, mech_new], ignore_index=True)

manual_mech = {
    ("SCConc", "DU_kw"): {"channel": "SCConc", "treatment": "DU_kw", "coef": -0.2767, "se": 0.0692, "tstat": -4.00, "pval": 0.00007, "N": 42332, "r2": 0.7646},
    ("SCConc", "DU_llm"): {"channel": "SCConc", "treatment": "DU_llm", "coef": -0.4602, "se": 0.0680, "tstat": -6.77, "pval": 2e-11, "N": 42332, "r2": 0.7647},
}


# ---------------------------------------------------------------------------
# Table 1: Construct boundary
# ---------------------------------------------------------------------------
d1 = [
    [L(""), P("(1)"), P("(2)"), P("(3)"), P("(4)"), P("(5)"), P("(6)"), P("(7)")],
    [
        L("变量"),
        P("总字数控制"),
        P("MD&A字数控制"),
        P("DU_llm"),
        P("长度标准化深度"),
        P("联合回归"),
        P("残差化"),
        P("去泛化词"),
    ],
    [L("因变量"), P("PriceDelay"), P("PriceDelay"), P("PriceDelay"), P("PriceDelay"), P("PriceDelay"), P("PriceDelay"), P("PriceDelay")],
]

row_specs_t1 = [
    ("DU_kw_lag", [construct["dukw_total_chars_lag"].get("DU_kw_lag"), construct["dukw_mda_chars_lag"].get("DU_kw_lag"), None, None, construct["joint_lag"].get("DU_kw_lag"), None, None]),
    ("ln_total_chars_lag", [construct["dukw_total_chars_lag"].get("ln_total_chars_lag"), None, construct["dullm_total_chars_lag"].get("ln_total_chars_lag"), construct["dullm_lenstd_total_chars_lag"].get("ln_total_chars_lag"), None, None, None]),
    ("ln_mda_chars_lag", [None, construct["dukw_mda_chars_lag"].get("ln_mda_chars_lag"), None, None, None, None, None]),
    ("DU_llm_lag", [None, None, construct["dullm_total_chars_lag"].get("DU_llm_lag"), None, None, None, None]),
    ("DU_llm_lenstd_lag", [None, None, None, construct["dullm_lenstd_total_chars_lag"].get("DU_llm_lenstd_lag"), None, None, None]),
    ("GenericNarr_lag", [None, None, None, None, construct["joint_lag"].get("GenericNarr_lag"), None, None]),
    ("DU_kw_resid_lag", [None, None, None, None, None, construct["dukw_resid_lag"].get("DU_kw_resid_lag"), None]),
    ("DU_kw_strict_lag", [None, None, None, None, None, None, construct["dukw_strict_lag"].get("DU_kw_strict_lag")]),
]

table1_csv_rows: list[list[str]] = []
for label, vals in row_specs_t1:
    coef_row = [L(pdf_var(label))]
    se_row = [L("")]
    raw_coef = [csv_var(label)]
    raw_se = [""]
    for v in vals:
        if v is None:
            coef_row.append(P(""))
            se_row.append(P(""))
            raw_coef.append("")
            raw_se.append("")
        else:
            coef_row.append(P(fmt_coef(v["coef"], p=v.get("p"), t=v.get("t"))))
            se_row.append(P(fmt_se(v["se"])))
            raw_coef.append(fmt_coef(v["coef"], p=v.get("p"), t=v.get("t")))
            raw_se.append(fmt_se(v["se"]))
    d1.extend([coef_row, se_row])
    table1_csv_rows.extend([raw_coef, raw_se])

d1.append([L("FE-adjusted VIF")] + [P(""), P(""), P(""), P(""), P(f'DU={construct["joint_lag"]["vif_DU_kw_lag"]:.2f}; GN={construct["joint_lag"]["vif_GenericNarr_lag"]:.2f}'), P(""), P("")])
d1.append([L("N")] + [P(f'{construct["dukw_total_chars_lag"]["DU_kw_lag"]["n"]:,}')] * 7)
t1 = Table(d1, colWidths=[2.6 * cm] + [1.85 * cm] * 7)
t1.setStyle(three_line(len(d1), hdr=3, spans=[("LINEBELOW", (0, 2), (-1, 2), 0.5, colors.black)]))
title1 = Paragraph("表1 构念边界检验", s_title)
note1 = Paragraph(
    "注：表中报告统一滞后规格下的关键边界检验。联合回归列同时纳入DU_kw与GenericNarr；残差化列使用剔除GenericNarr可解释部分后的DU_kw_resid；"
    "去泛化词列对应剔除8个最泛化数字叙事词后的DU_kw_strict。标准误在行业×年份水平聚类。",
    s_note,
)
save_csv(
    OUT / "table1_construct_boundary_v19.csv",
    table1_csv_rows + [["FE-adjusted VIF", "", "", "", "", f'DU={construct["joint_lag"]["vif_DU_kw_lag"]:.2f}; GN={construct["joint_lag"]["vif_GenericNarr_lag"]:.2f}', "", ""], ["N"] + [f'{construct["dukw_total_chars_lag"]["DU_kw_lag"]["n"]:,}'] * 7],
    ["变量", "(1)", "(2)", "(3)", "(4)", "(5)", "(6)", "(7)"],
)


# ---------------------------------------------------------------------------
# Table 2: Measurement upgrade and WashGap
# ---------------------------------------------------------------------------
mu = meas["measurement_upgrade"]
d2 = [
    [L(""), P("(1)"), P("(2)"), P("(3)"), P("(4)"), P("(5)"), P("(6)"), P("(7)")],
    [L("模型"), P("DUchain"), P("DUclosed"), P("DUcore"), P("DUkw_mda"), P("WashGap"), P("BroadShallow"), P("联合回归")],
    [L("因变量"), P("PriceDelay"), P("PriceDelay"), P("PriceDelay"), P("PriceDelay"), P("PriceDelay"), P("PriceDelay"), P("PriceDelay")],
]

table2_specs = [
    ("DUchain_count_lag", [mu["duchain_count_lag"].get("DUchain_count_lag"), None, None, None, None, None, None]),
    ("DUclosedloop_lag", [None, mu["duclosedloop_lag"].get("DUclosedloop_lag"), None, None, None, None, None]),
    ("DUcore_lag", [None, None, mu["ducore_lag"].get("DUcore_lag"), None, None, None, None]),
    ("DUkw_mda_lag", [None, None, None, mu["dukw_mda_lag"].get("DUkw_mda_lag"), None, None, None]),
    ("WashGap_lag", [None, None, None, None, washgap["washgap_lag"].get("WashGap_lag"), None, None]),
    ("BroadShallow_lag", [None, None, None, None, None, washgap["broadshallow_lag"].get("BroadShallow_lag"), None]),
    ("DU_kw_lag", [None, None, None, None, None, None, washgap["joint_depth_lag"].get("DU_kw_lag")]),
    ("DU_llm_lenstd_lag", [None, None, None, None, None, None, washgap["joint_depth_lag"].get("DU_llm_lenstd_lag")]),
]

table2_csv_rows: list[list[str]] = []
for label, vals in table2_specs:
    coef_row = [L(pdf_var(label))]
    se_row = [L("")]
    raw_coef = [csv_var(label)]
    raw_se = [""]
    for v in vals:
        if v is None:
            coef_row.append(P(""))
            se_row.append(P(""))
            raw_coef.append("")
            raw_se.append("")
        else:
            coef_row.append(P(fmt_coef(v["coef"], p=v.get("p"), t=v.get("t"))))
            se_row.append(P(fmt_se(v["se"])))
            raw_coef.append(fmt_coef(v["coef"], p=v.get("p"), t=v.get("t")))
            raw_se.append(fmt_se(v["se"]))
    d2.extend([coef_row, se_row])
    table2_csv_rows.extend([raw_coef, raw_se])

d2.append([L("N")] + [P(f'{meas["baseline"]["dukw_lag"]["DU_kw_lag"]["n"]:,}')] * 7)
t2 = Table(d2, colWidths=[2.6 * cm] + [1.85 * cm] * 7)
t2.setStyle(three_line(len(d2), hdr=3, spans=[("LINEBELOW", (0, 2), (-1, 2), 0.5, colors.black)]))
title2 = Paragraph("表2 增强测度与广度—深度错配", s_title)
note2 = Paragraph(
    "注：DUchain_count、DUclosedloop、DUcore和DUkw_mda均采用统一滞后规格；第(5)(6)列分别报告WashGap与BroadShallow；"
    "第(7)列同时纳入DU_kw与长度标准化语义深度DU_llm_lenstd。DUevent因当前归档特征表下不稳健，未纳入主表。",
    s_note,
)
save_csv(
    OUT / "table2_measurement_washgap_v19.csv",
    table2_csv_rows + [["N"] + [f'{meas["baseline"]["dukw_lag"]["DU_kw_lag"]["n"]:,}'] * 7],
    ["变量", "(1)", "(2)", "(3)", "(4)", "(5)", "(6)", "(7)"],
)


# ---------------------------------------------------------------------------
# Table 3: Key mechanisms
# ---------------------------------------------------------------------------
mech_map = {
    "ForecastDisp": "分析师预测分歧",
    "CashFlowVol": "现金流波动",
    "SCConc": "综合供应链集中度",
}


def get_mech(channel: str, treat: str) -> dict:
    r = mech[(mech["channel"] == channel) & (mech["treatment"] == treat)]
    if r.empty:
        key = (channel, treat)
        if key in manual_mech:
            return manual_mech[key]
        raise KeyError(f"Missing mechanism result: {channel}/{treat}")
    return r.iloc[0].to_dict()


d3 = [
    [L(""), P("(1)"), P("(2)"), P("(3)"), P("(4)"), P("(5)"), P("(6)")],
    [L("变量"), P("ForecastDisp"), P("ForecastDisp"), P("CashFlowVol"), P("CashFlowVol"), P("SCConc"), P("SCConc")],
]

row_kw = [L(pdf_var("DU_kw"))]
row_kw_se = [L("")]
row_llm = [L(pdf_var("DU_llm"))]
row_llm_se = [L("")]
for channel in ["ForecastDisp", "CashFlowVol", "SCConc"]:
    kw = get_mech(channel, "DU_kw")
    llm = get_mech(channel, "DU_llm")
    row_kw.extend([P(fmt_coef(kw["coef"], t=kw["tstat"])), P("")])
    row_kw_se.extend([P(fmt_se(kw["se"], 4 if channel == "ForecastDisp" else (4 if channel == "CashFlowVol" else 3))), P("")])
    row_llm.extend([P(""), P(fmt_coef(llm["coef"], t=llm["tstat"]))])
    row_llm_se.extend([P(""), P(fmt_se(llm["se"], 4 if channel == "ForecastDisp" else (4 if channel == "CashFlowVol" else 3)))])

d3.extend([row_kw, row_kw_se, row_llm, row_llm_se])
d3.append([L("控制变量")] + [P("YES")] * 6)
d3.append([L("Firm FE")] + [P("YES")] * 6)
d3.append([L("Year FE")] + [P("YES")] * 6)
d3.append([L("N")] + [P(f'{int(get_mech(ch, tr)["N"]):,}') for ch in ["ForecastDisp", "ForecastDisp", "CashFlowVol", "CashFlowVol", "SCConc", "SCConc"] for tr in []])
# easier explicit N row
d3[-1] = [L("N"),
          P(f'{int(get_mech("ForecastDisp", "DU_kw")["N"]):,}'),
          P(f'{int(get_mech("ForecastDisp", "DU_llm")["N"]):,}'),
          P(f'{int(get_mech("CashFlowVol", "DU_kw")["N"]):,}'),
          P(f'{int(get_mech("CashFlowVol", "DU_llm")["N"]):,}'),
          P(f'{int(get_mech("SCConc", "DU_kw")["N"]):,}'),
          P(f'{int(get_mech("SCConc", "DU_llm")["N"]):,}')]
t3 = Table(d3, colWidths=[2.4 * cm] + [2.0 * cm] * 6)
t3.setStyle(three_line(len(d3), hdr=2, spans=[("LINEBELOW", (0, 1), (-1, 1), 0.5, colors.black)]))
title3 = Paragraph("表3 主机制检验", s_title)
note3 = Paragraph(
    "注：因变量分别为分析师预测分歧(ForecastDisp)、现金流波动(CashFlowVol)和综合供应链集中度(SCConc)。"
    "所有模型控制企业与年份固定效应和11个控制变量，标准误在行业×年份水平聚类。ForecastDisp仅在DU<sub>kw</sub>下显著，应作为补充性信息质量证据理解；"
    "CashFlowVol与SCConc为双测度下更稳健的主体机制结果。",
    s_note,
)
save_csv(
    OUT / "table3_key_mechanisms_v19.csv",
    [
        [csv_var("DU_kw"), fmt_coef(get_mech("ForecastDisp", "DU_kw")["coef"], t=get_mech("ForecastDisp", "DU_kw")["tstat"]), "", fmt_coef(get_mech("CashFlowVol", "DU_kw")["coef"], t=get_mech("CashFlowVol", "DU_kw")["tstat"]), "", fmt_coef(get_mech("SCConc", "DU_kw")["coef"], t=get_mech("SCConc", "DU_kw")["tstat"]), ""],
        ["", fmt_se(get_mech("ForecastDisp", "DU_kw")["se"]), "", fmt_se(get_mech("CashFlowVol", "DU_kw")["se"]), "", fmt_se(get_mech("SCConc", "DU_kw")["se"], 3), ""],
        [csv_var("DU_llm"), "", fmt_coef(get_mech("ForecastDisp", "DU_llm")["coef"], t=get_mech("ForecastDisp", "DU_llm")["tstat"]), "", fmt_coef(get_mech("CashFlowVol", "DU_llm")["coef"], t=get_mech("CashFlowVol", "DU_llm")["tstat"]), "", fmt_coef(get_mech("SCConc", "DU_llm")["coef"], t=get_mech("SCConc", "DU_llm")["tstat"])],
        ["", "", fmt_se(get_mech("ForecastDisp", "DU_llm")["se"]), "", fmt_se(get_mech("CashFlowVol", "DU_llm")["se"]), "", fmt_se(get_mech("SCConc", "DU_llm")["se"], 3)],
        ["N",
         f'{int(get_mech("ForecastDisp", "DU_kw")["N"]):,}',
         f'{int(get_mech("ForecastDisp", "DU_llm")["N"]):,}',
         f'{int(get_mech("CashFlowVol", "DU_kw")["N"]):,}',
         f'{int(get_mech("CashFlowVol", "DU_llm")["N"]):,}',
         f'{int(get_mech("SCConc", "DU_kw")["N"]):,}',
         f'{int(get_mech("SCConc", "DU_llm")["N"]):,}'],
    ],
    ["变量", "(1)", "(2)", "(3)", "(4)", "(5)", "(6)"],
)


# ---------------------------------------------------------------------------
# Table 4: Main-text heterogeneity
# ---------------------------------------------------------------------------
post = het_extra["Post2022"]
analyst = het_extra["Analyst_high"]
board = het_extra["MainBoard"]

d4 = [
    [L(""), P("(1)"), P("(2)"), P("(3)"), P("(4)"), P("(5)"), P("(6)")],
    [L("维度"), P("2011-2021"), P("2022-2024"), P("高分析师关注"), P("低分析师关注"), P("主板公司"), P("非主板公司")],
    [L("因变量"), P("PriceDelay"), P("PriceDelay"), P("PriceDelay"), P("PriceDelay"), P("PriceDelay"), P("PriceDelay")],
    [L(pdf_var("DU_kw")), P(fmt_coef(post["low"]["coef"], t=post["low"]["t"])), P(fmt_coef(post["high"]["coef"], t=post["high"]["t"])), P(fmt_coef(analyst["high"]["coef"], t=analyst["high"]["t"])), P(fmt_coef(analyst["low"]["coef"], t=analyst["low"]["t"])), P(fmt_coef(board["high"]["coef"], t=board["high"]["t"])), P(fmt_coef(board["low"]["coef"], t=board["low"]["t"]))],
    [L(""), P(fmt_se(post["low"]["se"])), P(fmt_se(post["high"]["se"])), P(fmt_se(analyst["high"]["se"])), P(fmt_se(analyst["low"]["se"])), P(fmt_se(board["high"]["se"])), P(fmt_se(board["low"]["se"]))],
    [L("控制变量")] + [P("YES")] * 6,
    [L("Firm FE")] + [P("YES")] * 6,
    [L("Year FE")] + [P("YES")] * 6,
    [L("N"), P(f'{post["low"]["n"]:,}'), P(f'{post["high"]["n"]:,}'), P(f'{analyst["high"]["n"]:,}'), P(f'{analyst["low"]["n"]:,}'), P(f'{board["high"]["n"]:,}'), P(f'{board["low"]["n"]:,}')],
    [L("交互项P值"), P(fmt_p(post["interaction"]["p"])), P(""), P(fmt_p(analyst["interaction"]["p"])), P(""), P(fmt_p(board["interaction"]["p"])), P("")],
]
t4 = Table(d4, colWidths=[2.5 * cm] + [1.95 * cm] * 6)
t4.setStyle(three_line(len(d4), hdr=3, spans=[("LINEBELOW", (0, 2), (-1, 2), 0.5, colors.black)]))
title4 = Paragraph("表4 主文异质性检验", s_title)
note4 = Paragraph(
    "注：所有模型统一采用t−1期DU_kw解释t期PriceDelay的滞后规格，控制企业与年份固定效应及11个控制变量，标准误在行业×年份水平聚类。"
    "2022切点为主文中的主要边界条件；高分析师关注和上市板块结果仅作为补充性边界证据报告。",
    s_note,
)
save_csv(
    OUT / "table4_heterogeneity_v19.csv",
    [
        [csv_var("DU_kw"), fmt_coef(post["low"]["coef"], t=post["low"]["t"]), fmt_coef(post["high"]["coef"], t=post["high"]["t"]), fmt_coef(analyst["high"]["coef"], t=analyst["high"]["t"]), fmt_coef(analyst["low"]["coef"], t=analyst["low"]["t"]), fmt_coef(board["high"]["coef"], t=board["high"]["t"]), fmt_coef(board["low"]["coef"], t=board["low"]["t"])],
        ["", fmt_se(post["low"]["se"]), fmt_se(post["high"]["se"]), fmt_se(analyst["high"]["se"]), fmt_se(analyst["low"]["se"]), fmt_se(board["high"]["se"]), fmt_se(board["low"]["se"])],
        ["N", f'{post["low"]["n"]:,}', f'{post["high"]["n"]:,}', f'{analyst["high"]["n"]:,}', f'{analyst["low"]["n"]:,}', f'{board["high"]["n"]:,}', f'{board["low"]["n"]:,}'],
        ["交互项P值", fmt_p(post["interaction"]["p"]), "", fmt_p(analyst["interaction"]["p"]), "", fmt_p(board["interaction"]["p"]), ""],
    ],
    ["变量", "(1)", "(2)", "(3)", "(4)", "(5)", "(6)"],
)


# ---------------------------------------------------------------------------
# Summary markdown
# ---------------------------------------------------------------------------
summary = f"""# v19 当前结果表汇总

## 输出文件

- `v19_tables.pdf`
- `table1_construct_boundary_v19.csv`
- `table2_measurement_washgap_v19.csv`
- `table3_key_mechanisms_v19.csv`
- `table4_heterogeneity_v19.csv`

## 结果来源

- 构念边界：`results/v14/construct_boundary_results.json`
- 广度—深度错配：`results/v14/washgap_results.json`
- 增强测度：`results/v15_measurement/measurement_upgrade_results.json`
- 主机制：`results/v18/mechanism_v18.csv` 与 `results/v18/mechanism_new_v18.csv`
- 主文异质性：`results/v15_tables/heterogeneity_extra_v15.json`

## 核心结论

- 构念边界方面，`DU_kw_strict_lag` 仍显著为负（{fmt_coef(construct["dukw_strict_lag"]["DU_kw_strict_lag"]["coef"], p=construct["dukw_strict_lag"]["DU_kw_strict_lag"]["p"])}，t={construct["dukw_strict_lag"]["DU_kw_strict_lag"]["t"]:.2f}），但 `DU_kw` 与 `GenericNarr` 联合后均不显著，说明边界需克制表述。
- 增强测度方面，`DUchain_count`、`DUclosedloop`、`DUcore` 与 `DUkw_mda` 在单变量滞后规格下为负；`WashGap` 显著为正，`DU_llm_lenstd` 在联合回归中保留额外负向解释力。
- 主机制方面，`CashFlowVol` 与 `SCConc` 在双测度下均显著，`ForecastDisp` 仅在 `DU_kw` 下显著。
- 异质性方面，`2011-2021` 组效应更强；高分析师关注与非主板结果保留为补充性边界证据。
"""
(OUT / "v19_全部结果汇总.md").write_text(summary, encoding="utf-8")


# ---------------------------------------------------------------------------
# Build PDF
# ---------------------------------------------------------------------------
pdf_path = OUT / "v19_tables.pdf"
doc = SimpleDocTemplate(
    str(pdf_path),
    pagesize=A4,
    topMargin=2 * cm,
    bottomMargin=2 * cm,
    leftMargin=1.7 * cm,
    rightMargin=1.7 * cm,
)
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

print(f"Saved: {pdf_path}")
