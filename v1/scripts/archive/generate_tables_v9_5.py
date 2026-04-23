"""
生成朱康(2025)会计研究格式的全部表格PDF — v9.5
v9.5变更:
  - 中文字体改用宋体(Songti SC)以符合会计研究排版要求
  - 变量名DU_kw等添加下标渲染
  - 表头首列添加"变量"/"被解释变量="标签
  - 表3注释措辞调整
v9.4变更:
  - 表3: 新增第9列DU_kw_lead前导项检验
  - 表5: 机制检验替换为Analyst/Disp/absDA三渠道
  - 表7: 改为附表(附表1)
"""

import pandas as pd
import numpy as np
import pyfixest as pf
import json, warnings, os
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

# ============================================================
# 0. 字体注册 (v9.5: 宋体优先)
# ============================================================
from reportlab.pdfbase.pdfmetrics import registerFontFamily

CN_FONT = "Helvetica"

# 优先: macOS 宋体 (会计研究期刊标准字体)
SONGTI_PATH = "/System/Library/Fonts/Supplemental/Songti.ttc"
if os.path.exists(SONGTI_PATH):
    try:
        pdfmetrics.registerFont(TTFont('SongtiSC', SONGTI_PATH, subfontIndex=6))       # Regular
        pdfmetrics.registerFont(TTFont('SongtiSC-Bold', SONGTI_PATH, subfontIndex=1))   # Bold
        registerFontFamily('SongtiSC', normal='SongtiSC', bold='SongtiSC-Bold',
                           italic='SongtiSC', boldItalic='SongtiSC-Bold')
        CN_FONT = 'SongtiSC'
    except:
        pass

# 备选: NotoSansCJK / fc-list (Linux等环境)
if CN_FONT == "Helvetica":
    font_paths = [
        os.path.expanduser("~/Library/Fonts/NotoSansCJKsc-Regular.otf"),
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/noto-cjk/NotoSansCJKsc-Regular.otf",
    ]
    for fp in font_paths:
        if os.path.exists(fp):
            try:
                pdfmetrics.registerFont(TTFont('NotoSansCJK', fp, subfontIndex=0))
                CN_FONT = 'NotoSansCJK'
                break
            except:
                pass

if CN_FONT == "Helvetica":
    import subprocess
    result = subprocess.run(['fc-list', ':lang=zh', 'file'], capture_output=True, text=True)
    for line in result.stdout.strip().split('\n'):
        fp = line.split(':')[0].strip()
        if fp and os.path.exists(fp):
            try:
                pdfmetrics.registerFont(TTFont('CJKFont', fp, subfontIndex=0))
                CN_FONT = 'CJKFont'
                break
            except:
                try:
                    pdfmetrics.registerFont(TTFont('CJKFont', fp))
                    CN_FONT = 'CJKFont'
                    break
                except:
                    pass

print(f"Using font: {CN_FONT}")

# ============================================================
# 1. 数据加载 & 预处理
# ============================================================
print("Loading data...")
BASE = "/Users/mac/computerscience/15会计研究"
panel = pd.read_parquet(f"{BASE}/data_parquet/panel.parquet")
ar_feat = pd.read_parquet(f"{BASE}/data_parquet/annual_report_features.parquet")

panel = panel.merge(
    ar_feat[['Stkcd','year','kw_total','kw_per10k','substantive_count',
             'kw_data_stock','kw_data_dev','kw_data_app','kw_data_value','kw_data_gov']],
    on=['Stkcd','year'], how='left'
)

fi = pd.read_parquet(f"{BASE}/data_parquet/firm_info.parquet",
                     columns=['Symbol','EndDate','IndustryCodeC','LISTINGSTATE'])
fi = fi.rename(columns={'Symbol':'Stkcd'})
fi['EndDate'] = pd.to_datetime(fi['EndDate'])
fi['year'] = fi['EndDate'].dt.year
fi = fi.sort_values(['Stkcd','year','EndDate']).drop_duplicates(subset=['Stkcd','year'], keep='last')

# v9.1: 用公司众数填补IndustryCodeC缺失(修复Ind2缺失9512条的bug)
ind_mode = fi.groupby('Stkcd')['IndustryCodeC'].agg(
    lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else np.nan)
panel = panel.merge(ind_mode.rename('IndCode_mode').reset_index(), on='Stkcd', how='left')
panel = panel.merge(fi[['Stkcd','year','IndustryCodeC','LISTINGSTATE']], on=['Stkcd','year'], how='left')
panel['IndustryCodeC'] = panel['IndustryCodeC'].fillna(panel['IndCode_mode'])

# 样本筛选
mask_fin = panel['IndustryCodeC'].str.startswith('J', na=False)
mask_st = panel['LISTINGSTATE'].isin(['ST','*ST'])
mask_new = panel['Age'] <= 0
panel = panel[~mask_fin & ~mask_st & ~mask_new]

# 构造变量
panel['Ind2'] = panel['IndustryCodeC'].str[:3]
panel['IndYear'] = panel['Ind2'].astype(str) + '_' + panel['year'].astype(str)
panel['DU_kw'] = panel['kw_per10k']
panel['DU_kw_ln'] = np.log1p(panel['kw_total'])
panel['DU_sub_ln'] = np.log1p(panel['substantive_count'])

# SYNCH
synch = pd.read_parquet(f"{BASE}/data_parquet/price_synchronicity.parquet")
panel = panel.merge(synch[['Stkcd','year','SYNCH']], on=['Stkcd','year'], how='left')

# 主板标记
panel['MainBoard'] = panel['Stkcd'].astype(str).str.match(r'^(00[01]|6)')

# Winsorize
def winsorize(s):
    lo, hi = s.quantile([0.01, 0.99])
    return s.clip(lo, hi)

controls = ['Size','Lev','ROA','TobinQ','Age','Growth','BoardSize','IndepRatio',
            'Dual','Top1Share','SOE','InstHold','Amihud','Analyst','AuditType']

cont_vars = ['PriceDelay','DU_kw','DU_kw_ln','DU_sub_ln','SYNCH','FinAsset',
             'Lev','ROA','Growth','Size','TobinQ','Age','BoardSize','IndepRatio',
             'Top1Share','InstHold','Amihud','Analyst']
for v in cont_vars:
    if v in panel.columns and panel[v].notna().any():
        panel[v] = winsorize(panel[v])

reg = panel.dropna(subset=['PriceDelay','DU_kw'] + controls).copy()
reg['Stkcd_str'] = reg['Stkcd'].astype(str)
reg['year_str'] = reg['year'].astype(str)
print(f"Regression sample (pre-FE): N={len(reg):,}, firms={reg.Stkcd.nunique():,}")

ctrl_str = " + ".join(controls)

# ============================================================
# 2. Run ALL regressions
# ============================================================
print("Running regressions...")

def get_nobs(model): return model._N
def get_r2(model): return model._r2

def sig_stars(p):
    if p is None or pd.isna(p):
        return ""
    if p < 0.01: return "***"
    if p < 0.05: return "**"
    if p < 0.1:  return "*"
    return ""

def extract_coefs(model, vars_list):
    results = {}
    for var in vars_list:
        c = float(model.coef()[var])
        s = float(model.se()[var])
        p = float(model.pvalue()[var])
        results[var] = (c, s, sig_stars(p))
    return results

def get_intercept(model, data, y_var, x_vars):
    """从FE模型中还原截距项, delta method计算SE"""
    from scipy import stats as sp_stats
    try:
        beta = model.coef()
        y = data[y_var].values
        xb = np.zeros(len(data))
        for v in x_vars:
            if v in beta.index:
                xb += data[v].values * float(beta[v])
        intercept = np.mean(y - xb)
        # Delta method: SE(intercept) = sqrt(sigma2/N + xbar' V xbar)
        V = model._vcov
        xbar = np.array([float(data[v].mean()) for v in x_vars if v in beta.index])
        xVx = float(xbar @ V @ xbar)
        se = np.sqrt(model._rmse**2 / model._N + xVx)
        p = 2 * (1 - sp_stats.t.cdf(abs(intercept / se), model._N - model._k))
        return intercept, se, sig_stars(p)
    except:
        return None, None, ""

# ---- Baseline models ----
print("  Baseline models...")

m0 = pf.feols(f"PriceDelay ~ DU_kw | Stkcd_str + year_str",
              data=reg, vcov={"CRV1":"IndYear"})
m0_main = extract_coefs(m0, ["DU_kw"])
m0_N, m0_R2 = get_nobs(m0), get_r2(m0)

m1 = pf.feols(f"PriceDelay ~ DU_kw + {ctrl_str} | Stkcd_str + year_str",
              data=reg, vcov={"CRV1":"IndYear"})
m1_coefs = extract_coefs(m1, ["DU_kw"] + controls)
m1_N, m1_R2 = get_nobs(m1), get_r2(m1)

m2 = pf.feols(f"PriceDelay ~ DU_kw_ln + {ctrl_str} | Stkcd_str + year_str",
              data=reg, vcov={"CRV1":"IndYear"})
m2_main = extract_coefs(m2, ["DU_kw_ln"])
m2_coefs = extract_coefs(m2, ["DU_kw_ln"] + controls)
m2_N, m2_R2 = get_nobs(m2), get_r2(m2)

m3 = pf.feols(f"PriceDelay ~ DU_sub_ln + {ctrl_str} | Stkcd_str + year_str",
              data=reg, vcov={"CRV1":"IndYear"})
m3_main = extract_coefs(m3, ["DU_sub_ln"])
m3_coefs = extract_coefs(m3, ["DU_sub_ln"] + controls)
m3_N, m3_R2 = get_nobs(m3), get_r2(m3)

# v9.1: 用同一样本reg(Ind2已填补), 不再dropna
m4 = pf.feols(f"PriceDelay ~ DU_kw + {ctrl_str} | Ind2 + year_str",
              data=reg, vcov={"CRV1":"IndYear"})
m4_main = extract_coefs(m4, ["DU_kw"])
m4_coefs = extract_coefs(m4, ["DU_kw"] + controls)
m4_N, m4_R2 = get_nobs(m4), get_r2(m4)

reg_synch = reg.dropna(subset=['SYNCH']).copy()
m5 = pf.feols(f"SYNCH ~ DU_kw + {ctrl_str} | Stkcd_str + year_str",
              data=reg_synch, vcov={"CRV1":"IndYear"})
m5_main = extract_coefs(m5, ["DU_kw"])
m5_coefs = extract_coefs(m5, ["DU_kw"] + controls)
m5_N, m5_R2 = get_nobs(m5), get_r2(m5)

m6 = pf.feols(f"PriceDelay ~ DU_kw + FinAsset + {ctrl_str} | Stkcd_str + year_str",
              data=reg, vcov={"CRV1":"IndYear"})
m6_main = extract_coefs(m6, ["DU_kw", "FinAsset"])
m6_coefs = extract_coefs(m6, ["DU_kw", "FinAsset"] + controls)
m6_N, m6_R2 = get_nobs(m6), get_r2(m6)

# v9.1: 不再手算Constant (FE模型截距由reghdfe处理, 参见Stata do文件)
print(f"  Baseline done. m0={m0_main['DU_kw'][0]:.4f} N={m0_N}, m1={m1_coefs['DU_kw'][0]:.4f} N={m1_N}")

# ---- Robustness models (v9.2: Stata reghdfe verified results) ----
print("  Robustness models (Stata-verified hardcoded)...")

# Table 3 results from Stata reghdfe (see scripts/table3_robustness_stata.do)
# Format: (coef, se, sig, N, R2)
rob_stata = {
    'indyear':  (-0.0022, 0.0006, '***', 43778, 0.478),  # (1) Ind*Year FE
    'provyear': (-0.0041, 0.0009, '***', 43842, 0.435),  # (2) Prov*Year FE
    'indadj':   (-0.0020, 0.0006, '***', 43843, 0.424),  # (3) Industry-year mean adj
    'no2024':   (-0.0047, 0.0010, '***', 38916, 0.434),  # (4) Drop 2024
    'noit':     (-0.0040, 0.0009, '***', 40720, 0.425),  # (5) Drop IT industry
    'mainboard':(-0.0028, 0.0013, '**',  18837, 0.417),  # (6) Main board only
    'psm':      (-0.0040, 0.0009, '***', 34425, 0.434),  # (7) PSM matched
    'twoway':   (-0.0042, 0.0012, '***', 43843, 0.425),  # (8) Two-way cluster SE
    'lead':     (-0.0047, 0.0012, '***', 38567, 0.434),  # (9) DU_kw_lead placebo: DU_kw coef when controlling for lead
}
# DU_kw_lead placebo details: DU_kw_lead=0.0005(SE=0.0010) n.s. when DU_kw is included
rob_lead_detail = {'DU_kw_lead_coef': 0.0005, 'DU_kw_lead_se': 0.0010, 'DU_kw_lead_sig': ''}

# IV results (still from stored v4 results)
iv_results = json.load(open(f"{BASE}/results/endogeneity_v4/endogeneity_v4_results.json"))
rob_iv_coef = iv_results['peer_iv']['coef']
rob_iv_se = iv_results['peer_iv']['se']
rob_iv_p = iv_results['peer_iv']['p']
rob_iv_sig = "***" if rob_iv_p<0.01 else "**" if rob_iv_p<0.05 else "*" if rob_iv_p<0.1 else ""
rob_iv_N = iv_results['peer_iv']['N']
rob_iv_KPF = iv_results['peer_iv']['KP_F']

print("  Robustness done.")

# ---- Mechanism models (v9.4: hardcoded from Stata reghdfe on reg_sample_v4.dta) ----
print("  Mechanism models (v9.4: Analyst/Disp/absDA, Stata hardcoded)...")
mech_dvs = ['Analyst', 'Disp', 'absDA']
mech_results = {
    'Analyst': {"coef": 0.0255, "se": 0.0081, "sig": "***", "N": 43843, "R2": 0.794},
    'Disp':    {"coef": -0.0064, "se": 0.0015, "sig": "***", "N": 25092, "R2": 0.490},
    'absDA':   {"coef": -0.0005, "se": 0.0003, "sig": "*",   "N": 42542, "R2": 0.268},
}
# X+M->Y (b path): M coefficient when both DU_kw and M are in the PriceDelay regression
mech_b_path = {
    'Analyst': {"coef": -0.0103, "se": 0.0008, "sig": "***", "N": 43843, "R2": 0.432,
                "DU_kw_coef": -0.0042, "DU_kw_se": 0.0009, "DU_kw_sig": "***"},
    'Disp':    {"coef": 0.0167, "se": 0.0050, "sig": "***", "N": 25092, "R2": 0.436,
                "DU_kw_coef": -0.0047, "DU_kw_se": 0.0011, "DU_kw_sig": "***"},
    'absDA':   {"coef": 0.0345, "se": 0.0141, "sig": "**",  "N": 42542, "R2": 0.425,
                "DU_kw_coef": -0.0040, "DU_kw_se": 0.0009, "DU_kw_sig": "***"},
}
# Sobel test results
sobel_results = {
    'Analyst': {"indirect": -0.000262, "Z": -3.056, "p": 0.002, "pct": 5.9},
    'Disp':    {"indirect": -0.000107, "Z": -2.624, "p": 0.009, "pct": 2.2},
    'absDA':   {"indirect": -0.000018, "Z": -1.507, "p": 0.132, "pct": 0.4},
}
print("  Mechanism done.")

# ---- Heterogeneity models ----
# v9.3: 3 dimensions (SOE, InstHold, Analyst) with Fisher permutation p-values
print("  Heterogeneity models (v9.3: 3 dimensions)...")
base_fml = f"PriceDelay ~ DU_kw + {ctrl_str} | Stkcd_str + year_str"
het_results = {}

# Panel A: 分组回归 — 3个维度6列
# 维度1: 产权性质
for label, mask in [("国有企业", reg['SOE']==1), ("非国有企业", reg['SOE']==0)]:
    sub = reg[mask].copy()
    m = pf.feols(base_fml, data=sub, vcov={"CRV1":"IndYear"})
    c, s, sig = extract_coefs(m, ["DU_kw"])["DU_kw"]
    het_results[label] = {"coef": c, "se": s, "sig": sig, "N": get_nobs(m), "R2": get_r2(m)}

# 维度2: 机构持股
med_insthold = reg['InstHold'].median()
for label, mask in [("高机构持股", reg['InstHold']>=med_insthold), ("低机构持股", reg['InstHold']<med_insthold)]:
    sub = reg[mask].copy()
    m = pf.feols(base_fml, data=sub, vcov={"CRV1":"IndYear"})
    c, s, sig = extract_coefs(m, ["DU_kw"])["DU_kw"]
    het_results[label] = {"coef": c, "se": s, "sig": sig, "N": get_nobs(m), "R2": get_r2(m)}

# 维度3: 分析师覆盖
med_analyst = reg['Analyst'].median()
for label, mask in [("高分析师覆盖", reg['Analyst']>=med_analyst), ("低分析师覆盖", reg['Analyst']<med_analyst)]:
    sub = reg[mask].copy()
    m = pf.feols(base_fml, data=sub, vcov={"CRV1":"IndYear"})
    c, s, sig = extract_coefs(m, ["DU_kw"])["DU_kw"]
    het_results[label] = {"coef": c, "se": s, "sig": sig, "N": get_nobs(m), "R2": get_r2(m)}

# v9.3: Fisher permutation test p-values (Stata验证, 1000次置换)
# See results/v9_tables/stata_het/fisher_test.log
fisher_p = {
    '产权性质': 0.020,      # SOE: p=0.020**
    '机构持股': 0.015,      # HighInstHold: p=0.015**
    '分析师覆盖': 0.070,    # HighAnalyst: p=0.070*
}
print(f"  Fisher permutation p-values: {fisher_p}")

# Panel B: 连续变量交互项调节效应
print("  Continuous interaction models...")
cont_interact_results = {}
for mod_var, mod_label in [('TobinQ', 'TobinQ'), ('Amihud', 'Amihud')]:
    fml = f"PriceDelay ~ DU_kw + DU_kw:{mod_var} + {ctrl_str} | Stkcd_str + year_str"
    m = pf.feols(fml, data=reg, vcov={"CRV1":"IndYear"})
    interact_key = f"DU_kw:{mod_var}"
    c_main = float(m.coef()['DU_kw'])
    se_main = float(m.se()['DU_kw'])
    p_main = float(m.pvalue()['DU_kw'])
    c_inter = float(m.coef()[interact_key])
    se_inter = float(m.se()[interact_key])
    p_inter = float(m.pvalue()[interact_key])
    cont_interact_results[mod_label] = {
        'coef_main': c_main, 'se_main': se_main, 'sig_main': sig_stars(p_main),
        'coef_inter': c_inter, 'se_inter': se_inter, 'sig_inter': sig_stars(p_inter),
        'N': get_nobs(m), 'R2': get_r2(m)
    }
    print(f"    DU_kw x {mod_label}: coef={c_inter:.5f}, p={p_inter:.4f}")

print("  Heterogeneity done.")

# ============================================================
# 3. 生成PDF
# ============================================================
print("\nGenerating PDF...")

pdf_path = f"{BASE}/results/v9_tables/regression_tables_v9_5.pdf"
os.makedirs(os.path.dirname(pdf_path), exist_ok=True)

doc = SimpleDocTemplate(pdf_path, pagesize=A4,
                       leftMargin=1.5*cm, rightMargin=1.5*cm,
                       topMargin=2*cm, bottomMargin=2*cm)

# Custom styles
title_style = ParagraphStyle('TableTitle', fontName=CN_FONT, fontSize=10,
                             leading=14, alignment=TA_CENTER, spaceAfter=6)
note_style = ParagraphStyle('TableNote', fontName=CN_FONT, fontSize=7.5,
                            leading=10, alignment=TA_LEFT, spaceAfter=12)
cell_style = ParagraphStyle('CellText', fontName=CN_FONT, fontSize=8.5,
                            leading=11, alignment=TA_CENTER)
cell_left = ParagraphStyle('CellLeft', fontName=CN_FONT, fontSize=8.5,
                           leading=11, alignment=TA_LEFT)
header_style = ParagraphStyle('HeaderText', fontName=CN_FONT, fontSize=8.5,
                              leading=11, alignment=TA_CENTER)

def P(text, style=cell_style):
    return Paragraph(str(text), style)

# v9.5: 变量名下标映射
VAR_DISPLAY = {
    'DU_kw':           'DU<sub>kw</sub>',
    'DU_kw_ln':        'ln(1+DU<sub>kw</sub>)',
    'DU_sub_ln':       'DU<sub>sub</sub>(ln)',
    'DU_kw_adj':       'DU<sub>kw,adj</sub>',
    'DU_kw_lead':      'DU<sub>kw,lead</sub>',
    'DU_kw_lag':       'DU<sub>kw,lag</sub>',
    'DU_kw_peer':      'DU<sub>kw,peer</sub>',
    'DU_kw \u00d7 TobinQ':  'DU<sub>kw</sub> \u00d7 TobinQ',
    'DU_kw \u00d7 Amihud':  'DU<sub>kw</sub> \u00d7 Amihud',
    'Bartik_IV':       'Bartik<sub>IV</sub>',
    'Peer_lag':        'Peer<sub>lag</sub>',
}

def P_var(text):
    display = VAR_DISPLAY.get(text, text)
    return Paragraph(f"<i>{display}</i>", ParagraphStyle('Var', fontName='Times-Italic',
                     fontSize=8.5, leading=11, alignment=TA_LEFT))

def P_coef(c, sig, dec=3, superscript=True):
    txt = f"{c:.{dec}f}"
    if sig:
        if superscript:
            txt += f"<super>{sig}</super>"
        else:
            txt += sig
    return Paragraph(txt, ParagraphStyle('Coef', fontName='Times-Roman',
                     fontSize=8.5, leading=11, alignment=TA_CENTER))

def P_se(s, dec=3):
    return Paragraph(f"({s:.{dec}f})", ParagraphStyle('SE', fontName='Times-Roman',
                     fontSize=8.5, leading=11, alignment=TA_CENTER))

def P_num(n, dec=3):
    return Paragraph(f"{n:.{dec}f}", ParagraphStyle('Num', fontName='Times-Roman',
                     fontSize=8.5, leading=11, alignment=TA_CENTER))

def P_int(n):
    return Paragraph(f"{int(n):,}", ParagraphStyle('Int', fontName='Times-Roman',
                     fontSize=8.5, leading=11, alignment=TA_CENTER))

def P_hdr(text):
    return Paragraph(text, ParagraphStyle('Hdr', fontName=CN_FONT,
                     fontSize=7.5, leading=9.5, alignment=TA_CENTER))

story = []

# ============================================================
# 表1: 描述性统计
# ============================================================
print("  Table 1...")

desc_vars = ['PriceDelay','DU_kw','DU_kw_ln','Size','Lev','ROA','TobinQ',
             'Age','Growth','BoardSize','IndepRatio','Dual','Top1Share',
             'SOE','InstHold','Amihud','Analyst','AuditType']

# Use baseline model N for consistency
baseline_N = m1_N

t1_data = [[P('变量', cell_left), P('N', header_style), P('Mean', header_style),
            P('SD', header_style), P('Min', header_style),
            P('Median', header_style), P('Max', header_style)]]

for var in desc_vars:
    s = reg[var].dropna()
    t1_data.append([
        P_var(var), P_int(baseline_N),
        P_num(s.mean()), P_num(s.std()),
        P_num(s.min()), P_num(s.median()), P_num(s.max()),
    ])

table1 = Table(t1_data, colWidths=[3*cm, 2*cm, 2.2*cm, 2.2*cm, 2.2*cm, 2.2*cm, 2.2*cm])
table1.setStyle(TableStyle([
    ('FONT', (0,0), (-1,-1), CN_FONT, 8.5),
    ('ALIGN', (0,0), (-1,-1), 'CENTER'), ('ALIGN', (0,0), (0,-1), 'LEFT'),
    ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ('TOPPADDING', (0,0), (-1,-1), 2), ('BOTTOMPADDING', (0,0), (-1,-1), 2),
    ('LINEABOVE', (0,0), (-1,0), 1.2, colors.black),
    ('LINEBELOW', (0,0), (-1,0), 0.5, colors.black),
    ('LINEBELOW', (0,-1), (-1,-1), 1.2, colors.black),
]))

story.append(Paragraph("表1　　　　　　　　　　　变量的描述性统计", title_style))
story.append(table1)
story.append(Spacer(1, 6))
story.append(PageBreak())

# ============================================================
# 表2: 基准回归 (7 models)
# ============================================================
print("  Table 2...")

t2_data = []
# Header (1)-(7)
t2_data.append([P('', cell_left)] + [P(f'({i})', header_style) for i in range(1, 8)])
# DV row
dvs2 = ['PriceDelay']*5 + ['SYNCH', 'PriceDelay']
t2_data.append([P('被解释变量=', cell_left)] + [P(f'<i>{dv}</i>', ParagraphStyle('dv', fontName='Times-Italic',
               fontSize=8, leading=10, alignment=TA_CENTER)) for dv in dvs2])

# DU_kw row: columns (1)(2)(5)(6)(7) have DU_kw, others blank
du_kw_cols = {0: m0_main['DU_kw'], 1: m1_coefs['DU_kw'],
              4: m4_main['DU_kw'], 5: m5_main['DU_kw'], 6: m6_main['DU_kw']}
cr = [P_var('DU_kw')]
sr = [P('', cell_left)]
for i in range(7):
    if i in du_kw_cols:
        c, s, sig = du_kw_cols[i]
        # v9.1: model(5) SE very small (0.0005), use 4 dec to avoid (0.000)
        se_dec = 4 if i == 4 else 3
        cr.append(P_coef(c, sig))
        sr.append(P_se(s, dec=se_dec))
    else:
        cr.append(P('', cell_style))
        sr.append(P('', cell_style))
t2_data.append(cr)
t2_data.append(sr)

# DU_kw_ln row: only column (3)
cr_ln = [P_var('DU_kw_ln')]
sr_ln = [P('', cell_left)]
for i in range(7):
    if i == 2:
        c, s, sig = m2_main['DU_kw_ln']
        cr_ln.append(P_coef(c, sig))
        sr_ln.append(P_se(s))
    else:
        cr_ln.append(P('', cell_style))
        sr_ln.append(P('', cell_style))
t2_data.append(cr_ln)
t2_data.append(sr_ln)

# DU_sub_ln row: only column (4)
cr_sub = [P_var('DU_sub_ln')]
sr_sub = [P('', cell_left)]
for i in range(7):
    if i == 3:
        c, s, sig = m3_main['DU_sub_ln']
        cr_sub.append(P_coef(c, sig))
        sr_sub.append(P_se(s))
    else:
        cr_sub.append(P('', cell_style))
        sr_sub.append(P('', cell_style))
t2_data.append(cr_sub)
t2_data.append(sr_sub)

# FinAsset for model (7) only
fa_c_row = [P_var('FinAsset')]
fa_s_row = [P('', cell_left)]
for i in range(7):
    if i == 6:
        c, s, sig = m6_main['FinAsset']
        fa_c_row.append(P_coef(c, sig))
        fa_s_row.append(P_se(s))
    else:
        fa_c_row.append(P('', cell_style))
        fa_s_row.append(P('', cell_style))
t2_data.append(fa_c_row)
t2_data.append(fa_s_row)

# v9.1: Control variables with smart decimal places
# Top1Share, SOE, InstHold use 4 decimals; others use 3
vars_4dec = {'Top1Share', 'SOE', 'InstHold'}
all_ctrl_coefs = [None, m1_coefs, m2_coefs, m3_coefs, m4_coefs, m5_coefs, m6_coefs]
for ctrl in controls:
    dec = 4 if ctrl in vars_4dec else 3
    cr = [P_var(ctrl)]
    sr = [P('', cell_left)]
    for i in range(7):
        if i == 0:  # m0 no controls
            cr.append(P('', cell_style))
            sr.append(P('', cell_style))
        else:
            c, s, sig = all_ctrl_coefs[i][ctrl]
            cr.append(P_coef(c, sig, dec=dec))
            sr.append(P_se(s, dec=dec))
    t2_data.append(cr)
    t2_data.append(sr)

# v9.1: 不再报告手算Constant (FE吸收, 参见Stata reghdfe验证)
# Bottom rows: 加Controls YES/NO行
t2_data.append([P('Controls', cell_left)] + [P(v, cell_style) for v in ['NO','YES','YES','YES','YES','YES','YES']])
t2_data.append([P('Firm/Year FE', cell_left)] + [P(v, cell_style) for v in ['YES','YES','YES','YES','NO','YES','YES']])
t2_data.append([P('Ind/Year FE', cell_left)] + [P(v, cell_style) for v in ['NO','NO','NO','NO','YES','NO','NO']])
t2_data.append([P_var('N')] + [P_int(n) for n in [m0_N, m1_N, m2_N, m3_N, m4_N, m5_N, m6_N]])
t2_data.append([P_var('R<super>2</super>')] + [P_num(r) for r in [m0_R2, m1_R2, m2_R2, m3_R2, m4_R2, m5_R2, m6_R2]])

table2 = Table(t2_data, colWidths=[2.6*cm] + [2.05*cm]*7)
table2.setStyle(TableStyle([
    ('FONT', (0,0), (-1,-1), CN_FONT, 8.5),
    ('ALIGN', (0,0), (-1,-1), 'CENTER'), ('ALIGN', (0,0), (0,-1), 'LEFT'),
    ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ('TOPPADDING', (0,0), (-1,-1), 1.5), ('BOTTOMPADDING', (0,0), (-1,-1), 1.5),
    ('LINEABOVE', (0,0), (-1,0), 1.2, colors.black),
    ('LINEBELOW', (0,0), (-1,0), 0.5, colors.black),
    ('LINEBELOW', (0,1), (-1,1), 0.5, colors.black),
    ('LINEBELOW', (0,-1), (-1,-1), 1.2, colors.black),
]))

story.append(Paragraph("表2　　　　　　　　　　　基准回归检验", title_style))
story.append(table2)
story.append(Spacer(1, 4))
story.append(Paragraph(
    "注：括号内数值为经行业×年份层面聚类调整后的稳健标准误；<super>***</super>、<super>**</super>、<super>*</super>分别表示1%、5%、10%的水平上显著，下同。"
    "模型(1)仅控制固定效应不含控制变量，模型(3)解释变量为ln(1+关键词总数)，模型(4)为ln(1+实质利用次数)，"
    "模型(5)控制行业和年份固定效应，模型(6)被解释变量为股价同步性SYNCH，模型(7)额外控制金融资产占比。",
    note_style))
story.append(PageBreak())

# ============================================================
# 表3: 稳健性检验 (v9.2: Stata验证, 8列新结构)
# ============================================================
print("  Table 3...")

rob_labels = [
    "(1) 控制<br/>Ind×Year",
    "(2) 控制<br/>Prov×Year",
    "(3) 行业均<br/>值调整",
    "(4) 剔除<br/>2024年",
    "(5) 剔除信息<br/>技术业",
    "(6) 仅<br/>主板",
    "(7) 倾向得<br/>分匹配",
    "(8) 双向<br/>聚类SE",
    "(9) 前导项<br/>检验",
]
rob_dvs = ['PriceDelay']*9

rob_keys = ['indyear','provyear','indadj','no2024','noit','mainboard','psm','twoway','lead']
rob_models = [rob_stata[k] for k in rob_keys]

# Variable display names per column
rob_var_names = ['DU_kw','DU_kw','DU_kw_adj','DU_kw','DU_kw','DU_kw','DU_kw','DU_kw','DU_kw']

# FE structure rows
rob_fe_firm =    ['YES','YES','YES','YES','YES','YES','YES','YES','YES']
rob_fe_year =    ['NO', 'NO', 'YES','YES','YES','YES','YES','YES','YES']
rob_fe_indyear = ['YES','NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO']
rob_fe_provyear =['NO', 'YES','NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO']

t3_data = []
t3_data.append([P('', cell_left)] + [P_hdr(l) for l in rob_labels])
t3_data.append([P('被解释变量=', cell_left)] + [P(f'<i>{dv}</i>', ParagraphStyle(f'dv3_{i}', fontName='Times-Italic',
                fontSize=7.5, leading=9, alignment=TA_CENTER)) for i, dv in enumerate(rob_dvs)])

# Variable name row + coef (use 4 decimals for small values in cols 1,3)
cr = [P('', cell_left)]
sr = [P('', cell_left)]
for i, (c, s, sig, n, r2) in enumerate(rob_models):
    d = 4 if i in (0, 2) else 3
    cr.append(P_coef(c, sig, dec=d))
    sr.append(P_se(s, dec=d))
# Insert per-column variable names as a separate row
var_row = [P('', cell_left)] + [P_var(v) for v in rob_var_names]
t3_data.append(var_row)
t3_data.append(cr)
t3_data.append(sr)

# Col 9 extra row: DU_kw_lead coefficient
lead_cr = [P('', cell_left)] + [P('', cell_style)]*8 + \
          [P_coef(rob_lead_detail['DU_kw_lead_coef'], rob_lead_detail['DU_kw_lead_sig'], 4)]
lead_var = [P('', cell_left)] + [P('', cell_style)]*8 + [P_var('DU_kw_lead')]
lead_sr = [P('', cell_left)] + [P('', cell_style)]*8 + \
          [P_se(rob_lead_detail['DU_kw_lead_se'], 4)]
t3_data.append(lead_var)
t3_data.append(lead_cr)
t3_data.append(lead_sr)

t3_data.append([P('Controls', cell_left)] + [P('YES', cell_style)]*9)
t3_data.append([P('Firm FE', cell_left)] + [P(v, cell_style) for v in rob_fe_firm])
t3_data.append([P('Year FE', cell_left)] + [P(v, cell_style) for v in rob_fe_year])
t3_data.append([P('Ind×Year FE', cell_left)] + [P(v, cell_style) for v in rob_fe_indyear])
t3_data.append([P('Prov×Year FE', cell_left)] + [P(v, cell_style) for v in rob_fe_provyear])
t3_data.append([P_var('N')] + [P_int(n) for c,s,sig,n,r2 in rob_models])
t3_data.append([P_var('R<super>2</super>')] + [P_num(r2) for c,s,sig,n,r2 in rob_models])

table3 = Table(t3_data, colWidths=[2.2*cm] + [1.75*cm]*9)
table3.setStyle(TableStyle([
    ('FONT', (0,0), (-1,-1), CN_FONT, 7),
    ('ALIGN', (0,0), (-1,-1), 'CENTER'), ('ALIGN', (0,0), (0,-1), 'LEFT'),
    ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ('TOPPADDING', (0,0), (-1,-1), 1.5), ('BOTTOMPADDING', (0,0), (-1,-1), 1.5),
    ('LINEABOVE', (0,0), (-1,0), 1.2, colors.black),
    ('LINEBELOW', (0,0), (-1,0), 0.5, colors.black),
    ('LINEBELOW', (0,2), (-1,2), 0.5, colors.black),
    ('LINEBELOW', (0,-1), (-1,-1), 1.2, colors.black),
]))

story.append(Paragraph("表3　　　　　　　　　　　　　　稳健性检验", title_style))
story.append(table3)
story.append(Spacer(1, 4))
story.append(Paragraph(
    "注：括号内数值为稳健标准误(Stata reghdfe验证)。"
    "模型(1)控制企业和行业×年份交互固定效应，模型(2)控制企业和省份×年份交互固定效应，"
    "年份效应已被交互固定效应吸收。"
    "模型(3)的DU_kw_adj为行业年度均值调整后的DU_kw，即DU_kw减去同行业同年度样本均值。"
    "模型(7)以DU_kw中位数划分处理组，经Logit倾向得分1:1近邻匹配(caliper=0.05)后，"
    "在匹配样本上以连续DU_kw回归。"
    "模型(8)使用企业和年份双向聚类标准误。"
    "模型(9)同时纳入DU_kw和DU_kw_lead(t+1期)，DU_kw_lead系数不显著(t=0.57)，缓解反向因果担忧。", note_style))
story.append(PageBreak())

# ============================================================
# 表4: 内生性检验 (v9.3: ivreghdfe统一一二阶段, 样本完全一致)
# ============================================================
print("  Table 4...")

oster = iv_results['oster']

# All IV results from Stata ivreghdfe (unified 1st/2nd stage samples)
# First-stage: Format: (IV_name, coef, se, sig, N, R2)
fs_stata = {
    'peer':   ('DU_kw_peer', 0.5970, 0.0367, '***', 43779, None),
    'bartik': ('bartik_iv',  0.2889, 0.0365, '***', 43058, None),
    'lag':    ('peer_lag',   0.6018, 0.0444, '***', 37340, None),
}

# Second-stage: all from ivreghdfe (Stata), N matches first stage exactly
# Format: (coef, se, sig, N, R2)
iv_models_data = [
    (-0.004157, 0.000899, '***', 43843, 0.425),      # (1) OLS
    (-0.01470, 0.00459, '***', 43779, None),           # (2) Peer IV
    (-0.01116, 0.00644, '*', 43058, None),             # (3) Bartik IV
    (-0.003614, 0.000986, '***', 37404, None),         # (4) Lag OLS
    (-0.01647, 0.00526, '***', 37340, None),           # (5) Lag IV
]

# KP F and DWH p from ivreghdfe
iv_kp_f = {2: 265.1, 3: 62.5, 5: 183.5}
iv_dwh_p = {2: 0.009, 3: 0.268, 5: 0.004}

# ---- Panel A: First Stage ----
t4a_data = []
t4a_data.append([P('<b>Panel A: First Stage</b>', cell_left)] + [P('', cell_style)]*2)
# Column headers: (2) Peer IV, (3) Bartik IV, (5) Lag IV
fs_labels = ["(2) Peer IV", "(3) Bartik IV", "(5) Lag IV"]
t4a_data.append([P('', cell_left)] + [P_hdr(l) for l in fs_labels])
# DV row
t4a_data.append([P('被解释变量=', cell_left)] + [P('<i>DU<sub>kw</sub></i>', ParagraphStyle('dv', fontName='Times-Italic',
                fontSize=8, leading=10, alignment=TA_CENTER)),
                P('<i>DU<sub>kw</sub></i>', ParagraphStyle('dv2', fontName='Times-Italic',
                fontSize=8, leading=10, alignment=TA_CENTER)),
                P('<i>DU<sub>kw,lag</sub></i>', ParagraphStyle('dv3', fontName='Times-Italic',
                fontSize=8, leading=10, alignment=TA_CENTER))])

# IV variable names row
fs_keys = ['peer', 'bartik', 'lag']
fs_var_names = ['DU_kw_peer', 'Bartik_IV', 'Peer_lag']
t4a_data.append([P('', cell_left)] + [P_var(v) for v in fs_var_names])

# IV coefficients
cr = [P('', cell_left)]
sr = [P('', cell_left)]
for k in fs_keys:
    _, c, s, sig, _, _ = fs_stata[k]
    cr.append(P_coef(c, sig, 4))
    sr.append(P_se(s, 4))
t4a_data.append(cr)
t4a_data.append(sr)

for label in ['Controls', 'Firm/Year']:
    t4a_data.append([P(label, cell_left)] + [P('YES', cell_style)]*3)
t4a_data.append([P_var('N')] + [P_int(fs_stata[k][4]) for k in fs_keys])
t4a_data.append([P_var('R<super>2</super>')] + [P_num(fs_stata[k][5]) if fs_stata[k][5] is not None else P('—', cell_style) for k in fs_keys])

table4a = Table(t4a_data, colWidths=[2.5*cm] + [4.2*cm]*3)
table4a.setStyle(TableStyle([
    ('FONT', (0,0), (-1,-1), CN_FONT, 8.5),
    ('ALIGN', (0,0), (-1,-1), 'CENTER'), ('ALIGN', (0,0), (0,-1), 'LEFT'),
    ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ('TOPPADDING', (0,0), (-1,-1), 1.5), ('BOTTOMPADDING', (0,0), (-1,-1), 1.5),
    ('LINEABOVE', (0,0), (-1,0), 1.2, colors.black),
    ('LINEBELOW', (0,0), (-1,0), 0.5, colors.black),
    ('LINEBELOW', (0,2), (-1,2), 0.5, colors.black),
    ('LINEBELOW', (0,-1), (-1,-1), 0.5, colors.black),
]))

# ---- Panel B: Second Stage ----
t4b_data = []
t4b_data.append([P('<b>Panel B: Second Stage</b>', cell_left)] + [P('', cell_style)]*4)
iv_labels = ["(1) OLS", "(2) Peer IV", "(3) Bartik IV", "(4) Lag OLS", "(5) Lag IV"]
t4b_data.append([P('', cell_left)] + [P_hdr(l) for l in iv_labels])
t4b_data.append([P('被解释变量=', cell_left)] + [P(f'<i>PriceDelay</i>', ParagraphStyle('dv4b', fontName='Times-Italic',
                fontSize=8, leading=10, alignment=TA_CENTER)) for _ in range(5)])

cr = [P_var('DU_kw')]
sr = [P('', cell_left)]
for c, s, sig, n, r2 in iv_models_data:
    cr.append(P_coef(c, sig, 4))
    sr.append(P_se(s, 4))
t4b_data.append(cr)
t4b_data.append(sr)

for label in ['Controls', 'Firm/Year']:
    t4b_data.append([P(label, cell_left)] + [P('YES', cell_style)]*5)

kp_vals = [None, iv_kp_f[2], iv_kp_f[3], None, iv_kp_f[5]]
t4b_data.append([P('KP F', cell_left)] + [P(f"{v:.1f}" if v else '—', cell_style) for v in kp_vals])

dwh_vals = [None, iv_dwh_p[2], iv_dwh_p[3], None, iv_dwh_p[5]]
t4b_data.append([P('DWH p', cell_left)] + [P(f"{v:.3f}" if v else '—', cell_style) for v in dwh_vals])

t4b_data.append([P_var('N')] + [P_int(n) for c,s,sig,n,r2 in iv_models_data])
t4b_data.append([P_var('R<super>2</super>')] + [P_num(r2) if r2 is not None else P('—', cell_style) for c,s,sig,n,r2 in iv_models_data])

table4b = Table(t4b_data, colWidths=[2.2*cm] + [2.8*cm]*5)
table4b.setStyle(TableStyle([
    ('FONT', (0,0), (-1,-1), CN_FONT, 8.5),
    ('ALIGN', (0,0), (-1,-1), 'CENTER'), ('ALIGN', (0,0), (0,-1), 'LEFT'),
    ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ('TOPPADDING', (0,0), (-1,-1), 1.5), ('BOTTOMPADDING', (0,0), (-1,-1), 1.5),
    ('LINEABOVE', (0,0), (-1,0), 0.5, colors.black),
    ('LINEBELOW', (0,0), (-1,0), 0.5, colors.black),
    ('LINEBELOW', (0,2), (-1,2), 0.5, colors.black),
    ('LINEBELOW', (0,-1), (-1,-1), 1.2, colors.black),
]))

story.append(Paragraph("表4　　　　　　　　　　　　内生性检验", title_style))
story.append(table4a)
story.append(Spacer(1, 2))
story.append(table4b)
story.append(Spacer(1, 4))
story.append(Paragraph(
    f"注：Panel A报告工具变量第一阶段回归结果，Panel B报告第二阶段估计（均由Stata ivreghdfe统一估计，一二阶段样本完全一致）。"
    f"模型(2)以同行业均值为工具变量；模型(3)为Bartik移位份额工具变量；模型(4)-(5)使用滞后一期DU_kw。"
    f"KP F为Kleibergen-Paap rk Wald F统计量，DWH p为Durbin-Wu-Hausman内生性检验p值。"
    f"Oster (2019)系数稳定性检验：R<super>2</super><sub>max</sub>=1.3R<super>2</super>时"
    f"delta*={oster['delta_13']:.1f}，"
    f"R<super>2</super><sub>max</sub>=min(1,1.3R<super>2</super>)时delta*={oster['delta_con']:.1f}，"
    f"均远大于临界值1。", note_style))
story.append(PageBreak())

# ============================================================
# 表5: 机制检验 (v9.4: 三渠道 Panel A/B/C)
# ============================================================
print("  Table 5...")

mech_dv_labels = ['Analyst', 'Disp', 'absDA']
mech_dv_cn = {'Analyst': '分析师覆盖', 'Disp': '预测分歧度', 'absDA': '盈余管理'}

# --- Panel A: X -> M (第一阶段) ---
t5a_data = []
t5a_data.append([P('<b>Panel A: DU<sub>kw</sub> \u2192 M</b>', cell_left)] + [P('', cell_style)]*2)
t5a_data.append([P('', cell_left)] + [P(f'({i+1})', header_style) for i in range(3)])
t5a_data.append([P('被解释变量=', cell_left)] + [P(f'<i>{dv}</i>', ParagraphStyle(f'dv_{dv}', fontName='Times-Italic',
                fontSize=8.5, leading=11, alignment=TA_CENTER)) for dv in mech_dv_labels])

cr = [P_var('DU_kw')]
sr = [P('', cell_left)]
for dv in mech_dv_labels:
    r = mech_results[dv]
    cr.append(P_coef(r['coef'], r['sig'], 4))
    sr.append(P_se(r['se'], 4))
t5a_data.append(cr)
t5a_data.append(sr)

for label in ['Controls', 'Firm/Year FE']:
    t5a_data.append([P(label, cell_left)] + [P('YES', cell_style)]*3)
t5a_data.append([P_var('N')] + [P_int(mech_results[dv]['N']) for dv in mech_dv_labels])
t5a_data.append([P_var('R<super>2</super>')] + [P_num(mech_results[dv]['R2']) for dv in mech_dv_labels])

table5a = Table(t5a_data, colWidths=[3*cm] + [4*cm]*3)
table5a.setStyle(TableStyle([
    ('FONT', (0,0), (-1,-1), CN_FONT, 8.5),
    ('ALIGN', (0,0), (-1,-1), 'CENTER'), ('ALIGN', (0,0), (0,-1), 'LEFT'),
    ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ('TOPPADDING', (0,0), (-1,-1), 2), ('BOTTOMPADDING', (0,0), (-1,-1), 2),
    ('LINEABOVE', (0,0), (-1,0), 1.2, colors.black),
    ('LINEBELOW', (0,0), (-1,0), 0.5, colors.black),
    ('LINEBELOW', (0,2), (-1,2), 0.5, colors.black),
    ('LINEBELOW', (0,-1), (-1,-1), 0.5, colors.black),
]))

# --- Panel B: X + M -> Y (第二阶段) ---
t5b_data = []
t5b_data.append([P('<b>Panel B: DU<sub>kw</sub> + M \u2192 PriceDelay</b>', cell_left)] + [P('', cell_style)]*2)
t5b_data.append([P('', cell_left)] + [P(f'({i+4})', header_style) for i in range(3)])
t5b_data.append([P('被解释变量=', cell_left)] + [P(f'<i>PriceDelay</i>', ParagraphStyle(f'pd_{i}', fontName='Times-Italic',
                fontSize=8.5, leading=11, alignment=TA_CENTER)) for i in range(3)])

# DU_kw row
cr_x = [P_var('DU_kw')]
sr_x = [P('', cell_left)]
for dv in mech_dv_labels:
    b = mech_b_path[dv]
    cr_x.append(P_coef(b['DU_kw_coef'], b['DU_kw_sig'], 4))
    sr_x.append(P_se(b['DU_kw_se'], 4))
t5b_data.append(cr_x)
t5b_data.append(sr_x)

# M row
cr_m = [P('', cell_left)]
sr_m = [P('', cell_left)]
for dv in mech_dv_labels:
    b = mech_b_path[dv]
    cr_m.append(P_coef(b['coef'], b['sig'], 4))
    sr_m.append(P_se(b['se'], 4))
# M variable names
m_var_row = [P('', cell_left)] + [P_var(dv) for dv in mech_dv_labels]
t5b_data.append(m_var_row)
t5b_data.append(cr_m)
t5b_data.append(sr_m)

for label in ['Controls', 'Firm/Year FE']:
    t5b_data.append([P(label, cell_left)] + [P('YES', cell_style)]*3)
t5b_data.append([P_var('N')] + [P_int(mech_b_path[dv]['N']) for dv in mech_dv_labels])
t5b_data.append([P_var('R<super>2</super>')] + [P_num(mech_b_path[dv]['R2']) for dv in mech_dv_labels])

table5b = Table(t5b_data, colWidths=[3*cm] + [4*cm]*3)
table5b.setStyle(TableStyle([
    ('FONT', (0,0), (-1,-1), CN_FONT, 8.5),
    ('ALIGN', (0,0), (-1,-1), 'CENTER'), ('ALIGN', (0,0), (0,-1), 'LEFT'),
    ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ('TOPPADDING', (0,0), (-1,-1), 2), ('BOTTOMPADDING', (0,0), (-1,-1), 2),
    ('LINEABOVE', (0,0), (-1,0), 0.5, colors.black),
    ('LINEBELOW', (0,0), (-1,0), 0.5, colors.black),
    ('LINEBELOW', (0,2), (-1,2), 0.5, colors.black),
    ('LINEBELOW', (0,-1), (-1,-1), 0.5, colors.black),
]))

# --- Panel C: Sobel test ---
t5c_data = []
t5c_data.append([P('<b>Panel C: Sobel Test</b>', cell_left)] + [P('', cell_style)]*2)
t5c_data.append([P('', cell_left)] + [P_hdr(mech_dv_cn[dv]) for dv in mech_dv_labels])

sobel_rows = [
    ('Indirect', [f"{sobel_results[dv]['indirect']:.6f}" for dv in mech_dv_labels]),
    ('Z', [f"{sobel_results[dv]['Z']:.3f}" for dv in mech_dv_labels]),
    ('p', [f"{sobel_results[dv]['p']:.3f}" for dv in mech_dv_labels]),
    ('Proportion', [f"{sobel_results[dv]['pct']:.1f}%" for dv in mech_dv_labels]),
]
for label, vals in sobel_rows:
    t5c_data.append([P_var(label)] + [P(v, ParagraphStyle(f's_{label}', fontName='Times-Roman',
                    fontSize=8.5, leading=11, alignment=TA_CENTER)) for v in vals])

table5c = Table(t5c_data, colWidths=[3*cm] + [4*cm]*3)
table5c.setStyle(TableStyle([
    ('FONT', (0,0), (-1,-1), CN_FONT, 8.5),
    ('ALIGN', (0,0), (-1,-1), 'CENTER'), ('ALIGN', (0,0), (0,-1), 'LEFT'),
    ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ('TOPPADDING', (0,0), (-1,-1), 2), ('BOTTOMPADDING', (0,0), (-1,-1), 2),
    ('LINEABOVE', (0,0), (-1,0), 0.5, colors.black),
    ('LINEBELOW', (0,0), (-1,0), 0.5, colors.black),
    ('LINEBELOW', (0,1), (-1,1), 0.5, colors.black),
    ('LINEBELOW', (0,-1), (-1,-1), 1.2, colors.black),
]))

story.append(Paragraph("表5　　　　　　　　　　　　机制检验", title_style))
story.append(table5a)
story.append(Spacer(1, 1))
story.append(table5b)
story.append(Spacer(1, 1))
story.append(table5c)
story.append(Spacer(1, 4))
story.append(Paragraph(
    "注：Panel A报告DU_kw对中介变量的回归结果，Panel B报告同时纳入DU_kw和中介变量对PriceDelay的回归，"
    "Panel C报告Sobel中介效应检验。固定效应为企业+年份，标准误按行业×年份聚类。"
    "Analyst为分析师覆盖人数对数，Disp为分析师预测分歧度(每股收益预测标准差)，"
    "absDA为Kothari et al.(2005)业绩调整Jones模型可操纵应计绝对值。"
    "Proportion为中介效应占直接效应的比例。", note_style))
story.append(PageBreak())

# ============================================================
# 表6: 异质性分析 — 双Panel结构
# ============================================================
print("  Table 6...")

# --- Panel A: 分组回归 + Fisher permutation p值 (v9.3: 3维度6列) ---
het_groups = ["国有企业","非国有企业","高机构持股","低机构持股",
              "高分析师覆盖","低分析师覆盖"]
het_labels = [
    "(1) 国有", "(2) 非国有",
    "(3) 高机构<br/>持股", "(4) 低机构<br/>持股",
    "(5) 高分析师<br/>覆盖", "(6) 低分析师<br/>覆盖"
]

t6a_data = []
# Panel A header
t6a_data.append([P('<b>Panel A: 分组回归</b>', cell_left)] + [P('', cell_style)]*6)
t6a_data.append([P('', cell_left)] + [P_hdr(l) for l in het_labels])
t6a_data.append([P('被解释变量=', cell_left)] + [P('<i>PriceDelay</i>', ParagraphStyle('dv6a', fontName='Times-Italic',
                fontSize=7.5, leading=9, alignment=TA_CENTER)) for _ in range(6)])

cr = [P_var('DU_kw')]
sr = [P('', cell_left)]
for grp in het_groups:
    r = het_results[grp]
    cr.append(P_coef(r['coef'], r['sig'], superscript=False))
    sr.append(P_se(r['se']))
t6a_data.append(cr)
t6a_data.append(sr)

for label in ['Controls', 'Firm/Year FE']:
    t6a_data.append([P(label, cell_left)] + [P('YES', cell_style)]*6)

t6a_data.append([P_var('N')] + [P_int(het_results[g]['N']) for g in het_groups])
t6a_data.append([P_var('R<super>2</super>')] + [P_num(het_results[g]['R2']) for g in het_groups])

# v9.3: Fisher permutation test p值行 (Stata验证, 1000次置换)
fisher_row = [P('组间差异P值', cell_left)]
fisher_dims = ['产权性质', '机构持股', '分析师覆盖']
for dim in fisher_dims:
    p_val = fisher_p.get(dim, np.nan)
    p_text = f"{p_val:.3f}" if not np.isnan(p_val) else "—"
    p_sig = sig_stars(p_val) if not np.isnan(p_val) else ""
    if p_sig:
        p_text += p_sig
    fisher_row.append(Paragraph(p_text, ParagraphStyle('fp', fontName='Times-Roman',
                       fontSize=8.5, leading=11, alignment=TA_CENTER)))
    fisher_row.append(P('', cell_style))  # span placeholder
t6a_data.append(fisher_row)

# --- Panel B: 连续变量调节效应 (4列: 变量名 + TobinQ + 空隔 + Amihud) ---
r_tq = cont_interact_results['TobinQ']
r_am = cont_interact_results['Amihud']

t6b_data = []
t6b_data.append([P('<b>Panel B: 调节效应</b>', cell_left),
                 P('', cell_style), P('', cell_style), P('', cell_style)])
t6b_data.append([P('', cell_left),
                 P_hdr('(1)'), P('', cell_style), P_hdr('(2)')])
t6b_data.append([P('被解释变量=', cell_left),
                 P('<i>PriceDelay</i>', ParagraphStyle('dv6b1', fontName='Times-Italic',
                   fontSize=7.5, leading=9, alignment=TA_CENTER)),
                 P('', cell_style),
                 P('<i>PriceDelay</i>', ParagraphStyle('dv6b2', fontName='Times-Italic',
                   fontSize=7.5, leading=9, alignment=TA_CENTER))])

# DU_kw main effect
t6b_data.append([P_var('DU_kw'),
                 P_coef(r_tq['coef_main'], r_tq['sig_main']),
                 P('', cell_style),
                 P_coef(r_am['coef_main'], r_am['sig_main'])])
t6b_data.append([P('', cell_left),
                 P_se(r_tq['se_main']),
                 P('', cell_style),
                 P_se(r_am['se_main'])])

# DU_kw x TobinQ (only in col 1)
t6b_data.append([P_var('DU_kw \u00d7 TobinQ'),
                 P_coef(r_tq['coef_inter'], r_tq['sig_inter'], dec=4),
                 P('', cell_style),
                 P('', cell_style)])
t6b_data.append([P('', cell_left),
                 P_se(r_tq['se_inter'], dec=4),
                 P('', cell_style),
                 P('', cell_style)])

# DU_kw x Amihud (only in col 2)
t6b_data.append([P_var('DU_kw \u00d7 Amihud'),
                 P('', cell_style),
                 P('', cell_style),
                 P_coef(r_am['coef_inter'], r_am['sig_inter'], dec=4)])
t6b_data.append([P('', cell_left),
                 P('', cell_style),
                 P('', cell_style),
                 P_se(r_am['se_inter'], dec=4)])

for label in ['Controls', 'Firm/Year FE']:
    t6b_data.append([P(label, cell_left), P('YES', cell_style), P('', cell_style), P('YES', cell_style)])

t6b_data.append([P_var('N'), P_int(r_tq['N']), P('', cell_style), P_int(r_am['N'])])
t6b_data.append([P_var('R<super>2</super>'), P_num(r_tq['R2']), P('', cell_style), P_num(r_am['R2'])])

# Build Panel A table (v9.3: 6 cols instead of 8)
table6a = Table(t6a_data, colWidths=[2.2*cm] + [2.4*cm]*6)
table6a.setStyle(TableStyle([
    ('FONT', (0,0), (-1,-1), CN_FONT, 8),
    ('ALIGN', (0,0), (-1,-1), 'CENTER'), ('ALIGN', (0,0), (0,-1), 'LEFT'),
    ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ('TOPPADDING', (0,0), (-1,-1), 2), ('BOTTOMPADDING', (0,0), (-1,-1), 2),
    ('LINEABOVE', (0,0), (-1,0), 1.2, colors.black),
    ('LINEBELOW', (0,0), (-1,0), 0.5, colors.black),
    ('LINEBELOW', (0,2), (-1,2), 0.5, colors.black),
    ('LINEBELOW', (0,-1), (-1,-1), 0.5, colors.black),
    # Span Fisher p-value cells (each dim spans 2 cols)
    ('SPAN', (1,-1), (2,-1)),
    ('SPAN', (3,-1), (4,-1)),
    ('SPAN', (5,-1), (6,-1)),
]))

# Build Panel B table (4 cols: varname 2.2cm, TobinQ 5.6cm, gap 1.2cm, Amihud 5.6cm)
table6b = Table(t6b_data, colWidths=[2.2*cm, 5.6*cm, 1.2*cm, 5.6*cm])
table6b.setStyle(TableStyle([
    ('FONT', (0,0), (-1,-1), CN_FONT, 8),
    ('ALIGN', (0,0), (-1,-1), 'CENTER'), ('ALIGN', (0,0), (0,-1), 'LEFT'),
    ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ('TOPPADDING', (0,0), (-1,-1), 2), ('BOTTOMPADDING', (0,0), (-1,-1), 2),
    ('LINEABOVE', (0,0), (-1,0), 0.5, colors.black),
    ('LINEBELOW', (0,0), (-1,0), 0.5, colors.black),
    ('LINEBELOW', (0,2), (-1,2), 0.5, colors.black),
    ('LINEBELOW', (0,-1), (-1,-1), 1.2, colors.black),
]))

story.append(Paragraph("表6　　　　　　　　　　　　异质性分析", title_style))
story.append(table6a)
story.append(Spacer(1, 2))
story.append(table6b)
story.append(Spacer(1, 4))
story.append(Paragraph(
    "注：Panel A报告分组回归结果，组间差异P值基于费舍尔组合检验(Stata置换1000次)。"
    "高/低机构持股和高/低分析师覆盖均按全样本中位数分组。Panel B报告连续变量调节效应，"
    "TobinQ越高(成长性越好)或Amihud越大(流动性越差)时，数据要素的定价效率改善效果越弱。", note_style))
story.append(PageBreak())

# ============================================================
# 表7: 投资组合Alpha
# ============================================================
print("  Table 7...")

port_df = pd.read_csv(f"{BASE}/results/portfolio/portfolio_alpha.csv")

t7_data = []
t7_data.append([P('Portfolio', cell_left),
                P('CAPM', header_style), P('', header_style),
                P('FF3', header_style), P('', header_style),
                P('FF5+MOM', header_style), P('', header_style)])
t7_data.append([P('', cell_left),
                P('alpha', header_style), P('t', header_style),
                P('alpha', header_style), P('t', header_style),
                P('alpha', header_style), P('t', header_style)])

for port in ['Q1','Q2','Q3','Q4','Q5','DAT']:
    row = [P_var(port)]
    for model in ['CAPM','FF3','FF5+MOM']:
        sub = port_df[(port_df['Portfolio']==port) & (port_df['Model']==model)]
        if len(sub) > 0:
            a = sub.iloc[0]['alpha']
            t = sub.iloc[0]['t']
            p = sub.iloc[0]['p']
            sig = "***" if p<0.01 else "**" if p<0.05 else "*" if p<0.1 else ""
            row.append(P_coef(a*100, sig, 2))
            row.append(P(f"{t:.2f}", ParagraphStyle('t', fontName='Times-Roman',
                        fontSize=8.5, leading=11, alignment=TA_CENTER)))
        else:
            row.append(P('', cell_style))
            row.append(P('', cell_style))
    t7_data.append(row)

table7 = Table(t7_data, colWidths=[2*cm] + [2.2*cm]*6)
table7.setStyle(TableStyle([
    ('FONT', (0,0), (-1,-1), CN_FONT, 8.5),
    ('ALIGN', (0,0), (-1,-1), 'CENTER'), ('ALIGN', (0,0), (0,-1), 'LEFT'),
    ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ('TOPPADDING', (0,0), (-1,-1), 2), ('BOTTOMPADDING', (0,0), (-1,-1), 2),
    ('LINEABOVE', (0,0), (-1,0), 1.2, colors.black),
    ('LINEBELOW', (0,0), (-1,0), 0.5, colors.black),
    ('LINEBELOW', (0,1), (-1,1), 0.5, colors.black),
    ('LINEBELOW', (0,-1), (-1,-1), 1.2, colors.black),
    ('SPAN', (1,0), (2,0)), ('SPAN', (3,0), (4,0)), ('SPAN', (5,0), (6,0)),
]))

story.append(Paragraph("附表1　　　　　　　　　投资组合Alpha检验", title_style))
story.append(table7)
story.append(Spacer(1, 4))
story.append(Paragraph(
    "注：alpha单位为月度百分比。Q1至Q5为按DU_kw年度五分位构造的投资组合，"
    "DAT=Q5-Q1为数据资产因子。样本期间2011年7月至2024年6月，共156个月。", note_style))

# ============================================================
# Build PDF
# ============================================================
print("Building PDF...")
doc.build(story)
print(f"\nDone! PDF saved to: {pdf_path}")
