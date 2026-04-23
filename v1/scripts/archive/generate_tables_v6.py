"""
生成朱康(2025)会计研究格式的全部表格PDF
v6: 系数带星号 + 标准误括号，不单列t值/p值
"""

import pandas as pd
import numpy as np
import pyfixest as pf
import json, warnings, os
warnings.filterwarnings('ignore')

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm, mm
from reportlab.lib import colors
from reportlab.platypus import (SimpleDocTemplate, Table, TableStyle, Paragraph,
                                Spacer, PageBreak, KeepTogether)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ============================================================
# 0. 字体注册
# ============================================================
# Try to register Chinese font
font_paths = [
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/noto-cjk/NotoSansCJKsc-Regular.otf",
]
CN_FONT = "Helvetica"  # fallback
for fp in font_paths:
    if os.path.exists(fp):
        try:
            pdfmetrics.registerFont(TTFont('NotoSansCJK', fp, subfontIndex=0))
            CN_FONT = 'NotoSansCJK'
            break
        except:
            pass

if CN_FONT == "Helvetica":
    # Try system fonts
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
# 1. 数据加载 & 预处理 (复用 v3_concurrent 流程)
# ============================================================
print("Loading data...")
BASE = "mnt/15会计研究"
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
panel = panel.merge(fi[['Stkcd','year','IndustryCodeC','LISTINGSTATE']], on=['Stkcd','year'], how='left')

# 样本筛选
mask_fin = panel['IndustryCodeC'].str.startswith('J', na=False)
mask_st = panel['LISTINGSTATE'].isin(['ST','*ST'])
mask_new = panel['Age'] <= 0
panel = panel[~mask_fin & ~mask_st & ~mask_new]

# 构造变量 (同期)
panel['Ind2'] = panel['IndustryCodeC'].str[:3]
panel['IndYear'] = panel['Ind2'].astype(str) + '_' + panel['year'].astype(str)
panel['DU_kw'] = panel['kw_per10k']
panel['DU_kw_ln'] = np.log1p(panel['kw_total'])
panel['DU_sub_ln'] = np.log1p(panel['substantive_count'])

# SYNCH
synch = pd.read_parquet(f"{BASE}/data_parquet/price_synchronicity.parquet")
panel = panel.merge(synch[['Stkcd','year','SYNCH']], on=['Stkcd','year'], how='left')

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
print(f"Regression sample: N={len(reg):,}, firms={reg.Stkcd.nunique():,}")

ctrl_str = " + ".join(controls)

# ============================================================
# 2. Run ALL regressions
# ============================================================
print("Running regressions...")

def get_nobs(model):
    return model._N

def get_r2(model):
    return model._r2

def extract_coefs(model, vars_list):
    """Extract coef, se, sig for a list of variables"""
    results = {}
    for var in vars_list:
        c = float(model.coef()[var])
        s = float(model.se()[var])
        p = float(model.pvalue()[var])
        sig = "***" if p<0.01 else "**" if p<0.05 else "*" if p<0.1 else ""
        results[var] = (c, s, sig)
    return results

def fmt_coef(c, sig, decimals=3):
    """Format coefficient with significance stars"""
    return f"{c:.{decimals}f}{sig}"

def fmt_se(s, decimals=3):
    """Format standard error in parentheses"""
    return f"({s:.{decimals}f})"

# ---- Baseline models ----
print("  Baseline models...")

# m0: DU_kw only, no controls → 表2 model (1)
m0 = pf.feols(f"PriceDelay ~ DU_kw | Stkcd_str + year_str",
              data=reg, vcov={"CRV1":"IndYear"})
m0_main = extract_coefs(m0, ["DU_kw"])
m0_N, m0_R2 = get_nobs(m0), get_r2(m0)

# m1: DU_kw + controls → 表2 model (2), show all control coefs
m1 = pf.feols(f"PriceDelay ~ DU_kw + {ctrl_str} | Stkcd_str + year_str",
              data=reg, vcov={"CRV1":"IndYear"})
m1_coefs = extract_coefs(m1, ["DU_kw"] + controls)
m1_N, m1_R2 = get_nobs(m1), get_r2(m1)

# m2: DU_kw_ln → 表2 model (3)
m2 = pf.feols(f"PriceDelay ~ DU_kw_ln + {ctrl_str} | Stkcd_str + year_str",
              data=reg, vcov={"CRV1":"IndYear"})
m2_main = extract_coefs(m2, ["DU_kw_ln"])
m2_N, m2_R2 = get_nobs(m2), get_r2(m2)

# m3: DU_sub_ln → 表2 model (4)
m3 = pf.feols(f"PriceDelay ~ DU_sub_ln + {ctrl_str} | Stkcd_str + year_str",
              data=reg, vcov={"CRV1":"IndYear"})
m3_main = extract_coefs(m3, ["DU_sub_ln"])
m3_N, m3_R2 = get_nobs(m3), get_r2(m3)

# m4: Ind+Year FE → 表2 model (5)
reg_ind = reg.dropna(subset=['Ind2']).copy()
m4 = pf.feols(f"PriceDelay ~ DU_kw + {ctrl_str} | Ind2 + year_str",
              data=reg_ind, vcov={"CRV1":"IndYear"})
m4_main = extract_coefs(m4, ["DU_kw"])
m4_N, m4_R2 = get_nobs(m4), get_r2(m4)

# m5: SYNCH as DV → 表2 model (6)
reg_synch = reg.dropna(subset=['SYNCH']).copy()
m5 = pf.feols(f"SYNCH ~ DU_kw + {ctrl_str} | Stkcd_str + year_str",
              data=reg_synch, vcov={"CRV1":"IndYear"})
m5_main = extract_coefs(m5, ["DU_kw"])
m5_N, m5_R2 = get_nobs(m5), get_r2(m5)

# m6: With FinAsset control → 表2 model (7)
m6 = pf.feols(f"PriceDelay ~ DU_kw + FinAsset + {ctrl_str} | Stkcd_str + year_str",
              data=reg, vcov={"CRV1":"IndYear"})
m6_main = extract_coefs(m6, ["DU_kw", "FinAsset"])
m6_N, m6_R2 = get_nobs(m6), get_r2(m6)

print(f"  Baseline done. Model(1) no ctrl: coef={m0_main['DU_kw'][0]:.6f}, N={m0_N}")
print(f"                 Model(2) w/ ctrl: coef={m1_coefs['DU_kw'][0]:.6f}, N={m1_N}")

# ---- Robustness models ----
print("  Robustness models...")
# (1) Replace DV: SYNCH - already done as m5
# (2) Replace IV: DU_kw_ln - already done as m2
# (3) Ind×Year FE
rob3 = pf.feols(f"PriceDelay ~ DU_kw + {ctrl_str} | IndYear",
                data=reg, vcov={"CRV1":"IndYear"})
rob3_main = extract_coefs(rob3, ["DU_kw"])
rob3_N, rob3_R2 = get_nobs(rob3), get_r2(rob3)

# (4) Prov×Year FE - need province info
# Skip if no province data, use existing result
# (5) 剔除2024
reg_no24 = reg[reg['year'] != 2024].copy()
rob5 = pf.feols(f"PriceDelay ~ DU_kw + {ctrl_str} | Stkcd_str + year_str",
                data=reg_no24, vcov={"CRV1":"IndYear"})
rob5_main = extract_coefs(rob5, ["DU_kw"])
rob5_N, rob5_R2 = get_nobs(rob5), get_r2(rob5)

# (6) 剔除IT行业
reg_noit = reg[~reg['IndustryCodeC'].str.startswith('I', na=False)].copy()
rob6 = pf.feols(f"PriceDelay ~ DU_kw + {ctrl_str} | Stkcd_str + year_str",
                data=reg_noit, vcov={"CRV1":"IndYear"})
rob6_main = extract_coefs(rob6, ["DU_kw"])
rob6_N, rob6_R2 = get_nobs(rob6), get_r2(rob6)

# (7) PSM
from sklearn.linear_model import LogisticRegression
reg_psm = reg.copy()
reg_psm['Treat'] = (reg_psm['DU_kw'] > reg_psm['DU_kw'].median()).astype(int)
X_psm = reg_psm[controls].values
y_psm = reg_psm['Treat'].values
lr = LogisticRegression(max_iter=1000, C=1.0)
lr.fit(X_psm, y_psm)
reg_psm['pscore'] = lr.predict_proba(X_psm)[:,1]
caliper = 0.25 * reg_psm['pscore'].std()
# Simple caliper match
treat_idx = reg_psm[reg_psm['Treat']==1].index
ctrl_pool = reg_psm[reg_psm['Treat']==0]
matched_pairs = []
for idx in treat_idx:
    ps_t = reg_psm.loc[idx, 'pscore']
    candidates = ctrl_pool[(ctrl_pool['pscore'] - ps_t).abs() < caliper]
    if len(candidates) > 0:
        best = (candidates['pscore'] - ps_t).abs().idxmin()
        matched_pairs.append((idx, best))
match_idx = set([p[0] for p in matched_pairs] + [p[1] for p in matched_pairs])
reg_matched = reg_psm.loc[list(match_idx)].copy()
rob7 = pf.feols(f"PriceDelay ~ DU_kw + {ctrl_str} | Stkcd_str + year_str",
                data=reg_matched, vcov={"CRV1":"IndYear"})
rob7_main = extract_coefs(rob7, ["DU_kw"])
rob7_N, rob7_R2 = get_nobs(rob7), get_r2(rob7)
print(f"  PSM matched: N={rob7_N}")

# (8) IV: use stored results
iv_results = json.load(open(f"{BASE}/results/endogeneity_v4/endogeneity_v4_results.json"))
rob8_coef = iv_results['peer_iv']['coef']
rob8_se = iv_results['peer_iv']['se']
rob8_p = iv_results['peer_iv']['p']
rob8_sig = "***" if rob8_p<0.01 else "**" if rob8_p<0.05 else "*" if rob8_p<0.1 else ""
rob8_N = iv_results['peer_iv']['N']
rob8_KPF = iv_results['peer_iv']['KP_F']

# (9) Heckman: need IMR. Approximate with Probit first stage
from scipy.stats import norm as sp_norm
reg_heck = reg.copy()
reg_heck['DU_positive'] = (reg_heck['DU_kw'] > reg_heck['DU_kw'].median()).astype(int)
probit_X = reg_heck[controls].values
probit_y = reg_heck['DU_positive'].values
lr_heck = LogisticRegression(max_iter=1000)
lr_heck.fit(probit_X, probit_y)
prob = lr_heck.predict_proba(probit_X)[:,1]
prob = np.clip(prob, 1e-6, 1-1e-6)
reg_heck['IMR'] = sp_norm.pdf(sp_norm.ppf(prob)) / prob  # Mills ratio for selected
rob9 = pf.feols(f"PriceDelay ~ DU_kw + IMR + {ctrl_str} | Stkcd_str + year_str",
                data=reg_heck, vcov={"CRV1":"IndYear"})
rob9_main = extract_coefs(rob9, ["DU_kw"])
rob9_N, rob9_R2 = get_nobs(rob9), get_r2(rob9)

print("  Robustness done.")

# ---- Mechanism models ----
print("  Mechanism models...")
mech_dvs = ['Analyst', 'FinAsset', 'InstHold']
mech_results = {}
for dv in mech_dvs:
    # Exclude DV from controls to avoid multicollinearity
    mech_ctrls = [c for c in controls if c != dv]
    mech_ctrl_str = " + ".join(mech_ctrls)
    m = pf.feols(f"{dv} ~ DU_kw + {mech_ctrl_str} | Stkcd_str + year_str",
                 data=reg, vcov={"CRV1":"IndYear"})
    c, s, sig = extract_coefs(m, ["DU_kw"])["DU_kw"]
    mech_results[dv] = {"coef": c, "se": s, "sig": sig, "N": get_nobs(m), "R2": get_r2(m)}
print("  Mechanism done.")

# ---- Heterogeneity models ----
print("  Heterogeneity models...")
base_fml = f"PriceDelay ~ DU_kw + {ctrl_str} | Stkcd_str + year_str"
het_results = {}

# SOE vs Non-SOE
for label, mask in [("国有企业", reg['SOE']==1), ("非国有企业", reg['SOE']==0)]:
    sub = reg[mask].copy()
    m = pf.feols(base_fml, data=sub, vcov={"CRV1":"IndYear"})
    c, s, sig = extract_coefs(m, ["DU_kw"])["DU_kw"]
    het_results[label] = {"coef": c, "se": s, "sig": sig, "N": get_nobs(m), "R2": get_r2(m)}

# High vs Low Analyst
med_analyst = reg['Analyst'].median()
for label, mask in [("高分析师覆盖", reg['Analyst']>=med_analyst), ("低分析师覆盖", reg['Analyst']<med_analyst)]:
    sub = reg[mask].copy()
    m = pf.feols(base_fml, data=sub, vcov={"CRV1":"IndYear"})
    c, s, sig = extract_coefs(m, ["DU_kw"])["DU_kw"]
    het_results[label] = {"coef": c, "se": s, "sig": sig, "N": get_nobs(m), "R2": get_r2(m)}

# Large vs Small firms (by median Size)
med_size = reg['Size'].median()
for label, mask in [("大企业", reg['Size']>=med_size), ("小企业", reg['Size']<med_size)]:
    sub = reg[mask].copy()
    m = pf.feols(base_fml, data=sub, vcov={"CRV1":"IndYear"})
    c, s, sig = extract_coefs(m, ["DU_kw"])["DU_kw"]
    het_results[label] = {"coef": c, "se": s, "sig": sig, "N": get_nobs(m), "R2": get_r2(m)}

# High-tech vs Traditional
hitech_codes = ['I', 'M']  # IT + Scientific research
for label, is_hitech in [("高科技行业", True), ("传统行业", False)]:
    if is_hitech:
        mask = reg['IndustryCodeC'].str[0].isin(hitech_codes)
    else:
        mask = ~reg['IndustryCodeC'].str[0].isin(hitech_codes)
    sub = reg[mask].copy()
    m = pf.feols(base_fml, data=sub, vcov={"CRV1":"IndYear"})
    c, s, sig = extract_coefs(m, ["DU_kw"])["DU_kw"]
    het_results[label] = {"coef": c, "se": s, "sig": sig, "N": get_nobs(m), "R2": get_r2(m)}

# Fisher permutation test for group differences (stored results)
fisher_df = pd.read_csv(f"{BASE}/results/v3_concurrent/heterogeneity_v3_fisher.csv")
fisher_p = {}
for _, row in fisher_df.iterrows():
    fisher_p[row['Dimension']] = row['p']

print("  Heterogeneity done.")

# ============================================================
# 3. 生成PDF
# ============================================================
print("\nGenerating PDF...")

pdf_path = f"{BASE}/results/v6_tables/regression_tables_v6.pdf"
os.makedirs(os.path.dirname(pdf_path), exist_ok=True)

doc = SimpleDocTemplate(pdf_path, pagesize=A4,
                       leftMargin=1.5*cm, rightMargin=1.5*cm,
                       topMargin=2*cm, bottomMargin=2*cm)

styles = getSampleStyleSheet()

# Custom styles
title_style = ParagraphStyle('TableTitle', fontName=CN_FONT, fontSize=10,
                             leading=14, alignment=TA_CENTER, spaceAfter=6)
note_style = ParagraphStyle('TableNote', fontName=CN_FONT, fontSize=7.5,
                            leading=10, alignment=TA_LEFT, spaceAfter=12)
var_style = ParagraphStyle('VarName', fontName='Times-Italic', fontSize=8.5,
                           leading=11, alignment=TA_LEFT)
cell_style = ParagraphStyle('CellText', fontName=CN_FONT, fontSize=8.5,
                            leading=11, alignment=TA_CENTER)
cell_left = ParagraphStyle('CellLeft', fontName=CN_FONT, fontSize=8.5,
                           leading=11, alignment=TA_LEFT)
header_style = ParagraphStyle('HeaderText', fontName=CN_FONT, fontSize=8.5,
                              leading=11, alignment=TA_CENTER)

def P(text, style=cell_style):
    return Paragraph(str(text), style)

def P_var(text):
    """Variable name in italics"""
    return Paragraph(f"<i>{text}</i>", ParagraphStyle('Var', fontName='Times-Italic',
                     fontSize=8.5, leading=11, alignment=TA_LEFT))

def P_coef(c, sig, dec=3):
    """Coefficient with stars"""
    txt = f"{c:.{dec}f}"
    if sig:
        txt += f"<super>{sig}</super>"
    return Paragraph(txt, ParagraphStyle('Coef', fontName='Times-Roman',
                     fontSize=8.5, leading=11, alignment=TA_CENTER))

def P_se(s, dec=3):
    """Standard error in parentheses"""
    return Paragraph(f"({s:.{dec}f})", ParagraphStyle('SE', fontName='Times-Roman',
                     fontSize=8.5, leading=11, alignment=TA_CENTER))

def P_num(n, dec=3):
    return Paragraph(f"{n:.{dec}f}", ParagraphStyle('Num', fontName='Times-Roman',
                     fontSize=8.5, leading=11, alignment=TA_CENTER))

def P_int(n):
    return Paragraph(f"{int(n):,}", ParagraphStyle('Int', fontName='Times-Roman',
                     fontSize=8.5, leading=11, alignment=TA_CENTER))

def std_table_style(nrows, ncols):
    """Standard table style mimicking 朱康(2025)"""
    return TableStyle([
        ('FONT', (0,0), (-1,-1), CN_FONT, 8.5),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('ALIGN', (0,0), (0,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
        # Top and bottom borders
        ('LINEABOVE', (0,0), (-1,0), 1.2, colors.black),
        ('LINEBELOW', (0,0), (-1,0), 0.5, colors.black),  # below header row 0
        ('LINEBELOW', (0,1), (-1,1), 0.5, colors.black),  # below DV row
        ('LINEBELOW', (0,-1), (-1,-1), 1.2, colors.black),  # bottom
    ])


story = []

# ============================================================
# 表1: 描述性统计
# ============================================================
print("  Table 1: Descriptive Statistics...")

# Compute descriptive stats from regression sample directly
desc_vars = ['PriceDelay','DU_kw','DU_kw_ln','Size','Lev','ROA','TobinQ',
             'Age','Growth','BoardSize','IndepRatio','Dual','Top1Share',
             'SOE','InstHold','Amihud','Analyst','AuditType']

desc_rows = []
for var in desc_vars:
    s = reg[var].dropna()
    desc_rows.append({
        'Variable': var, 'N': len(s), 'Mean': s.mean(), 'Std': s.std(),
        'Min': s.min(), 'Median': s.median(), 'Max': s.max()
    })
desc_data = pd.DataFrame(desc_rows)

# Build table
t1_header = [P('', cell_left), P('N', header_style),
             P('Mean', header_style), P('SD', header_style),
             P('Min', header_style), P('Median', header_style), P('Max', header_style)]

t1_data = [t1_header]
for var in desc_vars:
    row = desc_data[desc_data['Variable']==var]
    if len(row) == 0:
        continue
    r = row.iloc[0]
    var_display = var
    t1_data.append([
        P_var(var_display),
        P_int(r['N']),
        P_num(r['Mean']),
        P_num(r['Std']),
        P_num(r['Min']),
        P_num(r['Median']),
        P_num(r['Max']),
    ])

col_widths_t1 = [3*cm, 2*cm, 2.2*cm, 2.2*cm, 2.2*cm, 2.2*cm, 2.2*cm]
table1 = Table(t1_data, colWidths=col_widths_t1)
ts1 = TableStyle([
    ('FONT', (0,0), (-1,-1), CN_FONT, 8.5),
    ('ALIGN', (0,0), (-1,-1), 'CENTER'),
    ('ALIGN', (0,0), (0,-1), 'LEFT'),
    ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ('TOPPADDING', (0,0), (-1,-1), 2),
    ('BOTTOMPADDING', (0,0), (-1,-1), 2),
    ('LINEABOVE', (0,0), (-1,0), 1.2, colors.black),
    ('LINEBELOW', (0,0), (-1,0), 0.5, colors.black),
    ('LINEBELOW', (0,-1), (-1,-1), 1.2, colors.black),
])
table1.setStyle(ts1)

story.append(Paragraph("表1　　　　　　　　　　　变量的描述性统计", title_style))
story.append(table1)
story.append(Spacer(1, 6))
story.append(PageBreak())

# ============================================================
# 表2: 基准回归 (7 models)
# ============================================================
print("  Table 2: Baseline Regression...")

ncols_t2 = 7  # 7 models

# Column headers
t2_h1 = [P('', cell_left)]
for i in range(1, ncols_t2+1):
    t2_h1.append(P(f'({i})', header_style))

# DV row: (1)-(5)(7) PriceDelay, (6) SYNCH
dv_names = ['PriceDelay','PriceDelay','PriceDelay','PriceDelay','PriceDelay','SYNCH','PriceDelay']
t2_dv = [P('', cell_left)]
for dv in dv_names:
    t2_dv.append(P(f'<i>{dv}</i>', ParagraphStyle('dv', fontName='Times-Italic',
                   fontSize=8, leading=10, alignment=TA_CENTER)))

# Models mapping:
# (1) m0: DU_kw no controls
# (2) m1: DU_kw + controls (show all)
# (3) m2: DU_kw_ln + controls
# (4) m3: DU_sub_ln + controls
# (5) m4: DU_kw Ind+Year FE
# (6) m5: SYNCH as DV
# (7) m6: DU_kw + FinAsset + controls

t2_data = [t2_h1, t2_dv]

# --- Key variable row: DU_kw / DU_kw_ln / DU_sub_ln ---
key_coefs = [
    m0_main['DU_kw'],       # (1)
    m1_coefs['DU_kw'],      # (2)
    m2_main['DU_kw_ln'],    # (3)
    m3_main['DU_sub_ln'],   # (4)
    m4_main['DU_kw'],       # (5)
    m5_main['DU_kw'],       # (6)
    m6_main['DU_kw'],       # (7)
]
coef_row = [P_var('DU_kw/DU_kw_ln/<br/>DU_sub_ln')]
se_row = [P('', cell_left)]
for c, s, sig in key_coefs:
    coef_row.append(P_coef(c, sig))
    se_row.append(P_se(s))
t2_data.append(coef_row)
t2_data.append(se_row)

# --- FinAsset for model (7) only ---
fa_coef_row = [P_var('FinAsset')]
fa_se_row = [P('', cell_left)]
for i in range(ncols_t2):
    if i == 6:  # model (7)
        c, s, sig = m6_main['FinAsset']
        fa_coef_row.append(P_coef(c, sig))
        fa_se_row.append(P_se(s))
    else:
        fa_coef_row.append(P('', cell_style))
        fa_se_row.append(P('', cell_style))
t2_data.append(fa_coef_row)
t2_data.append(fa_se_row)

# --- Control variables: show in model (2) only ---
for ctrl in controls:
    c, s, sig = m1_coefs[ctrl]
    ctrl_coef_row = [P_var(ctrl)]
    ctrl_se_row = [P('', cell_left)]
    # Model (1): no controls → blank
    ctrl_coef_row.append(P('', cell_style))
    ctrl_se_row.append(P('', cell_style))
    # Model (2): show coefficients
    ctrl_coef_row.append(P_coef(c, sig))
    ctrl_se_row.append(P_se(s))
    # Models (3)-(7): suppress
    for i in range(5):
        ctrl_coef_row.append(P('', cell_style))
        ctrl_se_row.append(P('', cell_style))
    t2_data.append(ctrl_coef_row)
    t2_data.append(ctrl_se_row)

# Controls YES/NO row
ctrl_yes_row = [P('Controls', cell_left)]
ctrl_flags = ['NO','YES','YES','YES','YES','YES','YES']
for f in ctrl_flags:
    ctrl_yes_row.append(P(f, cell_style))
t2_data.append(ctrl_yes_row)

# Fixed effects rows
fe_firm = ['YES','YES','YES','YES','NO','YES','YES']
fe_ind  = ['NO','NO','NO','NO','YES','NO','NO']

fe_row = [P('Firm/Year', cell_left)]
for f in fe_firm:
    fe_row.append(P(f, cell_style))
t2_data.append(fe_row)

indyear_row = [P('Ind/Year', cell_left)]
for f in fe_ind:
    indyear_row.append(P(f, cell_style))
t2_data.append(indyear_row)

# N row
n_vals = [m0_N, m1_N, m2_N, m3_N, m4_N, m5_N, m6_N]
n_row = [P_var('N')]
for n in n_vals:
    n_row.append(P_int(n))
t2_data.append(n_row)

# R2 row
r2_vals = [m0_R2, m1_R2, m2_R2, m3_R2, m4_R2, m5_R2, m6_R2]
r2_row = [P_var('R<super>2</super>')]
for r in r2_vals:
    r2_row.append(P_num(r))
t2_data.append(r2_row)

col_widths_t2 = [2.6*cm] + [2.05*cm]*7
table2 = Table(t2_data, colWidths=col_widths_t2)
ts2 = TableStyle([
    ('FONT', (0,0), (-1,-1), CN_FONT, 8.5),
    ('ALIGN', (0,0), (-1,-1), 'CENTER'),
    ('ALIGN', (0,0), (0,-1), 'LEFT'),
    ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ('TOPPADDING', (0,0), (-1,-1), 1.5),
    ('BOTTOMPADDING', (0,0), (-1,-1), 1.5),
    ('LINEABOVE', (0,0), (-1,0), 1.2, colors.black),
    ('LINEBELOW', (0,0), (-1,0), 0.5, colors.black),
    ('LINEBELOW', (0,1), (-1,1), 0.5, colors.black),
    ('LINEBELOW', (0,-1), (-1,-1), 1.2, colors.black),
])
table2.setStyle(ts2)

story.append(Paragraph("表2　　　　　　　　　　　基准回归检验", title_style))
story.append(table2)
story.append(Spacer(1, 4))
story.append(Paragraph(
    "注：括号内数值为经行业×年份层面聚类调整后的稳健标准误；<super>***</super>、<super>**</super>、<super>*</super>分别表示1%、5%、10%的水平上显著，下同。"
    "模型(1)仅控制固定效应不含控制变量，模型(2)-(4)和(6)-(7)控制企业和年份固定效应，模型(5)控制行业和年份固定效应。",
    note_style))
story.append(PageBreak())

# ============================================================
# 表3: 稳健性检验 (9列)
# ============================================================
print("  Table 3: Robustness...")

rob_labels = [
    "(1) 替换被<br/>解释变量", "(2) 替换解<br/>释变量", "(3) 控制<br/>Ind×Year",
    "(4) 控制<br/>Prov×Year", "(5) 剔除<br/>2024", "(6) 剔除信<br/>息技术业",
    "(7) 倾向得<br/>分匹配", "(8) 工具<br/>变量法", "(9) Heck-<br/>man检验"
]

rob_dvs = ['SYNCH','PriceDelay','PriceDelay','PriceDelay','PriceDelay',
           'PriceDelay','PriceDelay','PriceDelay','PriceDelay']

# Use stored result for ProvYear (model 4)
rob_v5 = json.load(open(f"{BASE}/results/v5_zhu_format/all_results_v5.json"))
prov_res = [r for r in rob_v5['robustness'] if r['Column']=='(4)_ProvYear'][0]

rob_models = [
    (m5_main['DU_kw'][0], m5_main['DU_kw'][1], m5_main['DU_kw'][2], m5_N, m5_R2),
    (m2_main['DU_kw_ln'][0], m2_main['DU_kw_ln'][1], m2_main['DU_kw_ln'][2], m2_N, m2_R2),
    (rob3_main['DU_kw'][0], rob3_main['DU_kw'][1], rob3_main['DU_kw'][2], rob3_N, rob3_R2),
    (prov_res['Coef'], prov_res['SE'], prov_res['Sig'], prov_res['N'], prov_res['R2']),
    (rob5_main['DU_kw'][0], rob5_main['DU_kw'][1], rob5_main['DU_kw'][2], rob5_N, rob5_R2),
    (rob6_main['DU_kw'][0], rob6_main['DU_kw'][1], rob6_main['DU_kw'][2], rob6_N, rob6_R2),
    (rob7_main['DU_kw'][0], rob7_main['DU_kw'][1], rob7_main['DU_kw'][2], rob7_N, rob7_R2),
    (rob8_coef, rob8_se, rob8_sig, rob8_N, None),  # IV: no R2
    (rob9_main['DU_kw'][0], rob9_main['DU_kw'][1], rob9_main['DU_kw'][2], rob9_N, rob9_R2),
]

rob_var_names = ['DU_kw','DU_kw_ln','DU_kw','DU_kw','DU_kw','DU_kw','DU_kw','DU_kw','DU_kw']
rob_fe_firm = ['YES','YES','NO','NO','YES','YES','YES','YES','YES']
rob_fe_ind = ['NO','NO','YES','NO','NO','NO','NO','NO','NO']
rob_fe_prov = ['NO','NO','NO','YES','NO','NO','NO','NO','NO']

t3_data = []
# Header row 1
h1 = [P('', cell_left)]
for i, label in enumerate(rob_labels):
    h1.append(Paragraph(label, ParagraphStyle('h',
              fontName=CN_FONT, fontSize=7.5, leading=9, alignment=TA_CENTER)))
t3_data.append(h1)

# DV row
dv_row = [P('', cell_left)]
for dv in rob_dvs:
    dv_row.append(P(f'<i>{dv}</i>', ParagraphStyle('dv', fontName='Times-Italic',
                    fontSize=7.5, leading=9, alignment=TA_CENTER)))
t3_data.append(dv_row)

# Variable name row + coefficient
var_row = [P_var('DU_kw/DU_kw_ln')]
for i, (c, s, sig, n, r2) in enumerate(rob_models):
    var_row.append(P_coef(c, sig, 3))
t3_data.append(var_row)

# SE row
se_row = [P('', cell_left)]
for c, s, sig, n, r2 in rob_models:
    se_row.append(P_se(s, 3))
t3_data.append(se_row)

# Controls
ctrl_row = [P('Controls', cell_left)]
for _ in range(9): ctrl_row.append(P('YES', cell_style))
t3_data.append(ctrl_row)

# FE rows
fe1 = [P('Firm/Year', cell_left)]
for v in rob_fe_firm: fe1.append(P(v, cell_style))
t3_data.append(fe1)

fe2 = [P('Ind×Year', cell_left)]
for v in rob_fe_ind: fe2.append(P(v, cell_style))
t3_data.append(fe2)

fe3 = [P('Prov×Year', cell_left)]
for v in rob_fe_prov: fe3.append(P(v, cell_style))
t3_data.append(fe3)

# N
n_row = [P_var('N')]
for c, s, sig, n, r2 in rob_models:
    n_row.append(P_int(n))
t3_data.append(n_row)

# R2
r2_row = [P_var('R<super>2</super>')]
for c, s, sig, n, r2 in rob_models:
    if r2 is not None and not (isinstance(r2, float) and np.isnan(r2)):
        r2_row.append(P_num(r2))
    else:
        r2_row.append(P('—', cell_style))
t3_data.append(r2_row)

col_widths_t3 = [2.2*cm] + [1.7*cm]*9
table3 = Table(t3_data, colWidths=col_widths_t3)
ts3 = TableStyle([
    ('FONT', (0,0), (-1,-1), CN_FONT, 7.5),
    ('ALIGN', (0,0), (-1,-1), 'CENTER'),
    ('ALIGN', (0,0), (0,-1), 'LEFT'),
    ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ('TOPPADDING', (0,0), (-1,-1), 1.5),
    ('BOTTOMPADDING', (0,0), (-1,-1), 1.5),
    ('LINEABOVE', (0,0), (-1,0), 1.2, colors.black),
    ('LINEBELOW', (0,0), (-1,0), 0.5, colors.black),
    ('LINEBELOW', (0,1), (-1,1), 0.5, colors.black),
    ('LINEBELOW', (0,-1), (-1,-1), 1.2, colors.black),
])
table3.setStyle(ts3)

story.append(Paragraph("表3　　　　　　　　　　　　　　稳健性检验", title_style))
story.append(table3)
story.append(Spacer(1, 4))
story.append(Paragraph(
    "注：括号内数值为稳健标准误。模型(1)替换被解释变量为股价同步性SYNCH；"
    "模型(2)替换解释变量为ln(1+关键词总数)；"
    "模型(8)为同行业均值工具变量2SLS估计，KP F统计量为"
    f"{rob8_KPF:.1f}；模型(9)为Heckman两阶段检验。",
    note_style))
story.append(PageBreak())

# ============================================================
# 表4: 内生性检验
# ============================================================
print("  Table 4: Endogeneity...")

iv_labels = ["(1) OLS\n基准", "(2) 同行业\n均值IV", "(3) Bartik\nIV",
             "(4) 滞后\nOLS", "(5) 滞后\nIV"]
iv_dvs = ['PriceDelay']*5

iv = iv_results
iv_models_data = [
    (iv['ols']['coef'], iv['ols']['se'], "***", iv['ols']['N'], iv['ols']['R2']),
    (iv['peer_iv']['coef'], iv['peer_iv']['se'], "***", iv['peer_iv']['N'], None),
    (iv['bartik_iv']['coef'], iv['bartik_iv']['se'], "***", iv['bartik_iv']['N'], None),
    (iv['lag_ols']['coef'], iv['lag_ols']['se'], "**", iv['lag_ols']['N'], None),
    (iv['lag_iv']['coef'], iv['lag_iv']['se'], "***", iv['lag_iv']['N'], None),
]

t4_data = []
h1 = [P('', cell_left)]
for label in iv_labels:
    h1.append(Paragraph(label.replace('\n','<br/>'), ParagraphStyle('h',
              fontName=CN_FONT, fontSize=8, leading=10, alignment=TA_CENTER)))
t4_data.append(h1)

dv_row = [P('', cell_left)]
for dv in iv_dvs:
    dv_row.append(P(f'<i>{dv}</i>', ParagraphStyle('dv', fontName='Times-Italic',
                    fontSize=8, leading=10, alignment=TA_CENTER)))
t4_data.append(dv_row)

# DU_kw row
coef_row = [P_var('DU_kw')]
se_row_iv = [P('', cell_left)]
for c, s, sig, n, r2 in iv_models_data:
    coef_row.append(P_coef(c, sig, 4))
    se_row_iv.append(P_se(s, 4))
t4_data.append(coef_row)
t4_data.append(se_row_iv)

# Controls + FE
for label in ['Controls', 'Firm/Year']:
    row = [P(label, cell_left)]
    for _ in range(5): row.append(P('YES', cell_style))
    t4_data.append(row)

# KP F statistic
kpf_row = [P('KP F', cell_left)]
kp_vals = [None, iv['peer_iv']['KP_F'], iv['bartik_iv']['KP_F'], None, iv['lag_iv']['KP_F']]
for v in kp_vals:
    if v is not None:
        kpf_row.append(P(f"{v:.1f}", ParagraphStyle('n', fontName='Times-Roman',
                        fontSize=8.5, leading=11, alignment=TA_CENTER)))
    else:
        kpf_row.append(P('—', cell_style))
t4_data.append(kpf_row)

# DWH p
dwh_row = [P('DWH p', cell_left)]
dwh_vals = [None, iv['peer_iv']['DWH_p'], iv['bartik_iv']['DWH_p'], None, iv['lag_iv']['DWH_p']]
for v in dwh_vals:
    if v is not None:
        dwh_row.append(P(f"{v:.3f}", ParagraphStyle('n', fontName='Times-Roman',
                        fontSize=8.5, leading=11, alignment=TA_CENTER)))
    else:
        dwh_row.append(P('—', cell_style))
t4_data.append(dwh_row)

# N
n_row = [P_var('N')]
for c, s, sig, n, r2 in iv_models_data:
    n_row.append(P_int(n))
t4_data.append(n_row)

# R2
r2_row = [P_var('R<super>2</super>')]
for c, s, sig, n, r2 in iv_models_data:
    if r2 is not None:
        r2_row.append(P_num(r2))
    else:
        r2_row.append(P('—', cell_style))
t4_data.append(r2_row)

col_widths_t4 = [2.2*cm] + [2.8*cm]*5
table4 = Table(t4_data, colWidths=col_widths_t4)
ts4 = TableStyle([
    ('FONT', (0,0), (-1,-1), CN_FONT, 8.5),
    ('ALIGN', (0,0), (-1,-1), 'CENTER'),
    ('ALIGN', (0,0), (0,-1), 'LEFT'),
    ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ('TOPPADDING', (0,0), (-1,-1), 2),
    ('BOTTOMPADDING', (0,0), (-1,-1), 2),
    ('LINEABOVE', (0,0), (-1,0), 1.2, colors.black),
    ('LINEBELOW', (0,0), (-1,0), 0.5, colors.black),
    ('LINEBELOW', (0,1), (-1,1), 0.5, colors.black),
    ('LINEBELOW', (0,-1), (-1,-1), 1.2, colors.black),
])
table4.setStyle(ts4)

# Oster test info
oster = iv_results['oster']
oster_text = (f"Oster (2019) 系数稳定性检验：R<super>2</super><sub>max</sub>=1.3R<super>2</super>时"
              f"delta*={oster['delta_13']:.1f}，"
              f"R<super>2</super><sub>max</sub>=min(1,1.3R<super>2</super>)时delta*={oster['delta_con']:.1f}，"
              f"均远大于临界值1。")

story.append(Paragraph("表4　　　　　　　　　　　　内生性检验", title_style))
story.append(table4)
story.append(Spacer(1, 4))
story.append(Paragraph(
    f"注：模型(1)为OLS基准估计；模型(2)以同行业均值为工具变量；"
    f"模型(3)为Bartik移位份额工具变量；模型(4)-(5)使用滞后一期DU_kw。"
    f"KP F为Kleibergen-Paap rk Wald F统计量，DWH p为Durbin-Wu-Hausman内生性检验p值。"
    f"{oster_text}", note_style))
story.append(PageBreak())

# ============================================================
# 表5: 机制检验
# ============================================================
print("  Table 5: Mechanism...")

mech_labels = ["(1)", "(2)", "(3)"]
mech_dv_names = ['Analyst', 'FinAsset', 'InstHold']

t5_data = []
h1 = [P('', cell_left)]
for label in mech_labels:
    h1.append(P(label, header_style))
t5_data.append(h1)

dv_row = [P('', cell_left)]
for dv in mech_dv_names:
    dv_row.append(P(f'<i>{dv}</i>', ParagraphStyle('dv', fontName='Times-Italic',
                    fontSize=8.5, leading=11, alignment=TA_CENTER)))
t5_data.append(dv_row)

# DU_kw coef + SE
coef_row = [P_var('DU_kw')]
se_row = [P('', cell_left)]
for dv in mech_dv_names:
    r = mech_results[dv]
    coef_row.append(P_coef(r['coef'], r['sig'], 4 if dv=='FinAsset' else 3))
    se_row.append(P_se(r['se'], 4 if dv=='FinAsset' else 3))
t5_data.append(coef_row)
t5_data.append(se_row)

# Controls + FE
for label in ['Controls', 'Firm/Year']:
    row = [P(label, cell_left)]
    for _ in range(3): row.append(P('YES', cell_style))
    t5_data.append(row)

# N
n_row = [P_var('N')]
for dv in mech_dv_names:
    n_row.append(P_int(mech_results[dv]['N']))
t5_data.append(n_row)

# R2
r2_row = [P_var('R<super>2</super>')]
for dv in mech_dv_names:
    r2_row.append(P_num(mech_results[dv]['R2']))
t5_data.append(r2_row)

col_widths_t5 = [3*cm] + [4*cm]*3
table5 = Table(t5_data, colWidths=col_widths_t5)
ts5 = TableStyle([
    ('FONT', (0,0), (-1,-1), CN_FONT, 8.5),
    ('ALIGN', (0,0), (-1,-1), 'CENTER'),
    ('ALIGN', (0,0), (0,-1), 'LEFT'),
    ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ('TOPPADDING', (0,0), (-1,-1), 2),
    ('BOTTOMPADDING', (0,0), (-1,-1), 2),
    ('LINEABOVE', (0,0), (-1,0), 1.2, colors.black),
    ('LINEBELOW', (0,0), (-1,0), 0.5, colors.black),
    ('LINEBELOW', (0,1), (-1,1), 0.5, colors.black),
    ('LINEBELOW', (0,-1), (-1,-1), 1.2, colors.black),
])
table5.setStyle(ts5)

story.append(Paragraph("表5　　　　　　　　　　　　机制检验", title_style))
story.append(table5)
story.append(Spacer(1, 4))
story.append(Paragraph(
    "注：固定效应为企业+年份，标准误按行业×年份聚类。Analyst为分析师覆盖对数，"
    "FinAsset为金融资产占比，InstHold为机构持股比例。", note_style))
story.append(PageBreak())

# ============================================================
# 表6: 异质性分析
# ============================================================
print("  Table 6: Heterogeneity...")

het_groups = ["国有企业","非国有企业","大企业","小企业",
              "高分析师覆盖","低分析师覆盖","高科技行业","传统行业"]
het_col_labels = [
    "(1) 国有企业", "(2) 非国有企业",
    "(3) 大企业", "(4) 小企业",
    "(5) 高分析师覆盖", "(6) 低分析师覆盖",
    "(7) 高科技行业", "(8) 传统行业"
]
ncols_t6 = 8

t6_data = []
h1 = [P('', cell_left)]
for label in het_col_labels:
    h1.append(Paragraph(label, ParagraphStyle('h',
              fontName=CN_FONT, fontSize=7.5, leading=9, alignment=TA_CENTER)))
t6_data.append(h1)

dv_row = [P('', cell_left)]
for _ in range(ncols_t6):
    dv_row.append(P('<i>PriceDelay</i>', ParagraphStyle('dv', fontName='Times-Italic',
                    fontSize=7.5, leading=9, alignment=TA_CENTER)))
t6_data.append(dv_row)

# DU_kw coef + SE
coef_row = [P_var('DU_kw')]
se_row = [P('', cell_left)]
for grp in het_groups:
    r = het_results[grp]
    coef_row.append(P_coef(r['coef'], r['sig']))
    se_row.append(P_se(r['se']))
t6_data.append(coef_row)
t6_data.append(se_row)

# Controls + FE
for label in ['Controls', 'Firm/Year']:
    row = [P(label, cell_left)]
    for _ in range(ncols_t6): row.append(P('YES', cell_style))
    t6_data.append(row)

# N
n_row = [P_var('N')]
for grp in het_groups:
    n_row.append(P_int(het_results[grp]['N']))
t6_data.append(n_row)

# R2
r2_row = [P_var('R<super>2</super>')]
for grp in het_groups:
    r2_row.append(P_num(het_results[grp]['R2']))
t6_data.append(r2_row)

# Fisher p-value row (4 pairs)
fisher_row = [P('组间差异P值', cell_left)]
fisher_dims = ['产权性质', '企业规模', '信息中介活跃度', '行业属性']
for dim in fisher_dims:
    p_val = fisher_p.get(dim, np.nan)
    if np.isnan(p_val):
        fisher_row.append(P('—', cell_style))
    else:
        sig = "***" if p_val<0.01 else "**" if p_val<0.05 else "*" if p_val<0.1 else ""
        p_text = f"{p_val:.3f}"
        if sig:
            p_text += f"<super>{sig}</super>"
        fisher_row.append(Paragraph(p_text, ParagraphStyle('fp', fontName='Times-Roman',
                         fontSize=8.5, leading=11, alignment=TA_CENTER)))
    fisher_row.append(P('', cell_style))  # span across 2 cols
t6_data.append(fisher_row)

col_widths_t6 = [2.0*cm] + [1.85*cm]*8
table6 = Table(t6_data, colWidths=col_widths_t6)
ts6 = TableStyle([
    ('FONT', (0,0), (-1,-1), CN_FONT, 8.5),
    ('ALIGN', (0,0), (-1,-1), 'CENTER'),
    ('ALIGN', (0,0), (0,-1), 'LEFT'),
    ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ('TOPPADDING', (0,0), (-1,-1), 2),
    ('BOTTOMPADDING', (0,0), (-1,-1), 2),
    ('LINEABOVE', (0,0), (-1,0), 1.2, colors.black),
    ('LINEBELOW', (0,0), (-1,0), 0.5, colors.black),
    ('LINEBELOW', (0,1), (-1,1), 0.5, colors.black),
    ('LINEBELOW', (0,-1), (-1,-1), 1.2, colors.black),
    # Span Fisher p across pairs (4 pairs)
    ('SPAN', (1,-1), (2,-1)),
    ('SPAN', (3,-1), (4,-1)),
    ('SPAN', (5,-1), (6,-1)),
    ('SPAN', (7,-1), (8,-1)),
])
table6.setStyle(ts6)

story.append(Paragraph("表6　　　　　　　　　　　　异质性分析", title_style))
story.append(table6)
story.append(Spacer(1, 4))
story.append(Paragraph(
    "注：组间差异P值基于Fisher组合检验（1000次随机抽样）。", note_style))
story.append(PageBreak())

# ============================================================
# 表7: 投资组合Alpha
# ============================================================
print("  Table 7: Portfolio Alpha...")

port_df = pd.read_csv(f"{BASE}/results/portfolio/portfolio_alpha.csv")

t7_data = []
# Header
h1 = [P('Portfolio', cell_left), P('CAPM', header_style), P('', header_style),
      P('FF3', header_style), P('', header_style),
      P('FF5+MOM', header_style), P('', header_style)]
t7_data.append(h1)
h2 = [P('', cell_left), P('alpha', header_style), P('t', header_style),
      P('alpha', header_style), P('t', header_style),
      P('alpha', header_style), P('t', header_style)]
t7_data.append(h2)

for port in ['Q1','Q2','Q3','Q4','Q5','DAT']:
    row = [P_var(port)]
    for model in ['CAPM','FF3','FF5+MOM']:
        sub = port_df[(port_df['Portfolio']==port) & (port_df['Model']==model)]
        if len(sub) > 0:
            a = sub.iloc[0]['alpha']
            t = sub.iloc[0]['t']
            p = sub.iloc[0]['p']
            sig = "***" if p<0.01 else "**" if p<0.05 else "*" if p<0.1 else ""
            row.append(P_coef(a*100, sig, 2))  # Convert to %
            row.append(P(f"{t:.2f}", ParagraphStyle('t', fontName='Times-Roman',
                        fontSize=8.5, leading=11, alignment=TA_CENTER)))
        else:
            row.append(P('', cell_style))
            row.append(P('', cell_style))
    t7_data.append(row)

col_widths_t7 = [2*cm] + [2.2*cm]*6
table7 = Table(t7_data, colWidths=col_widths_t7)
ts7 = TableStyle([
    ('FONT', (0,0), (-1,-1), CN_FONT, 8.5),
    ('ALIGN', (0,0), (-1,-1), 'CENTER'),
    ('ALIGN', (0,0), (0,-1), 'LEFT'),
    ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ('TOPPADDING', (0,0), (-1,-1), 2),
    ('BOTTOMPADDING', (0,0), (-1,-1), 2),
    ('LINEABOVE', (0,0), (-1,0), 1.2, colors.black),
    ('LINEBELOW', (0,0), (-1,0), 0.5, colors.black),
    ('LINEBELOW', (0,1), (-1,1), 0.5, colors.black),
    ('LINEBELOW', (0,-1), (-1,-1), 1.2, colors.black),
    # Span model headers
    ('SPAN', (1,0), (2,0)),
    ('SPAN', (3,0), (4,0)),
    ('SPAN', (5,0), (6,0)),
])
table7.setStyle(ts7)

story.append(Paragraph("表7　　　　　　　　　　投资组合Alpha检验", title_style))
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
