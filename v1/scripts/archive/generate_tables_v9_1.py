"""
生成朱康(2025)会计研究格式的全部表格PDF — v9.1
v9.1修复:
  - Ind2行业代码用公司众数填补(修复模型5样本泄漏bug)
  - 表2: 删除手算Constant, 加Controls YES/NO行, Top1Share/SOE/InstHold用4位小数
  - 本地路径替换
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
# 0. 字体注册
# ============================================================
font_paths = [
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/noto-cjk/NotoSansCJKsc-Regular.otf",
]
CN_FONT = "Helvetica"
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

# ---- Robustness models ----
print("  Robustness models...")

# Prov×Year (from stored results)
rob_v5 = json.load(open(f"{BASE}/results/v5_zhu_format/all_results_v5.json"))
prov_res = [r for r in rob_v5['robustness'] if r['Column']=='(4)_ProvYear'][0]

# 剔除2024
reg_no24 = reg[reg['year'] != 2024].copy()
rob_no24 = pf.feols(f"PriceDelay ~ DU_kw + {ctrl_str} | Stkcd_str + year_str",
                     data=reg_no24, vcov={"CRV1":"IndYear"})
rob_no24_main = extract_coefs(rob_no24, ["DU_kw"])
rob_no24_N, rob_no24_R2 = get_nobs(rob_no24), get_r2(rob_no24)

# 剔除IT
reg_noit = reg[~reg['IndustryCodeC'].str.startswith('I', na=False)].copy()
rob_noit = pf.feols(f"PriceDelay ~ DU_kw + {ctrl_str} | Stkcd_str + year_str",
                     data=reg_noit, vcov={"CRV1":"IndYear"})
rob_noit_main = extract_coefs(rob_noit, ["DU_kw"])
rob_noit_N, rob_noit_R2 = get_nobs(rob_noit), get_r2(rob_noit)

# 仅主板
reg_main = reg[reg['MainBoard']==True].copy()
rob_main = pf.feols(f"PriceDelay ~ DU_kw + {ctrl_str} | Stkcd_str + year_str",
                     data=reg_main, vcov={"CRV1":"IndYear"})
rob_main_coefs = extract_coefs(rob_main, ["DU_kw"])
rob_main_N, rob_main_R2 = get_nobs(rob_main), get_r2(rob_main)

# PSM
from sklearn.linear_model import LogisticRegression
reg_psm = reg.copy()
reg_psm['Treat'] = (reg_psm['DU_kw'] > reg_psm['DU_kw'].median()).astype(int)
X_psm = reg_psm[controls].values
y_psm = reg_psm['Treat'].values
lr = LogisticRegression(max_iter=1000, C=1.0)
lr.fit(X_psm, y_psm)
reg_psm['pscore'] = lr.predict_proba(X_psm)[:,1]
caliper = 0.25 * reg_psm['pscore'].std()
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
rob_psm = pf.feols(f"PriceDelay ~ DU_kw + {ctrl_str} | Stkcd_str + year_str",
                    data=reg_matched, vcov={"CRV1":"IndYear"})
rob_psm_main = extract_coefs(rob_psm, ["DU_kw"])
rob_psm_N, rob_psm_R2 = get_nobs(rob_psm), get_r2(rob_psm)
print(f"  PSM matched: N={rob_psm_N}")

# IV (stored)
iv_results = json.load(open(f"{BASE}/results/endogeneity_v4/endogeneity_v4_results.json"))
rob_iv_coef = iv_results['peer_iv']['coef']
rob_iv_se = iv_results['peer_iv']['se']
rob_iv_p = iv_results['peer_iv']['p']
rob_iv_sig = "***" if rob_iv_p<0.01 else "**" if rob_iv_p<0.05 else "*" if rob_iv_p<0.1 else ""
rob_iv_N = iv_results['peer_iv']['N']
rob_iv_KPF = iv_results['peer_iv']['KP_F']

# Heckman
from scipy.stats import norm as sp_norm
reg_heck = reg.copy()
reg_heck['DU_positive'] = (reg_heck['DU_kw'] > reg_heck['DU_kw'].median()).astype(int)
probit_X = reg_heck[controls].values
probit_y = reg_heck['DU_positive'].values
lr_heck = LogisticRegression(max_iter=1000)
lr_heck.fit(probit_X, probit_y)
prob = lr_heck.predict_proba(probit_X)[:,1]
prob = np.clip(prob, 1e-6, 1-1e-6)
reg_heck['IMR'] = sp_norm.pdf(sp_norm.ppf(prob)) / prob
rob_heck = pf.feols(f"PriceDelay ~ DU_kw + IMR + {ctrl_str} | Stkcd_str + year_str",
                     data=reg_heck, vcov={"CRV1":"IndYear"})
rob_heck_main = extract_coefs(rob_heck, ["DU_kw"])
rob_heck_N, rob_heck_R2 = get_nobs(rob_heck), get_r2(rob_heck)

print("  Robustness done.")

# ---- Mechanism models ----
print("  Mechanism models...")
mech_dvs = ['Analyst', 'FinAsset', 'InstHold']
mech_results = {}
for dv in mech_dvs:
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

# Panel A: 分组回归
for label, mask in [("国有企业", reg['SOE']==1), ("非国有企业", reg['SOE']==0)]:
    sub = reg[mask].copy()
    m = pf.feols(base_fml, data=sub, vcov={"CRV1":"IndYear"})
    c, s, sig = extract_coefs(m, ["DU_kw"])["DU_kw"]
    het_results[label] = {"coef": c, "se": s, "sig": sig, "N": get_nobs(m), "R2": get_r2(m)}

med_size = reg['Size'].median()
for label, mask in [("大企业", reg['Size']>=med_size), ("小企业", reg['Size']<med_size)]:
    sub = reg[mask].copy()
    m = pf.feols(base_fml, data=sub, vcov={"CRV1":"IndYear"})
    c, s, sig = extract_coefs(m, ["DU_kw"])["DU_kw"]
    het_results[label] = {"coef": c, "se": s, "sig": sig, "N": get_nobs(m), "R2": get_r2(m)}

med_analyst = reg['Analyst'].median()
for label, mask in [("高分析师覆盖", reg['Analyst']>=med_analyst), ("低分析师覆盖", reg['Analyst']<med_analyst)]:
    sub = reg[mask].copy()
    m = pf.feols(base_fml, data=sub, vcov={"CRV1":"IndYear"})
    c, s, sig = extract_coefs(m, ["DU_kw"])["DU_kw"]
    het_results[label] = {"coef": c, "se": s, "sig": sig, "N": get_nobs(m), "R2": get_r2(m)}

hitech_codes = ['I', 'M']
for label, is_hitech in [("高科技行业", True), ("传统行业", False)]:
    mask = reg['IndustryCodeC'].str[0].isin(hitech_codes) if is_hitech else ~reg['IndustryCodeC'].str[0].isin(hitech_codes)
    sub = reg[mask].copy()
    m = pf.feols(base_fml, data=sub, vcov={"CRV1":"IndYear"})
    c, s, sig = extract_coefs(m, ["DU_kw"])["DU_kw"]
    het_results[label] = {"coef": c, "se": s, "sig": sig, "N": get_nobs(m), "R2": get_r2(m)}

# Panel A: 交互项检验组间差异 (替代Fisher)
print("  Interaction term tests for group differences...")
interact_p = {}
# SOE交互项
m_soe = pf.feols(f"PriceDelay ~ DU_kw + DU_kw:SOE + {ctrl_str} | Stkcd_str + year_str",
                  data=reg, vcov={"CRV1":"IndYear"})
interact_p['产权性质'] = float(m_soe.pvalue()['DU_kw:SOE'])

# Size交互项
reg['HighSize'] = (reg['Size'] >= med_size).astype(int)
m_size = pf.feols(f"PriceDelay ~ DU_kw + DU_kw:HighSize + {ctrl_str} | Stkcd_str + year_str",
                   data=reg, vcov={"CRV1":"IndYear"})
interact_p['企业规模'] = float(m_size.pvalue()['DU_kw:HighSize'])

# Analyst交互项
reg['HighAnalyst'] = (reg['Analyst'] >= med_analyst).astype(int)
m_ana = pf.feols(f"PriceDelay ~ DU_kw + DU_kw:HighAnalyst + {ctrl_str} | Stkcd_str + year_str",
                  data=reg, vcov={"CRV1":"IndYear"})
interact_p['信息中介活跃度'] = float(m_ana.pvalue()['DU_kw:HighAnalyst'])

# HighTech交互项
reg['HighTech'] = reg['IndustryCodeC'].str[0].isin(hitech_codes).astype(int)
m_tech = pf.feols(f"PriceDelay ~ DU_kw + DU_kw:HighTech + {ctrl_str} | Stkcd_str + year_str",
                   data=reg, vcov={"CRV1":"IndYear"})
interact_p['行业属性'] = float(m_tech.pvalue()['DU_kw:HighTech'])
print(f"  Interaction p-values: {interact_p}")

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

pdf_path = f"{BASE}/results/v9_tables/regression_tables_v9_1.pdf"
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

def P_var(text):
    return Paragraph(f"<i>{text}</i>", ParagraphStyle('Var', fontName='Times-Italic',
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

t1_data = [[P('', cell_left), P('N', header_style), P('Mean', header_style),
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
t2_data.append([P('', cell_left)] + [P(f'<i>{dv}</i>', ParagraphStyle('dv', fontName='Times-Italic',
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
# 表3: 稳健性检验 (8列, 去掉Ind×Year, 加仅主板)
# ============================================================
print("  Table 3...")

rob_labels = [
    "(1) 替换被解<br/>释变量",
    "(2) 替换解<br/>释变量",
    "(3) 控制<br/>Prov×Year",
    "(4) 剔除<br/>2024年",
    "(5) 剔除信息<br/>技术业",
    "(6) 仅<br/>主板",
    "(7) 倾向得<br/>分匹配",
    "(8) Heckman<br/>两阶段",
]
rob_dvs = ['SYNCH'] + ['PriceDelay']*7

rob_models = [
    (m5_main['DU_kw'][0], m5_main['DU_kw'][1], m5_main['DU_kw'][2], m5_N, m5_R2),
    (m2_main['DU_kw_ln'][0], m2_main['DU_kw_ln'][1], m2_main['DU_kw_ln'][2], m2_N, m2_R2),
    (prov_res['Coef'], prov_res['SE'], prov_res['Sig'], prov_res['N'], prov_res['R2']),
    (rob_no24_main['DU_kw'][0], rob_no24_main['DU_kw'][1], rob_no24_main['DU_kw'][2], rob_no24_N, rob_no24_R2),
    (rob_noit_main['DU_kw'][0], rob_noit_main['DU_kw'][1], rob_noit_main['DU_kw'][2], rob_noit_N, rob_noit_R2),
    (rob_main_coefs['DU_kw'][0], rob_main_coefs['DU_kw'][1], rob_main_coefs['DU_kw'][2], rob_main_N, rob_main_R2),
    (rob_psm_main['DU_kw'][0], rob_psm_main['DU_kw'][1], rob_psm_main['DU_kw'][2], rob_psm_N, rob_psm_R2),
    (rob_heck_main['DU_kw'][0], rob_heck_main['DU_kw'][1], rob_heck_main['DU_kw'][2], rob_heck_N, rob_heck_R2),
]

rob_fe_firm = ['YES','YES','NO','YES','YES','YES','YES','YES']
rob_fe_prov = ['NO','NO','YES','NO','NO','NO','NO','NO']

t3_data = []
t3_data.append([P('', cell_left)] + [P_hdr(l) for l in rob_labels])
t3_data.append([P('', cell_left)] + [P(f'<i>{dv}</i>', ParagraphStyle('dv', fontName='Times-Italic',
                fontSize=7.5, leading=9, alignment=TA_CENTER)) for dv in rob_dvs])

# Variable name + coef
var_names_rob = ['DU_kw','DU_kw_ln','DU_kw','DU_kw','DU_kw','DU_kw','DU_kw','DU_kw']
cr = [P_var('DU_kw/DU_kw_ln')]
sr = [P('', cell_left)]
for c, s, sig, n, r2 in rob_models:
    cr.append(P_coef(c, sig))
    sr.append(P_se(s))
t3_data.append(cr)
t3_data.append(sr)

t3_data.append([P('Controls', cell_left)] + [P('YES', cell_style)]*8)
t3_data.append([P('Firm/Year', cell_left)] + [P(v, cell_style) for v in rob_fe_firm])
t3_data.append([P('Prov×Year', cell_left)] + [P(v, cell_style) for v in rob_fe_prov])
t3_data.append([P_var('N')] + [P_int(n) for c,s,sig,n,r2 in rob_models])
t3_data.append([P_var('R<super>2</super>')] + [P_num(r2) if r2 is not None else P('—', cell_style) for c,s,sig,n,r2 in rob_models])

table3 = Table(t3_data, colWidths=[2.2*cm] + [1.95*cm]*8)
table3.setStyle(TableStyle([
    ('FONT', (0,0), (-1,-1), CN_FONT, 7.5),
    ('ALIGN', (0,0), (-1,-1), 'CENTER'), ('ALIGN', (0,0), (0,-1), 'LEFT'),
    ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ('TOPPADDING', (0,0), (-1,-1), 1.5), ('BOTTOMPADDING', (0,0), (-1,-1), 1.5),
    ('LINEABOVE', (0,0), (-1,0), 1.2, colors.black),
    ('LINEBELOW', (0,0), (-1,0), 0.5, colors.black),
    ('LINEBELOW', (0,1), (-1,1), 0.5, colors.black),
    ('LINEBELOW', (0,-1), (-1,-1), 1.2, colors.black),
]))

story.append(Paragraph("表3　　　　　　　　　　　　　　稳健性检验", title_style))
story.append(table3)
story.append(Spacer(1, 4))
story.append(Paragraph(
    "注：括号内数值为稳健标准误。模型(1)替换被解释变量为股价同步性SYNCH；"
    "模型(2)替换解释变量为ln(1+关键词总数)；"
    "模型(6)仅保留主板上市公司样本。", note_style))
story.append(PageBreak())

# ============================================================
# 表4: 内生性检验
# ============================================================
print("  Table 4...")

iv = iv_results
iv_models_data = [
    (iv['ols']['coef'], iv['ols']['se'], sig_stars(iv['ols']['p']), iv['ols']['N'], iv['ols']['R2']),
    (iv['peer_iv']['coef'], iv['peer_iv']['se'], sig_stars(iv['peer_iv']['p']), iv['peer_iv']['N'], None),
    (iv['bartik_iv']['coef'], iv['bartik_iv']['se'], sig_stars(iv['bartik_iv']['p']), iv['bartik_iv']['N'], None),
    (iv['lag_ols']['coef'], iv['lag_ols']['se'], sig_stars(iv['lag_ols']['p']), iv['lag_ols']['N'], None),
    (iv['lag_iv']['coef'], iv['lag_iv']['se'], sig_stars(iv['lag_iv']['p']), iv['lag_iv']['N'], None),
]

t4_data = []
iv_labels = ["(1) OLS基准", "(2) 同行业均值IV", "(3) Bartik IV", "(4) 滞后OLS", "(5) 滞后IV"]
t4_data.append([P('', cell_left)] + [P_hdr(l) for l in iv_labels])
t4_data.append([P('', cell_left)] + [P(f'<i>PriceDelay</i>', ParagraphStyle('dv', fontName='Times-Italic',
                fontSize=8, leading=10, alignment=TA_CENTER)) for _ in range(5)])

cr = [P_var('DU_kw')]
sr = [P('', cell_left)]
for c, s, sig, n, r2 in iv_models_data:
    cr.append(P_coef(c, sig, 4))
    sr.append(P_se(s, 4))
t4_data.append(cr)
t4_data.append(sr)

for label in ['Controls', 'Firm/Year']:
    t4_data.append([P(label, cell_left)] + [P('YES', cell_style)]*5)

kp_vals = [None, iv['peer_iv']['KP_F'], iv['bartik_iv']['KP_F'], None, iv['lag_iv']['KP_F']]
t4_data.append([P('KP F', cell_left)] + [P(f"{v:.1f}" if v else '—', cell_style) for v in kp_vals])

dwh_vals = [None, iv['peer_iv']['DWH_p'], iv['bartik_iv']['DWH_p'], None, iv['lag_iv']['DWH_p']]
t4_data.append([P('DWH p', cell_left)] + [P(f"{v:.3f}" if v else '—', cell_style) for v in dwh_vals])

t4_data.append([P_var('N')] + [P_int(n) for c,s,sig,n,r2 in iv_models_data])
t4_data.append([P_var('R<super>2</super>')] + [P_num(r2) if r2 is not None else P('—', cell_style) for c,s,sig,n,r2 in iv_models_data])

table4 = Table(t4_data, colWidths=[2.2*cm] + [2.8*cm]*5)
table4.setStyle(TableStyle([
    ('FONT', (0,0), (-1,-1), CN_FONT, 8.5),
    ('ALIGN', (0,0), (-1,-1), 'CENTER'), ('ALIGN', (0,0), (0,-1), 'LEFT'),
    ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ('TOPPADDING', (0,0), (-1,-1), 2), ('BOTTOMPADDING', (0,0), (-1,-1), 2),
    ('LINEABOVE', (0,0), (-1,0), 1.2, colors.black),
    ('LINEBELOW', (0,0), (-1,0), 0.5, colors.black),
    ('LINEBELOW', (0,1), (-1,1), 0.5, colors.black),
    ('LINEBELOW', (0,-1), (-1,-1), 1.2, colors.black),
]))

oster = iv_results['oster']
story.append(Paragraph("表4　　　　　　　　　　　　内生性检验", title_style))
story.append(table4)
story.append(Spacer(1, 4))
story.append(Paragraph(
    f"注：模型(1)为OLS基准估计；模型(2)以同行业均值为工具变量；"
    f"模型(3)为Bartik移位份额工具变量；模型(4)-(5)使用滞后一期DU_kw。"
    f"KP F为Kleibergen-Paap rk Wald F统计量，DWH p为Durbin-Wu-Hausman内生性检验p值。"
    f"Oster (2019)系数稳定性检验：R<super>2</super><sub>max</sub>=1.3R<super>2</super>时"
    f"delta*={oster['delta_13']:.1f}，"
    f"R<super>2</super><sub>max</sub>=min(1,1.3R<super>2</super>)时delta*={oster['delta_con']:.1f}，"
    f"均远大于临界值1。", note_style))
story.append(PageBreak())

# ============================================================
# 表5: 机制检验
# ============================================================
print("  Table 5...")

t5_data = []
t5_data.append([P('', cell_left)] + [P(f'({i+1})', header_style) for i in range(3)])
t5_data.append([P('', cell_left)] + [P(f'<i>{dv}</i>', ParagraphStyle('dv', fontName='Times-Italic',
                fontSize=8.5, leading=11, alignment=TA_CENTER)) for dv in ['Analyst','FinAsset','InstHold']])

cr = [P_var('DU_kw')]
sr = [P('', cell_left)]
for dv in ['Analyst','FinAsset','InstHold']:
    r = mech_results[dv]
    dec = 4 if dv=='FinAsset' else 3
    cr.append(P_coef(r['coef'], r['sig'], dec))
    sr.append(P_se(r['se'], dec))
t5_data.append(cr)
t5_data.append(sr)

for label in ['Controls', 'Firm/Year']:
    t5_data.append([P(label, cell_left)] + [P('YES', cell_style)]*3)
t5_data.append([P_var('N')] + [P_int(mech_results[dv]['N']) for dv in ['Analyst','FinAsset','InstHold']])
t5_data.append([P_var('R<super>2</super>')] + [P_num(mech_results[dv]['R2']) for dv in ['Analyst','FinAsset','InstHold']])

table5 = Table(t5_data, colWidths=[3*cm] + [4*cm]*3)
table5.setStyle(TableStyle([
    ('FONT', (0,0), (-1,-1), CN_FONT, 8.5),
    ('ALIGN', (0,0), (-1,-1), 'CENTER'), ('ALIGN', (0,0), (0,-1), 'LEFT'),
    ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ('TOPPADDING', (0,0), (-1,-1), 2), ('BOTTOMPADDING', (0,0), (-1,-1), 2),
    ('LINEABOVE', (0,0), (-1,0), 1.2, colors.black),
    ('LINEBELOW', (0,0), (-1,0), 0.5, colors.black),
    ('LINEBELOW', (0,1), (-1,1), 0.5, colors.black),
    ('LINEBELOW', (0,-1), (-1,-1), 1.2, colors.black),
]))

story.append(Paragraph("表5　　　　　　　　　　　　机制检验", title_style))
story.append(table5)
story.append(Spacer(1, 4))
story.append(Paragraph(
    "注：固定效应为企业+年份，标准误按行业×年份聚类。Analyst为分析师覆盖对数，"
    "FinAsset为金融资产占比，InstHold为机构持股比例。", note_style))
story.append(PageBreak())

# ============================================================
# 表6: 异质性分析 — 双Panel结构
# ============================================================
print("  Table 6...")

# --- Panel A: 分组回归 + 交互项p值 ---
het_groups = ["国有企业","非国有企业","大企业","小企业",
              "高分析师覆盖","低分析师覆盖","高科技行业","传统行业"]
het_labels = [
    "(1) 国有", "(2) 非国有",
    "(3) 大企业", "(4) 小企业",
    "(5) 高分析师", "(6) 低分析师",
    "(7) 高科技", "(8) 传统"
]

t6a_data = []
# Panel A header
t6a_data.append([P('<b>Panel A: 分组回归</b>', cell_left)] + [P('', cell_style)]*8)
t6a_data.append([P('', cell_left)] + [P_hdr(l) for l in het_labels])
t6a_data.append([P('', cell_left)] + [P('<i>PriceDelay</i>', ParagraphStyle('dv', fontName='Times-Italic',
                fontSize=7.5, leading=9, alignment=TA_CENTER)) for _ in range(8)])

cr = [P_var('DU_kw')]
sr = [P('', cell_left)]
for grp in het_groups:
    r = het_results[grp]
    cr.append(P_coef(r['coef'], r['sig'], superscript=False))
    sr.append(P_se(r['se']))
t6a_data.append(cr)
t6a_data.append(sr)

for label in ['Controls', 'Firm/Year FE']:
    t6a_data.append([P(label, cell_left)] + [P('YES', cell_style)]*8)

t6a_data.append([P_var('N')] + [P_int(het_results[g]['N']) for g in het_groups])
t6a_data.append([P_var('R<super>2</super>')] + [P_num(het_results[g]['R2']) for g in het_groups])

# 交互项p值行 (替代Fisher)
interact_row = [P('交互项P值', cell_left)]
fisher_dims = ['产权性质', '企业规模', '信息中介活跃度', '行业属性']
for dim in fisher_dims:
    p_val = interact_p.get(dim, np.nan)
    p_text = f"{p_val:.3f}" if not np.isnan(p_val) else "—"
    p_sig = sig_stars(p_val) if not np.isnan(p_val) else ""
    if p_sig:
        p_text += p_sig
    interact_row.append(Paragraph(p_text, ParagraphStyle('fp', fontName='Times-Roman',
                       fontSize=8.5, leading=11, alignment=TA_CENTER)))
    interact_row.append(P('', cell_style))  # span placeholder
t6a_data.append(interact_row)

# --- Panel B: 连续变量调节效应 (4列: 变量名 + TobinQ + 空隔 + Amihud) ---
r_tq = cont_interact_results['TobinQ']
r_am = cont_interact_results['Amihud']

t6b_data = []
t6b_data.append([P('<b>Panel B: 调节效应</b>', cell_left),
                 P('', cell_style), P('', cell_style), P('', cell_style)])
t6b_data.append([P('', cell_left),
                 P_hdr('(1)'), P('', cell_style), P_hdr('(2)')])
t6b_data.append([P('', cell_left),
                 P('<i>PriceDelay</i>', ParagraphStyle('dv2', fontName='Times-Italic',
                   fontSize=7.5, leading=9, alignment=TA_CENTER)),
                 P('', cell_style),
                 P('<i>PriceDelay</i>', ParagraphStyle('dv3', fontName='Times-Italic',
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

# Build Panel A table
table6a = Table(t6a_data, colWidths=[2.2*cm] + [1.85*cm]*8)
table6a.setStyle(TableStyle([
    ('FONT', (0,0), (-1,-1), CN_FONT, 8),
    ('ALIGN', (0,0), (-1,-1), 'CENTER'), ('ALIGN', (0,0), (0,-1), 'LEFT'),
    ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ('TOPPADDING', (0,0), (-1,-1), 2), ('BOTTOMPADDING', (0,0), (-1,-1), 2),
    ('LINEABOVE', (0,0), (-1,0), 1.2, colors.black),
    ('LINEBELOW', (0,0), (-1,0), 0.5, colors.black),
    ('LINEBELOW', (0,2), (-1,2), 0.5, colors.black),
    ('LINEBELOW', (0,-1), (-1,-1), 0.5, colors.black),
    # Span interaction p-value cells (each dim spans 2 cols)
    ('SPAN', (1,-1), (2,-1)),
    ('SPAN', (3,-1), (4,-1)),
    ('SPAN', (5,-1), (6,-1)),
    ('SPAN', (7,-1), (8,-1)),
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
    "注：Panel A报告分组回归结果，交互项P值基于全样本DU_kw与分组虚拟变量的交互项回归。"
    "大/小企业按总资产对数中位数分组，高/低分析师覆盖按分析师跟踪人数中位数分组，"
    "高科技行业为证监会I类和M类。Panel B报告连续变量调节效应，"
    "TobinQ越高（成长性越好）或Amihud越大（流动性越差）时，数据要素的定价效率改善效果越弱。", note_style))
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
