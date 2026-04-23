"""
DML主结果 — Python (DoubleML)
模型: Y = theta * D + g(X) + u
方法: 部分线性模型 (PLR), CRE (Mundlak) + Lasso + RF
参考: Chernozhukov et al. (2018); Ahrens et al. (2024)

4规格:
  (1) Y=PriceDelay, D=DU_kw,    X=15控制变量 CRE
  (2) Y=PriceDelay, D=DU_kw,    X=28控制变量 CRE  ← 主规格
  (3) Y=PriceDelay, D=DU_kw_ln, X=28控制变量 CRE  ← 替换处理变量
  (4) Y=SYNCH,      D=DU_kw,    X=28控制变量 CRE  ← 替换被解释变量
"""

import numpy as np
import pandas as pd
import warnings, os, time
warnings.filterwarnings("ignore")

from doubleml import DoubleMLData, DoubleMLPLR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor

BASE = "/Users/mac/computerscience/15会计研究"
DATA = f"{BASE}/data_parquet"
OUT  = f"{BASE}/results/v12_tables"
os.makedirs(OUT, exist_ok=True)

# ================================================================
# 1. 加载数据
# ================================================================
print("=" * 60)
print("1. 加载数据")
print("=" * 60)

panel = pd.read_parquet(f"{DATA}/panel_dml.parquet")
feat  = pd.read_parquet(f"{DATA}/annual_report_features.parquet",
                         columns=["Stkcd", "year", "kw_per10k", "kw_total"])
feat["DU_kw"]    = feat["kw_per10k"]
feat["DU_kw_ln"] = np.log1p(feat["kw_total"])

synch = pd.read_parquet(f"{DATA}/price_synchronicity.parquet",
                         columns=["Stkcd", "year", "SYNCH"])

df = panel.merge(feat[["Stkcd", "year", "DU_kw", "DU_kw_ln"]], on=["Stkcd", "year"], how="left")
df = df.merge(synch, on=["Stkcd", "year"], how="left")
print(f"合并后: {len(df):,} obs")
print(f"  DU_kw非空:  {df['DU_kw'].notna().sum():,}")
print(f"  SYNCH非空:  {df['SYNCH'].notna().sum():,}")

# ================================================================
# 2. 变量定义
# ================================================================
controls_old = ["Size", "Lev", "ROA", "TobinQ", "Age", "Growth",
                "BoardSize", "IndepRatio", "Dual", "Top1Share", "SOE",
                "InstHold", "Amihud", "Analyst", "AuditType"]

controls_new = ["CFO", "RetVol", "Turnover", "Intangible", "PPE", "BM",
                "Employee", "ShareholderNum", "Manhold", "Opinion",
                "Balance", "Separation", "Market"]

all_controls = controls_old + controls_new  # 28个

# ================================================================
# 3. 行业代码 (OLS对比用)
# ================================================================
try:
    import pyfixest as pf
    fi = pd.read_parquet(f"{DATA}/firm_info.parquet", columns=["Symbol", "EndDate", "IndustryCode"])
    fi = fi.rename(columns={"Symbol": "Stkcd"})
    fi["EndDate"] = pd.to_datetime(fi["EndDate"])
    fi["year"]    = fi["EndDate"].dt.year
    fi = fi.sort_values(["Stkcd", "year"]).drop_duplicates(subset=["Stkcd", "year"], keep="last")
    fi["Ind2"] = fi["IndustryCode"].astype(str).str[:2]
    df = df.merge(fi[["Stkcd", "year", "Ind2"]], on=["Stkcd", "year"], how="left")
    df["IndYear_cluster"] = df["Ind2"].astype(str) + "_" + df["year"].astype(str)
    has_pyfixest = True
except Exception as e:
    print(f"pyfixest加载失败: {e}")
    has_pyfixest = False

# ================================================================
# 4. OLS基准 (双向FE + 行业×年份聚类)
# ================================================================
print("\n" + "=" * 60)
print("4. OLS基准 (双向FE + 行业×年份聚类SE)")
print("=" * 60)

ols_results = {}
if has_pyfixest:
    req_ols = ["PriceDelay", "DU_kw"] + all_controls + ["Stkcd", "year", "IndYear_cluster"]
    df_ols = df[req_ols].dropna().copy()
    print(f"OLS样本: {len(df_ols):,} obs, {df_ols['Stkcd'].nunique():,} firms")

    X_str_old = " + ".join(controls_old)
    X_str_all = " + ".join(all_controls)

    for label, fml in [
        ("OLS-15X", f"PriceDelay ~ DU_kw + {X_str_old} | Stkcd + year"),
        ("OLS-28X", f"PriceDelay ~ DU_kw + {X_str_all} | Stkcd + year"),
    ]:
        try:
            m = pf.feols(fml, data=df_ols, vcov={"CRV1": "IndYear_cluster"})
            ols_results[label] = {
                "coef": m.coef()["DU_kw"],
                "se":   m.se()["DU_kw"],
                "t":    m.tstat()["DU_kw"],
                "N":    len(df_ols),
            }
            print(f"  {label}: coef={ols_results[label]['coef']:.4f}, "
                  f"se={ols_results[label]['se']:.4f}, t={ols_results[label]['t']:.2f}")
        except Exception as e:
            print(f"  {label} 失败: {e}")
else:
    print("跳过OLS（无pyfixest）")

# ================================================================
# 5. DML辅助函数
# ================================================================
def make_learners():
    ml_l = Pipeline([("scaler", StandardScaler()),
                     ("model", LassoCV(cv=5, max_iter=3000, n_jobs=-1))])
    ml_m = RandomForestRegressor(n_estimators=500, max_depth=6,
                                  min_samples_leaf=10, n_jobs=-1, random_state=42)
    return ml_l, ml_m


def build_cre(df_in, y_var, d_var, ctrl_list, id_col="Stkcd", year_col="year"):
    """
    构造CRE (Mundlak) 变量：企业均值 + 年份虚拟变量
    返回 (X_mat, Y_vec, D_vec, n_obs)
    """
    req = [y_var, d_var] + ctrl_list + [id_col, year_col]
    df_c = df_in[req].dropna().copy()

    # Winsorize Y, D, controls
    for v in [y_var, d_var] + ctrl_list:
        lo = df_c[v].quantile(0.01)
        hi = df_c[v].quantile(0.99)
        df_c[v] = df_c[v].clip(lo, hi)

    # 企业均值
    fm_cols = []
    for v in ctrl_list:
        col = f"fm_{v}"
        df_c[col] = df_c.groupby(id_col)[v].transform("mean")
        fm_cols.append(col)

    # 年份虚拟变量
    yr_dummies = pd.get_dummies(df_c[year_col], prefix="yr", drop_first=True)
    yr_cols = yr_dummies.columns.tolist()
    df_c = pd.concat([df_c, yr_dummies], axis=1)

    X_cols = ctrl_list + fm_cols + yr_cols
    X_mat  = df_c[X_cols].values.astype(float)
    Y_vec  = df_c[y_var].values
    D_vec  = df_c[d_var].values

    return X_mat, Y_vec, D_vec, len(df_c)


def run_dml_spec(X_mat, Y_vec, D_vec, label="DML"):
    np.random.seed(42)
    dml_data = DoubleMLData.from_arrays(x=X_mat, y=Y_vec, d=D_vec)
    ml_l, ml_m = make_learners()
    plr = DoubleMLPLR(
        obj_dml_data=dml_data,
        ml_l=ml_l,
        ml_m=ml_m,
        n_folds=5,
        n_rep=3,
        score="partialling out",
    )
    t0 = time.time()
    plr.fit()
    elapsed = time.time() - t0

    ci  = plr.confint(level=0.95)
    res = {
        "coef": plr.coef[0],
        "se":   plr.se[0],
        "t":    plr.t_stat[0],
        "p":    plr.pval[0],
        "ci_lo": ci.iloc[0, 0],
        "ci_hi": ci.iloc[0, 1],
    }
    print(f"  {label}: coef={res['coef']:.4f}, se={res['se']:.4f}, "
          f"t={res['t']:.2f}, p={res['p']:.4f}  [{elapsed:.0f}s]")
    return res

# ================================================================
# 6. 运行4个DML规格
# ================================================================
print("\n" + "=" * 60)
print("6. DML-CRE: 4规格 (主结果)")
print("=" * 60)

specs = [
    # (label,         y_var,        d_var,      ctrl_list)
    ("DML-15X",      "PriceDelay", "DU_kw",    controls_old),
    ("DML-28X",      "PriceDelay", "DU_kw",    all_controls),
    ("DML-28X-ln",   "PriceDelay", "DU_kw_ln", all_controls),
    ("DML-28X-SYNCH","SYNCH",      "DU_kw",    all_controls),
]

dml_results = {}
for label, y_var, d_var, ctrl in specs:
    print(f"\n规格 {label} (Y={y_var}, D={d_var}, X={len(ctrl)}+CRE)")
    X_mat, Y_vec, D_vec, n = build_cre(df, y_var, d_var, ctrl)
    print(f"  样本量: {n:,}")
    res = run_dml_spec(X_mat, Y_vec, D_vec, label=label)
    res["N"] = n
    dml_results[label] = res

# ================================================================
# 7. 输出汇总
# ================================================================
print("\n" + "=" * 60)
print("7. 结果汇总")
print("=" * 60)

print("\nDML主结果:")
for k, v in dml_results.items():
    print(f"  {k:20s} coef={v['coef']:+.4f}  se={v['se']:.4f}  "
          f"t={v['t']:6.2f}  p={v['p']:.4f}  N={v['N']:,}")

# ================================================================
# 8. 保存CSV
# ================================================================
rows = []
for k, v in dml_results.items():
    rows.append({"Spec": k, **v})
pd.DataFrame(rows).to_csv(f"{OUT}/dml_main_results.csv", index=False)
print(f"\nCSV: {OUT}/dml_main_results.csv")

# OLS vs DML对比
ols_rows = []
for k, v in ols_results.items():
    ols_rows.append({"Spec": k, "coef": v["coef"], "se": v["se"], "t": v["t"],
                     "p": None, "ci_lo": None, "ci_hi": None, "N": v["N"]})
comparison = pd.DataFrame(ols_rows + rows)
comparison.to_csv(f"{OUT}/ols_vs_dml_comparison.csv", index=False)
print(f"对比CSV: {OUT}/ols_vs_dml_comparison.csv")

# ================================================================
# 9. LaTeX表格 — DML主结果 (4列)
# ================================================================
def stars(p):
    if p is None: return ""
    if p < 0.01: return "^{***}"
    if p < 0.05: return "^{**}"
    if p < 0.10: return "^{*}"
    return ""

col_labels = [
    ("DML-15X",       r"(1)\\PriceDelay"),
    ("DML-28X",       r"(2)\\PriceDelay"),
    ("DML-28X-ln",    r"(3)\\PriceDelay"),
    ("DML-28X-SYNCH", r"(4)\\SYNCH"),
]

d_labels = {
    "DML-15X":       "DU\\_kw",
    "DML-28X":       "DU\\_kw",
    "DML-28X-ln":    "DU\\_kw\\_ln",
    "DML-28X-SYNCH": "DU\\_kw",
}

ctrl_labels = {
    "DML-15X":       "15个 + CRE",
    "DML-28X":       "28个 + CRE",
    "DML-28X-ln":    "28个 + CRE",
    "DML-28X-SYNCH": "28个 + CRE",
}

latex = r"""\begin{table}[htbp]
\centering
\caption{基准回归：DML-CRE估计结果}
\label{tab:dml_main}
\begin{threeparttable}
\begin{tabular}{lcccc}
\toprule
"""
# 列标题
header_row = " & " + " & ".join([f"(\\text{{{i+1}}})" for i in range(4)]) + r" \\"
latex += header_row + "\n"
dv_row = " & " + " & ".join(["PriceDelay", "PriceDelay", "PriceDelay", "SYNCH"]) + r" \\"
latex += dv_row + "\n"
latex += r"\midrule" + "\n"

# 系数行
coef_row = "处理变量"
for k, _ in col_labels:
    r = dml_results[k]
    st = stars(r["p"])
    coef_row += f" & ${r['coef']:.4f}{st}$"
coef_row += r" \\" + "\n"
latex += coef_row

# SE行
se_row = ""
for k, _ in col_labels:
    r = dml_results[k]
    se_row += f" & $({r['se']:.4f})$"
se_row += r" \\" + "\n"
latex += se_row

latex += r"\midrule" + "\n"

# 底部统计信息
n_row = "$N$"
for k, _ in col_labels:
    n_row += f" & {dml_results[k]['N']:,}"
n_row += r" \\" + "\n"
latex += n_row

ctrl_row = "控制变量"
for k, _ in col_labels:
    ctrl_row += f" & {ctrl_labels[k]}"
ctrl_row += r" \\" + "\n"
latex += ctrl_row

latex += r"""企业FE & Mundlak & Mundlak & Mundlak & Mundlak \\
年份FE & 虚拟变量 & 虚拟变量 & 虚拟变量 & 虚拟变量 \\
ML学习器 & Lasso+RF & Lasso+RF & Lasso+RF & Lasso+RF \\
\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item 注：DML-CRE (Chernozhukov et al., 2018)。CRE = 关联随机效应 (Mundlak, 1978)，
用企业层面控制变量均值控制个体固定效应，加入年份虚拟变量控制时间效应。
学习器：E[Y|X]使用Lasso (LassoCV, 5折)，E[D|X]使用随机森林 (500棵, max\_depth=6)。
交叉拟合：5折×3次重复。标准误为White HC异方差稳健标准误。
***、**、* 分别表示在1\%、5\%、10\%水平显著。
\end{tablenotes}
\end{threeparttable}
\end{table}
"""

with open(f"{OUT}/dml_main_results.tex", "w", encoding="utf-8") as f:
    f.write(latex)
print(f"LaTeX: {OUT}/dml_main_results.tex")

# ================================================================
# 10. LaTeX表格 — OLS vs DML 对比 (稳健性)
# ================================================================
latex_comp = r"""\begin{table}[htbp]
\centering
\caption{稳健性检验：OLS与DML估计对比}
\label{tab:ols_vs_dml}
\begin{threeparttable}
\begin{tabular}{lcccc}
\toprule
 & \multicolumn{2}{c}{传统OLS} & \multicolumn{2}{c}{DML-CRE} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}
 & (1) OLS-15X & (2) OLS-28X & (3) DML-15X & (4) DML-28X \\
\midrule
"""

# 系数和SE
for method_label, data_dict, p_val in [
    ("DU\\_kw", ols_results.get("OLS-15X", {}), None),
    ("DU\\_kw", ols_results.get("OLS-28X", {}), None),
    ("DU\\_kw", dml_results.get("DML-15X",  {}), dml_results.get("DML-15X", {}).get("p")),
    ("DU\\_kw", dml_results.get("DML-28X",  {}), dml_results.get("DML-28X", {}).get("p")),
]:
    pass  # We'll build column by column below

all_specs_comp = [
    ("OLS-15X", ols_results.get("OLS-15X", {}), None),
    ("OLS-28X", ols_results.get("OLS-28X", {}), None),
    ("DML-15X", dml_results.get("DML-15X",  {}), dml_results.get("DML-15X", {}).get("p")),
    ("DML-28X", dml_results.get("DML-28X",  {}), dml_results.get("DML-28X", {}).get("p")),
]

coef_line = "DU\\_kw"
se_line   = ""
n_line    = "$N$"
for k, d, p in all_specs_comp:
    if d:
        c  = d.get("coef", float("nan"))
        s  = d.get("se",   float("nan"))
        n  = d.get("N",    0)
        st = stars(p)
        coef_line += f" & ${c:.4f}{st}$"
        se_line   += f" & $({s:.4f})$"
        n_line    += f" & {n:,}"
    else:
        coef_line += " & --"
        se_line   += " & --"
        n_line    += " & --"

latex_comp += coef_line + r" \\" + "\n"
latex_comp += se_line   + r" \\" + "\n"
latex_comp += r"\midrule" + "\n"
latex_comp += n_line    + r" \\" + "\n"
ctrl_row2 = "控制变量 & 15个 & 28个 & 15个+CRE & 28个+CRE" + r" \\" + "\n"
latex_comp += ctrl_row2
latex_comp += r"""企业FE & \checkmark & \checkmark & Mundlak & Mundlak \\
年份FE & \checkmark & \checkmark & 虚拟变量 & 虚拟变量 \\
\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item 注：OLS列使用双向固定效应 (企业+年份)，行业×年份聚类标准误。
DML列使用CRE-Mundlak装置，White HC稳健标准误。
***、**、* 分别表示在1\%、5\%、10\%水平显著。
\end{tablenotes}
\end{threeparttable}
\end{table}
"""

with open(f"{OUT}/ols_vs_dml_comparison.tex", "w", encoding="utf-8") as f:
    f.write(latex_comp)
print(f"对比LaTeX: {OUT}/ols_vs_dml_comparison.tex")

print("\nDone!")
