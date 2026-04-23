"""
四机制变量相关矩阵 + VIF sanity check
变量：Amihud / lnAmihud / CashFlowVol / SCConc / Comparability_med
数据：reg_sample_v18.dta + accounting_comparability.csv
2026-04-21
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")

ROOT = "/Users/mac/computerscience/0做完了/15会计研究"
OUT = f"{ROOT}/v2/results/mechanism_corr_vif.txt"

lines = []
def p(s=""):
    print(s)
    lines.append(str(s))

# ------------------------------------------------------------
# 1. Load
# ------------------------------------------------------------
df = pd.read_stata(f"{ROOT}/v1/data_stata/reg_sample_v18.dta", convert_categoricals=False)
cmp = pd.read_csv(f"{ROOT}/v1/data_parquet/accounting_comparability.csv")
df = df.merge(cmp[["Stkcd", "year", "Comparability_med"]], on=["Stkcd", "year"], how="left")

df["lnAmihud"] = np.log1p(df["Amihud"].clip(lower=0))

mech = ["Amihud", "lnAmihud", "CashFlowVol", "SCConc", "Comparability_med"]

# winsorize 1/99
for v in mech:
    lo, hi = df[v].quantile([0.01, 0.99])
    df[v] = df[v].clip(lo, hi)

p("=" * 70)
p("四机制变量 sanity check (N and missing)")
p("=" * 70)
p(df[mech].describe().round(5).to_string())
p()
p("Missing count:")
p(df[mech].isna().sum().to_string())
p(f"\n样本 N={len(df)}")

# ------------------------------------------------------------
# 2. Pearson correlation
# ------------------------------------------------------------
p("\n" + "=" * 70)
p("(2) Pearson 相关矩阵（全样本，winsorized）")
p("=" * 70)
sub = df[mech].dropna()
p(f"Non-missing pairwise sample: {len(sub)}")
corr = sub.corr(method="pearson")
p(corr.round(4).to_string())
p("\nPearson p-values:")
pv = pd.DataFrame(np.ones_like(corr), index=mech, columns=mech)
for i, a in enumerate(mech):
    for j, b in enumerate(mech):
        if i != j:
            r, p_ = pearsonr(sub[a], sub[b])
            pv.loc[a, b] = p_
p(pv.round(4).to_string())

# ------------------------------------------------------------
# 3. Spearman
# ------------------------------------------------------------
p("\n" + "=" * 70)
p("(3) Spearman 秩相关矩阵")
p("=" * 70)
rho, pval = spearmanr(sub[mech].values)
p("Rho:")
p(pd.DataFrame(rho, index=mech, columns=mech).round(4).to_string())
p("\nP-values:")
p(pd.DataFrame(pval, index=mech, columns=mech).round(4).to_string())

# ------------------------------------------------------------
# 4. Within-firm correlation (demeaned by Stkcd)
# ------------------------------------------------------------
p("\n" + "=" * 70)
p("(4) Within-firm 相关（demean by Stkcd）")
p("=" * 70)
dm = df[["Stkcd"] + mech].copy()
for v in mech:
    dm[v] = dm[v] - dm.groupby("Stkcd")[v].transform("mean")
dmc = dm[mech].dropna()
p(f"Within-firm sample: {len(dmc)}")
p(dmc.corr().round(4).to_string())

# ------------------------------------------------------------
# 5. VIF with controls
# ------------------------------------------------------------
p("\n" + "=" * 70)
p("(5) VIF：四机制 + 11 控制变量（Delay 作因变量，仅算 VIF）")
p("=" * 70)
ctrls = ["Size", "Lev", "ROA", "TobinQ", "Age", "Growth",
         "IndepRatio", "Dual", "Top1Share", "SOE", "CFO"]
# 用 lnAmihud 代替 Amihud
x_vars = ["lnAmihud", "CashFlowVol", "SCConc", "Comparability_med"] + ctrls
sub5 = df[x_vars + ["PriceDelay"]].dropna()
X = sm.add_constant(sub5[x_vars])
vif_data = pd.DataFrame({
    "var": X.columns,
    "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
})
p(f"VIF sample N={len(sub5)}")
p(vif_data.round(3).to_string(index=False))

# ------------------------------------------------------------
# 6. VIF mechanisms only
# ------------------------------------------------------------
p("\n" + "=" * 70)
p("(6) VIF：仅四机制（无控制变量）")
p("=" * 70)
x6 = ["lnAmihud", "CashFlowVol", "SCConc", "Comparability_med"]
sub6 = df[x6].dropna()
X6 = sm.add_constant(sub6)
vif6 = pd.DataFrame({
    "var": X6.columns,
    "VIF": [variance_inflation_factor(X6.values, i) for i in range(X6.shape[1])]
})
p(f"N={len(sub6)}")
p(vif6.round(3).to_string(index=False))

# ------------------------------------------------------------
# 7. Pairwise PriceDelay regression
# ------------------------------------------------------------
p("\n" + "=" * 70)
p("(7) Pairwise PriceDelay ~ 两两机制 + 控制（无 FE，快速诊断）")
p("=" * 70)
mechs_pair = ["lnAmihud", "CashFlowVol", "SCConc", "Comparability_med"]
sub7_base = df[["PriceDelay"] + mechs_pair + ctrls].dropna()
for i, a in enumerate(mechs_pair):
    for b in mechs_pair[i+1:]:
        X = sm.add_constant(sub7_base[[a, b] + ctrls])
        y = sub7_base["PriceDelay"]
        m = sm.OLS(y, X).fit(cov_type="HC1")
        p(f"\n--- {a} + {b} ---  N={len(sub7_base)}, R²={m.rsquared:.4f}")
        for v in [a, b]:
            p(f"  {v}: β={m.params[v]:.6f}  SE={m.bse[v]:.6f}  t={m.tvalues[v]:.2f}  p={m.pvalues[v]:.4f}")

# ------------------------------------------------------------
# 8. Joint four mechanisms
# ------------------------------------------------------------
p("\n" + "=" * 70)
p("(8) Joint 四机制同时进 PriceDelay（无 FE）")
p("=" * 70)
X8 = sm.add_constant(sub7_base[mechs_pair + ctrls])
y8 = sub7_base["PriceDelay"]
m8 = sm.OLS(y8, X8).fit(cov_type="HC1")
p(f"N={len(sub7_base)}, R²={m8.rsquared:.4f}")
for v in mechs_pair:
    p(f"  {v}: β={m8.params[v]:.6f}  SE={m8.bse[v]:.6f}  t={m8.tvalues[v]:.2f}  p={m8.pvalues[v]:.4f}")

# ------------------------------------------------------------
# save
# ------------------------------------------------------------
with open(OUT, "w") as f:
    f.write("\n".join(lines))
p(f"\n[done] 结果写入 {OUT}")
