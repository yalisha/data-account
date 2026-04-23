"""
DML主结果 v15: 4规格，全部27控制变量
  (1) Y=PriceDelay, D=DU_kw,     X=27 CRE  ← 主规格
  (2) Y=PriceDelay, D=DU_kw_ln,  X=27 CRE  ← 替换处理变量
  (3) Y=PriceDelay, D=DU_sub_ln, X=27 CRE  ← 实质利用
  (4) Y=SYNCH,      D=DU_kw,     X=27 CRE  ← 替换被解释变量
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
OUT  = f"{BASE}/results/v15_tables"
os.makedirs(OUT, exist_ok=True)

# ================================================================
# 1. 加载数据
# ================================================================
print("1. 加载数据")

panel = pd.read_parquet(f"{DATA}/panel_dml.parquet")
feat  = pd.read_parquet(f"{DATA}/annual_report_features.parquet",
                         columns=["Stkcd", "year", "kw_per10k", "kw_total", "substantive_count"])
feat["DU_kw"]     = feat["kw_per10k"]
feat["DU_kw_ln"]  = np.log1p(feat["kw_total"])
feat["DU_sub_ln"] = np.log1p(feat["substantive_count"])

synch = pd.read_parquet(f"{DATA}/price_synchronicity.parquet",
                         columns=["Stkcd", "year", "SYNCH"])

df = panel.merge(feat[["Stkcd", "year", "DU_kw", "DU_kw_ln", "DU_sub_ln"]], on=["Stkcd", "year"], how="left")
df = df.merge(synch, on=["Stkcd", "year"], how="left")

# Construct Inv and Hhi
bs = pd.read_parquet(f"{DATA}/balance_sheet.parquet")
bs['EndDate'] = pd.to_datetime(bs['Accper']); bs['year'] = bs['EndDate'].dt.year
bs = bs[bs['Typrep'] == 'A'].sort_values(['Stkcd','year','EndDate']).drop_duplicates(['Stkcd','year'], keep='last')
bs['Inv'] = bs['A001218000'] / bs['A001000000']
df = df.merge(bs[['Stkcd','year','Inv']], on=['Stkcd','year'], how='left')

fi = pd.read_parquet(f"{DATA}/firm_info.parquet", columns=['Symbol','EndDate','IndustryCodeC'])
fi = fi.rename(columns={'Symbol':'Stkcd'}); fi['EndDate'] = pd.to_datetime(fi['EndDate']); fi['year'] = fi['EndDate'].dt.year
fi = fi.sort_values(['Stkcd','year','EndDate']).drop_duplicates(['Stkcd','year'], keep='last')
df = df.merge(fi[['Stkcd','year','IndustryCodeC']], on=['Stkcd','year'], how='left')
df['Ind2'] = df['IndustryCodeC'].str[:3]

inc = pd.read_parquet(f"{DATA}/income_stmt.parquet")
inc['EndDate'] = pd.to_datetime(inc['Accper']); inc['year'] = inc['EndDate'].dt.year
inc = inc[inc['Typrep'] == 'A'].sort_values(['Stkcd','year','EndDate']).drop_duplicates(['Stkcd','year'], keep='last')
inc_ind = df[['Stkcd','year','Ind2']].merge(inc[['Stkcd','year','B001101000']], on=['Stkcd','year'], how='left')
hhi = inc_ind.groupby(['Ind2','year']).apply(
    lambda g: ((g['B001101000'] / g['B001101000'].sum()) ** 2).sum() if g['B001101000'].sum() > 0 else np.nan
).reset_index(name='Hhi')
df = df.merge(hhi, on=['Ind2','year'], how='left')
print(f"合并后: {len(df):,} obs (Inv: {df['Inv'].notna().sum()}, Hhi: {df['Hhi'].notna().sum()})")

# ================================================================
# 2. 11个控制变量 (移除机制渠道变量+冗余变量)
# ================================================================
controls_27 = ["Size", "Lev", "ROA", "TobinQ", "Age", "Growth",
               "IndepRatio", "Dual", "Top1Share", "SOE", "CFO", "Inv", "Hhi"]

print(f"控制变量: {len(controls_27)}个")

# ================================================================
# 3. DML辅助函数
# ================================================================
def make_learners():
    ml_l = Pipeline([("scaler", StandardScaler()),
                     ("model", LassoCV(cv=5, max_iter=3000, n_jobs=-1))])
    ml_m = RandomForestRegressor(n_estimators=500, max_depth=6,
                                  min_samples_leaf=10, n_jobs=-1, random_state=42)
    return ml_l, ml_m


def build_cre(df_in, y_var, d_var, ctrl_list, id_col="Stkcd", year_col="year"):
    req = [y_var, d_var] + ctrl_list + [id_col, year_col]
    df_c = df_in[req].dropna().copy()

    for v in [y_var, d_var] + ctrl_list:
        lo = df_c[v].quantile(0.01)
        hi = df_c[v].quantile(0.99)
        df_c[v] = df_c[v].clip(lo, hi)

    fm_cols = []
    for v in ctrl_list:
        col = f"fm_{v}"
        df_c[col] = df_c.groupby(id_col)[v].transform("mean")
        fm_cols.append(col)

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
        obj_dml_data=dml_data, ml_l=ml_l, ml_m=ml_m,
        n_folds=5, n_rep=3, score="partialling out",
    )
    t0 = time.time()
    plr.fit()
    elapsed = time.time() - t0

    ci  = plr.confint(level=0.95)
    res = {
        "coef": float(plr.coef[0]),
        "se":   float(plr.se[0]),
        "t":    float(plr.t_stat[0]),
        "p":    float(plr.pval[0]),
        "ci_lo": float(ci.iloc[0, 0]),
        "ci_hi": float(ci.iloc[0, 1]),
    }
    print(f"  {label}: coef={res['coef']:.4f}, se={res['se']:.4f}, "
          f"t={res['t']:.2f}, p={res['p']:.4f}  [{elapsed:.0f}s]")
    return res

# ================================================================
# 4. 运行4个DML规格 (全部27控制变量)
# ================================================================
print("\n4. DML-CRE: 4规格 (v15, 27控制变量)")

specs = [
    ("DML-27X",       "PriceDelay", "DU_kw",     controls_27),
    ("DML-27X-ln",    "PriceDelay", "DU_kw_ln",  controls_27),
    ("DML-27X-sub",   "PriceDelay", "DU_sub_ln", controls_27),
    ("DML-27X-SYNCH", "SYNCH",      "DU_kw",     controls_27),
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
# 5. 输出
# ================================================================
print("\n5. 结果汇总")
for k, v in dml_results.items():
    print(f"  {k:20s} coef={v['coef']:+.4f}  se={v['se']:.4f}  "
          f"t={v['t']:6.2f}  p={v['p']:.4f}  N={v['N']:,}")

# Save CSV (same format as before, for generate_tables_v13.py to read)
rows = []
for k, v in dml_results.items():
    rows.append({"Spec": k, **v})
pd.DataFrame(rows).to_csv(f"{OUT}/dml_main_results_v15.csv", index=False)
print(f"\nCSV: {OUT}/dml_main_results_v15.csv")

print("\nDone!")
