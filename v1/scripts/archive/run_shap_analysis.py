"""
SHAP分析 v2: 数据子维度的增量解释与稳健性

改进点:
1) 时点对齐: 使用t-1年报子维度解释t年PriceDelay
2) 目标残差化: 先用控制变量 + Firm/Year FE解释PriceDelay，再建模残差
3) 评估方式: walk-forward逐年外样本验证
4) 反事实设定: 仅做分位数提升与+1σ，并做P1-P99截断
5) 稳定性: bootstrap检验维度排序稳定性
"""

import os
import warnings

import numpy as np
import pandas as pd
import pyfixest as pf
from sklearn.metrics import mean_squared_error, r2_score

warnings.filterwarnings("ignore")

OUT_DIR = "/Users/mac/computerscience/15会计研究/data_parquet"
RES_DIR = "/Users/mac/computerscience/15会计研究/results"
os.makedirs(RES_DIR, exist_ok=True)

SEED = 42
MIN_TRAIN_YEARS = 5
MIN_TRAIN_OBS = 5000
MIN_TEST_OBS = 300
BOOTSTRAP_N = 200
MAX_SHAP_EVAL = 2500


print("=" * 60)
print("SHAP分析 v2: 数据要素子维度增量解释")
print("=" * 60)

# ================================================================
# 1. 数据准备
# ================================================================
panel = pd.read_parquet(f"{OUT_DIR}/panel.parquet")
ar_feat = pd.read_parquet(f"{OUT_DIR}/annual_report_features.parquet")

panel = panel.merge(
    ar_feat[
        [
            "Stkcd",
            "year",
            "kw_data_stock",
            "kw_data_dev",
            "kw_data_app",
            "kw_data_value",
            "kw_data_gov",
            "total_chars",
        ]
    ],
    on=["Stkcd", "year"],
    how="left",
)

# 行业与ST筛选
fi = pd.read_parquet(
    f"{OUT_DIR}/firm_info.parquet",
    columns=["Symbol", "EndDate", "IndustryCodeC", "LISTINGSTATE"],
)
fi = fi.rename(columns={"Symbol": "Stkcd"})
fi["EndDate"] = pd.to_datetime(fi["EndDate"])
fi["year"] = fi["EndDate"].dt.year
fi = fi.sort_values(["Stkcd", "year", "EndDate"]).drop_duplicates(
    subset=["Stkcd", "year"], keep="last"
)
panel = panel.merge(
    fi[["Stkcd", "year", "IndustryCodeC", "LISTINGSTATE"]],
    on=["Stkcd", "year"],
    how="left",
)

mask_fin = panel["IndustryCodeC"].str.startswith("J", na=False)
mask_st = panel["LISTINGSTATE"].isin(["ST", "*ST"])
mask_new = panel["Age"] <= 0
panel = panel[~mask_fin & ~mask_st & ~mask_new].copy()

# 子维度标准化（每万字）
for dim in [
    "kw_data_stock",
    "kw_data_dev",
    "kw_data_app",
    "kw_data_value",
    "kw_data_gov",
]:
    panel[f"{dim}_p10k"] = panel[dim] / panel["total_chars"].clip(lower=1) * 10000

data_dims = [
    "kw_data_stock_p10k",
    "kw_data_dev_p10k",
    "kw_data_app_p10k",
    "kw_data_value_p10k",
    "kw_data_gov_p10k",
]

# 时点对齐: 用t-1年年报维度解释t年PriceDelay
panel = panel.sort_values(["Stkcd", "year"])
for dim in data_dims:
    panel[dim] = panel.groupby("Stkcd")[dim].shift(1)

controls = [
    "Size",
    "Lev",
    "ROA",
    "TobinQ",
    "Age",
    "Growth",
    "BoardSize",
    "IndepRatio",
    "Dual",
    "Top1Share",
    "SOE",
    "InstHold",
    "Amihud",
    "Analyst",
    "AuditType",
]

base_cols = ["Stkcd", "year", "PriceDelay"] + controls + data_dims
df = panel[base_cols].dropna().copy()

print(f"  可用样本: {len(df):,} obs, {df.Stkcd.nunique():,} firms")
print(f"  年份范围: {int(df.year.min())}-{int(df.year.max())}")

# ================================================================
# 2. 残差化目标变量 (控制变量 + Firm/Year FE)
# ================================================================
print("\n--- 残差化 PriceDelay (controls + Firm/Year FE) ---")

df_fe = df.copy().reset_index(drop=True)
df_fe["Stkcd_fe"] = df_fe["Stkcd"].astype(str)
df_fe["year_fe"] = df_fe["year"].astype(str)
ctrl_str = " + ".join(controls)

m_fe = pf.feols(
    f"PriceDelay ~ {ctrl_str} | Stkcd_fe + year_fe",
    data=df_fe,
)

# feols会自动剔除singleton FE，后续样本以模型实际样本为准
df_fe = m_fe._data.copy().reset_index(drop=True)
df_fe["delay_resid"] = np.asarray(m_fe.resid())
print(
    f"  残差均值={df_fe['delay_resid'].mean():.6f}, "
    f"标准差={df_fe['delay_resid'].std():.6f}"
)

# 避免pyfixest与lightgbm/shap的OpenMP运行时冲突，延迟导入
import lightgbm as lgb
import matplotlib
import matplotlib.pyplot as plt
import shap

matplotlib.use("Agg")


def build_model(seed: int = SEED):
    return lgb.LGBMRegressor(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.2,
        reg_lambda=0.2,
        random_state=seed,
        verbose=-1,
        n_jobs=-1,
    )

# ================================================================
# 3. Walk-forward 外样本验证
# ================================================================
print("\n--- Walk-forward外样本验证 ---")

years = sorted(df_fe["year"].unique().tolist())
wf_rows = []

for test_year in years:
    train_years = [y for y in years if y < test_year]
    if len(train_years) < MIN_TRAIN_YEARS:
        continue

    train_df = df_fe[df_fe["year"].isin(train_years)]
    test_df = df_fe[df_fe["year"] == test_year]
    if len(train_df) < MIN_TRAIN_OBS or len(test_df) < MIN_TEST_OBS:
        continue

    model = build_model()
    model.fit(train_df[data_dims], train_df["delay_resid"])
    pred = model.predict(test_df[data_dims])
    r2 = r2_score(test_df["delay_resid"], pred)
    rmse = np.sqrt(mean_squared_error(test_df["delay_resid"], pred))

    wf_rows.append(
        {
            "test_year": int(test_year),
            "train_n": int(len(train_df)),
            "test_n": int(len(test_df)),
            "r2": float(r2),
            "rmse": float(rmse),
        }
    )
    print(
        f"  test_year={int(test_year)}: "
        f"R2={r2:.4f}, RMSE={rmse:.6f}, "
        f"train={len(train_df):,}, test={len(test_df):,}"
    )

wf_df = pd.DataFrame(wf_rows)
if wf_df.empty:
    raise RuntimeError("Walk-forward样本不足，无法继续。")

wf_df.to_csv(f"{RES_DIR}/shap_oos_metrics.csv", index=False)
weighted_r2 = np.average(wf_df["r2"], weights=wf_df["test_n"])
weighted_rmse = np.average(wf_df["rmse"], weights=wf_df["test_n"])
print(
    f"  加权平均: R2={weighted_r2:.4f}, RMSE={weighted_rmse:.6f} "
    f"(已保存 shap_oos_metrics.csv)"
)

# ================================================================
# 4. 最终训练/测试切分 (用于SHAP展示和反事实)
# ================================================================
holdout_years = sorted(wf_df["test_year"].unique())[-2:]
train_final = df_fe[~df_fe["year"].isin(holdout_years)].copy()
test_final = df_fe[df_fe["year"].isin(holdout_years)].copy()

print("\n--- 最终模型 (用于SHAP展示) ---")
print(f"  holdout years: {holdout_years}")
print(f"  train={len(train_final):,}, test={len(test_final):,}")

final_model = build_model(seed=SEED)
final_model.fit(train_final[data_dims], train_final["delay_resid"])

pred_train = final_model.predict(train_final[data_dims])
pred_test = final_model.predict(test_final[data_dims])
r2_train = r2_score(train_final["delay_resid"], pred_train)
r2_test = r2_score(test_final["delay_resid"], pred_test)
rmse_test = np.sqrt(mean_squared_error(test_final["delay_resid"], pred_test))
print(f"  Train R2={r2_train:.4f}")
print(f"  Test  R2={r2_test:.4f}")
print(f"  Test  RMSE={rmse_test:.6f}")

# ================================================================
# 5. SHAP重要性（外样本）
# ================================================================
print("\n--- 计算SHAP重要性 ---")

if len(test_final) > MAX_SHAP_EVAL:
    eval_df = test_final.sample(n=MAX_SHAP_EVAL, random_state=SEED)
else:
    eval_df = test_final.copy()

X_eval = eval_df[data_dims].values
explainer = shap.TreeExplainer(final_model)
shap_values = explainer.shap_values(X_eval)

label_map = {
    "kw_data_stock_p10k": "数据存量",
    "kw_data_dev_p10k": "数据开发能力",
    "kw_data_app_p10k": "数据驱动应用",
    "kw_data_value_p10k": "数据价值变现",
    "kw_data_gov_p10k": "数据治理",
}

importance = pd.DataFrame(
    {
        "feature": data_dims,
        "mean_abs_shap": np.abs(shap_values).mean(axis=0),
        "mean_shap": shap_values.mean(axis=0),
    }
).sort_values("mean_abs_shap", ascending=False)
importance["label"] = importance["feature"].map(label_map)
importance.to_csv(f"{RES_DIR}/shap_importance.csv", index=False)

print(importance[["label", "mean_abs_shap", "mean_shap"]].round(6).to_string(index=False))

total_data_shap = importance["mean_abs_shap"].sum()
for _, row in importance.iterrows():
    pct = row["mean_abs_shap"] / total_data_shap * 100
    direction = "降低delay" if row["mean_shap"] < 0 else "增加delay"
    print(
        f"  {row['label']}: |SHAP|={row['mean_abs_shap']:.6f} "
        f"({pct:.1f}%), {direction}"
    )

# SHAP图
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
labels = [label_map[d] for d in data_dims]

shap.summary_plot(
    shap_values,
    X_eval,
    feature_names=labels,
    plot_type="bar",
    show=False,
    max_display=5,
)
plt.tight_layout()
plt.savefig(f"{RES_DIR}/shap_summary_bar.png", dpi=150, bbox_inches="tight")
plt.close()

shap.summary_plot(
    shap_values,
    X_eval,
    feature_names=labels,
    show=False,
    max_display=5,
)
plt.tight_layout()
plt.savefig(f"{RES_DIR}/shap_summary_dot.png", dpi=150, bbox_inches="tight")
plt.close()

# ================================================================
# 6. 排序稳定性 (Bootstrap)
# ================================================================
print("\n--- Bootstrap排序稳定性 ---")

rng = np.random.default_rng(SEED)
rank_rows = []
train_len = len(train_final)

for b in range(BOOTSTRAP_N):
    boot_idx = rng.integers(0, train_len, train_len)
    boot_df = train_final.iloc[boot_idx]
    m_b = build_model(seed=SEED + b + 1)
    m_b.fit(boot_df[data_dims], boot_df["delay_resid"])

    shap_b = shap.TreeExplainer(m_b).shap_values(X_eval)
    imp_b = np.abs(shap_b).mean(axis=0)
    order = np.argsort(-imp_b)
    ranks = np.empty(len(data_dims), dtype=int)
    ranks[order] = np.arange(1, len(data_dims) + 1)

    for i, dim in enumerate(data_dims):
        rank_rows.append(
            {
                "bootstrap_id": b + 1,
                "feature": dim,
                "label": label_map[dim],
                "rank": int(ranks[i]),
                "mean_abs_shap": float(imp_b[i]),
            }
        )

    if (b + 1) % 25 == 0:
        print(f"  bootstrap {b + 1}/{BOOTSTRAP_N}")

rank_df = pd.DataFrame(rank_rows)
rank_df.to_csv(f"{RES_DIR}/shap_bootstrap_ranks.csv", index=False)

rank_summary = (
    rank_df.groupby(["feature", "label"])
    .agg(
        mean_rank=("rank", "mean"),
        std_rank=("rank", "std"),
        top1_share=("rank", lambda x: (x == 1).mean()),
        top2_share=("rank", lambda x: (x <= 2).mean()),
        mean_abs_shap=("mean_abs_shap", "mean"),
    )
    .reset_index()
    .sort_values("mean_rank")
)
rank_summary.to_csv(f"{RES_DIR}/shap_dim_rank_stability.csv", index=False)
print(rank_summary.round(4).to_string(index=False))

# ================================================================
# 7. 反事实分析（现实边界内）
# ================================================================
print("\n--- 反事实分析 (现实边界) ---")

X_base = test_final[data_dims].values.copy()
base_pred = final_model.predict(X_base)
base_delay_mean = test_final["PriceDelay"].mean()

q1 = train_final[data_dims].quantile(0.01)
q99 = train_final[data_dims].quantile(0.99)
q25 = train_final[data_dims].quantile(0.25)
q75 = train_final[data_dims].quantile(0.75)
stds = train_final[data_dims].std()

scenarios = []

# 情景1: 低位企业由P25提升至P75
X_cf1 = X_base.copy()
for i, dim in enumerate(data_dims):
    mask = X_cf1[:, i] < q25[dim]
    X_cf1[mask, i] = q75[dim]
    X_cf1[:, i] = np.clip(X_cf1[:, i], q1[dim], q99[dim])
pred_cf1 = final_model.predict(X_cf1)
delta1 = float((pred_cf1 - base_pred).mean())
scenarios.append(
    {
        "scenario": "P25_to_P75",
        "delta_delay_component": delta1,
        "pct_of_delay_mean": delta1 / base_delay_mean * 100,
    }
)

# 情景2: 各维度 +1σ，并截断到P99
X_cf2 = X_base.copy()
for i, dim in enumerate(data_dims):
    X_cf2[:, i] = X_cf2[:, i] + stds[dim]
    X_cf2[:, i] = np.clip(X_cf2[:, i], q1[dim], q99[dim])
pred_cf2 = final_model.predict(X_cf2)
delta2 = float((pred_cf2 - base_pred).mean())
scenarios.append(
    {
        "scenario": "plus_1sd_capped",
        "delta_delay_component": delta2,
        "pct_of_delay_mean": delta2 / base_delay_mean * 100,
    }
)

cf_df = pd.DataFrame(scenarios)
cf_df.to_csv(f"{RES_DIR}/shap_counterfactual.csv", index=False)
print(cf_df.round(6).to_string(index=False))

print("\n已保存:")
print(f"  - {RES_DIR}/shap_oos_metrics.csv")
print(f"  - {RES_DIR}/shap_importance.csv")
print(f"  - {RES_DIR}/shap_bootstrap_ranks.csv")
print(f"  - {RES_DIR}/shap_dim_rank_stability.csv")
print(f"  - {RES_DIR}/shap_counterfactual.csv")
print("完成！")
