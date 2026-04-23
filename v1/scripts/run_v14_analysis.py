"""
Run v14 construct-boundary, wash-gap, and policy-cutpoint analyses.

Outputs:
  - results/v14/v14_analysis_sample.parquet
  - results/v14/construct_boundary_results.json
  - results/v14/washgap_results.json
  - results/v14/heterogeneity_results.json
  - results/v14/v14_summary.md
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyfixest as pf

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parents[1]
DATA_STATA = BASE / "data_stata" / "reg_sample_v18.dta"
AR_FEAT = BASE / "data_parquet" / "annual_report_features.parquet"
PLACEBO = BASE / "data_parquet" / "placebo_features.parquet"
OUT_DIR = BASE / "results" / "v14"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CONTROLS = [
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

BUZZWORDS = [
    "数字化转型",
    "数字化",
    "智能化",
    "信息化",
    "人工智能",
    "大数据",
    "智能制造",
    "智慧城市",
]


def winsorize(s: pd.Series, lo: float = 0.01, hi: float = 0.99) -> pd.Series:
    q = s.quantile([lo, hi])
    return s.clip(q.iloc[0], q.iloc[1])


def lag_if_consecutive(df: pd.DataFrame, col: str) -> pd.Series:
    prev_year = df.groupby("Stkcd")["year"].shift(1)
    prev_val = df.groupby("Stkcd")[col].shift(1)
    return prev_val.where(df["year"].eq(prev_year + 1))


def add_fixef_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Stkcd_str"] = out["Stkcd"].astype(int).astype(str)
    out["year_str"] = out["year"].astype(int).astype(str)
    return out


def fit_feols(df: pd.DataFrame, y: str, xvars: list[str], cluster: str = "IndYear"):
    sample = df.dropna(subset=[y, *xvars, *CONTROLS, "Stkcd", "year", cluster]).copy()
    sample = add_fixef_cols(sample)
    rhs = " + ".join(xvars + CONTROLS)
    model = pf.feols(
        f"{y} ~ {rhs} | Stkcd_str + year_str",
        data=sample,
        vcov={"CRV1": cluster},
    )
    return sample, model


def coef_row(model, var: str) -> dict:
    return {
        "coef": float(model.coef()[var]),
        "se": float(model.se()[var]),
        "t": float(model.tstat()[var]),
        "p": float(model.pvalue()[var]),
        "n": int(model._N),
        "r2": float(model._r2),
        "r2_within": float(model._r2_within),
        "adj_r2_within": float(model._adj_r2_within),
    }


def interaction_group_results(
    df: pd.DataFrame,
    treat: str,
    group_col: str,
    group_hi_label: str,
    group_lo_label: str,
    sample_mask: pd.Series | None = None,
) -> dict:
    use = df.copy()
    if sample_mask is not None:
        use = use.loc[sample_mask].copy()

    hi_sample, hi_model = fit_feols(use.loc[use[group_col] == 1], "PriceDelay", [treat])
    lo_sample, lo_model = fit_feols(use.loc[use[group_col] == 0], "PriceDelay", [treat])

    inter = use.dropna(
        subset=["PriceDelay", treat, group_col, *CONTROLS, "Stkcd", "year", "IndYear"]
    ).copy()
    inter = add_fixef_cols(inter)
    model = pf.feols(
        f"PriceDelay ~ {treat} + {treat}:{group_col} + {' + '.join(CONTROLS)} | Stkcd_str + year_str",
        data=inter,
        vcov={"CRV1": "IndYear"},
    )
    inter_term = f"{treat}:{group_col}"

    return {
        "group_high": group_hi_label,
        "group_low": group_lo_label,
        "high": coef_row(hi_model, treat),
        "low": coef_row(lo_model, treat),
        "interaction": coef_row(model, inter_term),
        "sample_n": int(len(inter)),
    }


def fmt_signed(x: float, digits: int = 4) -> str:
    return f"{x:+.{digits}f}"


print("Loading v18 sample...")
reg = pd.read_stata(DATA_STATA)
reg["Stkcd"] = reg["Stkcd"].astype(int)
reg["year"] = reg["year"].astype(int)

feat = pd.read_parquet(
    AR_FEAT,
    columns=[
        "Stkcd",
        "year",
        "kw_total",
        "total_chars",
        "mda_chars",
        "substantive_ratio",
        "substantive_count",
        "mention_count",
    ],
)
placebo = pd.read_parquet(PLACEBO)

df = reg.merge(
    feat,
    on=["Stkcd", "year", "kw_total"],
    how="left",
    validate="1:1",
)
df = df.merge(placebo[["Stkcd", "year", "buzzword_count"]], on=["Stkcd", "year"], how="left")
df["buzzword_count"] = df["buzzword_count"].fillna(0)

print("Constructing v14 variables...")
df["GenericNarr"] = df["buzzword_count"] / df["total_chars"].clip(lower=1) * 10000
df["DU_kw_strict"] = (
    (df["kw_total"] - df["buzzword_count"]).clip(lower=0) / df["total_chars"].clip(lower=1) * 10000
)
df["DU_llm_lenstd"] = np.log1p(df["DU_kw"].clip(lower=0)) * (df["llm_score"] / 3.0)
df["ln_total_chars"] = np.log1p(df["total_chars"])
df["ln_mda_chars"] = np.log1p(df["mda_chars"])

for col in ["GenericNarr", "DU_kw_strict", "DU_llm_lenstd", "ln_total_chars", "ln_mda_chars"]:
    if df[col].notna().any():
        df[col] = winsorize(df[col])

df["DU_kw_z"] = (df["DU_kw"] - df["DU_kw"].mean()) / df["DU_kw"].std(ddof=0)
df["DU_llm_lenstd_z"] = (
    (df["DU_llm_lenstd"] - df["DU_llm_lenstd"].mean()) / df["DU_llm_lenstd"].std(ddof=0)
)
df["WashGap"] = winsorize(df["DU_kw_z"] - df["DU_llm_lenstd_z"])

du_rank = df.groupby("year")["DU_kw"].rank(pct=True, method="average")
llm_rank = df.groupby("year")["DU_llm_lenstd"].rank(pct=True, method="average")
df["BroadShallow"] = ((du_rank >= 2 / 3) & (llm_rank <= 1 / 3)).astype(int)

for col in [
    "DU_kw",
    "DU_llm",
    "DU_llm_lenstd",
    "GenericNarr",
    "DU_kw_strict",
    "WashGap",
    "BroadShallow",
    "ln_total_chars",
    "ln_mda_chars",
]:
    df[f"{col}_lag"] = lag_if_consecutive(df.sort_values(["Stkcd", "year"]), col)

df["Post2021"] = (df["year"] >= 2021).astype(int)
df["Post2022"] = (df["year"] >= 2022).astype(int)
df["Post2021_ex2020"] = np.where(df["year"] == 2020, np.nan, (df["year"] >= 2021).astype(float))

df.to_parquet(OUT_DIR / "v14_analysis_sample.parquet", index=False)

print("Running construct-boundary regressions...")
construct = {}

sample, model = fit_feols(df, "PriceDelay", ["DU_kw_lag", "ln_total_chars_lag"])
construct["dukw_total_chars_lag"] = {
    "DU_kw_lag": coef_row(model, "DU_kw_lag"),
    "ln_total_chars_lag": coef_row(model, "ln_total_chars_lag"),
}

sample, model = fit_feols(df, "PriceDelay", ["DU_kw_lag", "ln_mda_chars_lag"])
construct["dukw_mda_chars_lag"] = {
    "DU_kw_lag": coef_row(model, "DU_kw_lag"),
    "ln_mda_chars_lag": coef_row(model, "ln_mda_chars_lag"),
}

sample, model = fit_feols(df, "PriceDelay", ["DU_llm_lag", "ln_total_chars_lag"])
construct["dullm_total_chars_lag"] = {
    "DU_llm_lag": coef_row(model, "DU_llm_lag"),
    "ln_total_chars_lag": coef_row(model, "ln_total_chars_lag"),
}

sample, model = fit_feols(df, "PriceDelay", ["DU_llm_lenstd_lag", "ln_total_chars_lag"])
construct["dullm_lenstd_total_chars_lag"] = {
    "DU_llm_lenstd_lag": coef_row(model, "DU_llm_lenstd_lag"),
    "ln_total_chars_lag": coef_row(model, "ln_total_chars_lag"),
}

joint_sample, joint_model = fit_feols(df, "PriceDelay", ["DU_kw_lag", "GenericNarr_lag"])
joint_full = coef_row(joint_model, "DU_kw_lag")
joint_full["GenericNarr_lag"] = coef_row(joint_model, "GenericNarr_lag")

z_sample = joint_sample.copy()
for col in ["DU_kw_lag", "GenericNarr_lag"]:
    z_sample[f"z_{col}"] = (z_sample[col] - z_sample[col].mean()) / z_sample[col].std(ddof=0)
z_model = pf.feols(
    "PriceDelay ~ z_DU_kw_lag + z_GenericNarr_lag + "
    + " + ".join(CONTROLS)
    + " | Stkcd_str + year_str",
    data=z_sample,
    vcov={"CRV1": "IndYear"},
)

vif_du_model = pf.feols(
    "DU_kw_lag ~ GenericNarr_lag + "
    + " + ".join(CONTROLS)
    + " | Stkcd_str + year_str",
    data=joint_sample,
)
vif_gn_model = pf.feols(
    "GenericNarr_lag ~ DU_kw_lag + "
    + " + ".join(CONTROLS)
    + " | Stkcd_str + year_str",
    data=joint_sample,
)

reduced_du_sample, reduced_du_model = fit_feols(df, "PriceDelay", ["GenericNarr_lag"])
reduced_gn_sample, reduced_gn_model = fit_feols(df, "PriceDelay", ["DU_kw_lag"])

joint_model_du = pf.feols(
    "DU_kw_lag ~ GenericNarr_lag + "
    + " + ".join(CONTROLS)
    + " | Stkcd_str + year_str",
    data=joint_sample,
)
resid_sample = joint_model_du._data.copy()
resid_sample["DU_kw_resid_lag"] = joint_model_du.resid()
resid_model = pf.feols(
    "PriceDelay ~ DU_kw_resid_lag + "
    + " + ".join(CONTROLS)
    + " | Stkcd_str + year_str",
    data=resid_sample,
    vcov={"CRV1": "IndYear"},
)

strict_sample, strict_model = fit_feols(df, "PriceDelay", ["DU_kw_strict_lag"])

construct["joint_lag"] = {
    "DU_kw_lag": coef_row(joint_model, "DU_kw_lag"),
    "GenericNarr_lag": coef_row(joint_model, "GenericNarr_lag"),
    "std_beta_DU_kw_lag": float(z_model.coef()["z_DU_kw_lag"]),
    "std_beta_GenericNarr_lag": float(z_model.coef()["z_GenericNarr_lag"]),
    "vif_DU_kw_lag": float(1.0 / (1.0 - vif_du_model._r2_within)),
    "vif_GenericNarr_lag": float(1.0 / (1.0 - vif_gn_model._r2_within)),
    "delta_within_r2_DU_kw": float(joint_model._r2_within - reduced_du_model._r2_within),
    "delta_within_r2_GenericNarr": float(joint_model._r2_within - reduced_gn_model._r2_within),
}
construct["dukw_resid_lag"] = {"DU_kw_resid_lag": coef_row(resid_model, "DU_kw_resid_lag")}
construct["dukw_strict_lag"] = {"DU_kw_strict_lag": coef_row(strict_model, "DU_kw_strict_lag")}

print("Running wash-gap regressions...")
washgap = {}
sample, model = fit_feols(df, "PriceDelay", ["WashGap_lag"])
washgap["washgap_lag"] = {"WashGap_lag": coef_row(model, "WashGap_lag")}

sample, model = fit_feols(df, "PriceDelay", ["BroadShallow_lag"])
washgap["broadshallow_lag"] = {"BroadShallow_lag": coef_row(model, "BroadShallow_lag")}

sample, model = fit_feols(df, "PriceDelay", ["DU_kw_lag", "DU_llm_lenstd_lag"])
washgap["joint_depth_lag"] = {
    "DU_kw_lag": coef_row(model, "DU_kw_lag"),
    "DU_llm_lenstd_lag": coef_row(model, "DU_llm_lenstd_lag"),
}

print("Running heterogeneity regressions...")
heterogeneity = {}
lag_mask = df["DU_kw_lag"].notna()
heterogeneity["Post2021"] = interaction_group_results(
    df, "DU_kw_lag", "Post2021", "2021-2024", "2011-2020", sample_mask=lag_mask
)
heterogeneity["Post2022"] = interaction_group_results(
    df, "DU_kw_lag", "Post2022", "2022-2024", "2011-2021", sample_mask=lag_mask
)
heterogeneity["Post2021_ex2020"] = interaction_group_results(
    df,
    "DU_kw_lag",
    "Post2021_ex2020",
    "2021-2024",
    "2011-2019",
    sample_mask=lag_mask & df["Post2021_ex2020"].notna(),
)
heterogeneity["DigEconCore"] = interaction_group_results(
    df, "DU_kw_lag", "DigEconCore", "数字核心", "非数字核心", sample_mask=lag_mask
)

(OUT_DIR / "construct_boundary_results.json").write_text(
    json.dumps(construct, ensure_ascii=False, indent=2)
)
(OUT_DIR / "washgap_results.json").write_text(
    json.dumps(washgap, ensure_ascii=False, indent=2)
)
(OUT_DIR / "heterogeneity_results.json").write_text(
    json.dumps(heterogeneity, ensure_ascii=False, indent=2)
)

summary = f"""# v14 Results Summary

## Construct Boundary

- `DU_kw_lag + ln_total_chars_lag`: β={fmt_signed(construct["dukw_total_chars_lag"]["DU_kw_lag"]["coef"])}, t={construct["dukw_total_chars_lag"]["DU_kw_lag"]["t"]:.2f}
- `DU_kw_lag + ln_mda_chars_lag`: β={fmt_signed(construct["dukw_mda_chars_lag"]["DU_kw_lag"]["coef"])}, t={construct["dukw_mda_chars_lag"]["DU_kw_lag"]["t"]:.2f}
- Joint lag model:
  - `DU_kw_lag`: β={fmt_signed(construct["joint_lag"]["DU_kw_lag"]["coef"])}, t={construct["joint_lag"]["DU_kw_lag"]["t"]:.2f}
  - `GenericNarr_lag`: β={fmt_signed(construct["joint_lag"]["GenericNarr_lag"]["coef"])}, t={construct["joint_lag"]["GenericNarr_lag"]["t"]:.2f}
  - FE-adjusted VIF: DU={construct["joint_lag"]["vif_DU_kw_lag"]:.3f}, GenericNarr={construct["joint_lag"]["vif_GenericNarr_lag"]:.3f}
- `DU_kw_resid_lag`: β={fmt_signed(construct["dukw_resid_lag"]["DU_kw_resid_lag"]["coef"])}, t={construct["dukw_resid_lag"]["DU_kw_resid_lag"]["t"]:.2f}
- `DU_kw_strict_lag`: β={fmt_signed(construct["dukw_strict_lag"]["DU_kw_strict_lag"]["coef"])}, t={construct["dukw_strict_lag"]["DU_kw_strict_lag"]["t"]:.2f}

## WashGap

- `WashGap_lag`: β={fmt_signed(washgap["washgap_lag"]["WashGap_lag"]["coef"])}, t={washgap["washgap_lag"]["WashGap_lag"]["t"]:.2f}
- `BroadShallow_lag`: β={fmt_signed(washgap["broadshallow_lag"]["BroadShallow_lag"]["coef"])}, t={washgap["broadshallow_lag"]["BroadShallow_lag"]["t"]:.2f}
- Joint depth model:
  - `DU_kw_lag`: β={fmt_signed(washgap["joint_depth_lag"]["DU_kw_lag"]["coef"])}, t={washgap["joint_depth_lag"]["DU_kw_lag"]["t"]:.2f}
  - `DU_llm_lenstd_lag`: β={fmt_signed(washgap["joint_depth_lag"]["DU_llm_lenstd_lag"]["coef"])}, t={washgap["joint_depth_lag"]["DU_llm_lenstd_lag"]["t"]:.2f}

## Heterogeneity

- `Post2021`: interaction β={fmt_signed(heterogeneity["Post2021"]["interaction"]["coef"])}, p={heterogeneity["Post2021"]["interaction"]["p"]:.4f}
- `Post2022`: interaction β={fmt_signed(heterogeneity["Post2022"]["interaction"]["coef"])}, p={heterogeneity["Post2022"]["interaction"]["p"]:.4f}
- `Post2021_ex2020`: interaction β={fmt_signed(heterogeneity["Post2021_ex2020"]["interaction"]["coef"])}, p={heterogeneity["Post2021_ex2020"]["interaction"]["p"]:.4f}
- `DigEconCore`: interaction β={fmt_signed(heterogeneity["DigEconCore"]["interaction"]["coef"])}, p={heterogeneity["DigEconCore"]["interaction"]["p"]:.4f}
"""
(OUT_DIR / "v14_summary.md").write_text(summary)

print("Done. Results written to", OUT_DIR)
