"""
Run v15 measurement-upgrade analyses.

Focus:
  - DUevent (rule-based substantive-use proxy)
  - DUchain_count / DUclosedloop (value-chain completeness)
  - DUcore (narrower core-intensity proxy)
  - DUkw_mda (section-limited proxy)

Outputs:
  - results/v15_measurement/v15_analysis_sample.parquet
  - results/v15_measurement/measurement_upgrade_results.json
  - results/v15_measurement/measurement_upgrade_summary.md
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
OUT_DIR = BASE / "results" / "v15_measurement"
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

DIM_COLS = [
    "kw_data_stock",
    "kw_data_dev",
    "kw_data_app",
    "kw_data_value",
    "kw_data_gov",
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
        "r2_within": float(model._r2_within),
    }


def format_effect(row: dict) -> str:
    stars = ""
    if row["p"] < 0.01:
        stars = "***"
    elif row["p"] < 0.05:
        stars = "**"
    elif row["p"] < 0.10:
        stars = "*"
    return f"{row['coef']:+.4f}{stars} (t={row['t']:.2f}, p={row['p']:.4f})"


print("Loading base sample...")
reg = pd.read_stata(DATA_STATA)
reg["Stkcd"] = reg["Stkcd"].astype(int)
reg["year"] = reg["year"].astype(int)

feat = pd.read_parquet(AR_FEAT)
placebo = pd.read_parquet(PLACEBO, columns=["Stkcd", "year", "buzzword_count"])

df = reg.merge(feat, on=["Stkcd", "year", "kw_total"], how="left", validate="1:1")
df = df.merge(placebo, on=["Stkcd", "year"], how="left")
df["buzzword_count"] = df["buzzword_count"].fillna(0)

print("Constructing v15 measures...")
df["GenericNarr"] = df["buzzword_count"] / df["total_chars"].clip(lower=1) * 10000

# Rule-based substantive-use proxy from the existing text-feature pipeline.
df["DUevent_raw"] = df["substantive_count"].fillna(0)
df["DUevent"] = df["substantive_count"].fillna(0) / df["total_chars"].clip(lower=1) * 10000
df["DUevent_ratio"] = df["substantive_ratio"]

for col in DIM_COLS:
    df[col] = df[col].fillna(0)

df["DUchain_count"] = sum((df[col] > 0).astype(int) for col in DIM_COLS)
df["DUclosedloop"] = (
    (df["kw_data_stock"] > 0)
    & ((df["kw_data_app"] > 0) | (df["kw_data_dev"] > 0))
    & ((df["kw_data_gov"] > 0) | (df["kw_data_value"] > 0))
).astype(int)

shares = np.column_stack([df[col].to_numpy() for col in DIM_COLS])
row_sum = shares.sum(axis=1)
with np.errstate(divide="ignore", invalid="ignore"):
    probs = np.divide(shares, row_sum[:, None], where=row_sum[:, None] > 0)
    logp = np.where(probs > 0, np.log(probs), 0)
    entropy = -(probs * logp).sum(axis=1)
df["DUbalance"] = np.where(row_sum > 0, entropy / np.log(len(DIM_COLS)), np.nan)

# Narrower but still observable core proxy based on app + value + governance.
df["DUcore_raw"] = df["kw_data_app"] + df["kw_data_value"] + df["kw_data_gov"]
df["DUcore"] = df["DUcore_raw"] / df["total_chars"].clip(lower=1) * 10000

# Section-limited measure currently feasible from the archived feature table.
df["DUkw_mda"] = df["mda_kw_per10k"]

for dim, col in [
    ("DU_stock", "kw_data_stock"),
    ("DU_dev", "kw_data_dev"),
    ("DU_app", "kw_data_app"),
    ("DU_value", "kw_data_value"),
    ("DU_gov", "kw_data_gov"),
]:
    df[dim] = df[col] / df["total_chars"].clip(lower=1) * 10000

for col in [
    "GenericNarr",
    "DUevent",
    "DUevent_ratio",
    "DUchain_count",
    "DUbalance",
    "DUcore",
    "DUkw_mda",
    "DU_stock",
    "DU_dev",
    "DU_app",
    "DU_value",
    "DU_gov",
]:
    if df[col].notna().any():
        df[col] = winsorize(df[col])

ordered = df.sort_values(["Stkcd", "year"]).copy()
for col in [
    "DU_kw",
    "GenericNarr",
    "DUevent",
    "DUevent_ratio",
    "DUchain_count",
    "DUclosedloop",
    "DUbalance",
    "DUcore",
    "DUkw_mda",
    "DU_stock",
    "DU_dev",
    "DU_app",
    "DU_value",
    "DU_gov",
]:
    ordered[f"{col}_lag"] = lag_if_consecutive(ordered, col)

for col in [c for c in ordered.columns if c.endswith("_lag")]:
    df[col] = ordered[col].values

df.to_parquet(OUT_DIR / "v15_analysis_sample.parquet", index=False)

print("Running v15 regressions...")
results: dict[str, dict] = {
    "baseline": {},
    "measurement_upgrade": {},
    "dimension_decomposition": {},
}

for name, xvars in [
    ("dukw_lag", ["DU_kw_lag"]),
    ("duevent_lag", ["DUevent_lag"]),
    ("duevent_ratio_lag", ["DUevent_ratio_lag"]),
    ("duchain_count_lag", ["DUchain_count_lag"]),
    ("duclosedloop_lag", ["DUclosedloop_lag"]),
    ("ducore_lag", ["DUcore_lag"]),
    ("dukw_mda_lag", ["DUkw_mda_lag"]),
]:
    _, model = fit_feols(df, "PriceDelay", xvars)
    bucket = "baseline" if name == "dukw_lag" else "measurement_upgrade"
    results[bucket][name] = {var: coef_row(model, var) for var in xvars}

for name, xvars in [
    ("duevent_joint_dukw", ["DUevent_lag", "DU_kw_lag"]),
    ("duevent_joint_gn", ["DUevent_lag", "GenericNarr_lag"]),
    ("duchain_joint_dukw", ["DUchain_count_lag", "DU_kw_lag"]),
    ("duchain_joint_gn", ["DUchain_count_lag", "GenericNarr_lag"]),
    ("duclosedloop_joint_dukw", ["DUclosedloop_lag", "DU_kw_lag"]),
    ("duclosedloop_joint_gn", ["DUclosedloop_lag", "GenericNarr_lag"]),
    ("ducore_joint_dukw", ["DUcore_lag", "DU_kw_lag"]),
    ("ducore_joint_gn", ["DUcore_lag", "GenericNarr_lag"]),
    ("dukw_mda_joint_dukw", ["DUkw_mda_lag", "DU_kw_lag"]),
    ("dukw_mda_joint_gn", ["DUkw_mda_lag", "GenericNarr_lag"]),
]:
    _, model = fit_feols(df, "PriceDelay", xvars)
    results["measurement_upgrade"][name] = {var: coef_row(model, var) for var in xvars}

for name, xvars in [
    ("DU_stock_lag", ["DU_stock_lag"]),
    ("DU_dev_lag", ["DU_dev_lag"]),
    ("DU_app_lag", ["DU_app_lag"]),
    ("DU_value_lag", ["DU_value_lag"]),
    ("DU_gov_lag", ["DU_gov_lag"]),
]:
    _, model = fit_feols(df, "PriceDelay", xvars)
    results["dimension_decomposition"][name] = {var: coef_row(model, var) for var in xvars}

(OUT_DIR / "measurement_upgrade_results.json").write_text(
    json.dumps(results, ensure_ascii=False, indent=2)
)

summary = f"""# v15 Measurement Upgrade Summary

## Baseline

- `DU_kw_lag`: {format_effect(results["baseline"]["dukw_lag"]["DU_kw_lag"])}

## Measurement Upgrade

- `DUevent_lag`: {format_effect(results["measurement_upgrade"]["duevent_lag"]["DUevent_lag"])}
- `DUevent_ratio_lag`: {format_effect(results["measurement_upgrade"]["duevent_ratio_lag"]["DUevent_ratio_lag"])}
- `DUchain_count_lag`: {format_effect(results["measurement_upgrade"]["duchain_count_lag"]["DUchain_count_lag"])}
- `DUclosedloop_lag`: {format_effect(results["measurement_upgrade"]["duclosedloop_lag"]["DUclosedloop_lag"])}
- `DUcore_lag`: {format_effect(results["measurement_upgrade"]["ducore_lag"]["DUcore_lag"])}
- `DUkw_mda_lag`: {format_effect(results["measurement_upgrade"]["dukw_mda_lag"]["DUkw_mda_lag"])}

### Joint Models

- `DUevent_lag + DU_kw_lag`
  - `DUevent_lag`: {format_effect(results["measurement_upgrade"]["duevent_joint_dukw"]["DUevent_lag"])}
  - `DU_kw_lag`: {format_effect(results["measurement_upgrade"]["duevent_joint_dukw"]["DU_kw_lag"])}
- `DUevent_lag + GenericNarr_lag`
  - `DUevent_lag`: {format_effect(results["measurement_upgrade"]["duevent_joint_gn"]["DUevent_lag"])}
  - `GenericNarr_lag`: {format_effect(results["measurement_upgrade"]["duevent_joint_gn"]["GenericNarr_lag"])}
- `DUchain_count_lag + DU_kw_lag`
  - `DUchain_count_lag`: {format_effect(results["measurement_upgrade"]["duchain_joint_dukw"]["DUchain_count_lag"])}
  - `DU_kw_lag`: {format_effect(results["measurement_upgrade"]["duchain_joint_dukw"]["DU_kw_lag"])}
- `DUclosedloop_lag + DU_kw_lag`
  - `DUclosedloop_lag`: {format_effect(results["measurement_upgrade"]["duclosedloop_joint_dukw"]["DUclosedloop_lag"])}
  - `DU_kw_lag`: {format_effect(results["measurement_upgrade"]["duclosedloop_joint_dukw"]["DU_kw_lag"])}
- `DUclosedloop_lag + GenericNarr_lag`
  - `DUclosedloop_lag`: {format_effect(results["measurement_upgrade"]["duclosedloop_joint_gn"]["DUclosedloop_lag"])}
  - `GenericNarr_lag`: {format_effect(results["measurement_upgrade"]["duclosedloop_joint_gn"]["GenericNarr_lag"])}
- `DUcore_lag + DU_kw_lag`
  - `DUcore_lag`: {format_effect(results["measurement_upgrade"]["ducore_joint_dukw"]["DUcore_lag"])}
  - `DU_kw_lag`: {format_effect(results["measurement_upgrade"]["ducore_joint_dukw"]["DU_kw_lag"])}
- `DUcore_lag + GenericNarr_lag`
  - `DUcore_lag`: {format_effect(results["measurement_upgrade"]["ducore_joint_gn"]["DUcore_lag"])}
  - `GenericNarr_lag`: {format_effect(results["measurement_upgrade"]["ducore_joint_gn"]["GenericNarr_lag"])}
- `DUkw_mda_lag + DU_kw_lag`
  - `DUkw_mda_lag`: {format_effect(results["measurement_upgrade"]["dukw_mda_joint_dukw"]["DUkw_mda_lag"])}
  - `DU_kw_lag`: {format_effect(results["measurement_upgrade"]["dukw_mda_joint_dukw"]["DU_kw_lag"])}
- `DUkw_mda_lag + GenericNarr_lag`
  - `DUkw_mda_lag`: {format_effect(results["measurement_upgrade"]["dukw_mda_joint_gn"]["DUkw_mda_lag"])}
  - `GenericNarr_lag`: {format_effect(results["measurement_upgrade"]["dukw_mda_joint_gn"]["GenericNarr_lag"])}

## Dimension Decomposition

- `DU_stock_lag`: {format_effect(results["dimension_decomposition"]["DU_stock_lag"]["DU_stock_lag"])}
- `DU_dev_lag`: {format_effect(results["dimension_decomposition"]["DU_dev_lag"]["DU_dev_lag"])}
- `DU_app_lag`: {format_effect(results["dimension_decomposition"]["DU_app_lag"]["DU_app_lag"])}
- `DU_value_lag`: {format_effect(results["dimension_decomposition"]["DU_value_lag"]["DU_value_lag"])}
- `DU_gov_lag`: {format_effect(results["dimension_decomposition"]["DU_gov_lag"]["DU_gov_lag"])}

## Takeaways

- `DUclosedloop` and `DUchain_count` provide the cleanest supportive evidence that a more complete data-use chain is associated with lower stock price delay.
- `DUkw_mda` is significantly negative in the preferred lagged specification, suggesting that operating-section disclosure carries pricing content, but it does not survive joint models with `DU_kw` or `GenericNarr`.
- `DUevent` and `DUevent_ratio` do not provide stable standalone support in the current archived feature pipeline and should remain supplementary.
- `DUcore` is negative on a standalone basis but does not displace `DU_kw` once entered jointly, so it should be treated as a narrower supportive proxy rather than a replacement baseline.
"""

(OUT_DIR / "measurement_upgrade_summary.md").write_text(summary)
print("Done. Results written to", OUT_DIR)
