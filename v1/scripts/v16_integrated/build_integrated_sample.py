"""
Build the unified v16 integrated sample from v15_analysis_sample.parquet.

Outputs:
  - data_stata/reg_sample_v16_integrated.dta
  - results/v16_integrated/sample_summary.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path("/Users/mac/computerscience/0做完了/15会计研究")
V15_SAMPLE = BASE / "results" / "v15_measurement" / "v15_analysis_sample.parquet"
PLACEBO = BASE / "data_parquet" / "placebo_features.parquet"
OUT_DTA = BASE / "data_stata" / "reg_sample_v16_integrated.dta"
OUT_DIR = BASE / "results" / "v16_integrated"
OUT_SUMMARY = OUT_DIR / "sample_summary.csv"

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


def winsorize(series: pd.Series, lo: float = 0.01, hi: float = 0.99) -> pd.Series:
    q = series.quantile([lo, hi])
    return series.clip(lower=q.iloc[0], upper=q.iloc[1])


def lag_if_consecutive(df: pd.DataFrame, col: str) -> pd.Series:
    prev_year = df.groupby("Stkcd")["year"].shift(1)
    prev_val = df.groupby("Stkcd")[col].shift(1)
    return prev_val.where(df["year"].eq(prev_year + 1))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "logs").mkdir(parents=True, exist_ok=True)

    print("Loading v15 analysis sample...")
    df = pd.read_parquet(V15_SAMPLE).copy()
    pbo = pd.read_parquet(PLACEBO, columns=["Stkcd", "year", "placebo_count"]).copy()

    for obj in (df, pbo):
        obj["Stkcd"] = pd.to_numeric(obj["Stkcd"], errors="raise").astype(np.int64)
        obj["year"] = pd.to_numeric(obj["year"], errors="raise").astype(np.int64)

    assert df.duplicated(["Stkcd", "year"]).sum() == 0
    assert pbo.duplicated(["Stkcd", "year"]).sum() == 0

    if "placebo_count" in df.columns:
        df = df.drop(columns=["placebo_count"])
    df = df.merge(pbo, on=["Stkcd", "year"], how="left", validate="1:1")
    df["placebo_count"] = df["placebo_count"].fillna(0)

    df["DU_llm_lenstd"] = np.log1p(df["DU_kw"].clip(lower=0)) * (df["llm_score"] / 3.0)
    df["DU_llm_lenstd"] = winsorize(df["DU_llm_lenstd"])

    df["DU_kw_z"] = (df["DU_kw"] - df["DU_kw"].mean()) / df["DU_kw"].std(ddof=0)
    df["DU_llm_lenstd_z"] = (
        (df["DU_llm_lenstd"] - df["DU_llm_lenstd"].mean()) / df["DU_llm_lenstd"].std(ddof=0)
    )
    df["WashGap"] = winsorize(df["DU_kw_z"] - df["DU_llm_lenstd_z"])

    ordered = df.sort_values(["Stkcd", "year"]).copy()
    ordered["DU_llm_lenstd_lag"] = lag_if_consecutive(ordered, "DU_llm_lenstd")
    ordered["WashGap_lag"] = lag_if_consecutive(ordered, "WashGap")
    ordered["DU_llm_lag"] = lag_if_consecutive(ordered, "DU_llm")

    df = ordered

    summary_rows = [
        {"section": "sample", "key": "N", "value": str(len(df))},
        {"section": "sample", "key": "columns", "value": str(len(df.columns))},
        {"section": "sample", "key": "source", "value": str(V15_SAMPLE)},
        {"section": "spec", "key": "controls", "value": "; ".join(CONTROLS)},
        {"section": "spec", "key": "fixed_effects", "value": "firm + year"},
        {"section": "spec", "key": "cluster", "value": "IndYear_num"},
        {
            "section": "task",
            "key": "h2a_value_chain",
            "value": "DU_stock_lag; DU_dev_lag; DU_app_lag; DU_value_lag; DU_gov_lag",
        },
        {
            "section": "task",
            "key": "h2b_quality_direct",
            "value": "DUclosedloop_lag; DUcore_lag; DUchain_count_lag; DUkw_mda_lag; DU_llm_lenstd_lag",
        },
        {
            "section": "task",
            "key": "h2b_quality_joint",
            "value": "DU_kw_lag + quality_lag; DU_llm_lag + DU_llm_lenstd_lag",
        },
        {
            "section": "task",
            "key": "h2c_washgap",
            "value": "WashGap_lag(v14 formula); DU_kw_lag + WashGap_lag; DU_llm_lag + WashGap_lag",
        },
        {
            "section": "task",
            "key": "h3_downstream",
            "value": "ForecastDisp; CashFlowVol; SCConc with DU_kw and DU_llm",
        },
    ]
    pd.DataFrame(summary_rows).to_csv(OUT_SUMMARY, index=False)

    df.to_stata(OUT_DTA, write_index=False, version=118)
    print(f"Saved: {OUT_DTA}")

    assert df.shape[0] == 43735, f"样本量 {df.shape[0]} != 43,735"
    assert df["DU_llm_lenstd"].notna().mean() > 0.95, "DU_llm_lenstd 缺失率 > 5%"
    assert df["WashGap"].notna().mean() > 0.95, "WashGap 缺失率 > 5%"
    assert df["DU_llm_lenstd_lag"].notna().mean() > 0.85, "DU_llm_lenstd_lag 缺失率 > 15%"
    assert len(df.columns) >= 118, f"列数 {len(df.columns)} < 118"

    print("Task 1 smoke tests passed.")


if __name__ == "__main__":
    main()
