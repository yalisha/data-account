"""
Construct v16 disclosure-quality variables on top of reg_sample_v18.dta.
Outputs:
  - data_stata/reg_sample_v16_quality.dta
  - results/v16_quality/quality_vars_summary.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path("/Users/mac/computerscience/0做完了/15会计研究")
DATA_STATA = BASE / "data_stata"
DATA_PARQUET = BASE / "data_parquet"
RESULTS = BASE / "results" / "v16_quality"

EPS = 1e-6
QUALITY_VARS = [
    "Quality_sub",
    "Quality_mda",
    "Quality_subden",
    "BroadShallow_cont",
    "WashGap",
]
EXTRA_QUALITY_VARS = ["BroadShallow"]


def winsorize(series: pd.Series, lo: float = 0.01, hi: float = 0.99) -> pd.Series:
    q = series.quantile([lo, hi])
    return series.clip(lower=q.iloc[0], upper=q.iloc[1])


def zscore(series: pd.Series) -> pd.Series:
    std = series.std()
    if pd.isna(std) or std == 0:
        return pd.Series(np.nan, index=series.index)
    return (series - series.mean()) / std


def summary_table(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    for col in cols:
        s = df[col]
        rows.append(
            {
                "variable": col,
                "N": int(s.notna().sum()),
                "mean": float(s.mean()),
                "std": float(s.std()),
                "p25": float(s.quantile(0.25)),
                "p50": float(s.quantile(0.50)),
                "p75": float(s.quantile(0.75)),
                "min": float(s.min()),
                "max": float(s.max()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    panel = pd.read_stata(DATA_STATA / "reg_sample_v18.dta")
    feat = pd.read_parquet(
        DATA_PARQUET / "annual_report_features.parquet",
        columns=[
            "Stkcd",
            "year",
            "kw_total",
            "mda_kw_total",
            "kw_per10k",
            "substantive_ratio",
            "substantive_count",
            "total_chars",
            "has_mda",
        ],
    )
    pbo = pd.read_parquet(
        DATA_PARQUET / "placebo_features.parquet",
        columns=["Stkcd", "year", "placebo_count", "buzzword_count"],
    )

    print("Key dtypes before alignment:")
    print("  panel:", panel[["Stkcd", "year"]].dtypes.to_dict())
    print("  feat :", feat[["Stkcd", "year"]].dtypes.to_dict())
    print("  pbo  :", pbo[["Stkcd", "year"]].dtypes.to_dict())

    for df in (panel, feat, pbo):
        df["Stkcd"] = pd.to_numeric(df["Stkcd"], errors="raise").astype(np.int64)
        df["year"] = pd.to_numeric(df["year"], errors="raise").astype(np.int64)

    assert panel.duplicated(["Stkcd", "year"]).sum() == 0
    assert feat.duplicated(["Stkcd", "year"]).sum() == 0
    assert pbo.duplicated(["Stkcd", "year"]).sum() == 0

    panel = panel.merge(feat, on=["Stkcd", "year"], how="left", suffixes=("", "_feat"))
    panel = panel.merge(pbo, on=["Stkcd", "year"], how="left")

    for col in feat.columns:
        if col in {"Stkcd", "year"}:
            continue
        alt = f"{col}_feat"
        if alt in panel.columns:
            if col in panel.columns:
                panel[col] = panel[col].where(panel[col].notna(), panel[alt])
            else:
                panel[col] = panel[alt]
            panel = panel.drop(columns=[alt])

    print(f"Merged sample: {panel.shape}")
    for col in [
        "kw_total",
        "mda_kw_total",
        "kw_per10k",
        "substantive_ratio",
        "substantive_count",
        "total_chars",
        "has_mda",
        "placebo_count",
        "buzzword_count",
    ]:
        miss = panel[col].isna().mean()
        print(f"  missing rate {col}: {miss:.4%}")

    panel["Quality_sub"] = panel["substantive_ratio"]

    panel["Quality_mda"] = panel["mda_kw_total"] / (panel["kw_total"] + EPS)

    panel["Quality_subden"] = (
        panel["substantive_count"] / (panel["total_chars"] + EPS) * 1e4
    )

    p75 = panel.groupby(["Ind2", "year"])["kw_per10k"].transform(lambda x: x.quantile(0.75))
    p25 = panel.groupby(["Ind2", "year"])["substantive_ratio"].transform(lambda x: x.quantile(0.25))
    panel["BroadShallow"] = (
        (panel["kw_per10k"] > p75) & (panel["substantive_ratio"] < p25)
    ).astype(int)
    panel["BroadShallow_cont"] = panel["kw_per10k"] * (1 - panel["substantive_ratio"])

    panel["WashGap"] = (
        panel["buzzword_count"] + panel["placebo_count"]
    ) / (panel["substantive_count"] + 1)

    for col in QUALITY_VARS:
        panel[col] = winsorize(panel[col])

    for col in QUALITY_VARS + EXTRA_QUALITY_VARS:
        panel[f"{col}_z"] = zscore(panel[col])

    summary = summary_table(panel, QUALITY_VARS + EXTRA_QUALITY_VARS)
    summary.to_csv(RESULTS / "quality_vars_summary.csv", index=False)

    output_path = DATA_STATA / "reg_sample_v16_quality.dta"
    panel.to_stata(output_path, write_index=False, version=118)
    print(f"Saved: {output_path}")

    assert 43000 <= panel.shape[0] <= 44000, f"样本量异常: {panel.shape[0]}"
    assert panel["Quality_sub"].between(0, 1).all()
    assert panel["Quality_mda"].between(0, 1).all()
    assert set(panel["BroadShallow"].unique()).issubset({0, 1})
    assert (panel["Quality_subden"] >= 0).all()
    assert (panel["BroadShallow_cont"] >= 0).all()
    assert (panel["WashGap"] >= 0).all()

    for col in [
        "Quality_sub_z",
        "Quality_mda_z",
        "Quality_subden_z",
        "BroadShallow_cont_z",
        "BroadShallow_z",
        "WashGap_z",
    ]:
        assert abs(panel[col].mean()) < 0.01, f"{col} mean abnormal: {panel[col].mean()}"
        assert abs(panel[col].std() - 1) < 0.01, f"{col} std abnormal: {panel[col].std()}"

    print("Task 1 smoke tests passed.")


if __name__ == "__main__":
    main()
