#!/opt/miniconda3/bin/python
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path("/Users/mac/computerscience/0做完了/15会计研究")
BALANCE_PATH = BASE / "v1/data_parquet/balance_sheet.parquet"
REG_PATH = BASE / "v1/data_stata/reg_sample_iv_v16.dta"
OUTPUT_DTA = BASE / "v1/data_stata/reg_sample_asset_mv.dta"
BUILD_SUMMARY_PATH = BASE / "v2/results/asset_allocation_mv_build_summary.csv"

ASSET_COLS = [
    "A001107000",
    "A001202000",
    "A001211000",
    "A001229000",
    "A001218000",
    "A001219000",
    "A001000000",
]


def annual_report_filter(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Stkcd"] = pd.to_numeric(df["Stkcd"], errors="coerce").astype("Int64")
    df["Accper_str"] = df["Accper"].astype(str)
    mask = (df["Typrep"] == "A") & df["Accper_str"].str.endswith("12-31")
    df = df.loc[mask].copy()
    df["year"] = pd.to_datetime(df["Accper_str"], errors="coerce").dt.year.astype("Int64")
    for col in ASSET_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Stkcd", "year"])
    return df.drop_duplicates(["Stkcd", "year"], keep="last")


def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    out = numerator / denominator
    return out.where(denominator.ne(0))


def winsorize(series: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    clean = series.dropna()
    if clean.empty:
        return series
    q_low = clean.quantile(lower)
    q_high = clean.quantile(upper)
    return series.clip(lower=q_low, upper=q_high)


def build_candidate_frame(bs: pd.DataFrame) -> pd.DataFrame:
    work = bs[["Stkcd", "year"] + ASSET_COLS].copy()
    work = work.sort_values(["Stkcd", "year"]).reset_index(drop=True)

    trading = work["A001107000"].fillna(0)
    afs = work["A001202000"].fillna(0)
    real_estate = work["A001211000"].fillna(0)
    other_noncurrent = work["A001229000"].fillna(0)
    intangible = work["A001218000"].fillna(0)
    dev_outlay = work["A001219000"].fillna(0)
    total_assets = work["A001000000"]

    work["FinRatio_narrow_raw"] = safe_ratio(trading + afs + other_noncurrent, total_assets)
    work["FinRatio_trading_raw"] = safe_ratio(trading, total_assets)
    work["FinRatio_realEstate_raw"] = safe_ratio(real_estate, total_assets)
    work["FinRatio_other_raw"] = safe_ratio(other_noncurrent, total_assets)
    work["FinRatio_v4_raw"] = safe_ratio(trading + afs + real_estate + other_noncurrent, total_assets)
    work["IntangibleRatio_raw"] = safe_ratio(intangible, total_assets)
    work["DevOutlayRatio_raw"] = safe_ratio(dev_outlay, total_assets)

    work["FinRatio_v4_delta_raw"] = work.groupby("Stkcd")["FinRatio_v4_raw"].diff()
    work["FinRatio_v4_vol3y_raw"] = (
        work.groupby("Stkcd")["FinRatio_v4_raw"]
        .transform(lambda s: s.rolling(window=3, min_periods=3).std())
    )
    work["Fin_to_RD_raw"] = work["FinRatio_v4_raw"] / (work["DevOutlayRatio_raw"] + 0.001)

    candidates = [
        "FinRatio_narrow",
        "FinRatio_trading",
        "FinRatio_realEstate",
        "FinRatio_other",
        "FinRatio_v4",
        "IntangibleRatio",
        "DevOutlayRatio",
        "FinRatio_v4_delta",
        "FinRatio_v4_vol3y",
        "Fin_to_RD",
    ]
    for name in candidates:
        work[name] = winsorize(work[f"{name}_raw"])

    return work[["Stkcd", "year"] + candidates]


def summarize_candidates(df: pd.DataFrame, reg_merged: pd.DataFrame) -> pd.DataFrame:
    records = []
    for col in [
        "FinRatio_narrow",
        "FinRatio_trading",
        "FinRatio_realEstate",
        "FinRatio_other",
        "FinRatio_v4",
        "IntangibleRatio",
        "DevOutlayRatio",
        "FinRatio_v4_delta",
        "FinRatio_v4_vol3y",
        "Fin_to_RD",
    ]:
        full = df[col]
        merged = reg_merged[col]
        records.append(
            {
                "mv": col,
                "nonmissing_balance_annual": int(full.notna().sum()),
                "nonmissing_reg_sample": int(merged.notna().sum()),
                "p01": float(full.dropna().quantile(0.01)) if full.notna().any() else np.nan,
                "median": float(full.dropna().median()) if full.notna().any() else np.nan,
                "p99": float(full.dropna().quantile(0.99)) if full.notna().any() else np.nan,
                "min": float(full.dropna().min()) if full.notna().any() else np.nan,
                "max": float(full.dropna().max()) if full.notna().any() else np.nan,
            }
        )
    return pd.DataFrame.from_records(records)


def main() -> None:
    print("Loading balance sheet parquet...")
    bs = annual_report_filter(pd.read_parquet(BALANCE_PATH))
    print(f"Filtered annual-report balance sheet shape: {bs.shape}")

    dup_keys = int(bs.duplicated(["Stkcd", "year"]).sum())
    print(f"Duplicate (Stkcd, year) keys after filter: {dup_keys}")

    mv_df = build_candidate_frame(bs)
    print(f"Candidate MV frame shape: {mv_df.shape}")

    print("Loading lagged regression sample...")
    reg = pd.read_stata(REG_PATH)
    reg["Stkcd"] = pd.to_numeric(reg["Stkcd"], errors="coerce").astype("Int64")
    reg["year"] = pd.to_numeric(reg["year"], errors="coerce").astype("Int64")
    print(f"Regression sample shape: {reg.shape}")

    merged = reg.merge(mv_df, on=["Stkcd", "year"], how="left", validate="one_to_one")
    print(f"Merged sample shape: {merged.shape}")

    overlap = merged[["FinAsset", "FinRatio_v4"]].dropna()
    if not overlap.empty:
        overlap["abs_diff"] = (overlap["FinAsset"] - overlap["FinRatio_v4"]).abs()
        print(
            "FinRatio_v4 vs existing FinAsset: "
            f"n={len(overlap)}, "
            f"mean_abs_diff={overlap['abs_diff'].mean():.8f}, "
            f"max_abs_diff={overlap['abs_diff'].max():.8f}"
        )

    summary = summarize_candidates(mv_df, merged)
    summary.to_csv(BUILD_SUMMARY_PATH, index=False)
    print(f"Saved build summary to {BUILD_SUMMARY_PATH}")

    preview_cols = [
        "Stkcd",
        "year",
        "FinAsset",
        "FinRatio_v4",
        "FinRatio_narrow",
        "IntangibleRatio",
        "DevOutlayRatio",
        "FinRatio_v4_delta",
        "FinRatio_v4_vol3y",
        "Fin_to_RD",
    ]
    print("Merged preview:")
    print(merged[preview_cols].head().to_string(index=False))

    merged.to_stata(OUTPUT_DTA, write_index=False, version=118)
    print(f"Saved merged DTA to {OUTPUT_DTA}")


if __name__ == "__main__":
    main()
