#!/usr/bin/env python3
"""Extract DataAsset from the v1 parquet panel for Stata merging."""

from pathlib import Path

import pandas as pd


V1_PANEL = Path("/Users/mac/computerscience/0做完了/15会计研究/v1/data_parquet/panel.parquet")
OUT = Path("/Users/mac/computerscience/0做完了/15会计研究/v3/results/data/data_asset_from_panel.csv")


def main() -> None:
    df = pd.read_parquet(V1_PANEL, columns=["Stkcd", "year", "DataAsset"])
    df = df.rename(columns={"Stkcd": "Stkcd_num", "year": "year_num"})
    df["Stkcd_num"] = pd.to_numeric(df["Stkcd_num"], errors="coerce").astype("Int64")
    df["year_num"] = pd.to_numeric(df["year_num"], errors="coerce").astype("Int64")
    df["DataAsset"] = pd.to_numeric(df["DataAsset"], errors="coerce")
    df = df.dropna(subset=["Stkcd_num", "year_num"])
    df = df.sort_values(["Stkcd_num", "year_num"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)

    positive = df[df["DataAsset"].fillna(0) > 0]
    print(f"wrote {OUT}")
    print(f"rows={len(df)} positive={len(positive)}")
    if not positive.empty:
        print(positive.groupby("year_num").size().to_string())


if __name__ == "__main__":
    main()
