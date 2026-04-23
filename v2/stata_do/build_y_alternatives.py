#!/opt/miniconda3/bin/python
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


BASE = Path("/Users/mac/computerscience/0做完了/15会计研究")
DATA = BASE / "v1/data_parquet"
REG_PATH = BASE / "v1/data_stata/reg_sample_iv_v16.dta"
OUTPUT_DTA = BASE / "v1/data_stata/reg_sample_y_alt.dta"
SUMMARY_PATH = BASE / "v2/results/y_alternatives_build_summary.csv"

A_SHARE_TYPES = {1, 4, 16, 32}
CANDIDATE_VARS = [
    "Amihud_year",
    "Turnover_year",
    "ZeroRet_ratio",
    "SPI",
    "NCSKEW",
    "DUVOL",
    "CS_Spread",
]


def winsorize(series: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    clean = series.dropna()
    if clean.empty:
        return series
    q_low = clean.quantile(lower)
    q_high = clean.quantile(upper)
    return series.clip(lower=q_low, upper=q_high)


def load_reg_sample() -> pd.DataFrame:
    reg = pd.read_stata(REG_PATH, convert_categoricals=False)
    reg["Stkcd"] = pd.to_numeric(reg["Stkcd"], errors="coerce").astype("Int64")
    reg["year"] = pd.to_numeric(reg["year"], errors="coerce").astype("Int64")
    reg = reg.dropna(subset=["Stkcd", "year"]).copy()
    return reg


def sample_filters(reg: pd.DataFrame) -> tuple[set[int], set[int]]:
    years = set(reg["year"].dropna().astype(int).tolist())
    stocks = set(reg["Stkcd"].dropna().astype(int).tolist())
    return years, stocks


def load_filtered_daily(reg: pd.DataFrame) -> pd.DataFrame:
    years, stocks = sample_filters(reg)
    daily = pd.read_parquet(
        DATA / "daily_return.parquet",
        columns=["Stkcd", "Trddt", "Dretwd", "Dretnd", "Dnvaltrd", "Dsmvosd", "Markettype"],
    )
    daily["Stkcd"] = pd.to_numeric(daily["Stkcd"], errors="coerce").astype("Int64")
    daily["Trddt"] = pd.to_datetime(daily["Trddt"], errors="coerce")
    daily["year"] = daily["Trddt"].dt.year.astype("Int64")
    for col in ["Dretwd", "Dretnd", "Dnvaltrd", "Dsmvosd", "Markettype"]:
        daily[col] = pd.to_numeric(daily[col], errors="coerce")
    daily = daily[
        daily["Markettype"].isin(A_SHARE_TYPES)
        & daily["year"].isin(years)
        & daily["Stkcd"].isin(stocks)
    ].copy()
    return daily


def build_amihud_year(reg: pd.DataFrame) -> pd.DataFrame:
    years, stocks = sample_filters(reg)
    ami = pd.read_parquet(DATA / "amihud_daily.parquet", columns=["Stkcd", "Trddt", "ILLIQ"])
    ami["Stkcd"] = pd.to_numeric(ami["Stkcd"], errors="coerce").astype("Int64")
    ami["Trddt"] = pd.to_datetime(ami["Trddt"], errors="coerce")
    ami["year"] = ami["Trddt"].dt.year.astype("Int64")
    ami["ILLIQ"] = pd.to_numeric(ami["ILLIQ"], errors="coerce")
    ami = ami[ami["year"].isin(years) & ami["Stkcd"].isin(stocks)].copy()
    return (
        ami.groupby(["Stkcd", "year"], observed=True)["ILLIQ"]
        .mean()
        .reset_index(name="Amihud_year")
    )


def build_turnover_zero(daily: pd.DataFrame) -> pd.DataFrame:
    work = daily[["Stkcd", "year", "Dretnd", "Dnvaltrd", "Dsmvosd"]].copy()
    work["daily_turnover"] = work["Dnvaltrd"] / work["Dsmvosd"]
    work["daily_turnover"] = work["daily_turnover"].replace([np.inf, -np.inf], np.nan)

    def annualize(group: pd.DataFrame) -> pd.Series:
        turns = group["daily_turnover"].dropna()
        rets = group["Dretnd"].dropna()
        return pd.Series(
            {
                "Turnover_year": turns.mean() if len(turns) >= 60 else np.nan,
                "ZeroRet_ratio": (rets.eq(0).mean()) if len(rets) >= 60 else np.nan,
                "n_obs_daily": int(len(rets)),
            }
        )

    out = (
        work.groupby(["Stkcd", "year"], observed=True)
        .apply(annualize, include_groups=False)
        .reset_index()
    )
    return out.drop(columns=["n_obs_daily"])


def build_spi(reg: pd.DataFrame) -> pd.DataFrame:
    years, stocks = sample_filters(reg)
    synch = pd.read_parquet(DATA / "price_synchronicity.parquet", columns=["Stkcd", "year", "R2_synch"])
    synch["Stkcd"] = pd.to_numeric(synch["Stkcd"], errors="coerce").astype("Int64")
    synch["year"] = pd.to_numeric(synch["year"], errors="coerce").astype("Int64")
    synch["R2_synch"] = pd.to_numeric(synch["R2_synch"], errors="coerce")
    synch = synch[synch["year"].isin(years) & synch["Stkcd"].isin(stocks)].copy()
    synch["SPI"] = 1.0 - synch["R2_synch"]
    return synch[["Stkcd", "year", "SPI"]]


def calc_crash_risk(group: pd.DataFrame) -> pd.Series:
    valid = group[["Dretwd", "MktRet"]].dropna()
    result = {"NCSKEW": np.nan, "DUVOL": np.nan}
    if len(valid) < 30:
        return pd.Series(result)

    y = valid["Dretwd"].to_numpy(dtype=float)
    x = valid["MktRet"].to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(valid)), x])
    try:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        resid = y - X @ beta
    except np.linalg.LinAlgError:
        return pd.Series(result)

    resid = np.clip(resid, -0.99, None)
    w = np.log1p(resid)
    n = len(w)
    if n < 30:
        return pd.Series(result)

    sum_w2 = np.sum(w**2)
    sum_w3 = np.sum(w**3)
    if sum_w2 > 0 and n > 2:
        result["NCSKEW"] = -(n * (n - 1) ** 1.5 * sum_w3) / (
            (n - 1) * (n - 2) * (sum_w2 ** 1.5)
        )

    mean_w = np.mean(w)
    w_up = w[w > mean_w]
    w_down = w[w <= mean_w]
    if len(w_up) > 1 and len(w_down) > 1:
        sum_wu2 = np.sum(w_up**2)
        sum_wd2 = np.sum(w_down**2)
        if sum_wu2 > 0 and sum_wd2 > 0:
            result["DUVOL"] = np.log(((len(w_up) - 1) * sum_wd2) / ((len(w_down) - 1) * sum_wu2))

    return pd.Series(result)


def build_crash_risk(daily: pd.DataFrame) -> pd.DataFrame:
    mkt = pd.read_parquet(DATA / "market_index.parquet", columns=["Indexcd", "Trddt", "Retindex"])
    mkt["Indexcd"] = pd.to_numeric(mkt["Indexcd"], errors="coerce")
    mkt["Trddt"] = pd.to_datetime(mkt["Trddt"], errors="coerce")
    mkt["Retindex"] = pd.to_numeric(mkt["Retindex"], errors="coerce")
    mkt = mkt[mkt["Indexcd"] == 1][["Trddt", "Retindex"]].drop_duplicates("Trddt")
    mkt = mkt.rename(columns={"Retindex": "MktRet"})

    crash_input = daily[["Stkcd", "year", "Trddt", "Dretwd"]].merge(mkt, on="Trddt", how="left")
    crash = (
        crash_input.groupby(["Stkcd", "year"], observed=True)
        .apply(calc_crash_risk, include_groups=False)
        .reset_index()
    )
    return crash


def detect_cs_source() -> tuple[str, str]:
    high_tokens = {"high", "hi", "hiprc", "highest", "highprc"}
    low_tokens = {"low", "lo", "loprc", "lowest", "lowprc"}
    for path in sorted(DATA.glob("*.parquet")):
        cols = {col.lower() for col in pq.ParquetFile(path).schema.names}
        if "stkcd" not in cols:
            continue
        has_high = any(token in cols for token in high_tokens)
        has_low = any(token in cols for token in low_tokens)
        if has_high and has_low:
            return ("available", str(path))
    return ("missing", "No stock-level daily high/low parquet found under v1/data_parquet")


def summarize_candidates(candidate_frame: pd.DataFrame, merged: pd.DataFrame, cs_note: str) -> pd.DataFrame:
    records = []
    for col in CANDIDATE_VARS:
        source = candidate_frame[col] if col in candidate_frame.columns else pd.Series(dtype=float)
        reg_col = merged[col]
        note = ""
        if col == "CS_Spread":
            note = cs_note
        elif reg_col.notna().sum() < 25000:
            note = "sample_sparse"

        records.append(
            {
                "y_name": col,
                "nonmissing_candidate_frame": int(source.notna().sum()) if len(source) else 0,
                "nonmissing_reg_sample": int(reg_col.notna().sum()),
                "mean": float(reg_col.dropna().mean()) if reg_col.notna().any() else np.nan,
                "sd": float(reg_col.dropna().std()) if reg_col.notna().any() else np.nan,
                "min": float(reg_col.dropna().min()) if reg_col.notna().any() else np.nan,
                "p01": float(reg_col.dropna().quantile(0.01)) if reg_col.notna().any() else np.nan,
                "median": float(reg_col.dropna().median()) if reg_col.notna().any() else np.nan,
                "p99": float(reg_col.dropna().quantile(0.99)) if reg_col.notna().any() else np.nan,
                "max": float(reg_col.dropna().max()) if reg_col.notna().any() else np.nan,
                "note": note,
            }
        )
    return pd.DataFrame.from_records(records)


def compare_with_existing(merged: pd.DataFrame) -> None:
    for old_col, new_col in [("Amihud", "Amihud_year"), ("Turnover", "Turnover_year")]:
        overlap = merged[[old_col, new_col]].dropna()
        if overlap.empty:
            print(f"{new_col}: no overlap with existing {old_col}")
            continue
        diff = (overlap[old_col] - overlap[new_col]).abs()
        print(
            f"{new_col} vs {old_col}: "
            f"n={len(overlap)}, "
            f"corr={overlap.corr().iloc[0, 1]:.12f}, "
            f"mean_abs_diff={diff.mean():.12g}, "
            f"max_abs_diff={diff.max():.12g}"
        )


def main() -> None:
    print("Loading lagged regression sample...")
    reg = load_reg_sample()
    print(f"Regression sample shape: {reg.shape}")
    print(f"Regression sample years: {int(reg['year'].min())} - {int(reg['year'].max())}")

    print("\nLoading filtered daily data...")
    daily = load_filtered_daily(reg)
    print(
        f"Daily sample shape: {daily.shape}; "
        f"stocks={daily['Stkcd'].nunique()}, years={daily['year'].nunique()}"
    )

    print("\nBuilding Amihud_year...")
    amihud_year = build_amihud_year(reg)
    print(f"Amihud_year obs: {len(amihud_year)}")

    print("\nBuilding Turnover_year and ZeroRet_ratio...")
    turnover_zero = build_turnover_zero(daily)
    print(
        f"Turnover/Zero obs: {len(turnover_zero)}; "
        f"Turnover nonmissing={turnover_zero['Turnover_year'].notna().sum()}, "
        f"ZeroRet nonmissing={turnover_zero['ZeroRet_ratio'].notna().sum()}"
    )

    print("\nBuilding SPI from price_synchronicity.parquet...")
    spi = build_spi(reg)
    print(f"SPI obs: {len(spi)}")

    print("\nBuilding crash-risk variables (NCSKEW, DUVOL)...")
    crash = build_crash_risk(daily)
    print(
        f"Crash-risk obs: {len(crash)}; "
        f"NCSKEW nonmissing={crash['NCSKEW'].notna().sum()}, "
        f"DUVOL nonmissing={crash['DUVOL'].notna().sum()}"
    )

    print("\nChecking CS_Spread feasibility...")
    cs_status, cs_note = detect_cs_source()
    print(f"CS_Spread status: {cs_status} | {cs_note}")

    candidate = reg[["Stkcd", "year"]].drop_duplicates().copy()
    for frame in [amihud_year, turnover_zero, spi, crash]:
        candidate = candidate.merge(frame, on=["Stkcd", "year"], how="left", validate="one_to_one")
    if cs_status != "available":
        candidate["CS_Spread"] = np.nan

    merged = reg.merge(candidate, on=["Stkcd", "year"], how="left", validate="one_to_one")
    compare_with_existing(merged)
    for col in CANDIDATE_VARS:
        merged[col] = winsorize(merged[col])

    summary = summarize_candidates(candidate, merged, cs_note)
    summary.to_csv(SUMMARY_PATH, index=False)
    print(f"\nSaved build summary to {SUMMARY_PATH}")
    print(summary.to_string(index=False))

    preview_cols = [
        "Stkcd",
        "year",
        "PriceDelay",
        "SYNCH",
        "Amihud_year",
        "Turnover_year",
        "ZeroRet_ratio",
        "SPI",
        "NCSKEW",
        "DUVOL",
        "CS_Spread",
    ]
    print("\nMerged preview (first 5 rows):")
    print(merged[preview_cols].head().to_string(index=False))

    out = merged.copy()
    out["Stkcd"] = out["Stkcd"].astype("int32")
    out["year"] = out["year"].astype("int16")
    out.to_stata(OUTPUT_DTA, write_index=False, version=118)
    print(f"\nSaved merged DTA to {OUTPUT_DTA}")
    print(f"Final shape: {out.shape}")


if __name__ == "__main__":
    main()
