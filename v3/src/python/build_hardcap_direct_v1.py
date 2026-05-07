from __future__ import annotations

import re
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd


BASE = Path("/Users/mac/computerscience")
V3 = BASE / "0做完了/15会计研究/v3"
DATA_ROOT = BASE / "第三方资料/第三方数据资源/上市公司数据相关信息"
OUT_DATA = V3 / "results/data"
OUT_STATA = V3 / "results/stata"


FILES = {
    "dig_pat": DATA_ROOT / "数字发明专利授权情况表（年）190317870.zip",
    "dig_human": DATA_ROOT / "数字人力投入计划统计表（月）185606015.zip",
    "dig_capex": DATA_ROOT / "数字资本投入计划统计表（年）185545091.zip",
    "ai_inv": DATA_ROOT / "人工智能投资水平131920031(仅供沪江大学使用).zip",
    "pat_detail": DATA_ROOT / "专利明细情况093018090(仅供哈佛大学使用).zip",
}

DATA_PATENT_RE = re.compile(
    "|".join(
        [
            "数据",
            "大数据",
            "数据库",
            "数据处理",
            "数据分析",
            "数据挖掘",
            "数据安全",
            "数据治理",
            "数据平台",
            "数据中台",
            "人工智能",
            "机器学习",
            "深度学习",
            "算法",
            "云计算",
            "区块链",
            "物联网",
            "软件",
            "信息平台",
            "AI",
            "AIGC",
            "NLP",
        ]
    ),
    flags=re.IGNORECASE,
)


def read_zip_xlsx(path: Path) -> pd.DataFrame:
    with ZipFile(path) as zf:
        xlsx = [name for name in zf.namelist() if name.lower().endswith(".xlsx")]
        if not xlsx:
            raise FileNotFoundError(f"No xlsx file found in {path}")
        raw = zf.read(xlsx[0])
    return pd.read_excel(BytesIO(raw))


def normalize_stkcd(s: pd.Series) -> pd.Series:
    out = s.astype(str).str.strip()
    out = out.str.replace(r"\.0$", "", regex=True)
    out = out.str.extract(r"(\d+)", expand=False)
    out = pd.to_numeric(out, errors="coerce")
    return out.astype("Int64")


def numeric_col(s: pd.Series) -> pd.Series:
    out = s.astype(str).str.replace(",", "", regex=False).str.strip()
    out = out.replace({"": np.nan, "nan": np.nan, "None": np.nan, "没有单位": np.nan})
    return pd.to_numeric(out, errors="coerce")


def add_logs(df: pd.DataFrame, cols: list[str]) -> None:
    for col in cols:
        df[f"{col}_ln"] = np.log1p(df[col].clip(lower=0))


def component_summary(df: pd.DataFrame, component_cols: list[str]) -> pd.DataFrame:
    rows = []
    for col in component_cols:
        valid = df[["Stkcd_num", "year_num", col]].dropna()
        nonzero = valid[valid[col] > 0]
        rows.append(
            {
                "component": col,
                "N_nonmissing": int(len(valid)),
                "N_nonzero": int(len(nonzero)),
                "firms_nonmissing": int(valid["Stkcd_num"].nunique()),
                "firms_nonzero": int(nonzero["Stkcd_num"].nunique()),
                "min_year": int(valid["year_num"].min()) if len(valid) else np.nan,
                "max_year": int(valid["year_num"].max()) if len(valid) else np.nan,
                "mean": float(valid[col].mean()) if len(valid) else np.nan,
                "p50": float(valid[col].median()) if len(valid) else np.nan,
                "p95": float(valid[col].quantile(0.95)) if len(valid) else np.nan,
                "max": float(valid[col].max()) if len(valid) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def build_dig_pat() -> pd.DataFrame:
    df = read_zip_xlsx(FILES["dig_pat"])
    df["Stkcd_num"] = normalize_stkcd(df["symbol"])
    df["year_num"] = pd.to_numeric(df["sgnyear"], errors="coerce").astype("Int64")
    df["DigInvPatAut"] = numeric_col(df["diginvpatautnum"])
    df = df.dropna(subset=["Stkcd_num", "year_num"])
    df = df.groupby(["Stkcd_num", "year_num"], as_index=False, observed=True)["DigInvPatAut"].sum()
    return df


def build_dig_human() -> pd.DataFrame:
    df = read_zip_xlsx(FILES["dig_human"])
    df["Stkcd_num"] = normalize_stkcd(df["symbol"])
    month = df["sgnmonth"].astype(str).str.extract(r"(\d{4})", expand=False)
    df["year_num"] = pd.to_numeric(month, errors="coerce").astype("Int64")
    df["DigHumanDemand"] = numeric_col(df["releasedemandtimes"])
    df["DigHumanRecruit"] = numeric_col(df["recruitmentsnumber"])
    df = df.dropna(subset=["Stkcd_num", "year_num"])
    df = (
        df.groupby(["Stkcd_num", "year_num"], as_index=False, observed=True)[
            ["DigHumanDemand", "DigHumanRecruit"]
        ]
        .sum(min_count=1)
    )
    return df


def build_dig_capex() -> pd.DataFrame:
    df = read_zip_xlsx(FILES["dig_capex"])
    df["Stkcd_num"] = normalize_stkcd(df["symbol"])
    df["year_num"] = pd.to_numeric(df["sgnyear"], errors="coerce").astype("Int64")
    df["DigCapexItem"] = numeric_col(df["investitemnum"])
    df["DigCapexAmount"] = numeric_col(df["investtotalamount"])
    df = df.dropna(subset=["Stkcd_num", "year_num"])
    df = (
        df.groupby(["Stkcd_num", "year_num"], as_index=False, observed=True)[
            ["DigCapexItem", "DigCapexAmount"]
        ]
        .sum(min_count=1)
    )
    return df


def build_ai_inv() -> pd.DataFrame:
    df = read_zip_xlsx(FILES["ai_inv"])
    df["Stkcd_num"] = normalize_stkcd(df["Symbol"])
    df["year_num"] = pd.to_datetime(df["EndDate"], errors="coerce").dt.year.astype("Int64")
    df["Category_num"] = pd.to_numeric(df["Category"], errors="coerce")
    for col in [
        "AISoftInvest",
        "AISoftInvestValueAdd",
        "AIHardInvest",
        "AIHardInvestValueAdd",
        "AIInvestTotal",
        "AIInvestTotalValueAdd",
        "AIInvestLevel",
    ]:
        df[col] = numeric_col(df[col])
    df = df.dropna(subset=["Stkcd_num", "year_num"])

    # Prefer original-cost records (Category=1). If unavailable, keep book-value records.
    df["cat_priority"] = np.where(df["Category_num"] == 1, 0, 1)
    df = df.sort_values(["Stkcd_num", "year_num", "cat_priority"])
    value_cols = [
        "AISoftInvest",
        "AISoftInvestValueAdd",
        "AIHardInvest",
        "AIHardInvestValueAdd",
        "AIInvestTotal",
        "AIInvestTotalValueAdd",
        "AIInvestLevel",
    ]
    df = df.drop_duplicates(["Stkcd_num", "year_num"], keep="first")
    return df[["Stkcd_num", "year_num"] + value_cols]


def build_pat_detail_title() -> pd.DataFrame:
    df = read_zip_xlsx(FILES["pat_detail"])
    df["Stkcd_num"] = normalize_stkcd(df["Symbol"])
    df = df.dropna(subset=["Stkcd_num"])
    df["PatentName_clean"] = df["PatentName"].astype(str).str.strip()
    df["ApplicationDate_dt"] = pd.to_datetime(df["ApplicationDate"], errors="coerce")
    df["GrantDate_dt"] = pd.to_datetime(df["GrantDate"], errors="coerce")
    df["app_year"] = df["ApplicationDate_dt"].dt.year
    df["grant_year"] = df["GrantDate_dt"].dt.year
    df["is_title_data_patent"] = df["PatentName_clean"].str.contains(DATA_PATENT_RE, na=False)
    df["is_invention"] = df["PatentTypeCode"].astype(str).eq("S4901") | df["PatentType"].astype(str).str.contains("发明", na=False)

    key = df["ApplicationNumber"].astype(str).str.strip()
    fallback = df["PatentName_clean"] + "|" + df["ApplicationDate"].astype(str)
    df["pat_key"] = np.where(key.isna() | key.eq("") | key.eq("nan") | key.eq("没有单位"), fallback, key)
    df = df.drop_duplicates(["Stkcd_num", "pat_key"], keep="first")

    app = df.dropna(subset=["app_year"]).copy()
    app["year_num"] = app["app_year"].astype(int)
    app_agg = (
        app.groupby(["Stkcd_num", "year_num"], as_index=False, observed=True)
        .agg(
            PatentAllApp=("pat_key", "count"),
            PatentInvApp=("is_invention", "sum"),
            PatentTitleDataApp=("is_title_data_patent", "sum"),
        )
    )

    grant = df.dropna(subset=["grant_year"]).copy()
    grant["year_num"] = grant["grant_year"].astype(int)
    grant_agg = (
        grant.groupby(["Stkcd_num", "year_num"], as_index=False, observed=True)
        .agg(
            PatentAllGrant=("pat_key", "count"),
            PatentInvGrant=("is_invention", "sum"),
            PatentTitleDataGrant=("is_title_data_patent", "sum"),
        )
    )
    out = app_agg.merge(grant_agg, on=["Stkcd_num", "year_num"], how="outer")
    return out


def main() -> None:
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    OUT_STATA.mkdir(parents=True, exist_ok=True)

    pieces = [
        build_dig_pat(),
        build_dig_human(),
        build_dig_capex(),
        build_ai_inv(),
        build_pat_detail_title(),
    ]
    coverage: dict[str, tuple[int, int]] = {}
    for piece in pieces:
        years = piece["year_num"].dropna().astype(int)
        min_year, max_year = int(years.min()), int(years.max())
        for col in piece.columns:
            if col not in {"Stkcd_num", "year_num"}:
                coverage[col] = (min_year, max_year)

    out = pieces[0]
    for piece in pieces[1:]:
        out = out.merge(piece, on=["Stkcd_num", "year_num"], how="outer")

    count_cols = [
        "DigInvPatAut",
        "DigHumanDemand",
        "DigHumanRecruit",
        "DigCapexItem",
        "DigCapexAmount",
        "AISoftInvest",
        "AISoftInvestValueAdd",
        "AIHardInvest",
        "AIHardInvestValueAdd",
        "AIInvestTotal",
        "AIInvestTotalValueAdd",
        "PatentAllApp",
        "PatentInvApp",
        "PatentTitleDataApp",
        "PatentAllGrant",
        "PatentInvGrant",
        "PatentTitleDataGrant",
    ]
    for col in count_cols:
        if col in out.columns:
            min_year, max_year = coverage.get(col, (None, None))
            if min_year is not None:
                in_coverage = out["year_num"].between(min_year, max_year)
                out.loc[in_coverage, col] = out.loc[in_coverage, col].fillna(0)

    add_logs(
        out,
        [
            "DigInvPatAut",
            "DigHumanDemand",
            "DigHumanRecruit",
            "DigCapexItem",
            "DigCapexAmount",
            "AIInvestTotal",
            "AIInvestTotalValueAdd",
            "PatentTitleDataApp",
            "PatentTitleDataGrant",
        ],
    )

    out["Stkcd_num"] = out["Stkcd_num"].astype(int)
    out["year_num"] = out["year_num"].astype(int)
    out = out.sort_values(["Stkcd_num", "year_num"])

    component_cols = [
        "DigInvPatAut",
        "DigHumanDemand",
        "DigHumanRecruit",
        "DigCapexAmount",
        "AIInvestTotal",
        "AIInvestLevel",
        "PatentTitleDataApp",
        "PatentTitleDataGrant",
    ]
    summary = component_summary(out, component_cols)

    out.to_csv(OUT_DATA / "hardcap_direct_v1_components.csv", index=False)
    summary.to_csv(OUT_STATA / "hardcap_direct_v1_component_summary.csv", index=False)

    print(f"wrote {OUT_DATA / 'hardcap_direct_v1_components.csv'} rows={len(out):,}")
    print(f"wrote {OUT_STATA / 'hardcap_direct_v1_component_summary.csv'}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
