from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path("/Users/mac/computerscience")
V3 = BASE / "0做完了/15会计研究/v3"
RAR_PATH = BASE / "第三方资料/第三方数据资源/上市公司其他/上市公司招聘大数据2014-2026.3.rar"
CSV_IN_RAR = "上市公司招聘大数据2014-2026.3.csv"
OUT_DATA = V3 / "results/data"
OUT_STATA = V3 / "results/stata"

USECOLS = [
    "企业名称",
    "股票简称",
    "关联股票代码",
    "与上市公司关系",
    "招聘发布年份",
    "招聘岗位",
    "职位描述",
    "招聘人数",
    "招聘类别",
    "初级分类",
    "最低月薪",
    "最高月薪",
    "招聘发布日期",
]

TITLE_DATA_RE = re.compile(
    r"数据|大数据|数据库|数据仓库|数据湖|数据中台|数据平台|数据治理|数据安全|"
    r"数据分析|数据挖掘|数据开发|数据工程|数据建模|数据产品|BI\b|ETL\b|DBA\b|"
    r"算法|机器学习|深度学习|人工智能|AI\b|NLP\b|自然语言|计算机视觉|风控模型",
    flags=re.IGNORECASE,
)

TEXT_DATA_RE = re.compile(
    r"数据|大数据|数据库|数据仓库|数据湖|数据中台|数据平台|数据治理|数据安全|"
    r"数据分析|数据挖掘|数据开发|数据工程|数据建模|数据资产|数据产品|数据标注|"
    r"BI\b|ETL\b|DBA\b|SQL\b|Hadoop|Spark|Hive|Flink|Kafka|ClickHouse|Tableau|PowerBI|"
    r"算法|机器学习|深度学习|人工智能|AI\b|NLP\b|自然语言处理|计算机视觉|知识图谱|推荐系统",
    flags=re.IGNORECASE,
)

AI_RE = re.compile(
    r"人工智能|AI\b|AIGC|算法|机器学习|深度学习|NLP\b|自然语言处理|计算机视觉|"
    r"图像识别|语音识别|知识图谱|推荐系统|大模型|深度神经网络",
    flags=re.IGNORECASE,
)

SOFTWARE_RE = re.compile(
    r"软件|开发工程师|软件工程师|后端|前端|全栈|Java\b|Python\b|C\+\+|"
    r"Go\b|Golang|Android|iOS\b|测试开发|运维开发|架构师|系统开发|平台开发|"
    r"信息系统|ERP\b|MES\b|SaaS\b|PaaS\b",
    flags=re.IGNORECASE,
)

DATA_ENGINEER_RE = re.compile(
    r"数据工程|数据开发|大数据开发|数据仓库|数据平台|数据中台|ETL\b|DBA\b|"
    r"数据库|Hadoop|Spark|Hive|Flink|Kafka|ClickHouse",
    flags=re.IGNORECASE,
)


def normalize_stkcd(s: pd.Series) -> pd.Series:
    out = s.astype(str).str.strip()
    out = out.str.replace(r"\.0$", "", regex=True)
    out = out.str.extract(r"(\d+)", expand=False)
    return pd.to_numeric(out, errors="coerce").astype("Int64")


def parse_year(df: pd.DataFrame) -> pd.Series:
    year = pd.to_numeric(df["招聘发布年份"], errors="coerce")
    missing = year.isna()
    if missing.any():
        date_year = pd.to_datetime(df.loc[missing, "招聘发布日期"], errors="coerce").dt.year
        year.loc[missing] = date_year
    return year.astype("Int64")


def parse_recruits(s: pd.Series) -> pd.Series:
    raw = s.fillna("").astype(str).str.strip()
    nums = raw.str.replace(",", "", regex=False).str.extract(r"(\d+(?:\.\d+)?)", expand=False)
    val = pd.to_numeric(nums, errors="coerce")
    ruogan = raw.str.contains("若干|不限|多人|数名|若干名|若干人", na=False)
    val = val.mask(ruogan & val.isna(), 5)
    val = val.fillna(1)
    return val.clip(lower=0)


def numeric_col(s: pd.Series) -> pd.Series:
    raw = s.fillna("").astype(str).str.replace(",", "", regex=False).str.strip()
    num = raw.str.extract(r"(\d+(?:\.\d+)?)", expand=False)
    return pd.to_numeric(num, errors="coerce")


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    title = df["招聘岗位"].fillna("").astype(str)
    desc = df["职位描述"].fillna("").astype(str)
    cat = df["招聘类别"].fillna("").astype(str) + " " + df["初级分类"].fillna("").astype(str)
    text = title + " " + desc + " " + cat

    df["is_self"] = df["与上市公司关系"].fillna("").astype(str).str.contains("上市公司本身", na=False)
    df["is_data_title"] = title.str.contains(TITLE_DATA_RE, na=False)
    df["is_data_text"] = text.str.contains(TEXT_DATA_RE, na=False)
    df["is_ai_text"] = text.str.contains(AI_RE, na=False)
    df["is_software_text"] = text.str.contains(SOFTWARE_RE, na=False)
    df["is_data_engineer_text"] = text.str.contains(DATA_ENGINEER_RE, na=False)
    return df


def aggregate_chunk(df: pd.DataFrame) -> pd.DataFrame:
    df["Stkcd_num"] = normalize_stkcd(df["关联股票代码"])
    df["year_num"] = parse_year(df)
    df = df.dropna(subset=["Stkcd_num", "year_num"]).copy()
    df["year_num"] = df["year_num"].astype(int)
    df = df[df["year_num"].between(2014, 2026)].copy()
    if df.empty:
        return df

    df["recruits"] = parse_recruits(df["招聘人数"])
    df["salary_min"] = numeric_col(df["最低月薪"])
    df["salary_max"] = numeric_col(df["最高月薪"])
    df["salary_avg"] = df[["salary_min", "salary_max"]].mean(axis=1)
    df = add_indicators(df)

    for flag in [
        "is_self",
        "is_data_title",
        "is_data_text",
        "is_ai_text",
        "is_software_text",
        "is_data_engineer_text",
    ]:
        df[flag] = df[flag].astype(int)

    df["post"] = 1
    df["self_post"] = df["is_self"]
    df["self_recruits"] = df["recruits"] * df["is_self"]

    for flag, prefix in [
        ("is_data_text", "DataHiring"),
        ("is_data_title", "DataHiringTitle"),
        ("is_ai_text", "AIHiring"),
        ("is_software_text", "SoftwareHiring"),
        ("is_data_engineer_text", "DataEngineerHiring"),
    ]:
        df[f"{prefix}Posts"] = df[flag]
        df[f"{prefix}Recruits"] = df["recruits"] * df[flag]
        df[f"{prefix}Posts_self"] = df[flag] * df["is_self"]
        df[f"{prefix}Recruits_self"] = df["recruits"] * df[flag] * df["is_self"]

    df["DataSalary_sum"] = df["salary_avg"].where(df["is_data_text"].astype(bool), np.nan)
    df["DataSalary_n"] = df["DataSalary_sum"].notna().astype(int)
    df["DataSalary_sum"] = df["DataSalary_sum"].fillna(0)

    agg_cols = {
        "post": "sum",
        "recruits": "sum",
        "self_post": "sum",
        "self_recruits": "sum",
        "DataSalary_sum": "sum",
        "DataSalary_n": "sum",
    }
    for prefix in [
        "DataHiring",
        "DataHiringTitle",
        "AIHiring",
        "SoftwareHiring",
        "DataEngineerHiring",
    ]:
        agg_cols[f"{prefix}Posts"] = "sum"
        agg_cols[f"{prefix}Recruits"] = "sum"
        agg_cols[f"{prefix}Posts_self"] = "sum"
        agg_cols[f"{prefix}Recruits_self"] = "sum"

    out = df.groupby(["Stkcd_num", "year_num"], as_index=False, observed=True).agg(agg_cols)
    return out


def finalize(out: pd.DataFrame) -> pd.DataFrame:
    out = out.rename(
        columns={
            "post": "HiringPosts_all",
            "recruits": "HiringRecruits_all",
            "self_post": "HiringPosts_self",
            "self_recruits": "HiringRecruits_self",
        }
    )

    for prefix in [
        "DataHiring",
        "DataHiringTitle",
        "AIHiring",
        "SoftwareHiring",
        "DataEngineerHiring",
    ]:
        out[f"{prefix}Share"] = np.where(
            out["HiringPosts_all"] > 0, out[f"{prefix}Posts"] / out["HiringPosts_all"], np.nan
        )
        out[f"{prefix}RecruitShare"] = np.where(
            out["HiringRecruits_all"] > 0, out[f"{prefix}Recruits"] / out["HiringRecruits_all"], np.nan
        )
        out[f"{prefix}Share_self"] = np.where(
            out["HiringPosts_self"] > 0, out[f"{prefix}Posts_self"] / out["HiringPosts_self"], np.nan
        )
        out[f"{prefix}RecruitShare_self"] = np.where(
            out["HiringRecruits_self"] > 0,
            out[f"{prefix}Recruits_self"] / out["HiringRecruits_self"],
            np.nan,
        )
        out[f"{prefix}Posts_ln"] = np.log1p(out[f"{prefix}Posts"])
        out[f"{prefix}Recruits_ln"] = np.log1p(out[f"{prefix}Recruits"])

    out["HiringPosts_all_ln"] = np.log1p(out["HiringPosts_all"])
    out["HiringRecruits_all_ln"] = np.log1p(out["HiringRecruits_all"])
    out["DataSalary_avg"] = np.where(out["DataSalary_n"] > 0, out["DataSalary_sum"] / out["DataSalary_n"], np.nan)
    out["Stkcd_num"] = out["Stkcd_num"].astype(int)
    out["year_num"] = out["year_num"].astype(int)
    return out.sort_values(["Stkcd_num", "year_num"])


def summary_table(out: pd.DataFrame) -> pd.DataFrame:
    vars_ = [
        "HiringPosts_all",
        "HiringRecruits_all",
        "DataHiringPosts",
        "DataHiringRecruits",
        "DataHiringShare",
        "DataHiringTitlePosts",
        "AIHiringPosts",
        "SoftwareHiringPosts",
        "DataEngineerHiringPosts",
        "DataSalary_avg",
    ]
    rows = []
    for v in vars_:
        s = out[v].dropna()
        rows.append(
            {
                "component": v,
                "N_nonmissing": int(s.shape[0]),
                "N_nonzero": int((s > 0).sum()),
                "firms_nonmissing": int(out.loc[out[v].notna(), "Stkcd_num"].nunique()),
                "firms_nonzero": int(out.loc[out[v] > 0, "Stkcd_num"].nunique()),
                "min_year": int(out.loc[out[v].notna(), "year_num"].min()) if s.shape[0] else np.nan,
                "max_year": int(out.loc[out[v].notna(), "year_num"].max()) if s.shape[0] else np.nan,
                "mean": float(s.mean()) if s.shape[0] else np.nan,
                "p50": float(s.median()) if s.shape[0] else np.nan,
                "p95": float(s.quantile(0.95)) if s.shape[0] else np.nan,
                "max": float(s.max()) if s.shape[0] else np.nan,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    OUT_STATA.mkdir(parents=True, exist_ok=True)

    proc = subprocess.Popen(
        ["bsdtar", "-xOf", str(RAR_PATH), CSV_IN_RAR],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert proc.stdout is not None

    chunks = []
    rows_seen = 0
    try:
        reader = pd.read_csv(
            proc.stdout,
            usecols=USECOLS,
            dtype=str,
            encoding="utf-8-sig",
            chunksize=100_000,
            on_bad_lines="skip",
        )
        for i, chunk in enumerate(reader, start=1):
            rows_seen += len(chunk)
            agg = aggregate_chunk(chunk)
            if not agg.empty:
                chunks.append(agg)
            if i % 10 == 0:
                print(f"processed chunks={i}, raw_rows={rows_seen:,}, agg_chunks={len(chunks)}", flush=True)
    finally:
        stderr = proc.stderr.read().decode("utf-8", errors="ignore") if proc.stderr else ""
        rc = proc.wait()
        if rc not in (0,):
            print(stderr, file=sys.stderr)
            raise RuntimeError(f"bsdtar failed with code {rc}")

    if not chunks:
        raise RuntimeError("No valid hiring data was parsed")

    combined = pd.concat(chunks, ignore_index=True)
    combined = combined.groupby(["Stkcd_num", "year_num"], as_index=False, observed=True).sum(numeric_only=True)
    out = finalize(combined)
    summary = summary_table(out)

    out.to_csv(OUT_DATA / "hiring_custom_v1_firm_year.csv", index=False)
    compact_cols = {
        "Stkcd_num": "Stkcd_num",
        "year_num": "year_num",
        "HiringPosts_all": "HirePostsAll",
        "HiringRecruits_all": "HireRecruitsAll",
        "DataHiringPosts": "DataHireBroadPosts",
        "DataHiringShare": "DataHireBroadShare",
        "DataHiringTitlePosts": "DataHireTitlePosts",
        "DataHiringTitleShare": "DataHireTitleShare",
        "DataHiringTitlePosts_ln": "DataHireTitlePostsLn",
        "AIHiringPosts": "AIHirePosts",
        "AIHiringShare": "AIHireShare",
        "AIHiringPosts_ln": "AIHirePostsLn",
        "SoftwareHiringPosts": "SoftHirePosts",
        "SoftwareHiringShare": "SoftHireShare",
        "DataEngineerHiringPosts": "DataEngHirePosts",
        "DataEngineerHiringShare": "DataEngHireShare",
        "DataEngineerHiringPosts_ln": "DataEngHirePostsLn",
        "DataSalary_avg": "DataSalaryAvg",
    }
    out[list(compact_cols)].rename(columns=compact_cols).to_csv(
        OUT_DATA / "hiring_custom_v1_for_stata.csv", index=False
    )
    summary.to_csv(OUT_STATA / "hiring_custom_v1_summary.csv", index=False)

    by_year = out.groupby("year_num", as_index=False).agg(
        N=("Stkcd_num", "count"),
        firms_data=("DataHiringPosts", lambda s: int((s > 0).sum())),
        data_posts=("DataHiringPosts", "sum"),
        all_posts=("HiringPosts_all", "sum"),
        data_share_mean=("DataHiringShare", "mean"),
    )
    by_year.to_csv(OUT_STATA / "hiring_custom_v1_by_year.csv", index=False)

    print(f"raw rows seen: {rows_seen:,}")
    print(f"wrote {OUT_DATA / 'hiring_custom_v1_firm_year.csv'} rows={len(out):,}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
