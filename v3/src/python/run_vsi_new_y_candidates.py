#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import math
import warnings

import numpy as np
import pandas as pd
import pyfixest as pf


BASE = Path("/Users/mac/computerscience/0做完了/15会计研究")
DATA = BASE / "v1/data_parquet"
REG_PATH = BASE / "v1/data_stata/reg_sample_y_newdata.dta"
EVENT_PATH = BASE / "v2/results/y_newdata_event_level.csv"
OUT_DIR = BASE / "v3/results"

SAMPLE_PATH = OUT_DIR / "reg_sample_vsi_new_y.dta"
SUMMARY_PATH = OUT_DIR / "vsi_new_y_build_summary.csv"
RESULTS_PATH = OUT_DIR / "vsi_new_y_reg_results.csv"
REPORT_PATH = OUT_DIR / "vsi_new_y_results_20260430.md"

A_SHARE_TYPES = {1, 4, 16, 32}
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

NEW_VARS = [
    "AbsorbRatio60",
    "AbsorbLogRatio60",
    "AbsorbShare60",
    "ICOE_PEG",
    "ICOE_EY",
]

REGRESSION_VARS = NEW_VARS + ["PEAD60_abs", "CS_Spread", "PriceDelay"]


def winsorize(series: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return pd.to_numeric(series, errors="coerce")
    return pd.to_numeric(series, errors="coerce").clip(clean.quantile(lower), clean.quantile(upper))


def normalize_stock(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("Int64")


def load_reg_sample() -> pd.DataFrame:
    reg = pd.read_stata(REG_PATH, convert_categoricals=False)
    reg["Stkcd"] = normalize_stock(reg["Stkcd"])
    reg["year"] = pd.to_numeric(reg["year"], errors="coerce").astype("Int64")
    return reg.dropna(subset=["Stkcd", "year"]).copy()


def build_absorption_measures(reg: pd.DataFrame) -> pd.DataFrame:
    event = pd.read_csv(EVENT_PATH)
    event["Stkcd"] = normalize_stock(event["Stkcd"])
    event["year"] = pd.to_numeric(event["year"], errors="coerce").astype("Int64")
    event = event.dropna(subset=["Stkcd", "year"]).copy()
    event = event.sort_values(["Stkcd", "year"]).drop_duplicates(["Stkcd", "year"], keep="last")

    out = event[["Stkcd", "year", "PEAD60_abs", "EA_absCAR02"]].copy()
    out["PEAD60_abs"] = pd.to_numeric(out["PEAD60_abs"], errors="coerce")
    out["EA_absCAR02"] = pd.to_numeric(out["EA_absCAR02"], errors="coerce")

    eps = 1e-6
    out["AbsorbRatio60"] = out["PEAD60_abs"] / (out["EA_absCAR02"] + eps)
    out["AbsorbLogRatio60"] = np.log((out["PEAD60_abs"] + 1e-4) / (out["EA_absCAR02"] + 1e-4))
    out["AbsorbShare60"] = out["PEAD60_abs"] / (out["PEAD60_abs"] + out["EA_absCAR02"])
    out.loc[(out["PEAD60_abs"].isna()) | (out["EA_absCAR02"].isna()), NEW_VARS[:3]] = np.nan
    return out[["Stkcd", "year", "AbsorbRatio60", "AbsorbLogRatio60", "AbsorbShare60"]]


def build_year_end_prices() -> pd.DataFrame:
    daily = pd.read_parquet(DATA / "daily_return.parquet", columns=["Stkcd", "Trddt", "Clsprc", "Markettype"])
    daily["Stkcd"] = normalize_stock(daily["Stkcd"])
    daily["Trddt"] = pd.to_datetime(daily["Trddt"], errors="coerce")
    daily["Clsprc"] = pd.to_numeric(daily["Clsprc"], errors="coerce")
    daily["Markettype"] = pd.to_numeric(daily["Markettype"], errors="coerce")
    daily = daily[daily["Markettype"].isin(A_SHARE_TYPES)].dropna(subset=["Stkcd", "Trddt", "Clsprc"])
    daily = daily[daily["Clsprc"] > 0].copy()
    daily["year"] = daily["Trddt"].dt.year
    daily = daily.sort_values(["Stkcd", "year", "Trddt"])
    out = daily.groupby(["Stkcd", "year"], observed=True).tail(1)
    return out[["Stkcd", "year", "Clsprc"]].rename(columns={"Clsprc": "year_end_price"})


def build_icoe_measures(reg: pd.DataFrame) -> pd.DataFrame:
    stocks = set(reg["Stkcd"].dropna().astype(int))
    years = set(reg["year"].dropna().astype(int))
    fc = pd.read_parquet(
        DATA / "analyst_forecast.parquet",
        columns=["Stkcd", "Rptdt", "Fenddt", "ReportID", "DeclareDate", "AnanmID", "InstitutionID", "Feps"],
    )
    fc["Stkcd"] = normalize_stock(fc["Stkcd"])
    fc["forecast_dt"] = pd.to_datetime(fc["DeclareDate"], errors="coerce").fillna(
        pd.to_datetime(fc["Rptdt"], errors="coerce")
    )
    fc["fend_dt"] = pd.to_datetime(fc["Fenddt"], errors="coerce")
    fc["Feps"] = pd.to_numeric(fc["Feps"], errors="coerce")
    fc = fc.dropna(subset=["Stkcd", "forecast_dt", "fend_dt", "Feps"]).copy()
    fc = fc[fc["Stkcd"].isin(stocks)]
    fc = fc[fc["fend_dt"].dt.month.eq(12) & fc["fend_dt"].dt.day.eq(31)].copy()
    fc["year"] = fc["forecast_dt"].dt.year
    fc["target_year"] = fc["fend_dt"].dt.year
    fc["horizon"] = fc["target_year"] - fc["year"]
    fc = fc[fc["year"].isin(years) & fc["horizon"].isin([1, 2])].copy()
    fc = fc.drop_duplicates(["Stkcd", "year", "target_year", "ReportID", "AnanmID", "InstitutionID", "Feps"])

    # Keep each analyst-institution's latest forecast for a target fiscal year in the calendar year.
    fc = fc.sort_values("forecast_dt")
    fc = fc.drop_duplicates(["Stkcd", "year", "target_year", "AnanmID", "InstitutionID"], keep="last")
    cons = (
        fc.groupby(["Stkcd", "year", "horizon"], observed=True)
        .agg(Feps_median=("Feps", "median"), Feps_n=("Feps", "count"))
        .reset_index()
    )
    wide = cons.pivot(index=["Stkcd", "year"], columns="horizon", values=["Feps_median", "Feps_n"])
    wide.columns = [f"{name}{int(horizon)}" for name, horizon in wide.columns]
    wide = wide.reset_index()

    prices = build_year_end_prices()
    out = wide.merge(prices, on=["Stkcd", "year"], how="left")
    eps1 = pd.to_numeric(out.get("Feps_median1"), errors="coerce")
    eps2 = pd.to_numeric(out.get("Feps_median2"), errors="coerce")
    price = pd.to_numeric(out["year_end_price"], errors="coerce")
    growth_component = (eps2 - eps1) / price
    out["ICOE_PEG"] = np.sqrt(growth_component.where((growth_component > 0) & (eps1 > 0) & (price > 0)))
    out["ICOE_EY"] = (eps1 / price).where((eps1 > 0) & (price > 0))
    out = out.rename(columns={"Feps_median1": "ICOE_eps1", "Feps_median2": "ICOE_eps2", "Feps_n1": "ICOE_n1", "Feps_n2": "ICOE_n2"})
    keep = ["Stkcd", "year", "ICOE_PEG", "ICOE_EY", "ICOE_eps1", "ICOE_eps2", "ICOE_n1", "ICOE_n2", "year_end_price"]
    for col in keep:
        if col not in out.columns:
            out[col] = np.nan
    return out[keep]


def add_stata_style_lags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["Stkcd_num", "year_num"]).copy()
    for du in ["DU_kw", "DU_llm"]:
        prev_year = df.groupby("Stkcd_num", observed=True)["year_num"].shift(1)
        lag = df.groupby("Stkcd_num", observed=True)[du].shift(1)
        df[f"{du}_lag"] = lag.where(df["year_num"].eq(prev_year + 1))
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, float | int | str]] = []
    for col in NEW_VARS + ["PEAD60_abs", "CS_Spread"]:
        s = pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(dtype=float)
        clean = s.dropna()
        records.append(
            {
                "y_name": col,
                "nonmissing_reg_sample": int(clean.shape[0]),
                "mean": float(clean.mean()) if len(clean) else np.nan,
                "sd": float(clean.std()) if len(clean) else np.nan,
                "p01": float(clean.quantile(0.01)) if len(clean) else np.nan,
                "median": float(clean.median()) if len(clean) else np.nan,
                "p99": float(clean.quantile(0.99)) if len(clean) else np.nan,
            }
        )
    return pd.DataFrame.from_records(records)


def fit_one(df: pd.DataFrame, y: str, du: str) -> dict[str, float | int | str]:
    dulag = f"{du}_lag"
    needed = [y, dulag, "Stkcd_num", "year_num", "IndYear_num"] + CONTROLS
    available = df.dropna(subset=needed).copy()
    if available.empty:
        return {
            "y_name": y,
            "du_measure": du,
            "expected_sign": "negative",
            "coef": np.nan,
            "se": np.nan,
            "t_stat": np.nan,
            "p_value": np.nan,
            "N": 0,
            "status": "unavailable",
        }
    formula = f"{y} ~ {dulag} + {' + '.join(CONTROLS)} | Stkcd_num + year_num"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = pf.feols(formula, data=available, vcov={"CRV1": "IndYear_num"})
    coef = float(model.coef()[dulag])
    se = float(model.se()[dulag])
    t_stat = float(model.tstat()[dulag])
    p_value = float(model.pvalue()[dulag])
    status = "fail"
    if coef < 0 and abs(t_stat) >= 1.96:
        status = "pass"
    elif coef < 0 and abs(t_stat) >= 1.50:
        status = "marginal"
    return {
        "y_name": y,
        "du_measure": du,
        "expected_sign": "negative",
        "coef": coef,
        "se": se,
        "t_stat": t_stat,
        "p_value": p_value,
        "N": int(model._N),
        "status": status,
    }


def run_regressions(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for y in REGRESSION_VARS:
        for du in ["DU_kw", "DU_llm"]:
            rows.append(fit_one(df, y, du))
    return pd.DataFrame.from_records(rows)


def fmt_result(row: pd.Series) -> str:
    if pd.isna(row["coef"]):
        return "-"
    return f"{row['coef']:.4g} (t={row['t_stat']:.2f}, p={row['p_value']:.3g}; {row['status']})"


def write_report(summary: pd.DataFrame, results: pd.DataFrame) -> None:
    pivot = results.pivot(index="y_name", columns="du_measure", values=["coef", "se", "t_stat", "p_value", "N", "status"])
    rows = []
    for y in REGRESSION_VARS:
        sub = results[results["y_name"].eq(y)].set_index("du_measure")
        rows.append(
            {
                "Y": y,
                "DU_kw": fmt_result(sub.loc["DU_kw"]) if "DU_kw" in sub.index else "-",
                "DU_llm": fmt_result(sub.loc["DU_llm"]) if "DU_llm" in sub.index else "-",
                "N_kw": int(sub.loc["DU_kw", "N"]) if "DU_kw" in sub.index else 0,
            }
        )
    table = pd.DataFrame(rows)

    lines = [
        "# VSI 新 Y 试跑结果",
        "",
        "日期：2026-04-30",
        "",
        "## 构造口径",
        "",
        "- `AbsorbRatio60 = PEAD60_abs / (EA_absCAR02 + 1e-6)`，1%/99% winsorize。",
        "- `AbsorbLogRatio60 = log((PEAD60_abs + 1e-4)/(EA_absCAR02 + 1e-4))`，作为比例口径的稳健版本。",
        "- `AbsorbShare60 = PEAD60_abs / (PEAD60_abs + EA_absCAR02)`，表示公告后残余调整占即时反应与残余调整之和的比例。",
        "- `ICOE_PEG = sqrt((FEPS2 - FEPS1) / Price)`，其中 `FEPS1/FEPS2` 为同一自然年内分析师对下一年和下两年 EPS 的最新预测中位数，`Price` 为年末收盘价。",
        "- `ICOE_EY = FEPS1 / Price`，作为覆盖更高的预期盈利收益率补充口径。",
        "- 回归沿用既有口径：`Y_t = L.DU + controls + firm FE + year FE`，聚类到 `IndYear_num`。",
        "",
        "## 覆盖情况",
        "",
        summary.to_markdown(index=False),
        "",
        "## 回归结果",
        "",
        table.to_markdown(index=False),
        "",
        "## 初步判断",
        "",
    ]

    pass_rows = results[(results["status"].eq("pass")) & (results["y_name"].isin(NEW_VARS))]
    if pass_rows.empty:
        lines.append("- 两个新增方向暂无双测度显著主线。")
    else:
        strong = pass_rows.groupby("y_name")["du_measure"].nunique()
        both = strong[strong.eq(2)].index.tolist()
        if both:
            lines.append(f"- 双测度通过的新变量：{', '.join(both)}。")
        one = strong[strong.eq(1)].index.tolist()
        if one:
            lines.append(f"- 单测度通过的新变量：{', '.join(one)}。")
    lines.append("- `PEAD60_abs` 和 `CS_Spread` 保留为对照，便于判断新增变量是否真的超过既有 v3 候选。")
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    reg = load_reg_sample()
    absorption = build_absorption_measures(reg)
    icoe = build_icoe_measures(reg)

    merged = reg.merge(absorption, on=["Stkcd", "year"], how="left", validate="one_to_one")
    merged = merged.merge(icoe, on=["Stkcd", "year"], how="left", validate="one_to_one")
    for col in NEW_VARS:
        merged[col] = winsorize(merged[col])

    summary = summarize(merged)
    summary.to_csv(SUMMARY_PATH, index=False)

    out = merged.copy()
    out["Stkcd"] = out["Stkcd"].astype("int32")
    out["year"] = out["year"].astype("int16")
    out.to_stata(SAMPLE_PATH, write_index=False, version=118)

    reg_df = add_stata_style_lags(merged)
    results = run_regressions(reg_df)
    results.to_csv(RESULTS_PATH, index=False)
    write_report(summary, results)

    print(f"Saved sample: {SAMPLE_PATH}")
    print(f"Saved summary: {SUMMARY_PATH}")
    print(f"Saved regression results: {RESULTS_PATH}")
    print(f"Saved report: {REPORT_PATH}")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
