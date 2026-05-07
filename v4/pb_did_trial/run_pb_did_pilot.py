from pathlib import Path
import math
import re

import numpy as np
import pandas as pd
import pyfixest as pf


ROOT = Path("/Users/mac/computerscience/0做完了/15会计研究")
DATA = ROOT / "v1" / "data_parquet"
STATA = ROOT / "v1" / "data_stata"
OUT = ROOT / "v4" / "pb_did_trial"


def clean_city(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    s = s.replace(" ", "")
    return s


def read_annual_parquet(name, cols=None, date_col="EndDate", stk_col="Symbol"):
    df = pd.read_parquet(DATA / name, columns=cols)
    df["year"] = pd.to_datetime(df[date_col]).dt.year
    df["Stkcd"] = pd.to_numeric(df[stk_col], errors="coerce").astype("Int64")
    return df


def load_city_panel():
    cols = [
        "Symbol",
        "EndDate",
        "IndustryCodeC",
        "PROVINCE",
        "CITY",
        "CITYCODE",
        "OfficeAddress",
        "RegisterAddress",
    ]
    fi = read_annual_parquet("firm_info.parquet", cols=cols)
    fi["CITY"] = fi["CITY"].map(clean_city)
    fi["PROVINCE"] = fi["PROVINCE"].map(clean_city)
    fi = fi.sort_values(["Stkcd", "year"]).drop_duplicates(["Stkcd", "year"], keep="last")
    return fi[
        [
            "Stkcd",
            "year",
            "IndustryCodeC",
            "PROVINCE",
            "CITY",
            "CITYCODE",
            "OfficeAddress",
            "RegisterAddress",
        ]
    ]


def load_controller_panel():
    cols = [
        "Symbol",
        "EndDate",
        "EquityNature",
        "EquityNatureID",
        "ActualControllerName",
        "ActualControllerNatureID",
        "Founder",
        "LargestHolderRate",
        "TopTenHoldersRate",
    ]
    en = read_annual_parquet("equity_nature.parquet", cols=cols)
    nature = en["EquityNature"].astype(str)
    ac_id = en["ActualControllerNatureID"].astype(str)
    en["private"] = nature.str.contains("民营", na=False).astype(float)
    # CSMAR controller-nature codes with 31xx appear to be domestic natural-person controllers;
    # 32xx is kept as a broad natural-person/foreign-natural-person proxy for this pilot.
    en["natural_controller"] = ac_id.str.contains(r"31\d{2}|32\d{2}", regex=True, na=False).astype(float)
    en["founder_named"] = en["Founder"].notna().astype(float)
    en = en.sort_values(["Stkcd", "year"]).drop_duplicates(["Stkcd", "year"], keep="last")
    return en[
        [
            "Stkcd",
            "year",
            "private",
            "natural_controller",
            "founder_named",
            "LargestHolderRate",
            "TopTenHoldersRate",
        ]
    ]


def load_short_debt_panel():
    cols = ["Stkcd", "Accper", "Typrep", "A001000000", "A002000000", "A002100000"]
    bs = pd.read_parquet(DATA / "balance_sheet.parquet", columns=cols)
    bs = bs[bs["Typrep"].eq("A")].copy()
    bs["year"] = pd.to_datetime(bs["Accper"]).dt.year
    bs = bs.sort_values(["Stkcd", "year", "Accper"]).drop_duplicates(["Stkcd", "year"], keep="last")
    bs["short_liab_ratio"] = bs["A002100000"] / bs["A002000000"]
    bs["lev_bs"] = bs["A002000000"] / bs["A001000000"]
    bs.loc[~np.isfinite(bs["short_liab_ratio"]), "short_liab_ratio"] = np.nan
    bs.loc[~np.isfinite(bs["lev_bs"]), "lev_bs"] = np.nan
    return bs[["Stkcd", "year", "short_liab_ratio", "lev_bs"]]


def build_policy_flags(df):
    # Narrow hand-collected pilot coding from official/news/appendix sources found in this turn.
    # Effective-year is annualized. Late-December 2021 documents are coded from 2022.
    narrow_city_start = {
        "台州市": 2019,
        "温州市": 2019,
        "东营市": 2020,
        "深圳市": 2021,
        "成都市": 2022,
        "广州市": 2022,
        "无锡市": 2022,
    }
    # Broad coding adds province-wide court guidance as a rough sensitivity only.
    broad_province_start = {
        "浙江省": 2021,
        "江苏省": 2022,
    }
    df["pb_start_narrow"] = df["CITY"].map(narrow_city_start)
    df["pb_narrow"] = ((df["year"] >= df["pb_start_narrow"]) & df["pb_start_narrow"].notna()).astype(float)
    broad_city_start = df["pb_start_narrow"].copy()
    prov_start = df["PROVINCE"].map(broad_province_start)
    df["pb_start_broad"] = broad_city_start
    df.loc[df["pb_start_broad"].isna(), "pb_start_broad"] = prov_start[df["pb_start_broad"].isna()]
    df["pb_broad"] = ((df["year"] >= df["pb_start_broad"]) & df["pb_start_broad"].notna()).astype(float)
    return df


def first_nonmissing(s):
    s = s.dropna()
    return s.iloc[-1] if len(s) else np.nan


def add_pre_exposures(df):
    pre = df[df["year"].between(2016, 2018)].copy()
    firm_pre = pre.groupby("Stkcd").agg(
        private_pre=("private", "max"),
        natural_pre=("natural_controller", "max"),
        founder_pre=("founder_named", "max"),
        lev_pre=("Lev", "mean"),
        short_liab_pre=("short_liab_ratio", "mean"),
        top1_pre=("LargestHolderRate", "mean"),
    )
    firm_pre["high_lev_pre"] = (firm_pre["lev_pre"] > firm_pre["lev_pre"].median()).astype(float)
    firm_pre["high_short_liab_pre"] = (
        firm_pre["short_liab_pre"] > firm_pre["short_liab_pre"].median()
    ).astype(float)
    firm_pre["private_highlev_pre"] = firm_pre["private_pre"] * firm_pre["high_lev_pre"]
    firm_pre["natural_highlev_pre"] = firm_pre["natural_pre"] * firm_pre["high_lev_pre"]
    firm_pre["failure_cost_pre"] = (
        firm_pre[["private_pre", "natural_pre", "founder_pre", "high_lev_pre"]].fillna(0).sum(axis=1)
        / 4.0
    )
    return df.merge(firm_pre.reset_index(), on="Stkcd", how="left")


def get_stat(est, var):
    tidy = est.tidy().reset_index()
    rename = {}
    for c in tidy.columns:
        lc = c.lower()
        if lc in ["index", "coefficient"]:
            rename[c] = "term"
        elif lc in ["estimate", "coef"]:
            rename[c] = "estimate"
        elif lc in ["std. error", "std_error", "se"]:
            rename[c] = "std_error"
        elif lc in ["p-value", "pvalue", "pr(>|t|)"]:
            rename[c] = "pvalue"
        elif lc in ["t value", "tvalue", "statistic"]:
            rename[c] = "tvalue"
    tidy = tidy.rename(columns=rename)
    if "term" not in tidy.columns:
        tidy["term"] = tidy.index.astype(str)
    row = tidy[tidy["term"].eq(var)]
    if row.empty:
        return None
    row = row.iloc[0]
    estv = float(row.get("estimate", np.nan))
    sev = float(row.get("std_error", np.nan))
    tv = float(row.get("tvalue", estv / sev if sev else np.nan))
    pv = float(row.get("pvalue", np.nan))
    return estv, sev, tv, pv


def sig(p):
    if not np.isfinite(p):
        return ""
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


def run_models(df):
    controls = [
        "Size",
        "Lev",
        "ROA",
        "TobinQ",
        "Age",
        "Growth",
        "CFO",
        "IndepRatio",
        "Dual",
        "Top1Share",
        "SOE",
    ]
    outcomes = [
        "AuditFee",
        "absDA",
        "InvestIneff",
        "NCSKEW",
        "DUVOL",
        "short_liab_ratio",
        "Lev",
    ]
    exposures = [
        "private_pre",
        "natural_pre",
        "high_lev_pre",
        "private_highlev_pre",
        "natural_highlev_pre",
        "failure_cost_pre",
    ]
    pb_vars = ["pb_narrow", "pb_broad"]
    rows = []
    for pb in pb_vars:
        for exp in exposures:
            x = f"x_{pb}_{exp}"
            df[x] = df[pb] * df[exp]
            for y in outcomes:
                model_controls = [c for c in controls if c != y]
                cols = [y, x, "Stkcd_str", "city_year", "ind_year", "CITY"] + model_controls
                dat = df[cols].replace([np.inf, -np.inf], np.nan).dropna().copy()
                if dat.empty or dat[x].nunique() < 2 or dat[y].nunique() < 2:
                    rows.append(
                        {
                            "policy": pb,
                            "exposure": exp,
                            "outcome": y,
                            "status": "skip_no_variation",
                            "n": len(dat),
                        }
                    )
                    continue
                fml = f"{y} ~ {x} + {' + '.join(model_controls)} | Stkcd_str + city_year + ind_year"
                try:
                    est = pf.feols(
                        fml,
                        dat,
                        vcov={"CRV1": "CITY"},
                        fixef_rm="singleton",
                        lean=True,
                    )
                    stat = get_stat(est, x)
                    if stat is None:
                        rows.append(
                            {
                                "policy": pb,
                                "exposure": exp,
                                "outcome": y,
                                "status": "term_missing",
                                "n": len(dat),
                            }
                        )
                        continue
                    coef, se, t, p = stat
                    rows.append(
                        {
                            "policy": pb,
                            "exposure": exp,
                            "outcome": y,
                            "status": "ok",
                            "coef": coef,
                            "se_city_cluster": se,
                            "t": t,
                            "p": p,
                            "sig": sig(p),
                            "n_input": len(dat),
                            "treated_x_obs": int((dat[x] > 0).sum()),
                            "clusters_city": int(dat["CITY"].nunique()),
                            "y_mean": float(dat[y].mean()),
                            "x_mean": float(dat[x].mean()),
                        }
                    )
                except Exception as e:
                    rows.append(
                        {
                            "policy": pb,
                            "exposure": exp,
                            "outcome": y,
                            "status": f"error: {type(e).__name__}: {str(e)[:160]}",
                            "n": len(dat),
                        }
                    )
    return pd.DataFrame(rows)


def write_report(panel, results, exposure_summary, policy_summary):
    ok = results[results["status"].eq("ok")].copy()
    ok["abs_t"] = ok["t"].abs()
    top = ok.sort_values(["p", "abs_t"], ascending=[True, False]).head(20)
    report = []
    report.append("# 个人破产制度渐进 DID / DDD 试跑记录\n")
    report.append("日期：2026-05-04\n")
    report.append("## 重要说明\n")
    report.append(
        "这是方向判断用的 pilot，不是最终实证。政策表只使用本轮快速核到的城市/省份节点，"
        "自然人担保金额尚未拿到，因此 Exposure 先用民营、自然人实控人、高杠杆等代理变量。\n"
    )
    report.append("## 样本\n")
    report.append(f"- 回归底表观测值：{len(panel):,}\n")
    report.append(f"- 企业数：{panel['Stkcd'].nunique():,}\n")
    report.append(f"- 年份：{int(panel['year'].min())}-{int(panel['year'].max())}\n")
    report.append(f"- 城市数：{panel['CITY'].nunique():,}\n")
    report.append("\n## 政策编码概览\n\n")
    report.append(policy_summary.to_markdown(index=False))
    report.append("\n\n## 暴露度概览\n\n")
    report.append(exposure_summary.to_markdown(index=False))
    report.append("\n\n## 最靠前的回归信号\n\n")
    if top.empty:
        report.append("没有成功回归结果。\n")
    else:
        cols = [
            "policy",
            "exposure",
            "outcome",
            "coef",
            "se_city_cluster",
            "t",
            "p",
            "sig",
            "n_input",
            "treated_x_obs",
            "clusters_city",
        ]
        report.append(top[cols].to_markdown(index=False, floatfmt=".4f"))
    report.append("\n\n## 初步解释\n")
    report.append(
        "- `pb_narrow` 只含台州、温州、东营、深圳、成都、广州、无锡等快速核到的城市。\n"
    )
    report.append(
        "- `pb_broad` 在 narrow 基础上把浙江省 2021、江苏省 2022 省级指引粗略纳入，"
        "只作为敏感性，不宜直接写进论文。\n"
    )
    report.append(
        "- 固定效应使用企业、城市-年份、行业-年份；因此识别来自同一城市同一年中高暴露企业与低暴露企业的差异。\n"
    )
    report.append(
        "- 如果这一步有信号，下一步必须补自然人为本公司债务提供保证担保的真实 exposure，"
        "否则 A 路线不能进入正式论文。\n"
    )
    (OUT / "pb_did_pilot_report.md").write_text("".join(report), encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    panel = pd.read_stata(STATA / "reg_sample_v18.dta", convert_categoricals=False)
    panel = panel[panel["year"].between(2015, 2023)].copy()

    city = load_city_panel()
    ctrl = load_controller_panel()
    debt = load_short_debt_panel()

    panel = panel.merge(city, on=["Stkcd", "year"], how="left")
    panel = panel.merge(ctrl, on=["Stkcd", "year"], how="left")
    panel = panel.merge(debt, on=["Stkcd", "year"], how="left")
    panel = build_policy_flags(panel)
    panel = add_pre_exposures(panel)

    panel["Stkcd_str"] = panel["Stkcd"].astype(str)
    panel["CITY"] = panel["CITY"].map(clean_city)
    panel["city_year"] = panel["CITY"].astype(str) + "_" + panel["year"].astype(str)
    panel["ind_year"] = panel["Ind2"].astype(str) + "_" + panel["year"].astype(str)
    panel = panel[panel["CITY"].notna()].copy()

    exposure_cols = [
        "private_pre",
        "natural_pre",
        "founder_pre",
        "high_lev_pre",
        "high_short_liab_pre",
        "private_highlev_pre",
        "natural_highlev_pre",
        "failure_cost_pre",
    ]
    exposure_summary = []
    for col in exposure_cols:
        exposure_summary.append(
            {
                "exposure": col,
                "nonmissing": int(panel[col].notna().sum()),
                "mean": float(panel[col].mean()),
                "p50": float(panel[col].median()),
                "p90": float(panel[col].quantile(0.9)),
            }
        )
    exposure_summary = pd.DataFrame(exposure_summary)

    policy_summary = (
        panel.groupby(["pb_narrow", "pb_broad"])
        .agg(obs=("Stkcd", "size"), firms=("Stkcd", "nunique"), cities=("CITY", "nunique"))
        .reset_index()
    )

    results = run_models(panel)
    panel_cols = [
        "Stkcd",
        "year",
        "CITY",
        "PROVINCE",
        "pb_narrow",
        "pb_broad",
        "pb_start_narrow",
        "pb_start_broad",
    ] + exposure_cols
    panel[panel_cols].to_csv(OUT / "pb_did_pilot_panel_keyvars.csv", index=False)
    exposure_summary.to_csv(OUT / "pb_did_pilot_exposure_summary.csv", index=False)
    policy_summary.to_csv(OUT / "pb_did_pilot_policy_summary.csv", index=False)
    results.to_csv(OUT / "pb_did_pilot_results.csv", index=False)
    write_report(panel, results, exposure_summary, policy_summary)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
