"""
Run lagged-spec supplementary heterogeneity tests for v15.

Dimensions:
  - Post2022
  - Analyst_high
  - MainBoard

Outputs:
  - results/v15_tables/heterogeneity_extra_v15.csv
  - results/v15_tables/heterogeneity_extra_v15.json
  - results/v15_tables/heterogeneity_extra_v15_summary.md
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyfixest as pf


BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data_stata" / "reg_sample_v18.dta"
OUT = BASE / "results" / "v15_tables"
OUT.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUT / "heterogeneity_extra_v15.csv"
OUT_JSON = OUT / "heterogeneity_extra_v15.json"
OUT_MD = OUT / "heterogeneity_extra_v15_summary.md"

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


def lag_if_consecutive(df: pd.DataFrame, col: str) -> pd.Series:
    prev_year = df.groupby("Stkcd")["year"].shift(1)
    prev_val = df.groupby("Stkcd")[col].shift(1)
    return prev_val.where(df["year"].eq(prev_year + 1))


def prepare() -> pd.DataFrame:
    df = pd.read_stata(DATA)
    df["Stkcd"] = df["Stkcd"].astype(int)
    df["year"] = df["year"].astype(int)
    df = df.sort_values(["Stkcd", "year"]).copy()

    for col in ["DU_kw", "DU_llm"]:
        df[f"{col}_lag"] = lag_if_consecutive(df, col)

    analyst_median = df["Analyst"].median(skipna=True)
    df["Analyst_high"] = np.where(
        df["Analyst"].notna(), (df["Analyst"] >= analyst_median).astype(int), np.nan
    )
    df["Post2022"] = (df["year"] >= 2022).astype(int)

    df["Stkcd_str"] = df["Stkcd"].astype(str)
    df["year_str"] = df["year"].astype(str)
    return df


def fit_split(df: pd.DataFrame, treat: str, group: str, value: int) -> dict:
    controls = [c for c in CONTROLS if c != group]
    use = df[
        ["PriceDelay", treat, group, *controls, "Stkcd_str", "year_str", "IndYear"]
    ].dropna()
    use = use.loc[use[group] == value].copy()
    model = pf.feols(
        "PriceDelay ~ " + treat + " + " + " + ".join(controls) + " | Stkcd_str + year_str",
        data=use,
        vcov={"CRV1": "IndYear"},
    )
    return {
        "coef": float(model.coef()[treat]),
        "se": float(model.se()[treat]),
        "t": float(model.tstat()[treat]),
        "p": float(model.pvalue()[treat]),
        "n": int(model._N),
        "r2": float(model._r2),
    }


def fit_interaction(df: pd.DataFrame, treat: str, group: str) -> dict:
    controls = [c for c in CONTROLS if c != group]
    use = df[
        ["PriceDelay", treat, group, *controls, "Stkcd_str", "year_str", "IndYear"]
    ].dropna()
    model = pf.feols(
        "PriceDelay ~ "
        + treat
        + " + "
        + treat
        + ":"
        + group
        + " + "
        + " + ".join(controls)
        + " | Stkcd_str + year_str",
        data=use,
        vcov={"CRV1": "IndYear"},
    )
    term = f"{treat}:{group}"
    return {
        "coef": float(model.coef()[term]),
        "se": float(model.se()[term]),
        "t": float(model.tstat()[term]),
        "p": float(model.pvalue()[term]),
        "n": int(model._N),
    }


def main() -> None:
    df = prepare()

    dims = [
        ("Post2022", "2022-2024", "2011-2021"),
        ("Analyst_high", "高分析师关注", "低分析师关注"),
        ("MainBoard", "主板公司", "非主板公司"),
    ]

    rows = []
    payload = {}

    for group, high_label, low_label in dims:
        high = fit_split(df, "DU_kw_lag", group, 1)
        low = fit_split(df, "DU_kw_lag", group, 0)
        inter = fit_interaction(df, "DU_kw_lag", group)
        payload[group] = {
            "group_high": high_label,
            "group_low": low_label,
            "high": high,
            "low": low,
            "interaction": inter,
        }
        rows.append(
            {
                "dimension": group,
                "group_high": high_label,
                "coef_high": high["coef"],
                "t_high": high["t"],
                "n_high": high["n"],
                "group_low": low_label,
                "coef_low": low["coef"],
                "t_low": low["t"],
                "n_low": low["n"],
                "interaction_p": inter["p"],
                "interaction_t": inter["t"],
            }
        )

    pd.DataFrame(rows).to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# v15 补充异质性结果",
        "",
        "统一采用 `t-1 DU_kw -> t PriceDelay` 的滞后规格，控制企业和年份固定效应及11个控制变量，标准误在行业×年份层面聚类。",
        "",
    ]
    for group, high_label, low_label in dims:
        res = payload[group]
        lines.extend(
            [
                f"## {group}",
                f"- {high_label}: coef={res['high']['coef']:.4f}, t={res['high']['t']:.2f}, N={res['high']['n']}",
                f"- {low_label}: coef={res['low']['coef']:.4f}, t={res['low']['t']:.2f}, N={res['low']['n']}",
                f"- interaction p={res['interaction']['p']:.4f}, t={res['interaction']['t']:.2f}",
                "",
            ]
        )

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
