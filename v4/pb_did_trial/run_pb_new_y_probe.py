from pathlib import Path

import numpy as np
import pandas as pd
import pyfixest as pf

from run_pb_b_parallel_trends import (
    EVENT_BINS,
    add_event_terms,
    get_tidy,
    load_panel,
    wald_for_terms,
)
from run_pb_did_pilot import DATA, OUT, sig


EXPOSURES = [
    "natural_pre",
    "failure_cost_pre",
    "high_lev_pre",
    "natural_highlev_pre",
    "private_highlev_pre",
]

OUTCOMES = [
    "LossDummy",
    "LossMagnitude",
    "NegAccrual",
    "ConservAccrual",
    "IntangibleDecrease",
    "DevExpDecrease",
]

WEAK_PROXY_OUTCOMES = {"IntangibleDecrease", "DevExpDecrease"}

BASE_CONTROLS = [
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

CONTROL_EXCLUDE = {
    "LossDummy": {"ROA"},
    "LossMagnitude": {"ROA"},
    "NegAccrual": {"ROA", "CFO"},
    "ConservAccrual": {"ROA", "CFO"},
}


def winsorize(s, lo=0.01, hi=0.99):
    x = pd.to_numeric(s, errors="coerce")
    qlo, qhi = x.quantile([lo, hi])
    return x.clip(qlo, qhi)


def load_financial_new_y():
    inc_cols = ["Stkcd", "Accper", "Typrep", "B002000000"]
    cf_cols = ["Stkcd", "Accper", "Typrep", "C001000000"]
    bs_cols = ["Stkcd", "Accper", "Typrep", "A001000000", "A001218000", "A001219000"]

    inc = pd.read_parquet(DATA / "income_stmt.parquet", columns=inc_cols)
    cf = pd.read_parquet(DATA / "cashflow.parquet", columns=cf_cols)
    bs = pd.read_parquet(DATA / "balance_sheet.parquet", columns=bs_cols)

    def annual(df):
        out = df[df["Typrep"].eq("A")].copy()
        out["Accper"] = pd.to_datetime(out["Accper"], errors="coerce")
        out = out[out["Accper"].dt.month.eq(12)].copy()
        out["year"] = out["Accper"].dt.year
        out["Stkcd"] = pd.to_numeric(out["Stkcd"], errors="coerce").astype("Int64")
        out = out.sort_values(["Stkcd", "year", "Accper"]).drop_duplicates(["Stkcd", "year"], keep="last")
        return out.drop(columns=["Accper", "Typrep"])

    fin = annual(inc).merge(annual(cf), on=["Stkcd", "year"], how="outer")
    fin = fin.merge(annual(bs), on=["Stkcd", "year"], how="outer")
    fin = fin.sort_values(["Stkcd", "year"]).copy()

    for col in ["B002000000", "C001000000", "A001000000", "A001218000", "A001219000"]:
        fin[col] = pd.to_numeric(fin[col], errors="coerce")

    fin["lag_assets"] = fin.groupby("Stkcd")["A001000000"].shift(1)
    fin["lag_intangible"] = fin.groupby("Stkcd")["A001218000"].shift(1)
    fin["lag_devexp"] = fin.groupby("Stkcd")["A001219000"].shift(1)
    good_assets = fin["lag_assets"].gt(0)

    ni_scaled = (fin["B002000000"] / fin["lag_assets"]).where(good_assets)
    cfo_scaled = (fin["C001000000"] / fin["lag_assets"]).where(good_assets)
    accrual = (fin["B002000000"] - fin["C001000000"]) / fin["lag_assets"]
    intangible_delta = (fin["A001218000"] - fin["lag_intangible"]) / fin["lag_assets"]
    devexp_delta = (fin["A001219000"] - fin["lag_devexp"]) / fin["lag_assets"]

    fin["LossDummy"] = fin["B002000000"].lt(0).astype(float)
    fin.loc[fin["B002000000"].isna(), "LossDummy"] = np.nan
    fin["LossMagnitude"] = np.maximum(-ni_scaled, 0)
    fin["NegAccrual"] = np.maximum(-accrual.where(good_assets), 0)
    fin["ConservAccrual"] = -accrual.where(good_assets)
    fin["IntangibleDecrease"] = np.maximum(-intangible_delta.where(good_assets), 0)
    fin["DevExpDecrease"] = np.maximum(-devexp_delta.where(good_assets), 0)

    for col in [
        "LossMagnitude",
        "NegAccrual",
        "ConservAccrual",
        "IntangibleDecrease",
        "DevExpDecrease",
    ]:
        fin[col] = winsorize(fin[col])

    return fin[["Stkcd", "year"] + OUTCOMES]


def controls_for(outcome):
    drop = CONTROL_EXCLUDE.get(outcome, set())
    return [c for c in BASE_CONTROLS if c not in drop and c != outcome]


def add_exposure_group_year(dat, exposure):
    out = dat.copy()
    vals = pd.to_numeric(out[exposure], errors="coerce")
    if vals.dropna().nunique() <= 2:
        group = vals.fillna(0).gt(0).map({True: "high", False: "low"})
    else:
        med = vals.median()
        group = vals.fillna(vals.min()).gt(med).map({True: "high", False: "low"})
    out["exp_group_year"] = exposure + "_" + group.astype(str) + "_" + out["year"].astype(str)
    return out


def stat_from_est(est, term):
    tidy = get_tidy(est)
    row = tidy[tidy["term"].eq(term)]
    if row.empty:
        return None
    row = row.iloc[0]
    return {
        "coef": float(row["coef"]),
        "se_city_cluster": float(row["se"]),
        "t": float(row["t"]),
        "p": float(row["p"]),
        "sig": sig(float(row["p"])),
    }


def run_base(panel):
    rows = []
    for y in OUTCOMES:
        for exp in EXPOSURES:
            for spec, fe_rhs in [
                ("main_fe", "Stkcd_str + city_year + ind_year"),
                ("plus_exposure_group_year_fe", "Stkcd_str + city_year + ind_year + exp_group_year"),
            ]:
                dat = panel.copy()
                dat = add_exposure_group_year(dat, exp)
                x = f"x_{exp}"
                dat[x] = dat["pb_narrow"] * dat[exp]
                model_controls = controls_for(y)
                cols = [y, x, exp, "year", "Stkcd_str", "city_year", "ind_year", "exp_group_year", "CITY"] + model_controls
                dat = dat[cols].replace([np.inf, -np.inf], np.nan).dropna()
                if dat.empty or dat[y].nunique() < 2 or dat[x].nunique() < 2:
                    rows.append(
                        {
                            "outcome": y,
                            "exposure": exp,
                            "spec": spec,
                            "status": "skip_no_variation",
                            "n_input": int(len(dat)),
                        }
                    )
                    continue
                fml = f"{y} ~ {x} + {' + '.join(model_controls)} | {fe_rhs}"
                try:
                    est = pf.feols(fml, dat, vcov={"CRV1": "CITY"}, fixef_rm="singleton", lean=True)
                    stat = stat_from_est(est, x)
                    rows.append(
                        {
                            "outcome": y,
                            "exposure": exp,
                            "spec": spec,
                            "status": "ok" if stat else "term_missing",
                            **(stat or {}),
                            "n_input": int(len(dat)),
                            "treated_x_obs": int((dat[x] > 0).sum()),
                            "clusters_city": int(dat["CITY"].nunique()),
                            "y_mean": float(dat[y].mean()),
                            "weak_proxy": y in WEAK_PROXY_OUTCOMES,
                        }
                    )
                except Exception as exc:
                    rows.append(
                        {
                            "outcome": y,
                            "exposure": exp,
                            "spec": spec,
                            "status": f"error: {type(exc).__name__}: {str(exc)[:220]}",
                            "n_input": int(len(dat)),
                            "weak_proxy": y in WEAK_PROXY_OUTCOMES,
                        }
                    )
    return pd.DataFrame(rows)


def run_event_checks(panel, spec_list):
    rows = []
    coef_rows = []
    for outcome, exposure in spec_list:
        dat = add_event_terms(panel, exposure)
        terms = [f"es_{var}_{exposure}" for var, _, _ in EVENT_BINS]
        model_controls = controls_for(outcome)
        cols = [outcome, exposure, "Stkcd_str", "city_year", "ind_year", "CITY"] + model_controls + terms
        dat = dat[cols].replace([np.inf, -np.inf], np.nan).dropna()
        usable_terms = [t for t in terms if dat[t].nunique() > 1]
        if not usable_terms:
            rows.append({"outcome": outcome, "exposure": exposure, "status": "skip_no_event_variation", "n_input": len(dat)})
            continue
        fml = f"{outcome} ~ {' + '.join(usable_terms + model_controls)} | Stkcd_str + city_year + ind_year"
        try:
            est = pf.feols(fml, dat, vcov={"CRV1": "CITY"}, fixef_rm="singleton", lean=True)
            tidy = get_tidy(est)
            near_terms = [f"es_lead3_{exposure}", f"es_lead2_{exposure}"]
            all_leads = [f"es_lead4m_{exposure}", f"es_lead3_{exposure}", f"es_lead2_{exposure}"]
            near_chi2, near_p = wald_for_terms(est, near_terms)
            all_chi2, all_p = wald_for_terms(est, all_leads)
            for var, label, rel in EVENT_BINS:
                term = f"es_{var}_{exposure}"
                row = tidy[tidy["term"].eq(term)]
                if row.empty:
                    continue
                row = row.iloc[0]
                coef_rows.append(
                    {
                        "outcome": outcome,
                        "exposure": exposure,
                        "event": label,
                        "rel_year": rel,
                        "term": term,
                        "coef": float(row["coef"]),
                        "se_city_cluster": float(row["se"]),
                        "t": float(row["t"]),
                        "p": float(row["p"]),
                        "sig": sig(float(row["p"])),
                        "n_input": int(len(dat)),
                        "clusters_city": int(dat["CITY"].nunique()),
                    }
                )
            post0 = tidy[tidy["term"].eq(f"es_event0_{exposure}")]
            lag1 = tidy[tidy["term"].eq(f"es_lag1_{exposure}")]
            lag2 = tidy[tidy["term"].eq(f"es_lag2p_{exposure}")]
            rows.append(
                {
                    "outcome": outcome,
                    "exposure": exposure,
                    "status": "ok",
                    "n_input": int(len(dat)),
                    "clusters_city": int(dat["CITY"].nunique()),
                    "near_leads_chi2": near_chi2,
                    "near_leads_p": near_p,
                    "all_leads_chi2": all_chi2,
                    "all_leads_p": all_p,
                    "event0_coef": float(post0["coef"].iloc[0]) if not post0.empty else np.nan,
                    "event0_p": float(post0["p"].iloc[0]) if not post0.empty else np.nan,
                    "lag1_coef": float(lag1["coef"].iloc[0]) if not lag1.empty else np.nan,
                    "lag1_p": float(lag1["p"].iloc[0]) if not lag1.empty else np.nan,
                    "lag2p_coef": float(lag2["coef"].iloc[0]) if not lag2.empty else np.nan,
                    "lag2p_p": float(lag2["p"].iloc[0]) if not lag2.empty else np.nan,
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "outcome": outcome,
                    "exposure": exposure,
                    "status": f"error: {type(exc).__name__}: {str(exc)[:220]}",
                    "n_input": int(len(dat)),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(coef_rows)


def classify(base, events):
    main = base[(base["status"].eq("ok")) & (base["spec"].eq("main_fe"))].copy()
    strict = base[(base["status"].eq("ok")) & (base["spec"].eq("plus_exposure_group_year_fe"))].copy()
    merged = main.merge(
        strict[
            [
                "outcome",
                "exposure",
                "coef",
                "p",
                "sig",
                "n_input",
                "treated_x_obs",
                "clusters_city",
            ]
        ].rename(
            columns={
                "coef": "coef_strict",
                "p": "p_strict",
                "sig": "sig_strict",
                "n_input": "n_strict",
                "treated_x_obs": "treated_x_obs_strict",
                "clusters_city": "clusters_city_strict",
            }
        ),
        on=["outcome", "exposure"],
        how="left",
    )
    ev = events[events["status"].eq("ok")].copy()
    merged = merged.merge(
        ev[
            [
                "outcome",
                "exposure",
                "near_leads_p",
                "all_leads_p",
                "event0_coef",
                "event0_p",
                "lag1_coef",
                "lag1_p",
                "lag2p_coef",
                "lag2p_p",
            ]
        ],
        on=["outcome", "exposure"],
        how="left",
    )

    notes = []
    for row in merged.itertuples(index=False):
        signal = pd.notna(row.p) and row.p < 0.10
        strict_keeps = pd.notna(row.p_strict) and row.p_strict < 0.10 and np.sign(row.coef_strict) == np.sign(row.coef)
        pre_ok = pd.notna(row.near_leads_p) and row.near_leads_p >= 0.10
        if getattr(row, "weak_proxy", False):
            decision = "weak_proxy_only"
            note = "弱代理变量，不能直接写成减值或稳健性主证据。"
        elif signal and strict_keeps and pre_ok:
            decision = "candidate"
            note = "基准、ExposureGroup-Year FE 与近端前趋势同时过关，可进入下一轮。"
        elif signal and pre_ok:
            decision = "fragile_candidate"
            note = "基准和近端前趋势有信号，但严格趋势控制未稳住。"
        elif signal:
            decision = "base_only"
            note = "只有基准显著，不能直接写。"
        else:
            decision = "no_signal"
            note = "当前本地数据下没有可用信号。"
        notes.append((decision, note))
    if merged.empty:
        merged["decision"] = []
        merged["note"] = []
    else:
        merged["decision"] = [x[0] for x in notes]
        merged["note"] = [x[1] for x in notes]
    return merged


def write_report(base, events, event_coefs, decisions):
    ok = base[base["status"].eq("ok")].copy()
    ok["abs_t"] = ok["t"].abs()
    top = ok.sort_values(["p", "abs_t"], ascending=[True, False])
    dec = decisions.copy()
    dec["abs_t"] = dec["t"].abs()
    dec = dec.sort_values(["decision", "p", "abs_t"], ascending=[True, True, False])

    lines = []
    lines.append("# PB new Y probe: accounting recognition and weak write-down proxies\n\n")
    lines.append("Date: 2026-05-06\n\n")
    lines.append(
        "Design: `PB_narrow x pre-policy exposure`, firm FE, city-year FE, industry-year FE, city-clustered SE. "
        "Strict columns add `ExposureGroup x YearFE` to absorb nationwide high-exposure trends.\n\n"
    )
    lines.append("## What this run can and cannot test\n\n")
    lines.append("- Can test with current local data: loss recognition propensity, loss magnitude, accrual conservatism-style proxies.\n")
    lines.append("- Only weakly testable: decreases in intangible assets and development expenditure; these are not exact impairment-loss variables.\n")
    lines.append("- Not testable in the current local parquet set: corporate risk-disclosure specificity, debt-risk KAM text, exact asset/credit impairment loss.\n\n")

    lines.append("## Decision table\n\n")
    dec_cols = [
        "outcome",
        "exposure",
        "decision",
        "coef",
        "p",
        "sig",
        "coef_strict",
        "p_strict",
        "sig_strict",
        "near_leads_p",
        "lag1_coef",
        "lag1_p",
        "lag2p_coef",
        "lag2p_p",
        "n_input",
        "note",
    ]
    lines.append(dec[dec_cols].to_markdown(index=False, floatfmt=".4f"))
    lines.append("\n\n## Full base results\n\n")
    base_cols = [
        "outcome",
        "exposure",
        "spec",
        "coef",
        "se_city_cluster",
        "t",
        "p",
        "sig",
        "n_input",
        "treated_x_obs",
        "clusters_city",
        "weak_proxy",
    ]
    lines.append(top[base_cols].to_markdown(index=False, floatfmt=".4f"))
    lines.append("\n\n## Event-study summaries\n\n")
    event_cols = [
        "outcome",
        "exposure",
        "status",
        "n_input",
        "clusters_city",
        "near_leads_p",
        "all_leads_p",
        "event0_coef",
        "event0_p",
        "lag1_coef",
        "lag1_p",
        "lag2p_coef",
        "lag2p_p",
    ]
    if events.empty:
        lines.append("No event-study checks were run because no base signal crossed the screening threshold.\n")
    else:
        lines.append(events[event_cols].to_markdown(index=False, floatfmt=".4f"))
        lines.append("\n\nDetailed event coefficients are in `pb_new_y_probe_event_coefficients.csv`.\n")

    if not event_coefs.empty:
        lines.append("\n## Event coefficients\n\n")
        lines.append(
            event_coefs[
                ["outcome", "exposure", "event", "coef", "se_city_cluster", "p", "sig"]
            ].to_markdown(index=False, floatfmt=".4f")
        )
        lines.append("\n")

    path = OUT / "pb_new_y_probe_report.md"
    path.write_text("".join(lines), encoding="utf-8")
    return path


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    panel = load_panel().merge(load_financial_new_y(), on=["Stkcd", "year"], how="left")
    panel = panel[panel["year"].between(2015, 2023)].copy()

    base = run_base(panel)
    main_sig = base[
        (base["status"].eq("ok"))
        & (base["spec"].eq("main_fe"))
        & (base["p"].lt(0.10))
        & (~base["outcome"].isin(WEAK_PROXY_OUTCOMES))
    ].copy()
    main_sig["abs_t"] = main_sig["t"].abs()
    event_specs = list(
        main_sig.sort_values(["p", "abs_t"], ascending=[True, False])[
            ["outcome", "exposure"]
        ]
        .head(12)
        .itertuples(index=False, name=None)
    )
    events, event_coefs = run_event_checks(panel, event_specs)
    decisions = classify(base, events)

    base.to_csv(OUT / "pb_new_y_probe_base.csv", index=False)
    events.to_csv(OUT / "pb_new_y_probe_events.csv", index=False)
    event_coefs.to_csv(OUT / "pb_new_y_probe_event_coefficients.csv", index=False)
    decisions.to_csv(OUT / "pb_new_y_probe_decisions.csv", index=False)
    report_path = write_report(base, events, event_coefs, decisions)
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
