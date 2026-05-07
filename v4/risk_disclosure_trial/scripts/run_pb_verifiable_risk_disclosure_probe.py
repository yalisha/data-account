import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyfixest as pf

PB_DIR = Path("/Users/mac/computerscience/0做完了/15会计研究/v4/pb_did_trial")
sys.path.insert(0, str(PB_DIR))

from run_pb_b_parallel_trends import EVENT_BINS, add_event_terms, get_tidy, load_panel, wald_for_terms
from run_pb_did_pilot import sig


ROOT = Path("/Users/mac/computerscience/0做完了/15会计研究")
OUT = ROOT / "v4" / "risk_disclosure_trial"
RISK_FEATURES = OUT / "risk_disclosure_verifiable_features_2015_2023.parquet"
TEXT_EXPOSURE_PRE = OUT / "np_liability_text_exposure_pre_2016_2018.csv"

OUTCOMES = [
    "risk_verifiability_index",
    "np_liability_verifiability_index",
    "risk_verif_debt_distress_per10k",
    "risk_verif_guarantee_pledge_per10k",
    "risk_verif_legal_process_per10k",
    "np_liability_control_right_per10k",
]

EXPOSURES = [
    "natural_pre",
    "failure_cost_pre",
    "high_lev_pre",
    "natural_highlev_pre",
    "ar_np_guarantee_pre",
    "ar_controller_pledge_pre",
    "ar_np_liability_text_pre",
    "ar_np_liability_index_pre",
]

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


def load_risk_panel():
    panel = load_panel()
    risk = pd.read_parquet(RISK_FEATURES)
    cols = ["Stkcd", "year"] + OUTCOMES + [
        "has_risk_text",
        "risk_headers",
        "risk_preview",
        "risk_chars",
    ]
    panel = panel.merge(risk[cols], on=["Stkcd", "year"], how="left")
    if TEXT_EXPOSURE_PRE.exists():
        text_exp = pd.read_csv(TEXT_EXPOSURE_PRE)
        keep = ["Stkcd"] + [c for c in EXPOSURES if c.startswith("ar_") and c in text_exp.columns]
        panel = panel.merge(text_exp[keep], on="Stkcd", how="left")
    for col in [c for c in EXPOSURES if c.startswith("ar_")]:
        if col not in panel.columns:
            panel[col] = np.nan
    for col in OUTCOMES:
        panel[col] = panel[col].fillna(0)
    panel["is_finance"] = panel["IndustryCodeC"].astype(str).str.startswith("J").astype(float)
    return panel


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
                dat = add_exposure_group_year(panel, exp)
                x = f"x_{exp}"
                dat[x] = dat["pb_narrow"] * dat[exp]
                controls = [c for c in BASE_CONTROLS if c != y]
                cols = [y, x, exp, "year", "Stkcd_str", "city_year", "ind_year", "exp_group_year", "CITY"] + controls
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
                fml = f"{y} ~ {x} + {' + '.join(controls)} | {fe_rhs}"
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
                            "expected_sign": 1,
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
                            "expected_sign": 1,
                        }
                    )
    return pd.DataFrame(rows)


def run_event_checks(panel, spec_list):
    rows = []
    coef_rows = []
    for outcome, exposure in spec_list:
        dat = add_event_terms(panel.copy(), exposure)
        terms = [f"es_{var}_{exposure}" for var, _, _ in EVENT_BINS]
        controls = [c for c in BASE_CONTROLS if c != outcome]
        cols = [outcome, exposure, "Stkcd_str", "city_year", "ind_year", "CITY"] + controls + terms
        dat = dat[cols].replace([np.inf, -np.inf], np.nan).dropna()
        usable_terms = [t for t in terms if dat[t].nunique() > 1]
        if not usable_terms:
            rows.append(
                {
                    "outcome": outcome,
                    "exposure": exposure,
                    "status": "skip_no_event_variation",
                    "n_input": len(dat),
                }
            )
            continue
        fml = f"{outcome} ~ {' + '.join(usable_terms + controls)} | Stkcd_str + city_year + ind_year"
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
                        "coef": float(row["coef"]),
                        "se_city_cluster": float(row["se"]),
                        "p": float(row["p"]),
                        "sig": sig(float(row["p"])),
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
                    "near_leads_p": near_p,
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
        strict[["outcome", "exposure", "coef", "p", "sig"]].rename(
            columns={"coef": "coef_strict", "p": "p_strict", "sig": "sig_strict"}
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
    decisions = []
    notes = []
    for row in merged.itertuples(index=False):
        sign_ok = pd.notna(row.coef) and row.coef > 0
        signal = pd.notna(row.p) and row.p < 0.10 and sign_ok
        strict_ok = pd.notna(row.p_strict) and row.p_strict < 0.10 and pd.notna(row.coef_strict) and row.coef_strict > 0
        pre_ok = pd.notna(row.near_leads_p) and row.near_leads_p >= 0.10
        if signal and strict_ok and pre_ok:
            decisions.append("candidate")
            notes.append("主规格、严格趋势控制和近端前趋势同时过关。")
        elif signal and pre_ok:
            decisions.append("fragile_candidate")
            notes.append("主规格和近端前趋势可用，但严格趋势控制未稳住。")
        elif signal:
            decisions.append("base_only")
            notes.append("只有主规格方向显著，不能直接写。")
        else:
            decisions.append("no_signal")
            notes.append("当前口径下没有理论方向一致的可用信号。")
    merged["decision"] = decisions
    merged["note"] = notes
    return merged


def write_report(base, events, event_coefs, decisions, panel):
    ok = base[base["status"].eq("ok")].copy()
    ok["abs_t"] = ok["t"].abs()
    top = ok.sort_values(["p", "abs_t"], ascending=[True, False])
    dec = decisions.copy()
    dec["decision_rank"] = dec["decision"].map(
        {"candidate": 1, "fragile_candidate": 2, "base_only": 3, "no_signal": 4}
    ).fillna(9)
    dec["abs_t"] = dec["t"].abs()
    dec = dec.sort_values(["decision_rank", "p", "abs_t"], ascending=[True, True, False])

    coverage = panel.groupby("year").agg(
        n=("Stkcd", "count"),
        has_risk=("has_risk_text", "mean"),
        verif=("risk_verifiability_index", "mean"),
        np_verif=("np_liability_verifiability_index", "mean"),
    )

    lines = []
    lines.append("# PB x exposure -> verifiable risk disclosure probe\n\n")
    lines.append("Date: 2026-05-06\n\n")
    lines.append(
        "Design: `PB_narrow x pre-policy exposure`, firm FE, city-year FE, industry-year FE, city-clustered SE. "
        "Strict specification adds `ExposureGroup x YearFE`.\n\n"
    )
    lines.append("## Coverage\n\n")
    lines.append(coverage.to_markdown(floatfmt=".4f"))
    lines.append("\n\n## Decision table\n\n")
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
    lines.append(events[event_cols].to_markdown(index=False, floatfmt=".4f"))
    if not event_coefs.empty:
        lines.append("\n\n## Event coefficients\n\n")
        lines.append(event_coefs.to_markdown(index=False, floatfmt=".4f"))
    path = OUT / "pb_verifiable_risk_disclosure_probe_report.md"
    path.write_text("".join(lines), encoding="utf-8")
    return path


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    panel = load_risk_panel()
    base = run_base(panel)
    main = base[(base["status"].eq("ok")) & (base["spec"].eq("main_fe"))].copy()
    main = main[main["p"].lt(0.10) & main["coef"].gt(0)].copy()
    main["abs_t"] = main["t"].abs()
    event_specs = list(
        main.sort_values(["p", "abs_t"], ascending=[True, False])[["outcome", "exposure"]]
        .head(20)
        .itertuples(index=False, name=None)
    )
    events, event_coefs = run_event_checks(panel, event_specs)
    decisions = classify(base, events)

    base.to_csv(OUT / "pb_verifiable_risk_disclosure_probe_base.csv", index=False)
    events.to_csv(OUT / "pb_verifiable_risk_disclosure_probe_events.csv", index=False)
    event_coefs.to_csv(OUT / "pb_verifiable_risk_disclosure_probe_event_coefficients.csv", index=False)
    decisions.to_csv(OUT / "pb_verifiable_risk_disclosure_probe_decisions.csv", index=False)
    path = write_report(base, events, event_coefs, decisions, panel)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
