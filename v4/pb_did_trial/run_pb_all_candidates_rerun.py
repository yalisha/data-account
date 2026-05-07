from pathlib import Path

import numpy as np
import pandas as pd
import pyfixest as pf

from run_pb_b_followup_y_probe import add_audit
from run_pb_b_parallel_trends import EVENT_BINS, add_event_terms, get_tidy, load_panel, wald_for_terms
from run_pb_did_pilot import DATA, OUT, sig


EXPOSURES = [
    "private_pre",
    "natural_pre",
    "high_lev_pre",
    "private_highlev_pre",
    "natural_highlev_pre",
    "failure_cost_pre",
]

OUTCOME_GROUPS = {
    "A_positive_information": ["NCSKEW", "DUVOL", "absDA", "EarnQualAbs"],
    "A_audit_risk": ["AuditFee", "NonStdAudit", "AuditDelayLn", "AuditorSwitch"],
    "B_debt_pressure": ["short_liab_ratio", "Lev", "InvestIneff"],
    "B_supply_chain": ["CustConc", "SuppConc", "SCConc", "CustHHI"],
}

LOWER_IS_BETTER = {"NCSKEW", "DUVOL", "absDA", "EarnQualAbs", "InvestIneff", "Lev"}


def add_extra_outcomes(panel):
    out = add_audit(panel)
    eq = pd.read_parquet(DATA / "earnings_quality_dd.parquet", columns=["Stkcd", "year", "EarnQualAbs"])
    out = out.merge(eq, on=["Stkcd", "year"], how="left")
    return out


def controls_for(outcome):
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
    return [c for c in controls if c != outcome]


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
    outcome_to_group = {y: g for g, ys in OUTCOME_GROUPS.items() for y in ys}
    outcomes = [y for ys in OUTCOME_GROUPS.values() for y in ys]
    for outcome in outcomes:
        for exposure in EXPOSURES:
            dat = panel.copy()
            x = f"x_{exposure}"
            dat[x] = dat["pb_narrow"] * dat[exposure]
            model_controls = controls_for(outcome)
            cols = [outcome, x, "Stkcd_str", "city_year", "ind_year", "CITY"] + model_controls
            dat = dat[cols].replace([np.inf, -np.inf], np.nan).dropna()
            if dat.empty or dat[x].nunique() < 2 or dat[outcome].nunique() < 2:
                rows.append(
                    {
                        "group": outcome_to_group[outcome],
                        "outcome": outcome,
                        "exposure": exposure,
                        "status": "skip_no_variation",
                        "n_input": int(len(dat)),
                    }
                )
                continue
            fml = f"{outcome} ~ {x} + {' + '.join(model_controls)} | Stkcd_str + city_year + ind_year"
            try:
                est = pf.feols(fml, dat, vcov={"CRV1": "CITY"}, fixef_rm="singleton", lean=True)
                stat = stat_from_est(est, x)
                rows.append(
                    {
                        "group": outcome_to_group[outcome],
                        "outcome": outcome,
                        "exposure": exposure,
                        "status": "ok" if stat else "term_missing",
                        **(stat or {}),
                        "n_input": int(len(dat)),
                        "treated_x_obs": int((dat[x] > 0).sum()),
                        "clusters_city": int(dat["CITY"].nunique()),
                        "y_mean": float(dat[outcome].mean()),
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "group": outcome_to_group[outcome],
                        "outcome": outcome,
                        "exposure": exposure,
                        "status": f"error: {type(exc).__name__}: {str(exc)[:220]}",
                        "n_input": int(len(dat)),
                    }
                )
    return pd.DataFrame(rows)


def choose_event_specs(base):
    ok = base[base["status"].eq("ok")].copy()
    ok["abs_t"] = ok["t"].abs()

    forced = pd.DataFrame(
        [
            {"outcome": "short_liab_ratio", "exposure": "high_lev_pre"},
            {"outcome": "short_liab_ratio", "exposure": "private_highlev_pre"},
            {"outcome": "short_liab_ratio", "exposure": "natural_highlev_pre"},
            {"outcome": "NCSKEW", "exposure": "natural_pre"},
            {"outcome": "DUVOL", "exposure": "natural_pre"},
            {"outcome": "NCSKEW", "exposure": "failure_cost_pre"},
            {"outcome": "DUVOL", "exposure": "failure_cost_pre"},
            {"outcome": "absDA", "exposure": "natural_pre"},
            {"outcome": "EarnQualAbs", "exposure": "natural_pre"},
            {"outcome": "AuditFee", "exposure": "failure_cost_pre"},
            {"outcome": "CustConc", "exposure": "private_pre"},
            {"outcome": "CustConc", "exposure": "high_lev_pre"},
        ]
    )
    significant = ok[ok["p"].lt(0.10)].sort_values(["p", "abs_t"], ascending=[True, False])[
        ["outcome", "exposure"]
    ]
    specs = pd.concat([forced, significant], ignore_index=True).drop_duplicates()
    return list(specs.itertuples(index=False, name=None))


def run_events(panel, specs):
    rows = []
    coef_rows = []
    for outcome, exposure in specs:
        dat = add_event_terms(panel, exposure)
        event_terms = [f"es_{var}_{exposure}" for var, _, _ in EVENT_BINS]
        model_controls = controls_for(outcome)
        cols = [outcome, exposure, "Stkcd_str", "city_year", "ind_year", "CITY"] + model_controls + event_terms
        dat = dat[cols].replace([np.inf, -np.inf], np.nan).dropna()
        usable_terms = [t for t in event_terms if dat[t].nunique() > 1]
        if not usable_terms:
            rows.append({"outcome": outcome, "exposure": exposure, "status": "skip", "n_input": int(len(dat))})
            continue
        fml = f"{outcome} ~ {' + '.join(usable_terms + model_controls)} | Stkcd_str + city_year + ind_year"
        try:
            est = pf.feols(fml, dat, vcov={"CRV1": "CITY"}, fixef_rm="singleton", lean=True)
            tidy = get_tidy(est)
            near_terms = [f"es_lead3_{exposure}", f"es_lead2_{exposure}"]
            all_leads = [f"es_lead4m_{exposure}", f"es_lead3_{exposure}", f"es_lead2_{exposure}"]
            near_chi2, near_p = wald_for_terms(est, near_terms)
            all_chi2, all_p = wald_for_terms(est, all_leads)

            post_info = {}
            for var, label, _ in EVENT_BINS:
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
                        "term": term,
                        "coef": float(row["coef"]),
                        "se_city_cluster": float(row["se"]),
                        "t": float(row["t"]),
                        "p": float(row["p"]),
                        "sig": sig(float(row["p"])),
                        "ci_low": float(row["ci_low"]),
                        "ci_high": float(row["ci_high"]),
                        "n_input": int(len(dat)),
                        "clusters_city": int(dat["CITY"].nunique()),
                    }
                )
                if label in ["0", "+1", ">=+2"]:
                    post_info[f"{label}_coef"] = float(row["coef"])
                    post_info[f"{label}_p"] = float(row["p"])

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
                    **post_info,
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
    merged = base.merge(events, on=["outcome", "exposure"], how="left", suffixes=("", "_event"))
    rows = []
    for _, r in merged.iterrows():
        if r.get("status") != "ok":
            verdict = "not_estimated"
        elif not np.isfinite(r.get("p", np.nan)) or r["p"] >= 0.10:
            verdict = "weak_base"
        elif r.get("status_event") != "ok":
            verdict = "base_only_no_event"
        elif np.isfinite(r.get("near_leads_p", np.nan)) and r["near_leads_p"] < 0.10:
            verdict = "bad_pretrend"
        else:
            post_ps = [r.get("0_p", np.nan), r.get("+1_p", np.nan), r.get(">=+2_p", np.nan)]
            post_coefs = [r.get("0_coef", np.nan), r.get("+1_coef", np.nan), r.get(">=+2_coef", np.nan)]
            post_sig = [
                np.isfinite(p) and p < 0.10 and np.isfinite(c) and np.sign(c) == np.sign(r["coef"])
                for p, c in zip(post_ps, post_coefs)
            ]
            verdict = "dynamic_pass" if any(post_sig) else "pooled_only"

        if r["outcome"] in LOWER_IS_BETTER:
            direction = "good_if_negative"
        elif r["outcome"] in ["AuditFee", "NonStdAudit", "AuditDelayLn", "AuditorSwitch"]:
            direction = "risk_pricing_if_positive"
        else:
            direction = "ambiguous"

        rows.append(
            {
                "group": r.get("group"),
                "outcome": r["outcome"],
                "exposure": r["exposure"],
                "coef": r.get("coef"),
                "p": r.get("p"),
                "near_leads_p": r.get("near_leads_p"),
                "post0_p": r.get("0_p"),
                "post1_p": r.get("+1_p"),
                "post2p_p": r.get(">=+2_p"),
                "direction": direction,
                "verdict": verdict,
            }
        )
    return pd.DataFrame(rows)


def write_report(base, events, decisions):
    ok = base[base["status"].eq("ok")].copy()
    ok["abs_t"] = ok["t"].abs()
    top_base = ok.sort_values(["p", "abs_t"], ascending=[True, False]).head(40)
    top_dec = decisions[decisions["verdict"].ne("weak_base")].copy()
    top_dec["abs_coef"] = top_dec["coef"].abs()
    top_dec = top_dec.sort_values(["verdict", "p", "abs_coef"], ascending=[True, True, False])

    report = []
    report.append("# PB all-candidate rerun\n\n")
    report.append("Date: 2026-05-04\n\n")
    report.append("Design: `PB_narrow x pre-policy exposure`, firm FE, city-year FE, industry-year FE, city-clustered SE. Event-study base year is t=-1.\n\n")
    report.append("## Top base DDD signals\n\n")
    cols = [
        "group",
        "outcome",
        "exposure",
        "coef",
        "se_city_cluster",
        "t",
        "p",
        "sig",
        "n_input",
        "treated_x_obs",
        "clusters_city",
    ]
    report.append(top_base[cols].to_markdown(index=False, floatfmt=".4f"))
    report.append("\n\n## Decision table after event checks\n\n")
    dec_cols = [
        "group",
        "outcome",
        "exposure",
        "coef",
        "p",
        "near_leads_p",
        "post0_p",
        "post1_p",
        "post2p_p",
        "direction",
        "verdict",
    ]
    report.append(top_dec[dec_cols].head(60).to_markdown(index=False, floatfmt=".4f"))
    report.append("\n\n## Verdict labels\n")
    report.append("- `dynamic_pass`: base DDD significant, near leads clean, at least one post-event coefficient significant with same sign.\n")
    report.append("- `pooled_only`: base DDD significant and near leads clean, but event-study post coefficients are weak.\n")
    report.append("- `bad_pretrend`: base DDD significant but t=-3/-2 lead test fails.\n")
    report.append("- `weak_base`: base DDD not significant at 10%.\n")
    (OUT / "pb_all_candidates_rerun_report.md").write_text("".join(report), encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    panel = add_extra_outcomes(load_panel())
    base = run_base(panel)
    specs = choose_event_specs(base)
    events, event_coefs = run_events(panel, specs)
    decisions = classify(base, events)

    base.to_csv(OUT / "pb_all_candidates_base.csv", index=False)
    events.to_csv(OUT / "pb_all_candidates_events.csv", index=False)
    event_coefs.to_csv(OUT / "pb_all_candidates_event_coefficients.csv", index=False)
    decisions.to_csv(OUT / "pb_all_candidates_decisions.csv", index=False)
    write_report(base, events, decisions)
    print("wrote", OUT / "pb_all_candidates_rerun_report.md")


if __name__ == "__main__":
    main()
