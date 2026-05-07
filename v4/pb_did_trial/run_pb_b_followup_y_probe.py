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
    "private_pre",
    "natural_pre",
    "high_lev_pre",
    "private_highlev_pre",
    "natural_highlev_pre",
    "failure_cost_pre",
]

SUPPLY_OUTCOMES = ["SuppConc", "SCConc", "CustConc", "CustHHI"]
AUDIT_OUTCOMES = ["AuditFee", "NonStdAudit", "AuditDelayLn", "AuditorSwitch"]


def load_audit_outcomes():
    audit = pd.read_parquet(DATA / "audit.parquet")
    audit["Accper"] = pd.to_datetime(audit["Accper"], errors="coerce")
    audit["Annodt"] = pd.to_datetime(audit["Annodt"], errors="coerce")
    audit = audit[audit["Accper"].dt.month.eq(12)].copy()
    audit["year"] = audit["Accper"].dt.year
    audit["Stkcd"] = pd.to_numeric(audit["Stkcd"], errors="coerce").astype("Int64")
    audit = audit.sort_values(["Stkcd", "year", "Accper"]).drop_duplicates(["Stkcd", "year"], keep="last")

    fee = pd.to_numeric(audit["Tcost"], errors="coerce").fillna(pd.to_numeric(audit["Dcost"], errors="coerce"))
    audit["AuditFee_rebuilt"] = np.where(fee > 0, np.log(fee), np.nan)
    audit["NonStdAudit"] = audit["Audittyp"].ne("标准无保留意见").astype(float)
    delay = (audit["Annodt"] - audit["Accper"]).dt.days
    audit["AuditDelay"] = delay.where(delay.between(0, 365))
    audit["AuditDelayLn"] = np.log1p(audit["AuditDelay"])

    audit["audit_firm_id"] = audit["DadtunitID"].fillna(audit["Dadtunit"]).astype(str)
    audit = audit.sort_values(["Stkcd", "year"])
    prev = audit.groupby("Stkcd")["audit_firm_id"].shift(1)
    audit["AuditorSwitch"] = ((audit["audit_firm_id"].ne(prev)) & prev.notna()).astype(float)

    return audit[["Stkcd", "year", "AuditFee_rebuilt", "NonStdAudit", "AuditDelayLn", "AuditorSwitch"]]


def add_audit(panel):
    audit = load_audit_outcomes()
    out = panel.merge(audit, on=["Stkcd", "year"], how="left")
    out["AuditFee"] = out["AuditFee"].fillna(out["AuditFee_rebuilt"])
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
    outcomes = SUPPLY_OUTCOMES + AUDIT_OUTCOMES
    rows = []
    for y in outcomes:
        for exp in EXPOSURES:
            x = f"x_{exp}"
            dat = panel.copy()
            dat[x] = dat["pb_narrow"] * dat[exp]
            model_controls = [c for c in controls if c != y]
            cols = [y, x, "Stkcd_str", "city_year", "ind_year", "CITY"] + model_controls
            dat = dat[cols].replace([np.inf, -np.inf], np.nan).dropna()
            if dat.empty or dat[y].nunique() < 2 or dat[x].nunique() < 2:
                rows.append({"outcome": y, "exposure": exp, "status": "skip", "n_input": len(dat)})
                continue
            fml = f"{y} ~ {x} + {' + '.join(model_controls)} | Stkcd_str + city_year + ind_year"
            try:
                est = pf.feols(fml, dat, vcov={"CRV1": "CITY"}, fixef_rm="singleton", lean=True)
                stat = stat_from_est(est, x)
                rows.append(
                    {
                        "outcome": y,
                        "exposure": exp,
                        "status": "ok" if stat else "term_missing",
                        **(stat or {}),
                        "n_input": int(len(dat)),
                        "treated_x_obs": int((dat[x] > 0).sum()),
                        "clusters_city": int(dat["CITY"].nunique()),
                        "y_mean": float(dat[y].mean()),
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "outcome": y,
                        "exposure": exp,
                        "status": f"error: {type(exc).__name__}: {str(exc)[:180]}",
                        "n_input": int(len(dat)),
                    }
                )
    return pd.DataFrame(rows)


def run_event_checks(panel, spec_list):
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
    rows = []
    for outcome, exposure in spec_list:
        dat = add_event_terms(panel, exposure)
        terms = [f"es_{var}_{exposure}" for var, _, _ in EVENT_BINS]
        model_controls = [c for c in controls if c != outcome]
        cols = [outcome, exposure, "Stkcd_str", "city_year", "ind_year", "CITY"] + model_controls + terms
        dat = dat[cols].replace([np.inf, -np.inf], np.nan).dropna()
        usable_terms = [t for t in terms if dat[t].nunique() > 1]
        if not usable_terms:
            rows.append({"outcome": outcome, "exposure": exposure, "status": "skip", "n_input": len(dat)})
            continue
        fml = f"{outcome} ~ {' + '.join(usable_terms + model_controls)} | Stkcd_str + city_year + ind_year"
        try:
            est = pf.feols(fml, dat, vcov={"CRV1": "CITY"}, fixef_rm="singleton", lean=True)
            near_terms = [f"es_lead3_{exposure}", f"es_lead2_{exposure}"]
            near_chi2, near_p = wald_for_terms(est, near_terms)
            tidy = get_tidy(est)
            post2 = tidy[tidy["term"].eq(f"es_lag2p_{exposure}")]
            post0 = tidy[tidy["term"].eq(f"es_event0_{exposure}")]
            rows.append(
                {
                    "outcome": outcome,
                    "exposure": exposure,
                    "status": "ok",
                    "n_input": int(len(dat)),
                    "clusters_city": int(dat["CITY"].nunique()),
                    "near_leads_chi2": near_chi2,
                    "near_leads_p": near_p,
                    "event0_coef": float(post0["coef"].iloc[0]) if not post0.empty else np.nan,
                    "event0_p": float(post0["p"].iloc[0]) if not post0.empty else np.nan,
                    "lag2p_coef": float(post2["coef"].iloc[0]) if not post2.empty else np.nan,
                    "lag2p_p": float(post2["p"].iloc[0]) if not post2.empty else np.nan,
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "outcome": outcome,
                    "exposure": exposure,
                    "status": f"error: {type(exc).__name__}: {str(exc)[:180]}",
                    "n_input": int(len(dat)),
                }
            )
    return pd.DataFrame(rows)


def write_report(base, events):
    ok = base[base["status"].eq("ok")].copy()
    ok["abs_t"] = ok["t"].abs()
    top = ok.sort_values(["p", "abs_t"], ascending=[True, False])

    report = []
    report.append("# PB follow-up Y probe: supply chain and audit risk\n\n")
    report.append("Date: 2026-05-04\n\n")
    report.append("Design: same exposure DID/DDD as the B-route pilot: `PB_narrow x pre-policy debt exposure`, firm FE, city-year FE, industry-year FE, city-clustered SE.\n\n")
    report.append("## Base DDD results\n\n")
    cols = [
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
    report.append(top[cols].to_markdown(index=False, floatfmt=".4f"))
    report.append("\n\n## Event-study pretrend checks for strongest base signals\n\n")
    event_cols = [
        "outcome",
        "exposure",
        "status",
        "n_input",
        "clusters_city",
        "near_leads_p",
        "event0_coef",
        "event0_p",
        "lag2p_coef",
        "lag2p_p",
    ]
    report.append(events[event_cols].to_markdown(index=False, floatfmt=".4f"))
    report.append("\n")
    (OUT / "pb_b_followup_y_probe_report.md").write_text("".join(report), encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    panel = add_audit(load_panel())
    base = run_base(panel)
    ok = base[base["status"].eq("ok")].copy()
    ok = ok[ok["p"].lt(0.10)].copy()
    ok["abs_t"] = ok["t"].abs()
    event_specs = list(ok.sort_values(["p", "abs_t"], ascending=[True, False])[["outcome", "exposure"]].head(10).itertuples(index=False, name=None))
    events = run_event_checks(panel, event_specs)
    base.to_csv(OUT / "pb_b_followup_y_probe_base.csv", index=False)
    events.to_csv(OUT / "pb_b_followup_y_probe_events.csv", index=False)
    write_report(base, events)
    print("wrote", OUT / "pb_b_followup_y_probe_report.md")


if __name__ == "__main__":
    main()
