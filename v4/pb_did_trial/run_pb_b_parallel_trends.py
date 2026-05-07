from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyfixest as pf

from run_pb_did_pilot import (
    DATA,
    OUT,
    STATA,
    add_pre_exposures,
    build_policy_flags,
    clean_city,
    load_city_panel,
    load_controller_panel,
    load_short_debt_panel,
    sig,
)


EVENT_BINS = [
    ("lead4m", "<=-4", -4),
    ("lead3", "-3", -3),
    ("lead2", "-2", -2),
    ("event0", "0", 0),
    ("lag1", "+1", 1),
    ("lag2p", ">=+2", 2),
]

MAIN_SPECS = [
    ("short_liab_ratio", "high_lev_pre"),
    ("short_liab_ratio", "private_highlev_pre"),
    ("short_liab_ratio", "natural_highlev_pre"),
    ("Lev", "high_lev_pre"),
    ("Lev", "private_highlev_pre"),
    ("Lev", "natural_highlev_pre"),
    ("InvestIneff", "natural_pre"),
    ("InvestIneff", "failure_cost_pre"),
]


def get_tidy(est):
    tidy = est.tidy().reset_index()
    tidy = tidy.rename(
        columns={
            "Coefficient": "term",
            "Estimate": "coef",
            "Std. Error": "se",
            "t value": "t",
            "Pr(>|t|)": "p",
            "2.5%": "ci_low",
            "97.5%": "ci_high",
        }
    )
    return tidy


def load_panel():
    panel = pd.read_stata(STATA / "reg_sample_v18.dta", convert_categoricals=False)
    panel = panel[panel["year"].between(2015, 2023)].copy()

    panel = panel.merge(load_city_panel(), on=["Stkcd", "year"], how="left")
    panel = panel.merge(load_controller_panel(), on=["Stkcd", "year"], how="left")
    panel = panel.merge(load_short_debt_panel(), on=["Stkcd", "year"], how="left")
    panel = build_policy_flags(panel)
    panel = add_pre_exposures(panel)

    panel["Stkcd_str"] = panel["Stkcd"].astype(str)
    panel["CITY"] = panel["CITY"].map(clean_city)
    panel["city_year"] = panel["CITY"].astype(str) + "_" + panel["year"].astype(str)
    panel["ind_year"] = panel["Ind2"].astype(str) + "_" + panel["year"].astype(str)
    panel = panel[panel["CITY"].notna()].copy()
    panel["event_time"] = panel["year"] - panel["pb_start_narrow"]
    return panel


def add_event_terms(df, exposure):
    out = df.copy()
    treated_city = out["pb_start_narrow"].notna()
    for var, _, rel in EVENT_BINS:
        if var == "lead4m":
            flag = treated_city & (out["event_time"] <= -4)
        elif var == "lag2p":
            flag = treated_city & (out["event_time"] >= 2)
        else:
            flag = treated_city & (out["event_time"] == rel)
        out[f"es_{var}_{exposure}"] = flag.astype(float) * out[exposure].fillna(0)
    return out


def wald_for_terms(est, terms):
    coef_index = list(est.coef().index)
    keep = [t for t in terms if t in coef_index]
    if not keep:
        return np.nan, np.nan
    r = np.zeros((len(keep), len(coef_index)))
    for i, term in enumerate(keep):
        r[i, coef_index.index(term)] = 1.0
    try:
        result = est.wald_test(R=r, q=np.zeros(len(keep)), distribution="chi2")
        return float(result.get("statistic", np.nan)), float(result.get("pvalue", np.nan))
    except Exception:
        return np.nan, np.nan


def run_event_study(panel):
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
    coef_rows = []
    summary_rows = []

    for outcome, exposure in MAIN_SPECS:
        dat = add_event_terms(panel, exposure)
        event_terms = [f"es_{var}_{exposure}" for var, _, _ in EVENT_BINS]
        model_controls = [c for c in controls if c != outcome]
        cols = [
            outcome,
            exposure,
            "Stkcd_str",
            "city_year",
            "ind_year",
            "CITY",
            "event_time",
            "pb_start_narrow",
        ] + model_controls + event_terms
        dat = dat[cols].replace([np.inf, -np.inf], np.nan).dropna(subset=[outcome, "CITY", "Stkcd_str"])
        dat = dat.dropna(subset=model_controls)
        dat = dat[dat[exposure].notna()].copy()

        usable_terms = [t for t in event_terms if dat[t].nunique() > 1]
        rhs = " + ".join(usable_terms + model_controls)
        fml = f"{outcome} ~ {rhs} | Stkcd_str + city_year + ind_year"

        try:
            est = pf.feols(fml, dat, vcov={"CRV1": "CITY"}, fixef_rm="singleton", lean=True)
            tidy = get_tidy(est)
            for var, label, rel in EVENT_BINS:
                term = f"es_{var}_{exposure}"
                row = tidy[tidy["term"].eq(term)]
                if row.empty:
                    coef_rows.append(
                        {
                            "outcome": outcome,
                            "exposure": exposure,
                            "event": label,
                            "rel_year": rel,
                            "term": term,
                            "status": "term_missing",
                        }
                    )
                    continue
                row = row.iloc[0]
                coef_rows.append(
                    {
                        "outcome": outcome,
                        "exposure": exposure,
                        "event": label,
                        "rel_year": rel,
                        "term": term,
                        "status": "ok",
                        "coef": float(row["coef"]),
                        "se_city_cluster": float(row["se"]),
                        "t": float(row["t"]),
                        "p": float(row["p"]),
                        "ci_low": float(row["ci_low"]),
                        "ci_high": float(row["ci_high"]),
                        "sig": sig(float(row["p"])),
                        "n_input": int(len(dat)),
                        "treated_exposure_obs": int((dat[term] != 0).sum()),
                        "clusters_city": int(dat["CITY"].nunique()),
                    }
                )

            near_leads = [f"es_lead3_{exposure}", f"es_lead2_{exposure}"]
            all_leads = [f"es_lead4m_{exposure}", f"es_lead3_{exposure}", f"es_lead2_{exposure}"]
            near_f, near_p = wald_for_terms(est, near_leads)
            all_f, all_p = wald_for_terms(est, all_leads)
            summary_rows.append(
                {
                    "outcome": outcome,
                    "exposure": exposure,
                    "status": "ok",
                    "n_input": int(len(dat)),
                    "clusters_city": int(dat["CITY"].nunique()),
                    "near_leads_chi2": near_f,
                    "near_leads_p": near_p,
                    "all_leads_chi2": all_f,
                    "all_leads_p": all_p,
                }
            )
        except Exception as exc:
            summary_rows.append(
                {
                    "outcome": outcome,
                    "exposure": exposure,
                    "status": f"error: {type(exc).__name__}: {str(exc)[:220]}",
                    "n_input": int(len(dat)),
                }
            )

    return pd.DataFrame(coef_rows), pd.DataFrame(summary_rows)


def write_plots(coefs):
    ok = coefs[coefs["status"].eq("ok")].copy()
    if ok.empty:
        return None

    spec_labels = ok[["outcome", "exposure"]].drop_duplicates().to_records(index=False)
    n = len(spec_labels)
    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(11, 14), sharex=True)
    axes = axes.flatten()

    order = ["<=-4", "-3", "-2", "0", "+1", ">=+2"]
    x_map = {label: idx for idx, label in enumerate(order)}
    for ax, (outcome, exposure) in zip(axes, spec_labels):
        sub = ok[(ok["outcome"].eq(outcome)) & (ok["exposure"].eq(exposure))].copy()
        sub["x"] = sub["event"].map(x_map)
        sub = sub.sort_values("x")
        ax.axhline(0, color="#666666", linewidth=0.8)
        ax.axvline(2.5, color="#999999", linewidth=0.8, linestyle="--")
        ax.errorbar(
            sub["x"],
            sub["coef"],
            yerr=[sub["coef"] - sub["ci_low"], sub["ci_high"] - sub["coef"]],
            fmt="o-",
            color="#1f77b4",
            ecolor="#9ecae1",
            capsize=3,
            linewidth=1.2,
        )
        ax.set_title(f"{outcome} x {exposure}", fontsize=10)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order)
        ax.grid(axis="y", alpha=0.25)

    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("PB narrow event-study interactions; base period = -1", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    path = OUT / "pb_b_event_study_plot.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_report(coefs, summary, plot_path):
    ok = coefs[coefs["status"].eq("ok")].copy()
    lead_view = ok[ok["event"].isin(["<=-4", "-3", "-2"])].copy()
    post_view = ok[ok["event"].isin(["0", "+1", ">=+2"])].copy()

    report = []
    report.append("# PB B-route parallel-trends screen\n\n")
    report.append("Date: 2026-05-04\n\n")
    report.append("Design: event-study version of `PB_narrow x pre-policy exposure`, with firm FE, city-year FE, industry-year FE, city-clustered standard errors. Base event year is -1.\n\n")
    report.append("Interpretation: the key pre-trend check is whether event years -3 and -2 are jointly insignificant. The <=-4 bin is reported as a far-pre-period diagnostic.\n\n")
    report.append("## Joint lead tests\n\n")
    summary_cols = [
        "outcome",
        "exposure",
        "status",
        "n_input",
        "clusters_city",
        "near_leads_chi2",
        "near_leads_p",
        "all_leads_chi2",
        "all_leads_p",
    ]
    report.append(summary[summary_cols].to_markdown(index=False, floatfmt=".4f"))
    report.append("\n\n## Lead coefficients\n\n")
    lead_cols = [
        "outcome",
        "exposure",
        "event",
        "coef",
        "se_city_cluster",
        "t",
        "p",
        "sig",
        "treated_exposure_obs",
    ]
    report.append(lead_view[lead_cols].to_markdown(index=False, floatfmt=".4f"))
    report.append("\n\n## Post coefficients\n\n")
    report.append(post_view[lead_cols].to_markdown(index=False, floatfmt=".4f"))
    if plot_path:
        report.append(f"\n\nPlot: `{plot_path}`\n")
    (OUT / "pb_b_parallel_trends_report.md").write_text("".join(report), encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    panel = load_panel()
    coefs, summary = run_event_study(panel)
    coefs.to_csv(OUT / "pb_b_event_study_coefficients.csv", index=False)
    summary.to_csv(OUT / "pb_b_parallel_trends_summary.csv", index=False)
    plot_path = write_plots(coefs)
    write_report(coefs, summary, plot_path)
    print("wrote", OUT / "pb_b_parallel_trends_report.md")


if __name__ == "__main__":
    main()
