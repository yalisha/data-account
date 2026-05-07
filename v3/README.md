# v3 project layout

This folder keeps the special-issue pivot work separate from earlier v1/v2 manuscript scripts.

## Directories

- `src/stata/`: Stata do-files, including the current mismatch-X pilot.
- `src/python/`: Python helpers from earlier v3 exploratory runs.
- `docs/`: topic memos and literature/design notes.
- `docs/data_exchange_verifiable_disclosure/`: current canonical design docs for the data-exchange x verifiable-disclosure branch.
- `docs/research_memos/`: decision memos from the Y/X pivot process.
- `docs/handoff/`: coauthor-facing summaries and exported PDFs/HTML.
- `results/`: generated outputs only, grouped by file type/purpose.
- `results/data/`: constructed Stata-readable samples.
- `results/stata/`: Stata regression/result CSVs.
- `results/logs/`: Stata MCP logs.
- `bib/`: bibliography packs.

## Current X Decision

The current X is now defined as annual-report data-element narrative minus
verifiable hard capability:

```text
NarrHardGap = percentile(data narrative) - percentile(hard data capability)
```

The working design memo is:

```text
docs/research_memos/x_design_narrative_hard_capability_20260502.md
```

The data-supplement checklist for upgrading the hard-capability side is:

```text
docs/research_memos/data_supplement_checklist_for_x_20260503.md
```

The local inventory of newly downloaded hard-capability data is:

```text
docs/research_memos/local_hardcap_data_inventory_20260504.md
```

The first direct hard-capability build from small firm-year tables is:

```text
src/python/build_hardcap_direct_v1.py
src/stata/hardcap_direct_v1_screen.do
docs/research_memos/hardcap_direct_v1_pilot_report_20260504.md
results/data/hardcap_direct_v1_components.csv
results/data/hardcap_direct_v1_panel.dta
results/stata/hardcap_direct_v1_y_screen.csv
```

The hiring-refined hard-capability branch is:

```text
src/python/build_hiring_custom_v1.py
src/stata/hardcap_hiring_custom_v1_screen.do
src/stata/hardcap_hiring_custom_v1_compare.do
src/stata/hiring_pure_mismatch_v1_screen.do
src/stata/hiring_pure_mismatch_v1_compare.do
docs/research_memos/hiring_refined_hardcap_v1_report_20260504.md
results/data/hiring_custom_v1_for_stata.csv
results/data/hardcap_hiring_custom_v1_panel.dta
results/data/hiring_pure_mismatch_v1_panel.dta
results/stata/hardcap_hiring_custom_v1_y_screen.csv
results/stata/hiring_pure_mismatch_v1_y_screen.csv
```

Current judgment after the hiring run:

```text
Main X should remain the comprehensive hard-capability mismatch:
NarrHardGap_direct_v1 / NarrHardWashing_direct_v1 / NHWash_direct_v1_s.

Pure hiring mismatch is useful as a refined human-capital evidence branch:
NarrHireGap_title / NarrHireWash_title / NHireWash_title_s.
```

The current quasi-experiment screening branch is:

```text
src/stata/did_accounting_2024_v1.do
src/stata/did_data_ip_pilot_v1.do
src/stata/did_policy_first_stage_v1.do
src/stata/did_policy_first_stage_event_v1.do
src/stata/did_data_exchange_mismatch_ddd_v1.do
src/stata/did_data_exchange_mismatch_ddd_v2_pre2017.do
src/stata/did_data_exchange_mismatch_ddd_robust_v1.do
src/stata/did_data_exchange_mismatch_ddd_pretrend_slope_v1.do
docs/research_memos/quasi_experiment_did_iv_v1_report_20260504.md
docs/research_memos/data_exchange_mismatch_ddd_pilot_20260506.md
docs/research_memos/data_exchange_mismatch_ddd_robustness_20260506.md
docs/research_memos/research_design_data_exchange_verifiable_disclosure_20260506.md
docs/research_memos/data_exchange_verifiable_disclosure_pro_checks_20260506.md
results/stata/did_accounting_2024_v1_main.csv
results/stata/did_accounting_2024_v1_event.csv
results/stata/did_data_ip_pilot_v1_main.csv
results/stata/did_data_ip_pilot_v1_event.csv
results/stata/did_policy_first_stage_v1.csv
results/stata/did_policy_first_stage_event_v1.csv
results/stata/did_data_exchange_mismatch_ddd_v1_main.csv
results/stata/did_data_exchange_mismatch_ddd_v1_event.csv
results/stata/did_data_exchange_mismatch_ddd_v2_pre2017_main.csv
results/stata/did_data_exchange_mismatch_ddd_v2_pre2017_event.csv
results/stata/did_data_exchange_mismatch_ddd_robust_v1_static.csv
results/stata/did_data_exchange_mismatch_ddd_robust_v1_stacked_static.csv
results/stata/did_data_exchange_mismatch_ddd_robust_v1_stacked_pretrend_joint.csv
results/stata/did_data_exchange_mismatch_ddd_robust_v1_jackknife_summary.csv
results/stata/did_data_exchange_mismatch_ddd_pretrend_slope_v1_static.csv
results/stata/did_data_exchange_mismatch_ddd_pretrend_slope_v1_stacked_pretrend_joint.csv
results/stata/did_data_exchange_verifiable_disclosure_pro_v1.csv
```

Current quasi-experiment judgment:

```text
Do not use data-IP pilot as the main DID because pre-trends are visible.
Do not use these policy interactions as IVs because exclusion restrictions are weak.
The only usable quasi-experimental piece is a supplementary first-stage DID:
2024 accounting rule x pre-policy data-washing firms -> lower subsequent mismatch/washing.
The new data-exchange DDD branch is a stronger candidate main DID:
city data exchange x pre-2017 firm mismatch/washing -> lower later narrative-hard-capability mismatch.
Robustness checks show that static DDD survives city trends, province-year FE,
industry-year FE, not-yet/never stacked controls, and leave-one-city-out.
However, the mismatch outcomes still fail dynamic pretrend checks. A cleaner
main outcome is verifiable data-economy disclosure (`DU_kw`, `asset_trade_kw`,
and possibly `strict_at_kw`), with mismatch decline kept as supporting evidence.
Post-Pro checks further support `asset_trade_kw` as the safest lead outcome:
drop-2024, PreWash x 2024, and industry-2024 FE checks survive. The strongest
city-year FE + PreWash x year FE spec weakens broad `DU_kw`, but preserves
`asset_trade_kw`/`strict_at_kw` mainly for strict pre-washing firms.
The latest Y refinement removes institution-name terms from the verifiable
disclosure outcome. Current lead outcomes are now `asset_trade_noinst_kw` and
`strict_noinst_kw`, with `verif_noinst_kw`/`BookEntry` used for validation.
```

Current canonical docs for this branch:

```text
docs/data_exchange_verifiable_disclosure/README.md
docs/data_exchange_verifiable_disclosure/00_research_plan.md
docs/data_exchange_verifiable_disclosure/01_x_pre_data_washing_exposure.md
docs/data_exchange_verifiable_disclosure/02_y_verifiable_disclosure.md
docs/data_exchange_verifiable_disclosure/03_mechanisms_and_validation.md
```

Latest no-institution Y branch:

```text
src/python/build_verifiable_disclosure_y_noinst_v1.py
src/stata/did_data_exchange_verif_noinst_y_v1.do
docs/research_memos/verif_noinst_y_trial_20260506.md
results/data/verifiable_disclosure_y_noinst_v1.csv
results/data/did_data_exchange_verif_noinst_y_v1_panel.dta
results/stata/did_data_exchange_verif_noinst_y_v1_static.csv
results/stata/did_data_exchange_verif_noinst_y_v1_pretrend_joint.csv
results/stata/did_data_exchange_verif_noinst_y_v1_dataasset_validation.csv
```

The current runnable implementation is:

```text
src/stata/narrative_hard_capability_mismatch.do
```

Current main X candidates:

- `NarrHardGap_base`: continuous narrative-capability gap.
- `NarrHardWashing`: high narrative, low hard capability.
- `NarrHardHushing`: high hard capability, low narrative.
- `NarrHardVerified`: high narrative, high hard capability.

Data-asset booking is treated as accounting verification, not the main X,
because 2024 has only 22 positive booking observations after the project sample
merge.

## Earlier Pilot

The current runnable pilot is:

```text
src/stata/mismatch_x_y_screen_v0.do
```

It constructs:

- `Mismatch_sub`: annual-report data narrative rank minus substantive-expression rank.
- `DataWashing_sub`: high narrative, low substantive expression.
- `DataHushing_sub`: high substantive expression, low narrative.
- `Mismatch_potential`: narrative rank minus industry/location data-potential rank.
- `DataWashing_pot`: high narrative, low data potential.
- `DataHushing_pot`: high data potential, low narrative.
- `VerifiedData_pot`: high narrative and high data potential.

The `potential` version is only a coarse v0 proxy. It should be replaced or strengthened with patent, software copyright, data hiring, AI/data investment, or other hard capability data when available.

Current run report:

```text
docs/research_memos/mismatch_x_y_screen_v0_report.md
```

Data-asset booking verification pilot:

```text
src/python/extract_data_asset_for_stata.py
src/stata/data_asset_entry_mismatch_2024.do
docs/research_memos/data_asset_entry_mismatch_2024_pilot_report.md
```

Narrative-minus-hard-capability/accounting-verification pilot:

```text
src/stata/narrative_hard_capability_mismatch.do
docs/research_memos/narrative_hard_capability_mismatch_pilot_report.md
results/stata/narrative_hard_x_summary.csv
results/stata/narrative_hard_y_screen_panel.csv
results/stata/narrative_hard_y_screen_2024.csv
```

This pilot constructs `NarrHardGap_base`, `NarrHardWashing`,
`NarrHardHushing`, and 2024 accounting-augmented variants using
`DataAsset`/`BookEntry`. The accounting verification piece is currently a
supplement because 2024 has only 22 positive booking observations after the
project sample merge.
