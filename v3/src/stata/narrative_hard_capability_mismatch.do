clear all
set more off
capture log close

global V3 "/Users/mac/computerscience/0做完了/15会计研究/v3"
global V1 "/Users/mac/computerscience/0做完了/15会计研究/v1"

log using "$V3/results/logs/narrative_hard_capability_mismatch.log", replace text

display "=== Annual-report data narrative minus hard capability/accounting verification ==="
display "Input 1: v1/data_stata/reg_sample_v18.dta"
display "Input 2: v3/results/data/data_asset_from_panel.csv"
display "Panel spec: Y_t = L.X + controls + firm FE + year FE, cluster(IndYear_num)"
display "2024 spec: Y_2024 = X_2024 + controls + industry FE, robust SE"

capture which reghdfe
if _rc {
    display as error "reghdfe not installed"
    exit 199
}

* -----------------------------
* 0. Merge data-asset booking data
* -----------------------------

tempfile asset
import delimited using "$V3/results/data/data_asset_from_panel.csv", clear varnames(1)
capture rename stkcd_num Stkcd_num
capture rename year_num year_num
capture rename dataasset DataAsset
destring Stkcd_num year_num DataAsset, replace force
duplicates drop Stkcd_num year_num, force
save `asset', replace

use "$V1/data_stata/reg_sample_v18.dta", clear
merge 1:1 Stkcd_num year_num using `asset', keep(master match) nogen

sort Stkcd_num year_num
tsset Stkcd_num year_num

capture drop DataAsset0 DataAsset_ln BookEntry
gen DataAsset0 = DataAsset
replace DataAsset0 = 0 if missing(DataAsset0)
gen DataAsset_ln = ln(1 + DataAsset0)
gen BookEntry = (DataAsset0 > 0) if year_num == 2024

* -----------------------------
* 1. Construct full-panel hard-capability base score
* -----------------------------

capture drop lnLQ z_lnLQ
gen lnLQ = ln(1 + LQ) if LQ >= 0
egen z_lnLQ = std(lnLQ)

foreach v in HighTech StrategicEmerging DigEconCore IndustryCluster {
    capture drop z_`v'
    egen z_`v' = std(`v')
}

capture drop HardBaseRaw
egen HardBaseRaw = rowmean(z_HighTech z_StrategicEmerging z_DigEconCore z_IndustryCluster z_lnLQ)

* rank(...), field assigns smaller ranks to larger values in this Stata setup.
* Rank negative values so percentiles increase with the original construct.
capture drop neg_DU_kw neg_HardBaseRaw n_disc_y n_hard_y rank_disc_y rank_hard_y disc_pct_y hard_base_pct_y
gen neg_DU_kw = -DU_kw
gen neg_HardBaseRaw = -HardBaseRaw
bys year_num: egen n_disc_y = count(DU_kw)
bys year_num: egen n_hard_y = count(HardBaseRaw)
bys year_num: egen rank_disc_y = rank(neg_DU_kw) if !missing(neg_DU_kw), field
bys year_num: egen rank_hard_y = rank(neg_HardBaseRaw) if !missing(neg_HardBaseRaw), field
gen disc_pct_y = (rank_disc_y - 1) / (n_disc_y - 1) if n_disc_y > 1
gen hard_base_pct_y = (rank_hard_y - 1) / (n_hard_y - 1) if n_hard_y > 1

capture drop NarrHardGap_base NarrHardWashing NarrHardHushing NarrHardVerified
capture drop NarrHardWashing_strict NarrHardHushing_strict NarrHardVerified_strict
gen NarrHardGap_base = disc_pct_y - hard_base_pct_y
gen NarrHardWashing = (disc_pct_y >= .50 & hard_base_pct_y < .50) if !missing(disc_pct_y, hard_base_pct_y)
gen NarrHardHushing = (hard_base_pct_y >= .50 & disc_pct_y < .50) if !missing(disc_pct_y, hard_base_pct_y)
gen NarrHardVerified = (disc_pct_y >= .50 & hard_base_pct_y >= .50) if !missing(disc_pct_y, hard_base_pct_y)
gen NarrHardWashing_strict = (disc_pct_y >= .75 & hard_base_pct_y <= .25) if !missing(disc_pct_y, hard_base_pct_y)
gen NarrHardHushing_strict = (hard_base_pct_y >= .75 & disc_pct_y <= .25) if !missing(disc_pct_y, hard_base_pct_y)
gen NarrHardVerified_strict = (disc_pct_y >= .75 & hard_base_pct_y >= .75) if !missing(disc_pct_y, hard_base_pct_y)

* -----------------------------
* 2. Construct 2024 accounting-augmented hard score
* -----------------------------

capture drop z_HardBaseRaw_2024 z_DataAsset_ln_2024 z_BookEntry_2024 HardAcctRaw
egen z_HardBaseRaw_2024 = std(HardBaseRaw) if year_num == 2024
egen z_DataAsset_ln_2024 = std(DataAsset_ln) if year_num == 2024
egen z_BookEntry_2024 = std(BookEntry) if year_num == 2024
egen HardAcctRaw = rowmean(z_HardBaseRaw_2024 z_DataAsset_ln_2024 z_BookEntry_2024) if year_num == 2024

* Same direction correction for the 2024 accounting-augmented hard score.
capture drop neg_HardAcctRaw n_acct_2024 rank_acct_2024 hard_acct_pct_2024 disc_pct_2024
gen neg_HardAcctRaw = -HardAcctRaw
bys year_num: egen n_acct_2024 = count(HardAcctRaw)
bys year_num: egen rank_acct_2024 = rank(neg_HardAcctRaw) if year_num == 2024 & !missing(neg_HardAcctRaw), field
gen hard_acct_pct_2024 = (rank_acct_2024 - 1) / (n_acct_2024 - 1) if year_num == 2024 & n_acct_2024 > 1
gen disc_pct_2024 = disc_pct_y if year_num == 2024

capture drop NarrAcctHardGap NarrAcctHardWashing NarrAcctHardHushing NarrAcctHardVerified
capture drop NarrAcctHardWashing_strict NarrAcctHardHushing_strict NarrAcctHardVerified_strict
gen NarrAcctHardGap = disc_pct_2024 - hard_acct_pct_2024 if year_num == 2024
gen NarrAcctHardWashing = (disc_pct_2024 >= .50 & hard_acct_pct_2024 < .50) if year_num == 2024 & !missing(disc_pct_2024, hard_acct_pct_2024)
gen NarrAcctHardHushing = (hard_acct_pct_2024 >= .50 & disc_pct_2024 < .50) if year_num == 2024 & !missing(disc_pct_2024, hard_acct_pct_2024)
gen NarrAcctHardVerified = (disc_pct_2024 >= .50 & hard_acct_pct_2024 >= .50) if year_num == 2024 & !missing(disc_pct_2024, hard_acct_pct_2024)
gen NarrAcctHardWashing_strict = (disc_pct_2024 >= .75 & hard_acct_pct_2024 <= .25) if year_num == 2024 & !missing(disc_pct_2024, hard_acct_pct_2024)
gen NarrAcctHardHushing_strict = (hard_acct_pct_2024 >= .75 & disc_pct_2024 <= .25) if year_num == 2024 & !missing(disc_pct_2024, hard_acct_pct_2024)
gen NarrAcctHardVerified_strict = (disc_pct_2024 >= .75 & hard_acct_pct_2024 >= .75) if year_num == 2024 & !missing(disc_pct_2024, hard_acct_pct_2024)

label variable HardBaseRaw "Hard data capability/potential score: industry/location proxies"
label variable hard_base_pct_y "Within-year percentile of hard data capability/potential score"
label variable NarrHardGap_base "Percentile(data narrative) - percentile(hard capability), panel"
label variable NarrHardWashing "High data narrative, low hard capability"
label variable NarrHardHushing "High hard capability, low data narrative"
label variable NarrHardVerified "High data narrative and high hard capability"
label variable HardAcctRaw "2024 hard score augmented by data asset booking"
label variable hard_acct_pct_2024 "2024 percentile of hard score augmented by data asset booking"
label variable NarrAcctHardGap "2024 percentile(data narrative) - percentile(accounting-augmented hard score)"
label variable NarrAcctHardWashing "2024 high data narrative, low accounting-augmented hard score"
label variable NarrAcctHardHushing "2024 high accounting-augmented hard score, low data narrative"
label variable NarrAcctHardVerified "2024 high data narrative and high accounting-augmented hard score"

save "$V3/results/data/narrative_hard_capability_mismatch_panel.dta", replace

* -----------------------------
* 3. X construction summaries
* -----------------------------

tempfile xsummary
tempname xpost
postfile `xpost' str40 x_name str12 sample double N mean sd p25 median p75 positives using `xsummary', replace

foreach x in HardBaseRaw hard_base_pct_y NarrHardGap_base NarrHardWashing NarrHardHushing NarrHardVerified NarrHardWashing_strict NarrHardHushing_strict NarrHardVerified_strict {
    quietly summarize `x', detail
    local x_N = r(N)
    local x_mean = r(mean)
    local x_sd = r(sd)
    local x_p25 = r(p25)
    local x_p50 = r(p50)
    local x_p75 = r(p75)
    quietly count if `x' > 0 & !missing(`x')
    local x_pos = r(N)
    post `xpost' ("`x'") ("panel") (`x_N') (`x_mean') (`x_sd') (`x_p25') (`x_p50') (`x_p75') (`x_pos')
}

foreach x in DataAsset0 DataAsset_ln BookEntry HardAcctRaw hard_acct_pct_2024 NarrAcctHardGap NarrAcctHardWashing NarrAcctHardHushing NarrAcctHardVerified NarrAcctHardWashing_strict NarrAcctHardHushing_strict NarrAcctHardVerified_strict {
    quietly summarize `x' if year_num == 2024, detail
    local x_N = r(N)
    local x_mean = r(mean)
    local x_sd = r(sd)
    local x_p25 = r(p25)
    local x_p50 = r(p50)
    local x_p75 = r(p75)
    quietly count if year_num == 2024 & `x' > 0 & !missing(`x')
    local x_pos = r(N)
    post `xpost' ("`x'") ("2024") (`x_N') (`x_mean') (`x_sd') (`x_p25') (`x_p50') (`x_p75') (`x_pos')
}

postclose `xpost'
use `xsummary', clear
export delimited using "$V3/results/stata/narrative_hard_x_summary.csv", replace
save "$V3/results/stata/narrative_hard_x_summary.dta", replace

* Restore constructed panel.
use "$V3/results/data/narrative_hard_capability_mismatch_panel.dta", clear
sort Stkcd_num year_num
tsset Stkcd_num year_num

* -----------------------------
* 4. Full-panel lagged-X Y screen
* -----------------------------

local ctrls "Size Lev ROA TobinQ Age Growth IndepRatio Dual Top1Share SOE CFO"
local xlist_panel "NarrHardGap_base NarrHardWashing NarrHardHushing NarrHardVerified NarrHardWashing_strict NarrHardHushing_strict NarrHardVerified_strict"
local ylist "PriceDelay ForecastDisp FcstAcc Analyst ReportFreq RatingDisp TobinQ SA InstHold InstStable AuditFee absDA InvestIneff CashFlowVol SCConc SuppConc TFP"

tempfile out_panel
tempname postp
postfile `postp' str40 x_name str24 y_name str20 y_family str24 spec double coef se t_stat p_value N positives using `out_panel', replace

foreach y of local ylist {
    local family "other"
    if inlist("`y'", "ForecastDisp", "FcstAcc", "Analyst", "ReportFreq", "RatingDisp") local family "analyst_info"
    if inlist("`y'", "PriceDelay") local family "pricing_efficiency"
    if inlist("`y'", "TobinQ") local family "valuation"
    if inlist("`y'", "SA") local family "financing_constraint"
    if inlist("`y'", "InstHold", "InstStable") local family "investor_attention"
    if inlist("`y'", "AuditFee", "absDA") local family "audit_accounting"
    if inlist("`y'", "InvestIneff") local family "capital_allocation"
    if inlist("`y'", "CashFlowVol") local family "operating_resilience"
    if inlist("`y'", "SCConc", "SuppConc") local family "supply_chain"
    if inlist("`y'", "TFP") local family "productivity"

    capture confirm variable `y'
    if _rc continue

    foreach x of local xlist_panel {
        capture confirm variable `x'
        if _rc continue

        capture drop x_lag
        gen x_lag = L.`x'

        local regctrls ""
        local regmiss ""
        foreach c of local ctrls {
            capture confirm variable `c'
            if !_rc & "`c'" != "`y'" {
                local regctrls "`regctrls' `c'"
                local regmiss "`regmiss', `c'"
            }
        }

        quietly count if !missing(`y', x_lag `regmiss', Stkcd_num, year_num, IndYear_num)
        if r(N) == 0 {
            post `postp' ("`x'") ("`y'") ("`family'") ("firm_yearFE_lagX") (.) (.) (.) (.) (0) (0)
            continue
        }

        quietly count if !missing(`y', x_lag `regmiss', Stkcd_num, year_num, IndYear_num) & x_lag > 0
        local xpos = r(N)

        capture noisily quietly reghdfe `y' x_lag `regctrls', absorb(Stkcd_num year_num) cluster(IndYear_num)
        if _rc {
            post `postp' ("`x'") ("`y'") ("`family'") ("firm_yearFE_lagX") (.) (.) (.) (.) (0) (`xpos')
            continue
        }

        capture local coef = _b[x_lag]
        if _rc {
            post `postp' ("`x'") ("`y'") ("`family'") ("firm_yearFE_lagX") (.) (.) (.) (.) (e(N)) (`xpos')
            continue
        }
        local se = _se[x_lag]
        local tval = `coef' / `se'
        local pval = 2 * ttail(e(df_r), abs(`tval'))
        post `postp' ("`x'") ("`y'") ("`family'") ("firm_yearFE_lagX") (`coef') (`se') (`tval') (`pval') (e(N)) (`xpos')
    }
}

postclose `postp'
use `out_panel', clear
gen sig_10 = abs(t_stat) >= 1.645 if !missing(t_stat)
gen sig_05 = abs(t_stat) >= 1.96 if !missing(t_stat)
gen sig_01 = abs(t_stat) >= 2.576 if !missing(t_stat)
order x_name y_family y_name spec coef se t_stat p_value N positives sig_10 sig_05 sig_01
sort x_name y_family y_name
export delimited using "$V3/results/stata/narrative_hard_y_screen_panel.csv", replace
save "$V3/results/stata/narrative_hard_y_screen_panel.dta", replace

* -----------------------------
* 5. 2024 accounting-augmented Y screen
* -----------------------------

use "$V3/results/data/narrative_hard_capability_mismatch_panel.dta", clear

local xlist_2024 "NarrHardGap_base NarrHardWashing NarrHardHushing NarrHardVerified NarrAcctHardGap NarrAcctHardWashing NarrAcctHardHushing NarrAcctHardVerified NarrAcctHardWashing_strict NarrAcctHardHushing_strict NarrAcctHardVerified_strict BookEntry"

tempfile out_2024
tempname postc
postfile `postc' str40 x_name str24 y_name str20 y_family str24 spec double coef se t_stat p_value N positives using `out_2024', replace

foreach y of local ylist {
    local family "other"
    if inlist("`y'", "ForecastDisp", "FcstAcc", "Analyst", "ReportFreq", "RatingDisp") local family "analyst_info"
    if inlist("`y'", "PriceDelay") local family "pricing_efficiency"
    if inlist("`y'", "TobinQ") local family "valuation"
    if inlist("`y'", "SA") local family "financing_constraint"
    if inlist("`y'", "InstHold", "InstStable") local family "investor_attention"
    if inlist("`y'", "AuditFee", "absDA") local family "audit_accounting"
    if inlist("`y'", "InvestIneff") local family "capital_allocation"
    if inlist("`y'", "CashFlowVol") local family "operating_resilience"
    if inlist("`y'", "SCConc", "SuppConc") local family "supply_chain"
    if inlist("`y'", "TFP") local family "productivity"

    capture confirm variable `y'
    if _rc continue

    foreach x of local xlist_2024 {
        capture confirm variable `x'
        if _rc continue

        local regctrls ""
        local regmiss ""
        foreach c of local ctrls {
            capture confirm variable `c'
            if !_rc & "`c'" != "`y'" {
                local regctrls "`regctrls' `c'"
                local regmiss "`regmiss', `c'"
            }
        }

        quietly count if year_num == 2024 & !missing(`y', `x' `regmiss', IndYear_num)
        if r(N) == 0 {
            post `postc' ("`x'") ("`y'") ("`family'") ("indFE_controls") (.) (.) (.) (.) (0) (0)
            continue
        }

        quietly count if year_num == 2024 & !missing(`y', `x' `regmiss', IndYear_num) & `x' > 0
        local xpos = r(N)

        capture noisily quietly reghdfe `y' `x' `regctrls' if year_num == 2024, absorb(IndYear_num) vce(robust)
        if _rc {
            post `postc' ("`x'") ("`y'") ("`family'") ("indFE_controls") (.) (.) (.) (.) (0) (`xpos')
            continue
        }

        capture local coef = _b[`x']
        if _rc {
            post `postc' ("`x'") ("`y'") ("`family'") ("indFE_controls") (.) (.) (.) (.) (e(N)) (`xpos')
            continue
        }
        local se = _se[`x']
        local tval = `coef' / `se'
        local pval = 2 * ttail(e(df_r), abs(`tval'))
        post `postc' ("`x'") ("`y'") ("`family'") ("indFE_controls") (`coef') (`se') (`tval') (`pval') (e(N)) (`xpos')
    }
}

postclose `postc'
use `out_2024', clear
gen sig_10 = abs(t_stat) >= 1.645 if !missing(t_stat)
gen sig_05 = abs(t_stat) >= 1.96 if !missing(t_stat)
gen sig_01 = abs(t_stat) >= 2.576 if !missing(t_stat)
order x_name y_family y_name spec coef se t_stat p_value N positives sig_10 sig_05 sig_01
sort x_name y_family y_name
export delimited using "$V3/results/stata/narrative_hard_y_screen_2024.csv", replace
save "$V3/results/stata/narrative_hard_y_screen_2024.dta", replace

display _newline "=== X summary ==="
use "$V3/results/stata/narrative_hard_x_summary.dta", clear
list, abbreviate(24) sepby(sample)

display _newline "=== Panel highlights: |t| >= 1.96 ==="
use "$V3/results/stata/narrative_hard_y_screen_panel.dta", clear
list x_name y_family y_name coef t_stat p_value N positives if sig_05 == 1, sepby(x_name) abbreviate(24)

display _newline "=== 2024 highlights: |t| >= 1.96 ==="
use "$V3/results/stata/narrative_hard_y_screen_2024.dta", clear
list x_name y_family y_name coef t_stat p_value N positives if sig_05 == 1, sepby(x_name) abbreviate(24)

log close
