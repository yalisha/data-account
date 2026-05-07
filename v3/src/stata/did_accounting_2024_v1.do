clear all
set more off
capture log close

global V3 "/Users/mac/computerscience/0做完了/15会计研究/v3"

log using "$V3/results/logs/did_accounting_2024_v1.log", replace text

display "=== DID v1: 2024 data-resource accounting rule x pre-policy washing ==="
display "Policy: Enterprise data-resource accounting rule effective 2024-01-01"
display "Sample: 2021-2024; exposure measured by 2021-2023 pre-policy mismatch"
display "Spec: Y_it = exposure_i x Post2024_t + controls + firm FE + year FE"

capture which reghdfe
if _rc {
    display as error "reghdfe not installed"
    exit 199
}

use "$V3/results/data/hardcap_direct_v1_panel.dta", clear

* -----------------------------
* 1. Pre-policy exposure
* -----------------------------

bys Stkcd_num: egen PreGap2123 = mean(cond(inrange(year_num, 2021, 2023), NarrHardGap_direct_v1, .))
bys Stkcd_num: egen PreWashShare2123 = mean(cond(inrange(year_num, 2021, 2023), NarrHardWashing_direct_v1, .))
bys Stkcd_num: egen PreWashAny2123 = max(cond(inrange(year_num, 2021, 2023), NarrHardWashing_direct_v1, .))
bys Stkcd_num: egen PreStrictWashAny2123 = max(cond(inrange(year_num, 2021, 2023), NHWash_direct_v1_s, .))

gen PreWashMaj2123 = (PreWashShare2123 >= .5) if !missing(PreWashShare2123)
gen Post2024 = (year_num >= 2024)

gen acct_gap_post = PreGap2123 * Post2024
gen acct_wany_post = PreWashAny2123 * Post2024
gen acct_wmaj_post = PreWashMaj2123 * Post2024
gen acct_wstr_post = PreStrictWashAny2123 * Post2024

label variable PreGap2123 "Mean direct narrative-hard gap, 2021-2023"
label variable PreWashAny2123 "Any direct washing, 2021-2023"
label variable PreWashMaj2123 "Majority direct washing, 2021-2023"
label variable PreStrictWashAny2123 "Any strict direct washing, 2021-2023"
label variable acct_gap_post "PreGap2123 x Post2024"
label variable acct_wany_post "PreWashAny2123 x Post2024"
label variable acct_wmaj_post "PreWashMaj2123 x Post2024"
label variable acct_wstr_post "PreStrictWashAny2123 x Post2024"

* Event-style interactions with 2023 as baseline.
foreach yy in 2021 2022 2024 {
    gen egap_`yy' = PreGap2123 * (year_num == `yy')
    gen ewany_`yy' = PreWashAny2123 * (year_num == `yy')
    gen ewmaj_`yy' = PreWashMaj2123 * (year_num == `yy')
    gen ewstr_`yy' = PreStrictWashAny2123 * (year_num == `yy')
}

save "$V3/results/data/did_accounting_2024_v1_panel.dta", replace

* -----------------------------
* 2. Exposure summary
* -----------------------------

preserve
keep if year_num == 2023
tempfile exposum
tempname expost
postfile `expost' str28 exposure double N mean sd p25 median p75 positives using `exposum', replace

foreach x in PreGap2123 PreWashShare2123 PreWashAny2123 PreWashMaj2123 PreStrictWashAny2123 {
    quietly summarize `x', detail
    local x_N = r(N)
    local x_mean = r(mean)
    local x_sd = r(sd)
    local x_p25 = r(p25)
    local x_p50 = r(p50)
    local x_p75 = r(p75)
    quietly count if `x' > 0 & !missing(`x')
    local x_pos = r(N)
    post `expost' ("`x'") (`x_N') (`x_mean') (`x_sd') (`x_p25') (`x_p50') (`x_p75') (`x_pos')
}
postclose `expost'
use `exposum', clear
export delimited using "$V3/results/stata/did_accounting_2024_v1_exposure_summary.csv", replace
restore

* -----------------------------
* 3. Main DID screen
* -----------------------------

local ctrls "Size Lev ROA TobinQ Age Growth IndepRatio Dual Top1Share SOE CFO"
local ylist "TobinQ Analyst ForecastDisp RatingDisp ReportFreq AuditFee PriceDelay SA InstHold InstStable InvestIneff CashFlowVol TFP SCConc SuppConc"
local xlist "acct_gap_post acct_wany_post acct_wmaj_post acct_wstr_post"

tempfile mainout
tempname mainpost
postfile `mainpost' str28 x_name str24 y_name str20 y_family str24 spec double coef se t_stat p_value N positives using `mainout', replace

foreach y of local ylist {
    local family "other"
    if inlist("`y'", "ForecastDisp", "Analyst", "ReportFreq", "RatingDisp") local family "analyst_info"
    if inlist("`y'", "PriceDelay") local family "pricing_efficiency"
    if inlist("`y'", "TobinQ") local family "valuation"
    if inlist("`y'", "SA") local family "financing_constraint"
    if inlist("`y'", "InstHold", "InstStable") local family "investor_attention"
    if inlist("`y'", "AuditFee") local family "audit_accounting"
    if inlist("`y'", "InvestIneff") local family "capital_allocation"
    if inlist("`y'", "CashFlowVol") local family "operating_resilience"
    if inlist("`y'", "SCConc", "SuppConc") local family "supply_chain"
    if inlist("`y'", "TFP") local family "productivity"

    capture confirm variable `y'
    if _rc continue

    foreach x of local xlist {
        local regctrls ""
        local regmiss ""
        foreach c of local ctrls {
            capture confirm variable `c'
            if !_rc & "`c'" != "`y'" {
                local regctrls "`regctrls' `c'"
                local regmiss "`regmiss', `c'"
            }
        }

        quietly count if inrange(year_num, 2021, 2024) & !missing(`y', `x' `regmiss', Stkcd_num, year_num, IndYear_num)
        if r(N) == 0 {
            post `mainpost' ("`x'") ("`y'") ("`family'") ("acct2024_did") (.) (.) (.) (.) (0) (0)
            continue
        }

        quietly count if inrange(year_num, 2021, 2024) & !missing(`y', `x' `regmiss', Stkcd_num, year_num, IndYear_num) & `x' > 0
        local xpos = r(N)

        capture noisily quietly reghdfe `y' `x' `regctrls' if inrange(year_num, 2021, 2024), absorb(Stkcd_num year_num) cluster(IndYear_num)
        if _rc {
            post `mainpost' ("`x'") ("`y'") ("`family'") ("acct2024_did") (.) (.) (.) (.) (0) (`xpos')
            continue
        }

        capture local coef = _b[`x']
        if _rc {
            post `mainpost' ("`x'") ("`y'") ("`family'") ("acct2024_did") (.) (.) (.) (.) (e(N)) (`xpos')
            continue
        }
        local se = _se[`x']
        local tval = `coef' / `se'
        local pval = 2 * ttail(e(df_r), abs(`tval'))
        post `mainpost' ("`x'") ("`y'") ("`family'") ("acct2024_did") (`coef') (`se') (`tval') (`pval') (e(N)) (`xpos')
    }
}

postclose `mainpost'
use `mainout', clear
gen sig_10 = abs(t_stat) >= 1.645 if !missing(t_stat)
gen sig_05 = abs(t_stat) >= 1.96 if !missing(t_stat)
gen sig_01 = abs(t_stat) >= 2.576 if !missing(t_stat)
order x_name y_family y_name spec coef se t_stat p_value N positives sig_10 sig_05 sig_01
sort x_name y_family y_name
export delimited using "$V3/results/stata/did_accounting_2024_v1_main.csv", replace
save "$V3/results/stata/did_accounting_2024_v1_main.dta", replace

* -----------------------------
* 4. Event/pretrend screen: 2023 omitted
* -----------------------------

use "$V3/results/data/did_accounting_2024_v1_panel.dta", clear

tempfile evout
tempname evpost
postfile `evpost' str20 exposure str24 y_name str20 y_family int event_year double coef se t_stat p_value N using `evout', replace

foreach y of local ylist {
    local family "other"
    if inlist("`y'", "ForecastDisp", "Analyst", "ReportFreq", "RatingDisp") local family "analyst_info"
    if inlist("`y'", "PriceDelay") local family "pricing_efficiency"
    if inlist("`y'", "TobinQ") local family "valuation"
    if inlist("`y'", "SA") local family "financing_constraint"
    if inlist("`y'", "InstHold", "InstStable") local family "investor_attention"
    if inlist("`y'", "AuditFee") local family "audit_accounting"
    if inlist("`y'", "InvestIneff") local family "capital_allocation"
    if inlist("`y'", "CashFlowVol") local family "operating_resilience"
    if inlist("`y'", "SCConc", "SuppConc") local family "supply_chain"
    if inlist("`y'", "TFP") local family "productivity"

    capture confirm variable `y'
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

    foreach block in "gap egap" "wany ewany" "wmaj ewmaj" "wstr ewstr" {
        tokenize "`block'"
        local ename "`1'"
        local pref "`2'"

        capture noisily quietly reghdfe `y' `pref'_2021 `pref'_2022 `pref'_2024 `regctrls' if inrange(year_num, 2021, 2024), absorb(Stkcd_num year_num) cluster(IndYear_num)
        if _rc {
            foreach yy in 2021 2022 2024 {
                post `evpost' ("`ename'") ("`y'") ("`family'") (`yy') (.) (.) (.) (.) (0)
            }
            continue
        }
        foreach yy in 2021 2022 2024 {
            capture local coef = _b[`pref'_`yy']
            if _rc {
                post `evpost' ("`ename'") ("`y'") ("`family'") (`yy') (.) (.) (.) (.) (e(N))
                continue
            }
            local se = _se[`pref'_`yy']
            local tval = `coef' / `se'
            local pval = 2 * ttail(e(df_r), abs(`tval'))
            post `evpost' ("`ename'") ("`y'") ("`family'") (`yy') (`coef') (`se') (`tval') (`pval') (e(N))
        }
    }
}

postclose `evpost'
use `evout', clear
gen sig_10 = abs(t_stat) >= 1.645 if !missing(t_stat)
gen sig_05 = abs(t_stat) >= 1.96 if !missing(t_stat)
gen sig_01 = abs(t_stat) >= 2.576 if !missing(t_stat)
order exposure y_family y_name event_year coef se t_stat p_value N sig_10 sig_05 sig_01
sort exposure y_family y_name event_year
export delimited using "$V3/results/stata/did_accounting_2024_v1_event.csv", replace
save "$V3/results/stata/did_accounting_2024_v1_event.dta", replace

display _newline "=== Accounting DID main highlights: |t| >= 1.96 ==="
use "$V3/results/stata/did_accounting_2024_v1_main.dta", clear
list x_name y_family y_name coef t_stat p_value N positives if sig_05 == 1, sepby(x_name) abbreviate(28)

display _newline "=== Accounting DID event highlights: |t| >= 1.96 ==="
use "$V3/results/stata/did_accounting_2024_v1_event.dta", clear
list exposure y_family y_name event_year coef t_stat p_value N if sig_05 == 1, sepby(exposure y_name) abbreviate(28)

log close
