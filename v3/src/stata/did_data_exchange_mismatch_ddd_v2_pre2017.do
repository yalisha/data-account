clear all
set more off
capture log close

global V3 "/Users/mac/computerscience/0做完了/15会计研究/v3"
global EX "/Users/mac/computerscience/0做完了/16数据交易所"

log using "$V3/results/logs/did_data_exchange_mismatch_ddd_v2_pre2017.log", replace text

display "=== Data-exchange DID x pre-2016-2017 mismatch DDD v2 ==="
display "This version avoids overlap between pre-exposure construction and the 2018-2024 event-study window."

capture which reghdfe
if _rc {
    display as error "reghdfe not installed"
    exit 199
}

* ------------------------------------------------------------------
* 0. Build pre-2018 firm exposure from v3 full panel
* ------------------------------------------------------------------

tempfile preexp
use "$V3/results/data/hardcap_direct_v1_panel.dta", clear
bys Stkcd: egen PreGap1617 = mean(cond(inrange(year_num, 2016, 2017), NarrHardGap_direct_v1, .))
bys Stkcd: egen PreWashShare1617 = mean(cond(inrange(year_num, 2016, 2017), NarrHardWashing_direct_v1, .))
bys Stkcd: egen PreWashAny1617 = max(cond(inrange(year_num, 2016, 2017), NarrHardWashing_direct_v1, .))
bys Stkcd: egen PreStrictWashAny1617 = max(cond(inrange(year_num, 2016, 2017), NHWash_direct_v1_s, .))
bys Stkcd: egen PreHardCap1617 = mean(cond(inrange(year_num, 2016, 2017), HardCap_direct_v1, .))
bys Stkcd: egen PreDU1617 = mean(cond(inrange(year_num, 2016, 2017), DU_kw, .))
gen PreWashMaj1617 = (PreWashShare1617 >= .5) if !missing(PreWashShare1617)
gen PreHighGap1617 = (PreGap1617 >= 0) if !missing(PreGap1617)
keep Stkcd PreGap1617 PreHighGap1617 PreWashShare1617 PreWashAny1617 PreWashMaj1617 PreStrictWashAny1617 PreHardCap1617 PreDU1617
duplicates drop Stkcd, force
save `preexp', replace

* ------------------------------------------------------------------
* 1. Prepare project-16 data exchange DID and asset-trade keyword variables
* ------------------------------------------------------------------

tempfile exdid
use "$EX/data/did_sample_llm.dta", clear
rename year year_num
keep Stkcd year_num city_clean prov_clean treat_year DID_city event_time_city ///
    SYNCH Volatility score1 score2 has_data_content n_paragraphs llm_matched
duplicates drop Stkcd year_num, force
save `exdid', replace

tempfile assetkw
import delimited using "$EX/data/asset_trade_keywords.csv", clear varnames(1)
capture destring stkcd, gen(Stkcd) force
capture confirm variable Stkcd
if _rc {
    rename stkcd Stkcd
}
rename year year_num
gen asset_trade_d = (asset_trade_raw > 0) if asset_trade_raw < .
gen strict_at_d = (strict_at_raw > 0) if strict_at_raw < .
keep Stkcd year_num asset_trade_kw asset_trade_raw asset_trade_d strict_at_kw strict_at_raw strict_at_d
duplicates drop Stkcd year_num, force
save `assetkw', replace

* ------------------------------------------------------------------
* 2. Merge
* ------------------------------------------------------------------

use "$V3/results/data/hardcap_direct_v1_panel.dta", clear
merge 1:1 Stkcd year_num using `exdid', keep(match) nogen
merge 1:1 Stkcd year_num using `assetkw', keep(master match) nogen
merge m:1 Stkcd using `preexp', keep(master match) nogen

egen firm_id = group(Stkcd)
egen city_id = group(city_clean)

replace has_data_content = . if has_data_content < 0
replace n_paragraphs = . if n_paragraphs < 0
gen ln_n_paragraphs = ln(1 + n_paragraphs) if n_paragraphs < .

gen dx_gap17 = DID_city * PreGap1617
gen dx_highgap17 = DID_city * PreHighGap1617
gen dx_wany17 = DID_city * PreWashAny1617
gen dx_wmaj17 = DID_city * PreWashMaj1617
gen dx_wstr17 = DID_city * PreStrictWashAny1617

gen et = event_time_city
replace et = -3 if event_time_city <= -3 & event_time_city != .
replace et = 3 if event_time_city >= 3 & event_time_city != .

foreach k in m3 m2 p0 p1 p2 p3 {
    gen evt_`k' = 0
}
replace evt_m3 = (et == -3) if et != .
replace evt_m2 = (et == -2) if et != .
replace evt_p0 = (et == 0) if et != .
replace evt_p1 = (et == 1) if et != .
replace evt_p2 = (et == 2) if et != .
replace evt_p3 = (et == 3) if et != .

foreach e in m3 m2 p0 p1 p2 p3 {
    gen egap17_`e' = evt_`e' * PreGap1617
    gen ehighgap17_`e' = evt_`e' * PreHighGap1617
    gen ewany17_`e' = evt_`e' * PreWashAny1617
    gen ewmaj17_`e' = evt_`e' * PreWashMaj1617
    gen ewstr17_`e' = evt_`e' * PreStrictWashAny1617
}

save "$V3/results/data/did_data_exchange_mismatch_ddd_v2_pre2017_panel.dta", replace

* ------------------------------------------------------------------
* 3. Exposure summary
* ------------------------------------------------------------------

preserve
keep if year_num == 2018
tempfile exposum
tempname expost
postfile `expost' str32 exposure double N mean sd p25 median p75 positives using `exposum', replace

foreach x in PreGap1617 PreHighGap1617 PreWashShare1617 PreWashAny1617 PreWashMaj1617 PreStrictWashAny1617 PreHardCap1617 PreDU1617 {
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
export delimited using "$V3/results/stata/did_data_exchange_mismatch_ddd_v2_pre2017_exposure_summary.csv", replace
restore

* ------------------------------------------------------------------
* 4. Static DDD screen
* ------------------------------------------------------------------

local ctrls "Size Lev ROA TobinQ Age Growth IndepRatio Dual Top1Share SOE CFO InstHold Analyst"
local xlist "dx_gap17 dx_highgap17 dx_wany17 dx_wmaj17 dx_wstr17"
local ylist "NarrHardGap_direct_v1 NarrHardWashing_direct_v1 NarrHardVerified_direct_v1 HardCap_direct_v1 DU_kw asset_trade_kw strict_at_kw asset_trade_d strict_at_d score1 score2 has_data_content ln_n_paragraphs TobinQ Analyst RatingDisp ReportFreq AuditFee PriceDelay SYNCH"
local mismatch_outcomes "NarrHardGap_direct_v1 NarrHardWashing_direct_v1 NarrHardVerified_direct_v1"
local first_stage_outcomes "HardCap_direct_v1 DU_kw asset_trade_kw strict_at_kw asset_trade_d strict_at_d score1 score2 has_data_content ln_n_paragraphs"
local analyst_outcomes "Analyst ReportFreq RatingDisp"
local pricing_outcomes "PriceDelay SYNCH"

tempfile mainout
tempname mainpost
postfile `mainpost' str24 x_name str32 y_name str24 y_family str24 spec double coef se t_stat p_value N positives using `mainout', replace

foreach y of local ylist {
    local family "other"
    if strpos(" `mismatch_outcomes' ", " `y' ") local family "mismatch_x"
    if strpos(" `first_stage_outcomes' ", " `y' ") local family "first_stage"
    if strpos(" `analyst_outcomes' ", " `y' ") local family "analyst_info"
    if strpos(" `pricing_outcomes' ", " `y' ") local family "pricing_efficiency"
    if "`y'" == "TobinQ" local family "valuation"
    if "`y'" == "AuditFee" local family "audit_accounting"

    capture confirm variable `y'
    if _rc continue

    foreach x of local xlist {
        local regctrls ""
        foreach c of local ctrls {
            capture confirm variable `c'
            if !_rc & "`c'" != "`y'" {
                local regctrls "`regctrls' `c'"
            }
        }

        quietly count if inrange(year_num, 2018, 2024) & !missing(`y', DID_city, `x', firm_id, year_num, city_id)
        if r(N) == 0 {
            post `mainpost' ("`x'") ("`y'") ("`family'") ("exchange_pre2017_ddd") (.) (.) (.) (.) (0) (0)
            continue
        }

        quietly count if inrange(year_num, 2018, 2024) & !missing(`y', DID_city, `x', firm_id, year_num, city_id) & `x' > 0
        local xpos = r(N)

        capture noisily quietly reghdfe `y' DID_city `x' `regctrls' if inrange(year_num, 2018, 2024), absorb(firm_id year_num) cluster(city_id)
        if _rc {
            post `mainpost' ("`x'") ("`y'") ("`family'") ("exchange_pre2017_ddd") (.) (.) (.) (.) (0) (`xpos')
            continue
        }

        capture local coef = _b[`x']
        if _rc {
            post `mainpost' ("`x'") ("`y'") ("`family'") ("exchange_pre2017_ddd") (.) (.) (.) (.) (e(N)) (`xpos')
            continue
        }
        local se = _se[`x']
        local tval = `coef' / `se'
        local pval = 2 * ttail(e(df_r), abs(`tval'))
        post `mainpost' ("`x'") ("`y'") ("`family'") ("exchange_pre2017_ddd") (`coef') (`se') (`tval') (`pval') (e(N)) (`xpos')
    }
}

postclose `mainpost'
use `mainout', clear
gen sig_10 = abs(t_stat) >= 1.645 if !missing(t_stat)
gen sig_05 = abs(t_stat) >= 1.96 if !missing(t_stat)
gen sig_01 = abs(t_stat) >= 2.576 if !missing(t_stat)
order x_name y_family y_name spec coef se t_stat p_value N positives sig_10 sig_05 sig_01
sort x_name y_family y_name
export delimited using "$V3/results/stata/did_data_exchange_mismatch_ddd_v2_pre2017_main.csv", replace
save "$V3/results/stata/did_data_exchange_mismatch_ddd_v2_pre2017_main.dta", replace

* ------------------------------------------------------------------
* 5. Event DDD screen
* ------------------------------------------------------------------

use "$V3/results/data/did_data_exchange_mismatch_ddd_v2_pre2017_panel.dta", clear

tempfile evout
tempname evpost
postfile `evpost' str20 exposure str32 y_name str24 y_family str8 event_time double coef se t_stat p_value N using `evout', replace

local evmain "evt_m3 evt_m2 evt_p0 evt_p1 evt_p2 evt_p3"

foreach y of local ylist {
    local family "other"
    if strpos(" `mismatch_outcomes' ", " `y' ") local family "mismatch_x"
    if strpos(" `first_stage_outcomes' ", " `y' ") local family "first_stage"
    if strpos(" `analyst_outcomes' ", " `y' ") local family "analyst_info"
    if strpos(" `pricing_outcomes' ", " `y' ") local family "pricing_efficiency"
    if "`y'" == "TobinQ" local family "valuation"
    if "`y'" == "AuditFee" local family "audit_accounting"

    capture confirm variable `y'
    if _rc continue

    local regctrls ""
    foreach c of local ctrls {
        capture confirm variable `c'
        if !_rc & "`c'" != "`y'" {
            local regctrls "`regctrls' `c'"
        }
    }

    foreach block in "gap17 egap17" "highgap17 ehighgap17" "wany17 ewany17" "wmaj17 ewmaj17" "wstr17 ewstr17" {
        tokenize "`block'"
        local ename "`1'"
        local pref "`2'"

        capture noisily quietly reghdfe `y' `evmain' `pref'_m3 `pref'_m2 `pref'_p0 `pref'_p1 `pref'_p2 `pref'_p3 `regctrls' if inrange(year_num, 2018, 2024), absorb(firm_id year_num) cluster(city_id)
        if _rc {
            foreach e in m3 m2 p0 p1 p2 p3 {
                post `evpost' ("`ename'") ("`y'") ("`family'") ("`e'") (.) (.) (.) (.) (0)
            }
            continue
        }

        foreach e in m3 m2 p0 p1 p2 p3 {
            capture local coef = _b[`pref'_`e']
            if _rc {
                post `evpost' ("`ename'") ("`y'") ("`family'") ("`e'") (.) (.) (.) (.) (e(N))
                continue
            }
            local se = _se[`pref'_`e']
            local tval = `coef' / `se'
            local pval = 2 * ttail(e(df_r), abs(`tval'))
            post `evpost' ("`ename'") ("`y'") ("`family'") ("`e'") (`coef') (`se') (`tval') (`pval') (e(N))
        }
    }
}

postclose `evpost'
use `evout', clear
gen sig_10 = abs(t_stat) >= 1.645 if !missing(t_stat)
gen sig_05 = abs(t_stat) >= 1.96 if !missing(t_stat)
gen sig_01 = abs(t_stat) >= 2.576 if !missing(t_stat)
order exposure y_family y_name event_time coef se t_stat p_value N sig_10 sig_05 sig_01
sort exposure y_family y_name event_time
export delimited using "$V3/results/stata/did_data_exchange_mismatch_ddd_v2_pre2017_event.csv", replace
save "$V3/results/stata/did_data_exchange_mismatch_ddd_v2_pre2017_event.dta", replace

display _newline "=== Static DDD v2 highlights: |t| >= 1.96 ==="
use "$V3/results/stata/did_data_exchange_mismatch_ddd_v2_pre2017_main.dta", clear
list x_name y_family y_name coef t_stat p_value N positives if sig_05 == 1, sepby(x_name) abbreviate(28)

display _newline "=== Event DDD v2 highlights: |t| >= 1.96 ==="
use "$V3/results/stata/did_data_exchange_mismatch_ddd_v2_pre2017_event.dta", clear
list exposure y_family y_name event_time coef t_stat p_value N if sig_05 == 1, sepby(exposure y_name) abbreviate(28)

log close
