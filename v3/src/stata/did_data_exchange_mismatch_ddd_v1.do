clear all
set more off
capture log close

global V3 "/Users/mac/computerscience/0做完了/15会计研究/v3"
global EX "/Users/mac/computerscience/0做完了/16数据交易所"

log using "$V3/results/logs/did_data_exchange_mismatch_ddd_v1.log", replace text

display "=== Data-exchange DID x pre-mismatch DDD v1 ==="
display "Design: city data-exchange establishment x firm pre-2018-2020 data-washing exposure"
display "Main question: whether data exchanges constrain annual-report narrative-hardcap mismatch"

capture which reghdfe
if _rc {
    display as error "reghdfe not installed"
    exit 199
}

* ------------------------------------------------------------------
* 0. Prepare city DID variables from project 16
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
* 1. Merge with v3 direct hardcap panel
* ------------------------------------------------------------------

use "$V3/results/data/hardcap_direct_v1_panel.dta", clear
merge 1:1 Stkcd year_num using `exdid', keep(match) nogen
merge 1:1 Stkcd year_num using `assetkw', keep(master match) nogen

egen firm_id = group(Stkcd_num)
egen city_id = group(city_clean)
egen city_year_id = group(city_clean year_num)

* Make a few project-16 text outcomes safer.
replace has_data_content = . if has_data_content < 0
replace n_paragraphs = . if n_paragraphs < 0
gen ln_n_paragraphs = ln(1 + n_paragraphs) if n_paragraphs < .

* ------------------------------------------------------------------
* 2. Predetermined firm exposure, measured before the 2021-2024 exchange wave
* ------------------------------------------------------------------

bys Stkcd_num: egen PreGap1820 = mean(cond(inrange(year_num, 2018, 2020), NarrHardGap_direct_v1, .))
bys Stkcd_num: egen PreWashShare1820 = mean(cond(inrange(year_num, 2018, 2020), NarrHardWashing_direct_v1, .))
bys Stkcd_num: egen PreWashAny1820 = max(cond(inrange(year_num, 2018, 2020), NarrHardWashing_direct_v1, .))
bys Stkcd_num: egen PreStrictWashAny1820 = max(cond(inrange(year_num, 2018, 2020), NHWash_direct_v1_s, .))
bys Stkcd_num: egen PreHardCap1820 = mean(cond(inrange(year_num, 2018, 2020), HardCap_direct_v1, .))
bys Stkcd_num: egen PreDU1820 = mean(cond(inrange(year_num, 2018, 2020), DU_kw, .))

gen PreWashMaj1820 = (PreWashShare1820 >= .5) if !missing(PreWashShare1820)
gen PreHighGap1820 = (PreGap1820 >= 0) if !missing(PreGap1820)

gen dx_gap = DID_city * PreGap1820
gen dx_highgap = DID_city * PreHighGap1820
gen dx_wany = DID_city * PreWashAny1820
gen dx_wmaj = DID_city * PreWashMaj1820
gen dx_wstr = DID_city * PreStrictWashAny1820

label variable PreGap1820 "Pre-wave mean narrative-hardcap gap, 2018-2020"
label variable PreWashAny1820 "Any pre-wave data washing, 2018-2020"
label variable PreWashMaj1820 "Majority pre-wave data washing, 2018-2020"
label variable PreStrictWashAny1820 "Any strict pre-wave data washing, 2018-2020"
label variable dx_gap "Data exchange x PreGap1820"
label variable dx_highgap "Data exchange x PreHighGap1820"
label variable dx_wany "Data exchange x PreWashAny1820"
label variable dx_wmaj "Data exchange x PreWashMaj1820"
label variable dx_wstr "Data exchange x PreStrictWashAny1820"

* Event-time bins and DDD event interactions; -1 is the omitted base.
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
    gen egap_`e' = evt_`e' * PreGap1820
    gen ehighgap_`e' = evt_`e' * PreHighGap1820
    gen ewany_`e' = evt_`e' * PreWashAny1820
    gen ewmaj_`e' = evt_`e' * PreWashMaj1820
    gen ewstr_`e' = evt_`e' * PreStrictWashAny1820
}

save "$V3/results/data/did_data_exchange_mismatch_ddd_v1_panel.dta", replace

* ------------------------------------------------------------------
* 3. Exposure summary
* ------------------------------------------------------------------

preserve
keep if year_num == 2020
tempfile exposum
tempname expost
postfile `expost' str32 exposure double N mean sd p25 median p75 positives using `exposum', replace

foreach x in PreGap1820 PreHighGap1820 PreWashShare1820 PreWashAny1820 PreWashMaj1820 PreStrictWashAny1820 PreHardCap1820 PreDU1820 DID_city {
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
export delimited using "$V3/results/stata/did_data_exchange_mismatch_ddd_v1_exposure_summary.csv", replace
restore

* ------------------------------------------------------------------
* 4. Static DDD screen
* ------------------------------------------------------------------

local ctrls "Size Lev ROA TobinQ Age Growth IndepRatio Dual Top1Share SOE CFO InstHold Analyst"
local xlist "dx_gap dx_highgap dx_wany dx_wmaj dx_wstr"
local ylist "NarrHardGap_direct_v1 NarrHardWashing_direct_v1 NarrHardHushing_direct_v1 NarrHardVerified_direct_v1 HardCap_direct_v1 DU_kw asset_trade_kw strict_at_kw asset_trade_d strict_at_d score1 score2 has_data_content ln_n_paragraphs TobinQ Analyst ForecastDisp RatingDisp ReportFreq AuditFee PriceDelay SYNCH"
local mismatch_outcomes "NarrHardGap_direct_v1 NarrHardWashing_direct_v1 NarrHardHushing_direct_v1 NarrHardVerified_direct_v1"
local first_stage_outcomes "HardCap_direct_v1 DU_kw asset_trade_kw strict_at_kw asset_trade_d strict_at_d score1 score2 has_data_content ln_n_paragraphs"
local analyst_outcomes "ForecastDisp Analyst ReportFreq RatingDisp"
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
        local regmiss ""
        foreach c of local ctrls {
            capture confirm variable `c'
            if !_rc & "`c'" != "`y'" {
                local regctrls "`regctrls' `c'"
                local regmiss "`regmiss', `c'"
            }
        }

        quietly count if inrange(year_num, 2018, 2024) & !missing(`y', DID_city, `x', firm_id, year_num, city_id)
        if r(N) == 0 {
            post `mainpost' ("`x'") ("`y'") ("`family'") ("exchange_preexp_ddd") (.) (.) (.) (.) (0) (0)
            continue
        }

        quietly count if inrange(year_num, 2018, 2024) & !missing(`y', DID_city, `x', firm_id, year_num, city_id) & `x' > 0
        local xpos = r(N)

        capture noisily quietly reghdfe `y' DID_city `x' `regctrls' if inrange(year_num, 2018, 2024), absorb(firm_id year_num) cluster(city_id)
        if _rc {
            post `mainpost' ("`x'") ("`y'") ("`family'") ("exchange_preexp_ddd") (.) (.) (.) (.) (0) (`xpos')
            continue
        }

        capture local coef = _b[`x']
        if _rc {
            post `mainpost' ("`x'") ("`y'") ("`family'") ("exchange_preexp_ddd") (.) (.) (.) (.) (e(N)) (`xpos')
            continue
        }
        local se = _se[`x']
        local tval = `coef' / `se'
        local pval = 2 * ttail(e(df_r), abs(`tval'))
        post `mainpost' ("`x'") ("`y'") ("`family'") ("exchange_preexp_ddd") (`coef') (`se') (`tval') (`pval') (e(N)) (`xpos')
    }
}

postclose `mainpost'
use `mainout', clear
gen sig_10 = abs(t_stat) >= 1.645 if !missing(t_stat)
gen sig_05 = abs(t_stat) >= 1.96 if !missing(t_stat)
gen sig_01 = abs(t_stat) >= 2.576 if !missing(t_stat)
order x_name y_family y_name spec coef se t_stat p_value N positives sig_10 sig_05 sig_01
sort x_name y_family y_name
export delimited using "$V3/results/stata/did_data_exchange_mismatch_ddd_v1_main.csv", replace
save "$V3/results/stata/did_data_exchange_mismatch_ddd_v1_main.dta", replace

* ------------------------------------------------------------------
* 5. DDD event-study screen, with event time -1 omitted
* ------------------------------------------------------------------

use "$V3/results/data/did_data_exchange_mismatch_ddd_v1_panel.dta", clear

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

    foreach block in "gap egap" "highgap ehighgap" "wany ewany" "wmaj ewmaj" "wstr ewstr" {
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
export delimited using "$V3/results/stata/did_data_exchange_mismatch_ddd_v1_event.csv", replace
save "$V3/results/stata/did_data_exchange_mismatch_ddd_v1_event.dta", replace

display _newline "=== Static DDD highlights: |t| >= 1.96 ==="
use "$V3/results/stata/did_data_exchange_mismatch_ddd_v1_main.dta", clear
list x_name y_family y_name coef t_stat p_value N positives if sig_05 == 1, sepby(x_name) abbreviate(28)

display _newline "=== Event DDD highlights: |t| >= 1.96 ==="
use "$V3/results/stata/did_data_exchange_mismatch_ddd_v1_event.dta", clear
list exposure y_family y_name event_time coef t_stat p_value N if sig_05 == 1, sepby(exposure y_name) abbreviate(28)

log close
