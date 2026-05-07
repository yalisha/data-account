clear all
set more off
capture log close

global V3 "/Users/mac/computerscience/0做完了/15会计研究/v3"

log using "$V3/results/logs/did_data_exchange_mismatch_ddd_robust_v1.log", replace text

display "=== Robustness checks for data-exchange x pre-2017 mismatch DDD ==="

capture which reghdfe
if _rc {
    display as error "reghdfe not installed"
    exit 199
}

use "$V3/results/data/did_data_exchange_mismatch_ddd_v2_pre2017_panel.dta", clear

gen trend = year_num - 2018
egen prov_year_id = group(prov_clean year_num)
capture confirm variable IndYear_num
if _rc {
    egen IndYear_num = group(Ind2 year_num)
}

gen DID_city_nolaunch = DID_city
replace DID_city_nolaunch = 0 if event_time_city == 0
gen dx_wany17_nolaunch = DID_city_nolaunch * PreWashAny1617
gen dx_wstr17_nolaunch = DID_city_nolaunch * PreStrictWashAny1617

local ctrls "Size Lev ROA TobinQ Age Growth IndepRatio Dual Top1Share SOE CFO InstHold Analyst"
local ylist "NarrHardGap_direct_v1 NarrHardWashing_direct_v1 DU_kw asset_trade_kw strict_at_kw TobinQ"
local xlist "dx_wany17 dx_wstr17"

* ------------------------------------------------------------------
* 1. Static robustness: stronger FE, timing restrictions
* ------------------------------------------------------------------

tempfile staticout
tempname staticpost
postfile `staticpost' str32 spec str24 x_name str32 y_name double coef se t_stat p_value N using `staticout', replace

foreach spec in baseline city_linear_trend province_year_fe industry_year_fe drop_2024_cohort no_launch_year {
    local didvar "DID_city"
    local xs "`xlist'"
    local absorb "firm_id year_num"
    local sample "inrange(year_num, 2018, 2024)"

    if "`spec'" == "city_linear_trend" {
        local absorb "firm_id year_num city_id#c.trend"
    }
    if "`spec'" == "province_year_fe" {
        local absorb "firm_id prov_year_id"
    }
    if "`spec'" == "industry_year_fe" {
        local absorb "firm_id IndYear_num"
    }
    if "`spec'" == "drop_2024_cohort" {
        local sample "inrange(year_num, 2018, 2024) & (missing(treat_year) | treat_year != 2024)"
    }
    if "`spec'" == "no_launch_year" {
        local didvar "DID_city_nolaunch"
        local xs "dx_wany17_nolaunch dx_wstr17_nolaunch"
    }

    foreach y of local ylist {
        local regctrls ""
        foreach c of local ctrls {
            capture confirm variable `c'
            if !_rc & "`c'" != "`y'" {
                local regctrls "`regctrls' `c'"
            }
        }

        foreach x of local xs {
            capture noisily quietly reghdfe `y' `didvar' `x' `regctrls' if `sample', absorb(`absorb') cluster(city_id)
            if _rc {
                post `staticpost' ("`spec'") ("`x'") ("`y'") (.) (.) (.) (.) (0)
                continue
            }

            capture local b = _b[`x']
            if _rc {
                post `staticpost' ("`spec'") ("`x'") ("`y'") (.) (.) (.) (.) (e(N))
                continue
            }
            local s = _se[`x']
            local t = `b' / `s'
            local p = 2 * ttail(e(df_r), abs(`t'))
            post `staticpost' ("`spec'") ("`x'") ("`y'") (`b') (`s') (`t') (`p') (e(N))
        }
    }
}

postclose `staticpost'
use `staticout', clear
gen sig_10 = abs(t_stat) >= 1.645 if !missing(t_stat)
gen sig_05 = abs(t_stat) >= 1.96 if !missing(t_stat)
gen sig_01 = abs(t_stat) >= 2.576 if !missing(t_stat)
order spec x_name y_name coef se t_stat p_value N sig_10 sig_05 sig_01
sort y_name x_name spec
export delimited using "$V3/results/stata/did_data_exchange_mismatch_ddd_robust_v1_static.csv", replace
save "$V3/results/stata/did_data_exchange_mismatch_ddd_robust_v1_static.dta", replace

* ------------------------------------------------------------------
* 2. Build stacked not-yet-treated / never-treated sample
* ------------------------------------------------------------------

tempfile base stackall
use "$V3/results/data/did_data_exchange_mismatch_ddd_v2_pre2017_panel.dta", clear
gen trend = year_num - 2018
egen prov_year_id = group(prov_clean year_num)
capture confirm variable IndYear_num
if _rc {
    egen IndYear_num = group(Ind2 year_num)
}
save `base', replace

local first 1
foreach g in 2021 2022 2023 2024 {
    use `base', clear
    gen stack = `g'
    gen stack_treated = (treat_year == `g')
    gen rel = year_num - `g'
    keep if inrange(rel, -3, 3)
    keep if stack_treated == 1 | missing(treat_year) | treat_year > year_num
    gen spost = (stack_treated == 1 & year_num >= `g')
    gen sdx_wany17 = spost * PreWashAny1617
    gen sdx_wstr17 = spost * PreStrictWashAny1617
    replace sdx_wany17 = 0 if spost == 0
    replace sdx_wstr17 = 0 if spost == 0

    foreach e in m3 m2 p0 p1 p2 p3 {
        gen sev_`e' = 0
    }
    replace sev_m3 = (stack_treated == 1 & rel <= -3)
    replace sev_m2 = (stack_treated == 1 & rel == -2)
    replace sev_p0 = (stack_treated == 1 & rel == 0)
    replace sev_p1 = (stack_treated == 1 & rel == 1)
    replace sev_p2 = (stack_treated == 1 & rel == 2)
    replace sev_p3 = (stack_treated == 1 & rel >= 3)

    foreach e in m3 m2 p0 p1 p2 p3 {
        gen swany17_`e' = sev_`e' * PreWashAny1617
        gen swstr17_`e' = sev_`e' * PreStrictWashAny1617
        replace swany17_`e' = 0 if sev_`e' == 0
        replace swstr17_`e' = 0 if sev_`e' == 0
    }

    if `first' {
        save `stackall', replace
        local first 0
    }
    else {
        append using `stackall'
        save `stackall', replace
    }
}

use `stackall', clear
egen stack_firm_id = group(stack firm_id)
egen stack_year_id = group(stack year_num)
save "$V3/results/data/did_data_exchange_mismatch_ddd_robust_v1_stacked_panel.dta", replace

* ------------------------------------------------------------------
* 3. Stacked static DID with not-yet/never controls
* ------------------------------------------------------------------

tempfile stackedstatic
tempname stspost
postfile `stspost' str32 spec str24 x_name str32 y_name double coef se t_stat p_value N using `stackedstatic', replace

foreach y of local ylist {
    local regctrls ""
    foreach c of local ctrls {
        capture confirm variable `c'
        if !_rc & "`c'" != "`y'" {
            local regctrls "`regctrls' `c'"
        }
    }

    foreach x in sdx_wany17 sdx_wstr17 {
        capture noisily quietly reghdfe `y' spost `x' `regctrls', absorb(stack_firm_id stack_year_id) cluster(city_id)
        if _rc {
            post `stspost' ("stacked_notyet_static") ("`x'") ("`y'") (.) (.) (.) (.) (0)
            continue
        }
        capture local b = _b[`x']
        if _rc {
            post `stspost' ("stacked_notyet_static") ("`x'") ("`y'") (.) (.) (.) (.) (e(N))
            continue
        }
        local s = _se[`x']
        local t = `b' / `s'
        local p = 2 * ttail(e(df_r), abs(`t'))
        post `stspost' ("stacked_notyet_static") ("`x'") ("`y'") (`b') (`s') (`t') (`p') (e(N))
    }
}

postclose `stspost'
use `stackedstatic', clear
gen sig_10 = abs(t_stat) >= 1.645 if !missing(t_stat)
gen sig_05 = abs(t_stat) >= 1.96 if !missing(t_stat)
gen sig_01 = abs(t_stat) >= 2.576 if !missing(t_stat)
order spec x_name y_name coef se t_stat p_value N sig_10 sig_05 sig_01
sort y_name x_name
export delimited using "$V3/results/stata/did_data_exchange_mismatch_ddd_robust_v1_stacked_static.csv", replace
save "$V3/results/stata/did_data_exchange_mismatch_ddd_robust_v1_stacked_static.dta", replace

* ------------------------------------------------------------------
* 4. Stacked event-study DDD
* ------------------------------------------------------------------

use "$V3/results/data/did_data_exchange_mismatch_ddd_robust_v1_stacked_panel.dta", clear

tempfile stackedevent stackedjoint
tempname evpost jointpost
postfile `evpost' str24 exposure str32 y_name str8 event_time double coef se t_stat p_value N using `stackedevent', replace
postfile `jointpost' str24 exposure str32 y_name double pretrend_p N using `stackedjoint', replace

local evmain "sev_m3 sev_m2 sev_p0 sev_p1 sev_p2 sev_p3"

foreach y of local ylist {
    local regctrls ""
    foreach c of local ctrls {
        capture confirm variable `c'
        if !_rc & "`c'" != "`y'" {
            local regctrls "`regctrls' `c'"
        }
    }

    foreach block in "wany17 swany17" "wstr17 swstr17" {
        tokenize "`block'"
        local ename "`1'"
        local pref "`2'"

        capture noisily quietly reghdfe `y' `evmain' `pref'_m3 `pref'_m2 `pref'_p0 `pref'_p1 `pref'_p2 `pref'_p3 `regctrls', absorb(stack_firm_id stack_year_id) cluster(city_id)
        if _rc {
            foreach e in m3 m2 p0 p1 p2 p3 {
                post `evpost' ("`ename'") ("`y'") ("`e'") (.) (.) (.) (.) (0)
            }
            post `jointpost' ("`ename'") ("`y'") (.) (0)
            continue
        }

        capture test `pref'_m3 `pref'_m2
        if _rc {
            local ppre = .
        }
        else {
            local ppre = r(p)
        }
        post `jointpost' ("`ename'") ("`y'") (`ppre') (e(N))

        foreach e in m3 m2 p0 p1 p2 p3 {
            capture local b = _b[`pref'_`e']
            if _rc {
                post `evpost' ("`ename'") ("`y'") ("`e'") (.) (.) (.) (.) (e(N))
                continue
            }
            local s = _se[`pref'_`e']
            local t = `b' / `s'
            local p = 2 * ttail(e(df_r), abs(`t'))
            post `evpost' ("`ename'") ("`y'") ("`e'") (`b') (`s') (`t') (`p') (e(N))
        }
    }
}

postclose `evpost'
postclose `jointpost'

use `stackedevent', clear
gen sig_10 = abs(t_stat) >= 1.645 if !missing(t_stat)
gen sig_05 = abs(t_stat) >= 1.96 if !missing(t_stat)
gen sig_01 = abs(t_stat) >= 2.576 if !missing(t_stat)
order exposure y_name event_time coef se t_stat p_value N sig_10 sig_05 sig_01
sort y_name exposure event_time
export delimited using "$V3/results/stata/did_data_exchange_mismatch_ddd_robust_v1_stacked_event.csv", replace
save "$V3/results/stata/did_data_exchange_mismatch_ddd_robust_v1_stacked_event.dta", replace

use `stackedjoint', clear
gen pass_10 = pretrend_p >= .10 if !missing(pretrend_p)
gen pass_05 = pretrend_p >= .05 if !missing(pretrend_p)
order exposure y_name pretrend_p N pass_10 pass_05
sort y_name exposure
export delimited using "$V3/results/stata/did_data_exchange_mismatch_ddd_robust_v1_stacked_pretrend_joint.csv", replace
save "$V3/results/stata/did_data_exchange_mismatch_ddd_robust_v1_stacked_pretrend_joint.dta", replace

* ------------------------------------------------------------------
* 5. Leave-one-treated-city-out, main exposure only
* ------------------------------------------------------------------

use "$V3/results/data/did_data_exchange_mismatch_ddd_v2_pre2017_panel.dta", clear
levelsof city_id if !missing(treat_year), local(treated_city_ids)

tempfile jackout
tempname jackpost
postfile `jackpost' str40 dropped_city int dropped_city_id str32 y_name double coef se t_stat p_value N using `jackout', replace

foreach cid of local treated_city_ids {
    preserve
    keep if city_id == `cid'
    local cname = city_clean[1]
    restore

    foreach y of local ylist {
        local regctrls ""
        foreach c of local ctrls {
            capture confirm variable `c'
            if !_rc & "`c'" != "`y'" {
                local regctrls "`regctrls' `c'"
            }
        }

        capture noisily quietly reghdfe `y' DID_city dx_wany17 `regctrls' if inrange(year_num, 2018, 2024) & city_id != `cid', absorb(firm_id year_num) cluster(city_id)
        if _rc {
            post `jackpost' ("`cname'") (`cid') ("`y'") (.) (.) (.) (.) (0)
            continue
        }
        capture local b = _b[dx_wany17]
        if _rc {
            post `jackpost' ("`cname'") (`cid') ("`y'") (.) (.) (.) (.) (e(N))
            continue
        }
        local s = _se[dx_wany17]
        local t = `b' / `s'
        local p = 2 * ttail(e(df_r), abs(`t'))
        post `jackpost' ("`cname'") (`cid') ("`y'") (`b') (`s') (`t') (`p') (e(N))
    }
}

postclose `jackpost'
use `jackout', clear
gen sig_10 = abs(t_stat) >= 1.645 if !missing(t_stat)
gen sig_05 = abs(t_stat) >= 1.96 if !missing(t_stat)
gen sig_01 = abs(t_stat) >= 2.576 if !missing(t_stat)
order y_name dropped_city dropped_city_id coef se t_stat p_value N sig_10 sig_05 sig_01
sort y_name dropped_city_id
export delimited using "$V3/results/stata/did_data_exchange_mismatch_ddd_robust_v1_jackknife.csv", replace
save "$V3/results/stata/did_data_exchange_mismatch_ddd_robust_v1_jackknife.dta", replace

preserve
collapse (count) n=coef (mean) mean_coef=coef mean_t=t_stat mean_sig05=sig_05 (min) min_coef=coef min_t=t_stat (max) max_coef=coef max_t=t_stat, by(y_name)
export delimited using "$V3/results/stata/did_data_exchange_mismatch_ddd_robust_v1_jackknife_summary.csv", replace
restore

display _newline "=== Static robustness, significant at 5 percent ==="
use "$V3/results/stata/did_data_exchange_mismatch_ddd_robust_v1_static.dta", clear
list spec x_name y_name coef t_stat p_value N if sig_05 == 1, sepby(y_name x_name) abbreviate(28)

display _newline "=== Stacked static ==="
use "$V3/results/stata/did_data_exchange_mismatch_ddd_robust_v1_stacked_static.dta", clear
list spec x_name y_name coef t_stat p_value N, sepby(y_name) abbreviate(28)

display _newline "=== Stacked pretrend joint tests ==="
use "$V3/results/stata/did_data_exchange_mismatch_ddd_robust_v1_stacked_pretrend_joint.dta", clear
list exposure y_name pretrend_p N pass_10 pass_05, sepby(y_name) abbreviate(28)

display _newline "=== Jackknife summary ==="
import delimited using "$V3/results/stata/did_data_exchange_mismatch_ddd_robust_v1_jackknife_summary.csv", clear
list, sepby(y_name) abbreviate(28)

log close
