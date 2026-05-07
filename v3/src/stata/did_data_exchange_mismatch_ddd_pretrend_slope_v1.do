clear all
set more off
capture log close

global V3 "/Users/mac/computerscience/0做完了/15会计研究/v3"

log using "$V3/results/logs/did_data_exchange_mismatch_ddd_pretrend_slope_v1.log", replace text

display "=== Group-specific pretrend slope checks for data-exchange DDD ==="

capture which reghdfe
if _rc {
    display as error "reghdfe not installed"
    exit 199
}

local ctrls "Size Lev ROA TobinQ Age Growth IndepRatio Dual Top1Share SOE CFO InstHold Analyst"
local ylist "NarrHardGap_direct_v1 NarrHardWashing_direct_v1 DU_kw asset_trade_kw strict_at_kw TobinQ"

* ------------------------------------------------------------------
* 1. Static DDD with treated-ever x pre-exposure linear trend
* ------------------------------------------------------------------

use "$V3/results/data/did_data_exchange_mismatch_ddd_v2_pre2017_panel.dta", clear
gen trend = year_num - 2018
gen TreatedEver = !missing(treat_year)
gen te_wany_trend = TreatedEver * PreWashAny1617 * trend
gen te_wstr_trend = TreatedEver * PreStrictWashAny1617 * trend
gen te_gap_trend = TreatedEver * PreGap1617 * trend

tempfile staticout
tempname staticpost
postfile `staticpost' str32 spec str24 x_name str32 y_name double coef se t_stat p_value N using `staticout', replace

foreach block in "dx_wany17 te_wany_trend" "dx_wstr17 te_wstr_trend" {
    tokenize "`block'"
    local x "`1'"
    local slope "`2'"

    foreach y of local ylist {
        local regctrls ""
        foreach c of local ctrls {
            capture confirm variable `c'
            if !_rc & "`c'" != "`y'" {
                local regctrls "`regctrls' `c'"
            }
        }

        capture noisily quietly reghdfe `y' DID_city `x' `slope' `regctrls' if inrange(year_num, 2018, 2024), absorb(firm_id year_num) cluster(city_id)
        if _rc {
            post `staticpost' ("treated_exposure_linear_trend") ("`x'") ("`y'") (.) (.) (.) (.) (0)
            continue
        }
        capture local b = _b[`x']
        if _rc {
            post `staticpost' ("treated_exposure_linear_trend") ("`x'") ("`y'") (.) (.) (.) (.) (e(N))
            continue
        }
        local s = _se[`x']
        local t = `b' / `s'
        local p = 2 * ttail(e(df_r), abs(`t'))
        post `staticpost' ("treated_exposure_linear_trend") ("`x'") ("`y'") (`b') (`s') (`t') (`p') (e(N))
    }
}

postclose `staticpost'
use `staticout', clear
gen sig_10 = abs(t_stat) >= 1.645 if !missing(t_stat)
gen sig_05 = abs(t_stat) >= 1.96 if !missing(t_stat)
gen sig_01 = abs(t_stat) >= 2.576 if !missing(t_stat)
order spec x_name y_name coef se t_stat p_value N sig_10 sig_05 sig_01
sort y_name x_name
export delimited using "$V3/results/stata/did_data_exchange_mismatch_ddd_pretrend_slope_v1_static.csv", replace
save "$V3/results/stata/did_data_exchange_mismatch_ddd_pretrend_slope_v1_static.dta", replace

* ------------------------------------------------------------------
* 2. Stacked event DDD with treated-cohort x exposure x rel trend
* ------------------------------------------------------------------

use "$V3/results/data/did_data_exchange_mismatch_ddd_robust_v1_stacked_panel.dta", clear
gen st_wany_rel = stack_treated * PreWashAny1617 * rel
gen st_wstr_rel = stack_treated * PreStrictWashAny1617 * rel

tempfile evout jointout
tempname evpost jointpost
postfile `evpost' str24 exposure str32 y_name str8 event_time double coef se t_stat p_value N using `evout', replace
postfile `jointpost' str24 exposure str32 y_name double pretrend_p N using `jointout', replace

local evmain "sev_m3 sev_m2 sev_p0 sev_p1 sev_p2 sev_p3"

foreach y of local ylist {
    local regctrls ""
    foreach c of local ctrls {
        capture confirm variable `c'
        if !_rc & "`c'" != "`y'" {
            local regctrls "`regctrls' `c'"
        }
    }

    foreach block in "wany17 swany17 st_wany_rel" "wstr17 swstr17 st_wstr_rel" {
        tokenize "`block'"
        local ename "`1'"
        local pref "`2'"
        local slope "`3'"

        capture noisily quietly reghdfe `y' `evmain' `pref'_m3 `pref'_m2 `pref'_p0 `pref'_p1 `pref'_p2 `pref'_p3 `slope' `regctrls', absorb(stack_firm_id stack_year_id) cluster(city_id)
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

use `evout', clear
gen sig_10 = abs(t_stat) >= 1.645 if !missing(t_stat)
gen sig_05 = abs(t_stat) >= 1.96 if !missing(t_stat)
gen sig_01 = abs(t_stat) >= 2.576 if !missing(t_stat)
order exposure y_name event_time coef se t_stat p_value N sig_10 sig_05 sig_01
sort y_name exposure event_time
export delimited using "$V3/results/stata/did_data_exchange_mismatch_ddd_pretrend_slope_v1_stacked_event.csv", replace
save "$V3/results/stata/did_data_exchange_mismatch_ddd_pretrend_slope_v1_stacked_event.dta", replace

use `jointout', clear
gen pass_10 = pretrend_p >= .10 if !missing(pretrend_p)
gen pass_05 = pretrend_p >= .05 if !missing(pretrend_p)
order exposure y_name pretrend_p N pass_10 pass_05
sort y_name exposure
export delimited using "$V3/results/stata/did_data_exchange_mismatch_ddd_pretrend_slope_v1_stacked_pretrend_joint.csv", replace
save "$V3/results/stata/did_data_exchange_mismatch_ddd_pretrend_slope_v1_stacked_pretrend_joint.dta", replace

display _newline "=== Static with treated exposure trend ==="
use "$V3/results/stata/did_data_exchange_mismatch_ddd_pretrend_slope_v1_static.dta", clear
list spec x_name y_name coef t_stat p_value N, sepby(y_name) abbreviate(28)

display _newline "=== Trend-adjusted stacked pretrend joint tests ==="
use "$V3/results/stata/did_data_exchange_mismatch_ddd_pretrend_slope_v1_stacked_pretrend_joint.dta", clear
list exposure y_name pretrend_p N pass_10 pass_05, sepby(y_name) abbreviate(28)

log close
