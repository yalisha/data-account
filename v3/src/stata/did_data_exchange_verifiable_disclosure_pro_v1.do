clear all
set more off
capture log close

global V3 "/Users/mac/computerscience/0做完了/15会计研究/v3"

log using "$V3/results/logs/did_data_exchange_verifiable_disclosure_pro_v1.log", replace text

display "=== Pro-feedback checks: verifiable disclosure design ==="

capture which reghdfe
if _rc {
    display as error "reghdfe not installed"
    exit 199
}

use "$V3/results/data/did_data_exchange_mismatch_ddd_v2_pre2017_panel.dta", clear

egen city_year_id = group(city_id year_num)
egen washany_year_id = group(PreWashAny1617 year_num)
egen wstr_year_id = group(PreStrictWashAny1617 year_num)
egen ind2024_id = group(Ind2 year_num) if year_num == 2024
replace ind2024_id = 0 if missing(ind2024_id)

gen post2024 = (year_num == 2024)
gen wany2024 = PreWashAny1617 * post2024
gen wstr2024 = PreStrictWashAny1617 * post2024

replace asset_trade_kw = 0 if missing(asset_trade_kw)
replace strict_at_kw = 0 if missing(strict_at_kw)

gen asset_trade_ratio = asset_trade_kw / (DU_kw + 1) if !missing(asset_trade_kw, DU_kw)
gen strict_at_ratio = strict_at_kw / (DU_kw + 1) if !missing(strict_at_kw, DU_kw)
gen asset_trade_share = asset_trade_kw / (asset_trade_kw + DU_kw + 1) if !missing(asset_trade_kw, DU_kw)
gen strict_at_share = strict_at_kw / (strict_at_kw + DU_kw + 1) if !missing(strict_at_kw, DU_kw)

local ctrls "Size Lev ROA TobinQ Age Growth IndepRatio Dual Top1Share SOE CFO InstHold Analyst"
local ylist "DU_kw asset_trade_kw strict_at_kw asset_trade_ratio strict_at_ratio asset_trade_share strict_at_share"

tempfile out
tempname posth
postfile `posth' str36 spec str24 exposure str32 y_name double coef se t_stat p_value N using `out', replace

foreach y of local ylist {
    local regctrls ""
    foreach c of local ctrls {
        capture confirm variable `c'
        if !_rc & "`c'" != "`y'" {
            local regctrls "`regctrls' `c'"
        }
    }

    * Baseline for comparison: firm FE + year FE.
    foreach x in dx_wany17 dx_wstr17 {
        capture noisily quietly reghdfe `y' DID_city `x' `regctrls' if inrange(year_num, 2018, 2024), absorb(firm_id year_num) cluster(city_id)
        if _rc {
            post `posth' ("baseline") ("`x'") ("`y'") (.) (.) (.) (.) (0)
        }
        else {
            local b = _b[`x']
            local s = _se[`x']
            local t = `b' / `s'
            local p = 2 * ttail(e(df_r), abs(`t'))
            post `posth' ("baseline") ("`x'") ("`y'") (`b') (`s') (`t') (`p') (e(N))
        }
    }

    * City-year FE + PreWashAny x year FE. DID main effect is absorbed.
    capture noisily quietly reghdfe `y' dx_wany17 `regctrls' if inrange(year_num, 2018, 2024), absorb(firm_id city_year_id washany_year_id) cluster(city_id)
    if _rc {
        post `posth' ("cityyear_prewashyear_fe") ("dx_wany17") ("`y'") (.) (.) (.) (.) (0)
    }
    else {
        local b = _b[dx_wany17]
        local s = _se[dx_wany17]
        local t = `b' / `s'
        local p = 2 * ttail(e(df_r), abs(`t'))
        post `posth' ("cityyear_prewashyear_fe") ("dx_wany17") ("`y'") (`b') (`s') (`t') (`p') (e(N))
    }

    * City-year FE + strict-washing x year FE.
    capture noisily quietly reghdfe `y' dx_wstr17 `regctrls' if inrange(year_num, 2018, 2024), absorb(firm_id city_year_id wstr_year_id) cluster(city_id)
    if _rc {
        post `posth' ("cityyear_prewashyear_fe") ("dx_wstr17") ("`y'") (.) (.) (.) (.) (0)
    }
    else {
        local b = _b[dx_wstr17]
        local s = _se[dx_wstr17]
        local t = `b' / `s'
        local p = 2 * ttail(e(df_r), abs(`t'))
        post `posth' ("cityyear_prewashyear_fe") ("dx_wstr17") ("`y'") (`b') (`s') (`t') (`p') (e(N))
    }

    * Drop 2024 observations to address national data-resource accounting rule.
    foreach x in dx_wany17 dx_wstr17 {
        capture noisily quietly reghdfe `y' DID_city `x' `regctrls' if inrange(year_num, 2018, 2023), absorb(firm_id year_num) cluster(city_id)
        if _rc {
            post `posth' ("drop_year_2024") ("`x'") ("`y'") (.) (.) (.) (.) (0)
        }
        else {
            local b = _b[`x']
            local s = _se[`x']
            local t = `b' / `s'
            local p = 2 * ttail(e(df_r), abs(`t'))
            post `posth' ("drop_year_2024") ("`x'") ("`y'") (`b') (`s') (`t') (`p') (e(N))
        }
    }

    * Absorb/partial out 2024 accounting shock for pre-washing firms.
    capture noisily quietly reghdfe `y' DID_city dx_wany17 wany2024 `regctrls' if inrange(year_num, 2018, 2024), absorb(firm_id year_num) cluster(city_id)
    if _rc {
        post `posth' ("add_prewash_x_2024") ("dx_wany17") ("`y'") (.) (.) (.) (.) (0)
    }
    else {
        local b = _b[dx_wany17]
        local s = _se[dx_wany17]
        local t = `b' / `s'
        local p = 2 * ttail(e(df_r), abs(`t'))
        post `posth' ("add_prewash_x_2024") ("dx_wany17") ("`y'") (`b') (`s') (`t') (`p') (e(N))
    }

    capture noisily quietly reghdfe `y' DID_city dx_wstr17 wstr2024 `regctrls' if inrange(year_num, 2018, 2024), absorb(firm_id year_num) cluster(city_id)
    if _rc {
        post `posth' ("add_prewash_x_2024") ("dx_wstr17") ("`y'") (.) (.) (.) (.) (0)
    }
    else {
        local b = _b[dx_wstr17]
        local s = _se[dx_wstr17]
        local t = `b' / `s'
        local p = 2 * ttail(e(df_r), abs(`t'))
        post `posth' ("add_prewash_x_2024") ("dx_wstr17") ("`y'") (`b') (`s') (`t') (`p') (e(N))
    }

    * Industry-specific 2024 accounting shock.
    foreach x in dx_wany17 dx_wstr17 {
        capture noisily quietly reghdfe `y' DID_city `x' `regctrls' if inrange(year_num, 2018, 2024), absorb(firm_id year_num ind2024_id) cluster(city_id)
        if _rc {
            post `posth' ("industry_2024_fe") ("`x'") ("`y'") (.) (.) (.) (.) (0)
        }
        else {
            local b = _b[`x']
            local s = _se[`x']
            local t = `b' / `s'
            local p = 2 * ttail(e(df_r), abs(`t'))
            post `posth' ("industry_2024_fe") ("`x'") ("`y'") (`b') (`s') (`t') (`p') (e(N))
        }
    }
}

postclose `posth'
use `out', clear
gen sig_10 = abs(t_stat) >= 1.645 if !missing(t_stat)
gen sig_05 = abs(t_stat) >= 1.96 if !missing(t_stat)
gen sig_01 = abs(t_stat) >= 2.576 if !missing(t_stat)
order spec exposure y_name coef se t_stat p_value N sig_10 sig_05 sig_01
sort y_name exposure spec
export delimited using "$V3/results/stata/did_data_exchange_verifiable_disclosure_pro_v1.csv", replace
save "$V3/results/stata/did_data_exchange_verifiable_disclosure_pro_v1.dta", replace

display _newline "=== Pro-feedback check highlights ==="
list spec exposure y_name coef t_stat p_value N if inlist(y_name, "DU_kw", "asset_trade_kw", "asset_trade_ratio", "asset_trade_share"), sepby(y_name exposure) abbreviate(30)

log close
