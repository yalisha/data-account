clear all
set more off
capture log close

global V3 "/Users/mac/computerscience/0做完了/15会计研究/v3"

log using "$V3/results/logs/did_data_exchange_actual_data_asset_validation_v1.log", replace text

display "=== Actual data-asset validation for data-exchange verifiable disclosure design ==="
display "Input 1: did_data_exchange_mismatch_ddd_v2_pre2017_panel.dta"
display "Input 2: data_asset_from_panel.csv"
display "Role: validation layer, because positive DataAsset observations are concentrated in 2024."

capture which reghdfe
if _rc {
    display as error "reghdfe not installed"
    exit 199
}

* ------------------------------------------------------------------
* 0. Merge actual data-resource booking amount
* ------------------------------------------------------------------

tempfile asset
import delimited using "$V3/results/data/data_asset_from_panel.csv", clear varnames(1)
capture rename stkcd_num Stkcd_num
capture rename year_num year_num
capture rename dataasset DataAsset
destring Stkcd_num year_num DataAsset, replace force
gen Stkcd = Stkcd_num
keep Stkcd year_num DataAsset
duplicates drop Stkcd year_num, force
save `asset', replace

use "$V3/results/data/did_data_exchange_mismatch_ddd_v2_pre2017_panel.dta", clear
merge 1:1 Stkcd year_num using `asset', keep(master match) nogen

capture drop DataAsset0 DataAsset_ln BookEntry
gen DataAsset0 = DataAsset
replace DataAsset0 = 0 if missing(DataAsset0)
gen DataAsset_ln = ln(1 + DataAsset0)
gen BookEntry = (DataAsset0 > 0) if !missing(DataAsset0)

replace asset_trade_kw = 0 if missing(asset_trade_kw)
replace strict_at_kw = 0 if missing(strict_at_kw)

capture drop asset_trade_ratio strict_at_ratio asset_trade_share strict_at_share
capture drop asset_trade_pos strict_at_pos DU_pos
gen asset_trade_ratio = asset_trade_kw / (DU_kw + 1) if !missing(asset_trade_kw, DU_kw)
gen strict_at_ratio = strict_at_kw / (DU_kw + 1) if !missing(strict_at_kw, DU_kw)
gen asset_trade_share = asset_trade_kw / (asset_trade_kw + DU_kw + 1) if !missing(asset_trade_kw, DU_kw)
gen strict_at_share = strict_at_kw / (strict_at_kw + DU_kw + 1) if !missing(strict_at_kw, DU_kw)
gen asset_trade_pos = (asset_trade_kw > 0) if !missing(asset_trade_kw)
gen strict_at_pos = (strict_at_kw > 0) if !missing(strict_at_kw)
gen DU_pos = (DU_kw > 0) if !missing(DU_kw)

capture drop city_year_id washany_year_id wstr_year_id
egen city_year_id = group(city_id year_num)
egen washany_year_id = group(PreWashAny1617 year_num)
egen wstr_year_id = group(PreStrictWashAny1617 year_num)

capture drop post2024 wany2024 wstr2024
gen post2024 = (year_num == 2024)
gen wany2024 = PreWashAny1617 * post2024
gen wstr2024 = PreStrictWashAny1617 * post2024

capture drop ind_id
egen ind_id = group(Ind2)

label variable DataAsset0 "Actual booked data-resource/data-asset amount, missing set to zero"
label variable DataAsset_ln "ln(1 + actual booked data-resource/data-asset amount)"
label variable BookEntry "Positive actual data-resource/data-asset booking"
label variable asset_trade_ratio "Asset/trade keyword intensity divided by DU_kw+1"
label variable strict_at_ratio "Strict asset/trade keyword intensity divided by DU_kw+1"
label variable asset_trade_share "Asset/trade keyword share in broad data narrative"
label variable strict_at_share "Strict asset/trade keyword share in broad data narrative"

save "$V3/results/data/did_data_exchange_actual_data_asset_validation_v1_panel.dta", replace

* ------------------------------------------------------------------
* 1. Coverage of actual data assets
* ------------------------------------------------------------------

preserve
collapse (count) N=DataAsset0 (sum) book_positive=BookEntry (mean) book_rate=BookEntry ///
    (mean) mean_data_asset=DataAsset0 (max) max_data_asset=DataAsset0, by(year_num)
export delimited using "$V3/results/stata/did_data_exchange_actual_data_asset_validation_v1_coverage.csv", replace
restore

preserve
keep if year_num == 2024
contract asset_trade_pos strict_at_pos BookEntry
rename _freq firm_years
export delimited using "$V3/results/stata/did_data_exchange_actual_data_asset_validation_v1_crosstab_2024.csv", replace
restore

* ------------------------------------------------------------------
* 2. 2024 measurement validation:
*    Do asset/trade text measures predict actual data-asset booking?
* ------------------------------------------------------------------

local ctrls "Size Lev ROA TobinQ Age Growth IndepRatio Dual Top1Share SOE CFO InstHold Analyst"
local xlist "DU_kw asset_trade_kw strict_at_kw asset_trade_ratio strict_at_ratio asset_trade_share strict_at_share asset_trade_pos strict_at_pos"
local ylist "BookEntry DataAsset_ln"

tempfile validout
tempname validpost
postfile `validpost' str28 spec str32 x_name str20 y_name double coef se t_stat p_value N y_positive x_positive both_positive corr using `validout', replace

foreach y of local ylist {
    foreach x of local xlist {
        local regctrls ""
        local regmiss ""
        foreach c of local ctrls {
            capture confirm variable `c'
            if !_rc & "`c'" != "`y'" & "`c'" != "`x'" {
                local regctrls "`regctrls' `c'"
                local regmiss "`regmiss', `c'"
            }
        }

        quietly count if year_num == 2024 & !missing(`y', `x' `regmiss', ind_id)
        local nobs = r(N)
        quietly count if year_num == 2024 & !missing(`y', `x' `regmiss', ind_id) & `y' > 0
        local ypos = r(N)
        quietly count if year_num == 2024 & !missing(`y', `x' `regmiss', ind_id) & `x' > 0
        local xpos = r(N)
        quietly count if year_num == 2024 & !missing(`y', `x' `regmiss', ind_id) & `y' > 0 & `x' > 0
        local bothpos = r(N)

        capture quietly correlate `y' `x' if year_num == 2024 & !missing(`y', `x')
        if _rc local rho = .
        else {
            matrix C = r(C)
            local rho = C[1,2]
        }

        if `nobs' == 0 {
            post `validpost' ("industry_fe_2024") ("`x'") ("`y'") (.) (.) (.) (.) (0) (`ypos') (`xpos') (`bothpos') (`rho')
            continue
        }

        capture noisily quietly reghdfe `y' `x' `regctrls' if year_num == 2024, absorb(ind_id) vce(robust)
        if _rc {
            post `validpost' ("industry_fe_2024") ("`x'") ("`y'") (.) (.) (.) (.) (0) (`ypos') (`xpos') (`bothpos') (`rho')
            continue
        }

        capture local b = _b[`x']
        if _rc {
            post `validpost' ("industry_fe_2024") ("`x'") ("`y'") (.) (.) (.) (.) (e(N)) (`ypos') (`xpos') (`bothpos') (`rho')
            continue
        }
        local s = _se[`x']
        local t = `b' / `s'
        local p = 2 * ttail(e(df_r), abs(`t'))
        post `validpost' ("industry_fe_2024") ("`x'") ("`y'") (`b') (`s') (`t') (`p') (e(N)) (`ypos') (`xpos') (`bothpos') (`rho')
    }
}

postclose `validpost'
use `validout', clear
gen sig_10 = abs(t_stat) >= 1.645 if !missing(t_stat)
gen sig_05 = abs(t_stat) >= 1.96 if !missing(t_stat)
gen sig_01 = abs(t_stat) >= 2.576 if !missing(t_stat)
order spec y_name x_name coef se t_stat p_value N y_positive x_positive both_positive corr sig_10 sig_05 sig_01
sort y_name x_name
export delimited using "$V3/results/stata/did_data_exchange_actual_data_asset_validation_v1_measurement.csv", replace
save "$V3/results/stata/did_data_exchange_actual_data_asset_validation_v1_measurement.dta", replace

* ------------------------------------------------------------------
* 3. Exploratory DDD using actual data assets as Y.
*    This is not promoted as a main DID because the dependent variable is
*    effectively a 2024 accounting-rule realization.
* ------------------------------------------------------------------

use "$V3/results/data/did_data_exchange_actual_data_asset_validation_v1_panel.dta", clear

tempfile didout
tempname didpost
postfile `didpost' str36 spec str24 exposure str20 y_name double coef se t_stat p_value N y_positive using `didout', replace

foreach y in BookEntry DataAsset_ln {
    local regctrls ""
    foreach c of local ctrls {
        capture confirm variable `c'
        if !_rc & "`c'" != "`y'" {
            local regctrls "`regctrls' `c'"
        }
    }

    foreach x in dx_wany17 dx_wstr17 {
        quietly count if inrange(year_num, 2018, 2024) & !missing(`y', DID_city, `x', firm_id, year_num, city_id)
        local nobs = r(N)
        quietly count if inrange(year_num, 2018, 2024) & !missing(`y', DID_city, `x', firm_id, year_num, city_id) & `y' > 0
        local ypos = r(N)

        capture noisily quietly reghdfe `y' DID_city `x' `regctrls' if inrange(year_num, 2018, 2024), absorb(firm_id year_num) cluster(city_id)
        if _rc {
            post `didpost' ("baseline_firm_year_fe") ("`x'") ("`y'") (.) (.) (.) (.) (0) (`ypos')
        }
        else {
            local b = _b[`x']
            local s = _se[`x']
            local t = `b' / `s'
            local p = 2 * ttail(e(df_r), abs(`t'))
            post `didpost' ("baseline_firm_year_fe") ("`x'") ("`y'") (`b') (`s') (`t') (`p') (e(N)) (`ypos')
        }

        capture noisily quietly reghdfe `y' DID_city `x' wany2024 `regctrls' if inrange(year_num, 2018, 2024) & "`x'" == "dx_wany17", absorb(firm_id year_num) cluster(city_id)
        if !_rc & "`x'" == "dx_wany17" {
            local b = _b[`x']
            local s = _se[`x']
            local t = `b' / `s'
            local p = 2 * ttail(e(df_r), abs(`t'))
            post `didpost' ("add_prewash_x_2024") ("`x'") ("`y'") (`b') (`s') (`t') (`p') (e(N)) (`ypos')
        }

        capture noisily quietly reghdfe `y' DID_city `x' wstr2024 `regctrls' if inrange(year_num, 2018, 2024) & "`x'" == "dx_wstr17", absorb(firm_id year_num) cluster(city_id)
        if !_rc & "`x'" == "dx_wstr17" {
            local b = _b[`x']
            local s = _se[`x']
            local t = `b' / `s'
            local p = 2 * ttail(e(df_r), abs(`t'))
            post `didpost' ("add_prewash_x_2024") ("`x'") ("`y'") (`b') (`s') (`t') (`p') (e(N)) (`ypos')
        }
    }

    capture noisily quietly reghdfe `y' dx_wany17 `regctrls' if inrange(year_num, 2018, 2024), absorb(firm_id city_year_id washany_year_id) cluster(city_id)
    if _rc {
        post `didpost' ("cityyear_prewashyear_fe") ("dx_wany17") ("`y'") (.) (.) (.) (.) (0) (.)
    }
    else {
        quietly count if e(sample) & `y' > 0
        local ypos = r(N)
        local b = _b[dx_wany17]
        local s = _se[dx_wany17]
        local t = `b' / `s'
        local p = 2 * ttail(e(df_r), abs(`t'))
        post `didpost' ("cityyear_prewashyear_fe") ("dx_wany17") ("`y'") (`b') (`s') (`t') (`p') (e(N)) (`ypos')
    }

    capture noisily quietly reghdfe `y' dx_wstr17 `regctrls' if inrange(year_num, 2018, 2024), absorb(firm_id city_year_id wstr_year_id) cluster(city_id)
    if _rc {
        post `didpost' ("cityyear_prewashyear_fe") ("dx_wstr17") ("`y'") (.) (.) (.) (.) (0) (.)
    }
    else {
        quietly count if e(sample) & `y' > 0
        local ypos = r(N)
        local b = _b[dx_wstr17]
        local s = _se[dx_wstr17]
        local t = `b' / `s'
        local p = 2 * ttail(e(df_r), abs(`t'))
        post `didpost' ("cityyear_prewashyear_fe") ("dx_wstr17") ("`y'") (`b') (`s') (`t') (`p') (e(N)) (`ypos')
    }
}

postclose `didpost'
use `didout', clear
gen sig_10 = abs(t_stat) >= 1.645 if !missing(t_stat)
gen sig_05 = abs(t_stat) >= 1.96 if !missing(t_stat)
gen sig_01 = abs(t_stat) >= 2.576 if !missing(t_stat)
order spec exposure y_name coef se t_stat p_value N y_positive sig_10 sig_05 sig_01
sort y_name exposure spec
export delimited using "$V3/results/stata/did_data_exchange_actual_data_asset_validation_v1_did.csv", replace
save "$V3/results/stata/did_data_exchange_actual_data_asset_validation_v1_did.dta", replace

display _newline "=== Coverage ==="
import delimited using "$V3/results/stata/did_data_exchange_actual_data_asset_validation_v1_coverage.csv", clear varnames(1)
list, abbreviate(24)

display _newline "=== Measurement validation highlights ==="
use "$V3/results/stata/did_data_exchange_actual_data_asset_validation_v1_measurement.dta", clear
list y_name x_name coef t_stat p_value N y_positive x_positive both_positive corr if inlist(x_name, "DU_kw", "asset_trade_kw", "strict_at_kw", "asset_trade_ratio", "strict_at_ratio"), sepby(y_name) abbreviate(24)

display _newline "=== Exploratory actual-data-asset DID ==="
use "$V3/results/stata/did_data_exchange_actual_data_asset_validation_v1_did.dta", clear
list spec exposure y_name coef t_stat p_value N y_positive, sepby(y_name exposure) abbreviate(28)

log close
