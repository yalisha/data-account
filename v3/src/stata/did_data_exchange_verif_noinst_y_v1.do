clear all
set more off
capture log close

global V3 "/Users/mac/computerscience/0做完了/15会计研究/v3"

log using "$V3/results/logs/did_data_exchange_verif_noinst_y_v1.log", replace text

display "=== Data exchange DDD with no-institution verifiable disclosure Y ==="
display "Main idea: remove institution-name terms such as data exchange / data trading center from Y."

capture which reghdfe
if _rc {
    display as error "reghdfe not installed"
    exit 199
}

* ------------------------------------------------------------------
* 0. Merge new no-institution outcomes
* ------------------------------------------------------------------

tempfile noinst
import delimited using "$V3/results/data/verifiable_disclosure_y_noinst_v1.csv", clear varnames(1)
capture destring stkcd, gen(Stkcd) force
capture confirm variable Stkcd
if _rc {
    rename stkcd Stkcd
}
rename year year_num
duplicates drop Stkcd year_num, force
save `noinst', replace

tempfile asset
import delimited using "$V3/results/data/data_asset_from_panel.csv", clear varnames(1)
capture rename stkcd_num Stkcd_num
capture rename dataasset DataAsset
destring Stkcd_num year_num DataAsset, replace force
gen Stkcd = Stkcd_num
keep Stkcd year_num DataAsset
duplicates drop Stkcd year_num, force
save `asset', replace

use "$V3/results/data/did_data_exchange_mismatch_ddd_v2_pre2017_panel.dta", clear
merge 1:1 Stkcd year_num using `noinst', keep(master match) nogen
merge 1:1 Stkcd year_num using `asset', keep(master match) nogen

foreach v in inst_term_kw asset_trade_noinst_kw strict_noinst_kw acct_kw rights_kw pricing_kw product_tx_noinst_kw verif_noinst_kw {
    replace `v' = 0 if missing(`v')
}

foreach v in inst_term_raw asset_trade_noinst_raw strict_noinst_raw acct_raw rights_raw pricing_raw product_tx_noinst_raw verif_noinst_raw {
    replace `v' = 0 if missing(`v')
}

capture drop asset_trade_noinst_ratio strict_noinst_ratio verif_noinst_ratio product_tx_noinst_ratio
capture drop asset_trade_noinst_share strict_noinst_share verif_noinst_share product_tx_noinst_share
capture drop asset_trade_noinst_pos strict_noinst_pos verif_noinst_pos product_tx_noinst_pos inst_term_pos

gen asset_trade_noinst_ratio = asset_trade_noinst_kw / (DU_kw + 1) if !missing(asset_trade_noinst_kw, DU_kw)
gen strict_noinst_ratio = strict_noinst_kw / (DU_kw + 1) if !missing(strict_noinst_kw, DU_kw)
gen verif_noinst_ratio = verif_noinst_kw / (DU_kw + 1) if !missing(verif_noinst_kw, DU_kw)
gen product_tx_noinst_ratio = product_tx_noinst_kw / (DU_kw + 1) if !missing(product_tx_noinst_kw, DU_kw)

gen asset_trade_noinst_share = asset_trade_noinst_kw / (asset_trade_noinst_kw + DU_kw + 1) if !missing(asset_trade_noinst_kw, DU_kw)
gen strict_noinst_share = strict_noinst_kw / (strict_noinst_kw + DU_kw + 1) if !missing(strict_noinst_kw, DU_kw)
gen verif_noinst_share = verif_noinst_kw / (verif_noinst_kw + DU_kw + 1) if !missing(verif_noinst_kw, DU_kw)
gen product_tx_noinst_share = product_tx_noinst_kw / (product_tx_noinst_kw + DU_kw + 1) if !missing(product_tx_noinst_kw, DU_kw)

gen asset_trade_noinst_pos = (asset_trade_noinst_raw > 0) if !missing(asset_trade_noinst_raw)
gen strict_noinst_pos = (strict_noinst_raw > 0) if !missing(strict_noinst_raw)
gen verif_noinst_pos = (verif_noinst_raw > 0) if !missing(verif_noinst_raw)
gen product_tx_noinst_pos = (product_tx_noinst_raw > 0) if !missing(product_tx_noinst_raw)
gen inst_term_pos = (inst_term_raw > 0) if !missing(inst_term_raw)

capture drop DataAsset0 DataAsset_ln BookEntry
gen DataAsset0 = DataAsset
replace DataAsset0 = 0 if missing(DataAsset0)
gen DataAsset_ln = ln(1 + DataAsset0)
gen BookEntry = (DataAsset0 > 0) if !missing(DataAsset0)

capture drop city_year_id washany_year_id wstr_year_id ind2024_id indyear_id post2024 wany2024 wstr2024 ind_id
egen city_year_id = group(city_id year_num)
egen washany_year_id = group(PreWashAny1617 year_num)
egen wstr_year_id = group(PreStrictWashAny1617 year_num)
egen ind2024_id = group(Ind2 year_num) if year_num == 2024
replace ind2024_id = 0 if missing(ind2024_id)
egen indyear_id = group(Ind2 year_num)
egen ind_id = group(Ind2)

gen post2024 = (year_num == 2024)
gen wany2024 = PreWashAny1617 * post2024
gen wstr2024 = PreStrictWashAny1617 * post2024

save "$V3/results/data/did_data_exchange_verif_noinst_y_v1_panel.dta", replace

* ------------------------------------------------------------------
* 1. Coverage summary
* ------------------------------------------------------------------

preserve
tempfile sumout
tempname sumpost
postfile `sumpost' str32 varname double N mean sd p50 positives positive_share using `sumout', replace
foreach v in DU_kw asset_trade_kw strict_at_kw inst_term_kw asset_trade_noinst_kw strict_noinst_kw verif_noinst_kw acct_kw rights_kw pricing_kw product_tx_noinst_kw asset_trade_noinst_share strict_noinst_share verif_noinst_share {
    quietly summarize `v' if inrange(year_num, 2018, 2024), detail
    local n = r(N)
    local mu = r(mean)
    local sd = r(sd)
    local p50 = r(p50)
    quietly count if inrange(year_num, 2018, 2024) & `v' > 0 & !missing(`v')
    local pos = r(N)
    local pshare = `pos' / `n'
    post `sumpost' ("`v'") (`n') (`mu') (`sd') (`p50') (`pos') (`pshare')
}
postclose `sumpost'
use `sumout', clear
export delimited using "$V3/results/stata/did_data_exchange_verif_noinst_y_v1_summary.csv", replace
restore

* ------------------------------------------------------------------
* 2. Static DDD specs
* ------------------------------------------------------------------

local ctrls "Size Lev ROA TobinQ Age Growth IndepRatio Dual Top1Share SOE CFO InstHold Analyst"
local ylist "DU_kw asset_trade_kw strict_at_kw inst_term_kw asset_trade_noinst_kw strict_noinst_kw verif_noinst_kw acct_kw rights_kw pricing_kw product_tx_noinst_kw asset_trade_noinst_share strict_noinst_share verif_noinst_share product_tx_noinst_share"

tempfile staticout
tempname staticpost
postfile `staticpost' str36 spec str24 exposure str32 y_name double coef se t_stat p_value N positives using `staticout', replace

foreach y of local ylist {
    local regctrls ""
    foreach c of local ctrls {
        capture confirm variable `c'
        if !_rc & "`c'" != "`y'" {
            local regctrls "`regctrls' `c'"
        }
    }

    foreach x in dx_wany17 dx_wstr17 {
        quietly count if inrange(year_num, 2018, 2024) & !missing(`y', DID_city, `x', firm_id, year_num, city_id)
        quietly count if inrange(year_num, 2018, 2024) & !missing(`y', DID_city, `x', firm_id, year_num, city_id) & `x' > 0
        local xpos = r(N)

        capture noisily quietly reghdfe `y' DID_city `x' `regctrls' if inrange(year_num, 2018, 2024), absorb(firm_id year_num) cluster(city_id)
        if _rc {
            post `staticpost' ("baseline") ("`x'") ("`y'") (.) (.) (.) (.) (0) (`xpos')
        }
        else {
            local b = _b[`x']
            local s = _se[`x']
            local t = `b' / `s'
            local p = 2 * ttail(e(df_r), abs(`t'))
            post `staticpost' ("baseline") ("`x'") ("`y'") (`b') (`s') (`t') (`p') (e(N)) (`xpos')
        }

        capture noisily quietly reghdfe `y' DID_city `x' `regctrls' if inrange(year_num, 2018, 2023), absorb(firm_id year_num) cluster(city_id)
        if _rc {
            post `staticpost' ("drop_year_2024") ("`x'") ("`y'") (.) (.) (.) (.) (0) (`xpos')
        }
        else {
            local b = _b[`x']
            local s = _se[`x']
            local t = `b' / `s'
            local p = 2 * ttail(e(df_r), abs(`t'))
            post `staticpost' ("drop_year_2024") ("`x'") ("`y'") (`b') (`s') (`t') (`p') (e(N)) (`xpos')
        }

        capture noisily quietly reghdfe `y' DID_city `x' `regctrls' if inrange(year_num, 2018, 2024), absorb(firm_id year_num ind2024_id) cluster(city_id)
        if _rc {
            post `staticpost' ("industry_2024_fe") ("`x'") ("`y'") (.) (.) (.) (.) (0) (`xpos')
        }
        else {
            local b = _b[`x']
            local s = _se[`x']
            local t = `b' / `s'
            local p = 2 * ttail(e(df_r), abs(`t'))
            post `staticpost' ("industry_2024_fe") ("`x'") ("`y'") (`b') (`s') (`t') (`p') (e(N)) (`xpos')
        }

        capture noisily quietly reghdfe `y' DID_city `x' `regctrls' if inrange(year_num, 2018, 2024), absorb(firm_id year_num indyear_id) cluster(city_id)
        if _rc {
            post `staticpost' ("industry_year_fe") ("`x'") ("`y'") (.) (.) (.) (.) (0) (`xpos')
        }
        else {
            local b = _b[`x']
            local s = _se[`x']
            local t = `b' / `s'
            local p = 2 * ttail(e(df_r), abs(`t'))
            post `staticpost' ("industry_year_fe") ("`x'") ("`y'") (`b') (`s') (`t') (`p') (e(N)) (`xpos')
        }
    }

    capture noisily quietly reghdfe `y' DID_city dx_wany17 wany2024 `regctrls' if inrange(year_num, 2018, 2024), absorb(firm_id year_num) cluster(city_id)
    if _rc {
        post `staticpost' ("add_prewash_x_2024") ("dx_wany17") ("`y'") (.) (.) (.) (.) (0) (.)
    }
    else {
        quietly count if e(sample) & dx_wany17 > 0
        local xpos = r(N)
        local b = _b[dx_wany17]
        local s = _se[dx_wany17]
        local t = `b' / `s'
        local p = 2 * ttail(e(df_r), abs(`t'))
        post `staticpost' ("add_prewash_x_2024") ("dx_wany17") ("`y'") (`b') (`s') (`t') (`p') (e(N)) (`xpos')
    }

    capture noisily quietly reghdfe `y' DID_city dx_wstr17 wstr2024 `regctrls' if inrange(year_num, 2018, 2024), absorb(firm_id year_num) cluster(city_id)
    if _rc {
        post `staticpost' ("add_prewash_x_2024") ("dx_wstr17") ("`y'") (.) (.) (.) (.) (0) (.)
    }
    else {
        quietly count if e(sample) & dx_wstr17 > 0
        local xpos = r(N)
        local b = _b[dx_wstr17]
        local s = _se[dx_wstr17]
        local t = `b' / `s'
        local p = 2 * ttail(e(df_r), abs(`t'))
        post `staticpost' ("add_prewash_x_2024") ("dx_wstr17") ("`y'") (`b') (`s') (`t') (`p') (e(N)) (`xpos')
    }

    capture noisily quietly reghdfe `y' dx_wany17 `regctrls' if inrange(year_num, 2018, 2024), absorb(firm_id city_year_id washany_year_id) cluster(city_id)
    if _rc {
        post `staticpost' ("cityyear_prewashyear_fe") ("dx_wany17") ("`y'") (.) (.) (.) (.) (0) (.)
    }
    else {
        quietly count if e(sample) & dx_wany17 > 0
        local xpos = r(N)
        local b = _b[dx_wany17]
        local s = _se[dx_wany17]
        local t = `b' / `s'
        local p = 2 * ttail(e(df_r), abs(`t'))
        post `staticpost' ("cityyear_prewashyear_fe") ("dx_wany17") ("`y'") (`b') (`s') (`t') (`p') (e(N)) (`xpos')
    }

    capture noisily quietly reghdfe `y' dx_wstr17 `regctrls' if inrange(year_num, 2018, 2024), absorb(firm_id city_year_id wstr_year_id) cluster(city_id)
    if _rc {
        post `staticpost' ("cityyear_prewashyear_fe") ("dx_wstr17") ("`y'") (.) (.) (.) (.) (0) (.)
    }
    else {
        quietly count if e(sample) & dx_wstr17 > 0
        local xpos = r(N)
        local b = _b[dx_wstr17]
        local s = _se[dx_wstr17]
        local t = `b' / `s'
        local p = 2 * ttail(e(df_r), abs(`t'))
        post `staticpost' ("cityyear_prewashyear_fe") ("dx_wstr17") ("`y'") (`b') (`s') (`t') (`p') (e(N)) (`xpos')
    }
}

postclose `staticpost'
use `staticout', clear
gen sig_10 = abs(t_stat) >= 1.645 if !missing(t_stat)
gen sig_05 = abs(t_stat) >= 1.96 if !missing(t_stat)
gen sig_01 = abs(t_stat) >= 2.576 if !missing(t_stat)
order spec exposure y_name coef se t_stat p_value N positives sig_10 sig_05 sig_01
sort y_name exposure spec
export delimited using "$V3/results/stata/did_data_exchange_verif_noinst_y_v1_static.csv", replace
save "$V3/results/stata/did_data_exchange_verif_noinst_y_v1_static.dta", replace

* ------------------------------------------------------------------
* 3. Event/pretrend screen for key new outcomes
* ------------------------------------------------------------------

use "$V3/results/data/did_data_exchange_verif_noinst_y_v1_panel.dta", clear

local evmain "evt_m3 evt_m2 evt_p0 evt_p1 evt_p2 evt_p3"
local ev_ylist "asset_trade_noinst_kw strict_noinst_kw verif_noinst_kw asset_trade_noinst_share strict_noinst_share verif_noinst_share product_tx_noinst_kw inst_term_kw"

tempfile evout
tempname evpost
postfile `evpost' str20 exposure str32 y_name str8 event_time double coef se t_stat p_value N pretrend_p using `evout', replace

foreach y of local ev_ylist {
    local regctrls ""
    foreach c of local ctrls {
        capture confirm variable `c'
        if !_rc & "`c'" != "`y'" {
            local regctrls "`regctrls' `c'"
        }
    }

    foreach block in "wany17 ewany17" "wstr17 ewstr17" {
        tokenize "`block'"
        local ename "`1'"
        local pref "`2'"

        capture noisily quietly reghdfe `y' `evmain' `pref'_m3 `pref'_m2 `pref'_p0 `pref'_p1 `pref'_p2 `pref'_p3 `regctrls' if inrange(year_num, 2018, 2024), absorb(firm_id year_num) cluster(city_id)
        if _rc {
            foreach e in m3 m2 p0 p1 p2 p3 {
                post `evpost' ("`ename'") ("`y'") ("`e'") (.) (.) (.) (.) (0) (.)
            }
            continue
        }

        capture test `pref'_m3 `pref'_m2
        if _rc local pre_p = .
        else local pre_p = r(p)

        foreach e in m3 m2 p0 p1 p2 p3 {
            capture local b = _b[`pref'_`e']
            if _rc {
                post `evpost' ("`ename'") ("`y'") ("`e'") (.) (.) (.) (.) (e(N)) (`pre_p')
                continue
            }
            local s = _se[`pref'_`e']
            local t = `b' / `s'
            local p = 2 * ttail(e(df_r), abs(`t'))
            post `evpost' ("`ename'") ("`y'") ("`e'") (`b') (`s') (`t') (`p') (e(N)) (`pre_p')
        }
    }
}

postclose `evpost'
use `evout', clear
gen sig_10 = abs(t_stat) >= 1.645 if !missing(t_stat)
gen sig_05 = abs(t_stat) >= 1.96 if !missing(t_stat)
gen sig_01 = abs(t_stat) >= 2.576 if !missing(t_stat)
order exposure y_name event_time coef se t_stat p_value N pretrend_p sig_10 sig_05 sig_01
sort y_name exposure event_time
export delimited using "$V3/results/stata/did_data_exchange_verif_noinst_y_v1_event.csv", replace
save "$V3/results/stata/did_data_exchange_verif_noinst_y_v1_event.dta", replace

preserve
keep exposure y_name pretrend_p
duplicates drop
sort y_name exposure
export delimited using "$V3/results/stata/did_data_exchange_verif_noinst_y_v1_pretrend_joint.csv", replace
restore

* ------------------------------------------------------------------
* 4. Actual data-asset validation, 2024 cross-section
* ------------------------------------------------------------------

use "$V3/results/data/did_data_exchange_verif_noinst_y_v1_panel.dta", clear

local valx "DU_kw asset_trade_kw strict_at_kw inst_term_kw asset_trade_noinst_kw strict_noinst_kw verif_noinst_kw asset_trade_noinst_pos strict_noinst_pos verif_noinst_pos asset_trade_noinst_share strict_noinst_share verif_noinst_share acct_kw rights_kw pricing_kw product_tx_noinst_kw"

tempfile valout
tempname valpost
postfile `valpost' str20 y_name str32 x_name double coef se t_stat p_value N y_positive x_positive both_positive corr using `valout', replace

foreach y in BookEntry DataAsset_ln {
    foreach x of local valx {
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

        capture noisily quietly reghdfe `y' `x' `regctrls' if year_num == 2024, absorb(ind_id) vce(robust)
        if _rc {
            post `valpost' ("`y'") ("`x'") (.) (.) (.) (.) (0) (`ypos') (`xpos') (`bothpos') (`rho')
            continue
        }

        capture local b = _b[`x']
        if _rc {
            post `valpost' ("`y'") ("`x'") (.) (.) (.) (.) (e(N)) (`ypos') (`xpos') (`bothpos') (`rho')
            continue
        }
        local s = _se[`x']
        local t = `b' / `s'
        local p = 2 * ttail(e(df_r), abs(`t'))
        post `valpost' ("`y'") ("`x'") (`b') (`s') (`t') (`p') (e(N)) (`ypos') (`xpos') (`bothpos') (`rho')
    }
}

postclose `valpost'
use `valout', clear
gen sig_10 = abs(t_stat) >= 1.645 if !missing(t_stat)
gen sig_05 = abs(t_stat) >= 1.96 if !missing(t_stat)
gen sig_01 = abs(t_stat) >= 2.576 if !missing(t_stat)
order y_name x_name coef se t_stat p_value N y_positive x_positive both_positive corr sig_10 sig_05 sig_01
sort y_name x_name
export delimited using "$V3/results/stata/did_data_exchange_verif_noinst_y_v1_dataasset_validation.csv", replace
save "$V3/results/stata/did_data_exchange_verif_noinst_y_v1_dataasset_validation.dta", replace

display _newline "=== Static highlights ==="
use "$V3/results/stata/did_data_exchange_verif_noinst_y_v1_static.dta", clear
list spec exposure y_name coef t_stat p_value N if inlist(y_name, "asset_trade_noinst_kw", "strict_noinst_kw", "verif_noinst_kw", "asset_trade_noinst_share", "strict_noinst_share", "verif_noinst_share") & inlist(spec, "baseline", "cityyear_prewashyear_fe", "drop_year_2024"), sepby(y_name exposure) abbreviate(32)

display _newline "=== Pretrend p-values ==="
import delimited using "$V3/results/stata/did_data_exchange_verif_noinst_y_v1_pretrend_joint.csv", clear varnames(1)
list, abbreviate(32)

display _newline "=== DataAsset validation highlights ==="
use "$V3/results/stata/did_data_exchange_verif_noinst_y_v1_dataasset_validation.dta", clear
list y_name x_name coef t_stat p_value N y_positive x_positive both_positive corr if y_name == "BookEntry" & inlist(x_name, "asset_trade_kw", "strict_at_kw", "inst_term_kw", "asset_trade_noinst_kw", "strict_noinst_kw", "verif_noinst_kw", "asset_trade_noinst_share", "strict_noinst_share", "verif_noinst_share"), sepby(y_name) abbreviate(32)

log close
