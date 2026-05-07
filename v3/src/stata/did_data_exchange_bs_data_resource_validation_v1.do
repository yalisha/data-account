clear all
set more off
capture log close

global V3 "/Users/mac/computerscience/0做完了/15会计研究/v3"

log using "$V3/results/logs/did_data_exchange_bs_data_resource_validation_v1.log", replace text

display "=== Balance-sheet data-resource booking validation for v3 ==="
display "Input 1: did_data_exchange_verif_noinst_y_v1_panel.dta"
display "Input 2: balance_sheet_data_resource_annual.csv"
display "Role: external validation of verifiable disclosure Y; not a long-panel main Y."

capture which reghdfe
if _rc {
    display as error "reghdfe not installed"
    exit 199
}

* ------------------------------------------------------------------
* 0. Import annual balance-sheet data-resource booking fields
* ------------------------------------------------------------------

tempfile bs
import delimited using "$V3/results/data/balance_sheet_data_resource_annual.csv", clear varnames(1) stringcols(_all)

capture rename stkcd Stkcd_raw
capture rename Stkcd Stkcd_raw
capture rename shortname ShortName_bs
capture rename ShortName ShortName_bs
capture rename accper Accper_bs
capture rename Accper Accper_bs
capture rename typrep Typrep_bs
capture rename Typrep Typrep_bs

gen Stkcd = real(Stkcd_raw)
destring year_num datares_inventory datares_intangible datares_devexp ///
    bookdataresourceamount bookdataresource lnbookdataresource ///
    bookdataresourceratio totalassets, replace force

rename datares_inventory DataRes_inventory_bs
rename datares_intangible DataRes_intangible_bs
rename datares_devexp DataRes_devexp_bs
rename bookdataresourceamount BookDataResourceAmount_bs
rename bookdataresource BookDataResource_bs
rename lnbookdataresource lnBookDataResource_bs
rename bookdataresourceratio BookDataResourceRatio_bs
rename totalassets TotalAssets_bs

keep Stkcd year_num Accper_bs Typrep_bs ShortName_bs ///
    DataRes_inventory_bs DataRes_intangible_bs DataRes_devexp_bs ///
    BookDataResourceAmount_bs BookDataResource_bs lnBookDataResource_bs ///
    BookDataResourceRatio_bs TotalAssets_bs
duplicates drop Stkcd year_num, force
save `bs', replace

* ------------------------------------------------------------------
* 1. Merge into current no-institution validation panel
* ------------------------------------------------------------------

use "$V3/results/data/did_data_exchange_verif_noinst_y_v1_panel.dta", clear
merge 1:1 Stkcd year_num using `bs', keep(master match) nogen

foreach v in DataRes_inventory_bs DataRes_intangible_bs DataRes_devexp_bs ///
    BookDataResourceAmount_bs BookDataResource_bs lnBookDataResource_bs ///
    BookDataResourceRatio_bs {
    replace `v' = 0 if missing(`v')
}

capture drop BookEntry_old DataAsset_ln_old DataAsset0_old
gen BookEntry_old = BookEntry
gen DataAsset_ln_old = DataAsset_ln
gen DataAsset0_old = DataAsset0

label variable DataRes_inventory_bs "Balance-sheet data resources in inventory"
label variable DataRes_intangible_bs "Balance-sheet data resources in intangible assets"
label variable DataRes_devexp_bs "Balance-sheet data resources in development expenditure"
label variable BookDataResourceAmount_bs "Balance-sheet booked data-resource amount"
label variable BookDataResource_bs "Positive balance-sheet data-resource booking"
label variable lnBookDataResource_bs "ln(1 + balance-sheet booked data-resource amount)"
label variable BookDataResourceRatio_bs "Booked data-resource amount scaled by total assets"
label variable BookEntry_old "Old text/parquet DataAsset positive indicator"

capture drop city_year_id washany_year_id wstr_year_id ind_id
egen city_year_id = group(city_id year_num)
egen washany_year_id = group(PreWashAny1617 year_num)
egen wstr_year_id = group(PreStrictWashAny1617 year_num)
egen ind_id = group(Ind2)

save "$V3/results/data/did_data_exchange_bs_data_resource_validation_v1_panel.dta", replace

* ------------------------------------------------------------------
* 2. Coverage and old-vs-new comparison
* ------------------------------------------------------------------

preserve
collapse (count) N=BookDataResource_bs ///
    (sum) positive=BookDataResource_bs ///
    (mean) positive_rate=BookDataResource_bs ///
    (sum) total_amount=BookDataResourceAmount_bs ///
    (mean) mean_amount=BookDataResourceAmount_bs ///
    (max) max_amount=BookDataResourceAmount_bs, by(year_num)
export delimited using "$V3/results/stata/did_data_exchange_bs_data_resource_validation_v1_coverage.csv", replace
restore

preserve
keep if year_num == 2024
contract BookEntry_old BookDataResource_bs
rename _freq firm_years
export delimited using "$V3/results/stata/did_data_exchange_bs_data_resource_validation_v1_old_new_crosstab_2024.csv", replace
restore

preserve
keep if year_num == 2024
contract asset_trade_noinst_pos strict_noinst_pos verif_noinst_pos BookDataResource_bs
rename _freq firm_years
export delimited using "$V3/results/stata/did_data_exchange_bs_data_resource_validation_v1_text_crosstab_2024.csv", replace
restore

* ------------------------------------------------------------------
* 3. 2024 measurement validation:
*    Do verifiable text measures predict balance-sheet data-resource booking?
* ------------------------------------------------------------------

local ctrls "Size Lev ROA TobinQ Age Growth IndepRatio Dual Top1Share SOE CFO InstHold Analyst"
local xlist "DU_kw asset_trade_kw strict_at_kw inst_term_kw asset_trade_noinst_kw strict_noinst_kw verif_noinst_kw acct_kw rights_kw pricing_kw product_tx_noinst_kw asset_trade_noinst_pos strict_noinst_pos verif_noinst_pos asset_trade_noinst_share strict_noinst_share verif_noinst_share product_tx_noinst_share"
local ylist "BookDataResource_bs lnBookDataResource_bs BookDataResourceRatio_bs"

tempfile validout
tempname validpost
postfile `validpost' str32 y_name str32 x_name double coef se t_stat p_value N y_positive x_positive both_positive corr using `validout', replace

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

        capture noisily quietly reghdfe `y' `x' `regctrls' if year_num == 2024, absorb(ind_id) vce(robust)
        if _rc {
            post `validpost' ("`y'") ("`x'") (.) (.) (.) (.) (0) (`ypos') (`xpos') (`bothpos') (`rho')
            continue
        }

        capture local b = _b[`x']
        if _rc {
            post `validpost' ("`y'") ("`x'") (.) (.) (.) (.) (e(N)) (`ypos') (`xpos') (`bothpos') (`rho')
            continue
        }
        local s = _se[`x']
        local t = `b' / `s'
        local p = 2 * ttail(e(df_r), abs(`t'))
        post `validpost' ("`y'") ("`x'") (`b') (`s') (`t') (`p') (e(N)) (`ypos') (`xpos') (`bothpos') (`rho')
    }
}

postclose `validpost'
use `validout', clear
gen sig_10 = abs(t_stat) >= 1.645 if !missing(t_stat)
gen sig_05 = abs(t_stat) >= 1.96 if !missing(t_stat)
gen sig_01 = abs(t_stat) >= 2.576 if !missing(t_stat)
order y_name x_name coef se t_stat p_value N y_positive x_positive both_positive corr sig_10 sig_05 sig_01
sort y_name x_name
export delimited using "$V3/results/stata/did_data_exchange_bs_data_resource_validation_v1_measurement.csv", replace
save "$V3/results/stata/did_data_exchange_bs_data_resource_validation_v1_measurement.dta", replace

* ------------------------------------------------------------------
* 4. Exploratory DDD with balance-sheet data-resource booking as Y.
*    Keep as diagnostic only because the Y appears only after the 2024 rule.
* ------------------------------------------------------------------

use "$V3/results/data/did_data_exchange_bs_data_resource_validation_v1_panel.dta", clear

tempfile didout
tempname didpost
postfile `didpost' str36 spec str24 exposure str32 y_name double coef se t_stat p_value N y_positive using `didout', replace

foreach y in BookDataResource_bs lnBookDataResource_bs BookDataResourceRatio_bs {
    local regctrls ""
    foreach c of local ctrls {
        capture confirm variable `c'
        if !_rc & "`c'" != "`y'" {
            local regctrls "`regctrls' `c'"
        }
    }

    foreach x in dx_wany17 dx_wstr17 {
        capture noisily quietly reghdfe `y' DID_city `x' `regctrls' if inrange(year_num, 2018, 2024), absorb(firm_id year_num) cluster(city_id)
        if _rc {
            post `didpost' ("baseline_firm_year_fe") ("`x'") ("`y'") (.) (.) (.) (.) (0) (.)
        }
        else {
            quietly count if e(sample) & `y' > 0
            local ypos = r(N)
            local b = _b[`x']
            local s = _se[`x']
            local t = `b' / `s'
            local p = 2 * ttail(e(df_r), abs(`t'))
            post `didpost' ("baseline_firm_year_fe") ("`x'") ("`y'") (`b') (`s') (`t') (`p') (e(N)) (`ypos')
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
export delimited using "$V3/results/stata/did_data_exchange_bs_data_resource_validation_v1_did.csv", replace
save "$V3/results/stata/did_data_exchange_bs_data_resource_validation_v1_did.dta", replace

display _newline "=== Balance-sheet data-resource coverage ==="
import delimited using "$V3/results/stata/did_data_exchange_bs_data_resource_validation_v1_coverage.csv", clear varnames(1)
list if year_num >= 2022, abbreviate(28)

display _newline "=== Old vs balance-sheet data-resource cross-tab, 2024 ==="
import delimited using "$V3/results/stata/did_data_exchange_bs_data_resource_validation_v1_old_new_crosstab_2024.csv", clear varnames(1)
list, abbreviate(28)

display _newline "=== Measurement validation highlights ==="
use "$V3/results/stata/did_data_exchange_bs_data_resource_validation_v1_measurement.dta", clear
gen highlight = 0
replace highlight = 1 if x_name == "DU_kw"
replace highlight = 1 if x_name == "asset_trade_noinst_kw"
replace highlight = 1 if x_name == "strict_noinst_kw"
replace highlight = 1 if x_name == "verif_noinst_kw"
replace highlight = 1 if x_name == "asset_trade_noinst_pos"
replace highlight = 1 if x_name == "strict_noinst_pos"
replace highlight = 1 if x_name == "verif_noinst_pos"
replace highlight = 1 if x_name == "asset_trade_noinst_share"
replace highlight = 1 if x_name == "strict_noinst_share"
replace highlight = 1 if x_name == "verif_noinst_share"
list y_name x_name coef t_stat p_value N y_positive x_positive both_positive corr if y_name == "BookDataResource_bs" & highlight == 1, sepby(y_name) abbreviate(32)

display _newline "=== Exploratory balance-sheet data-resource DID ==="
use "$V3/results/stata/did_data_exchange_bs_data_resource_validation_v1_did.dta", clear
list spec exposure y_name coef t_stat p_value N y_positive, sepby(y_name exposure) abbreviate(32)

log close
