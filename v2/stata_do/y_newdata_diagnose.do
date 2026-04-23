clear all
set more off
capture log close

log using "/Users/mac/computerscience/0做完了/15会计研究/v2/results/y_newdata_diagnose.log", replace text

display "=== New-data candidate Y diagnose ==="
display "Sample: reg_sample_y_newdata.dta"
display "Regressor: lagged DU_kw and DU_llm, aligned with main H1"

capture which reghdfe
if _rc {
    display as error "reghdfe not installed"
    exit 199
}

use "/Users/mac/computerscience/0做完了/15会计研究/v1/data_stata/reg_sample_y_newdata.dta", clear

sort Stkcd_num year_num
tsset Stkcd_num year_num

local ctrls "Size Lev ROA TobinQ Age Growth IndepRatio Dual Top1Share SOE CFO"
local ylist "CS_Spread HL_Range_year PEAD20_signed PEAD60_signed PEAD20_abs PEAD60_abs EA_CAR02 EA_absCAR02 SUE_price SUE_abs_price AF_Dispersion AF_Coverage AF_Error_abs AF_Optimism Bench_Accuracy Bench_Optimism Bench_BuyShare NCSKEW_off DUVOL_off CRASH_off"

foreach du in DU_kw DU_llm {
    capture drop `du'_lag
    gen `du'_lag = L.`du'
}

tempfile step1
tempname posth
postfile `posth' str24 y_name str8 du_measure str12 expected_sign ///
    double coef se t_stat p_value N str12 status str20 baseline_direction_match ///
    using `step1', replace

foreach du in DU_kw DU_llm {
    local dulag = "`du'_lag"

    quietly count if !missing(PriceDelay, `dulag')
    if r(N) == 0 {
        post `posth' ("PriceDelay") ("`du'") ("negative") (.) (.) (.) (.) (0) ("unavailable") ("baseline")
        local price_sign_`du' = 0
    }
    else {
        quietly reghdfe PriceDelay `dulag' `ctrls', absorb(Stkcd_num year_num) cluster(IndYear_num)
        local coef = _b[`dulag']
        local se = _se[`dulag']
        local tval = `coef' / `se'
        local pval = 2 * ttail(e(df_r), abs(`tval'))
        local price_sign_`du' = 0
        if `coef' > 0 local price_sign_`du' = 1
        if `coef' < 0 local price_sign_`du' = -1
        local status = "fail"
        if `coef' < 0 & abs(`tval') >= 1.96 local status = "pass"
        else if `coef' < 0 & abs(`tval') >= 1.50 local status = "marginal"
        post `posth' ("PriceDelay") ("`du'") ("negative") (`coef') (`se') (`tval') (`pval') (e(N)) ("`status'") ("baseline")
    }

    quietly count if !missing(SYNCH, `dulag')
    if r(N) == 0 {
        post `posth' ("SYNCH") ("`du'") ("positive") (.) (.) (.) (.) (0) ("unavailable") ("baseline")
    }
    else {
        quietly reghdfe SYNCH `dulag' `ctrls', absorb(Stkcd_num year_num) cluster(IndYear_num)
        local coef = _b[`dulag']
        local se = _se[`dulag']
        local tval = `coef' / `se'
        local pval = 2 * ttail(e(df_r), abs(`tval'))
        local status = "fail"
        if `coef' > 0 & abs(`tval') >= 1.96 local status = "pass"
        else if `coef' > 0 & abs(`tval') >= 1.50 local status = "marginal"
        post `posth' ("SYNCH") ("`du'") ("positive") (`coef') (`se') (`tval') (`pval') (e(N)) ("`status'") ("baseline")
    }
}

foreach y of local ylist {
    local expected = "negative"
    if inlist("`y'", "EA_CAR02", "AF_Coverage", "Bench_BuyShare") {
        local expected = "positive"
    }
    if inlist("`y'", "SUE_price", "AF_Optimism", "Bench_Optimism") {
        local expected = "mixed"
    }

    foreach du in DU_kw DU_llm {
        local dulag = "`du'_lag"
        local baseline_match = "mismatch"

        capture confirm variable `y'
        if _rc {
            post `posth' ("`y'") ("`du'") ("`expected'") (.) (.) (.) (.) (0) ("unavailable") ("unavailable")
            continue
        }

        quietly count if !missing(`y', `dulag')
        if r(N) == 0 {
            post `posth' ("`y'") ("`du'") ("`expected'") (.) (.) (.) (.) (0) ("unavailable") ("unavailable")
            continue
        }

        capture noisily quietly reghdfe `y' `dulag' `ctrls', absorb(Stkcd_num year_num) cluster(IndYear_num)
        if _rc {
            post `posth' ("`y'") ("`du'") ("`expected'") (.) (.) (.) (.) (0) ("unavailable") ("unavailable")
            continue
        }

        local coef = _b[`dulag']
        local se = _se[`dulag']
        local tval = `coef' / `se'
        local pval = 2 * ttail(e(df_r), abs(`tval'))

        local direction_ok = 0
        if "`expected'" == "positive" & `coef' > 0 local direction_ok = 1
        if "`expected'" == "negative" & `coef' < 0 local direction_ok = 1

        local coef_sign = 0
        if `coef' > 0 local coef_sign = 1
        if `coef' < 0 local coef_sign = -1

        if `price_sign_`du'' == 0 {
            local baseline_match = "zero_baseline"
        }
        else if `coef_sign' == `price_sign_`du'' {
            local baseline_match = "match_delay"
        }
        else {
            local baseline_match = "mismatch"
        }

        local status = "fail"
        if "`expected'" == "mixed" {
            local status = "informative"
        }
        else if `direction_ok' == 1 & abs(`tval') >= 1.96 {
            local status = "pass"
        }
        else if `direction_ok' == 1 & abs(`tval') >= 1.50 {
            local status = "marginal"
        }

        post `posth' ("`y'") ("`du'") ("`expected'") (`coef') (`se') (`tval') (`pval') (e(N)) ("`status'") ("`baseline_match'")
    }
}

postclose `posth'

use `step1', clear
order y_name du_measure expected_sign coef se t_stat p_value N status baseline_direction_match
export delimited using "/Users/mac/computerscience/0做完了/15会计研究/v2/results/y_newdata_step1.csv", replace
save "/Users/mac/computerscience/0做完了/15会计研究/v2/results/y_newdata_step1.dta", replace

display _newline "=== New-data summary by status ==="
tab status

log close
