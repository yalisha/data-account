clear all
set more off
capture log close

log using "/Users/mac/computerscience/0做完了/15会计研究/v3/results/broad_y_screen_stata.log", replace text

display "=== Broad Y screen: DU_kw only, non-pricing-efficiency candidates ==="
display "Spec: Y_t = L.DU_kw + controls + firm FE + year FE, cluster(IndYear_num)"

capture which reghdfe
if _rc {
    display as error "reghdfe not installed"
    exit 199
}

use "/Users/mac/computerscience/0做完了/15会计研究/v3/results/reg_sample_broad_y_candidates.dta", clear
sort Stkcd_num year_num
tsset Stkcd_num year_num

capture drop DU_kw_lag
gen DU_kw_lag = L.DU_kw

local ctrls "Size Lev ROA TobinQ Age Growth IndepRatio Dual Top1Share SOE CFO"

tempfile out
tempname posth
postfile `posth' str24 y_name str16 family str8 expected ///
    double coef se t_stat p_value N str12 status ///
    using `out', replace

local ylist "TFP CashFlowVol SCConc CustConc SuppConc CustHHI InvestIneff FinAsset absDA AuditFee InstStable InstHold SA ForecastDisp FcstAcc ReportFreq RatingDisp Analyst RetVol IdioVol NCSKEW DUVOL"

foreach y of local ylist {
    local family "other"
    local expected "negative"

    if inlist("`y'", "TFP") {
        local family "productivity"
        local expected "positive"
    }
    if inlist("`y'", "CashFlowVol", "RetVol", "IdioVol", "NCSKEW", "DUVOL") {
        local family "resilience"
        local expected "negative"
    }
    if inlist("`y'", "SCConc", "CustConc", "SuppConc", "CustHHI") {
        local family "supply_chain"
        local expected "negative"
    }
    if inlist("`y'", "InvestIneff", "FinAsset") {
        local family "capital_allocation"
        local expected "negative"
    }
    if inlist("`y'", "absDA", "AuditFee") {
        local family "accounting_quality"
        local expected "negative"
    }
    if inlist("`y'", "InstStable", "InstHold") {
        local family "investor_base"
        local expected "positive"
    }
    if inlist("`y'", "SA") {
        local family "financing_constraint"
        local expected "negative"
    }
    if inlist("`y'", "ForecastDisp", "RatingDisp") {
        local family "analyst_info"
        local expected "negative"
    }
    if inlist("`y'", "FcstAcc", "ReportFreq", "Analyst") {
        local family "analyst_info"
        local expected "positive"
    }

    capture confirm variable `y'
    if _rc {
        post `posth' ("`y'") ("`family'") ("`expected'") (.) (.) (.) (.) (0) ("unavailable")
        continue
    }

    quietly count if !missing(`y', DU_kw_lag, Size, Lev, ROA, TobinQ, Age, Growth, IndepRatio, Dual, Top1Share, SOE, CFO, Stkcd_num, year_num, IndYear_num)
    if r(N) == 0 {
        post `posth' ("`y'") ("`family'") ("`expected'") (.) (.) (.) (.) (0) ("unavailable")
        continue
    }

    capture noisily quietly reghdfe `y' DU_kw_lag `ctrls', absorb(Stkcd_num year_num) cluster(IndYear_num)
    if _rc {
        post `posth' ("`y'") ("`family'") ("`expected'") (.) (.) (.) (.) (0) ("unavailable")
        continue
    }

    local coef = _b[DU_kw_lag]
    local se = _se[DU_kw_lag]
    local tval = `coef' / `se'
    local pval = 2 * ttail(e(df_r), abs(`tval'))

    local direction_ok = 0
    if "`expected'" == "positive" & `coef' > 0 local direction_ok = 1
    if "`expected'" == "negative" & `coef' < 0 local direction_ok = 1

    local status = "fail"
    if `direction_ok' == 1 & abs(`tval') >= 1.96 local status = "pass"
    else if `direction_ok' == 1 & abs(`tval') >= 1.50 local status = "marginal"

    post `posth' ("`y'") ("`family'") ("`expected'") (`coef') (`se') (`tval') (`pval') (e(N)) ("`status'")
}

postclose `posth'
use `out', clear
order family y_name expected coef se t_stat p_value N status
sort family y_name
export delimited using "/Users/mac/computerscience/0做完了/15会计研究/v3/results/broad_y_screen_stata.csv", replace
save "/Users/mac/computerscience/0做完了/15会计研究/v3/results/broad_y_screen_stata.dta", replace

display _newline "=== Broad screen status ==="
tab status
display _newline "=== Passed or marginal ==="
list family y_name expected coef t_stat p_value N status if inlist(status, "pass", "marginal"), sepby(family)

log close
