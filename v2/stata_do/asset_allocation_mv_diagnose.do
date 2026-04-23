clear all
set more off
capture log close

log using "/Users/mac/computerscience/0做完了/15会计研究/v2/results/asset_allocation_mv_diagnose.log", replace text

display "=== Asset allocation MV Step 2 diagnose ==="
display "Gate rules:"
display "1) pass: |t| >= 1.96 and sign direction matches expectation"
display "2) marginal: 1.50 <= |t| < 1.96 and sign direction matches expectation"
display "3) fail: everything else"
display "Direction map used in this run:"
display "   positive to PriceDelay: FinRatio_* and Fin_to_RD"
display "   negative to PriceDelay: IntangibleRatio and DevOutlayRatio"

use "/Users/mac/computerscience/0做完了/15会计研究/v1/data_stata/reg_sample_asset_mv.dta", clear

tsset Stkcd_num year_num

local ctrls "Size Lev ROA TobinQ Age Growth IndepRatio Dual Top1Share SOE CFO"
local mvlist "FinRatio_narrow FinRatio_trading FinRatio_realEstate FinRatio_other FinRatio_v4 IntangibleRatio DevOutlayRatio FinRatio_v4_delta FinRatio_v4_vol3y Fin_to_RD"

foreach mv of local mvlist {
    capture drop `mv'_lag
    gen `mv'_lag = L.`mv'
}

tempfile step2
tempname posth
postfile `posth' str32 mv str8 expected_sign double coef se t_stat p_value N r2 str12 status using `step2', replace

foreach mv of local mvlist {
    local expected = "positive"
    if inlist("`mv'", "IntangibleRatio", "DevOutlayRatio") {
        local expected = "negative"
    }

    quietly count if !missing(`mv')
    local current_n = r(N)
    quietly count if !missing(`mv'_lag)
    local lag_n = r(N)

    display _newline "=== Step 2 candidate: `mv' ==="
    display "current non-missing = `current_n'; lag non-missing = `lag_n'"
    quietly reghdfe PriceDelay `mv'_lag `ctrls', absorb(Stkcd_num year_num) cluster(IndYear_num)

    local coef = _b[`mv'_lag]
    local se = _se[`mv'_lag]
    local tval = `coef' / `se'
    local pval = 2 * ttail(e(df_r), abs(`tval'))
    local nobs = e(N)
    local r2 = e(r2)
    local direction_ok = 0

    if "`expected'" == "positive" & `coef' > 0 {
        local direction_ok = 1
    }
    if "`expected'" == "negative" & `coef' < 0 {
        local direction_ok = 1
    }

    local status = "fail"
    if `direction_ok' == 1 & abs(`tval') >= 1.96 {
        local status = "pass"
    }
    else if `direction_ok' == 1 & abs(`tval') >= 1.50 {
        local status = "marginal"
    }

    display "coef = " %9.6f `coef' " ; se = " %9.6f `se' " ; t = " %9.3f `tval' " ; p = " %9.4f `pval'
    display "expected sign = `expected' ; status = `status'"

    post `posth' ("`mv'") ("`expected'") (`coef') (`se') (`tval') (`pval') (`nobs') (`r2') ("`status'")
}

postclose `posth'

use `step2', clear
order mv expected_sign coef se t_stat p_value N r2 status
export delimited using "/Users/mac/computerscience/0做完了/15会计研究/v2/results/asset_allocation_mv_step2.csv", replace
save "/Users/mac/computerscience/0做完了/15会计研究/v2/results/asset_allocation_mv_step2.dta", replace

quietly count if inlist(status, "pass", "marginal")
local survivors = r(N)
display _newline "=== Step 2 survivors (pass + marginal) = `survivors' ==="

file open gatefh using "/Users/mac/computerscience/0做完了/15会计研究/v2/results/asset_allocation_mv_gate.txt", write replace
file write gatefh "survivors=`survivors'" _n
if `survivors' == 0 {
    display as error "资产配置全部 MV Step 2 fail，机制路线确诊死亡，建议降级"
    file write gatefh "message=资产配置全部 MV Step 2 fail，机制路线确诊死亡，建议降级" _n
}
else {
    file write gatefh "message=存在 pass/marginal 候选，可进入三步检验" _n
}
file close gatefh

log close
