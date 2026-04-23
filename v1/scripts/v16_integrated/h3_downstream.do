clear all
set more off

local base "/Users/mac/computerscience/0做完了/15会计研究"
cap mkdir "`base'/results/v16_integrated"
cap mkdir "`base'/results/v16_integrated/logs"

capture log close _all
log using "`base'/results/v16_integrated/logs/h3_downstream.log", replace text

capture which reghdfe
if _rc {
    display as error "reghdfe not installed"
    exit 198
}

use "`base'/data_stata/reg_sample_v16_integrated.dta", clear

local controls "Size Lev ROA TobinQ Age Growth IndepRatio Dual Top1Share SOE CFO"
tempfile out
tempname memhold

postfile `memhold' str20 channel str12 treatment double(coef se tstat pval) long(N) double(r2) using `out', replace

foreach Y in ForecastDisp CashFlowVol SCConc {
    foreach X in DU_kw DU_llm {
        quietly reghdfe `Y' `X' `controls', absorb(Stkcd_num year_num) cluster(IndYear_num)
        local b = _b[`X']
        local se = _se[`X']
        local t = `b' / `se'
        local p = 2 * ttail(e(df_r), abs(`t'))
        local n = e(N)
        local r2 = e(r2)
        post `memhold' ("`Y'") ("`X'") (`b') (`se') (`t') (`p') (`n') (`r2')
        display as text "`Y' / `X': coef=" %9.4f `b' " t=" %6.2f `t' " N=" `n'
    }
}

postclose `memhold'

use `out', clear
export delimited using "`base'/results/v16_integrated/h3_downstream.csv", replace
list, noobs abbreviate(20)

log close
