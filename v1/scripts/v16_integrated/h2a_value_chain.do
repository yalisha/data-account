clear all
set more off

local base "/Users/mac/computerscience/0做完了/15会计研究"
cap mkdir "`base'/results/v16_integrated"
cap mkdir "`base'/results/v16_integrated/logs"

capture log close _all
log using "`base'/results/v16_integrated/logs/h2a_value_chain.log", replace text

capture which reghdfe
if _rc {
    display as error "reghdfe not installed"
    exit 198
}

use "`base'/data_stata/reg_sample_v16_integrated.dta", clear

local controls "Size Lev ROA TobinQ Age Growth IndepRatio Dual Top1Share SOE CFO"
tempfile out
tempname memhold

postfile `memhold' str20 model str20 regressor double(coef se tstat pval) long(N) double(r2) using `out', replace

foreach V in DU_stock DU_dev DU_app DU_value DU_gov {
    quietly reghdfe PriceDelay `V'_lag `controls', absorb(Stkcd_num year_num) cluster(IndYear_num)
    local b = _b[`V'_lag]
    local se = _se[`V'_lag]
    local t = `b' / `se'
    local p = 2 * ttail(e(df_r), abs(`t'))
    local n = e(N)
    local r2 = e(r2)
    post `memhold' ("`V'") ("`V'_lag") (`b') (`se') (`t') (`p') (`n') (`r2')
    display as text "`V'_lag: coef=" %9.4f `b' " t=" %6.2f `t' " N=" `n'
}

postclose `memhold'

use `out', clear
export delimited using "`base'/results/v16_integrated/h2a_value_chain.csv", replace
list, noobs abbreviate(20)

log close
