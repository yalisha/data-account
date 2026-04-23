clear all
set more off

local base "/Users/mac/computerscience/0做完了/15会计研究"
cap mkdir "`base'/results/v16_integrated"
cap mkdir "`base'/results/v16_integrated/logs"

capture log close _all
log using "`base'/results/v16_integrated/logs/h2c_washgap.log", replace text

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

quietly reghdfe PriceDelay WashGap_lag `controls', absorb(Stkcd_num year_num) cluster(IndYear_num)
foreach R in WashGap_lag {
    local b = _b[`R']
    local se = _se[`R']
    local t = `b' / `se'
    local p = 2 * ttail(e(df_r), abs(`t'))
    local n = e(N)
    local r2 = e(r2)
    post `memhold' ("washgap_direct") ("`R'") (`b') (`se') (`t') (`p') (`n') (`r2')
}

quietly reghdfe PriceDelay DU_kw_lag WashGap_lag `controls', absorb(Stkcd_num year_num) cluster(IndYear_num)
foreach R in DU_kw_lag WashGap_lag {
    local b = _b[`R']
    local se = _se[`R']
    local t = `b' / `se'
    local p = 2 * ttail(e(df_r), abs(`t'))
    local n = e(N)
    local r2 = e(r2)
    post `memhold' ("washgap_joint_kw") ("`R'") (`b') (`se') (`t') (`p') (`n') (`r2')
}

quietly reghdfe PriceDelay DU_llm_lag WashGap_lag `controls', absorb(Stkcd_num year_num) cluster(IndYear_num)
foreach R in DU_llm_lag WashGap_lag {
    local b = _b[`R']
    local se = _se[`R']
    local t = `b' / `se'
    local p = 2 * ttail(e(df_r), abs(`t'))
    local n = e(N)
    local r2 = e(r2)
    post `memhold' ("washgap_joint_llm") ("`R'") (`b') (`se') (`t') (`p') (`n') (`r2')
}

postclose `memhold'

use `out', clear
export delimited using "`base'/results/v16_integrated/h2c_washgap.csv", replace
list, noobs abbreviate(20)

log close
