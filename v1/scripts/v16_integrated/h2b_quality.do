clear all
set more off

local base "/Users/mac/computerscience/0做完了/15会计研究"
cap mkdir "`base'/results/v16_integrated"
cap mkdir "`base'/results/v16_integrated/logs"

capture log close _all
log using "`base'/results/v16_integrated/logs/h2b_quality.log", replace text

capture which reghdfe
if _rc {
    display as error "reghdfe not installed"
    exit 198
}

local controls "Size Lev ROA TobinQ Age Growth IndepRatio Dual Top1Share SOE CFO"

use "`base'/data_stata/reg_sample_v16_integrated.dta", clear

tempfile direct_out joint_out
tempname direct_hold joint_hold

postfile `direct_hold' str20 model str20 regressor double(coef se tstat pval) long(N) double(r2) using `direct_out', replace

quietly reghdfe PriceDelay DU_llm_lenstd_lag `controls', absorb(Stkcd_num year_num) cluster(IndYear_num)
local b = _b[DU_llm_lenstd_lag]
local se = _se[DU_llm_lenstd_lag]
local t = `b' / `se'
local p = 2 * ttail(e(df_r), abs(`t'))
local n = e(N)
local r2 = e(r2)
post `direct_hold' ("direct_DU_llm_lenstd") ("DU_llm_lenstd_lag") (`b') (`se') (`t') (`p') (`n') (`r2')
display as text "Direct DU_llm_lenstd_lag: coef=" %9.4f `b' " t=" %6.2f `t' " N=" `n'
postclose `direct_hold'

tempfile direct_patch
use `direct_out', clear
save `direct_patch', replace
capture confirm file "`base'/results/v16_integrated/h2b_quality_direct.csv"
if !_rc {
    import delimited using "`base'/results/v16_integrated/h2b_quality_direct.csv", clear varnames(1)
    drop if regressor == "DU_llm_lenstd_lag"
    append using `direct_patch'
    capture confirm variable n
    if !_rc {
        replace N = n if missing(N)
        drop n
    }
}
else {
    use `direct_patch', clear
}
sort model regressor
export delimited using "`base'/results/v16_integrated/h2b_quality_direct.csv", replace
list, noobs abbreviate(20)

use "`base'/data_stata/reg_sample_v16_integrated.dta", clear
postfile `joint_hold' str24 model str20 regressor double(coef se tstat pval) long(N) double(r2) using `joint_out', replace

quietly reghdfe PriceDelay DU_kw_lag DU_llm_lenstd_lag `controls', absorb(Stkcd_num year_num) cluster(IndYear_num)
foreach R in DU_kw_lag DU_llm_lenstd_lag {
    local b = _b[`R']
    local se = _se[`R']
    local t = `b' / `se'
    local p = 2 * ttail(e(df_r), abs(`t'))
    local n = e(N)
    local r2 = e(r2)
    post `joint_hold' ("joint_dukw_lenstd") ("`R'") (`b') (`se') (`t') (`p') (`n') (`r2')
}
display as text "Joint DU_kw_lag + DU_llm_lenstd_lag complete"

postclose `joint_hold'

tempfile joint_patch
use `joint_out', clear
save `joint_patch', replace
capture confirm file "`base'/results/v16_integrated/h2b_quality_joint.csv"
if !_rc {
    import delimited using "`base'/results/v16_integrated/h2b_quality_joint.csv", clear varnames(1)
    drop if model == "joint_dukw_lenstd"
    drop if model == "joint_dullm_lenstd"
    append using `joint_patch'
    capture confirm variable n
    if !_rc {
        replace N = n if missing(N)
        drop n
    }
}
else {
    use `joint_patch', clear
}
sort model regressor
export delimited using "`base'/results/v16_integrated/h2b_quality_joint.csv", replace
list, noobs abbreviate(24)

log close
