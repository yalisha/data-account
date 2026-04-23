clear all
set more off

local base "/Users/mac/computerscience/0做完了/15会计研究"
cap mkdir "`base'/results/v16_quality"
cap mkdir "`base'/results/v16_quality/logs"

capture log close _all
log using "`base'/results/v16_quality/logs/run_quality_resid.log", replace text

capture which reghdfe
if _rc {
    display as error "reghdfe not installed"
    exit 198
}

use "`base'/data_stata/reg_sample_v16_quality.dta", clear

local resid_controls "Size Lev ROA IndepRatio SOE"
local controls "Size Lev ROA TobinQ Age Growth IndepRatio Dual Top1Share SOE CFO"

foreach Q in Quality_sub Quality_mda Quality_subden BroadShallow_cont WashGap {
    capture drop `Q'_resid
    capture drop `Q'_resid_z
    quietly reghdfe `Q' `resid_controls', absorb(Stkcd_num year_num) residuals(`Q'_resid)
    egen `Q'_resid_z = std(`Q'_resid)
    display as text "Residualized `Q'"
}

tempfile resid_out
tempname resid_hold

postfile `resid_hold' str20 quality str12 treatment double(coef se tstat pval) long(N) double(r2) using `resid_out', replace

foreach Q in Quality_sub Quality_mda Quality_subden BroadShallow_cont WashGap {
    capture drop DUxQ
    gen double DUxQ = DU_kw * `Q'_resid_z
    quietly reghdfe PriceDelay DU_kw `Q'_resid_z DUxQ `controls', absorb(Stkcd_num year_num) cluster(IndYear_num)
    local b = _b[DUxQ]
    local se = _se[DUxQ]
    local t = `b' / `se'
    local p = 2 * ttail(e(df_r), abs(`t'))
    local n = e(N)
    local r2 = e(r2)
    post `resid_hold' ("`Q'") ("DU_kw") (`b') (`se') (`t') (`p') (`n') (`r2')
    display as text "`Q'_resid_z / DU_kw: coef=" %9.4f `b' " t=" %6.2f `t' " N=" `n'
    drop DUxQ
}

postclose `resid_hold'

use `resid_out', clear
export delimited using "`base'/results/v16_quality/quality_resid.csv", replace
list, noobs abbreviate(20)

log close
