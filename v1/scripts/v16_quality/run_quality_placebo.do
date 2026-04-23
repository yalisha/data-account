clear all
set more off

local base "/Users/mac/computerscience/0做完了/15会计研究"
cap mkdir "`base'/results/v16_quality"
cap mkdir "`base'/results/v16_quality/logs"

capture log close _all
log using "`base'/results/v16_quality/logs/run_quality_placebo.log", replace text

capture which reghdfe
if _rc {
    display as error "reghdfe not installed"
    exit 198
}

use "`base'/data_stata/reg_sample_v16_quality.dta", clear

local controls "Size Lev ROA TobinQ Age Growth IndepRatio Dual Top1Share SOE CFO"

set seed 20260419

tempfile placebo_out
tempname placebo_hold

postfile `placebo_hold' int(iter) double(coef se tstat pval) long(N) double(r2) using `placebo_out', replace

forvalues i = 1/500 {
    preserve
    tempvar rand Q_shuffled DUxQ
    bysort Ind2 year: gen double `rand' = runiform()
    bysort Ind2 year (`rand'): gen double `Q_shuffled' = Quality_sub_z[_n+1]
    bysort Ind2 year (`rand'): replace `Q_shuffled' = Quality_sub_z[1] if missing(`Q_shuffled')

    gen double `DUxQ' = DU_kw * `Q_shuffled'
    quietly reghdfe PriceDelay DU_kw `Q_shuffled' `DUxQ' `controls', absorb(Stkcd_num year_num) cluster(IndYear_num)

    local b = _b[`DUxQ']
    local se = _se[`DUxQ']
    local t = `b' / `se'
    local p = 2 * ttail(e(df_r), abs(`t'))
    local n = e(N)
    local r2 = e(r2)
    post `placebo_hold' (`i') (`b') (`se') (`t') (`p') (`n') (`r2')
    restore

    if mod(`i', 50) == 0 {
        display as text "Completed placebo iteration `i'/500"
    }
}

postclose `placebo_hold'

use `placebo_out', clear
export delimited using "`base'/results/v16_quality/quality_placebo_500.csv", replace

histogram coef, frequency ///
    xtitle("Placebo b3 (Quality_sub random shuffle)") ///
    graphregion(color(white)) ///
    bgcolor(white)
graph export "`base'/results/v16_quality/quality_placebo_hist.png", replace width(1200)

summarize coef, detail

log close
