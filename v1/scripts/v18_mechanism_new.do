* ============================================================
* v18 New Mechanism Channels: TFP, CrashRisk, CashFlowVol
* ============================================================

clear all
set more off

use "/Users/mac/computerscience/15会计研究/data_stata/reg_sample_v18.dta", clear

global controls "Size Lev ROA TobinQ Age Growth IndepRatio Dual Top1Share SOE CFO"
local outdir "/Users/mac/computerscience/15会计研究/results/v18"

local new_channels "TFP NCSKEW DUVOL CashFlowVol"

tempname memhold
postfile `memhold' str20 channel str10 treatment double(coef se tstat pval) long(N) double(r2) using "`outdir'/mechanism_new_v18.dta", replace

foreach m of local new_channels {
    capture confirm variable `m'
    if _rc {
        display "  SKIP: `m' not found"
        continue
    }

    quietly count if !missing(`m') & !missing(DU_kw)
    local n_obs = r(N)
    if `n_obs' < 500 {
        display "  SKIP: `m' only `n_obs' obs"
        continue
    }

    * DU_kw
    display _n "=== `m' ~ DU_kw ==="
    quietly reghdfe `m' DU_kw $controls, absorb(Stkcd_num year_num) cluster(IndYear_num)
    local b = _b[DU_kw]
    local se = _se[DU_kw]
    local t = `b'/`se'
    local p = 2*ttail(e(df_r), abs(`t'))
    post `memhold' ("`m'") ("DU_kw") (`b') (`se') (`t') (`p') (e(N)) (e(r2))
    display "  coef=" %9.4f `b' "  t=" %6.2f `t' "  N=" e(N)

    * DU_llm
    display "=== `m' ~ DU_llm ==="
    quietly reghdfe `m' DU_llm $controls, absorb(Stkcd_num year_num) cluster(IndYear_num)
    local b = _b[DU_llm]
    local se = _se[DU_llm]
    local t = `b'/`se'
    local p = 2*ttail(e(df_r), abs(`t'))
    post `memhold' ("`m'") ("DU_llm") (`b') (`se') (`t') (`p') (e(N)) (e(r2))
    display "  coef=" %9.4f `b' "  t=" %6.2f `t' "  N=" e(N)
}

postclose `memhold'

use "`outdir'/mechanism_new_v18.dta", clear
export delimited using "`outdir'/mechanism_new_v18.csv", replace
list, separator(2) abbreviate(20)

* ════════════════════════════════════════════
* Also run conditional mechanism analysis:
* Mechanism channels BY StrategicEmerging and IndustryCluster
* ════════════════════════════════════════════

use "/Users/mac/computerscience/15会计研究/data_stata/reg_sample_v18.dta", clear

display _n "============================================"
display "CONDITIONAL MECHANISM: by StrategicEmerging"
display "============================================"

local key_channels "Analyst ForecastDisp NCSKEW CashFlowVol"

foreach m of local key_channels {
    capture confirm variable `m'
    if _rc continue

    foreach grp in 0 1 {
        display _n "=== `m' ~ DU_kw | StrategicEmerging==`grp' ==="
        quietly count if !missing(`m') & !missing(DU_kw) & StrategicEmerging==`grp'
        if r(N) < 500 {
            display "  SKIP: only " r(N) " obs"
            continue
        }
        quietly reghdfe `m' DU_kw $controls if StrategicEmerging==`grp', absorb(Stkcd_num year_num) cluster(IndYear_num)
        display "  coef=" %9.4f _b[DU_kw] "  t=" %6.2f _b[DU_kw]/_se[DU_kw] "  N=" e(N)
    }
}

display _n "============================================"
display "CONDITIONAL MECHANISM: by IndustryCluster"
display "============================================"

foreach m of local key_channels {
    capture confirm variable `m'
    if _rc continue

    foreach grp in 0 1 {
        display _n "=== `m' ~ DU_kw | IndustryCluster==`grp' ==="
        quietly count if !missing(`m') & !missing(DU_kw) & IndustryCluster==`grp'
        if r(N) < 500 {
            display "  SKIP: only " r(N) " obs"
            continue
        }
        quietly reghdfe `m' DU_kw $controls if IndustryCluster==`grp', absorb(Stkcd_num year_num) cluster(IndYear_num)
        display "  coef=" %9.4f _b[DU_kw] "  t=" %6.2f _b[DU_kw]/_se[DU_kw] "  N=" e(N)
    }
}

display _n "=== New mechanism + conditional analysis complete ==="
