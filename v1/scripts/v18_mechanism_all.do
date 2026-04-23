* ============================================================
* v18 Mechanism Tests: ALL channels
* DU_kw + DU_llm -> each mechanism variable
* reghdfe with firm+year FE, cluster(IndYear)
* ============================================================

clear all
set more off

use "/Users/mac/computerscience/15会计研究/data_stata/reg_sample_v18.dta", clear

* Controls
global controls "Size Lev ROA TobinQ Age Growth IndepRatio Dual Top1Share SOE CFO"

* Output directory
local outdir "/Users/mac/computerscience/15会计研究/results/v18"

* ── Mechanism variables list ──
* Existing: Analyst Amihud RetVol InstHold Turnover
* New: ForecastDisp FcstAcc absDA InvestIneff AuditFee RetAutoCorr IdioVol TurnoverVol InstStable RatingDisp ReportFreq

local mech_vars "Analyst Amihud RetVol InstHold Turnover ForecastDisp FcstAcc absDA InvestIneff AuditFee RetAutoCorr IdioVol TurnoverVol InstStable RatingDisp ReportFreq"

* Note: mechanism vars already winsorized in Python construction step

* ════════════════════════════════════════════
* Run all mechanism regressions
* ════════════════════════════════════════════

* Create results file
tempname memhold
postfile `memhold' str20 channel str10 treatment double(coef se tstat pval) long(N) double(r2) using "`outdir'/mechanism_v18.dta", replace

foreach m of local mech_vars {
    capture confirm variable `m'
    if _rc {
        display "  SKIP: `m' not found"
        continue
    }

    * Count non-missing
    quietly count if !missing(`m') & !missing(DU_kw)
    local n_obs = r(N)
    if `n_obs' < 500 {
        display "  SKIP: `m' only `n_obs' obs"
        continue
    }

    * DU_kw -> mechanism
    display _n "=== `m' ~ DU_kw ==="
    quietly reghdfe `m' DU_kw $controls, absorb(Stkcd_num year_num) cluster(IndYear_num)
    local b = _b[DU_kw]
    local se = _se[DU_kw]
    local t = `b'/`se'
    local p = 2*ttail(e(df_r), abs(`t'))
    local n = e(N)
    local r2 = e(r2)
    post `memhold' ("`m'") ("DU_kw") (`b') (`se') (`t') (`p') (`n') (`r2')
    display "  coef=" %9.4f `b' "  t=" %6.2f `t' "  N=" `n'

    * DU_llm -> mechanism
    display "=== `m' ~ DU_llm ==="
    quietly reghdfe `m' DU_llm $controls, absorb(Stkcd_num year_num) cluster(IndYear_num)
    local b = _b[DU_llm]
    local se = _se[DU_llm]
    local t = `b'/`se'
    local p = 2*ttail(e(df_r), abs(`t'))
    local n = e(N)
    local r2 = e(r2)
    post `memhold' ("`m'") ("DU_llm") (`b') (`se') (`t') (`p') (`n') (`r2')
    display "  coef=" %9.4f `b' "  t=" %6.2f `t' "  N=" `n'
}

postclose `memhold'

* ── Export to CSV ──
use "`outdir'/mechanism_v18.dta", clear
export delimited using "`outdir'/mechanism_v18.csv", replace
list, separator(2) abbreviate(20)

display _n "=== Mechanism tests complete ==="
