* ============================================================
* v18 Heterogeneity Tests: ALL dimensions
* Split sample by median (or binary), regress PriceDelay ~ DU_kw
* Fisher permutation test (500 reps for speed)
* ============================================================

clear all
set more off
set seed 20260330

use "/Users/mac/computerscience/15会计研究/data_stata/reg_sample_v18.dta", clear

global controls "Size Lev ROA TobinQ Age Growth IndepRatio Dual Top1Share SOE CFO"

local outdir "/Users/mac/computerscience/15会计研究/results/v18"

* ════════════════════════════════════════════
* Define heterogeneity dimensions
* ════════════════════════════════════════════

* Continuous vars: split by median
* Binary vars: use as-is
* Special: Post2020, HighTech, MainBoard, Big4, SOE are binary

local cont_dims "Analyst Size Hhi InstHold ProvDigital2016 Market SA ShareholderNum BM Intangible LQ"
local bin_dims "SOE HighTech Post2020 MainBoard Big4 StrategicEmerging DigEconCore IndustryCluster"

* ── Create median splits for continuous vars ──
foreach v of local cont_dims {
    capture confirm variable `v'
    if _rc continue
    quietly summarize `v' if !missing(PriceDelay) & !missing(DU_kw), detail
    local med = r(p50)
    generate byte `v'_high = (`v' >= `med') if !missing(`v')
    label variable `v'_high "`v' >= median"
}

* ── Binary vars: create _high versions ──
foreach v of local bin_dims {
    capture confirm variable `v'
    if _rc continue
    generate byte `v'_high = `v' if !missing(`v')
}

* ════════════════════════════════════════════
* Run split-sample regressions + Fisher test
* ════════════════════════════════════════════

local all_dims "`cont_dims' `bin_dims'"
local n_reps = 500

tempname memhold
postfile `memhold' str20 dimension str10 treatment double(coef_high se_high t_high) long(N_high) double(r2_high coef_low se_low t_low) long(N_low) double(r2_low fisher_p) using "`outdir'/heterogeneity_v18.dta", replace

foreach dim of local all_dims {
    capture confirm variable `dim'_high
    if _rc {
        display "  SKIP: `dim' not available"
        continue
    }

    * Check sample sizes
    quietly count if `dim'_high == 1 & !missing(PriceDelay) & !missing(DU_kw)
    local n1 = r(N)
    quietly count if `dim'_high == 0 & !missing(PriceDelay) & !missing(DU_kw)
    local n0 = r(N)
    if `n1' < 500 | `n0' < 500 {
        display "  SKIP: `dim' groups too small (high=`n1', low=`n0')"
        continue
    }

    * === DU_kw ===
    display _n "=== Heterogeneity: `dim' (DU_kw) ==="

    * High group
    quietly reghdfe PriceDelay DU_kw $controls if `dim'_high == 1, absorb(Stkcd_num year_num) cluster(IndYear_num)
    local b_h = _b[DU_kw]
    local se_h = _se[DU_kw]
    local t_h = `b_h'/`se_h'
    local n_h = e(N)
    local r2_h = e(r2)

    * Low group
    quietly reghdfe PriceDelay DU_kw $controls if `dim'_high == 0, absorb(Stkcd_num year_num) cluster(IndYear_num)
    local b_l = _b[DU_kw]
    local se_l = _se[DU_kw]
    local t_l = `b_l'/`se_l'
    local n_l = e(N)
    local r2_l = e(r2)

    * Actual difference
    local actual_diff = `b_h' - `b_l'

    * Fisher permutation test
    local n_extreme = 0
    forvalues i = 1/`n_reps' {
        * Randomly permute group assignment
        tempvar perm_group
        generate double `perm_group' = runiform() if !missing(`dim'_high) & !missing(PriceDelay) & !missing(DU_kw)
        quietly summarize `perm_group', detail
        local perm_med = r(p50)
        tempvar perm_high
        generate byte `perm_high' = (`perm_group' >= `perm_med') if !missing(`perm_group')

        capture {
            quietly reghdfe PriceDelay DU_kw $controls if `perm_high' == 1, absorb(Stkcd_num year_num) cluster(IndYear_num)
            local pb_h = _b[DU_kw]
            quietly reghdfe PriceDelay DU_kw $controls if `perm_high' == 0, absorb(Stkcd_num year_num) cluster(IndYear_num)
            local pb_l = _b[DU_kw]
            local perm_diff = `pb_h' - `pb_l'
            if abs(`perm_diff') >= abs(`actual_diff') {
                local n_extreme = `n_extreme' + 1
            }
        }
        drop `perm_group' `perm_high'
    }
    local fisher_p = (`n_extreme' + 1) / (`n_reps' + 1)

    display "  High: coef=" %9.4f `b_h' " t=" %6.2f `t_h' " N=" `n_h'
    display "  Low:  coef=" %9.4f `b_l' " t=" %6.2f `t_l' " N=" `n_l'
    display "  Fisher P (DU_kw) = " %6.3f `fisher_p'

    post `memhold' ("`dim'") ("DU_kw") (`b_h') (`se_h') (`t_h') (`n_h') (`r2_h') (`b_l') (`se_l') (`t_l') (`n_l') (`r2_l') (`fisher_p')

    * === DU_llm ===
    display _n "=== Heterogeneity: `dim' (DU_llm) ==="

    quietly reghdfe PriceDelay DU_llm $controls if `dim'_high == 1, absorb(Stkcd_num year_num) cluster(IndYear_num)
    local b_h = _b[DU_llm]
    local se_h = _se[DU_llm]
    local t_h = `b_h'/`se_h'
    local n_h = e(N)
    local r2_h = e(r2)

    quietly reghdfe PriceDelay DU_llm $controls if `dim'_high == 0, absorb(Stkcd_num year_num) cluster(IndYear_num)
    local b_l = _b[DU_llm]
    local se_l = _se[DU_llm]
    local t_l = `b_l'/`se_l'
    local n_l = e(N)
    local r2_l = e(r2)

    local actual_diff = `b_h' - `b_l'

    local n_extreme = 0
    forvalues i = 1/`n_reps' {
        tempvar perm_group
        generate double `perm_group' = runiform() if !missing(`dim'_high) & !missing(PriceDelay) & !missing(DU_llm)
        quietly summarize `perm_group', detail
        local perm_med = r(p50)
        tempvar perm_high
        generate byte `perm_high' = (`perm_group' >= `perm_med') if !missing(`perm_group')

        capture {
            quietly reghdfe PriceDelay DU_llm $controls if `perm_high' == 1, absorb(Stkcd_num year_num) cluster(IndYear_num)
            local pb_h = _b[DU_llm]
            quietly reghdfe PriceDelay DU_llm $controls if `perm_high' == 0, absorb(Stkcd_num year_num) cluster(IndYear_num)
            local pb_l = _b[DU_llm]
            local perm_diff = `pb_h' - `pb_l'
            if abs(`perm_diff') >= abs(`actual_diff') {
                local n_extreme = `n_extreme' + 1
            }
        }
        drop `perm_group' `perm_high'
    }
    local fisher_p = (`n_extreme' + 1) / (`n_reps' + 1)

    display "  High: coef=" %9.4f `b_h' " t=" %6.2f `t_h' " N=" `n_h'
    display "  Low:  coef=" %9.4f `b_l' " t=" %6.2f `t_l' " N=" `n_l'
    display "  Fisher P (DU_llm) = " %6.3f `fisher_p'

    post `memhold' ("`dim'") ("DU_llm") (`b_h') (`se_h') (`t_h') (`n_h') (`r2_h') (`b_l') (`se_l') (`t_l') (`n_l') (`r2_l') (`fisher_p')
}

postclose `memhold'

* ── Export ──
use "`outdir'/heterogeneity_v18.dta", clear
export delimited using "`outdir'/heterogeneity_v18.csv", replace
list, separator(2) abbreviate(20)

display _n "=== Heterogeneity tests complete ==="
