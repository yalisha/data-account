* ============================================================
* v18 Supply Chain: Mechanism + Heterogeneity
* ============================================================

clear all
set more off
set seed 20260330

use "/Users/mac/computerscience/15会计研究/data_stata/reg_sample_v18.dta", clear

global controls "Size Lev ROA TobinQ Age Growth IndepRatio Dual Top1Share SOE CFO"
local outdir "/Users/mac/computerscience/15会计研究/results/v18"

* ════════════════════════════════════════════
* 1. Mechanism: DU -> Supply Chain Concentration
* ════════════════════════════════════════════

display _n "============================================"
display "MECHANISM: DU -> Supply Chain Variables"
display "============================================"

foreach m in CustConc SuppConc SCConc CustHHI {
    capture confirm variable `m'
    if _rc continue

    quietly count if !missing(`m') & !missing(DU_kw)
    if r(N) < 500 continue

    display _n "=== `m' ~ DU_kw ==="
    quietly reghdfe `m' DU_kw $controls, absorb(Stkcd_num year_num) cluster(IndYear_num)
    display "  coef=" %9.4f _b[DU_kw] "  t=" %6.2f _b[DU_kw]/_se[DU_kw] "  N=" e(N) "  R2=" %6.4f e(r2)

    display "=== `m' ~ DU_llm ==="
    quietly reghdfe `m' DU_llm $controls, absorb(Stkcd_num year_num) cluster(IndYear_num)
    display "  coef=" %9.4f _b[DU_llm] "  t=" %6.2f _b[DU_llm]/_se[DU_llm] "  N=" e(N) "  R2=" %6.4f e(r2)
}

* ════════════════════════════════════════════
* 2. Heterogeneity: Split by CustConc median
* ════════════════════════════════════════════

display _n "============================================"
display "HETEROGENEITY: by Customer Concentration"
display "============================================"

quietly summarize CustConc if !missing(PriceDelay) & !missing(DU_kw), detail
local med = r(p50)
generate byte CustConc_high = (CustConc >= `med') if !missing(CustConc)

foreach treat in DU_kw DU_llm {
    display _n "--- `treat' ---"

    * High concentration
    quietly reghdfe PriceDelay `treat' $controls if CustConc_high == 1, absorb(Stkcd_num year_num) cluster(IndYear_num)
    local b_h = _b[`treat']
    local t_h = `b_h' / _se[`treat']
    local n_h = e(N)

    * Low concentration
    quietly reghdfe PriceDelay `treat' $controls if CustConc_high == 0, absorb(Stkcd_num year_num) cluster(IndYear_num)
    local b_l = _b[`treat']
    local t_l = `b_l' / _se[`treat']
    local n_l = e(N)

    local actual_diff = `b_h' - `b_l'

    * Fisher test (500 reps)
    local n_extreme = 0
    forvalues i = 1/500 {
        tempvar pg ph
        generate double `pg' = runiform() if !missing(CustConc_high) & !missing(PriceDelay) & !missing(`treat')
        quietly summarize `pg', detail
        generate byte `ph' = (`pg' >= r(p50)) if !missing(`pg')
        capture {
            quietly reghdfe PriceDelay `treat' $controls if `ph' == 1, absorb(Stkcd_num year_num) cluster(IndYear_num)
            local pb_h = _b[`treat']
            quietly reghdfe PriceDelay `treat' $controls if `ph' == 0, absorb(Stkcd_num year_num) cluster(IndYear_num)
            local pb_l = _b[`treat']
            if abs(`pb_h' - `pb_l') >= abs(`actual_diff') {
                local n_extreme = `n_extreme' + 1
            }
        }
        drop `pg' `ph'
    }
    local fisher_p = (`n_extreme' + 1) / 501

    display "  High CustConc: coef=" %9.4f `b_h' " t=" %6.2f `t_h' " N=" `n_h'
    display "  Low  CustConc: coef=" %9.4f `b_l' " t=" %6.2f `t_l' " N=" `n_l'
    display "  Fisher P = " %6.3f `fisher_p'
}

* ════════════════════════════════════════════
* 3. Heterogeneity: Split by SCConc (综合供应链集中度)
* ════════════════════════════════════════════

display _n "============================================"
display "HETEROGENEITY: by Supply Chain Concentration"
display "============================================"

quietly summarize SCConc if !missing(PriceDelay) & !missing(DU_kw), detail
local med = r(p50)
generate byte SCConc_high = (SCConc >= `med') if !missing(SCConc)

foreach treat in DU_kw DU_llm {
    display _n "--- `treat' ---"

    quietly reghdfe PriceDelay `treat' $controls if SCConc_high == 1, absorb(Stkcd_num year_num) cluster(IndYear_num)
    local b_h = _b[`treat']
    local t_h = `b_h' / _se[`treat']
    local n_h = e(N)

    quietly reghdfe PriceDelay `treat' $controls if SCConc_high == 0, absorb(Stkcd_num year_num) cluster(IndYear_num)
    local b_l = _b[`treat']
    local t_l = `b_l' / _se[`treat']
    local n_l = e(N)

    local actual_diff = `b_h' - `b_l'

    local n_extreme = 0
    forvalues i = 1/500 {
        tempvar pg ph
        generate double `pg' = runiform() if !missing(SCConc_high) & !missing(PriceDelay) & !missing(`treat')
        quietly summarize `pg', detail
        generate byte `ph' = (`pg' >= r(p50)) if !missing(`pg')
        capture {
            quietly reghdfe PriceDelay `treat' $controls if `ph' == 1, absorb(Stkcd_num year_num) cluster(IndYear_num)
            local pb_h = _b[`treat']
            quietly reghdfe PriceDelay `treat' $controls if `ph' == 0, absorb(Stkcd_num year_num) cluster(IndYear_num)
            local pb_l = _b[`treat']
            if abs(`pb_h' - `pb_l') >= abs(`actual_diff') {
                local n_extreme = `n_extreme' + 1
            }
        }
        drop `pg' `ph'
    }
    local fisher_p = (`n_extreme' + 1) / 501

    display "  High SCConc: coef=" %9.4f `b_h' " t=" %6.2f `t_h' " N=" `n_h'
    display "  Low  SCConc: coef=" %9.4f `b_l' " t=" %6.2f `t_l' " N=" `n_l'
    display "  Fisher P = " %6.3f `fisher_p'
}

display _n "=== Supply chain analysis complete ==="
