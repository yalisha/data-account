* =============================================================
* Table 2: OLS Baseline (Stata, consistent with IV sample)
* =============================================================
clear all
set more off

use "data_stata/reg_sample_iv_v16.dta", clear

* Controls
global controls Size Lev ROA TobinQ Age Growth IndepRatio Dual Top1Share SOE CFO

* ---- Model (1): DU_kw, no controls ----
reghdfe PriceDelay DU_kw, absorb(Stkcd_num year_num) cluster(IndYear_num)
est store m1
local b1 = _b[DU_kw]
local se1 = _se[DU_kw]
local n1 = e(N)
local r2_1 = e(r2)

* ---- Model (2): DU_kw, with controls ----
reghdfe PriceDelay DU_kw $controls, absorb(Stkcd_num year_num) cluster(IndYear_num)
est store m2
local b2 = _b[DU_kw]
local se2 = _se[DU_kw]
local n2 = e(N)
local r2_2 = e(r2)

* ---- Model (3): DU_llm, no controls ----
cap reghdfe PriceDelay DU_llm, absorb(Stkcd_num year_num) cluster(IndYear_num)
if _rc == 0 {
    est store m3
    local b3 = _b[DU_llm]
    local se3 = _se[DU_llm]
    local n3 = e(N)
    local r2_3 = e(r2)
}

* ---- Model (4): DU_llm, with controls ----
cap reghdfe PriceDelay DU_llm $controls, absorb(Stkcd_num year_num) cluster(IndYear_num)
if _rc == 0 {
    est store m4
    local b4 = _b[DU_llm]
    local se4 = _se[DU_llm]
    local n4 = e(N)
    local r2_4 = e(r2)
}

* ---- Model (5): llm_binary, with controls ----
cap reghdfe PriceDelay llm_binary $controls, absorb(Stkcd_num year_num) cluster(IndYear_num)
if _rc == 0 {
    est store m5
    local b5 = _b[llm_binary]
    local se5 = _se[llm_binary]
    local n5 = e(N)
    local r2_5 = e(r2)
}

* ---- Print results ----
di "============================================="
di "Table 2: OLS Baseline Results (Stata)"
di "============================================="
di "Model (1) DU_kw no ctrl:   b=`b1'  se=`se1'  N=`n1'  R2=`r2_1'"
di "Model (2) DU_kw w/ ctrl:   b=`b2'  se=`se2'  N=`n2'  R2=`r2_2'"
di "Model (3) DU_llm no ctrl:  b=`b3'  se=`se3'  N=`n3'  R2=`r2_3'"
di "Model (4) DU_llm w/ ctrl:  b=`b4'  se=`se4'  N=`n4'  R2=`r2_4'"
di "Model (5) llm_binary ctrl: b=`b5'  se=`se5'  N=`n5'  R2=`r2_5'"

* ---- SYNCH as alternative DV ----
cap reghdfe SYNCH DU_kw $controls, absorb(Stkcd_num year_num) cluster(IndYear_num)
if _rc == 0 {
    di "SYNCH ~ DU_kw:  b=" _b[DU_kw] "  se=" _se[DU_kw] "  N=" e(N)
}

cap reghdfe SYNCH DU_llm $controls, absorb(Stkcd_num year_num) cluster(IndYear_num)
if _rc == 0 {
    di "SYNCH ~ DU_llm: b=" _b[DU_llm] "  se=" _se[DU_llm] "  N=" e(N)
}

* ---- Check: does DU_llm exist? ----
cap confirm variable DU_llm
if _rc != 0 {
    di "WARNING: DU_llm not found in dataset. Check variable names."
    describe llm*
    describe DU*
}
