* =============================================================
* Table 4: Robustness Checks (Stata, unified sample)
* =============================================================
clear all
set more off
use "data_stata/reg_sample_iv_v16.dta", clear

global controls Size Lev ROA TobinQ Age Growth IndepRatio Dual Top1Share SOE CFO

di "============================================="
di "Table 4: Robustness"
di "============================================="

* ---- (1) Industry×Year FE ----
reghdfe PriceDelay DU_kw $controls, absorb(IndYear_num) cluster(IndYear_num)
di "IndYear FE: b=" _b[DU_kw] " se=" _se[DU_kw] " N=" e(N)

* ---- (2) Drop last year (2024) ----
reghdfe PriceDelay DU_kw $controls if year_num < 2024, absorb(Stkcd_num year_num) cluster(IndYear_num)
di "Drop 2024: b=" _b[DU_kw] " se=" _se[DU_kw] " N=" e(N)

* ---- (3) Drop IT industry (I63-I65: Ind2_num 60,61,62) ----
reghdfe PriceDelay DU_kw $controls if !inlist(Ind2_num, 60, 61, 62), absorb(Stkcd_num year_num) cluster(IndYear_num)
di "Drop IT: b=" _b[DU_kw] " se=" _se[DU_kw] " N=" e(N)

* ---- (4) Lagged controls ----
xtset Stkcd_num year_num
reghdfe PriceDelay DU_kw L.Size L.Lev L.ROA L.TobinQ L.Age L.Growth L.IndepRatio L.Dual L.Top1Share L.SOE L.CFO, absorb(Stkcd_num year_num) cluster(IndYear_num)
di "Lag ctrl: b=" _b[DU_kw] " se=" _se[DU_kw] " N=" e(N)

* ---- (5) PSM ----
* Treatment: DU_kw > industry-year median
bysort Ind2_num year_num: egen med_DU = median(DU_kw)
gen treat_psm = (DU_kw > med_DU) if DU_kw != .

* Logit propensity score
logit treat_psm $controls
predict pscore, pr

* Nearest neighbor 1:1 matching with caliper 0.05
* Use psmatch2 if available, otherwise manual
cap which psmatch2
if _rc != 0 {
    ssc install psmatch2, replace
}
psmatch2 treat_psm, pscore(pscore) neighbor(1) caliper(0.05) common
* Run regression on matched sample
reghdfe PriceDelay DU_kw $controls if _weight != ., absorb(Stkcd_num year_num) cluster(IndYear_num)
di "PSM: b=" _b[DU_kw] " se=" _se[DU_kw] " N=" e(N)

* Balance check
pstest $controls, both

* ---- (6) Two-way clustering (firm + year) ----
reghdfe PriceDelay DU_kw $controls, absorb(Stkcd_num year_num) cluster(Stkcd_num year_num)
di "2way cluster: b=" _b[DU_kw] " se=" _se[DU_kw] " N=" e(N)

* ---- (7) Alternative IV: DU_sub_ln ----
reghdfe PriceDelay DU_sub_ln $controls, absorb(Stkcd_num year_num) cluster(IndYear_num)
di "DU_sub_ln: b=" _b[DU_sub_ln] " se=" _se[DU_sub_ln] " N=" e(N)
