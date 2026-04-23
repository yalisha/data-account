* =============================================================
* Fisher Permutation Test for Heterogeneity (1000 reps)
* Uses FWL approach for speed: partial out FE+controls first,
* then permute on residualized data
* =============================================================
clear all
set more off
set seed 20260326
use "data_stata/reg_sample_het.dta", clear

global controls Size Lev ROA TobinQ Age Growth IndepRatio Dual Top1Share SOE CFO

* ---- Construct grouping variables ----
gen Post2020 = (year_num >= 2020)

egen med_roastd = median(Roastd)
gen Hi_Roastd = (Roastd >= med_roastd) if Roastd != .

egen med_shnum = median(ShareholderNum)
gen Hi_ShNum = (ShareholderNum >= med_shnum) if ShareholderNum != .

* =============================================================
* Step 1: Partial out FE + controls using reghdfe (done ONCE)
* =============================================================
di "Partialling out FE + controls..."

* For DU_kw tests
reghdfe PriceDelay $controls, absorb(Stkcd_num year_num) resid(e_pd)
reghdfe DU_kw $controls, absorb(Stkcd_num year_num) resid(e_dukw)
reghdfe DU_llm $controls, absorb(Stkcd_num year_num) resid(e_dullm)

di "Residualization done."
di ""

* =============================================================
* Step 2: Actual coefficient differences
* =============================================================
di "============================================="
di "Actual Sub-sample Coefficients"
di "============================================="

* --- Roastd ---
reg e_pd e_dukw if Hi_Roastd == 1, robust
scalar b_roastd_kw_hi = _b[e_dukw]
reg e_pd e_dukw if Hi_Roastd == 0, robust
scalar b_roastd_kw_lo = _b[e_dukw]
scalar diff_roastd_kw = abs(b_roastd_kw_hi - b_roastd_kw_lo)
di "Roastd DU_kw: Hi=" %9.5f b_roastd_kw_hi " Lo=" %9.5f b_roastd_kw_lo " diff=" %9.5f diff_roastd_kw

reg e_pd e_dullm if Hi_Roastd == 1, robust
scalar b_roastd_llm_hi = _b[e_dullm]
reg e_pd e_dullm if Hi_Roastd == 0, robust
scalar b_roastd_llm_lo = _b[e_dullm]
scalar diff_roastd_llm = abs(b_roastd_llm_hi - b_roastd_llm_lo)
di "Roastd DU_llm: Hi=" %9.5f b_roastd_llm_hi " Lo=" %9.5f b_roastd_llm_lo " diff=" %9.5f diff_roastd_llm

* --- ShareholderNum ---
reg e_pd e_dukw if Hi_ShNum == 1, robust
scalar b_shnum_kw_hi = _b[e_dukw]
reg e_pd e_dukw if Hi_ShNum == 0, robust
scalar b_shnum_kw_lo = _b[e_dukw]
scalar diff_shnum_kw = abs(b_shnum_kw_hi - b_shnum_kw_lo)
di "ShNum DU_kw: Hi=" %9.5f b_shnum_kw_hi " Lo=" %9.5f b_shnum_kw_lo " diff=" %9.5f diff_shnum_kw

reg e_pd e_dullm if Hi_ShNum == 1, robust
scalar b_shnum_llm_hi = _b[e_dullm]
reg e_pd e_dullm if Hi_ShNum == 0, robust
scalar b_shnum_llm_lo = _b[e_dullm]
scalar diff_shnum_llm = abs(b_shnum_llm_hi - b_shnum_llm_lo)
di "ShNum DU_llm: Hi=" %9.5f b_shnum_llm_hi " Lo=" %9.5f b_shnum_llm_lo " diff=" %9.5f diff_shnum_llm

* --- Post2020 ---
reg e_pd e_dukw if Post2020 == 1, robust
scalar b_post_kw_hi = _b[e_dukw]
reg e_pd e_dukw if Post2020 == 0, robust
scalar b_post_kw_lo = _b[e_dukw]
scalar diff_post_kw = abs(b_post_kw_hi - b_post_kw_lo)
di "Post2020 DU_kw: Post=" %9.5f b_post_kw_hi " Pre=" %9.5f b_post_kw_lo " diff=" %9.5f diff_post_kw

reg e_pd e_dullm if Post2020 == 1, robust
scalar b_post_llm_hi = _b[e_dullm]
reg e_pd e_dullm if Post2020 == 0, robust
scalar b_post_llm_lo = _b[e_dullm]
scalar diff_post_llm = abs(b_post_llm_hi - b_post_llm_lo)
di "Post2020 DU_llm: Post=" %9.5f b_post_llm_hi " Pre=" %9.5f b_post_llm_lo " diff=" %9.5f diff_post_llm

* =============================================================
* Step 3: Fisher Permutation (1000 reps on residualized data)
* =============================================================
di ""
di "============================================="
di "Fisher Permutation Tests (1000 reps)"
di "============================================="

* Initialize counters
local count_roastd_kw = 0
local count_roastd_llm = 0
local count_shnum_kw = 0
local count_shnum_llm = 0
local count_post_kw = 0
local count_post_llm = 0

* Get sample sizes for each group
count if Hi_Roastd == 1 & e_pd != . & e_dukw != .
local n_roastd_hi = r(N)
count if Hi_Roastd != . & e_pd != . & e_dukw != .
local n_roastd_total = r(N)

count if Hi_ShNum == 1 & e_pd != . & e_dukw != .
local n_shnum_hi = r(N)
count if Hi_ShNum != . & e_pd != . & e_dukw != .
local n_shnum_total = r(N)

count if Post2020 == 1 & e_pd != . & e_dukw != .
local n_post_hi = r(N)
count if Post2020 != . & e_pd != . & e_dukw != .
local n_post_total = r(N)

di "Sample sizes: Roastd=" `n_roastd_total' " ShNum=" `n_shnum_total' " Post=" `n_post_total'

* ----- Roastd permutation -----
di "Running Roastd permutation..."
preserve
keep if Hi_Roastd != . & e_pd != . & e_dukw != . & e_dullm != .
local N = _N

forval i = 1/1000 {
    tempvar rand perm_g
    gen double `rand' = runiform()
    sort `rand'
    gen byte `perm_g' = (_n <= `n_roastd_hi')

    quietly reg e_pd e_dukw if `perm_g' == 1
    scalar pb1 = _b[e_dukw]
    quietly reg e_pd e_dukw if `perm_g' == 0
    scalar pb0 = _b[e_dukw]
    if abs(pb1 - pb0) >= scalar(diff_roastd_kw) local count_roastd_kw = `count_roastd_kw' + 1

    quietly reg e_pd e_dullm if `perm_g' == 1
    scalar pb1 = _b[e_dullm]
    quietly reg e_pd e_dullm if `perm_g' == 0
    scalar pb0 = _b[e_dullm]
    if abs(pb1 - pb0) >= scalar(diff_roastd_llm) local count_roastd_llm = `count_roastd_llm' + 1
}
restore

di "Roastd Fisher p (DU_kw) = " %6.3f (`count_roastd_kw' / 1000)
di "Roastd Fisher p (DU_llm) = " %6.3f (`count_roastd_llm' / 1000)

* ----- ShareholderNum permutation -----
di "Running ShNum permutation..."
preserve
keep if Hi_ShNum != . & e_pd != . & e_dukw != . & e_dullm != .
local N = _N

forval i = 1/1000 {
    tempvar rand perm_g
    gen double `rand' = runiform()
    sort `rand'
    gen byte `perm_g' = (_n <= `n_shnum_hi')

    quietly reg e_pd e_dukw if `perm_g' == 1
    scalar pb1 = _b[e_dukw]
    quietly reg e_pd e_dukw if `perm_g' == 0
    scalar pb0 = _b[e_dukw]
    if abs(pb1 - pb0) >= scalar(diff_shnum_kw) local count_shnum_kw = `count_shnum_kw' + 1

    quietly reg e_pd e_dullm if `perm_g' == 1
    scalar pb1 = _b[e_dullm]
    quietly reg e_pd e_dullm if `perm_g' == 0
    scalar pb0 = _b[e_dullm]
    if abs(pb1 - pb0) >= scalar(diff_shnum_llm) local count_shnum_llm = `count_shnum_llm' + 1
}
restore

di "ShNum Fisher p (DU_kw) = " %6.3f (`count_shnum_kw' / 1000)
di "ShNum Fisher p (DU_llm) = " %6.3f (`count_shnum_llm' / 1000)

* ----- Post2020 permutation -----
di "Running Post2020 permutation..."
preserve
keep if Post2020 != . & e_pd != . & e_dukw != . & e_dullm != .
local N = _N

forval i = 1/1000 {
    tempvar rand perm_g
    gen double `rand' = runiform()
    sort `rand'
    gen byte `perm_g' = (_n <= `n_post_hi')

    quietly reg e_pd e_dukw if `perm_g' == 1
    scalar pb1 = _b[e_dukw]
    quietly reg e_pd e_dukw if `perm_g' == 0
    scalar pb0 = _b[e_dukw]
    if abs(pb1 - pb0) >= scalar(diff_post_kw) local count_post_kw = `count_post_kw' + 1

    quietly reg e_pd e_dullm if `perm_g' == 1
    scalar pb1 = _b[e_dullm]
    quietly reg e_pd e_dullm if `perm_g' == 0
    scalar pb0 = _b[e_dullm]
    if abs(pb1 - pb0) >= scalar(diff_post_llm) local count_post_llm = `count_post_llm' + 1
}
restore

di "Post2020 Fisher p (DU_kw) = " %6.3f (`count_post_kw' / 1000)
di "Post2020 Fisher p (DU_llm) = " %6.3f (`count_post_llm' / 1000)

di ""
di "============================================="
di "SUMMARY"
di "============================================="
di "Roastd:    DU_kw p=" %5.3f (`count_roastd_kw'/1000) "  DU_llm p=" %5.3f (`count_roastd_llm'/1000)
di "ShNum:     DU_kw p=" %5.3f (`count_shnum_kw'/1000) "  DU_llm p=" %5.3f (`count_shnum_llm'/1000)
di "Post2020:  DU_kw p=" %5.3f (`count_post_kw'/1000) "  DU_llm p=" %5.3f (`count_post_llm'/1000)

di ""
di "DONE"
