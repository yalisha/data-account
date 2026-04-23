* =============================================================
* Fisher Permutation Test: ALL dimensions (1000 reps, FWL)
* =============================================================
clear all
set more off
set seed 20260326
use "data_stata/reg_sample_het.dta", clear

global controls Size Lev ROA TobinQ Age Growth IndepRatio Dual Top1Share SOE CFO

* ---- Step 1: Partial out FE + controls (once) ----
di "Partialling out FE + controls..."
reghdfe PriceDelay $controls, absorb(Stkcd_num year_num) resid(e_pd)
reghdfe DU_kw $controls, absorb(Stkcd_num year_num) resid(e_dukw)
reghdfe DU_llm $controls, absorb(Stkcd_num year_num) resid(e_dullm)
di "Done."

* ---- Step 2: Construct ALL grouping variables ----
* Old 4
egen med_size = median(Size)
gen Hi_Size = (Size >= med_size) if Size != .

egen med_analyst = median(Analyst)
gen Hi_Analyst = (Analyst >= med_analyst) if Analyst != .

gen hitech = inlist(Ind2_num, 60, 61, 62, 70, 71, 72)

* New round 1
gen Post2020 = (year_num >= 2020)

egen med_intang = median(Intangible)
gen Hi_Intang = (Intangible >= med_intang) if Intangible != .

egen med_mkt = median(Market)
gen Hi_Market = (Market >= med_mkt) if Market != .

* Round 2
egen med_roastd = median(Roastd)
gen Hi_Roastd = (Roastd >= med_roastd) if Roastd != .

egen med_shnum = median(ShareholderNum)
gen Hi_ShNum = (ShareholderNum >= med_shnum) if ShareholderNum != .

egen med_age = median(Age)
gen Old = (Age >= med_age) if Age != .

egen med_growth = median(Growth)
gen Hi_Growth = (Growth >= med_growth) if Growth != .

egen med_bm = median(BM)
gen Hi_BM = (BM >= med_bm) if BM != .

egen med_hhi = median(Hhi)
gen Hi_Hhi = (Hhi >= med_hhi) if Hhi != .

* ---- Step 3: Fisher test program ----
capture program drop fisher_test
program define fisher_test, rclass
    syntax, group(varname) nreps(integer)

    * Restrict to non-missing
    tempvar use_obs
    gen byte `use_obs' = (`group' != . & e_pd != . & e_dukw != . & e_dullm != .)

    * Count group sizes
    quietly count if `group' == 1 & `use_obs'
    local n_hi = r(N)
    quietly count if `use_obs'
    local n_total = r(N)

    * Actual differences
    quietly reg e_pd e_dukw if `group' == 1 & `use_obs'
    scalar ab1_kw = _b[e_dukw]
    quietly reg e_pd e_dukw if `group' == 0 & `use_obs'
    scalar ab0_kw = _b[e_dukw]
    scalar adiff_kw = abs(ab1_kw - ab0_kw)

    quietly reg e_pd e_dullm if `group' == 1 & `use_obs'
    scalar ab1_llm = _b[e_dullm]
    quietly reg e_pd e_dullm if `group' == 0 & `use_obs'
    scalar ab0_llm = _b[e_dullm]
    scalar adiff_llm = abs(ab1_llm - ab0_llm)

    * Permutation
    local ck = 0
    local cl = 0

    preserve
    keep if `use_obs'

    forval i = 1/`nreps' {
        tempvar rand pg
        gen double `rand' = runiform()
        sort `rand'
        gen byte `pg' = (_n <= `n_hi')

        quietly reg e_pd e_dukw if `pg' == 1
        scalar pb1 = _b[e_dukw]
        quietly reg e_pd e_dukw if `pg' == 0
        scalar pb0 = _b[e_dukw]
        if abs(pb1 - pb0) >= scalar(adiff_kw) local ck = `ck' + 1

        quietly reg e_pd e_dullm if `pg' == 1
        scalar pb1 = _b[e_dullm]
        quietly reg e_pd e_dullm if `pg' == 0
        scalar pb0 = _b[e_dullm]
        if abs(pb1 - pb0) >= scalar(adiff_llm) local cl = `cl' + 1
    }

    restore

    return scalar p_kw = `ck' / `nreps'
    return scalar p_llm = `cl' / `nreps'
    return scalar diff_kw = scalar(adiff_kw)
    return scalar diff_llm = scalar(adiff_llm)
    return scalar b1_kw = scalar(ab1_kw)
    return scalar b0_kw = scalar(ab0_kw)
    return scalar b1_llm = scalar(ab1_llm)
    return scalar b0_llm = scalar(ab0_llm)
    return scalar n_total = `n_total'
end

* ---- Step 4: Run all tests ----
di ""
di "============================================="
di "Fisher Permutation: All Dimensions (1000 reps)"
di "============================================="
di ""
di "Dimension          | G1_kw      G0_kw      Fisher_kw | G1_llm     G0_llm     Fisher_llm | N"
di "-------------------|----------------------------------|----------------------------------|------"

foreach dim in SOE Hi_Size Hi_Analyst hitech Post2020 Hi_Intang Hi_Market Big4 Hi_Roastd Hi_ShNum Old Hi_Growth Hi_BM Hi_Hhi {
    fisher_test, group(`dim') nreps(1000)
    di "`dim'" _col(20) "| " %9.5f r(b1_kw) "  " %9.5f r(b0_kw) "  " %5.3f r(p_kw) "    | " %9.5f r(b1_llm) "  " %9.5f r(b0_llm) "  " %5.3f r(p_llm) "    | " %6.0f r(n_total)
}

di ""
di "DONE"
