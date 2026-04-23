* =============================================================
* Final Heterogeneity: 3 dimensions × 2 measures, full results
* For manuscript Table 7
* =============================================================
clear all
set more off
use "data_stata/reg_sample_het.dta", clear

global controls Size Lev ROA TobinQ Age Growth IndepRatio Dual Top1Share SOE CFO

* Construct grouping variables
egen med_analyst = median(Analyst)
gen Hi_Analyst = (Analyst >= med_analyst) if Analyst != .

egen med_shnum = median(ShareholderNum)
gen Hi_ShNum = (ShareholderNum >= med_shnum) if ShareholderNum != .

di "============================================="
di "Table 7: Heterogeneity (3 dimensions × 2 measures)"
di "============================================="

* --- (1) SOE ---
di ""
di "=== SOE ==="
foreach du in DU_kw DU_llm {
    reghdfe PriceDelay `du' $controls if SOE == 1, absorb(Stkcd_num year_num) cluster(IndYear_num)
    di "SOE=1 `du': b=" %9.4f _b[`du'] " se=" %9.4f _se[`du'] " t=" %6.2f (_b[`du']/_se[`du']) " N=" e(N)

    reghdfe PriceDelay `du' $controls if SOE == 0, absorb(Stkcd_num year_num) cluster(IndYear_num)
    di "SOE=0 `du': b=" %9.4f _b[`du'] " se=" %9.4f _se[`du'] " t=" %6.2f (_b[`du']/_se[`du']) " N=" e(N)
}

* --- (2) Analyst ---
di ""
di "=== Analyst ==="
foreach du in DU_kw DU_llm {
    reghdfe PriceDelay `du' $controls if Hi_Analyst == 1, absorb(Stkcd_num year_num) cluster(IndYear_num)
    di "Hi_Analyst `du': b=" %9.4f _b[`du'] " se=" %9.4f _se[`du'] " t=" %6.2f (_b[`du']/_se[`du']) " N=" e(N)

    reghdfe PriceDelay `du' $controls if Hi_Analyst == 0, absorb(Stkcd_num year_num) cluster(IndYear_num)
    di "Lo_Analyst `du': b=" %9.4f _b[`du'] " se=" %9.4f _se[`du'] " t=" %6.2f (_b[`du']/_se[`du']) " N=" e(N)
}

* --- (3) ShNum ---
di ""
di "=== ShareholderNum ==="
foreach du in DU_kw DU_llm {
    reghdfe PriceDelay `du' $controls if Hi_ShNum == 1, absorb(Stkcd_num year_num) cluster(IndYear_num)
    di "Hi_ShNum `du': b=" %9.4f _b[`du'] " se=" %9.4f _se[`du'] " t=" %6.2f (_b[`du']/_se[`du']) " N=" e(N)

    reghdfe PriceDelay `du' $controls if Hi_ShNum == 0, absorb(Stkcd_num year_num) cluster(IndYear_num)
    di "Lo_ShNum `du': b=" %9.4f _b[`du'] " se=" %9.4f _se[`du'] " t=" %6.2f (_b[`du']/_se[`du']) " N=" e(N)
}

di ""
di "DONE"
