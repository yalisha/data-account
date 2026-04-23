* =============================================================
* Table 6: Mechanism + Table 7: Heterogeneity (Stata)
* =============================================================
clear all
set more off
use "data_stata/reg_sample_iv_v16.dta", clear

global controls Size Lev ROA TobinQ Age Growth IndepRatio Dual Top1Share SOE CFO

di "============================================="
di "Table 6: Mechanism Channels"
di "============================================="

* ---- 5 channels × 2 measures ----
foreach dv in Analyst Amihud RetVol Turnover InstHold {
    foreach du in DU_kw DU_llm {
        cap reghdfe `dv' `du' $controls, absorb(Stkcd_num year_num) cluster(IndYear_num)
        if _rc == 0 {
            di "`dv' ~ `du': b=" %9.4f _b[`du'] " se=" %9.4f _se[`du'] " t=" %6.2f (_b[`du']/_se[`du']) " N=" e(N)
        }
    }
}

di ""
di "============================================="
di "Table 7: Heterogeneity"
di "============================================="

* ---- (1) SOE split ----
reghdfe PriceDelay DU_kw $controls if SOE == 1, absorb(Stkcd_num year_num) cluster(IndYear_num)
di "SOE=1: b=" %9.4f _b[DU_kw] " se=" %9.4f _se[DU_kw] " N=" e(N)

reghdfe PriceDelay DU_kw $controls if SOE == 0, absorb(Stkcd_num year_num) cluster(IndYear_num)
di "SOE=0: b=" %9.4f _b[DU_kw] " se=" %9.4f _se[DU_kw] " N=" e(N)

* SOE interaction
reghdfe PriceDelay c.DU_kw##i.SOE $controls, absorb(Stkcd_num year_num) cluster(IndYear_num)
di "SOE interaction: b=" %9.4f _b[c.DU_kw#1.SOE] " se=" %9.4f _se[c.DU_kw#1.SOE] " t=" %6.2f (_b[c.DU_kw#1.SOE]/_se[c.DU_kw#1.SOE]) " p=" %6.4f (2*ttail(e(df_r), abs(_b[c.DU_kw#1.SOE]/_se[c.DU_kw#1.SOE])))

* ---- (2) Size split (median) ----
egen med_size = median(Size)
gen big = (Size >= med_size) if Size != .

reghdfe PriceDelay DU_kw $controls if big == 1, absorb(Stkcd_num year_num) cluster(IndYear_num)
di "Big: b=" %9.4f _b[DU_kw] " se=" %9.4f _se[DU_kw] " N=" e(N)

reghdfe PriceDelay DU_kw $controls if big == 0, absorb(Stkcd_num year_num) cluster(IndYear_num)
di "Small: b=" %9.4f _b[DU_kw] " se=" %9.4f _se[DU_kw] " N=" e(N)

* Size interaction
reghdfe PriceDelay c.DU_kw##i.big $controls, absorb(Stkcd_num year_num) cluster(IndYear_num)
di "Size interaction: b=" %9.4f _b[c.DU_kw#1.big] " se=" %9.4f _se[c.DU_kw#1.big] " t=" %6.2f (_b[c.DU_kw#1.big]/_se[c.DU_kw#1.big]) " p=" %6.4f (2*ttail(e(df_r), abs(_b[c.DU_kw#1.big]/_se[c.DU_kw#1.big])))

* ---- (3) Analyst coverage split (median) ----
egen med_analyst = median(Analyst)
gen hi_analyst = (Analyst >= med_analyst) if Analyst != .

reghdfe PriceDelay DU_kw $controls if hi_analyst == 1, absorb(Stkcd_num year_num) cluster(IndYear_num)
di "Hi Analyst: b=" %9.4f _b[DU_kw] " se=" %9.4f _se[DU_kw] " N=" e(N)

reghdfe PriceDelay DU_kw $controls if hi_analyst == 0, absorb(Stkcd_num year_num) cluster(IndYear_num)
di "Lo Analyst: b=" %9.4f _b[DU_kw] " se=" %9.4f _se[DU_kw] " N=" e(N)

* Analyst interaction
reghdfe PriceDelay c.DU_kw##i.hi_analyst $controls, absorb(Stkcd_num year_num) cluster(IndYear_num)
di "Analyst interaction: b=" %9.4f _b[c.DU_kw#1.hi_analyst] " se=" %9.4f _se[c.DU_kw#1.hi_analyst] " t=" %6.2f (_b[c.DU_kw#1.hi_analyst]/_se[c.DU_kw#1.hi_analyst]) " p=" %6.4f (2*ttail(e(df_r), abs(_b[c.DU_kw#1.hi_analyst]/_se[c.DU_kw#1.hi_analyst])))

* ---- (4) Industry split (high-tech vs traditional) ----
* High-tech = I63,I64,I65,M73,M74,M75 = Ind2_num 60,61,62,70,71,72
gen hitech = inlist(Ind2_num, 60, 61, 62, 70, 71, 72)

* High-tech: use industry+year FE (too few firms for firm FE)
reghdfe PriceDelay DU_kw $controls if hitech == 1, absorb(Ind2_num year_num) cluster(IndYear_num)
di "HiTech: b=" %9.4f _b[DU_kw] " se=" %9.4f _se[DU_kw] " N=" e(N)

reghdfe PriceDelay DU_kw $controls if hitech == 0, absorb(Stkcd_num year_num) cluster(IndYear_num)
di "Traditional: b=" %9.4f _b[DU_kw] " se=" %9.4f _se[DU_kw] " N=" e(N)

* Industry interaction (full sample, firm+year FE)
reghdfe PriceDelay c.DU_kw##i.hitech $controls, absorb(Stkcd_num year_num) cluster(IndYear_num)
di "Industry interaction: b=" %9.4f _b[c.DU_kw#1.hitech] " se=" %9.4f _se[c.DU_kw#1.hitech] " t=" %6.2f (_b[c.DU_kw#1.hitech]/_se[c.DU_kw#1.hitech]) " p=" %6.4f (2*ttail(e(df_r), abs(_b[c.DU_kw#1.hitech]/_se[c.DU_kw#1.hitech])))
