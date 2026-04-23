* =============================================================
* 诊断异质性回归：检查变量分布和Winsorize状态
* =============================================================
clear all
set more off
use "data_stata/reg_sample_iv_v16.dta", clear

* 1. DU_kw和PriceDelay是否已Winsorize？
di "============================================="
di "核心变量极端值检查"
di "============================================="
foreach v in PriceDelay DU_kw DU_llm {
    quietly sum `v', detail
    di "`v': p1=" %9.4f r(p1) " p5=" %9.4f r(p5) " p95=" %9.4f r(p95) " p99=" %9.4f r(p99) " max=" %9.4f r(max)
    * Check if max > p99 (sign of no Winsorization)
    if r(max) > r(p99) * 1.1 {
        di "  WARNING: max远超p99，可能未Winsorize"
    }
}

* 2. 控制变量是否Winsorize？
di ""
di "控制变量极端值："
foreach v in Size Lev ROA TobinQ Age Growth IndepRatio Top1Share CFO {
    quietly sum `v', detail
    local ratio = r(max) / r(p99)
    if `ratio' > 1.05 {
        di "`v': max/p99=" %5.2f `ratio' " → 可能未Winsorize"
    }
}

* 3. Winsorize DU_kw和PriceDelay，看异质性结果是否变化
di ""
di "============================================="
di "Winsorize DU_kw/PriceDelay后重跑异质性"
di "============================================="

foreach v in PriceDelay DU_kw DU_llm {
    quietly sum `v', detail
    local p1 = r(p1)
    local p99 = r(p99)
    gen `v'_w = `v'
    replace `v'_w = `p1' if `v' < `p1' & `v' != .
    replace `v'_w = `p99' if `v' > `p99' & `v' != .
}

global controls Size Lev ROA TobinQ Age Growth IndepRatio Dual Top1Share SOE CFO

* --- Baseline with Winsorized vars ---
reghdfe PriceDelay_w DU_kw_w $controls, absorb(Stkcd_num year_num) cluster(IndYear_num)
di "Baseline (winsorized): b=" %9.4f _b[DU_kw_w] " t=" %6.2f (_b[DU_kw_w]/_se[DU_kw_w]) " N=" e(N)

* --- Size split ---
egen med_size = median(Size)
gen big = (Size >= med_size) if Size != .

reghdfe PriceDelay_w DU_kw_w $controls if big == 1, absorb(Stkcd_num year_num) cluster(IndYear_num)
di "Big (w): b=" %9.4f _b[DU_kw_w] " t=" %6.2f (_b[DU_kw_w]/_se[DU_kw_w]) " N=" e(N)

reghdfe PriceDelay_w DU_kw_w $controls if big == 0, absorb(Stkcd_num year_num) cluster(IndYear_num)
di "Small (w): b=" %9.4f _b[DU_kw_w] " t=" %6.2f (_b[DU_kw_w]/_se[DU_kw_w]) " N=" e(N)

reghdfe PriceDelay_w c.DU_kw_w##i.big $controls, absorb(Stkcd_num year_num) cluster(IndYear_num)
di "Size interaction (w): b=" %9.4f _b[c.DU_kw_w#1.big] " t=" %6.2f (_b[c.DU_kw_w#1.big]/_se[c.DU_kw_w#1.big]) " p=" %6.4f (2*ttail(e(df_r), abs(_b[c.DU_kw_w#1.big]/_se[c.DU_kw_w#1.big])))

* --- SOE split ---
reghdfe PriceDelay_w DU_kw_w $controls if SOE == 1, absorb(Stkcd_num year_num) cluster(IndYear_num)
di "SOE=1 (w): b=" %9.4f _b[DU_kw_w] " t=" %6.2f (_b[DU_kw_w]/_se[DU_kw_w]) " N=" e(N)

reghdfe PriceDelay_w DU_kw_w $controls if SOE == 0, absorb(Stkcd_num year_num) cluster(IndYear_num)
di "SOE=0 (w): b=" %9.4f _b[DU_kw_w] " t=" %6.2f (_b[DU_kw_w]/_se[DU_kw_w]) " N=" e(N)

reghdfe PriceDelay_w c.DU_kw_w##i.SOE $controls, absorb(Stkcd_num year_num) cluster(IndYear_num)
di "SOE interaction (w): b=" %9.4f _b[c.DU_kw_w#1.SOE] " t=" %6.2f (_b[c.DU_kw_w#1.SOE]/_se[c.DU_kw_w#1.SOE]) " p=" %6.4f (2*ttail(e(df_r), abs(_b[c.DU_kw_w#1.SOE]/_se[c.DU_kw_w#1.SOE])))

* --- Analyst split ---
egen med_analyst = median(Analyst)
gen hi_analyst = (Analyst >= med_analyst) if Analyst != .

reghdfe PriceDelay_w DU_kw_w $controls if hi_analyst == 1, absorb(Stkcd_num year_num) cluster(IndYear_num)
di "Hi Analyst (w): b=" %9.4f _b[DU_kw_w] " t=" %6.2f (_b[DU_kw_w]/_se[DU_kw_w]) " N=" e(N)

reghdfe PriceDelay_w DU_kw_w $controls if hi_analyst == 0, absorb(Stkcd_num year_num) cluster(IndYear_num)
di "Lo Analyst (w): b=" %9.4f _b[DU_kw_w] " t=" %6.2f (_b[DU_kw_w]/_se[DU_kw_w]) " N=" e(N)

reghdfe PriceDelay_w c.DU_kw_w##i.hi_analyst $controls, absorb(Stkcd_num year_num) cluster(IndYear_num)
di "Analyst interaction (w): b=" %9.4f _b[c.DU_kw_w#1.hi_analyst] " t=" %6.2f (_b[c.DU_kw_w#1.hi_analyst]/_se[c.DU_kw_w#1.hi_analyst]) " p=" %6.4f (2*ttail(e(df_r), abs(_b[c.DU_kw_w#1.hi_analyst]/_se[c.DU_kw_w#1.hi_analyst])))

* --- Industry split ---
gen hitech = inlist(Ind2_num, 60, 61, 62, 70, 71, 72)

reghdfe PriceDelay_w DU_kw_w $controls if hitech == 1, absorb(Ind2_num year_num) cluster(IndYear_num)
di "HiTech (w): b=" %9.4f _b[DU_kw_w] " t=" %6.2f (_b[DU_kw_w]/_se[DU_kw_w]) " N=" e(N)

reghdfe PriceDelay_w DU_kw_w $controls if hitech == 0, absorb(Stkcd_num year_num) cluster(IndYear_num)
di "Traditional (w): b=" %9.4f _b[DU_kw_w] " t=" %6.2f (_b[DU_kw_w]/_se[DU_kw_w]) " N=" e(N)

reghdfe PriceDelay_w c.DU_kw_w##i.hitech $controls, absorb(Stkcd_num year_num) cluster(IndYear_num)
di "Industry interaction (w): b=" %9.4f _b[c.DU_kw_w#1.hitech] " t=" %6.2f (_b[c.DU_kw_w#1.hitech]/_se[c.DU_kw_w#1.hitech]) " p=" %6.4f (2*ttail(e(df_r), abs(_b[c.DU_kw_w#1.hitech]/_se[c.DU_kw_w#1.hitech])))

di ""
di "============================================="
di "对比：未Winsorize原始结果"
di "============================================="
reghdfe PriceDelay DU_kw $controls, absorb(Stkcd_num year_num) cluster(IndYear_num)
di "Baseline (raw): b=" %9.4f _b[DU_kw] " t=" %6.2f (_b[DU_kw]/_se[DU_kw])

reghdfe PriceDelay c.DU_kw##i.big $controls, absorb(Stkcd_num year_num) cluster(IndYear_num)
di "Size interaction (raw): p=" %6.4f (2*ttail(e(df_r), abs(_b[c.DU_kw#1.big]/_se[c.DU_kw#1.big])))
