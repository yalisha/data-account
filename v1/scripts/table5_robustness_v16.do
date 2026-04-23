* ============================================================
* 表5 稳健性检验 v16 — 11个控制变量, Stata统一样本
*
* (1) 行业×年份 FE
* (2) 剔除末期年份 (2024)
* (3) 剔除IT行业
* (4) 滞后控制变量
* (5) PSM匹配
* (6) 双向聚类标准误 (Firm + Year)
* (7) 替换自变量: DU_sub_ln
*
* 数据: reg_sample_iv_v16.dta (11 controls, N=43,735)
* ============================================================

clear all
set more off
set matsize 11000

use "/Users/mac/computerscience/15会计研究/data_stata/reg_sample_iv_v16.dta", clear

describe, short
count

* 定义11个控制变量
local controls Size Lev ROA TobinQ Age Growth IndepRatio Dual Top1Share SOE CFO

* ============================================================
* (1) 行业×年份 FE
* ============================================================
di _n "=========================================="
di "MODEL (1): Ind*Year FE"
di "=========================================="
eststo clear
eststo r1: reghdfe PriceDelay DU_kw `controls', absorb(Stkcd_num IndYear_num) vce(cluster IndYear_num)
di "coef = " _b[DU_kw]
di "se = " _se[DU_kw]
di "t = " _b[DU_kw]/_se[DU_kw]
di "N = " e(N)

* ============================================================
* (2) 剔除末期年份 (2024)
* ============================================================
di _n "=========================================="
di "MODEL (2): Drop 2024"
di "=========================================="
preserve
drop if year_num == 2024
eststo r2: reghdfe PriceDelay DU_kw `controls', absorb(Stkcd_num year_num) vce(cluster IndYear_num)
di "coef = " _b[DU_kw]
di "se = " _se[DU_kw]
di "t = " _b[DU_kw]/_se[DU_kw]
di "N = " e(N)
restore

* ============================================================
* (3) 剔除IT行业 (信息传输、软件和信息技术服务业, I类)
* ============================================================
di _n "=========================================="
di "MODEL (3): Drop IT industry"
di "=========================================="
preserve
drop if substr(Ind2, 1, 1) == "I"
eststo r3: reghdfe PriceDelay DU_kw `controls', absorb(Stkcd_num year_num) vce(cluster IndYear_num)
di "coef = " _b[DU_kw]
di "se = " _se[DU_kw]
di "t = " _b[DU_kw]/_se[DU_kw]
di "N = " e(N)
restore

* ============================================================
* (4) 滞后控制变量
* ============================================================
di _n "=========================================="
di "MODEL (4): Lagged controls"
di "=========================================="
* xtset生成滞后变量
xtset Stkcd_num year_num
foreach v of local controls {
    cap gen L_`v' = L.`v'
}
local lag_controls L_Size L_Lev L_ROA L_TobinQ L_Age L_Growth L_IndepRatio L_Dual L_Top1Share L_SOE L_CFO

preserve
* 删除滞后变量缺失的观测
foreach v of local lag_controls {
    drop if missing(`v')
}
eststo r4: reghdfe PriceDelay DU_kw `lag_controls', absorb(Stkcd_num year_num) vce(cluster IndYear_num)
di "coef = " _b[DU_kw]
di "se = " _se[DU_kw]
di "t = " _b[DU_kw]/_se[DU_kw]
di "N = " e(N)
restore

* ============================================================
* (5) PSM匹配
* ============================================================
di _n "=========================================="
di "MODEL (5): PSM matched sample"
di "=========================================="

* Logit倾向得分: DU_kw_high已在数据集中(行业-年份中位数以上=1)
logit DU_kw_high `controls', nolog
predict pscore, pr

* 近邻匹配 (1:1, caliper=0.05, 无放回)
psmatch2 DU_kw_high, pscore(pscore) caliper(0.05) noreplacement common

* 在匹配样本上回归
preserve
keep if _support == 1 & _weight != .
count
eststo r5: reghdfe PriceDelay DU_kw `controls', absorb(Stkcd_num year_num) vce(cluster IndYear_num)
di "coef = " _b[DU_kw]
di "se = " _se[DU_kw]
di "t = " _b[DU_kw]/_se[DU_kw]
di "N = " e(N)
restore

* 清除PSM临时变量
cap drop pscore _pscore _treated _support _weight _id _n1 _nn _pdif

* ============================================================
* (6) 双向聚类标准误 (Firm + Year)
* ============================================================
di _n "=========================================="
di "MODEL (6): Two-way cluster SE (Firm + Year)"
di "=========================================="
eststo r6: reghdfe PriceDelay DU_kw `controls', absorb(Stkcd_num year_num) vce(cluster Stkcd_num year_num)
di "coef = " _b[DU_kw]
di "se = " _se[DU_kw]
di "t = " _b[DU_kw]/_se[DU_kw]
di "N = " e(N)

* ============================================================
* (7) 替换自变量: DU_sub_ln
* ============================================================
di _n "=========================================="
di "MODEL (7): Alternative DV: DU_sub_ln"
di "=========================================="
eststo r7: reghdfe PriceDelay DU_sub_ln `controls', absorb(Stkcd_num year_num) vce(cluster IndYear_num)
di "coef = " _b[DU_sub_ln]
di "se = " _se[DU_sub_ln]
di "t = " _b[DU_sub_ln]/_se[DU_sub_ln]
di "N = " e(N)

* ============================================================
* 汇总输出
* ============================================================
di _n "=========================================="
di "SUMMARY: Table 5 Robustness v16"
di "=========================================="
esttab r1 r2 r3 r4 r5 r6 r7, ///
    b(4) se(4) ///
    star(* 0.10 ** 0.05 *** 0.01) ///
    order(DU_kw DU_sub_ln) ///
    keep(DU_kw DU_sub_ln) ///
    stats(N r2, fmt(%12.0fc %9.4f) labels("N" "R-squared")) ///
    mtitles("(1)IndYrFE" "(2)No2024" "(3)NoIT" "(4)LagCtrl" "(5)PSM" "(6)TwoClust" "(7)DU_sub") ///
    title("Table 5: Robustness Tests (v16, 11 controls)")

* LaTeX输出
cap mkdir "/Users/mac/computerscience/15会计研究/results/v16_tables"
esttab r1 r2 r3 r4 r5 r6 r7 using "/Users/mac/computerscience/15会计研究/results/v16_tables/table5_robustness_v16.tex", replace ///
    b(4) se(4) ///
    star(* 0.10 ** 0.05 *** 0.01) ///
    order(DU_kw DU_sub_ln) ///
    keep(DU_kw DU_sub_ln) ///
    stats(N r2, fmt(%12.0fc %9.4f) labels("$N$" "$R^2$")) ///
    mtitles("(1)" "(2)" "(3)" "(4)" "(5)" "(6)" "(7)") ///
    title("Table 5: Robustness Tests") ///
    booktabs compress

di _n "=========================================="
di "DONE: table5_robustness_v16.do"
di "=========================================="
