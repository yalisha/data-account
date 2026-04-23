* ============================================================
* 表4 稳健性检验 v14 — 27个控制变量
*
* (1) Ind*Year FE
* (2) Prov*Year FE
* (3) 行业年度均值调整 DU_kw
* (4) 剔除2024年
* (5) 剔除信息技术业 (I类)
* (6) 仅主板
* (7) 倾向得分匹配 (PSM)
* (8) 双向聚类标准误 (Firm + Year)
* (9) 前导项检验 (DU_kw + DU_kw_lead)
*
* 数据: reg_sample_v5.dta (27 controls)
* ============================================================

clear all
set more off
set matsize 11000

use "/Users/mac/computerscience/15会计研究/data_stata/reg_sample_v5.dta", clear

* 检查数据
describe, short
count

* 定义13个控制变量 (移除机制渠道+冗余, 加Inv+Hhi)
local controls Size Lev ROA TobinQ Age Growth IndepRatio Dual Top1Share SOE CFO Inv Hhi

* 统一样本: 删除Ind2缺失
drop if missing(Ind2_num)
count

* ============================================================
* 回归模型
* ============================================================
eststo clear

* ---- (1) 控制 Ind*Year FE ----
di _n "=========================================="
di "MODEL (1): Ind*Year FE"
di "=========================================="
eststo r1: reghdfe PriceDelay DU_kw `controls', absorb(Stkcd_num IndYear_num) vce(cluster IndYear_num)
estadd local FirmFE "YES"
estadd local YearFE "NO"
estadd local IndYearFE "YES"
estadd local ProvYearFE "NO"
estadd local Controls "YES"

* ---- (2) 控制 Prov*Year FE ----
di _n "=========================================="
di "MODEL (2): Prov*Year FE"
di "=========================================="
preserve
drop if missing(Prov_num) | missing(ProvYear_num)
eststo r2: reghdfe PriceDelay DU_kw `controls', absorb(Stkcd_num ProvYear_num) vce(cluster IndYear_num)
estadd local FirmFE "YES"
estadd local YearFE "NO"
estadd local IndYearFE "NO"
estadd local ProvYearFE "YES"
estadd local Controls "YES"
restore

* ---- (3) 行业年度均值调整 ----
di _n "=========================================="
di "MODEL (3): Industry-year mean adjusted DU_kw"
di "=========================================="
* DU_kw_indadj已在Python中构造
eststo r3: reghdfe PriceDelay DU_kw_indadj `controls', absorb(Stkcd_num year_num) vce(cluster IndYear_num)
estadd local FirmFE "YES"
estadd local YearFE "YES"
estadd local IndYearFE "NO"
estadd local ProvYearFE "NO"
estadd local Controls "YES"

* ---- (4) 剔除2024年 ----
di _n "=========================================="
di "MODEL (4): Drop 2024"
di "=========================================="
preserve
drop if year == 2024
eststo r4: reghdfe PriceDelay DU_kw `controls', absorb(Stkcd_num year_num) vce(cluster IndYear_num)
estadd local FirmFE "YES"
estadd local YearFE "YES"
estadd local IndYearFE "NO"
estadd local ProvYearFE "NO"
estadd local Controls "YES"
restore

* ---- (5) 剔除信息技术业 (Ind2以I开头) ----
di _n "=========================================="
di "MODEL (5): Drop IT industry"
di "=========================================="
preserve
drop if substr(Ind2, 1, 1) == "I"
eststo r5: reghdfe PriceDelay DU_kw `controls', absorb(Stkcd_num year_num) vce(cluster IndYear_num)
estadd local FirmFE "YES"
estadd local YearFE "YES"
estadd local IndYearFE "NO"
estadd local ProvYearFE "NO"
estadd local Controls "YES"
restore

* ---- (6) 仅主板 ----
di _n "=========================================="
di "MODEL (6): Main board only"
di "=========================================="
preserve
keep if MainBoard == 1
eststo r6: reghdfe PriceDelay DU_kw `controls', absorb(Stkcd_num year_num) vce(cluster IndYear_num)
estadd local FirmFE "YES"
estadd local YearFE "YES"
estadd local IndYearFE "NO"
estadd local ProvYearFE "NO"
estadd local Controls "YES"
restore

* ---- (7) PSM匹配 ----
di _n "=========================================="
di "MODEL (7): PSM matched sample"
di "=========================================="
* DU_kw_high已在Python中构造 (DU_kw > 中位数)
* Logit倾向得分
logit DU_kw_high `controls', nolog
predict pscore, pr

* 近邻匹配 (1:1, caliper=0.05)
psmatch2 DU_kw_high, pscore(pscore) caliper(0.05) noreplacement common

* 在匹配样本上回归
preserve
keep if _support == 1 & _weight != .
count
eststo r7: reghdfe PriceDelay DU_kw `controls', absorb(Stkcd_num year_num) vce(cluster IndYear_num)
estadd local FirmFE "YES"
estadd local YearFE "YES"
estadd local IndYearFE "NO"
estadd local ProvYearFE "NO"
estadd local Controls "YES"
restore

* 清除PSM临时变量
cap drop pscore _pscore _treated _support _weight _id _n1 _nn _pdif

* ---- (8) 双向聚类标准误 (Firm + Year) ----
di _n "=========================================="
di "MODEL (8): Two-way cluster SE"
di "=========================================="
eststo r8: reghdfe PriceDelay DU_kw `controls', absorb(Stkcd_num year_num) vce(cluster Stkcd_num year_num)
estadd local FirmFE "YES"
estadd local YearFE "YES"
estadd local IndYearFE "NO"
estadd local ProvYearFE "NO"
estadd local Controls "YES"

* ---- (9) 前导项检验 ----
di _n "=========================================="
di "MODEL (9): Lead test (DU_kw + DU_kw_lead)"
di "=========================================="
preserve
drop if missing(DU_kw_lead)
eststo r9: reghdfe PriceDelay DU_kw DU_kw_lead `controls', absorb(Stkcd_num year_num) vce(cluster IndYear_num)
estadd local FirmFE "YES"
estadd local YearFE "YES"
estadd local IndYearFE "NO"
estadd local ProvYearFE "NO"
estadd local Controls "YES"
restore

* ============================================================
* 输出结果
* ============================================================

* 变量标签
label variable DU_kw "DU_kw"
label variable DU_kw_indadj "DU_kw_indadj"
label variable DU_kw_lead "DU_kw_lead"

* 纯文本输出(供核对)
di _n "=========================================="
di "FULL RESULTS (4 decimal places)"
di "=========================================="
esttab r1 r2 r3 r4 r5 r6 r7 r8 r9, ///
    b(4) se(4) ///
    star(* 0.10 ** 0.05 *** 0.01) ///
    order(DU_kw DU_kw_indadj DU_kw_lead) ///
    keep(DU_kw DU_kw_indadj DU_kw_lead) ///
    stats(Controls FirmFE YearFE IndYearFE ProvYearFE N r2, ///
        fmt(%s %s %s %s %s %12.0fc %9.3f) ///
        labels("Controls" "Firm FE" "Year FE" "Ind*Year FE" "Prov*Year FE" "N" "R-squared")) ///
    mtitles("(1)IndYear" "(2)ProvYear" "(3)IndAdj" "(4)No2024" "(5)NoIT" "(6)MainBrd" "(7)PSM" "(8)TwoClust" "(9)Lead") ///
    title("Table 4: Robustness Tests (27 controls, v14)")

* LaTeX输出
cap mkdir "/Users/mac/computerscience/15会计研究/results/v15_tables"
esttab r1 r2 r3 r4 r5 r6 r7 r8 r9 using "/Users/mac/computerscience/15会计研究/results/v15_tables/table4_robustness_v14.tex", replace ///
    b(4) se(4) ///
    star(* 0.10 ** 0.05 *** 0.01) ///
    order(DU_kw DU_kw_indadj DU_kw_lead) ///
    keep(DU_kw DU_kw_indadj DU_kw_lead) ///
    stats(Controls FirmFE YearFE IndYearFE ProvYearFE N r2, ///
        fmt(%s %s %s %s %s %12.0fc %9.3f) ///
        labels("Controls" "Firm FE" "Year FE" "Ind$\times$Year FE" "Prov$\times$Year FE" "$N$" "$R^2$")) ///
    mtitles("(1)" "(2)" "(3)" "(4)" "(5)" "(6)" "(7)" "(8)" "(9)") ///
    title("Table 4: Robustness Tests (27 controls)") ///
    booktabs compress

di _n "=========================================="
di "DONE: table4_robustness_v14.do"
di "=========================================="
