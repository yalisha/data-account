* ============================================================
* 表3 稳健性检验 — Stata reghdfe (8列)
*
* (1) 控制Ind×Year FE
* (2) 控制Prov×Year FE
* (3) 行业年度均值调整 DU_kw
* (4) 剔除2024年
* (5) 剔除信息技术业 (I类)
* (6) 仅主板
* (7) 倾向得分匹配 (PSM)
* (8) 双向聚类标准误 (Firm + Year)
*
* 数据: reg_sample_v3.dta
* ============================================================

clear all
set more off

use "/Users/mac/computerscience/15会计研究/data_stata/reg_sample_v3.dta", clear

* 检查数据
describe, short
count

* 定义控制变量
global controls "Size Lev ROA TobinQ Age Growth BoardSize IndepRatio Dual Top1Share SOE InstHold Amihud Analyst AuditType"

* 统一样本: 删除Ind2缺失
drop if missing(Ind2_num)
count

* ============================================================
* 回归模型
* ============================================================
eststo clear

* ---- (1) 控制 Ind×Year FE ----
* absorb企业FE + 行业×年份交互FE
eststo r1: reghdfe PriceDelay DU_kw $controls, absorb(Stkcd_num IndYear_num) vce(cluster IndYear_num)
estadd local FirmFE "YES"
estadd local YearFE "NO"
estadd local IndYearFE "YES"
estadd local ProvYearFE "NO"
estadd local Controls "YES"

* ---- (2) 控制 Prov×Year FE ----
* 需要删除Province缺失
preserve
drop if missing(Prov_num)
eststo r2: reghdfe PriceDelay DU_kw $controls, absorb(Stkcd_num ProvYear_num) vce(cluster IndYear_num)
estadd local FirmFE "YES"
estadd local YearFE "NO"
estadd local IndYearFE "NO"
estadd local ProvYearFE "YES"
estadd local Controls "YES"
restore

* ---- (3) 行业年度均值调整 ----
eststo r3: reghdfe PriceDelay DU_kw_indadj $controls, absorb(Stkcd_num year_num) vce(cluster IndYear_num)
estadd local FirmFE "YES"
estadd local YearFE "YES"
estadd local IndYearFE "NO"
estadd local ProvYearFE "NO"
estadd local Controls "YES"

* ---- (4) 剔除2024年 ----
preserve
drop if year == 2024
eststo r4: reghdfe PriceDelay DU_kw $controls, absorb(Stkcd_num year_num) vce(cluster IndYear_num)
estadd local FirmFE "YES"
estadd local YearFE "YES"
estadd local IndYearFE "NO"
estadd local ProvYearFE "NO"
estadd local Controls "YES"
restore

* ---- (5) 剔除信息技术业 (Ind2以I开头) ----
preserve
drop if substr(Ind2, 1, 1) == "I"
eststo r5: reghdfe PriceDelay DU_kw $controls, absorb(Stkcd_num year_num) vce(cluster IndYear_num)
estadd local FirmFE "YES"
estadd local YearFE "YES"
estadd local IndYearFE "NO"
estadd local ProvYearFE "NO"
estadd local Controls "YES"
restore

* ---- (6) 仅主板 ----
preserve
keep if MainBoard == 1
eststo r6: reghdfe PriceDelay DU_kw $controls, absorb(Stkcd_num year_num) vce(cluster IndYear_num)
estadd local FirmFE "YES"
estadd local YearFE "YES"
estadd local IndYearFE "NO"
estadd local ProvYearFE "NO"
estadd local Controls "YES"
restore

* ---- (7) PSM匹配 ----
* 第一步: 生成处理组 (DU_kw > 中位数)
summarize DU_kw, detail
local med = r(p50)
gen Treat_psm = (DU_kw > `med')

* 第二步: Logit倾向得分
logit Treat_psm $controls, nolog
predict pscore, pr

* 第三步: 近邻匹配 (1:1, caliper=0.05)
* 需要psmatch2
psmatch2 Treat_psm, pscore(pscore) caliper(0.05) noreplacement common

* 第四步: 在匹配样本上回归
preserve
keep if _support == 1 & _weight != .
eststo r7: reghdfe PriceDelay DU_kw $controls, absorb(Stkcd_num year_num) vce(cluster IndYear_num)
estadd local FirmFE "YES"
estadd local YearFE "YES"
estadd local IndYearFE "NO"
estadd local ProvYearFE "NO"
estadd local Controls "YES"
restore

* ---- (8) 双向聚类标准误 (Firm + Year) ----
* 同一模型不同推断方式, 系数不变SE变化
eststo r8: reghdfe PriceDelay DU_kw $controls, absorb(Stkcd_num year_num) vce(cluster Stkcd_num year_num)
estadd local FirmFE "YES"
estadd local YearFE "YES"
estadd local IndYearFE "NO"
estadd local ProvYearFE "NO"
estadd local Controls "YES"

* ============================================================
* 输出表格
* ============================================================

* 变量标签
label variable DU_kw "DU\_kw"
label variable DU_kw_indadj "DU\_kw\_indadj"

* LaTeX输出
esttab r1 r2 r3 r4 r5 r6 r7 r8 using "/Users/mac/computerscience/15会计研究/results/v9_tables/table3_stata.tex", replace ///
    b(3) se(3) ///
    star(* 0.10 ** 0.05 *** 0.01) ///
    order(DU_kw DU_kw_indadj) ///
    keep(DU_kw DU_kw_indadj) ///
    stats(Controls FirmFE YearFE IndYearFE ProvYearFE N r2, ///
        fmt(%s %s %s %s %s %12.0fc %9.3f) ///
        labels("Controls" "Firm FE" "Year FE" "Ind\$\times\$Year FE" "Prov\$\times\$Year FE" "\$N\$" "\$R^2\$")) ///
    mtitles("(1)" "(2)" "(3)" "(4)" "(5)" "(6)" "(7)" "(8)") ///
    mgroups("Ind\$\times\$Year" "Prov\$\times\$Year" "IndAdj" "No 2024" "No IT" "MainBoard" "PSM" "Two-way", ///
        pattern(1 1 1 1 1 1 1 1)) ///
    nonotes addnotes( ///
        "注：括号内数值为稳健标准误。" ///
        "***、**、*分别表示1\%、5\%、10\%的水平上显著，下同。" ///
        "模型(1)控制企业和行业×年份交互固定效应。" ///
        "模型(2)控制企业和省份×年份交互固定效应。" ///
        "模型(3)使用行业年度均值调整后的DU\_kw。" ///
        "模型(7)使用倾向得分匹配后的样本。" ///
        "模型(8)使用企业和年份双向聚类标准误。") ///
    title("表3 稳健性检验") ///
    booktabs alignment(S) compress

* 纯文本输出(供核对)
esttab r1 r2 r3 r4 r5 r6 r7 r8, ///
    b(4) se(4) ///
    star(* 0.10 ** 0.05 *** 0.01) ///
    order(DU_kw DU_kw_indadj) ///
    keep(DU_kw DU_kw_indadj) ///
    stats(Controls FirmFE YearFE IndYearFE ProvYearFE N r2, ///
        fmt(%s %s %s %s %s %12.0fc %9.3f) ///
        labels("Controls" "Firm FE" "Year FE" "Ind*Year FE" "Prov*Year FE" "N" "R-squared")) ///
    mtitles("(1)" "(2)" "(3)" "(4)" "(5)" "(6)" "(7)" "(8)") ///
    title("Table 3: Robustness (4 decimal places for review)")
