* ============================================================
* 表2 基准回归检验 — Stata reghdfe 复现
* 数据要素利用、资产配置与资本定价效率
*
* 数据准备: Python脚本导出 data_stata/reg_sample_v2.dta
*   - panel.parquet + annual_report_features.parquet 合并
*   - 行业代码用公司众数填补(修复原Ind2缺失9512条的bug)
*   - 样本筛选: 非金融/非ST/Age>0
*   - 连续变量1%/99%缩尾
*
* 依赖包: reghdfe, ftools, estout (ssc install)
* ============================================================

clear all
set more off

* ---- 加载数据 ----
use "/Users/mac/computerscience/15会计研究/data_stata/reg_sample_v2.dta", clear

* 检查数据
describe, short
summarize PriceDelay DU_kw DU_kw_ln DU_sub_ln SYNCH FinAsset, detail

* 定义控制变量
global controls "Size Lev ROA TobinQ Age Growth BoardSize IndepRatio Dual Top1Share SOE InstHold Amihud Analyst AuditType"

* 删除Ind2缺失观测, 保证模型(5)与其他模型样本一致
drop if missing(Ind2_num)
count

* ============================================================
* 回归模型
* ============================================================
eststo clear

* (1) 仅DU_kw, 企业+年份FE, 无控制变量
eststo m1: reghdfe PriceDelay DU_kw, absorb(Stkcd_num year_num) vce(cluster IndYear_num)
estadd local FirmYearFE "YES"
estadd local IndYearFE "NO"
estadd local Controls "NO"

* (2) DU_kw + 控制变量, 企业+年份FE
eststo m2: reghdfe PriceDelay DU_kw $controls, absorb(Stkcd_num year_num) vce(cluster IndYear_num)
estadd local FirmYearFE "YES"
estadd local IndYearFE "NO"
estadd local Controls "YES"

* (3) 替换解释变量: ln(1+关键词总数), 企业+年份FE
eststo m3: reghdfe PriceDelay DU_kw_ln $controls, absorb(Stkcd_num year_num) vce(cluster IndYear_num)
estadd local FirmYearFE "YES"
estadd local IndYearFE "NO"
estadd local Controls "YES"

* (4) 替换解释变量: ln(1+实质利用次数), 企业+年份FE
eststo m4: reghdfe PriceDelay DU_sub_ln $controls, absorb(Stkcd_num year_num) vce(cluster IndYear_num)
estadd local FirmYearFE "YES"
estadd local IndYearFE "NO"
estadd local Controls "YES"

* (5) 替换固定效应: 行业+年份FE (同样本)
eststo m5: reghdfe PriceDelay DU_kw $controls, absorb(Ind2_num year_num) vce(cluster IndYear_num)
estadd local FirmYearFE "NO"
estadd local IndYearFE "YES"
estadd local Controls "YES"

* (6) 替换被解释变量: SYNCH, 企业+年份FE
eststo m6: reghdfe SYNCH DU_kw $controls, absorb(Stkcd_num year_num) vce(cluster IndYear_num)
estadd local FirmYearFE "YES"
estadd local IndYearFE "NO"
estadd local Controls "YES"

* (7) 额外控制金融资产占比, 企业+年份FE
eststo m7: reghdfe PriceDelay DU_kw FinAsset $controls, absorb(Stkcd_num year_num) vce(cluster IndYear_num)
estadd local FirmYearFE "YES"
estadd local IndYearFE "NO"
estadd local Controls "YES"

* ============================================================
* 输出表格
* ============================================================

* 变量排序
local varorder "DU_kw DU_kw_ln DU_sub_ln FinAsset Size Lev ROA TobinQ Age Growth BoardSize IndepRatio Dual Top1Share SOE InstHold Amihud Analyst AuditType _cons"

* 变量标签(LaTeX转义)
label variable DU_kw "DU\_kw"
label variable DU_kw_ln "DU\_kw\_ln"
label variable DU_sub_ln "DU\_sub\_ln"
label variable FinAsset "FinAsset"
label variable Size "Size"
label variable Lev "Lev"
label variable ROA "ROA"
label variable TobinQ "TobinQ"
label variable Age "Age"
label variable Growth "Growth"
label variable BoardSize "BoardSize"
label variable IndepRatio "IndepRatio"
label variable Dual "Dual"
label variable Top1Share "Top1Share"
label variable SOE "SOE"
label variable InstHold "InstHold"
label variable Amihud "Amihud"
label variable Analyst "Analyst"
label variable AuditType "AuditType"

* LaTeX输出 (3位小数, Top1Share/InstHold/SOE需手动改为4位)
esttab m1 m2 m3 m4 m5 m6 m7 using "/Users/mac/computerscience/15会计研究/results/v9_tables/table2_stata.tex", replace ///
    b(3) se(3) ///
    star(* 0.10 ** 0.05 *** 0.01) ///
    order(`varorder') ///
    stats(Controls FirmYearFE IndYearFE N r2, ///
        fmt(%s %s %s %12.0fc %9.3f) ///
        labels("Controls" "Firm/Year FE" "Ind/Year FE" "\$N\$" "\$R^2\$")) ///
    mtitles("(1)" "(2)" "(3)" "(4)" "(5)" "(6)" "(7)") ///
    mgroups("PriceDelay" "PriceDelay" "PriceDelay" "PriceDelay" "PriceDelay" "SYNCH" "PriceDelay", ///
        pattern(1 1 1 1 1 1 1)) ///
    nonotes addnotes( ///
        "注：括号内数值为经行业×年份层面聚类调整后的稳健标准误。" ///
        "***、**、*分别表示1\%、5\%、10\%的水平上显著，下同。" ///
        "模型(1)仅控制固定效应不含控制变量。" ///
        "模型(3)解释变量为ln(1+关键词总数)，模型(4)为ln(1+实质利用次数)。" ///
        "模型(5)控制行业和年份固定效应。" ///
        "模型(6)被解释变量为股价同步性SYNCH。" ///
        "模型(7)额外控制金融资产占比。") ///
    title("表2 基准回归检验") ///
    booktabs alignment(S) compress

* 4位小数纯文本输出(供核对)
esttab m1 m2 m3 m4 m5 m6 m7, ///
    b(4) se(4) ///
    star(* 0.10 ** 0.05 *** 0.01) ///
    order(`varorder') ///
    stats(Controls FirmYearFE IndYearFE N r2, ///
        fmt(%s %s %s %12.0fc %9.3f) ///
        labels("Controls" "Firm/Year FE" "Ind/Year FE" "N" "R-squared")) ///
    mtitles("(1)" "(2)" "(3)" "(4)" "(5)" "(6)" "(7)") ///
    title("Table 2: Baseline (4 decimal places for review)")
