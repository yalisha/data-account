* ============================================================
* v16 分析师覆盖异质性 + Analyst变量已merge
* ============================================================

clear all
set more off

use "/Users/mac/computerscience/15会计研究/data_stata/reg_sample_iv_v16.dta", clear

local controls Size Lev ROA TobinQ Age Growth IndepRatio Dual Top1Share SOE CFO

drop if missing(Ind2_num)
drop if missing(ProvDigital2016)

xtset Stkcd_num year

di "=========================================="
di "ANALYST COVERAGE HETEROGENEITY"
di "=========================================="

summ Analyst, detail
local ana_med = r(p50)
di "Analyst中位数: `ana_med'"

gen Analyst_high = (Analyst >= `ana_med') if !missing(Analyst)
tab Analyst_high

* 分组回归
di _n "--- 高分析师覆盖 ---"
reghdfe PriceDelay DU_kw `controls' if Analyst_high == 1, absorb(Stkcd_num year_num) vce(cluster IndYear_num)
local b_ana_h = _b[DU_kw]
local t_ana_h = _b[DU_kw]/_se[DU_kw]
di "高覆盖: β=`b_ana_h', t=`t_ana_h'"

di _n "--- 低分析师覆盖 ---"
reghdfe PriceDelay DU_kw `controls' if Analyst_high == 0, absorb(Stkcd_num year_num) vce(cluster IndYear_num)
local b_ana_l = _b[DU_kw]
local t_ana_l = _b[DU_kw]/_se[DU_kw]
di "低覆盖: β=`b_ana_l', t=`t_ana_l'"

* 交互项检验
di _n "--- 交互项检验 ---"
reghdfe PriceDelay c.DU_kw##i.Analyst_high `controls', absorb(Stkcd_num year_num) vce(cluster IndYear_num)
local ana_inter_b = _b[1.Analyst_high#c.DU_kw]
local ana_inter_t = _b[1.Analyst_high#c.DU_kw]/_se[1.Analyst_high#c.DU_kw]
local ana_inter_p = 2*ttail(e(df_r), abs(`ana_inter_t'))
di "交互项: β=`ana_inter_b', t=`ana_inter_t', p=`ana_inter_p'"

* 同时测一下三分位
di _n "--- 三分位分组 ---"
xtile Analyst_tercile = Analyst, nq(3)
tab Analyst_tercile

forvalues t = 1/3 {
    reghdfe PriceDelay DU_kw `controls' if Analyst_tercile == `t', absorb(Stkcd_num year_num) vce(cluster IndYear_num)
    di "Tercile `t': β=" _b[DU_kw] ", t=" _b[DU_kw]/_se[DU_kw] ", N=" e(N)
}

di _n "=========================================="
di "DONE"
di "=========================================="
