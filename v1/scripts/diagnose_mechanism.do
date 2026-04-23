* =============================================================
* 诊断机制变量：分布、组内变异、异常值
* =============================================================
clear all
set more off
use "data_stata/reg_sample_iv_v16.dta", clear

* 1. 基本描述统计
di "============================================="
di "机制变量描述统计"
di "============================================="
foreach v in Analyst Amihud RetVol Turnover InstHold {
    quietly sum `v', detail
    di "`v': N=" r(N) " mean=" %9.4f r(mean) " sd=" %9.4f r(sd) " p1=" %9.4f r(p1) " p99=" %9.4f r(p99) " min=" %9.4f r(min) " max=" %9.4f r(max)
}

* 2. Analyst到底是什么？原始计数还是log？
di ""
di "Analyst: 看看是否已经取了log"
sum Analyst if Analyst > 0, detail
* 如果mean~5-6且有小数，说明是log。如果mean~10-20且为整数，是原始计数

* 3. 组内变异（firm FE absorbs between-firm variation）
di ""
di "============================================="
di "组内变异 (within-firm std dev)"
di "============================================="
foreach v in Analyst Amihud RetVol Turnover InstHold DU_kw DU_llm {
    quietly {
        bysort Stkcd_num: egen `v'_mean = mean(`v')
        gen `v'_within = `v' - `v'_mean
        sum `v'_within
    }
    di "`v': within_sd=" %9.4f r(sd) " overall_sd=" %9.4f .
    quietly sum `v'
    di "  overall_sd=" %9.4f r(sd) " ratio=" %5.2f (r(sd) > 0 ? . : .)
    quietly {
        sum `v'_within
        local wsd = r(sd)
        sum `v'
        local osd = r(sd)
    }
    di "  within/overall=" %5.3f (`wsd'/`osd')
    drop `v'_mean `v'_within
}

* 4. Turnover极端值检查
di ""
di "============================================="
di "Turnover分布详情"
di "============================================="
sum Turnover, detail

* 5. 检查：是否需要Winsorize机制变量？
di ""
di "Turnover p1/p99:"
quietly sum Turnover, detail
di "  p1=" r(p1) " p99=" r(p99) " min=" r(min) " max=" r(max)

* 6. Winsorize后重跑Turnover
quietly {
    sum Turnover, detail
    local p1 = r(p1)
    local p99 = r(p99)
    gen Turnover_w = Turnover
    replace Turnover_w = `p1' if Turnover < `p1' & Turnover != .
    replace Turnover_w = `p99' if Turnover > `p99' & Turnover != .
}
di ""
di "Winsorized Turnover机制回归:"
global controls Size Lev ROA TobinQ Age Growth IndepRatio Dual Top1Share SOE CFO

reghdfe Turnover_w DU_kw $controls, absorb(Stkcd_num year_num) cluster(IndYear_num)
di "Turnover_w ~ DU_kw: b=" %9.4f _b[DU_kw] " t=" %6.2f (_b[DU_kw]/_se[DU_kw])

reghdfe Turnover_w DU_llm $controls, absorb(Stkcd_num year_num) cluster(IndYear_num)
di "Turnover_w ~ DU_llm: b=" %9.4f _b[DU_llm] " t=" %6.2f (_b[DU_llm]/_se[DU_llm])

* 7. 同样Winsorize Amihud和RetVol
foreach v in Amihud RetVol {
    quietly {
        sum `v', detail
        local p1 = r(p1)
        local p99 = r(p99)
        gen `v'_w = `v'
        replace `v'_w = `p1' if `v' < `p1' & `v' != .
        replace `v'_w = `p99' if `v' > `p99' & `v' != .
    }
    reghdfe `v'_w DU_kw $controls, absorb(Stkcd_num year_num) cluster(IndYear_num)
    di "`v'_w ~ DU_kw: b=" %9.4f _b[DU_kw] " t=" %6.2f (_b[DU_kw]/_se[DU_kw])

    reghdfe `v'_w DU_llm $controls, absorb(Stkcd_num year_num) cluster(IndYear_num)
    di "`v'_w ~ DU_llm: b=" %9.4f _b[DU_llm] " t=" %6.2f (_b[DU_llm]/_se[DU_llm])
}
