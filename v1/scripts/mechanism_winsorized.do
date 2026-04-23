* =============================================================
* 机制渠道：Winsorize机制变量后重跑
* =============================================================
clear all
set more off
use "data_stata/reg_sample_iv_v16.dta", clear

global controls Size Lev ROA TobinQ Age Growth IndepRatio Dual Top1Share SOE CFO

* Winsorize mechanism variables at 1%/99%
foreach v in Analyst Amihud RetVol Turnover InstHold {
    quietly sum `v', detail
    local p1 = r(p1)
    local p99 = r(p99)
    gen `v'_w = `v'
    replace `v'_w = `p1' if `v' < `p1' & `v' != .
    replace `v'_w = `p99' if `v' > `p99' & `v' != .
    quietly sum `v'_w, detail
    di "`v'_w: mean=" %9.4f r(mean) " sd=" %9.4f r(sd) " min=" %9.4f r(min) " max=" %9.4f r(max)
}

di ""
di "============================================="
di "Winsorized机制回归"
di "============================================="

foreach dv in Analyst_w Amihud_w RetVol_w Turnover_w InstHold_w {
    foreach du in DU_kw DU_llm {
        cap reghdfe `dv' `du' $controls, absorb(Stkcd_num year_num) cluster(IndYear_num)
        if _rc == 0 {
            di "`dv' ~ `du': b=" %9.4f _b[`du'] " se=" %9.4f _se[`du'] " t=" %6.2f (_b[`du']/_se[`du']) " N=" e(N)
        }
    }
}

di ""
di "============================================="
di "对比：未Winsorize的原始结果"
di "============================================="

foreach dv in Analyst Amihud RetVol Turnover InstHold {
    foreach du in DU_kw DU_llm {
        cap reghdfe `dv' `du' $controls, absorb(Stkcd_num year_num) cluster(IndYear_num)
        if _rc == 0 {
            di "`dv' ~ `du': b=" %9.4f _b[`du'] " se=" %9.4f _se[`du'] " t=" %6.2f (_b[`du']/_se[`du']) " N=" e(N)
        }
    }
}
