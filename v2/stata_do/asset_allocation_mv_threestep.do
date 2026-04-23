clear all
set more off
capture log close

log using "/Users/mac/computerscience/0做完了/15会计研究/v2/results/asset_allocation_mv_threestep.log", replace text

display "=== Asset allocation MV three-step run ==="

tempfile survivors
import delimited using "/Users/mac/computerscience/0做完了/15会计研究/v2/results/asset_allocation_mv_step2.csv", clear
keep if inlist(status, "pass", "marginal")
count
local survivors_n = r(N)

if `survivors_n' == 0 {
    display as error "资产配置全部 MV Step 2 fail，机制路线确诊死亡，建议降级"
    log close
    exit 0
}

levelsof mv, local(survivors_list)
save `survivors', replace

use "/Users/mac/computerscience/0做完了/15会计研究/v1/data_stata/reg_sample_asset_mv.dta", clear

tsset Stkcd_num year_num

local ctrls "Size Lev ROA TobinQ Age Growth IndepRatio Dual Top1Share SOE CFO"
local dulist "DU_kw_lag DU_llm_lag"

capture drop DU_kw_lag
capture drop DU_llm_lag
gen DU_kw_lag = L.DU_kw
gen DU_llm_lag = L.DU_llm

foreach mv of local survivors_list {
    capture drop `mv'_lag
    gen `mv'_lag = L.`mv'
}

local all_models

foreach mv of local survivors_list {
    capture log close mvlog
    log using "/Users/mac/computerscience/0做完了/15会计研究/v2/results/asset_allocation_mv_`mv'.log", replace text name(mvlog)
    display "=== Candidate `mv' ==="

    foreach du of local dulist {
        local du_short = cond("`du'" == "DU_kw_lag", "kw", "llm")

        reghdfe `mv' `du' `ctrls', absorb(Stkcd_num year_num) cluster(IndYear_num)
        estimates store s1_`mv'_`du_short'
        local all_models "`all_models' s1_`mv'_`du_short'"

        reghdfe PriceDelay `mv'_lag `ctrls', absorb(Stkcd_num year_num) cluster(IndYear_num)
        estimates store s2_`mv'_`du_short'
        local all_models "`all_models' s2_`mv'_`du_short'"

        reghdfe PriceDelay `du' `mv' `ctrls', absorb(Stkcd_num year_num) cluster(IndYear_num)
        estimates store s3_`mv'_`du_short'
        local all_models "`all_models' s3_`mv'_`du_short'"

        reghdfe PriceDelay `du' `mv'_lag `ctrls', absorb(Stkcd_num year_num) cluster(IndYear_num)
        estimates store s3b_`mv'_`du_short'
        local all_models "`all_models' s3b_`mv'_`du_short'"
    }

    log close mvlog
}

esttab `all_models' using "/Users/mac/computerscience/0做完了/15会计研究/v2/results/asset_allocation_mv_threestep.csv", ///
    replace csv se r2 star(* 0.10 ** 0.05 *** 0.01) nogaps compress ///
    title("Asset allocation MV three-step diagnostics")

log close
