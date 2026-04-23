* Descriptive statistics on Stata sample (consistent with OLS/IV)
clear all
use "data_stata/reg_sample_iv_v16.dta", clear

local vars PriceDelay SYNCH DU_kw DU_llm Size Lev ROA TobinQ Age Growth IndepRatio Dual Top1Share SOE CFO

foreach v of local vars {
    cap quietly sum `v', detail
    if _rc == 0 {
        di "`v': N=" r(N) " mean=" %9.4f r(mean) " sd=" %9.4f r(sd) " min=" %9.4f r(min) " p50=" %9.4f r(p50) " max=" %9.4f r(max)
    }
}

* LLM score distribution
tab llm_score if llm_score != .
