clear all
set more off

local base "/Users/mac/computerscience/0做完了/15会计研究"
cap mkdir "`base'/results/v16_quality"
cap mkdir "`base'/results/v16_quality/logs"

capture log close _all
log using "`base'/results/v16_quality/logs/run_quality_interactions.log", replace text

capture which reghdfe
if _rc {
    display as error "reghdfe not installed"
    exit 198
}

use "`base'/data_stata/reg_sample_v16_quality.dta", clear

local controls "Size Lev ROA TobinQ Age Growth IndepRatio Dual Top1Share SOE CFO"

tempfile kw_out llm_out
tempname kw_hold llm_hold

postfile `kw_hold' str20 quality str12 treatment double(coef se tstat pval) long(N) double(r2) using `kw_out', replace
scalar max_abs_t_kw = 0

foreach Q in Quality_sub_z Quality_mda_z Quality_subden_z BroadShallow_cont_z WashGap_z {
    capture drop DUxQ
    gen double DUxQ = DU_kw * `Q'
    quietly reghdfe PriceDelay DU_kw `Q' DUxQ `controls', absorb(Stkcd_num year_num) cluster(IndYear_num)
    local b = _b[DUxQ]
    local se = _se[DUxQ]
    local t = `b' / `se'
    local p = 2 * ttail(e(df_r), abs(`t'))
    local n = e(N)
    local r2 = e(r2)
    if abs(`t') > max_abs_t_kw {
        scalar max_abs_t_kw = abs(`t')
    }
    post `kw_hold' ("`Q'") ("DU_kw") (`b') (`se') (`t') (`p') (`n') (`r2')
    display as text "`Q' / DU_kw: coef=" %9.4f `b' " t=" %6.2f `t' " N=" `n'
    drop DUxQ
}
postclose `kw_hold'

use `kw_out', clear
export delimited using "`base'/results/v16_quality/quality_interactions.csv", replace
list, noobs abbreviate(20)

use "`base'/data_stata/reg_sample_v16_quality.dta", clear

postfile `llm_hold' str20 quality str12 treatment double(coef se tstat pval) long(N) double(r2) using `llm_out', replace
scalar max_abs_t_llm = 0

foreach Q in Quality_sub_z Quality_mda_z Quality_subden_z BroadShallow_cont_z WashGap_z {
    capture drop DUxQ
    gen double DUxQ = DU_llm * `Q'
    quietly reghdfe PriceDelay DU_llm `Q' DUxQ `controls', absorb(Stkcd_num year_num) cluster(IndYear_num)
    local b = _b[DUxQ]
    local se = _se[DUxQ]
    local t = `b' / `se'
    local p = 2 * ttail(e(df_r), abs(`t'))
    local n = e(N)
    local r2 = e(r2)
    if abs(`t') > max_abs_t_llm {
        scalar max_abs_t_llm = abs(`t')
    }
    post `llm_hold' ("`Q'") ("DU_llm") (`b') (`se') (`t') (`p') (`n') (`r2')
    display as text "`Q' / DU_llm: coef=" %9.4f `b' " t=" %6.2f `t' " N=" `n'
    drop DUxQ
}
postclose `llm_hold'

use `llm_out', clear
export delimited using "`base'/results/v16_quality/quality_interactions_llm.csv", replace
list, noobs abbreviate(20)

display as text "Max |t| DU_kw block  = " %6.3f scalar(max_abs_t_kw)
display as text "Max |t| DU_llm block = " %6.3f scalar(max_abs_t_llm)

log close
