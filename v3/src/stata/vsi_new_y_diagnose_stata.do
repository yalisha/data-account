clear all
set more off
capture log close

log using "/Users/mac/computerscience/0做完了/15会计研究/v3/results/vsi_new_y_diagnose_stata.log", replace text

display "=== VSI new Y diagnose: Stata MCP / reghdfe ==="
display "Input: v3/results/reg_sample_vsi_new_y.dta"
display "Spec: Y_t = L.DU + controls + firm FE + year FE, cluster(IndYear_num)"

capture which reghdfe
if _rc {
    display as error "reghdfe not installed"
    exit 199
}

use "/Users/mac/computerscience/0做完了/15会计研究/v3/results/reg_sample_vsi_new_y.dta", clear

sort Stkcd_num year_num
tsset Stkcd_num year_num

local ctrls "Size Lev ROA TobinQ Age Growth IndepRatio Dual Top1Share SOE CFO"
local ylist "AbsorbRatio60 AbsorbLogRatio60 AbsorbShare60 ICOE_PEG ICOE_EY PEAD60_abs CS_Spread PriceDelay"

foreach du in DU_kw DU_llm {
    capture drop `du'_lag
    gen `du'_lag = L.`du'
}

tempfile step1
tempname posth
postfile `posth' str28 y_name str8 du_measure str12 expected_sign ///
    double coef se t_stat p_value N str12 status ///
    using `step1', replace

foreach y of local ylist {
    foreach du in DU_kw DU_llm {
        local dulag = "`du'_lag"

        capture confirm variable `y'
        if _rc {
            post `posth' ("`y'") ("`du'") ("negative") (.) (.) (.) (.) (0) ("unavailable")
            continue
        }

        quietly count if !missing(`y', `dulag', Size, Lev, ROA, TobinQ, Age, Growth, IndepRatio, Dual, Top1Share, SOE, CFO, Stkcd_num, year_num, IndYear_num)
        if r(N) == 0 {
            post `posth' ("`y'") ("`du'") ("negative") (.) (.) (.) (.) (0) ("unavailable")
            continue
        }

        capture noisily quietly reghdfe `y' `dulag' `ctrls', absorb(Stkcd_num year_num) cluster(IndYear_num)
        if _rc {
            post `posth' ("`y'") ("`du'") ("negative") (.) (.) (.) (.) (0) ("unavailable")
            continue
        }

        local coef = _b[`dulag']
        local se = _se[`dulag']
        local tval = `coef' / `se'
        local pval = 2 * ttail(e(df_r), abs(`tval'))

        local status = "fail"
        if `coef' < 0 & abs(`tval') >= 1.96 local status = "pass"
        else if `coef' < 0 & abs(`tval') >= 1.50 local status = "marginal"

        post `posth' ("`y'") ("`du'") ("negative") (`coef') (`se') (`tval') (`pval') (e(N)) ("`status'")
    }
}

postclose `posth'

use `step1', clear
order y_name du_measure expected_sign coef se t_stat p_value N status
export delimited using "/Users/mac/computerscience/0做完了/15会计研究/v3/results/vsi_new_y_reg_results_stata.csv", replace
save "/Users/mac/computerscience/0做完了/15会计研究/v3/results/vsi_new_y_reg_results_stata.dta", replace

display _newline "=== Status summary ==="
tab status

log close
