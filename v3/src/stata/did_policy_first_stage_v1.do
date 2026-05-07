clear all
set more off
capture log close

global V3 "/Users/mac/computerscience/0做完了/15会计研究/v3"

log using "$V3/results/logs/did_policy_first_stage_v1.log", replace text

display "=== Policy first-stage checks for mismatch X ==="
display "Checks whether policy shocks move narrative-hard mismatch itself."

capture which reghdfe
if _rc {
    display as error "reghdfe not installed"
    exit 199
}

local xoutcomes "NarrHardGap_direct_v1 NarrHardWashing_direct_v1 NarrHardHushing_direct_v1 NarrHardVerified_direct_v1 DU_kw HardCap_direct_v1"
local ctrls "Size Lev ROA TobinQ Age Growth IndepRatio Dual Top1Share SOE CFO"

tempfile allout
tempname posth
postfile `posth' str20 policy str28 treatment str28 y_name str24 spec double coef se t_stat p_value N positives using `allout', replace

* -----------------------------
* 1. 2024 accounting rule first-stage
* -----------------------------

use "$V3/results/data/did_accounting_2024_v1_panel.dta", clear

foreach y of local xoutcomes {
    capture confirm variable `y'
    if _rc continue

    foreach x in acct_gap_post acct_wany_post acct_wmaj_post acct_wstr_post {
        local regctrls ""
        local regmiss ""
        foreach c of local ctrls {
            capture confirm variable `c'
            if !_rc & "`c'" != "`y'" {
                local regctrls "`regctrls' `c'"
                local regmiss "`regmiss', `c'"
            }
        }

        quietly count if inrange(year_num, 2021, 2024) & !missing(`y', `x' `regmiss', Stkcd_num, year_num, IndYear_num)
        if r(N) == 0 {
            post `posth' ("acct2024") ("`x'") ("`y'") ("firm_yearFE") (.) (.) (.) (.) (0) (0)
            continue
        }
        quietly count if inrange(year_num, 2021, 2024) & !missing(`y', `x' `regmiss', Stkcd_num, year_num, IndYear_num) & `x' > 0
        local xpos = r(N)

        capture noisily quietly reghdfe `y' `x' `regctrls' if inrange(year_num, 2021, 2024), absorb(Stkcd_num year_num) cluster(IndYear_num)
        if _rc {
            post `posth' ("acct2024") ("`x'") ("`y'") ("firm_yearFE") (.) (.) (.) (.) (0) (`xpos')
            continue
        }
        capture local coef = _b[`x']
        if _rc {
            post `posth' ("acct2024") ("`x'") ("`y'") ("firm_yearFE") (.) (.) (.) (.) (e(N)) (`xpos')
            continue
        }
        local se = _se[`x']
        local tval = `coef' / `se'
        local pval = 2 * ttail(e(df_r), abs(`tval'))
        post `posth' ("acct2024") ("`x'") ("`y'") ("firm_yearFE") (`coef') (`se') (`tval') (`pval') (e(N)) (`xpos')
    }
}

* -----------------------------
* 2. Data-IP pilot first-stage
* -----------------------------

use "$V3/results/data/did_data_ip_pilot_v1_panel.dta", clear

foreach y of local xoutcomes {
    capture confirm variable `y'
    if _rc continue

    foreach x in dip_gap dip_wany dip_wmaj dip_wstr {
        local regctrls ""
        local regmiss ""
        foreach c of local ctrls {
            capture confirm variable `c'
            if !_rc & "`c'" != "`y'" {
                local regctrls "`regctrls' `c'"
                local regmiss "`regmiss', `c'"
            }
        }

        quietly count if inrange(year_num, 2019, 2024) & !missing(`y', DIPilot, `x' `regmiss', Stkcd_num, year_num, Prov_num)
        if r(N) == 0 {
            post `posth' ("dataip") ("`x'") ("`y'") ("firm_yearFE_provclu") (.) (.) (.) (.) (0) (0)
            continue
        }
        quietly count if inrange(year_num, 2019, 2024) & !missing(`y', DIPilot, `x' `regmiss', Stkcd_num, year_num, Prov_num) & `x' > 0
        local xpos = r(N)

        capture noisily quietly reghdfe `y' DIPilot `x' `regctrls' if inrange(year_num, 2019, 2024), absorb(Stkcd_num year_num) cluster(Prov_num)
        if _rc {
            post `posth' ("dataip") ("`x'") ("`y'") ("firm_yearFE_provclu") (.) (.) (.) (.) (0) (`xpos')
            continue
        }
        capture local coef = _b[`x']
        if _rc {
            post `posth' ("dataip") ("`x'") ("`y'") ("firm_yearFE_provclu") (.) (.) (.) (.) (e(N)) (`xpos')
            continue
        }
        local se = _se[`x']
        local tval = `coef' / `se'
        local pval = 2 * ttail(e(df_r), abs(`tval'))
        post `posth' ("dataip") ("`x'") ("`y'") ("firm_yearFE_provclu") (`coef') (`se') (`tval') (`pval') (e(N)) (`xpos')
    }
}

postclose `posth'
use `allout', clear
gen sig_10 = abs(t_stat) >= 1.645 if !missing(t_stat)
gen sig_05 = abs(t_stat) >= 1.96 if !missing(t_stat)
gen sig_01 = abs(t_stat) >= 2.576 if !missing(t_stat)
order policy treatment y_name spec coef se t_stat p_value N positives sig_10 sig_05 sig_01
sort policy treatment y_name
export delimited using "$V3/results/stata/did_policy_first_stage_v1.csv", replace
save "$V3/results/stata/did_policy_first_stage_v1.dta", replace

display _newline "=== First-stage highlights: |t| >= 1.96 ==="
list policy treatment y_name coef t_stat p_value N positives if sig_05 == 1, sepby(policy treatment) abbreviate(28)

log close
