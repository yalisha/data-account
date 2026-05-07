clear all
set more off
capture log close

global V3 "/Users/mac/computerscience/0做完了/15会计研究/v3"

log using "$V3/results/logs/did_policy_first_stage_event_v1.log", replace text

display "=== Dynamic first-stage checks for policy shocks and mismatch X ==="

capture which reghdfe
if _rc {
    display as error "reghdfe not installed"
    exit 199
}

local xoutcomes "NarrHardGap_direct_v1 NarrHardWashing_direct_v1 NarrHardHushing_direct_v1 NarrHardVerified_direct_v1 DU_kw HardCap_direct_v1"
local ctrls "Size Lev ROA TobinQ Age Growth IndepRatio Dual Top1Share SOE CFO"

tempfile evout
tempname evpost
postfile `evpost' str20 policy str20 exposure str28 y_name int event_year double coef se t_stat p_value N using `evout', replace

* -----------------------------
* 1. Accounting 2024: 2023 omitted
* -----------------------------

use "$V3/results/data/did_accounting_2024_v1_panel.dta", clear

foreach y of local xoutcomes {
    capture confirm variable `y'
    if _rc continue

    local regctrls ""
    foreach c of local ctrls {
        capture confirm variable `c'
        if !_rc & "`c'" != "`y'" {
            local regctrls "`regctrls' `c'"
        }
    }

    foreach block in "gap egap" "wany ewany" "wmaj ewmaj" "wstr ewstr" {
        tokenize "`block'"
        local ename "`1'"
        local pref "`2'"

        capture noisily quietly reghdfe `y' `pref'_2021 `pref'_2022 `pref'_2024 `regctrls' if inrange(year_num, 2021, 2024), absorb(Stkcd_num year_num) cluster(IndYear_num)
        if _rc {
            foreach yy in 2021 2022 2024 {
                post `evpost' ("acct2024") ("`ename'") ("`y'") (`yy') (.) (.) (.) (.) (0)
            }
            continue
        }
        foreach yy in 2021 2022 2024 {
            capture local coef = _b[`pref'_`yy']
            if _rc {
                post `evpost' ("acct2024") ("`ename'") ("`y'") (`yy') (.) (.) (.) (.) (e(N))
                continue
            }
            local se = _se[`pref'_`yy']
            local tval = `coef' / `se'
            local pval = 2 * ttail(e(df_r), abs(`tval'))
            post `evpost' ("acct2024") ("`ename'") ("`y'") (`yy') (`coef') (`se') (`tval') (`pval') (e(N))
        }
    }
}

* -----------------------------
* 2. Data-IP first pilots: 2022 omitted
* -----------------------------

use "$V3/results/data/did_data_ip_pilot_v1_panel.dta", clear
local fdlist "fd_2019 fd_2020 fd_2021 fd_2023 fd_2024"

foreach y of local xoutcomes {
    capture confirm variable `y'
    if _rc continue

    local regctrls ""
    foreach c of local ctrls {
        capture confirm variable `c'
        if !_rc & "`c'" != "`y'" {
            local regctrls "`regctrls' `c'"
        }
    }

    foreach block in "gap fgap" "wany fwany" "wmaj fwmaj" "wstr fwstr" {
        tokenize "`block'"
        local ename "`1'"
        local pref "`2'"

        capture noisily quietly reghdfe `y' `fdlist' `pref'_2019 `pref'_2020 `pref'_2021 `pref'_2023 `pref'_2024 `regctrls' if inrange(year_num, 2019, 2024), absorb(Stkcd_num year_num) cluster(Prov_num)
        if _rc {
            foreach yy in 2019 2020 2021 2023 2024 {
                post `evpost' ("dataip") ("`ename'") ("`y'") (`yy') (.) (.) (.) (.) (0)
            }
            continue
        }
        foreach yy in 2019 2020 2021 2023 2024 {
            capture local coef = _b[`pref'_`yy']
            if _rc {
                post `evpost' ("dataip") ("`ename'") ("`y'") (`yy') (.) (.) (.) (.) (e(N))
                continue
            }
            local se = _se[`pref'_`yy']
            local tval = `coef' / `se'
            local pval = 2 * ttail(e(df_r), abs(`tval'))
            post `evpost' ("dataip") ("`ename'") ("`y'") (`yy') (`coef') (`se') (`tval') (`pval') (e(N))
        }
    }
}

postclose `evpost'
use `evout', clear
gen sig_10 = abs(t_stat) >= 1.645 if !missing(t_stat)
gen sig_05 = abs(t_stat) >= 1.96 if !missing(t_stat)
gen sig_01 = abs(t_stat) >= 2.576 if !missing(t_stat)
order policy exposure y_name event_year coef se t_stat p_value N sig_10 sig_05 sig_01
sort policy exposure y_name event_year
export delimited using "$V3/results/stata/did_policy_first_stage_event_v1.csv", replace
save "$V3/results/stata/did_policy_first_stage_event_v1.dta", replace

display _newline "=== Dynamic first-stage highlights: |t| >= 1.96 ==="
list policy exposure y_name event_year coef t_stat p_value N if sig_05 == 1, sepby(policy exposure y_name) abbreviate(28)

log close
