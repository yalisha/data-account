clear all
set more off
capture log close

global V3 "/Users/mac/computerscience/0做完了/15会计研究/v3"
global V1 "/Users/mac/computerscience/0做完了/15会计研究/v1"

log using "$V3/results/logs/mismatch_x_y_screen_v0.log", replace text

display "=== v0 data mismatch X screen ==="
display "Input: v1/data_stata/reg_sample_v18.dta"
display "Spec: Y_t = L.X + controls + firm FE + year FE, cluster(IndYear_num)"

capture which reghdfe
if _rc {
    display as error "reghdfe not installed"
    exit 199
}

use "$V1/data_stata/reg_sample_v18.dta", clear

sort Stkcd_num year_num
tsset Stkcd_num year_num

* -----------------------------
* 1. Construct mismatch Xs
* -----------------------------

* Disclosure side: annual-report data narrative intensity.
capture drop n_year n_indyear
bys year_num: egen n_year = count(DU_kw)
bys IndYear_num: egen n_indyear = count(DU_kw)

capture drop rank_disc_y rank_sub_y rank_disc_iy rank_sub_iy
bys year_num: egen rank_disc_y = rank(DU_kw) if !missing(DU_kw), field
bys year_num: egen rank_sub_y = rank(DU_sub_ln) if !missing(DU_sub_ln), field
bys IndYear_num: egen rank_disc_iy = rank(DU_kw) if !missing(DU_kw), field
bys IndYear_num: egen rank_sub_iy = rank(DU_sub_ln) if !missing(DU_sub_ln), field

capture drop disc_pct_y sub_pct_y disc_pct_iy sub_pct_iy
gen disc_pct_y = (rank_disc_y - 1) / (n_year - 1) if n_year > 1
gen sub_pct_y = (rank_sub_y - 1) / (n_year - 1) if n_year > 1
gen disc_pct_iy = (rank_disc_iy - 1) / (n_indyear - 1) if n_indyear > 1
gen sub_pct_iy = (rank_sub_iy - 1) / (n_indyear - 1) if n_indyear > 1

capture drop Mismatch_sub Mismatch_sub_iy DataWashing_sub DataHushing_sub DataWashing_sub_strict DataHushing_sub_strict
gen Mismatch_sub = disc_pct_y - sub_pct_y
gen Mismatch_sub_iy = disc_pct_iy - sub_pct_iy
gen DataWashing_sub = (disc_pct_y >= .50 & sub_pct_y < .50) if !missing(disc_pct_y, sub_pct_y)
gen DataHushing_sub = (sub_pct_y >= .50 & disc_pct_y < .50) if !missing(disc_pct_y, sub_pct_y)
gen DataWashing_sub_strict = (disc_pct_y >= .75 & sub_pct_y <= .25) if !missing(disc_pct_y, sub_pct_y)
gen DataHushing_sub_strict = (sub_pct_y >= .75 & disc_pct_y <= .25) if !missing(disc_pct_y, sub_pct_y)

* Coarse ex ante data-potential side.
* This is NOT a final true-capability measure. It proxies data potential with industry/location conditions.
capture drop lnLQ z_lnLQ DataPotentialRaw rank_pot_y pot_pct_y Mismatch_potential
gen lnLQ = ln(1 + LQ) if LQ >= 0
egen z_lnLQ = std(lnLQ)
gen DataPotentialRaw = HighTech + DigEconCore + StrategicEmerging + IndustryCluster + z_lnLQ
bys year_num: egen rank_pot_y = rank(DataPotentialRaw) if !missing(DataPotentialRaw), field
gen pot_pct_y = (rank_pot_y - 1) / (n_year - 1) if n_year > 1
gen Mismatch_potential = disc_pct_y - pot_pct_y

capture drop DataWashing_pot DataHushing_pot VerifiedData_pot DataWashing_pot_strict DataHushing_pot_strict VerifiedData_pot_strict
gen DataWashing_pot = (disc_pct_y >= .50 & pot_pct_y < .50) if !missing(disc_pct_y, pot_pct_y)
gen DataHushing_pot = (pot_pct_y >= .50 & disc_pct_y < .50) if !missing(disc_pct_y, pot_pct_y)
gen VerifiedData_pot = (pot_pct_y >= .50 & disc_pct_y >= .50) if !missing(disc_pct_y, pot_pct_y)
gen DataWashing_pot_strict = (disc_pct_y >= .75 & pot_pct_y <= .25) if !missing(disc_pct_y, pot_pct_y)
gen DataHushing_pot_strict = (pot_pct_y >= .75 & disc_pct_y <= .25) if !missing(disc_pct_y, pot_pct_y)
gen VerifiedData_pot_strict = (pot_pct_y >= .75 & disc_pct_y >= .75) if !missing(disc_pct_y, pot_pct_y)

label variable Mismatch_sub "Disclosure rank - substantive-expression rank"
label variable Mismatch_sub_iy "Disclosure rank - substantive-expression rank, industry-year"
label variable DataWashing_sub "High disclosure, low substantive expression"
label variable DataHushing_sub "High substantive expression, low disclosure"
label variable Mismatch_potential "Disclosure rank - data-potential rank"
label variable DataWashing_pot "High disclosure, low data potential"
label variable DataHushing_pot "High data potential, low disclosure"
label variable VerifiedData_pot "High disclosure, high data potential"

save "$V3/results/data/mismatch_x_panel_v0.dta", replace

* Summary table for X construction.
tempfile xsummary
tempname xpost
postfile `xpost' str32 x_name double N mean sd p25 median p75 using `xsummary', replace

foreach x in Mismatch_sub Mismatch_sub_iy DataWashing_sub DataHushing_sub DataWashing_sub_strict DataHushing_sub_strict Mismatch_potential DataWashing_pot DataHushing_pot VerifiedData_pot DataWashing_pot_strict DataHushing_pot_strict VerifiedData_pot_strict {
    quietly summarize `x', detail
    post `xpost' ("`x'") (r(N)) (r(mean)) (r(sd)) (r(p25)) (r(p50)) (r(p75))
}
postclose `xpost'
use `xsummary', clear
export delimited using "$V3/results/stata/mismatch_x_summary_v0.csv", replace

* Restore full panel for regressions.
use "$V3/results/data/mismatch_x_panel_v0.dta", clear
sort Stkcd_num year_num
tsset Stkcd_num year_num

* -----------------------------
* 2. Screen possible Ys
* -----------------------------

local ctrls "Size Lev ROA TobinQ Age Growth IndepRatio Dual Top1Share SOE CFO"
local xlist "Mismatch_sub DataWashing_sub DataHushing_sub DataWashing_sub_strict DataHushing_sub_strict Mismatch_potential DataWashing_pot DataHushing_pot VerifiedData_pot DataWashing_pot_strict DataHushing_pot_strict VerifiedData_pot_strict"
local ylist "ForecastDisp FcstAcc Analyst ReportFreq RatingDisp TobinQ SA InstHold InstStable AuditFee absDA InvestIneff CashFlowVol SCConc SuppConc TFP"

tempfile out
tempname posth
postfile `posth' str32 x_name str24 y_name str20 y_family double coef se t_stat p_value N using `out', replace

foreach y of local ylist {
    local family "other"
    if inlist("`y'", "ForecastDisp", "FcstAcc", "Analyst", "ReportFreq", "RatingDisp") local family "analyst_info"
    if inlist("`y'", "TobinQ") local family "valuation"
    if inlist("`y'", "SA") local family "financing_constraint"
    if inlist("`y'", "InstHold", "InstStable") local family "investor_attention"
    if inlist("`y'", "AuditFee", "absDA") local family "audit_accounting"
    if inlist("`y'", "InvestIneff") local family "capital_allocation"
    if inlist("`y'", "CashFlowVol") local family "operating_resilience"
    if inlist("`y'", "SCConc", "SuppConc") local family "supply_chain"
    if inlist("`y'", "TFP") local family "productivity"

    capture confirm variable `y'
    if _rc continue

    foreach x of local xlist {
        capture confirm variable `x'
        if _rc continue
        capture drop x_lag
        gen x_lag = L.`x'

        local regctrls ""
        local regmiss ""
        foreach c of local ctrls {
            if "`c'" != "`y'" {
                local regctrls "`regctrls' `c'"
                local regmiss "`regmiss', `c'"
            }
        }

        quietly count if !missing(`y', x_lag `regmiss', Stkcd_num, year_num, IndYear_num)
        if r(N) == 0 {
            post `posth' ("`x'") ("`y'") ("`family'") (.) (.) (.) (.) (0)
            continue
        }

        capture noisily quietly reghdfe `y' x_lag `regctrls', absorb(Stkcd_num year_num) cluster(IndYear_num)
        if _rc {
            post `posth' ("`x'") ("`y'") ("`family'") (.) (.) (.) (.) (0)
            continue
        }

        local coef = _b[x_lag]
        local se = _se[x_lag]
        local tval = `coef' / `se'
        local pval = 2 * ttail(e(df_r), abs(`tval'))
        post `posth' ("`x'") ("`y'") ("`family'") (`coef') (`se') (`tval') (`pval') (e(N))
    }
}

postclose `posth'
use `out', clear
gen sig_10 = abs(t_stat) >= 1.645 if !missing(t_stat)
gen sig_05 = abs(t_stat) >= 1.96 if !missing(t_stat)
gen sig_01 = abs(t_stat) >= 2.576 if !missing(t_stat)
order x_name y_family y_name coef se t_stat p_value N sig_10 sig_05 sig_01
sort x_name y_family y_name
export delimited using "$V3/results/stata/mismatch_x_y_screen_v0.csv", replace
save "$V3/results/stata/mismatch_x_y_screen_v0.dta", replace

display _newline "=== X summary ==="
use "$V3/results/data/mismatch_x_panel_v0.dta", clear
summarize Mismatch_sub Mismatch_potential DataWashing_sub DataHushing_sub DataWashing_pot DataHushing_pot VerifiedData_pot

display _newline "=== Screen highlights: |t| >= 1.96 ==="
use "$V3/results/stata/mismatch_x_y_screen_v0.dta", clear
list x_name y_family y_name coef t_stat p_value N if sig_05 == 1, sepby(x_name) abbreviate(24)

log close
