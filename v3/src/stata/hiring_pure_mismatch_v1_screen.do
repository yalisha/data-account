clear all
set more off
capture log close

global V3 "/Users/mac/computerscience/0做完了/15会计研究/v3"

log using "$V3/results/logs/hiring_pure_mismatch_v1_screen.log", replace text

display "=== Pure hiring-based narrative mismatch v1 screen ==="
display "Input: v3/results/data/hardcap_hiring_custom_v1_panel.dta"
display "Spec: Y_t = L.X + controls + firm FE + year FE, cluster(IndYear_num)"
display "Sample note: pure hiring measures start in 2014, so lag-X regressions mainly use 2015+ observations."

capture which reghdfe
if _rc {
    display as error "reghdfe not installed"
    exit 199
}

use "$V3/results/data/hardcap_hiring_custom_v1_panel.dta", clear
sort Stkcd_num year_num
tsset Stkcd_num year_num

* -----------------------------
* 1. Pure hiring capability scores
* -----------------------------

foreach v in DataHireBroadShare DataHireTitleShare DataHireTitlePostsLn AIHireShare AIHirePostsLn DataEngHireShare DataEngHirePostsLn DataSalaryAvg {
    capture drop zh_`v'
    capture confirm variable `v'
    if !_rc {
        egen zh_`v' = std(`v') if inrange(year_num, 2014, 2024)
    }
}

capture drop HireCap_title HireCap_engineer HireCap_posts HireCap_broad
gen HireCap_title = zh_DataHireTitleShare if inrange(year_num, 2014, 2024)
egen HireCap_engineer = rowmean(zh_DataEngHireShare zh_AIHireShare) if inrange(year_num, 2014, 2024)
egen HireCap_posts = rowmean(zh_DataHireTitlePostsLn zh_DataEngHirePostsLn zh_AIHirePostsLn) if inrange(year_num, 2014, 2024)
gen HireCap_broad = zh_DataHireBroadShare if inrange(year_num, 2014, 2024)

label variable HireCap_title "Pure hiring capability: data-related job title share"
label variable HireCap_engineer "Pure hiring capability: data-engineer and AI hiring shares"
label variable HireCap_posts "Pure hiring capability: data/AI hiring post counts"
label variable HireCap_broad "Pure hiring capability: broad text-based data hiring share"

* Disclosure percentile.
capture drop neg_DU_kw_h n_disc_h rank_disc_h disc_pct_h
gen neg_DU_kw_h = -DU_kw if inrange(year_num, 2014, 2024)
bys year_num: egen n_disc_h = count(DU_kw) if inrange(year_num, 2014, 2024)
bys year_num: egen rank_disc_h = rank(neg_DU_kw_h) if !missing(neg_DU_kw_h), field
gen disc_pct_h = (rank_disc_h - 1) / (n_disc_h - 1) if n_disc_h > 1

foreach h in HireCap_title HireCap_engineer HireCap_posts HireCap_broad {
    capture drop neg_`h' n_`h' rank_`h' pct_`h'
    gen neg_`h' = -`h' if inrange(year_num, 2014, 2024)
    bys year_num: egen n_`h' = count(`h') if inrange(year_num, 2014, 2024)
    bys year_num: egen rank_`h' = rank(neg_`h') if !missing(neg_`h'), field
    gen pct_`h' = (rank_`h' - 1) / (n_`h' - 1) if n_`h' > 1
}

capture drop NarrHireGap_title NarrHireWash_title NarrHireHush_title NarrHireVer_title
capture drop NHireWash_title_s NHireHush_title_s NHireVer_title_s
gen NarrHireGap_title = disc_pct_h - pct_HireCap_title
gen NarrHireWash_title = (disc_pct_h >= .50 & pct_HireCap_title < .50) if !missing(disc_pct_h, pct_HireCap_title)
gen NarrHireHush_title = (pct_HireCap_title >= .50 & disc_pct_h < .50) if !missing(disc_pct_h, pct_HireCap_title)
gen NarrHireVer_title = (disc_pct_h >= .50 & pct_HireCap_title >= .50) if !missing(disc_pct_h, pct_HireCap_title)
gen NHireWash_title_s = (disc_pct_h >= .75 & pct_HireCap_title <= .25) if !missing(disc_pct_h, pct_HireCap_title)
gen NHireHush_title_s = (pct_HireCap_title >= .75 & disc_pct_h <= .25) if !missing(disc_pct_h, pct_HireCap_title)
gen NHireVer_title_s = (disc_pct_h >= .75 & pct_HireCap_title >= .75) if !missing(disc_pct_h, pct_HireCap_title)

capture drop NarrHireGap_engineer NarrHireWash_engineer NarrHireHush_engineer NarrHireVer_engineer
gen NarrHireGap_engineer = disc_pct_h - pct_HireCap_engineer
gen NarrHireWash_engineer = (disc_pct_h >= .50 & pct_HireCap_engineer < .50) if !missing(disc_pct_h, pct_HireCap_engineer)
gen NarrHireHush_engineer = (pct_HireCap_engineer >= .50 & disc_pct_h < .50) if !missing(disc_pct_h, pct_HireCap_engineer)
gen NarrHireVer_engineer = (disc_pct_h >= .50 & pct_HireCap_engineer >= .50) if !missing(disc_pct_h, pct_HireCap_engineer)

capture drop NarrHireGap_posts NarrHireWash_posts NarrHireHush_posts NarrHireVer_posts
gen NarrHireGap_posts = disc_pct_h - pct_HireCap_posts
gen NarrHireWash_posts = (disc_pct_h >= .50 & pct_HireCap_posts < .50) if !missing(disc_pct_h, pct_HireCap_posts)
gen NarrHireHush_posts = (pct_HireCap_posts >= .50 & disc_pct_h < .50) if !missing(disc_pct_h, pct_HireCap_posts)
gen NarrHireVer_posts = (disc_pct_h >= .50 & pct_HireCap_posts >= .50) if !missing(disc_pct_h, pct_HireCap_posts)

capture drop NarrHireGap_broad NarrHireWash_broad NarrHireHush_broad NarrHireVer_broad
gen NarrHireGap_broad = disc_pct_h - pct_HireCap_broad
gen NarrHireWash_broad = (disc_pct_h >= .50 & pct_HireCap_broad < .50) if !missing(disc_pct_h, pct_HireCap_broad)
gen NarrHireHush_broad = (pct_HireCap_broad >= .50 & disc_pct_h < .50) if !missing(disc_pct_h, pct_HireCap_broad)
gen NarrHireVer_broad = (disc_pct_h >= .50 & pct_HireCap_broad >= .50) if !missing(disc_pct_h, pct_HireCap_broad)

save "$V3/results/data/hiring_pure_mismatch_v1_panel.dta", replace

* -----------------------------
* 2. X summary
* -----------------------------

tempfile xsummary
tempname xpost
postfile `xpost' str36 x_name double N mean sd p25 median p75 positives using `xsummary', replace

foreach x in HireCap_title HireCap_engineer HireCap_posts HireCap_broad NarrHireGap_title NarrHireWash_title NarrHireHush_title NarrHireVer_title NHireWash_title_s NHireHush_title_s NHireVer_title_s NarrHireGap_engineer NarrHireWash_engineer NarrHireHush_engineer NarrHireVer_engineer NarrHireGap_posts NarrHireWash_posts NarrHireHush_posts NarrHireVer_posts NarrHireGap_broad NarrHireWash_broad NarrHireHush_broad NarrHireVer_broad {
    capture confirm variable `x'
    if _rc continue
    quietly summarize `x', detail
    local x_N = r(N)
    local x_mean = r(mean)
    local x_sd = r(sd)
    local x_p25 = r(p25)
    local x_p50 = r(p50)
    local x_p75 = r(p75)
    quietly count if `x' > 0 & !missing(`x')
    local x_pos = r(N)
    post `xpost' ("`x'") (`x_N') (`x_mean') (`x_sd') (`x_p25') (`x_p50') (`x_p75') (`x_pos')
}

postclose `xpost'
use `xsummary', clear
export delimited using "$V3/results/stata/hiring_pure_mismatch_v1_x_summary.csv", replace
save "$V3/results/stata/hiring_pure_mismatch_v1_x_summary.dta", replace

* -----------------------------
* 3. Panel Y screen
* -----------------------------

use "$V3/results/data/hiring_pure_mismatch_v1_panel.dta", clear
sort Stkcd_num year_num
tsset Stkcd_num year_num

local ctrls "Size Lev ROA TobinQ Age Growth IndepRatio Dual Top1Share SOE CFO"
local xlist "NarrHireGap_title NarrHireWash_title NarrHireHush_title NarrHireVer_title NHireWash_title_s NHireHush_title_s NHireVer_title_s NarrHireGap_engineer NarrHireWash_engineer NarrHireHush_engineer NarrHireVer_engineer NarrHireGap_posts NarrHireWash_posts NarrHireHush_posts NarrHireVer_posts NarrHireGap_broad NarrHireWash_broad NarrHireHush_broad NarrHireVer_broad"
local ylist "PriceDelay ForecastDisp FcstAcc Analyst ReportFreq RatingDisp TobinQ SA InstHold InstStable AuditFee absDA InvestIneff CashFlowVol SCConc SuppConc TFP"

tempfile out
tempname posth
postfile `posth' str36 x_name str24 y_name str20 y_family str24 spec double coef se t_stat p_value N positives using `out', replace

foreach y of local ylist {
    local family "other"
    if inlist("`y'", "ForecastDisp", "FcstAcc", "Analyst", "ReportFreq", "RatingDisp") local family "analyst_info"
    if inlist("`y'", "PriceDelay") local family "pricing_efficiency"
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
            capture confirm variable `c'
            if !_rc & "`c'" != "`y'" {
                local regctrls "`regctrls' `c'"
                local regmiss "`regmiss', `c'"
            }
        }

        quietly count if !missing(`y', x_lag `regmiss', Stkcd_num, year_num, IndYear_num)
        if r(N) == 0 {
            post `posth' ("`x'") ("`y'") ("`family'") ("firm_yearFE_lagX") (.) (.) (.) (.) (0) (0)
            continue
        }

        quietly count if !missing(`y', x_lag `regmiss', Stkcd_num, year_num, IndYear_num) & x_lag > 0
        local xpos = r(N)

        capture noisily quietly reghdfe `y' x_lag `regctrls', absorb(Stkcd_num year_num) cluster(IndYear_num)
        if _rc {
            post `posth' ("`x'") ("`y'") ("`family'") ("firm_yearFE_lagX") (.) (.) (.) (.) (0) (`xpos')
            continue
        }

        capture local coef = _b[x_lag]
        if _rc {
            post `posth' ("`x'") ("`y'") ("`family'") ("firm_yearFE_lagX") (.) (.) (.) (.) (e(N)) (`xpos')
            continue
        }
        local se = _se[x_lag]
        local tval = `coef' / `se'
        local pval = 2 * ttail(e(df_r), abs(`tval'))
        post `posth' ("`x'") ("`y'") ("`family'") ("firm_yearFE_lagX") (`coef') (`se') (`tval') (`pval') (e(N)) (`xpos')
    }
}

postclose `posth'
use `out', clear
gen sig_10 = abs(t_stat) >= 1.645 if !missing(t_stat)
gen sig_05 = abs(t_stat) >= 1.96 if !missing(t_stat)
gen sig_01 = abs(t_stat) >= 2.576 if !missing(t_stat)
order x_name y_family y_name spec coef se t_stat p_value N positives sig_10 sig_05 sig_01
sort x_name y_family y_name
export delimited using "$V3/results/stata/hiring_pure_mismatch_v1_y_screen.csv", replace
save "$V3/results/stata/hiring_pure_mismatch_v1_y_screen.dta", replace

display _newline "=== X summary ==="
use "$V3/results/stata/hiring_pure_mismatch_v1_x_summary.dta", clear
list, abbreviate(28)

display _newline "=== Screen highlights: |t| >= 1.96 ==="
use "$V3/results/stata/hiring_pure_mismatch_v1_y_screen.dta", clear
list x_name y_family y_name coef t_stat p_value N positives if sig_05 == 1, sepby(x_name) abbreviate(28)

log close
