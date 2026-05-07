clear all
set more off
capture log close

global V3 "/Users/mac/computerscience/0做完了/15会计研究/v3"
global V1 "/Users/mac/computerscience/0做完了/15会计研究/v1"

log using "$V3/results/logs/hardcap_hiring_custom_v1_screen.log", replace text

display "=== HardCap hiring custom v1 screen ==="
display "Input 1: v1/data_stata/reg_sample_v18.dta"
display "Input 2: v3/results/data/hardcap_direct_v1_components.csv"
display "Input 3: v3/results/data/hiring_custom_v1_for_stata.csv"
display "Spec: Y_t = L.X + controls + firm FE + year FE, cluster(IndYear_num)"

capture which reghdfe
if _rc {
    display as error "reghdfe not installed"
    exit 199
}

* -----------------------------
* 0. Import direct hard-capability components
* -----------------------------

tempfile hardcap
import delimited using "$V3/results/data/hardcap_direct_v1_components.csv", clear varnames(1)

capture rename stkcd_num Stkcd_num
capture rename year_num year_num
capture rename diginvpataut DigInvPatAut
capture rename digcapexamount DigCapexAmount
capture rename aiinvesttotal AIInvestTotal
capture rename aiinvestlevel AIInvestLevel
capture rename patenttitledataapp PatentTitleDataApp
capture rename diginvpataut_ln DigInvPatAut_ln
capture rename digcapexamount_ln DigCapexAmount_ln
capture rename aiinvesttotal_ln AIInvestTotal_ln
capture rename patenttitledataapp_ln PatentTitleDataApp_ln

foreach v of varlist _all {
    capture destring `v', replace force
}
duplicates drop Stkcd_num year_num, force
save `hardcap', replace

tempfile hire
import delimited using "$V3/results/data/hiring_custom_v1_for_stata.csv", clear varnames(1)

capture rename stkcd_num Stkcd_num
capture rename year_num year_num
capture rename hirepostsall HirePostsAll
capture rename hirerecruitsall HireRecruitsAll
capture rename datahirebroadposts DataHireBroadPosts
capture rename datahirebroadshare DataHireBroadShare
capture rename datahiretitleposts DataHireTitlePosts
capture rename datahiretitleshare DataHireTitleShare
capture rename datahiretitlepostsln DataHireTitlePostsLn
capture rename aihireposts AIHirePosts
capture rename aihireshare AIHireShare
capture rename aihirepostsln AIHirePostsLn
capture rename softhireposts SoftHirePosts
capture rename softhireshare SoftHireShare
capture rename dataenghireposts DataEngHirePosts
capture rename dataenghireshare DataEngHireShare
capture rename dataenghirepostsln DataEngHirePostsLn
capture rename datasalaryavg DataSalaryAvg

foreach v of varlist _all {
    capture destring `v', replace force
}
duplicates drop Stkcd_num year_num, force
save `hire', replace

use "$V1/data_stata/reg_sample_v18.dta", clear
merge 1:1 Stkcd_num year_num using `hardcap', keep(master match) nogen
merge 1:1 Stkcd_num year_num using `hire', keep(master match) nogen

* Direct components: zero only inside source coverage.
foreach v in DigInvPatAut DigInvPatAut_ln DigCapexAmount DigCapexAmount_ln {
    capture replace `v' = 0 if missing(`v') & inrange(year_num, 2011, 2024)
}
foreach v in AIInvestTotal AIInvestTotal_ln AIInvestLevel {
    capture replace `v' = 0 if missing(`v') & inrange(year_num, 2007, 2025)
}
foreach v in PatentTitleDataApp PatentTitleDataApp_ln {
    capture replace `v' = 0 if missing(`v') & inrange(year_num, 1980, 2025)
}

* Custom hiring data covers 2014-2026.
foreach v in HirePostsAll HireRecruitsAll DataHireBroadPosts DataHireTitlePosts AIHirePosts SoftHirePosts DataEngHirePosts DataHireTitlePostsLn AIHirePostsLn DataEngHirePostsLn {
    capture replace `v' = 0 if missing(`v') & inrange(year_num, 2014, 2026)
}
foreach v in DataHireBroadShare DataHireTitleShare AIHireShare SoftHireShare DataEngHireShare {
    capture replace `v' = 0 if missing(`v') & inrange(year_num, 2014, 2026)
}

* Recompute logs after imputation.
foreach pair in "DigInvPatAut DigInvPatAut_ln" "DigCapexAmount DigCapexAmount_ln" "AIInvestTotal AIInvestTotal_ln" "PatentTitleDataApp PatentTitleDataApp_ln" "DataHireTitlePosts DataHireTitlePostsLn" "AIHirePosts AIHirePostsLn" "DataEngHirePosts DataEngHirePostsLn" {
    tokenize "`pair'"
    capture replace `2' = ln(1 + `1') if !missing(`1')
}

sort Stkcd_num year_num
tsset Stkcd_num year_num

* -----------------------------
* 1. Construct custom-hiring hard capability scores
* -----------------------------

foreach v in DigInvPatAut_ln DigCapexAmount_ln AIInvestTotal_ln AIInvestLevel PatentTitleDataApp_ln DataHireBroadShare DataHireTitleShare DataHireTitlePostsLn AIHireShare AIHirePostsLn DataEngHireShare DataEngHirePostsLn {
    capture drop z_`v'
    capture confirm variable `v'
    if !_rc {
        egen z_`v' = std(`v')
    }
}

capture drop HCap_hire_title HCap_hire_engineer HCap_hire_posts HCap_hire_broad
egen HCap_hire_title = rowmean(z_DigInvPatAut_ln z_DataHireTitleShare z_AIInvestLevel z_DigCapexAmount_ln z_PatentTitleDataApp_ln)
egen HCap_hire_engineer = rowmean(z_DigInvPatAut_ln z_DataEngHireShare z_AIHireShare z_AIInvestLevel z_DigCapexAmount_ln z_PatentTitleDataApp_ln)
egen HCap_hire_posts = rowmean(z_DigInvPatAut_ln z_DataHireTitlePostsLn z_DataEngHirePostsLn z_AIHirePostsLn z_AIInvestLevel z_DigCapexAmount_ln z_PatentTitleDataApp_ln)
egen HCap_hire_broad = rowmean(z_DigInvPatAut_ln z_DataHireBroadShare z_AIInvestLevel z_DigCapexAmount_ln z_PatentTitleDataApp_ln)

label variable HCap_hire_title "Hard capability with title-based data hiring share"
label variable HCap_hire_engineer "Hard capability with data-engineer and AI hiring shares"
label variable HCap_hire_posts "Hard capability with custom data/AI hiring post counts"
label variable HCap_hire_broad "Hard capability with broad text-based data hiring share"

* Disclosure percentile.
capture drop neg_DU_kw n_disc_y rank_disc_y disc_pct_y
gen neg_DU_kw = -DU_kw
bys year_num: egen n_disc_y = count(DU_kw)
bys year_num: egen rank_disc_y = rank(neg_DU_kw) if !missing(neg_DU_kw), field
gen disc_pct_y = (rank_disc_y - 1) / (n_disc_y - 1) if n_disc_y > 1

foreach h in HCap_hire_title HCap_hire_engineer HCap_hire_posts HCap_hire_broad {
    capture drop neg_`h' n_`h' rank_`h' pct_`h'
    gen neg_`h' = -`h'
    bys year_num: egen n_`h' = count(`h')
    bys year_num: egen rank_`h' = rank(neg_`h') if !missing(neg_`h'), field
    gen pct_`h' = (rank_`h' - 1) / (n_`h' - 1) if n_`h' > 1
}

capture drop NHGap_hire_title NHWash_hire_title NHHush_hire_title NHVer_hire_title
capture drop NHWash_hire_title_s NHHush_hire_title_s NHVer_hire_title_s
gen NHGap_hire_title = disc_pct_y - pct_HCap_hire_title
gen NHWash_hire_title = (disc_pct_y >= .50 & pct_HCap_hire_title < .50) if !missing(disc_pct_y, pct_HCap_hire_title)
gen NHHush_hire_title = (pct_HCap_hire_title >= .50 & disc_pct_y < .50) if !missing(disc_pct_y, pct_HCap_hire_title)
gen NHVer_hire_title = (disc_pct_y >= .50 & pct_HCap_hire_title >= .50) if !missing(disc_pct_y, pct_HCap_hire_title)
gen NHWash_hire_title_s = (disc_pct_y >= .75 & pct_HCap_hire_title <= .25) if !missing(disc_pct_y, pct_HCap_hire_title)
gen NHHush_hire_title_s = (pct_HCap_hire_title >= .75 & disc_pct_y <= .25) if !missing(disc_pct_y, pct_HCap_hire_title)
gen NHVer_hire_title_s = (disc_pct_y >= .75 & pct_HCap_hire_title >= .75) if !missing(disc_pct_y, pct_HCap_hire_title)

capture drop NHGap_hire_engineer NHWash_hire_engineer NHHush_hire_engineer NHVer_hire_engineer
gen NHGap_hire_engineer = disc_pct_y - pct_HCap_hire_engineer
gen NHWash_hire_engineer = (disc_pct_y >= .50 & pct_HCap_hire_engineer < .50) if !missing(disc_pct_y, pct_HCap_hire_engineer)
gen NHHush_hire_engineer = (pct_HCap_hire_engineer >= .50 & disc_pct_y < .50) if !missing(disc_pct_y, pct_HCap_hire_engineer)
gen NHVer_hire_engineer = (disc_pct_y >= .50 & pct_HCap_hire_engineer >= .50) if !missing(disc_pct_y, pct_HCap_hire_engineer)

capture drop NHGap_hire_posts NHWash_hire_posts NHHush_hire_posts NHVer_hire_posts
gen NHGap_hire_posts = disc_pct_y - pct_HCap_hire_posts
gen NHWash_hire_posts = (disc_pct_y >= .50 & pct_HCap_hire_posts < .50) if !missing(disc_pct_y, pct_HCap_hire_posts)
gen NHHush_hire_posts = (pct_HCap_hire_posts >= .50 & disc_pct_y < .50) if !missing(disc_pct_y, pct_HCap_hire_posts)
gen NHVer_hire_posts = (disc_pct_y >= .50 & pct_HCap_hire_posts >= .50) if !missing(disc_pct_y, pct_HCap_hire_posts)

capture drop NHGap_hire_broad
gen NHGap_hire_broad = disc_pct_y - pct_HCap_hire_broad

save "$V3/results/data/hardcap_hiring_custom_v1_panel.dta", replace

* -----------------------------
* 2. X summary
* -----------------------------

tempfile xsummary
tempname xpost
postfile `xpost' str36 x_name double N mean sd p25 median p75 positives using `xsummary', replace

foreach x in HirePostsAll DataHireBroadShare DataHireTitleShare AIHireShare DataEngHireShare HCap_hire_title HCap_hire_engineer HCap_hire_posts HCap_hire_broad NHGap_hire_title NHWash_hire_title NHHush_hire_title NHVer_hire_title NHWash_hire_title_s NHHush_hire_title_s NHVer_hire_title_s NHGap_hire_engineer NHWash_hire_engineer NHHush_hire_engineer NHVer_hire_engineer NHGap_hire_posts NHWash_hire_posts NHHush_hire_posts NHVer_hire_posts NHGap_hire_broad {
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
export delimited using "$V3/results/stata/hardcap_hiring_custom_v1_x_summary.csv", replace
save "$V3/results/stata/hardcap_hiring_custom_v1_x_summary.dta", replace

* -----------------------------
* 3. Panel Y screen
* -----------------------------

use "$V3/results/data/hardcap_hiring_custom_v1_panel.dta", clear
sort Stkcd_num year_num
tsset Stkcd_num year_num

local ctrls "Size Lev ROA TobinQ Age Growth IndepRatio Dual Top1Share SOE CFO"
local xlist "NHGap_hire_title NHWash_hire_title NHHush_hire_title NHVer_hire_title NHWash_hire_title_s NHHush_hire_title_s NHVer_hire_title_s NHGap_hire_engineer NHWash_hire_engineer NHHush_hire_engineer NHVer_hire_engineer NHGap_hire_posts NHWash_hire_posts NHHush_hire_posts NHVer_hire_posts NHGap_hire_broad"
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
export delimited using "$V3/results/stata/hardcap_hiring_custom_v1_y_screen.csv", replace
save "$V3/results/stata/hardcap_hiring_custom_v1_y_screen.dta", replace

display _newline "=== X summary ==="
use "$V3/results/stata/hardcap_hiring_custom_v1_x_summary.dta", clear
list, abbreviate(28)

display _newline "=== Screen highlights: |t| >= 1.96 ==="
use "$V3/results/stata/hardcap_hiring_custom_v1_y_screen.dta", clear
list x_name y_family y_name coef t_stat p_value N positives if sig_05 == 1, sepby(x_name) abbreviate(28)

log close
