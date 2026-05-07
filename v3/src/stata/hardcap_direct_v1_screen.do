clear all
set more off
capture log close

global V3 "/Users/mac/computerscience/0做完了/15会计研究/v3"
global V1 "/Users/mac/computerscience/0做完了/15会计研究/v1"

log using "$V3/results/logs/hardcap_direct_v1_screen.log", replace text

display "=== HardCap direct v1 screen ==="
display "Input 1: v1/data_stata/reg_sample_v18.dta"
display "Input 2: v3/results/data/hardcap_direct_v1_components.csv"
display "Spec: Y_t = L.X + controls + firm FE + year FE, cluster(IndYear_num)"

capture which reghdfe
if _rc {
    display as error "reghdfe not installed"
    exit 199
}

* -----------------------------
* 0. Import hard-capability components
* -----------------------------

tempfile hardcap
import delimited using "$V3/results/data/hardcap_direct_v1_components.csv", clear varnames(1)

capture rename stkcd_num Stkcd_num
capture rename year_num year_num
capture rename diginvpataut DigInvPatAut
capture rename dighumandemand DigHumanDemand
capture rename dighumanrecruit DigHumanRecruit
capture rename digcapexitem DigCapexItem
capture rename digcapexamount DigCapexAmount
capture rename aisoftinvest AISoftInvest
capture rename aisoftinvestvalueadd AISoftInvestValueAdd
capture rename aihardinvest AIHardInvest
capture rename aihardinvestvalueadd AIHardInvestValueAdd
capture rename aiinvesttotal AIInvestTotal
capture rename aiinvesttotalvalueadd AIInvestTotalValueAdd
capture rename aiinvestlevel AIInvestLevel
capture rename patentallapp PatentAllApp
capture rename patentinvapp PatentInvApp
capture rename patenttitledataapp PatentTitleDataApp
capture rename patentallgrant PatentAllGrant
capture rename patentinvgrant PatentInvGrant
capture rename patenttitledatagrant PatentTitleDataGrant
capture rename diginvpataut_ln DigInvPatAut_ln
capture rename dighumandemand_ln DigHumanDemand_ln
capture rename dighumanrecruit_ln DigHumanRecruit_ln
capture rename digcapexitem_ln DigCapexItem_ln
capture rename digcapexamount_ln DigCapexAmount_ln
capture rename aiinvesttotal_ln AIInvestTotal_ln
capture rename aiinvesttotalvalueadd_ln AIInvestTotalValueAdd_ln
capture rename patenttitledataapp_ln PatentTitleDataApp_ln
capture rename patenttitledatagrant_ln PatentTitleDataGrant_ln

foreach v of varlist _all {
    capture destring `v', replace force
}

duplicates drop Stkcd_num year_num, force
save `hardcap', replace

use "$V1/data_stata/reg_sample_v18.dta", clear
merge 1:1 Stkcd_num year_num using `hardcap', keep(master match) nogen

* Treat missing values as zero only inside each source's documented coverage.
foreach v in DigInvPatAut DigInvPatAut_ln DigCapexItem DigCapexItem_ln DigCapexAmount DigCapexAmount_ln {
    capture replace `v' = 0 if missing(`v') & inrange(year_num, 2011, 2024)
}
foreach v in DigHumanDemand DigHumanDemand_ln DigHumanRecruit DigHumanRecruit_ln {
    capture replace `v' = 0 if missing(`v') & inrange(year_num, 2022, 2026)
}
foreach v in AISoftInvest AISoftInvestValueAdd AIHardInvest AIHardInvestValueAdd AIInvestTotal AIInvestTotalValueAdd AIInvestTotal_ln AIInvestTotalValueAdd_ln AIInvestLevel {
    capture replace `v' = 0 if missing(`v') & inrange(year_num, 2007, 2025)
}
foreach v in PatentAllApp PatentInvApp PatentTitleDataApp PatentTitleDataApp_ln PatentAllGrant PatentInvGrant PatentTitleDataGrant PatentTitleDataGrant_ln {
    capture replace `v' = 0 if missing(`v') & inrange(year_num, 1980, 2025)
}

* Recompute logs after merge/imputation.
foreach pair in "DigInvPatAut DigInvPatAut_ln" "DigHumanDemand DigHumanDemand_ln" "DigHumanRecruit DigHumanRecruit_ln" "DigCapexItem DigCapexItem_ln" "DigCapexAmount DigCapexAmount_ln" "AIInvestTotal AIInvestTotal_ln" "AIInvestTotalValueAdd AIInvestTotalValueAdd_ln" "PatentTitleDataApp PatentTitleDataApp_ln" "PatentTitleDataGrant PatentTitleDataGrant_ln" {
    tokenize "`pair'"
    capture replace `2' = ln(1 + `1') if !missing(`1')
}

sort Stkcd_num year_num
tsset Stkcd_num year_num

* -----------------------------
* 1. Construct direct hard capability scores
* -----------------------------

foreach v in DigInvPatAut_ln DigHumanDemand_ln DigHumanRecruit_ln DigCapexAmount_ln AIInvestTotal_ln AIInvestLevel PatentTitleDataApp_ln PatentTitleDataGrant_ln {
    capture drop z_`v'
    capture confirm variable `v'
    if !_rc {
        egen z_`v' = std(`v')
    }
}

capture drop HardCap_direct_v1 HardCap_direct_amt HardCap_direct_patent_human
egen HardCap_direct_v1 = rowmean(z_DigInvPatAut_ln z_DigHumanDemand_ln z_AIInvestLevel z_DigCapexAmount_ln z_PatentTitleDataApp_ln)
egen HardCap_direct_amt = rowmean(z_DigInvPatAut_ln z_DigHumanDemand_ln z_AIInvestTotal_ln z_DigCapexAmount_ln z_PatentTitleDataApp_ln)
egen HardCap_direct_patent_human = rowmean(z_DigInvPatAut_ln z_DigHumanDemand_ln z_PatentTitleDataApp_ln)

label variable HardCap_direct_v1 "Direct hard capability: digital patent, hiring, AI investment level, capex, title data patent"
label variable HardCap_direct_amt "Direct hard capability using AI investment amount"
label variable HardCap_direct_patent_human "Direct hard capability using patent and hiring only"

* Disclosure side.
capture drop neg_DU_kw n_disc_y rank_disc_y disc_pct_y
gen neg_DU_kw = -DU_kw
bys year_num: egen n_disc_y = count(DU_kw)
bys year_num: egen rank_disc_y = rank(neg_DU_kw) if !missing(neg_DU_kw), field
gen disc_pct_y = (rank_disc_y - 1) / (n_disc_y - 1) if n_disc_y > 1

foreach h in HardCap_direct_v1 HardCap_direct_amt HardCap_direct_patent_human {
    capture drop neg_`h' n_`h' rank_`h' pct_`h'
    gen neg_`h' = -`h'
    bys year_num: egen n_`h' = count(`h')
    bys year_num: egen rank_`h' = rank(neg_`h') if !missing(neg_`h'), field
    gen pct_`h' = (rank_`h' - 1) / (n_`h' - 1) if n_`h' > 1
}

capture drop NarrHardGap_direct_v1 NarrHardWashing_direct_v1 NarrHardHushing_direct_v1 NarrHardVerified_direct_v1
capture drop NHWash_direct_v1_s NHHush_direct_v1_s NHVer_direct_v1_s
gen NarrHardGap_direct_v1 = disc_pct_y - pct_HardCap_direct_v1
gen NarrHardWashing_direct_v1 = (disc_pct_y >= .50 & pct_HardCap_direct_v1 < .50) if !missing(disc_pct_y, pct_HardCap_direct_v1)
gen NarrHardHushing_direct_v1 = (pct_HardCap_direct_v1 >= .50 & disc_pct_y < .50) if !missing(disc_pct_y, pct_HardCap_direct_v1)
gen NarrHardVerified_direct_v1 = (disc_pct_y >= .50 & pct_HardCap_direct_v1 >= .50) if !missing(disc_pct_y, pct_HardCap_direct_v1)
gen NHWash_direct_v1_s = (disc_pct_y >= .75 & pct_HardCap_direct_v1 <= .25) if !missing(disc_pct_y, pct_HardCap_direct_v1)
gen NHHush_direct_v1_s = (pct_HardCap_direct_v1 >= .75 & disc_pct_y <= .25) if !missing(disc_pct_y, pct_HardCap_direct_v1)
gen NHVer_direct_v1_s = (disc_pct_y >= .75 & pct_HardCap_direct_v1 >= .75) if !missing(disc_pct_y, pct_HardCap_direct_v1)

capture drop NarrHardGap_direct_amt NarrHardWashing_direct_amt NarrHardHushing_direct_amt NarrHardVerified_direct_amt
gen NarrHardGap_direct_amt = disc_pct_y - pct_HardCap_direct_amt
gen NarrHardWashing_direct_amt = (disc_pct_y >= .50 & pct_HardCap_direct_amt < .50) if !missing(disc_pct_y, pct_HardCap_direct_amt)
gen NarrHardHushing_direct_amt = (pct_HardCap_direct_amt >= .50 & disc_pct_y < .50) if !missing(disc_pct_y, pct_HardCap_direct_amt)
gen NarrHardVerified_direct_amt = (disc_pct_y >= .50 & pct_HardCap_direct_amt >= .50) if !missing(disc_pct_y, pct_HardCap_direct_amt)

capture drop NarrHardGap_patent_human NarrHardWashing_patent_human NarrHardHushing_patent_human NarrHardVerified_patent_human
gen NarrHardGap_patent_human = disc_pct_y - pct_HardCap_direct_patent_human
gen NarrHardWashing_patent_human = (disc_pct_y >= .50 & pct_HardCap_direct_patent_human < .50) if !missing(disc_pct_y, pct_HardCap_direct_patent_human)
gen NarrHardHushing_patent_human = (pct_HardCap_direct_patent_human >= .50 & disc_pct_y < .50) if !missing(disc_pct_y, pct_HardCap_direct_patent_human)
gen NarrHardVerified_patent_human = (disc_pct_y >= .50 & pct_HardCap_direct_patent_human >= .50) if !missing(disc_pct_y, pct_HardCap_direct_patent_human)

save "$V3/results/data/hardcap_direct_v1_panel.dta", replace

* -----------------------------
* 2. X and component summary
* -----------------------------

tempfile xsummary
tempname xpost
postfile `xpost' str44 x_name double N mean sd p25 median p75 positives using `xsummary', replace

foreach x in DigInvPatAut DigHumanDemand DigHumanRecruit DigCapexAmount AIInvestTotal AIInvestLevel PatentTitleDataApp HardCap_direct_v1 HardCap_direct_amt HardCap_direct_patent_human NarrHardGap_direct_v1 NarrHardWashing_direct_v1 NarrHardHushing_direct_v1 NarrHardVerified_direct_v1 NHWash_direct_v1_s NHHush_direct_v1_s NHVer_direct_v1_s NarrHardGap_direct_amt NarrHardWashing_direct_amt NarrHardHushing_direct_amt NarrHardVerified_direct_amt NarrHardGap_patent_human NarrHardWashing_patent_human NarrHardHushing_patent_human NarrHardVerified_patent_human {
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
export delimited using "$V3/results/stata/hardcap_direct_v1_x_summary.csv", replace
save "$V3/results/stata/hardcap_direct_v1_x_summary.dta", replace

* -----------------------------
* 3. Panel Y screen
* -----------------------------

use "$V3/results/data/hardcap_direct_v1_panel.dta", clear
sort Stkcd_num year_num
tsset Stkcd_num year_num

local ctrls "Size Lev ROA TobinQ Age Growth IndepRatio Dual Top1Share SOE CFO"
local xlist "NarrHardGap_direct_v1 NarrHardWashing_direct_v1 NarrHardHushing_direct_v1 NarrHardVerified_direct_v1 NHWash_direct_v1_s NHHush_direct_v1_s NHVer_direct_v1_s NarrHardGap_direct_amt NarrHardWashing_direct_amt NarrHardHushing_direct_amt NarrHardVerified_direct_amt NarrHardGap_patent_human NarrHardWashing_patent_human NarrHardHushing_patent_human NarrHardVerified_patent_human"
local ylist "PriceDelay ForecastDisp FcstAcc Analyst ReportFreq RatingDisp TobinQ SA InstHold InstStable AuditFee absDA InvestIneff CashFlowVol SCConc SuppConc TFP"

tempfile out
tempname posth
postfile `posth' str44 x_name str24 y_name str20 y_family str24 spec double coef se t_stat p_value N positives using `out', replace

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
export delimited using "$V3/results/stata/hardcap_direct_v1_y_screen.csv", replace
save "$V3/results/stata/hardcap_direct_v1_y_screen.dta", replace

display _newline "=== X summary ==="
use "$V3/results/stata/hardcap_direct_v1_x_summary.dta", clear
list, abbreviate(28)

display _newline "=== Screen highlights: |t| >= 1.96 ==="
use "$V3/results/stata/hardcap_direct_v1_y_screen.dta", clear
list x_name y_family y_name coef t_stat p_value N positives if sig_05 == 1, sepby(x_name) abbreviate(28)

log close
