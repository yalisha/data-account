clear all
set more off
capture log close

global V3 "/Users/mac/computerscience/0做完了/15会计研究/v3"
global V1 "/Users/mac/computerscience/0做完了/15会计研究/v1"

log using "$V3/results/logs/data_asset_entry_mismatch_2024.log", replace text

display "=== Data asset entry mismatch pilot, 2024 cross-section ==="
display "Input 1: v1/data_stata/reg_sample_v18.dta"
display "Input 2: v3/results/data/data_asset_from_panel.csv"
display "Spec: Y_2024 = X_2024 + controls + industry FE, robust SE"

capture which reghdfe
if _rc {
    display as error "reghdfe not installed"
    exit 199
}

tempfile asset
import delimited using "$V3/results/data/data_asset_from_panel.csv", clear varnames(1)
capture rename stkcd_num Stkcd_num
capture rename year_num year_num
capture rename dataasset DataAsset
destring Stkcd_num year_num DataAsset, replace force
duplicates drop Stkcd_num year_num, force
save `asset', replace

use "$V1/data_stata/reg_sample_v18.dta", clear
merge 1:1 Stkcd_num year_num using `asset', keep(master match) nogen

capture drop DataAsset0 DataAsset_ln BookEntry
gen DataAsset0 = DataAsset
replace DataAsset0 = 0 if missing(DataAsset0)
gen DataAsset_ln = ln(1 + DataAsset0)
gen BookEntry = (DataAsset0 > 0) if year_num == 2024

* -----------------------------
* 1. Construct 2024 narrative-booking mismatch Xs
* -----------------------------

capture drop n_year rank_disc_2024 rank_book_2024 disc_pct_2024 book_pct_2024
bys year_num: egen n_year = count(DU_kw)
bys year_num: egen rank_disc_2024 = rank(DU_kw) if year_num == 2024 & !missing(DU_kw), field
bys year_num: egen rank_book_2024 = rank(DataAsset_ln) if year_num == 2024 & !missing(DataAsset_ln), field
gen disc_pct_2024 = (rank_disc_2024 - 1) / (n_year - 1) if year_num == 2024 & n_year > 1
gen book_pct_2024 = (rank_book_2024 - 1) / (n_year - 1) if year_num == 2024 & n_year > 1

capture drop z_disc_2024 z_book_2024 NarrativeBookGap_z NarrativeBookGap_pct
egen z_disc_2024 = std(DU_kw) if year_num == 2024
egen z_book_2024 = std(DataAsset_ln) if year_num == 2024
gen NarrativeBookGap_z = z_disc_2024 - z_book_2024 if year_num == 2024
gen NarrativeBookGap_pct = disc_pct_2024 - book_pct_2024 if year_num == 2024

capture drop TalkNoBook TalkNoBook_strict VerifiedBook VerifiedBook_strict BookNoTalk BookNoTalk_strict AnyBook
gen TalkNoBook = (disc_pct_2024 >= .50 & BookEntry == 0) if year_num == 2024 & !missing(disc_pct_2024, BookEntry)
gen TalkNoBook_strict = (disc_pct_2024 >= .75 & BookEntry == 0) if year_num == 2024 & !missing(disc_pct_2024, BookEntry)
gen VerifiedBook = (disc_pct_2024 >= .50 & BookEntry == 1) if year_num == 2024 & !missing(disc_pct_2024, BookEntry)
gen VerifiedBook_strict = (disc_pct_2024 >= .75 & BookEntry == 1) if year_num == 2024 & !missing(disc_pct_2024, BookEntry)
gen BookNoTalk = (BookEntry == 1 & disc_pct_2024 < .50) if year_num == 2024 & !missing(disc_pct_2024, BookEntry)
gen BookNoTalk_strict = (BookEntry == 1 & disc_pct_2024 <= .25) if year_num == 2024 & !missing(disc_pct_2024, BookEntry)
gen AnyBook = BookEntry if year_num == 2024

label variable DataAsset0 "Data asset amount/ratio from parquet, missing set to zero"
label variable BookEntry "Positive data asset booking in 2024"
label variable NarrativeBookGap_z "z(data narrative) - z(data asset booking), 2024"
label variable NarrativeBookGap_pct "Percentile(data narrative) - percentile(data asset booking), 2024"
label variable TalkNoBook "High data narrative, no data asset booking, 2024"
label variable TalkNoBook_strict "Top-quartile data narrative, no data asset booking, 2024"
label variable VerifiedBook "High data narrative, positive data asset booking, 2024"
label variable VerifiedBook_strict "Top-quartile data narrative, positive data asset booking, 2024"
label variable BookNoTalk "Positive data asset booking, low data narrative, 2024"
label variable BookNoTalk_strict "Positive data asset booking, bottom-quartile data narrative, 2024"

save "$V3/results/data/data_asset_entry_mismatch_2024_panel.dta", replace

* X summary.
tempfile xsummary
tempname xpost
postfile `xpost' str32 x_name double N mean sd p25 median p75 positives using `xsummary', replace

foreach x in DataAsset0 DataAsset_ln BookEntry NarrativeBookGap_z NarrativeBookGap_pct TalkNoBook TalkNoBook_strict VerifiedBook VerifiedBook_strict BookNoTalk BookNoTalk_strict AnyBook {
    quietly summarize `x' if year_num == 2024, detail
    local x_N = r(N)
    local x_mean = r(mean)
    local x_sd = r(sd)
    local x_p25 = r(p25)
    local x_p50 = r(p50)
    local x_p75 = r(p75)
    quietly count if year_num == 2024 & `x' > 0 & !missing(`x')
    local x_pos = r(N)
    post `xpost' ("`x'") (`x_N') (`x_mean') (`x_sd') (`x_p25') (`x_p50') (`x_p75') (`x_pos')
}
postclose `xpost'
use `xsummary', clear
export delimited using "$V3/results/stata/data_asset_entry_x_summary_2024.csv", replace

* Restore full constructed panel.
use "$V3/results/data/data_asset_entry_mismatch_2024_panel.dta", clear

* -----------------------------
* 2. 2024 cross-sectional Y screen
* -----------------------------

local ctrls "Size Lev ROA TobinQ Age Growth IndepRatio Dual Top1Share SOE CFO"
local xlist "NarrativeBookGap_z NarrativeBookGap_pct TalkNoBook TalkNoBook_strict VerifiedBook VerifiedBook_strict BookNoTalk BookNoTalk_strict AnyBook"
local ylist "PriceDelay ForecastDisp FcstAcc Analyst ReportFreq RatingDisp TobinQ SA InstHold InstStable AuditFee absDA InvestIneff CashFlowVol SCConc SuppConc TFP"

tempfile out
tempname posth
postfile `posth' str32 x_name str24 y_name str20 y_family str24 spec double coef se t_stat p_value N positives using `out', replace

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

        local regctrls ""
        local regmiss ""
        foreach c of local ctrls {
            capture confirm variable `c'
            if !_rc & "`c'" != "`y'" {
                local regctrls "`regctrls' `c'"
                local regmiss "`regmiss', `c'"
            }
        }

        quietly count if year_num == 2024 & !missing(`y', `x' `regmiss', IndYear_num)
        if r(N) == 0 {
            post `posth' ("`x'") ("`y'") ("`family'") ("indFE_controls") (.) (.) (.) (.) (0) (0)
            continue
        }

        quietly count if year_num == 2024 & !missing(`y', `x' `regmiss', IndYear_num) & `x' > 0
        local xpos = r(N)

        capture noisily quietly reghdfe `y' `x' `regctrls' if year_num == 2024, absorb(IndYear_num) vce(robust)
        if _rc {
            post `posth' ("`x'") ("`y'") ("`family'") ("indFE_controls") (.) (.) (.) (.) (0) (`xpos')
            continue
        }

        capture local coef = _b[`x']
        if _rc {
            post `posth' ("`x'") ("`y'") ("`family'") ("indFE_controls") (.) (.) (.) (.) (e(N)) (`xpos')
            continue
        }
        local se = _se[`x']
        local tval = `coef' / `se'
        local pval = 2 * ttail(e(df_r), abs(`tval'))
        post `posth' ("`x'") ("`y'") ("`family'") ("indFE_controls") (`coef') (`se') (`tval') (`pval') (e(N)) (`xpos')
    }
}

postclose `posth'
use `out', clear
gen sig_10 = abs(t_stat) >= 1.645 if !missing(t_stat)
gen sig_05 = abs(t_stat) >= 1.96 if !missing(t_stat)
gen sig_01 = abs(t_stat) >= 2.576 if !missing(t_stat)
order x_name y_family y_name spec coef se t_stat p_value N positives sig_10 sig_05 sig_01
sort x_name y_family y_name
export delimited using "$V3/results/stata/data_asset_entry_y_screen_2024.csv", replace
save "$V3/results/stata/data_asset_entry_y_screen_2024.dta", replace

display _newline "=== X summary, 2024 ==="
use "$V3/results/data/data_asset_entry_mismatch_2024_panel.dta", clear
summarize DataAsset0 BookEntry NarrativeBookGap_z NarrativeBookGap_pct TalkNoBook VerifiedBook BookNoTalk if year_num == 2024
tab BookEntry if year_num == 2024
tab TalkNoBook if year_num == 2024
tab VerifiedBook if year_num == 2024
tab BookNoTalk if year_num == 2024

display _newline "=== Screen highlights: |t| >= 1.96 ==="
use "$V3/results/stata/data_asset_entry_y_screen_2024.dta", clear
list x_name y_family y_name coef t_stat p_value N positives if sig_05 == 1, sepby(x_name) abbreviate(24)

log close
