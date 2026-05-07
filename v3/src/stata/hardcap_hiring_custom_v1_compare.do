clear all
set more off
capture log close

global V3 "/Users/mac/computerscience/0做完了/15会计研究/v3"

log using "$V3/results/logs/hardcap_hiring_custom_v1_compare.log", replace text

display "=== Compare hiring-refined X with direct v1 X ==="

use "$V3/results/data/hardcap_direct_v1_panel.dta", clear
keep Stkcd_num year_num ///
    HardCap_direct_v1 HardCap_direct_amt HardCap_direct_patent_human ///
    NarrHardGap_direct_v1 NarrHardWashing_direct_v1 NarrHardHushing_direct_v1 NarrHardVerified_direct_v1 ///
    NHWash_direct_v1_s NHHush_direct_v1_s NHVer_direct_v1_s
tempfile direct
save `direct', replace

use "$V3/results/data/hardcap_hiring_custom_v1_panel.dta", clear
merge 1:1 Stkcd_num year_num using `direct', keep(match) nogen

tempfile corrout
tempname posth
postfile `posth' str24 category str36 left_var str36 right_var double N rho using `corrout', replace

foreach pair in ///
    "HCap_hire_title HardCap_direct_v1" ///
    "HCap_hire_engineer HardCap_direct_v1" ///
    "HCap_hire_posts HardCap_direct_v1" ///
    "HCap_hire_broad HardCap_direct_v1" ///
    "HCap_hire_title HardCap_direct_amt" ///
    "HCap_hire_title HardCap_direct_patent_human" ///
    "HCap_hire_engineer HardCap_direct_amt" ///
    "HCap_hire_engineer HardCap_direct_patent_human" ///
    "HCap_hire_posts HardCap_direct_amt" ///
    "HCap_hire_posts HardCap_direct_patent_human" {
    tokenize "`pair'"
    quietly correlate `1' `2' if !missing(`1', `2')
    matrix C = r(C)
    post `posth' ("hardcap_score") ("`1'") ("`2'") (r(N)) (C[1,2])
}

foreach pair in ///
    "NHGap_hire_title NarrHardGap_direct_v1" ///
    "NHGap_hire_engineer NarrHardGap_direct_v1" ///
    "NHGap_hire_posts NarrHardGap_direct_v1" ///
    "NHGap_hire_broad NarrHardGap_direct_v1" ///
    "NHWash_hire_title NarrHardWashing_direct_v1" ///
    "NHWash_hire_engineer NarrHardWashing_direct_v1" ///
    "NHWash_hire_posts NarrHardWashing_direct_v1" ///
    "NHHush_hire_title NarrHardHushing_direct_v1" ///
    "NHHush_hire_engineer NarrHardHushing_direct_v1" ///
    "NHHush_hire_posts NarrHardHushing_direct_v1" ///
    "NHVer_hire_title NarrHardVerified_direct_v1" ///
    "NHVer_hire_engineer NarrHardVerified_direct_v1" ///
    "NHVer_hire_posts NarrHardVerified_direct_v1" ///
    "NHWash_hire_title_s NHWash_direct_v1_s" ///
    "NHHush_hire_title_s NHHush_direct_v1_s" ///
    "NHVer_hire_title_s NHVer_direct_v1_s" {
    tokenize "`pair'"
    quietly correlate `1' `2' if !missing(`1', `2')
    matrix C = r(C)
    post `posth' ("x_vs_direct") ("`1'") ("`2'") (r(N)) (C[1,2])
}

foreach pair in ///
    "HCap_hire_title HCap_hire_engineer" ///
    "HCap_hire_title HCap_hire_posts" ///
    "HCap_hire_title HCap_hire_broad" ///
    "HCap_hire_engineer HCap_hire_posts" ///
    "NHGap_hire_title NHGap_hire_engineer" ///
    "NHGap_hire_title NHGap_hire_posts" ///
    "NHGap_hire_title NHGap_hire_broad" ///
    "NHWash_hire_title NHWash_hire_engineer" ///
    "NHWash_hire_title NHWash_hire_posts" {
    tokenize "`pair'"
    quietly correlate `1' `2' if !missing(`1', `2')
    matrix C = r(C)
    post `posth' ("within_hiring") ("`1'") ("`2'") (r(N)) (C[1,2])
}

postclose `posth'
use `corrout', clear
sort category left_var right_var
export delimited using "$V3/results/stata/hardcap_hiring_custom_v1_correlations.csv", replace
save "$V3/results/stata/hardcap_hiring_custom_v1_correlations.dta", replace

list, abbreviate(28)

log close
