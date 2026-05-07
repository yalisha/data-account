clear all
set more off
capture log close

global V3 "/Users/mac/computerscience/0做完了/15会计研究/v3"

log using "$V3/results/logs/hiring_pure_mismatch_v1_compare.log", replace text

display "=== Compare pure hiring mismatch with direct and hiring-composite versions ==="

use "$V3/results/data/hardcap_direct_v1_panel.dta", clear
keep Stkcd_num year_num ///
    HardCap_direct_v1 NarrHardGap_direct_v1 NarrHardWashing_direct_v1 ///
    NarrHardHushing_direct_v1 NarrHardVerified_direct_v1
tempfile direct
save `direct', replace

use "$V3/results/data/hardcap_hiring_custom_v1_panel.dta", clear
keep Stkcd_num year_num HCap_hire_title HCap_hire_engineer HCap_hire_posts ///
    NHGap_hire_title NHWash_hire_title NHHush_hire_title NHVer_hire_title ///
    NHGap_hire_engineer NHWash_hire_engineer NHHush_hire_engineer NHVer_hire_engineer ///
    NHGap_hire_posts NHWash_hire_posts NHHush_hire_posts NHVer_hire_posts
tempfile composite
save `composite', replace

use "$V3/results/data/hiring_pure_mismatch_v1_panel.dta", clear
merge 1:1 Stkcd_num year_num using `direct', keep(master match) nogen
merge 1:1 Stkcd_num year_num using `composite', keep(master match) nogen

tempfile corrout
tempname posth
postfile `posth' str24 category str36 left_var str36 right_var double N rho using `corrout', replace

foreach pair in ///
    "HireCap_title HardCap_direct_v1" ///
    "HireCap_engineer HardCap_direct_v1" ///
    "HireCap_posts HardCap_direct_v1" ///
    "HireCap_broad HardCap_direct_v1" ///
    "NarrHireGap_title NarrHardGap_direct_v1" ///
    "NarrHireGap_engineer NarrHardGap_direct_v1" ///
    "NarrHireGap_posts NarrHardGap_direct_v1" ///
    "NarrHireGap_broad NarrHardGap_direct_v1" ///
    "NarrHireWash_title NarrHardWashing_direct_v1" ///
    "NarrHireWash_engineer NarrHardWashing_direct_v1" ///
    "NarrHireWash_posts NarrHardWashing_direct_v1" ///
    "NarrHireHush_title NarrHardHushing_direct_v1" ///
    "NarrHireHush_engineer NarrHardHushing_direct_v1" ///
    "NarrHireHush_posts NarrHardHushing_direct_v1" ///
    "NarrHireVer_title NarrHardVerified_direct_v1" ///
    "NarrHireVer_engineer NarrHardVerified_direct_v1" ///
    "NarrHireVer_posts NarrHardVerified_direct_v1" {
    tokenize "`pair'"
    quietly correlate `1' `2' if !missing(`1', `2')
    matrix C = r(C)
    post `posth' ("pure_vs_direct") ("`1'") ("`2'") (r(N)) (C[1,2])
}

foreach pair in ///
    "HireCap_title HCap_hire_title" ///
    "HireCap_engineer HCap_hire_engineer" ///
    "HireCap_posts HCap_hire_posts" ///
    "NarrHireGap_title NHGap_hire_title" ///
    "NarrHireGap_engineer NHGap_hire_engineer" ///
    "NarrHireGap_posts NHGap_hire_posts" ///
    "NarrHireWash_title NHWash_hire_title" ///
    "NarrHireWash_engineer NHWash_hire_engineer" ///
    "NarrHireWash_posts NHWash_hire_posts" ///
    "NarrHireHush_title NHHush_hire_title" ///
    "NarrHireHush_engineer NHHush_hire_engineer" ///
    "NarrHireHush_posts NHHush_hire_posts" ///
    "NarrHireVer_title NHVer_hire_title" ///
    "NarrHireVer_engineer NHVer_hire_engineer" ///
    "NarrHireVer_posts NHVer_hire_posts" {
    tokenize "`pair'"
    quietly correlate `1' `2' if !missing(`1', `2')
    matrix C = r(C)
    post `posth' ("pure_vs_composite") ("`1'") ("`2'") (r(N)) (C[1,2])
}

foreach pair in ///
    "HireCap_title HireCap_engineer" ///
    "HireCap_title HireCap_posts" ///
    "HireCap_title HireCap_broad" ///
    "HireCap_engineer HireCap_posts" ///
    "NarrHireGap_title NarrHireGap_engineer" ///
    "NarrHireGap_title NarrHireGap_posts" ///
    "NarrHireGap_title NarrHireGap_broad" ///
    "NarrHireWash_title NarrHireWash_engineer" ///
    "NarrHireWash_title NarrHireWash_posts" {
    tokenize "`pair'"
    quietly correlate `1' `2' if !missing(`1', `2')
    matrix C = r(C)
    post `posth' ("within_pure") ("`1'") ("`2'") (r(N)) (C[1,2])
}

postclose `posth'
use `corrout', clear
sort category left_var right_var
export delimited using "$V3/results/stata/hiring_pure_mismatch_v1_correlations.csv", replace
save "$V3/results/stata/hiring_pure_mismatch_v1_correlations.dta", replace

list, abbreviate(28)

log close
