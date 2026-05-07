clear all
set more off
capture log close

global V3 "/Users/mac/computerscience/0做完了/15会计研究/v3"

log using "$V3/results/logs/did_data_ip_pilot_v1.log", replace text

display "=== DID/DDD v1: data IP pilot x pre-policy washing ==="
display "Policy: CNIPA data intellectual-property local pilots"
display "First pilots: Beijing, Shanghai, Jiangsu, Zhejiang, Fujian, Shandong, Guangdong/Shenzhen; start coded as 2023"
display "2024 added pilots: Tianjin, Hebei, Shanxi, Anhui, Henan, Hubei, Hunan, Guizhou, Shaanxi; start coded as 2024"
display "Sample: 2019-2024; exposure measured by 2020-2021 pre-policy mismatch"
display "Spec: Y_it = pilot_pt + exposure_i x pilot_pt + controls + firm FE + year FE, cluster(province)"

capture which reghdfe
if _rc {
    display as error "reghdfe not installed"
    exit 199
}

use "$V3/results/data/hardcap_direct_v1_panel.dta", clear

* -----------------------------
* 1. Policy and pre-policy exposure
* -----------------------------

gen FirstDIP = inlist(Prov_short, "北京", "上海", "江苏", "浙江", "福建", "山东", "广东")
gen NewDIP2024 = inlist(Prov_short, "天津", "河北", "山西", "安徽", "河南", "湖北", "湖南", "贵州", "陕西")
gen DIPilot = (FirstDIP == 1 & year_num >= 2023) | (NewDIP2024 == 1 & year_num >= 2024)
gen FirstDIPPost = FirstDIP == 1 & year_num >= 2023

bys Stkcd_num: egen PreGap2021 = mean(cond(inrange(year_num, 2020, 2021), NarrHardGap_direct_v1, .))
bys Stkcd_num: egen PreWashShare2021 = mean(cond(inrange(year_num, 2020, 2021), NarrHardWashing_direct_v1, .))
bys Stkcd_num: egen PreWashAny2021 = max(cond(inrange(year_num, 2020, 2021), NarrHardWashing_direct_v1, .))
bys Stkcd_num: egen PreStrictWashAny2021 = max(cond(inrange(year_num, 2020, 2021), NHWash_direct_v1_s, .))

gen PreWashMaj2021 = (PreWashShare2021 >= .5) if !missing(PreWashShare2021)

gen dip_gap = DIPilot * PreGap2021
gen dip_wany = DIPilot * PreWashAny2021
gen dip_wmaj = DIPilot * PreWashMaj2021
gen dip_wstr = DIPilot * PreStrictWashAny2021

label variable FirstDIP "First data-IP pilot provinces, 2022 notice"
label variable NewDIP2024 "New 2024 data-IP pilot provinces"
label variable DIPilot "Data-IP pilot province-year"
label variable PreGap2021 "Mean direct narrative-hard gap, 2020-2021"
label variable PreWashAny2021 "Any direct washing, 2020-2021"
label variable PreWashMaj2021 "Majority direct washing, 2020-2021"
label variable PreStrictWashAny2021 "Any strict direct washing, 2020-2021"
label variable dip_gap "PreGap2021 x DataIPPilot"
label variable dip_wany "PreWashAny2021 x DataIPPilot"
label variable dip_wmaj "PreWashMaj2021 x DataIPPilot"
label variable dip_wstr "PreStrictWashAny2021 x DataIPPilot"

* First-pilot event/parallel-trend variables; 2022 omitted.
foreach yy in 2019 2020 2021 2023 2024 {
    gen fd_`yy' = FirstDIP * (year_num == `yy')
    gen fgap_`yy' = FirstDIP * PreGap2021 * (year_num == `yy')
    gen fwany_`yy' = FirstDIP * PreWashAny2021 * (year_num == `yy')
    gen fwmaj_`yy' = FirstDIP * PreWashMaj2021 * (year_num == `yy')
    gen fwstr_`yy' = FirstDIP * PreStrictWashAny2021 * (year_num == `yy')
}

save "$V3/results/data/did_data_ip_pilot_v1_panel.dta", replace

* -----------------------------
* 2. Policy/exposure summary
* -----------------------------

preserve
keep if inrange(year_num, 2019, 2024)
tempfile exposum
tempname expost
postfile `expost' str30 item double N mean sd p25 median p75 positives using `exposum', replace

foreach x in FirstDIP NewDIP2024 DIPilot PreGap2021 PreWashShare2021 PreWashAny2021 PreWashMaj2021 PreStrictWashAny2021 dip_gap dip_wany dip_wmaj dip_wstr {
    quietly summarize `x', detail
    local x_N = r(N)
    local x_mean = r(mean)
    local x_sd = r(sd)
    local x_p25 = r(p25)
    local x_p50 = r(p50)
    local x_p75 = r(p75)
    quietly count if `x' > 0 & !missing(`x')
    local x_pos = r(N)
    post `expost' ("`x'") (`x_N') (`x_mean') (`x_sd') (`x_p25') (`x_p50') (`x_p75') (`x_pos')
}

postclose `expost'
use `exposum', clear
export delimited using "$V3/results/stata/did_data_ip_pilot_v1_exposure_summary.csv", replace
restore

* -----------------------------
* 3. Main DDD screen
* -----------------------------

local ctrls "Size Lev ROA TobinQ Age Growth IndepRatio Dual Top1Share SOE CFO"
local ylist "TobinQ Analyst ForecastDisp RatingDisp ReportFreq AuditFee PriceDelay SA InstHold InstStable InvestIneff CashFlowVol TFP SCConc SuppConc"
local xlist "dip_gap dip_wany dip_wmaj dip_wstr"

tempfile mainout
tempname mainpost
postfile `mainpost' str28 x_name str24 y_name str20 y_family str24 spec double coef se t_stat p_value N positives using `mainout', replace

foreach y of local ylist {
    local family "other"
    if inlist("`y'", "ForecastDisp", "Analyst", "ReportFreq", "RatingDisp") local family "analyst_info"
    if inlist("`y'", "PriceDelay") local family "pricing_efficiency"
    if inlist("`y'", "TobinQ") local family "valuation"
    if inlist("`y'", "SA") local family "financing_constraint"
    if inlist("`y'", "InstHold", "InstStable") local family "investor_attention"
    if inlist("`y'", "AuditFee") local family "audit_accounting"
    if inlist("`y'", "InvestIneff") local family "capital_allocation"
    if inlist("`y'", "CashFlowVol") local family "operating_resilience"
    if inlist("`y'", "SCConc", "SuppConc") local family "supply_chain"
    if inlist("`y'", "TFP") local family "productivity"

    capture confirm variable `y'
    if _rc continue

    foreach x of local xlist {
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
            post `mainpost' ("`x'") ("`y'") ("`family'") ("dataip_ddd_provclu") (.) (.) (.) (.) (0) (0)
            continue
        }

        quietly count if inrange(year_num, 2019, 2024) & !missing(`y', DIPilot, `x' `regmiss', Stkcd_num, year_num, Prov_num) & `x' > 0
        local xpos = r(N)

        capture noisily quietly reghdfe `y' DIPilot `x' `regctrls' if inrange(year_num, 2019, 2024), absorb(Stkcd_num year_num) cluster(Prov_num)
        if _rc {
            post `mainpost' ("`x'") ("`y'") ("`family'") ("dataip_ddd_provclu") (.) (.) (.) (.) (0) (`xpos')
            continue
        }

        capture local coef = _b[`x']
        if _rc {
            post `mainpost' ("`x'") ("`y'") ("`family'") ("dataip_ddd_provclu") (.) (.) (.) (.) (e(N)) (`xpos')
            continue
        }
        local se = _se[`x']
        local tval = `coef' / `se'
        local pval = 2 * ttail(e(df_r), abs(`tval'))
        post `mainpost' ("`x'") ("`y'") ("`family'") ("dataip_ddd_provclu") (`coef') (`se') (`tval') (`pval') (e(N)) (`xpos')
    }
}

postclose `mainpost'
use `mainout', clear
gen sig_10 = abs(t_stat) >= 1.645 if !missing(t_stat)
gen sig_05 = abs(t_stat) >= 1.96 if !missing(t_stat)
gen sig_01 = abs(t_stat) >= 2.576 if !missing(t_stat)
order x_name y_family y_name spec coef se t_stat p_value N positives sig_10 sig_05 sig_01
sort x_name y_family y_name
export delimited using "$V3/results/stata/did_data_ip_pilot_v1_main.csv", replace
save "$V3/results/stata/did_data_ip_pilot_v1_main.dta", replace

* -----------------------------
* 4. First-pilot event/pretrend screen: 2022 omitted
* -----------------------------

use "$V3/results/data/did_data_ip_pilot_v1_panel.dta", clear

tempfile evout
tempname evpost
postfile `evpost' str20 exposure str24 y_name str20 y_family int event_year double coef se t_stat p_value N using `evout', replace

local fdlist "fd_2019 fd_2020 fd_2021 fd_2023 fd_2024"

foreach y of local ylist {
    local family "other"
    if inlist("`y'", "ForecastDisp", "Analyst", "ReportFreq", "RatingDisp") local family "analyst_info"
    if inlist("`y'", "PriceDelay") local family "pricing_efficiency"
    if inlist("`y'", "TobinQ") local family "valuation"
    if inlist("`y'", "SA") local family "financing_constraint"
    if inlist("`y'", "InstHold", "InstStable") local family "investor_attention"
    if inlist("`y'", "AuditFee") local family "audit_accounting"
    if inlist("`y'", "InvestIneff") local family "capital_allocation"
    if inlist("`y'", "CashFlowVol") local family "operating_resilience"
    if inlist("`y'", "SCConc", "SuppConc") local family "supply_chain"
    if inlist("`y'", "TFP") local family "productivity"

    capture confirm variable `y'
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

    foreach block in "gap fgap" "wany fwany" "wmaj fwmaj" "wstr fwstr" {
        tokenize "`block'"
        local ename "`1'"
        local pref "`2'"

        capture noisily quietly reghdfe `y' `fdlist' `pref'_2019 `pref'_2020 `pref'_2021 `pref'_2023 `pref'_2024 `regctrls' if inrange(year_num, 2019, 2024), absorb(Stkcd_num year_num) cluster(Prov_num)
        if _rc {
            foreach yy in 2019 2020 2021 2023 2024 {
                post `evpost' ("`ename'") ("`y'") ("`family'") (`yy') (.) (.) (.) (.) (0)
            }
            continue
        }
        foreach yy in 2019 2020 2021 2023 2024 {
            capture local coef = _b[`pref'_`yy']
            if _rc {
                post `evpost' ("`ename'") ("`y'") ("`family'") (`yy') (.) (.) (.) (.) (e(N))
                continue
            }
            local se = _se[`pref'_`yy']
            local tval = `coef' / `se'
            local pval = 2 * ttail(e(df_r), abs(`tval'))
            post `evpost' ("`ename'") ("`y'") ("`family'") (`yy') (`coef') (`se') (`tval') (`pval') (e(N))
        }
    }
}

postclose `evpost'
use `evout', clear
gen sig_10 = abs(t_stat) >= 1.645 if !missing(t_stat)
gen sig_05 = abs(t_stat) >= 1.96 if !missing(t_stat)
gen sig_01 = abs(t_stat) >= 2.576 if !missing(t_stat)
order exposure y_family y_name event_year coef se t_stat p_value N sig_10 sig_05 sig_01
sort exposure y_family y_name event_year
export delimited using "$V3/results/stata/did_data_ip_pilot_v1_event.csv", replace
save "$V3/results/stata/did_data_ip_pilot_v1_event.dta", replace

display _newline "=== Data-IP DDD main highlights: |t| >= 1.96 ==="
use "$V3/results/stata/did_data_ip_pilot_v1_main.dta", clear
list x_name y_family y_name coef t_stat p_value N positives if sig_05 == 1, sepby(x_name) abbreviate(28)

display _newline "=== Data-IP first-pilot event highlights: |t| >= 1.96 ==="
use "$V3/results/stata/did_data_ip_pilot_v1_event.dta", clear
list exposure y_family y_name event_year coef t_stat p_value N if sig_05 == 1, sepby(exposure y_name) abbreviate(28)

log close
