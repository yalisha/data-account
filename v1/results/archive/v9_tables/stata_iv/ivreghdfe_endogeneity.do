* ==============================================================================
* Table 4: Endogeneity tests using ivreghdfe (unified 1st/2nd stage samples)
* ==============================================================================
* IV specs:
*   (1) OLS baseline
*   (2) Peer IV: leave-one-out industry-year mean of DU_kw
*   (3) Bartik IV: base_exposure * leave-out-industry growth
*   (4) Lag OLS: DU_kw_{t-1}
*   (5) Lag IV: peer mean of DU_kw_{t-1}
* DWH computed via first-stage residual augmented regression
* ==============================================================================

clear all
set more off

local BASE "/Users/mac/computerscience/15会计研究"
local OUTDIR "`BASE'/results/v9_tables/stata_iv"

use "`BASE'/data_stata/reg_sample_v3.dta", clear

* Drop obs with missing Ind2 (only 210 out of 44,067)
drop if missing(Ind2) | Ind2 == ""

* Numeric encoding for Ind2 (needed for grouping)
encode Ind2, gen(Ind2_enc)

global controls "Size Lev ROA TobinQ Age Growth BoardSize IndepRatio Dual Top1Share SOE InstHold Amihud Analyst AuditType"

* ==============================================================================
* Construct IV variables
* ==============================================================================

* --- 1. Peer IV: leave-one-out industry-year mean ---
bysort Ind2_enc year_num: egen iy_sum = total(DU_kw)
bysort Ind2_enc year_num: egen iy_count = count(DU_kw)
gen DU_kw_peer = (iy_sum - DU_kw) / (iy_count - 1) if iy_count > 1
drop iy_sum iy_count

* --- 2. Bartik IV ---
su year_num, meanonly
local base_yr = r(min)
bysort Ind2_enc: egen base_DU = mean(cond(year_num == `base_yr', DU_kw, .))

bysort year_num: egen nat_sum = total(DU_kw)
bysort year_num: egen nat_n = count(DU_kw)
bysort Ind2_enc year_num: egen ind_sum = total(DU_kw)
bysort Ind2_enc year_num: egen ind_n = count(DU_kw)
gen excl_mean = (nat_sum - ind_sum) / (nat_n - ind_n) if (nat_n - ind_n) > 0
bysort Ind2_enc: egen excl_mean_base = mean(cond(year_num == `base_yr', excl_mean, .))
gen growth_excl = excl_mean / max(excl_mean_base, 0.001)
gen bartik_iv = base_DU * growth_excl
drop nat_sum nat_n ind_sum ind_n excl_mean excl_mean_base growth_excl base_DU

* --- 3. Lag variables ---
sort Stkcd_num year_num
by Stkcd_num: gen DU_kw_lag = DU_kw[_n-1] if year_num == year_num[_n-1] + 1

bysort Ind2_enc year_num: egen iyl_sum = total(DU_kw_lag) if !missing(DU_kw_lag)
bysort Ind2_enc year_num: egen iyl_count = count(DU_kw_lag) if !missing(DU_kw_lag)
gen peer_lag = (iyl_sum - DU_kw_lag) / (iyl_count - 1) if iyl_count > 1 & !missing(DU_kw_lag)
drop iyl_sum iyl_count

* Summary
di "Total obs: " _N
count if !missing(DU_kw_peer)
di "Peer IV non-missing: " r(N)
count if !missing(bartik_iv)
di "Bartik IV non-missing: " r(N)
count if !missing(DU_kw_lag)
di "DU_kw_lag non-missing: " r(N)
count if !missing(peer_lag)
di "peer_lag non-missing: " r(N)

save "`OUTDIR'/iv_sample.dta", replace

* ==============================================================================
* (1) OLS Baseline
* ==============================================================================
di _n "============================================================"
di "Model (1): OLS Baseline"
di "============================================================"

reghdfe PriceDelay DU_kw $controls, absorb(Stkcd_num year_num) vce(cluster IndYear_num)
local ols_b = _b[DU_kw]
local ols_se = _se[DU_kw]
local ols_N = e(N)
local ols_r2 = e(r2)
di "OLS: b=" %9.6f `ols_b' " se=" %9.6f `ols_se' " N=" `ols_N' " R2=" %6.4f `ols_r2'

* ==============================================================================
* (2) Peer IV
* ==============================================================================
di _n "============================================================"
di "Model (2): Peer IV (ivreghdfe)"
di "============================================================"

ivreghdfe PriceDelay $controls (DU_kw = DU_kw_peer), absorb(Stkcd_num year_num) cluster(IndYear_num) first savefirst savefprefix(fs2_)
local peer_b = _b[DU_kw]
local peer_se = _se[DU_kw]
local peer_N = e(N)
local peer_KPF = e(widstat)

* DWH via first-stage residual method
* Step 1: get first-stage residuals (on same sample)
estimates restore fs2_DU_kw
local peer_fs_b = _b[DU_kw_peer]
local peer_fs_se = _se[DU_kw_peer]
local peer_fs_N = e(N)
local peer_fs_r2 = e(r2)
estimates drop fs2_DU_kw

* Step 2: manually compute residuals using reghdfe on same sample
preserve
keep if !missing(DU_kw_peer)
reghdfe DU_kw DU_kw_peer $controls, absorb(Stkcd_num year_num) resid(_resid_fs2)
* Step 3: augmented regression
reghdfe PriceDelay DU_kw _resid_fs2 $controls, absorb(Stkcd_num year_num) vce(cluster IndYear_num)
local peer_DWH_t = _b[_resid_fs2] / _se[_resid_fs2]
local peer_DWH_F = `peer_DWH_t'^2
local peer_DWH_p = 2 * ttail(e(df_r), abs(`peer_DWH_t'))
restore

di "Peer IV 2SLS: b=" %9.6f `peer_b' " se=" %9.6f `peer_se' " N=" `peer_N'
di "KP F=" %8.1f `peer_KPF' " DWH F=" %8.2f `peer_DWH_F' " DWH p=" %6.4f `peer_DWH_p'
di "1st stage: b=" %9.4f `peer_fs_b' " se=" %9.4f `peer_fs_se' " N=" `peer_fs_N' " R2=" %6.3f `peer_fs_r2'

* ==============================================================================
* (3) Bartik IV
* ==============================================================================
di _n "============================================================"
di "Model (3): Bartik IV (ivreghdfe)"
di "============================================================"

ivreghdfe PriceDelay $controls (DU_kw = bartik_iv), absorb(Stkcd_num year_num) cluster(IndYear_num) first savefirst savefprefix(fs3_)
local bk_b = _b[DU_kw]
local bk_se = _se[DU_kw]
local bk_N = e(N)
local bk_KPF = e(widstat)

estimates restore fs3_DU_kw
local bk_fs_b = _b[bartik_iv]
local bk_fs_se = _se[bartik_iv]
local bk_fs_N = e(N)
local bk_fs_r2 = e(r2)
estimates drop fs3_DU_kw

* DWH
preserve
keep if !missing(bartik_iv)
reghdfe DU_kw bartik_iv $controls, absorb(Stkcd_num year_num) resid(_resid_fs3)
reghdfe PriceDelay DU_kw _resid_fs3 $controls, absorb(Stkcd_num year_num) vce(cluster IndYear_num)
local bk_DWH_t = _b[_resid_fs3] / _se[_resid_fs3]
local bk_DWH_F = `bk_DWH_t'^2
local bk_DWH_p = 2 * ttail(e(df_r), abs(`bk_DWH_t'))
restore

di "Bartik 2SLS: b=" %9.6f `bk_b' " se=" %9.6f `bk_se' " N=" `bk_N'
di "KP F=" %8.1f `bk_KPF' " DWH F=" %8.2f `bk_DWH_F' " DWH p=" %6.4f `bk_DWH_p'
di "1st stage: b=" %9.4f `bk_fs_b' " se=" %9.4f `bk_fs_se' " N=" `bk_fs_N' " R2=" %6.3f `bk_fs_r2'

* ==============================================================================
* (4) Lag OLS
* ==============================================================================
di _n "============================================================"
di "Model (4): Lag OLS"
di "============================================================"

reghdfe PriceDelay DU_kw_lag $controls, absorb(Stkcd_num year_num) vce(cluster IndYear_num)
local lag_ols_b = _b[DU_kw_lag]
local lag_ols_se = _se[DU_kw_lag]
local lag_ols_N = e(N)
di "Lag OLS: b=" %9.6f `lag_ols_b' " se=" %9.6f `lag_ols_se' " N=" `lag_ols_N'

* ==============================================================================
* (5) Lag IV
* ==============================================================================
di _n "============================================================"
di "Model (5): Lag IV (ivreghdfe)"
di "============================================================"

ivreghdfe PriceDelay $controls (DU_kw_lag = peer_lag), absorb(Stkcd_num year_num) cluster(IndYear_num) first savefirst savefprefix(fs5_)
local lag_iv_b = _b[DU_kw_lag]
local lag_iv_se = _se[DU_kw_lag]
local lag_iv_N = e(N)
local lag_iv_KPF = e(widstat)

estimates restore fs5_DU_kw_lag
local lag_fs_b = _b[peer_lag]
local lag_fs_se = _se[peer_lag]
local lag_fs_N = e(N)
local lag_fs_r2 = e(r2)
estimates drop fs5_DU_kw_lag

* DWH
preserve
keep if !missing(peer_lag)
reghdfe DU_kw_lag peer_lag $controls, absorb(Stkcd_num year_num) resid(_resid_fs5)
reghdfe PriceDelay DU_kw_lag _resid_fs5 $controls, absorb(Stkcd_num year_num) vce(cluster IndYear_num)
local lag_iv_DWH_t = _b[_resid_fs5] / _se[_resid_fs5]
local lag_iv_DWH_F = `lag_iv_DWH_t'^2
local lag_iv_DWH_p = 2 * ttail(e(df_r), abs(`lag_iv_DWH_t'))
restore

di "Lag IV 2SLS: b=" %9.6f `lag_iv_b' " se=" %9.6f `lag_iv_se' " N=" `lag_iv_N'
di "KP F=" %8.1f `lag_iv_KPF' " DWH F=" %8.2f `lag_iv_DWH_F' " DWH p=" %6.4f `lag_iv_DWH_p'
di "1st stage: b=" %9.4f `lag_fs_b' " se=" %9.4f `lag_fs_se' " N=" `lag_fs_N' " R2=" %6.3f `lag_fs_r2'

* ==============================================================================
* Summary
* ==============================================================================
di _n "================================================================"
di "SUMMARY: Endogeneity Test Results (ivreghdfe, unified samples)"
di "================================================================"
di ""
di "Model (1) OLS:       b=" %9.4f `ols_b'     " se=" %9.4f `ols_se'     " N=" %7.0f `ols_N'   " R2=" %6.3f `ols_r2'
di "Model (2) Peer IV:   b=" %9.4f `peer_b'    " se=" %9.4f `peer_se'    " N=" %7.0f `peer_N'   " KP_F=" %8.1f `peer_KPF'  " DWH_p=" %6.3f `peer_DWH_p'
di "  1st stage:         b=" %9.4f `peer_fs_b'  " se=" %9.4f `peer_fs_se'  " N=" %7.0f `peer_fs_N' " R2=" %6.3f `peer_fs_r2'
di "Model (3) Bartik IV: b=" %9.4f `bk_b'      " se=" %9.4f `bk_se'      " N=" %7.0f `bk_N'     " KP_F=" %8.1f `bk_KPF'    " DWH_p=" %6.3f `bk_DWH_p'
di "  1st stage:         b=" %9.4f `bk_fs_b'    " se=" %9.4f `bk_fs_se'    " N=" %7.0f `bk_fs_N'   " R2=" %6.3f `bk_fs_r2'
di "Model (4) Lag OLS:   b=" %9.4f `lag_ols_b'  " se=" %9.4f `lag_ols_se'  " N=" %7.0f `lag_ols_N'
di "Model (5) Lag IV:    b=" %9.4f `lag_iv_b'   " se=" %9.4f `lag_iv_se'   " N=" %7.0f `lag_iv_N'  " KP_F=" %8.1f `lag_iv_KPF' " DWH_p=" %6.3f `lag_iv_DWH_p'
di "  1st stage:         b=" %9.4f `lag_fs_b'   " se=" %9.4f `lag_fs_se'   " N=" %7.0f `lag_fs_N'  " R2=" %6.3f `lag_fs_r2'
di ""
di "KEY CHECK: 1st stage N must equal 2nd stage N"
di "  Peer:   1st=" `peer_fs_N'  " 2nd=" `peer_N'   " match=" cond(`peer_fs_N'==`peer_N', "YES", "NO")
di "  Bartik: 1st=" `bk_fs_N'    " 2nd=" `bk_N'     " match=" cond(`bk_fs_N'==`bk_N', "YES", "NO")
di "  Lag IV: 1st=" `lag_fs_N'   " 2nd=" `lag_iv_N'  " match=" cond(`lag_fs_N'==`lag_iv_N', "YES", "NO")

* ==============================================================================
* Save results to dta
* ==============================================================================
clear
set obs 5
gen str20 model = ""
gen double coef = .
gen double se = .
gen double N = .
gen double R2 = .
gen double KP_F = .
gen double DWH_p = .
gen double fs_coef = .
gen double fs_se = .
gen double fs_N = .
gen double fs_R2 = .

replace model = "OLS"       in 1
replace coef = `ols_b'      in 1
replace se = `ols_se'        in 1
replace N = `ols_N'          in 1
replace R2 = `ols_r2'        in 1

replace model = "Peer_IV"    in 2
replace coef = `peer_b'      in 2
replace se = `peer_se'        in 2
replace N = `peer_N'          in 2
replace KP_F = `peer_KPF'    in 2
replace DWH_p = `peer_DWH_p' in 2
replace fs_coef = `peer_fs_b' in 2
replace fs_se = `peer_fs_se'  in 2
replace fs_N = `peer_fs_N'    in 2
replace fs_R2 = `peer_fs_r2'  in 2

replace model = "Bartik_IV"   in 3
replace coef = `bk_b'         in 3
replace se = `bk_se'           in 3
replace N = `bk_N'             in 3
replace KP_F = `bk_KPF'       in 3
replace DWH_p = `bk_DWH_p'    in 3
replace fs_coef = `bk_fs_b'    in 3
replace fs_se = `bk_fs_se'     in 3
replace fs_N = `bk_fs_N'       in 3
replace fs_R2 = `bk_fs_r2'     in 3

replace model = "Lag_OLS"     in 4
replace coef = `lag_ols_b'    in 4
replace se = `lag_ols_se'      in 4
replace N = `lag_ols_N'        in 4

replace model = "Lag_IV"       in 5
replace coef = `lag_iv_b'      in 5
replace se = `lag_iv_se'        in 5
replace N = `lag_iv_N'          in 5
replace KP_F = `lag_iv_KPF'    in 5
replace DWH_p = `lag_iv_DWH_p' in 5
replace fs_coef = `lag_fs_b'    in 5
replace fs_se = `lag_fs_se'     in 5
replace fs_N = `lag_fs_N'       in 5
replace fs_R2 = `lag_fs_r2'     in 5

save "`OUTDIR'/ivreghdfe_results.dta", replace
list, sep(0) noobs
