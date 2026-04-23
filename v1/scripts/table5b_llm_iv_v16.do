* ============================================================
* v16 DU_llm 工具变量估计
* IV1: 同行业跨省留一均值 (for DU_llm)
* IV2: 省级大数据发展指数(2016) × year趋势
* ============================================================

clear all
set more off

use "/Users/mac/computerscience/15会计研究/data_stata/reg_sample_iv_v16.dta", clear

local controls Size Lev ROA TobinQ Age Growth IndepRatio Dual Top1Share SOE CFO

drop if missing(Ind2_num)
drop if missing(ProvDigital2016)
drop if missing(DU_llm)

xtset Stkcd_num year
count
di "DU_llm样本: " r(N)

summ DU_llm, detail
summ llm_score, detail

* ============================================================
* 构造IV for DU_llm
* ============================================================

* IV1: 同行业跨省留一均值 (DU_llm)
bysort Ind2_num year_num: egen llm_iy_sum = total(DU_llm)
bysort Ind2_num year_num: egen llm_iy_n = count(DU_llm)
bysort Ind2_num year_num Prov_num: egen llm_iyp_sum = total(DU_llm)
bysort Ind2_num year_num Prov_num: egen llm_iyp_n = count(DU_llm)
gen double IV_llm_peer = (llm_iy_sum - llm_iyp_sum) / (llm_iy_n - llm_iyp_n) if (llm_iy_n - llm_iyp_n) > 0

* IV2: 省级数字化 × year (same as before)
gen double IV_prov_trend = ProvDigital2016 * (year - 2011)

* ============================================================
* (1) OLS: DU_llm → PriceDelay
* ============================================================
di _n "=========================================="
di "(1) OLS: DU_llm -> PriceDelay"
di "=========================================="

eststo llm_ols: reghdfe PriceDelay DU_llm `controls', absorb(Stkcd_num year_num) vce(cluster IndYear_num)
di "DU_llm OLS: coef=" _b[DU_llm] ", t=" _b[DU_llm]/_se[DU_llm]

* ============================================================
* (2) IV1: 跨省peer → DU_llm → PriceDelay
* ============================================================
di _n "=========================================="
di "(2) IV1: Cross-Province Peer for DU_llm"
di "=========================================="

preserve
drop if missing(IV_llm_peer)

* First stage
reghdfe DU_llm IV_llm_peer `controls', absorb(Stkcd_num year_num) vce(cluster IndYear_num) resid
local fs1_t = _b[IV_llm_peer]/_se[IV_llm_peer]
local fs1_F = `fs1_t'^2
di "First stage: coef=" _b[IV_llm_peer] ", t=`fs1_t', F=`fs1_F'"

predict llm_resid1, resid
gen llm_hat1 = DU_llm - llm_resid1

* Second stage
eststo llm_iv1: reghdfe PriceDelay llm_hat1 `controls', absorb(Stkcd_num year_num) vce(cluster IndYear_num)
di "IV1 second stage: coef=" _b[llm_hat1] ", se=" _se[llm_hat1] ", t=" _b[llm_hat1]/_se[llm_hat1]

* DWH
reghdfe PriceDelay DU_llm llm_resid1 `controls', absorb(Stkcd_num year_num) vce(cluster IndYear_num)
local dwh1_t = _b[llm_resid1]/_se[llm_resid1]
local dwh1_p = 2*ttail(e(df_r), abs(`dwh1_t'))
di "DWH: t=`dwh1_t', p=`dwh1_p'"

restore

* ============================================================
* (3) IV2: 省级数字化 → DU_llm → PriceDelay
* ============================================================
di _n "=========================================="
di "(3) IV2: Province Digital Index for DU_llm"
di "=========================================="

preserve
drop if missing(IV_prov_trend)

reghdfe DU_llm IV_prov_trend `controls', absorb(Stkcd_num year_num) vce(cluster IndYear_num) resid
local fs2_t = _b[IV_prov_trend]/_se[IV_prov_trend]
local fs2_F = `fs2_t'^2
di "First stage: coef=" _b[IV_prov_trend] ", t=`fs2_t', F=`fs2_F'"

predict llm_resid2, resid
gen llm_hat2 = DU_llm - llm_resid2

eststo llm_iv2: reghdfe PriceDelay llm_hat2 `controls', absorb(Stkcd_num year_num) vce(cluster IndYear_num)
di "IV2 second stage: coef=" _b[llm_hat2] ", se=" _se[llm_hat2] ", t=" _b[llm_hat2]/_se[llm_hat2]

* DWH
reghdfe PriceDelay DU_llm llm_resid2 `controls', absorb(Stkcd_num year_num) vce(cluster IndYear_num)
local dwh2_t = _b[llm_resid2]/_se[llm_resid2]
local dwh2_p = 2*ttail(e(df_r), abs(`dwh2_t'))
di "DWH: t=`dwh2_t', p=`dwh2_p'"

restore

* ============================================================
* (4) OLS + IV for llm_binary (二值化)
* ============================================================
di _n "=========================================="
di "(4) llm_binary OLS + IV"
di "=========================================="

* OLS
eststo bin_ols: reghdfe PriceDelay llm_binary `controls', absorb(Stkcd_num year_num) vce(cluster IndYear_num)
di "llm_binary OLS: coef=" _b[llm_binary] ", t=" _b[llm_binary]/_se[llm_binary]

* IV1 for llm_binary
bysort Ind2_num year_num: egen bin_iy_sum = total(llm_binary)
bysort Ind2_num year_num: egen bin_iy_n = count(llm_binary)
bysort Ind2_num year_num Prov_num: egen bin_iyp_sum = total(llm_binary)
bysort Ind2_num year_num Prov_num: egen bin_iyp_n = count(llm_binary)
gen double IV_bin_peer = (bin_iy_sum - bin_iyp_sum) / (bin_iy_n - bin_iyp_n) if (bin_iy_n - bin_iyp_n) > 0

preserve
drop if missing(IV_bin_peer)

reghdfe llm_binary IV_bin_peer `controls', absorb(Stkcd_num year_num) vce(cluster IndYear_num) resid
local fs_bin_t = _b[IV_bin_peer]/_se[IV_bin_peer]
local fs_bin_F = `fs_bin_t'^2
di "llm_binary First stage: t=`fs_bin_t', F=`fs_bin_F'"

predict bin_resid, resid
gen bin_hat = llm_binary - bin_resid

eststo bin_iv: reghdfe PriceDelay bin_hat `controls', absorb(Stkcd_num year_num) vce(cluster IndYear_num)
di "llm_binary IV: coef=" _b[bin_hat] ", t=" _b[bin_hat]/_se[bin_hat]

restore

* ============================================================
* (5) SYNCH as DV
* ============================================================
di _n "=========================================="
di "(5) DU_llm -> SYNCH"
di "=========================================="

preserve
drop if missing(SYNCH)

eststo llm_synch: reghdfe SYNCH DU_llm `controls', absorb(Stkcd_num year_num) vce(cluster IndYear_num)
di "DU_llm -> SYNCH OLS: coef=" _b[DU_llm] ", t=" _b[DU_llm]/_se[DU_llm]

restore

* ============================================================
* 汇总
* ============================================================
di _n "=========================================="
di "SUMMARY: DU_llm IV Results"
di "=========================================="

esttab llm_ols llm_iv1 llm_iv2 bin_ols bin_iv, ///
    b(4) se(4) ///
    star(* 0.10 ** 0.05 *** 0.01) ///
    keep(DU_llm llm_hat1 llm_hat2 llm_binary bin_hat) ///
    stats(N r2, fmt(%12.0fc %9.4f) labels("N" "R-squared")) ///
    mtitles("(1) LLM OLS" "(2) LLM IV1" "(3) LLM IV2" "(4) Bin OLS" "(5) Bin IV") ///
    title("DU_llm Endogeneity Tests")

di _n "=========================================="
di "DONE"
di "=========================================="
