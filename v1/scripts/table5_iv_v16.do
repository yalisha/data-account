* ============================================================
* 表5 内生性检验 v16
* IV1: 同行业跨省留一均值 (Cross-Province Industry Peer Mean)
* IV2: 省级大数据发展指数(2016) × year趋势
* 补充: 滞后OLS, 滞后IV, Oster bounds, 简化型
*
* 数据: reg_sample_iv_v16.dta
* ============================================================

clear all
set more off
set matsize 11000

use "/Users/mac/computerscience/15会计研究/data_stata/reg_sample_iv_v16.dta", clear

* 控制变量 (11个, v16规格)
local controls Size Lev ROA TobinQ Age Growth IndepRatio Dual Top1Share SOE CFO

* 清理样本
drop if missing(Ind2_num)
drop if missing(ProvDigital2016)
count

* ============================================================
* 构造工具变量
* ============================================================

* --- IV1: 同行业跨省留一均值 ---
* 行业-年份总和
bysort Ind2_num year_num: egen iy_sum = total(DU_kw)
bysort Ind2_num year_num: egen iy_n = count(DU_kw)

* 行业-年份-省份总和
bysort Ind2_num year_num Prov_num: egen iyp_sum = total(DU_kw)
bysort Ind2_num year_num Prov_num: egen iyp_n = count(DU_kw)

* 跨省均值 = (行业年总和 - 本省行业年总和) / (行业年计数 - 本省行业年计数)
gen double other_sum = iy_sum - iyp_sum
gen double other_n = iy_n - iyp_n
gen double IV_peer_xprov = other_sum / other_n if other_n > 0
label var IV_peer_xprov "同行业跨省留一均值"

summ IV_peer_xprov, detail
di "IV1 有效观测: " r(N)

* --- IV2: 省级大数据指数(2016) × (year - 2011) ---
gen double IV_prov_trend = ProvDigital2016 * (year - 2011)
label var IV_prov_trend "省级数字化指数(2016)×年份趋势"

summ IV_prov_trend, detail

* --- 滞后变量 ---
xtset Stkcd_num year
gen DU_kw_lag = L.DU_kw

* 滞后的跨省peer
bysort Ind2_num year_num: egen iyl_sum = total(DU_kw_lag)
bysort Ind2_num year_num: egen iyl_n = count(DU_kw_lag)
bysort Ind2_num year_num Prov_num: egen iylp_sum = total(DU_kw_lag)
bysort Ind2_num year_num Prov_num: egen iylp_n = count(DU_kw_lag)
gen double IV_peer_lag_xprov = (iyl_sum - iylp_sum) / (iyl_n - iylp_n) if (iyl_n - iylp_n) > 0

* ============================================================
* (1) OLS 基准
* ============================================================
di _n "=========================================="
di "MODEL (1): OLS Baseline"
di "=========================================="
eststo clear
eststo ols: reghdfe PriceDelay DU_kw `controls', absorb(Stkcd_num year_num) vce(cluster IndYear_num)

* ============================================================
* (2) IV1: 同行业跨省留一均值 — 手动2SLS
* ============================================================
di _n "=========================================="
di "MODEL (2): IV1 - Cross-Province Industry Peer Mean"
di "=========================================="

preserve
drop if missing(IV_peer_xprov)
count

* 第一阶段
eststo iv1_fs: reghdfe DU_kw IV_peer_xprov `controls', absorb(Stkcd_num year_num) vce(cluster IndYear_num) resid
di "First stage coef = " _b[IV_peer_xprov] ", se = " _se[IV_peer_xprov]
local fs1_t = _b[IV_peer_xprov]/_se[IV_peer_xprov]
local fs1_F = `fs1_t'^2
di "First stage t = `fs1_t', F = `fs1_F'"

predict DU_kw_resid1, resid
gen DU_kw_hat1 = DU_kw - DU_kw_resid1

* 第二阶段
eststo iv1_ss: reghdfe PriceDelay DU_kw_hat1 `controls', absorb(Stkcd_num year_num) vce(cluster IndYear_num)
di "Second stage coef = " _b[DU_kw_hat1] ", se = " _se[DU_kw_hat1]

* DWH检验
reghdfe PriceDelay DU_kw DU_kw_resid1 `controls', absorb(Stkcd_num year_num) vce(cluster IndYear_num)
local dwh1_t = _b[DU_kw_resid1]/_se[DU_kw_resid1]
local dwh1_p = 2*ttail(e(df_r), abs(`dwh1_t'))
di "DWH test: resid t = `dwh1_t', p = `dwh1_p'"

restore

* ============================================================
* (3) IV2: 省级大数据指数(2016) × year趋势 — 手动2SLS
* ============================================================
di _n "=========================================="
di "MODEL (3): IV2 - Province Digital Index (2016) x Year Trend"
di "=========================================="

preserve
drop if missing(IV_prov_trend)
count

* 第一阶段
eststo iv2_fs: reghdfe DU_kw IV_prov_trend `controls', absorb(Stkcd_num year_num) vce(cluster IndYear_num) resid
di "First stage coef = " _b[IV_prov_trend] ", se = " _se[IV_prov_trend]
local fs2_t = _b[IV_prov_trend]/_se[IV_prov_trend]
local fs2_F = `fs2_t'^2
di "First stage t = `fs2_t', F = `fs2_F'"

predict DU_kw_resid2, resid
gen DU_kw_hat2 = DU_kw - DU_kw_resid2

* 第二阶段
eststo iv2_ss: reghdfe PriceDelay DU_kw_hat2 `controls', absorb(Stkcd_num year_num) vce(cluster IndYear_num)
di "Second stage coef = " _b[DU_kw_hat2] ", se = " _se[DU_kw_hat2]

* DWH检验
reghdfe PriceDelay DU_kw DU_kw_resid2 `controls', absorb(Stkcd_num year_num) vce(cluster IndYear_num)
local dwh2_t = _b[DU_kw_resid2]/_se[DU_kw_resid2]
local dwh2_p = 2*ttail(e(df_r), abs(`dwh2_t'))
di "DWH test: resid t = `dwh2_t', p = `dwh2_p'"

restore

* ============================================================
* (4) 滞后 OLS
* ============================================================
di _n "=========================================="
di "MODEL (4): Lag OLS"
di "=========================================="

preserve
drop if missing(DU_kw_lag)
eststo lag_ols: reghdfe PriceDelay DU_kw_lag `controls', absorb(Stkcd_num year_num) vce(cluster IndYear_num)
restore

* ============================================================
* (5) 滞后 IV (滞后跨省peer作为IV)
* ============================================================
di _n "=========================================="
di "MODEL (5): Lag IV - Lag Cross-Province Peer"
di "=========================================="

preserve
drop if missing(DU_kw_lag) | missing(IV_peer_lag_xprov)
count

eststo iv3_fs: reghdfe DU_kw_lag IV_peer_lag_xprov `controls', absorb(Stkcd_num year_num) vce(cluster IndYear_num) resid
local fs3_t = _b[IV_peer_lag_xprov]/_se[IV_peer_lag_xprov]
local fs3_F = `fs3_t'^2
di "First stage t = `fs3_t', F = `fs3_F'"

predict DUlag_resid, resid
gen DUlag_hat = DU_kw_lag - DUlag_resid

eststo iv3_ss: reghdfe PriceDelay DUlag_hat `controls', absorb(Stkcd_num year_num) vce(cluster IndYear_num)
di "Second stage coef = " _b[DUlag_hat] ", se = " _se[DUlag_hat]

* DWH
reghdfe PriceDelay DU_kw_lag DUlag_resid `controls', absorb(Stkcd_num year_num) vce(cluster IndYear_num)
local dwh3_t = _b[DUlag_resid]/_se[DUlag_resid]
local dwh3_p = 2*ttail(e(df_r), abs(`dwh3_t'))
di "DWH test: resid t = `dwh3_t', p = `dwh3_p'"

restore

* ============================================================
* (6) 简化型回归 (Reduced Form)
* ============================================================
di _n "=========================================="
di "Reduced Form: IV -> PriceDelay"
di "=========================================="

* IV1 reduced form
preserve
drop if missing(IV_peer_xprov)
reghdfe PriceDelay IV_peer_xprov `controls', absorb(Stkcd_num year_num) vce(cluster IndYear_num)
di "RF IV1: coef = " _b[IV_peer_xprov] ", t = " _b[IV_peer_xprov]/_se[IV_peer_xprov]
restore

* IV2 reduced form
preserve
drop if missing(IV_prov_trend)
reghdfe PriceDelay IV_prov_trend `controls', absorb(Stkcd_num year_num) vce(cluster IndYear_num)
di "RF IV2: coef = " _b[IV_prov_trend] ", t = " _b[IV_prov_trend]/_se[IV_prov_trend]
restore

* ============================================================
* (7) Oster (2019) bounds
* ============================================================
di _n "=========================================="
di "Oster (2019) bounds"
di "=========================================="

* Short regression (no controls)
quietly reghdfe PriceDelay DU_kw, absorb(Stkcd_num year_num) vce(cluster IndYear_num)
local beta_short = _b[DU_kw]
local r2_short = e(r2)
di "Short: beta = `beta_short', R2 = `r2_short'"

* Long regression (with controls)
quietly reghdfe PriceDelay DU_kw `controls', absorb(Stkcd_num year_num) vce(cluster IndYear_num)
local beta_long = _b[DU_kw]
local r2_long = e(r2)
di "Long:  beta = `beta_long', R2 = `r2_long'"

* delta* = beta_long * (R2_max - R2_long) / ((beta_short - beta_long) * (R2_long - R2_short))
* R_max = 1.3 * R2_long
local r2_max = min(1.3 * `r2_long', 1)
local denom = (`beta_short' - `beta_long') * (`r2_long' - `r2_short')
if abs(`denom') > 1e-12 {
    local delta_star = `beta_long' * (`r2_max' - `r2_long') / `denom'
}
else {
    local delta_star = 999
}
di "Oster delta* (Rmax=1.3R2): `delta_star'"

* Conservative: R_max = min(2*R2_long - R2_short, 1)
local r2_max2 = min(2*`r2_long' - `r2_short', 1)
local denom2 = (`beta_short' - `beta_long') * (`r2_long' - `r2_short')
if abs(`denom2') > 1e-12 {
    local delta_star2 = `beta_long' * (`r2_max2' - `r2_long') / `denom2'
}
else {
    local delta_star2 = 999
}
di "Oster delta* (Rmax=2R2-R2s): `delta_star2'"

* ============================================================
* 汇总输出
* ============================================================
di _n "=========================================="
di "SUMMARY: Second Stage Results"
di "=========================================="
esttab ols iv1_ss iv2_ss lag_ols iv3_ss, ///
    b(4) se(4) ///
    star(* 0.10 ** 0.05 *** 0.01) ///
    keep(DU_kw DU_kw_hat1 DU_kw_hat2 DU_kw_lag DUlag_hat) ///
    stats(N r2, fmt(%12.0fc %9.4f) labels("N" "R-squared")) ///
    mtitles("(1) OLS" "(2) IV1:XProv" "(3) IV2:ProvDig" "(4) LagOLS" "(5) LagIV") ///
    title("Table 5: Endogeneity Tests v16")

di _n "=========================================="
di "SUMMARY: First Stage Results"
di "=========================================="
esttab iv1_fs iv2_fs iv3_fs, ///
    b(4) se(4) ///
    star(* 0.10 ** 0.05 *** 0.01) ///
    keep(IV_peer_xprov IV_prov_trend IV_peer_lag_xprov) ///
    stats(N r2, fmt(%12.0fc %9.4f) labels("N" "R-squared")) ///
    mtitles("(2) FS:XProv" "(3) FS:ProvDig" "(5) FS:LagXProv") ///
    title("Table 5: First Stage v16")

* LaTeX输出
cap mkdir "/Users/mac/computerscience/15会计研究/results/v16_tables"

esttab ols iv1_ss iv2_ss lag_ols iv3_ss using "/Users/mac/computerscience/15会计研究/results/v16_tables/table5_2nd_v16.tex", replace ///
    b(4) se(4) ///
    star(* 0.10 ** 0.05 *** 0.01) ///
    keep(DU_kw DU_kw_hat1 DU_kw_hat2 DU_kw_lag DUlag_hat) ///
    stats(N r2, fmt(%12.0fc %9.4f) labels("$N$" "$R^2$")) ///
    mtitles("(1) OLS" "(2) IV1" "(3) IV2" "(4) LagOLS" "(5) LagIV") ///
    booktabs compress

esttab iv1_fs iv2_fs iv3_fs using "/Users/mac/computerscience/15会计研究/results/v16_tables/table5_1st_v16.tex", replace ///
    b(4) se(4) ///
    star(* 0.10 ** 0.05 *** 0.01) ///
    keep(IV_peer_xprov IV_prov_trend IV_peer_lag_xprov) ///
    stats(N r2, fmt(%12.0fc %9.4f) labels("$N$" "$R^2$")) ///
    mtitles("(2) FS:XProv" "(3) FS:ProvDig" "(5) FS:LagXProv") ///
    booktabs compress

di _n "=========================================="
di "DONE: table5_iv_v16.do"
di "=========================================="
