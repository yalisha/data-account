* ============================================================
* 表5 内生性检验 v14 — 27个控制变量
* 使用手动2SLS (reghdfe) 避免ivreghdfe版本冲突
*
* (1) OLS基准
* (2) 同行业均值IV (Peer Mean)
* (3) Bartik移位份额IV (Shift-Share)
* (4) 滞后OLS
* (5) 滞后IV (Lag Peer)
*
* 数据: reg_sample_v5.dta (27 controls)
* ============================================================

clear all
set more off
set matsize 11000

use "/Users/mac/computerscience/15会计研究/data_stata/reg_sample_v5.dta", clear

* 定义13个控制变量 (移除机制渠道+冗余, 加Inv+Hhi)
local controls Size Lev ROA TobinQ Age Growth IndepRatio Dual Top1Share SOE CFO Inv Hhi

* 统一样本: 删除Ind2缺失
drop if missing(Ind2_num)
count

* ============================================================
* 构造工具变量
* ============================================================

* (A) Peer mean IV: leave-one-out industry-year mean
bysort Ind2_num year_num: egen iy_sum = total(DU_kw)
bysort Ind2_num year_num: egen iy_count = count(DU_kw)
gen DU_kw_peer = (iy_sum - DU_kw) / (iy_count - 1) if iy_count > 1

* (B) Bartik IV: shift-share instrument
* base_exposure = industry mean in base year
* shift = national growth excluding own industry
summ year, meanonly
local base_year = r(min)
bysort Ind2_num: egen base_DU_sum = total(DU_kw * (year == `base_year'))
bysort Ind2_num: egen base_DU_n = total(year == `base_year')
gen base_exposure = base_DU_sum / base_DU_n if base_DU_n > 0

bysort year_num: egen nat_sum = total(DU_kw)
bysort year_num: egen nat_n = count(DU_kw)
gen excl_mean = (nat_sum - iy_sum) / (nat_n - iy_count) if (nat_n - iy_count) > 0
gen excl_base_temp = excl_mean if year == `base_year'
bysort Ind2_num: egen excl_base = max(excl_base_temp)
drop excl_base_temp
gen growth_excl = excl_mean / excl_base if excl_base > 0.001
gen bartik_iv = base_exposure * growth_excl

* (C) Lag variables
xtset Stkcd_num year
gen DU_kw_lag = L.DU_kw
bysort Ind2_num year_num: egen iyl_sum = total(DU_kw_lag)
bysort Ind2_num year_num: egen iyl_count = count(DU_kw_lag)
gen peer_lag = (iyl_sum - DU_kw_lag) / (iyl_count - 1) if iyl_count > 1

* ============================================================
* (1) OLS基准
* ============================================================
di _n "=========================================="
di "MODEL (1): OLS Baseline (27 controls)"
di "=========================================="
eststo clear
eststo ols: reghdfe PriceDelay DU_kw `controls', absorb(Stkcd_num year_num) vce(cluster IndYear_num)

* ============================================================
* (2) 同行业均值IV — 手动2SLS
* ============================================================
di _n "=========================================="
di "MODEL (2): Peer Mean IV"
di "=========================================="

preserve
drop if missing(DU_kw_peer)

* 第一阶段: DU_kw ~ DU_kw_peer + controls + FE
eststo iv1_fs: reghdfe DU_kw DU_kw_peer `controls', absorb(Stkcd_num year_num) vce(cluster IndYear_num) resid
di "First stage: coef = " _b[DU_kw_peer] ", se = " _se[DU_kw_peer] ", t = " _b[DU_kw_peer]/_se[DU_kw_peer]
local fs1_F = (_b[DU_kw_peer]/_se[DU_kw_peer])^2
di "KP F (approx) = `fs1_F'"

* 第二阶段: 用拟合值替换
predict DU_kw_resid, resid
gen DU_kw_fitted = DU_kw - DU_kw_resid

eststo iv1_ss: reghdfe PriceDelay DU_kw_fitted `controls', absorb(Stkcd_num year_num) vce(cluster IndYear_num)
di "Second stage: coef = " _b[DU_kw_fitted] ", se = " _se[DU_kw_fitted]

* DWH检验: 将第一阶段残差加入OLS
eststo iv1_dwh: reghdfe PriceDelay DU_kw DU_kw_resid `controls', absorb(Stkcd_num year_num) vce(cluster IndYear_num)
di "DWH: resid coef = " _b[DU_kw_resid] ", t = " _b[DU_kw_resid]/_se[DU_kw_resid] ", p = " 2*ttail(e(df_r), abs(_b[DU_kw_resid]/_se[DU_kw_resid]))

restore

* ============================================================
* (3) Bartik IV — 手动2SLS
* ============================================================
di _n "=========================================="
di "MODEL (3): Bartik IV"
di "=========================================="

preserve
drop if missing(bartik_iv)

eststo iv2_fs: reghdfe DU_kw bartik_iv `controls', absorb(Stkcd_num year_num) vce(cluster IndYear_num) resid
di "First stage: coef = " _b[bartik_iv] ", se = " _se[bartik_iv] ", t = " _b[bartik_iv]/_se[bartik_iv]
local fs2_F = (_b[bartik_iv]/_se[bartik_iv])^2
di "KP F (approx) = `fs2_F'"

predict DU_kw_resid2, resid
gen DU_kw_fitted2 = DU_kw - DU_kw_resid2

eststo iv2_ss: reghdfe PriceDelay DU_kw_fitted2 `controls', absorb(Stkcd_num year_num) vce(cluster IndYear_num)
di "Second stage: coef = " _b[DU_kw_fitted2] ", se = " _se[DU_kw_fitted2]

* DWH
eststo iv2_dwh: reghdfe PriceDelay DU_kw DU_kw_resid2 `controls', absorb(Stkcd_num year_num) vce(cluster IndYear_num)
di "DWH: resid coef = " _b[DU_kw_resid2] ", t = " _b[DU_kw_resid2]/_se[DU_kw_resid2] ", p = " 2*ttail(e(df_r), abs(_b[DU_kw_resid2]/_se[DU_kw_resid2]))

restore

* ============================================================
* (4) 滞后OLS
* ============================================================
di _n "=========================================="
di "MODEL (4): Lag OLS"
di "=========================================="

preserve
drop if missing(DU_kw_lag)
eststo lag_ols: reghdfe PriceDelay DU_kw_lag `controls', absorb(Stkcd_num year_num) vce(cluster IndYear_num)
restore

* ============================================================
* (5) 滞后IV — 手动2SLS
* ============================================================
di _n "=========================================="
di "MODEL (5): Lag IV (peer_lag as IV for DU_kw_lag)"
di "=========================================="

preserve
drop if missing(DU_kw_lag) | missing(peer_lag)

eststo iv3_fs: reghdfe DU_kw_lag peer_lag `controls', absorb(Stkcd_num year_num) vce(cluster IndYear_num) resid
di "First stage: coef = " _b[peer_lag] ", se = " _se[peer_lag] ", t = " _b[peer_lag]/_se[peer_lag]
local fs3_F = (_b[peer_lag]/_se[peer_lag])^2
di "KP F (approx) = `fs3_F'"

predict DUlag_resid, resid
gen DUlag_fitted = DU_kw_lag - DUlag_resid

eststo iv3_ss: reghdfe PriceDelay DUlag_fitted `controls', absorb(Stkcd_num year_num) vce(cluster IndYear_num)
di "Second stage: coef = " _b[DUlag_fitted] ", se = " _se[DUlag_fitted]

* DWH
eststo iv3_dwh: reghdfe PriceDelay DU_kw_lag DUlag_resid `controls', absorb(Stkcd_num year_num) vce(cluster IndYear_num)
di "DWH: resid coef = " _b[DUlag_resid] ", t = " _b[DUlag_resid]/_se[DUlag_resid] ", p = " 2*ttail(e(df_r), abs(_b[DUlag_resid]/_se[DUlag_resid]))

restore

* ============================================================
* 汇总输出
* ============================================================

di _n "=========================================="
di "SUMMARY: Second Stage (27 controls)"
di "=========================================="
esttab ols iv1_ss iv2_ss lag_ols iv3_ss, ///
    b(4) se(4) ///
    star(* 0.10 ** 0.05 *** 0.01) ///
    keep(DU_kw DU_kw_fitted DU_kw_fitted2 DU_kw_lag DUlag_fitted) ///
    stats(N r2, fmt(%12.0fc %9.3f) labels("N" "R-squared")) ///
    mtitles("(1) OLS" "(2) Peer IV" "(3) Bartik IV" "(4) Lag OLS" "(5) Lag IV") ///
    title("Table 5: Second Stage (27 controls, v14)")

di _n "=========================================="
di "SUMMARY: First Stage (27 controls)"
di "=========================================="
esttab iv1_fs iv2_fs iv3_fs, ///
    b(4) se(4) ///
    star(* 0.10 ** 0.05 *** 0.01) ///
    keep(DU_kw_peer bartik_iv peer_lag) ///
    stats(N r2, fmt(%12.0fc %9.3f) labels("N" "R-squared")) ///
    mtitles("(2) First: Peer" "(3) First: Bartik" "(5) First: PeerLag") ///
    title("Table 5: First Stage (27 controls, v14)")

* LaTeX输出
cap mkdir "/Users/mac/computerscience/15会计研究/results/v15_tables"

* Second stage
esttab ols iv1_ss iv2_ss lag_ols iv3_ss using "/Users/mac/computerscience/15会计研究/results/v15_tables/table5_endogeneity_2nd_v14.tex", replace ///
    b(4) se(4) ///
    star(* 0.10 ** 0.05 *** 0.01) ///
    keep(DU_kw DU_kw_fitted DU_kw_fitted2 DU_kw_lag DUlag_fitted) ///
    stats(N r2, fmt(%12.0fc %9.3f) labels("$N$" "$R^2$")) ///
    mtitles("(1) OLS" "(2) Peer IV" "(3) Bartik IV" "(4) Lag OLS" "(5) Lag IV") ///
    title("Table 5 Panel B: Second Stage (27 controls)") ///
    booktabs compress

* First stage
esttab iv1_fs iv2_fs iv3_fs using "/Users/mac/computerscience/15会计研究/results/v15_tables/table5_endogeneity_1st_v14.tex", replace ///
    b(4) se(4) ///
    star(* 0.10 ** 0.05 *** 0.01) ///
    keep(DU_kw_peer bartik_iv peer_lag) ///
    stats(N r2, fmt(%12.0fc %9.3f) labels("$N$" "$R^2$")) ///
    mtitles("(2) First: Peer" "(3) First: Bartik" "(5) First: PeerLag") ///
    title("Table 5 Panel A: First Stage (27 controls)") ///
    booktabs compress

di _n "=========================================="
di "DONE: table5_endogeneity_v14.do"
di "=========================================="
