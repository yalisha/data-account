* ============================================================
* Heckman两阶段检验 (朱康2025方法)
* 第一阶段: Probit DU_kw_high ~ controls
* 第二阶段: 加入IMR后回归
* ============================================================

clear all
set more off

use "/Users/mac/computerscience/15会计研究/data_stata/reg_sample_v5.dta", clear

* 定义13个控制变量
local controls Size Lev ROA TobinQ Age Growth IndepRatio Dual Top1Share SOE CFO Inv Hhi

* 统一样本
drop if missing(Ind2_num)
count

* ---- Heckman第一阶段: Probit ----
di _n "=========================================="
di "Heckman Stage 1: Probit"
di "=========================================="

probit DU_kw_high `controls', nolog
predict imr_val, xb
* 计算逆米尔斯比率
gen phi_val = normalden(imr_val)
gen Phi_val = normal(imr_val)
gen IMR = phi_val / Phi_val if DU_kw_high == 1
replace IMR = -phi_val / (1 - Phi_val) if DU_kw_high == 0

* ---- Heckman第二阶段: 加入IMR ----
di _n "=========================================="
di "Heckman Stage 2: reghdfe with IMR"
di "=========================================="

eststo clear
eststo heckman: reghdfe PriceDelay DU_kw IMR `controls', absorb(Stkcd_num year_num) vce(cluster IndYear_num)

esttab heckman, ///
    b(4) se(4) ///
    star(* 0.10 ** 0.05 *** 0.01) ///
    keep(DU_kw IMR) ///
    stats(N r2, fmt(%12.0fc %9.3f) labels("N" "R-squared")) ///
    mtitles("Heckman") ///
    title("Heckman Two-Stage Test")

di _n "=========================================="
di "DONE: Heckman test"
di "=========================================="
