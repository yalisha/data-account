* ============================================================
* 表5 机制检验 (新版)
* 渠道1: Analyst (信息中介)
* 渠道2: Zeros (信息不对称/交易摩擦)
* 渠道3: absDA (会计信息质量)
* ============================================================

clear all
set more off

use "/Users/mac/computerscience/15会计研究/data_stata/reg_sample_v4.dta", clear

* ============================================================
* 渠道1: DU_kw -> Analyst (去掉Analyst本身作为控制)
* ============================================================
di _n "===== 渠道1: Analyst (信息中介) ====="
reghdfe Analyst DU_kw Size Lev ROA TobinQ Age Growth BoardSize IndepRatio ///
    Dual Top1Share SOE InstHold Amihud AuditType, ///
    absorb(Stkcd_num year_num) vce(cluster IndYear_num)
est store m1

* ============================================================
* 渠道2: DU_kw -> Zeros (信息不对称)
* ============================================================
di _n "===== 渠道2: Zeros (信息不对称) ====="
reghdfe Zeros DU_kw Size Lev ROA TobinQ Age Growth BoardSize IndepRatio ///
    Dual Top1Share SOE InstHold Amihud Analyst AuditType, ///
    absorb(Stkcd_num year_num) vce(cluster IndYear_num)
est store m2

* ============================================================
* 渠道3: DU_kw -> absDA (会计信息质量)
* ============================================================
di _n "===== 渠道3: absDA (会计信息质量) ====="
reghdfe absDA DU_kw Size Lev ROA TobinQ Age Growth BoardSize IndepRatio ///
    Dual Top1Share SOE InstHold Amihud Analyst AuditType, ///
    absorb(Stkcd_num year_num) vce(cluster IndYear_num)
est store m3

* ============================================================
* 输出结果
* ============================================================
esttab m1 m2 m3, se star(* 0.10 ** 0.05 *** 0.01) ///
    b(%9.4f) se(%9.4f) ///
    keep(DU_kw) ///
    scalars(N r2) ///
    mtitles("Analyst" "Zeros" "absDA") ///
    title("表5 机制检验")

* 完整结果
esttab m1 m2 m3, se star(* 0.10 ** 0.05 *** 0.01) ///
    b(%9.4f) se(%9.4f) ///
    scalars(N r2) ///
    mtitles("Analyst" "Zeros" "absDA") ///
    title("表5 机制检验 (完整)")
