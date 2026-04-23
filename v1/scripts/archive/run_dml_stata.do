* ============================================================
* DML稳健性检验 — Stata ddml
*
* 模型: PriceDelay = theta * DU_kw + g(X) + u
* 方法: 部分线性模型 (PLM), Stacking (Lasso+RF+GradBoost)
* FE处理: CRE (Mundlak关联随机效应)
* 参考: Chernozhukov et al. (2018); Ahrens et al. (2024 SJ)
* ============================================================

clear all
set more off
set seed 42

* ============================================================
* 0. 安装 (首次运行取消注释)
* ============================================================
* ssc install ddml, replace
* ssc install pystacked, replace
* ssc install reghdfe, replace
* ssc install ftools, replace

* ============================================================
* 1. 加载数据
* ============================================================
use "/Users/mac/computerscience/15会计研究/data_stata/panel_dml.dta", clear

describe, short
count

* ============================================================
* 2. 定义变量
* ============================================================
* 因变量
global Y PriceDelay

* 处理变量
global D DU_kw

* 原始控制变量 (15个)
global controls_old "Size Lev ROA TobinQ Age Growth BoardSize IndepRatio Dual Top1Share SOE InstHold Amihud Analyst AuditType"

* 新增控制变量 (13个, 不含Roastd避免样本损失)
global controls_new "CFO RetVol Turnover Intangible PPE BM Employee ShareholderNum Manhold Opinion Balance Separation Market"

* 全部控制变量 (28个)
global X $controls_old $controls_new

* 删除缺失
foreach v of varlist $Y $D $X {
    drop if missing(`v')
}
count
di "回归样本: `r(N)' obs"

* ============================================================
* 3. OLS基准 (对比用)
* ============================================================
di _n "============================================================"
di "OLS基准回归 (15个控制变量)"
di "============================================================"

* 3a. 原始控制变量 OLS
reghdfe $Y $D $controls_old, absorb(Stkcd_num year) vce(cluster IndYear_num)
est store ols_base
di "OLS基准: coef = " _b[$D] " se = " _se[$D]

* 3b. 扩展控制变量 OLS
di _n "OLS扩展 (28个控制变量)"
reghdfe $Y $D $X, absorb(Stkcd_num year) vce(cluster IndYear_num)
est store ols_ext
di "OLS扩展: coef = " _b[$D] " se = " _se[$D]

* ============================================================
* 4. CRE变量构造 (Mundlak装置)
* ============================================================
di _n "============================================================"
di "构造CRE变量 (企业均值 + 年份虚拟变量)"
di "============================================================"

* 企业层面均值
foreach v of varlist $X {
    capture drop fm_`v'
    bysort Stkcd_num: egen fm_`v' = mean(`v')
}

* 年份虚拟变量
tab year, gen(yr_)

* CRE扩展控制集
unab fm_vars: fm_*
unab yr_vars: yr_*
global X_CRE $X `fm_vars' `yr_vars'

di "原始控制变量数: " `: word count $X'
di "CRE扩展后变量数: " `: word count $X_CRE'

* ============================================================
* 5. DML — CRE方法 (推荐)
* ============================================================
di _n "============================================================"
di "DML-CRE: 部分线性模型, Stacking"
di "============================================================"

* 初始化: 5折交叉验证, 5次重复, 按企业聚类分折
ddml init partial, kfolds(5) reps(5) fcluster(Stkcd_num)

* E[Y|X] 的学习器: Stacking
ddml E[Y|X]: pystacked $Y $X_CRE || ///
    method(lassocv) || ///
    method(ridgecv) || ///
    method(rf) opt(n_estimators(500) max_depth(6) min_samples_leaf(10)) || ///
    method(gradboost) opt(n_estimators(500) learning_rate(0.05) max_depth(4)), ///
    type(reg)

* E[D|X] 的学习器: Stacking
ddml E[D|X]: pystacked $D $X_CRE || ///
    method(lassocv) || ///
    method(ridgecv) || ///
    method(rf) opt(n_estimators(500) max_depth(6) min_samples_leaf(10)) || ///
    method(gradboost) opt(n_estimators(500) learning_rate(0.05) max_depth(4)), ///
    type(reg)

* 交叉拟合
ddml crossfit

* 估计 (行业x年份聚类标准误)
ddml estimate, cluster(IndYear_num)

* 保存结果
est store dml_cre

* 查看详细信息
ddml describe

* ============================================================
* 6. DML — 去均值方法 (稳健性)
* ============================================================
di _n "============================================================"
di "DML-Demean: 先去均值再DML (稳健性)"
di "============================================================"

* 去均值
foreach v of varlist $Y $D $X {
    capture drop dm_`v'
    qui reghdfe `v', absorb(Stkcd_num year) residuals(dm_`v')
}

unab dm_controls: dm_Size dm_Lev dm_ROA dm_TobinQ dm_Age dm_Growth ///
    dm_BoardSize dm_IndepRatio dm_Dual dm_Top1Share dm_SOE dm_InstHold ///
    dm_Amihud dm_Analyst dm_AuditType ///
    dm_CFO dm_RetVol dm_Turnover dm_Intangible dm_PPE dm_BM ///
    dm_Employee dm_ShareholderNum dm_Manhold dm_Opinion ///
    dm_Balance dm_Separation dm_Market

ddml init partial, kfolds(5) reps(5) fcluster(Stkcd_num) mname(m_dm)

ddml E[Y|X], mname(m_dm): pystacked dm_$Y `dm_controls' || ///
    method(lassocv) || ///
    method(ridgecv) || ///
    method(rf) opt(n_estimators(500) max_depth(6)) || ///
    method(gradboost) opt(n_estimators(500) learning_rate(0.05)), ///
    type(reg)

ddml E[D|X], mname(m_dm): pystacked dm_$D `dm_controls' || ///
    method(lassocv) || ///
    method(ridgecv) || ///
    method(rf) opt(n_estimators(500) max_depth(6)) || ///
    method(gradboost) opt(n_estimators(500) learning_rate(0.05)), ///
    type(reg)

ddml crossfit, mname(m_dm)
ddml estimate, mname(m_dm) cluster(IndYear_num)
est store dml_dm

* ============================================================
* 7. 结果对比
* ============================================================
di _n "============================================================"
di "结果汇总"
di "============================================================"

esttab ols_base ols_ext dml_cre dml_dm, ///
    b(4) se(4) ///
    star(* 0.10 ** 0.05 *** 0.01) ///
    keep($D) ///
    stats(N, fmt(%12.0fc) labels("N")) ///
    mtitles("OLS-15X" "OLS-28X" "DML-CRE" "DML-Demean") ///
    title("DML vs OLS: DU_kw -> PriceDelay")

* 保存结果
esttab ols_base ols_ext dml_cre dml_dm ///
    using "/Users/mac/computerscience/15会计研究/results/v12_tables/dml_results.tex", replace ///
    b(4) se(4) ///
    star(* 0.10 ** 0.05 *** 0.01) ///
    keep($D) ///
    stats(N, fmt(%12.0fc) labels("\$N\$")) ///
    mtitles("OLS" "OLS-Ext" "DML-CRE" "DML-Demean") ///
    booktabs alignment(S)

di _n "Done! Results saved."
