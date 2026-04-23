* ============================================================
* v16 补充内生性与稳健性检验
*
* (A) Heckman两阶段 (样本选择偏差)
* (B) 安慰剂/前置期检验 (DU_kw_lead → PriceDelay)
* (C) 异质性组间差异检验 (suest)
* (D) 替换自变量 DU_sub_ln
* (E) IV for SYNCH (因变量替换)
*
* 数据: reg_sample_iv_v16.dta
* ============================================================

clear all
set more off
set matsize 11000

use "/Users/mac/computerscience/15会计研究/data_stata/reg_sample_iv_v16.dta", clear

local controls Size Lev ROA TobinQ Age Growth IndepRatio Dual Top1Share SOE CFO

drop if missing(Ind2_num)
drop if missing(ProvDigital2016)

* 面板设定
xtset Stkcd_num year

* 构造辅助变量
cap gen DU_kw_lag = L.DU_kw
gen DU_kw_positive = (DU_kw > 0) if !missing(DU_kw)
* DU_kw_lead已在数据集中 (export_reg_sample时已构造)

* 跨省peer IV (复用table5_iv_v16.do的逻辑)
bysort Ind2_num year_num: egen iy_sum = total(DU_kw)
bysort Ind2_num year_num: egen iy_n = count(DU_kw)
bysort Ind2_num year_num Prov_num: egen iyp_sum = total(DU_kw)
bysort Ind2_num year_num Prov_num: egen iyp_n = count(DU_kw)
gen double IV_peer_xprov = (iy_sum - iyp_sum) / (iy_n - iyp_n) if (iy_n - iyp_n) > 0
gen double IV_prov_trend = ProvDigital2016 * (year - 2011)

count
di "基础样本: " r(N)

* ============================================================
* (A) Heckman两阶段: 样本选择偏差
* ============================================================
di _n "=========================================="
di "(A) HECKMAN TWO-STAGE"
di "=========================================="

* 思路: 第一阶段probit估计"企业是否披露数据要素信息"的概率
* 第二阶段在披露子样本中估计DU_kw对PriceDelay的效应, 加入IMR
*
* 排他性限制条件: 省级数字化指数×year (影响披露概率但不直接影响PriceDelay)
* 同行业跨省均值 (行业披露浪潮影响本企业披露概率)

* 第一阶段: probit
di "--- Heckman Stage 1: Probit ---"
probit DU_kw_positive IV_peer_xprov IV_prov_trend `controls' i.year_num i.Ind2_num, vce(cluster IndYear_num)
di "Probit N = " e(N) ", Pseudo-R2 = " e(r2_p)

* 预测IMR
predict double xb_heck, xb
gen double imr = normalden(xb_heck) / normal(xb_heck) if DU_kw_positive == 1
label var imr "Inverse Mills Ratio"
summ imr, detail

* 第二阶段: 在DU_kw>0子样本中加入IMR
di "--- Heckman Stage 2: OLS with IMR ---"
eststo heck: reghdfe PriceDelay DU_kw imr `controls' if DU_kw_positive == 1, absorb(Stkcd_num year_num) vce(cluster IndYear_num)
di "DU_kw coef = " _b[DU_kw] ", se = " _se[DU_kw] ", t = " _b[DU_kw]/_se[DU_kw]
di "IMR coef = " _b[imr] ", se = " _se[imr] ", t = " _b[imr]/_se[imr]
local imr_t = _b[imr]/_se[imr]
local imr_p = 2*ttail(e(df_r), abs(`imr_t'))
di "IMR显著性: t = `imr_t', p = `imr_p'"

* ============================================================
* (B) 安慰剂检验: 前置期 DU_kw(t+1) → PriceDelay(t)
* ============================================================
di _n "=========================================="
di "(B) PLACEBO: Lead DU_kw -> PriceDelay"
di "=========================================="

* 如果因果方向正确, 未来的DU不应该影响当期PriceDelay
preserve
drop if missing(DU_kw_lead)
count
di "安慰剂样本: " r(N)

eststo placebo: reghdfe PriceDelay DU_kw_lead `controls', absorb(Stkcd_num year_num) vce(cluster IndYear_num)
di "DU_kw_lead coef = " _b[DU_kw_lead] ", se = " _se[DU_kw_lead]
local plac_t = _b[DU_kw_lead]/_se[DU_kw_lead]
local plac_p = 2*ttail(e(df_r), abs(`plac_t'))
di "Lead placebo: t = `plac_t', p = `plac_p'"

* 同时放入当期和前置期
eststo placebo2: reghdfe PriceDelay DU_kw DU_kw_lead `controls', absorb(Stkcd_num year_num) vce(cluster IndYear_num)
di "当期DU_kw coef = " _b[DU_kw] ", t = " _b[DU_kw]/_se[DU_kw]
di "前置DU_kw_lead coef = " _b[DU_kw_lead] ", t = " _b[DU_kw_lead]/_se[DU_kw_lead]

restore

* ============================================================
* (C) 异质性组间差异检验: suest
* ============================================================
di _n "=========================================="
di "(C) HETEROGENEITY GROUP DIFFERENCE TESTS"
di "=========================================="

* --- C1: SOE vs Non-SOE ---
di "--- C1: SOE差异检验 ---"
reghdfe PriceDelay DU_kw `controls' if SOE == 1, absorb(Stkcd_num year_num) vce(cluster IndYear_num)
est store het_soe1
local b_soe1 = _b[DU_kw]

reghdfe PriceDelay DU_kw `controls' if SOE == 0, absorb(Stkcd_num year_num) vce(cluster IndYear_num)
est store het_soe0
local b_soe0 = _b[DU_kw]

* 交互项检验 (更稳健的方法)
reghdfe PriceDelay c.DU_kw##i.SOE `controls', absorb(Stkcd_num year_num) vce(cluster IndYear_num)
local soe_inter_t = _b[1.SOE#c.DU_kw]/_se[1.SOE#c.DU_kw]
local soe_inter_p = 2*ttail(e(df_r), abs(`soe_inter_t'))
di "SOE: 国企β=`b_soe1', 民企β=`b_soe0'"
di "SOE交互项: t = `soe_inter_t', p = `soe_inter_p'"

* --- C2: 规模 (Size中位数分组) ---
di "--- C2: Size差异检验 ---"
summ Size, detail
local size_med = r(p50)
gen Size_large = (Size >= `size_med') if !missing(Size)

reghdfe PriceDelay DU_kw `controls' if Size_large == 1, absorb(Stkcd_num year_num) vce(cluster IndYear_num)
est store het_large
local b_large = _b[DU_kw]

reghdfe PriceDelay DU_kw `controls' if Size_large == 0, absorb(Stkcd_num year_num) vce(cluster IndYear_num)
est store het_small
local b_small = _b[DU_kw]

reghdfe PriceDelay c.DU_kw##i.Size_large `controls', absorb(Stkcd_num year_num) vce(cluster IndYear_num)
local size_inter_t = _b[1.Size_large#c.DU_kw]/_se[1.Size_large#c.DU_kw]
local size_inter_p = 2*ttail(e(df_r), abs(`size_inter_t'))
di "Size: 大企业β=`b_large', 小企业β=`b_small'"
di "Size交互项: t = `size_inter_t', p = `size_inter_p'"

* --- C3: 行业 (IT vs Non-IT) ---
di "--- C3: Industry差异检验 ---"
gen IT = (Ind2 == "I6") if !missing(Ind2)

reghdfe PriceDelay DU_kw `controls' if IT == 1, absorb(Stkcd_num year_num) vce(cluster IndYear_num)
est store het_it
local b_it = _b[DU_kw]

reghdfe PriceDelay DU_kw `controls' if IT == 0, absorb(Stkcd_num year_num) vce(cluster IndYear_num)
est store het_nonit
local b_nonit = _b[DU_kw]

reghdfe PriceDelay c.DU_kw##i.IT `controls', absorb(Stkcd_num year_num) vce(cluster IndYear_num)
local it_inter_t = _b[1.IT#c.DU_kw]/_se[1.IT#c.DU_kw]
local it_inter_p = 2*ttail(e(df_r), abs(`it_inter_t'))
di "Industry: IT β=`b_it', Non-IT β=`b_nonit'"
di "Industry交互项: t = `it_inter_t', p = `it_inter_p'"

* --- C4: 分析师覆盖 (高/低) ---
di "--- C4: Analyst差异检验 ---"
* Analyst不在当前数据集中,用panel_dml中的
capture confirm variable Analyst
if _rc == 0 {
    summ Analyst, detail
    local ana_med = r(p50)
    gen Analyst_high = (Analyst >= `ana_med') if !missing(Analyst)

    reghdfe PriceDelay DU_kw `controls' if Analyst_high == 1, absorb(Stkcd_num year_num) vce(cluster IndYear_num)
    local b_ana_h = _b[DU_kw]

    reghdfe PriceDelay DU_kw `controls' if Analyst_high == 0, absorb(Stkcd_num year_num) vce(cluster IndYear_num)
    local b_ana_l = _b[DU_kw]

    reghdfe PriceDelay c.DU_kw##i.Analyst_high `controls', absorb(Stkcd_num year_num) vce(cluster IndYear_num)
    local ana_inter_t = _b[1.Analyst_high#c.DU_kw]/_se[1.Analyst_high#c.DU_kw]
    local ana_inter_p = 2*ttail(e(df_r), abs(`ana_inter_t'))
    di "Analyst: 高覆盖β=`b_ana_h', 低覆盖β=`b_ana_l'"
    di "Analyst交互项: t = `ana_inter_t', p = `ana_inter_p'"
}
else {
    di "Analyst变量不在数据集中, 跳过"
}

* ============================================================
* (D) 替换自变量: DU_sub_ln
* ============================================================
di _n "=========================================="
di "(D) ALTERNATIVE TREATMENT: DU_sub_ln"
di "=========================================="

* OLS with DU_sub_ln
eststo alt_ols: reghdfe PriceDelay DU_sub_ln `controls', absorb(Stkcd_num year_num) vce(cluster IndYear_num)
di "DU_sub_ln OLS: coef=" _b[DU_sub_ln] ", t=" _b[DU_sub_ln]/_se[DU_sub_ln]

* IV1 with DU_sub_ln
* 构造跨省peer for DU_sub_ln
bysort Ind2_num year_num: egen sub_iy_sum = total(DU_sub_ln)
bysort Ind2_num year_num: egen sub_iy_n = count(DU_sub_ln)
bysort Ind2_num year_num Prov_num: egen sub_iyp_sum = total(DU_sub_ln)
bysort Ind2_num year_num Prov_num: egen sub_iyp_n = count(DU_sub_ln)
gen double IV_sub_peer = (sub_iy_sum - sub_iyp_sum) / (sub_iy_n - sub_iyp_n) if (sub_iy_n - sub_iyp_n) > 0

preserve
drop if missing(IV_sub_peer)

* First stage
reghdfe DU_sub_ln IV_sub_peer `controls', absorb(Stkcd_num year_num) vce(cluster IndYear_num) resid
local fs_sub_t = _b[IV_sub_peer]/_se[IV_sub_peer]
local fs_sub_F = `fs_sub_t'^2
di "DU_sub_ln First stage: t=`fs_sub_t', F=`fs_sub_F'"

predict sub_resid, resid
gen sub_hat = DU_sub_ln - sub_resid

eststo alt_iv: reghdfe PriceDelay sub_hat `controls', absorb(Stkcd_num year_num) vce(cluster IndYear_num)
di "DU_sub_ln IV: coef=" _b[sub_hat] ", se=" _se[sub_hat]

restore

* ============================================================
* (E) IV for SYNCH (因变量替换)
* ============================================================
di _n "=========================================="
di "(E) IV FOR SYNCH"
di "=========================================="

* OLS: DU_kw → SYNCH
preserve
drop if missing(SYNCH) | missing(IV_peer_xprov)

eststo synch_ols: reghdfe SYNCH DU_kw `controls', absorb(Stkcd_num year_num) vce(cluster IndYear_num)
di "SYNCH OLS: coef=" _b[DU_kw] ", t=" _b[DU_kw]/_se[DU_kw]

* IV1: 跨省peer → SYNCH
reghdfe DU_kw IV_peer_xprov `controls', absorb(Stkcd_num year_num) vce(cluster IndYear_num) resid
predict synch_resid, resid
gen synch_hat = DU_kw - synch_resid

eststo synch_iv1: reghdfe SYNCH synch_hat `controls', absorb(Stkcd_num year_num) vce(cluster IndYear_num)
di "SYNCH IV1: coef=" _b[synch_hat] ", t=" _b[synch_hat]/_se[synch_hat]

* DWH for SYNCH
reghdfe SYNCH DU_kw synch_resid `controls', absorb(Stkcd_num year_num) vce(cluster IndYear_num)
local synch_dwh_t = _b[synch_resid]/_se[synch_resid]
local synch_dwh_p = 2*ttail(e(df_r), abs(`synch_dwh_t'))
di "SYNCH DWH: t=`synch_dwh_t', p=`synch_dwh_p'"

restore

* ============================================================
* 汇总
* ============================================================
di _n "=========================================="
di "SUMMARY"
di "=========================================="

di _n "--- Heckman ---"
est restore heck
di "DU_kw: " _b[DU_kw] " (" _se[DU_kw] ")"
di "IMR:   " _b[imr] " (" _se[imr] ")"

di _n "--- Placebo ---"
est restore placebo
di "DU_kw_lead: " _b[DU_kw_lead] " (" _se[DU_kw_lead] "), t=" _b[DU_kw_lead]/_se[DU_kw_lead]

di _n "--- Heterogeneity Group Differences ---"
di "SOE interaction: t=`soe_inter_t', p=`soe_inter_p'"
di "Size interaction: t=`size_inter_t', p=`size_inter_p'"
di "Industry interaction: t=`it_inter_t', p=`it_inter_p'"

di _n "--- Alternative Treatment ---"
est restore alt_ols
di "DU_sub_ln OLS: " _b[DU_sub_ln] " (" _se[DU_sub_ln] ")"
est restore alt_iv
di "DU_sub_ln IV: " _b[sub_hat] " (" _se[sub_hat] ")"

di _n "--- SYNCH IV ---"
est restore synch_ols
di "SYNCH OLS: " _b[DU_kw] " (" _se[DU_kw] ")"
est restore synch_iv1
di "SYNCH IV1: " _b[synch_hat] " (" _se[synch_hat] ")"

di _n "=========================================="
di "DONE: table5b_supplementary_v16.do"
di "=========================================="
