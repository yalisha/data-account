* ==============================================================
* H2 资产配置机制 — 路径 B：滞后 DU 版本
* 动机：防止同期反向因果 (高金融化企业可能更积极披露数据要素利用以"洗白"主业)
* 数据: reg_sample_iv_v16.dta
* 三步回归 (lagged 版本) + 基准
* 新增日期：2026-04-20
* ==============================================================

clear all
set more off
capture log close

log using "/Users/mac/computerscience/0做完了/15会计研究/v2/results/h2_asset_allocation_lagged.log", replace text

use "/Users/mac/computerscience/0做完了/15会计研究/v1/data_stata/reg_sample_iv_v16.dta", clear

* Winsorize FinAsset 1%/99%
qui sum FinAsset, detail
local p1 = r(p1)
local p99 = r(p99)
replace FinAsset = `p1' if FinAsset < `p1' & !missing(FinAsset)
replace FinAsset = `p99' if FinAsset > `p99' & !missing(FinAsset)

* 设置面板结构构造滞后项
tsset Stkcd_num year_num

* 构造 DU 的一阶滞后
gen DU_kw_lag = L.DU_kw
gen DU_kw_ln_lag = L.DU_kw_ln
gen DU_llm_lag = L.DU_llm

* 构造 FinAsset 的一阶滞后 (用于 STEP 2B)
gen FinAsset_lag = L.FinAsset

* 检查：滞后变量描述
display _newline "=== 滞后变量描述统计 ==="
sum DU_kw DU_kw_lag DU_llm DU_llm_lag FinAsset FinAsset_lag

local ctrls "Size Lev ROA TobinQ Age Growth IndepRatio Dual Top1Share SOE CFO"

* ============================================================
* STEP 0B: 基准对照 (lagged DU 版本 Delay 主效应)
* ============================================================
display _newline "=== STEP 0B: Baseline Delay = DU_{t-1} + X ==="
reghdfe PriceDelay DU_kw_lag `ctrls', absorb(Stkcd_num year_num) cluster(IndYear_num)
estimates store base_kw_lag
reghdfe PriceDelay DU_llm_lag `ctrls', absorb(Stkcd_num year_num) cluster(IndYear_num)
estimates store base_llm_lag

* ============================================================
* STEP 1B: H2 滞后版 DU_{t-1} -> FinAsset_t
* 朱康 2025 范式的因果方向
* ============================================================
display _newline "=== STEP 1B: 滞后版 H2 FinAsset = DU_{t-1} + X ==="
reghdfe FinAsset DU_kw_lag `ctrls', absorb(Stkcd_num year_num) cluster(IndYear_num)
estimates store h2_kw_lag
reghdfe FinAsset DU_llm_lag `ctrls', absorb(Stkcd_num year_num) cluster(IndYear_num)
estimates store h2_llm_lag
reghdfe FinAsset DU_kw_ln_lag `ctrls', absorb(Stkcd_num year_num) cluster(IndYear_num)
estimates store h2_kwln_lag

* ============================================================
* STEP 2B: 滞后版 FinAsset_{t-1} -> Delay_t
* 防止同期因果倒置
* ============================================================
display _newline "=== STEP 2B: 滞后版 Delay = FinAsset_{t-1} + X ==="
reghdfe PriceDelay FinAsset_lag `ctrls', absorb(Stkcd_num year_num) cluster(IndYear_num)
estimates store fin_pd_lag

* ============================================================
* STEP 3B: 联合回归 (滞后 DU + 同期 FinAsset -> Delay)
* ============================================================
display _newline "=== STEP 3B: 联合滞后 Delay = DU_{t-1} + FinAsset + X ==="
reghdfe PriceDelay DU_kw_lag FinAsset `ctrls', absorb(Stkcd_num year_num) cluster(IndYear_num)
estimates store joint_kw_lag
reghdfe PriceDelay DU_llm_lag FinAsset `ctrls', absorb(Stkcd_num year_num) cluster(IndYear_num)
estimates store joint_llm_lag

* STEP 3C: DU_{t-1} + FinAsset_{t-1} 都滞后 (最保守因果推断)
display _newline "=== STEP 3C: 全滞后 Delay = DU_{t-1} + FinAsset_{t-1} + X ==="
reghdfe PriceDelay DU_kw_lag FinAsset_lag `ctrls', absorb(Stkcd_num year_num) cluster(IndYear_num)
estimates store joint_kw_fullag
reghdfe PriceDelay DU_llm_lag FinAsset_lag `ctrls', absorb(Stkcd_num year_num) cluster(IndYear_num)
estimates store joint_llm_fullag

* ============================================================
* 导出
* ============================================================
esttab base_kw_lag base_llm_lag h2_kw_lag h2_llm_lag h2_kwln_lag ///
       fin_pd_lag joint_kw_lag joint_llm_lag joint_kw_fullag joint_llm_fullag ///
    using "/Users/mac/computerscience/0做完了/15会计研究/v2/results/h2_asset_allocation_lagged.csv", ///
    replace csv se r2 ar2 star(* 0.10 ** 0.05 *** 0.01) ///
    nogaps compress ///
    mtitles("Base_PD_KW_lag" "Base_PD_LLM_lag" "H2_Fin_KW_lag" "H2_Fin_LLM_lag" "H2_Fin_KWln_lag" ///
            "Fin_PD_lag" "Joint_KW_lag" "Joint_LLM_lag" "Joint_KW_fullag" "Joint_LLM_fullag") ///
    title("H2 资产配置机制 — 路径 B：滞后 DU/FinAsset 版本")

display _newline "=== 路径 B 全部完成 ==="

log close
