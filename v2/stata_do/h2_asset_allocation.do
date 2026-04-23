* ==============================================================
* H2 资产配置机制：江艇 2022 consistent-with 范式
* 数据: reg_sample_iv_v16.dta (N=43,735, 非金融非ST A股 2011-2024)
* 三步回归 + 基准对照
* 新增日期：2026-04-20（方案 B 重构）
* ==============================================================

clear all
set more off
capture log close

log using "/Users/mac/computerscience/0做完了/15会计研究/v2/results/h2_asset_allocation.log", replace text

use "/Users/mac/computerscience/0做完了/15会计研究/v1/data_stata/reg_sample_iv_v16.dta", clear

* Winsorize FinAsset 1%/99% 内联实现（避免依赖 winsor2）
qui sum FinAsset, detail
local p1 = r(p1)
local p99 = r(p99)
replace FinAsset = `p1' if FinAsset < `p1' & !missing(FinAsset)
replace FinAsset = `p99' if FinAsset > `p99' & !missing(FinAsset)
display "FinAsset 已按 [" `p1' ", " `p99' "] winsorize"

* 对照：使用 DU_kw_ln 作为 DU_kw 的对数稳健变体
* 控制变量 (v16 口径)
local ctrls "Size Lev ROA TobinQ Age Growth IndepRatio Dual Top1Share SOE CFO"

* ============================================================
* STEP 0: 基准 PriceDelay = DU + X + FE (作为联合回归的对照)
* ============================================================
display _newline "=== STEP 0: Baseline PriceDelay = DU + X ==="
reghdfe PriceDelay DU_kw `ctrls', absorb(Stkcd_num year_num) cluster(IndYear_num)
estimates store base_kw
reghdfe PriceDelay DU_llm `ctrls', absorb(Stkcd_num year_num) cluster(IndYear_num)
estimates store base_llm

* ============================================================
* STEP 1: H2 主检验 DU -> FinAsset (去金融化)
* 预期 β < 0
* ============================================================
display _newline "=== STEP 1: H2 主检验 FinAsset = DU + X ==="
reghdfe FinAsset DU_kw `ctrls', absorb(Stkcd_num year_num) cluster(IndYear_num)
estimates store h2_kw
reghdfe FinAsset DU_llm `ctrls', absorb(Stkcd_num year_num) cluster(IndYear_num)
estimates store h2_llm

* 用 DU_kw_ln 稳健
reghdfe FinAsset DU_kw_ln `ctrls', absorb(Stkcd_num year_num) cluster(IndYear_num)
estimates store h2_kwln

* ============================================================
* STEP 2: FinAsset -> PriceDelay (金融化 → 定价效率下降)
* 预期 δ > 0
* ============================================================
display _newline "=== STEP 2: PriceDelay = FinAsset + X ==="
reghdfe PriceDelay FinAsset `ctrls', absorb(Stkcd_num year_num) cluster(IndYear_num)
estimates store fin_pd

* ============================================================
* STEP 3: 联合回归 (consistent-with / 江艇 2022 范式)
* PriceDelay = α + β·DU + δ·FinAsset + X + FE
* 对比 STEP 0 β_DU 与此步 β_DU
* ============================================================
display _newline "=== STEP 3: 联合 PriceDelay = DU + FinAsset + X ==="
reghdfe PriceDelay DU_kw FinAsset `ctrls', absorb(Stkcd_num year_num) cluster(IndYear_num)
estimates store joint_kw
reghdfe PriceDelay DU_llm FinAsset `ctrls', absorb(Stkcd_num year_num) cluster(IndYear_num)
estimates store joint_llm

* ============================================================
* 导出：全部七列到 CSV
* ============================================================
display _newline "=== 导出结果到 CSV ==="
esttab base_kw base_llm h2_kw h2_llm h2_kwln fin_pd joint_kw joint_llm ///
    using "/Users/mac/computerscience/0做完了/15会计研究/v2/results/h2_asset_allocation.csv", ///
    replace csv se r2 ar2 star(* 0.10 ** 0.05 *** 0.01) ///
    nogaps compress ///
    mtitles("Base_PD_KW" "Base_PD_LLM" "H2_Fin_KW" "H2_Fin_LLM" "H2_Fin_KWln" "Fin_PD" "Joint_KW" "Joint_LLM") ///
    title("H2 资产配置机制：江艇 2022 consistent-with 范式")

* 原始系数/SE/t/p 到另一个 CSV（便于程序解析）
estout base_kw base_llm h2_kw h2_llm h2_kwln fin_pd joint_kw joint_llm ///
    using "/Users/mac/computerscience/0做完了/15会计研究/v2/results/h2_asset_allocation_raw.csv", ///
    replace cells("b(fmt(6)) se(fmt(6)) t(fmt(3)) p(fmt(4))") ///
    stats(N r2) delimiter(",") varlabels(_cons "Constant") ///
    mlabels("Base_PD_KW" "Base_PD_LLM" "H2_Fin_KW" "H2_Fin_LLM" "H2_Fin_KWln" "Fin_PD" "Joint_KW" "Joint_LLM")

display _newline "=== 全部完成 ==="

log close
