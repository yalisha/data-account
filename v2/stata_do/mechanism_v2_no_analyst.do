* ==============================================================
* 机制重构：去 analyst 版本
* 保留 CashFlowVol + SCConc（v18 已跑过，此处用统一 v16 控制变量口径重跑）
* 新增 Amihud 作为市场端信息不对称测度（Merton 1987 + Amihud-Mendelson 1986）
* 数据: reg_sample_v18.dta (N=43,735)
* 2026-04-20
* ==============================================================

clear all
set more off
capture log close

log using "/Users/mac/computerscience/0做完了/15会计研究/v2/results/mechanism_v2_no_analyst.log", replace text

use "/Users/mac/computerscience/0做完了/15会计研究/v1/data_stata/reg_sample_v18.dta", clear

* 描述性：确认机制变量分布
display _newline "=== 机制变量描述 ==="
sum CashFlowVol SCConc Amihud, detail

* Amihud 分布偏态大，对数化
gen lnAmihud = ln(1 + Amihud)
sum lnAmihud

* winsorize 机制变量 1%/99%
foreach v in CashFlowVol SCConc Amihud lnAmihud {
    qui sum `v', detail
    local p1 = r(p1)
    local p99 = r(p99)
    replace `v' = `p1' if `v' < `p1' & !missing(`v')
    replace `v' = `p99' if `v' > `p99' & !missing(`v')
    display "`v' winsorize [" `p1' ", " `p99' "]"
}

local ctrls "Size Lev ROA TobinQ Age Growth IndepRatio Dual Top1Share SOE CFO"

* ============================================================
* MECH 1: CashFlowVol (基本面不确定性)
* ============================================================
display _newline "=== MECH 1: CashFlowVol = DU + X ==="
reghdfe CashFlowVol DU_kw `ctrls', absorb(Stkcd_num year_num) cluster(IndYear_num)
estimates store cfv_kw
reghdfe CashFlowVol DU_llm `ctrls', absorb(Stkcd_num year_num) cluster(IndYear_num)
estimates store cfv_llm

* ============================================================
* MECH 2: SCConc (价值实现不确定性 / 供应链)
* ============================================================
display _newline "=== MECH 2: SCConc = DU + X ==="
reghdfe SCConc DU_kw `ctrls', absorb(Stkcd_num year_num) cluster(IndYear_num)
estimates store sc_kw
reghdfe SCConc DU_llm `ctrls', absorb(Stkcd_num year_num) cluster(IndYear_num)
estimates store sc_llm

* ============================================================
* MECH 3: Amihud 非流动性（市场端信息不对称 — 新增）
* ============================================================
display _newline "=== MECH 3a: Amihud (原始) = DU + X ==="
reghdfe Amihud DU_kw `ctrls', absorb(Stkcd_num year_num) cluster(IndYear_num)
estimates store amh_kw
reghdfe Amihud DU_llm `ctrls', absorb(Stkcd_num year_num) cluster(IndYear_num)
estimates store amh_llm

display _newline "=== MECH 3b: lnAmihud (对数) = DU + X ==="
reghdfe lnAmihud DU_kw `ctrls', absorb(Stkcd_num year_num) cluster(IndYear_num)
estimates store lnamh_kw
reghdfe lnAmihud DU_llm `ctrls', absorb(Stkcd_num year_num) cluster(IndYear_num)
estimates store lnamh_llm

* ============================================================
* 导出
* ============================================================
esttab cfv_kw cfv_llm sc_kw sc_llm amh_kw amh_llm lnamh_kw lnamh_llm ///
    using "/Users/mac/computerscience/0做完了/15会计研究/v2/results/mechanism_v2_no_analyst.csv", ///
    replace csv se r2 ar2 star(* 0.10 ** 0.05 *** 0.01) ///
    nogaps compress ///
    mtitles("CFV_KW" "CFV_LLM" "SC_KW" "SC_LLM" "Amihud_KW" "Amihud_LLM" "lnAmihud_KW" "lnAmihud_LLM") ///
    title("机制重构 v2：CashFlowVol + SCConc + Amihud (去 analyst)")

display _newline "=== 机制 v2 全部完成 ==="

log close
