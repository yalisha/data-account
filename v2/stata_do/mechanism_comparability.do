* ==============================================================
* 新机制候选：会计信息可比性 Comparability (De Franco-Kothari-Verdi 2011 TAR)
* 预期方向：披露 → Comparability ↑（即越接近 0）
* 2026-04-20
* ==============================================================

clear all
set more off
capture log close

log using "/Users/mac/computerscience/0做完了/15会计研究/v2/results/mechanism_comparability.log", replace text

use "/Users/mac/computerscience/0做完了/15会计研究/v1/data_stata/reg_sample_v18.dta", clear
display _newline "v18 样本 N = " _N

preserve
    import delimited using "/Users/mac/computerscience/0做完了/15会计研究/v1/data_parquet/accounting_comparability.csv", encoding("utf-8") clear
    rename stkcd Stkcd
    rename year year
    keep Stkcd year comparability_med comparability_top4
    tempfile comp
    save `comp', replace
restore

merge 1:1 Stkcd year using `comp', keep(master match) nogen
display _newline "合并 Comparability 后 N = " _N
count if !missing(comparability_med)
count if !missing(comparability_top4)

* Winsorize (Python 已做，以防万一)
foreach v in comparability_med comparability_top4 {
    qui sum `v', detail
    local p1 = r(p1)
    local p99 = r(p99)
    replace `v' = `p1' if `v' < `p1' & !missing(`v')
    replace `v' = `p99' if `v' > `p99' & !missing(`v')
}

local ctrls "Size Lev ROA TobinQ Age Growth IndepRatio Dual Top1Share SOE CFO"

* ============================================================
* Comparability (Ind2 中位数版本)
* ============================================================
display _newline "=== Comparability_med (Ind2 中位数) = DU + X ==="
reghdfe comparability_med DU_kw `ctrls', absorb(Stkcd_num year_num) cluster(IndYear_num)
estimates store cm_kw
reghdfe comparability_med DU_llm `ctrls', absorb(Stkcd_num year_num) cluster(IndYear_num)
estimates store cm_llm

* ============================================================
* Comparability (Top-4 均值版本 - 朱康/袁蓉丽同口径)
* ============================================================
display _newline "=== Comparability_top4 (Top-4 均值) = DU + X ==="
reghdfe comparability_top4 DU_kw `ctrls', absorb(Stkcd_num year_num) cluster(IndYear_num)
estimates store ct_kw
reghdfe comparability_top4 DU_llm `ctrls', absorb(Stkcd_num year_num) cluster(IndYear_num)
estimates store ct_llm

* ============================================================
* 导出
* ============================================================
esttab cm_kw cm_llm ct_kw ct_llm ///
    using "/Users/mac/computerscience/0做完了/15会计研究/v2/results/mechanism_comparability.csv", ///
    replace csv se r2 ar2 star(* 0.10 ** 0.05 *** 0.01) ///
    nogaps compress ///
    mtitles("Comp_med_KW" "Comp_med_LLM" "Comp_top4_KW" "Comp_top4_LLM") ///
    title("会计信息可比性 De Franco 2011 机制")

display _newline "=== 可比性 完成 ==="
log close
