* ==============================================================
* 新机制候选：盈余质量 EarnQual (Dechow-Dichev 2002)
* 预期方向：披露 → EarnQual ↓（质量 ↑）
* 2026-04-20
* ==============================================================

clear all
set more off
capture log close

log using "/Users/mac/computerscience/0做完了/15会计研究/v2/results/mechanism_earnqual_dd.log", replace text

* Merge EarnQual into v18 sample (CSV 已由 Python 产出)
use "/Users/mac/computerscience/0做完了/15会计研究/v1/data_stata/reg_sample_v18.dta", clear
display _newline "v18 样本 N = " _N

preserve
    import delimited using "/Users/mac/computerscience/0做完了/15会计研究/v1/data_parquet/earnings_quality_dd.csv", encoding("utf-8") clear
    rename stkcd Stkcd
    rename year year
    keep Stkcd year earnqual earnqualabs
    tempfile aq
    save `aq', replace
restore

merge 1:1 Stkcd year using `aq', keep(master match) nogen
display _newline "合并 EarnQual 后 N = " _N
count if !missing(earnqual)
count if !missing(earnqualabs)

* Winsorize 1%/99% (已在 Python 做过，但以防万一)
foreach v in earnqual earnqualabs {
    qui sum `v', detail
    local p1 = r(p1)
    local p99 = r(p99)
    replace `v' = `p1' if `v' < `p1' & !missing(`v')
    replace `v' = `p99' if `v' > `p99' & !missing(`v')
}

local ctrls "Size Lev ROA TobinQ Age Growth IndepRatio Dual Top1Share SOE CFO"

* ============================================================
* EarnQual (rolling σ 版本，N ≈ 27,253)
* ============================================================
display _newline "=== EarnQual (rolling σ) = DU + X ==="
reghdfe earnqual DU_kw `ctrls', absorb(Stkcd_num year_num) cluster(IndYear_num)
estimates store eq_kw
reghdfe earnqual DU_llm `ctrls', absorb(Stkcd_num year_num) cluster(IndYear_num)
estimates store eq_llm

* ============================================================
* EarnQualAbs (|ε| 版本，N ≈ 36,378，覆盖更全)
* ============================================================
display _newline "=== EarnQualAbs (|ε|) = DU + X ==="
reghdfe earnqualabs DU_kw `ctrls', absorb(Stkcd_num year_num) cluster(IndYear_num)
estimates store eqa_kw
reghdfe earnqualabs DU_llm `ctrls', absorb(Stkcd_num year_num) cluster(IndYear_num)
estimates store eqa_llm

* ============================================================
* 导出
* ============================================================
esttab eq_kw eq_llm eqa_kw eqa_llm ///
    using "/Users/mac/computerscience/0做完了/15会计研究/v2/results/mechanism_earnqual_dd.csv", ///
    replace csv se r2 ar2 star(* 0.10 ** 0.05 *** 0.01) ///
    nogaps compress ///
    mtitles("EarnQual_KW" "EarnQual_LLM" "EarnQualAbs_KW" "EarnQualAbs_LLM") ///
    title("盈余质量 DD 2002 机制")

display _newline "=== 盈余质量 完成 ==="
log close
