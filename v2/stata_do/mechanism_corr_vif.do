* ==============================================================
* 四机制变量相关矩阵 + VIF sanity check
* 变量：Amihud / lnAmihud / CashFlowVol / SCConc / Comparability_med
* 数据：reg_sample_v18.dta + merge mechanism_comparability 结果
* 2026-04-21
* ==============================================================

clear all
set more off
capture log close

log using "/Users/mac/computerscience/0做完了/15会计研究/v2/results/mechanism_corr_vif.log", replace text

use "/Users/mac/computerscience/0做完了/15会计研究/v1/data_stata/reg_sample_v18.dta", clear

* merge Comparability_med
capture confirm variable Comparability_med
if _rc {
    preserve
        import delimited using "/Users/mac/computerscience/0做完了/15会计研究/v1/data_parquet/accounting_comparability.csv", clear
        keep Stkcd year Comparability_med
        tempfile cmp
        save `cmp'
    restore
    merge 1:1 Stkcd year using `cmp', keep(match master) nogen
}

gen lnAmihud = ln(1 + Amihud)

* winsorize 1%/99%（与主回归一致）
foreach v in Amihud lnAmihud CashFlowVol SCConc Comparability_med {
    qui sum `v', detail
    local p1 = r(p1)
    local p99 = r(p99)
    replace `v' = `p1' if `v' < `p1' & !missing(`v')
    replace `v' = `p99' if `v' > `p99' & !missing(`v')
}

display _newline "=== (1) 样本覆盖 ==="
misstable summarize Amihud lnAmihud CashFlowVol SCConc Comparability_med

display _newline "=== (2) Pearson 相关矩阵（全样本） ==="
pwcorr Amihud lnAmihud CashFlowVol SCConc Comparability_med, sig

display _newline "=== (3) Spearman rank 相关矩阵 ==="
spearman Amihud lnAmihud CashFlowVol SCConc Comparability_med, stats(rho p)

display _newline "=== (4) Within-firm 相关（demean by Stkcd 后） ==="
foreach v in Amihud lnAmihud CashFlowVol SCConc Comparability_med {
    bysort Stkcd: egen `v'_m = mean(`v')
    gen `v'_dm = `v' - `v'_m
}
pwcorr Amihud_dm lnAmihud_dm CashFlowVol_dm SCConc_dm Comparability_med_dm, sig

display _newline "=== (5) VIF：把四机制当自变量，PriceDelay 当因变量 ==="
reg PriceDelay lnAmihud CashFlowVol SCConc Comparability_med ///
    Size Lev ROA TobinQ Age Growth IndepRatio Dual Top1Share SOE CFO
vif

display _newline "=== (6) VIF 仅四机制（无控制变量） ==="
reg PriceDelay lnAmihud CashFlowVol SCConc Comparability_med
vif

display _newline "=== (7) Pairwise 条件回归：PriceDelay 同时对两两机制 ==="
foreach m1 in lnAmihud CashFlowVol SCConc Comparability_med {
    foreach m2 in lnAmihud CashFlowVol SCConc Comparability_med {
        if "`m1'" < "`m2'" {
            display _newline "--- PriceDelay ~ `m1' + `m2' + controls + FE ---"
            reghdfe PriceDelay `m1' `m2' Size Lev ROA TobinQ Age Growth ///
                IndepRatio Dual Top1Share SOE CFO, ///
                absorb(Stkcd_num year_num) cluster(IndYear_num)
        }
    }
}

display _newline "=== (8) Joint 回归：四机制同时进 PriceDelay ==="
reghdfe PriceDelay lnAmihud CashFlowVol SCConc Comparability_med ///
    Size Lev ROA TobinQ Age Growth IndepRatio Dual Top1Share SOE CFO, ///
    absorb(Stkcd_num year_num) cluster(IndYear_num)

display _newline "=== 完成 ==="

log close
