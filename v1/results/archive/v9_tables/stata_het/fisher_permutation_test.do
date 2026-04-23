
/*==============================================================
  Table 6 Panel A: 异质性分析 - Fisher组合检验
  方法: Freedman-Lane残差化 + 1000次置换检验
  参照朱康(2025)会计研究的做法
==============================================================*/

clear all
set more off
set seed 20250306

* Load data
use "/Users/mac/computerscience/15会计研究/data_stata/reg_sample_het.dta", clear

* Controls
global controls Size Lev ROA TobinQ Age Growth BoardSize IndepRatio ///
    Dual Top1Share SOE InstHold Amihud Analyst AuditType

* Quick summary
sum PriceDelay DU_kw SOE HighInfoEnv HighSA HighInstHold

/*--------------------------------------------------------------
  Step 1: Residualize Y and X using reghdfe (full sample)
  This absorbs Firm FE + Year FE + controls
  After this, each permutation only needs simple OLS
--------------------------------------------------------------*/
di _n "=== Step 1: Residualizing Y and X ==="

* Residualize PriceDelay
reghdfe PriceDelay $controls, absorb(Stkcd_num year_num) resid(resid_y)
* Residualize DU_kw  
reghdfe DU_kw $controls, absorb(Stkcd_num year_num) resid(resid_x)

di "Residualization done."

/*--------------------------------------------------------------
  Step 2: For each grouping dimension, run Fisher permutation test
--------------------------------------------------------------*/

* Define grouping dimensions to test
* We test 6 dimensions, pick the best 3-4
local dim_list "SOE HighInfoEnv HighSA HighInstHold HighSize HighAnalyst"
local dim_labels `" "产权性质" "信息环境(行业年度)" "融资约束(SA)" "机构持股" "企业规模" "分析师覆盖(全样本)" "'

* Store results
tempname results
postfile `results' str30 dimension double(b_high b_low diff_obs p_fisher n_high n_low) using "/Users/mac/computerscience/15会计研究/results/v9_tables/fisher_test_results.dta", replace

local nreps 1000

local i = 1
foreach dim of local dim_list {
    local label : word `i' of `dim_labels'
    di _n "=== Testing dimension: `dim' (`label') ==="
    
    * Observed coefficients from residual regression
    qui reg resid_y resid_x if `dim' == 1, nocons
    local b_high = _b[resid_x]
    local n_high = e(N)
    
    qui reg resid_y resid_x if `dim' == 0, nocons
    local b_low = _b[resid_x]
    local n_low = e(N)
    
    local diff_obs = abs(`b_high' - `b_low')
    
    di "  High group: b = " %9.6f `b_high' ", N = `n_high'"
    di "  Low group:  b = " %9.6f `b_low' ", N = `n_low'"
    di "  |Difference| = " %9.6f `diff_obs'
    
    * Fisher permutation test
    * For firm-level groups (SOE), permute at firm level
    * For time-varying groups, permute at observation level
    
    * Create firm-level group indicator
    preserve
    collapse (first) `dim', by(Stkcd_num)
    rename `dim' orig_group
    * Check if group is time-invariant (all same within firm)
    restore
    preserve
    bys Stkcd_num: egen sd_group = sd(`dim')
    sum sd_group
    local is_timevarying = (r(max) > 0.001)
    restore
    
    di "  Time-varying: `is_timevarying'"
    
    * Permutation loop
    local pcount = 0
    
    if `is_timevarying' == 0 {
        * Time-invariant: permute at firm level
        preserve
        collapse (first) `dim', by(Stkcd_num)
        local n_firms = _N
        local n_treated = 0
        count if `dim' == 1
        local n_treated = r(N)
        tempfile firm_orig
        save `firm_orig'
        restore
        
        forval rep = 1/`nreps' {
            preserve
            * Create permuted group at firm level
            merge m:1 Stkcd_num using `firm_orig', keepusing(`dim') nogen
            
            * Generate random firm-level permutation
            tempfile obs_data
            save `obs_data'
            
            use `firm_orig', clear
            gen _rand = runiform()
            sort _rand
            gen `dim'_perm = (_n <= `n_treated')
            keep Stkcd_num `dim'_perm
            tempfile perm_groups
            save `perm_groups'
            
            use `obs_data', clear
            merge m:1 Stkcd_num using `perm_groups', nogen
            
            qui reg resid_y resid_x if `dim'_perm == 1, nocons
            local b1_perm = _b[resid_x]
            qui reg resid_y resid_x if `dim'_perm == 0, nocons
            local b0_perm = _b[resid_x]
            local diff_perm = abs(`b1_perm' - `b0_perm')
            
            if `diff_perm' >= `diff_obs' {
                local pcount = `pcount' + 1
            }
            restore
        }
    }
    else {
        * Time-varying: permute at observation level
        forval rep = 1/`nreps' {
            preserve
            local n_treated_obs = `n_high'
            gen _rand = runiform()
            sort _rand
            gen `dim'_perm = (_n <= `n_treated_obs')
            
            qui reg resid_y resid_x if `dim'_perm == 1, nocons
            local b1_perm = _b[resid_x]
            qui reg resid_y resid_x if `dim'_perm == 0, nocons
            local b0_perm = _b[resid_x]
            local diff_perm = abs(`b1_perm' - `b0_perm')
            
            if `diff_perm' >= `diff_obs' {
                local pcount = `pcount' + 1
            }
            restore
        }
    }
    
    local p_fisher = `pcount' / `nreps'
    di "  Fisher p-value = `p_fisher' (`pcount'/`nreps')"
    
    * Store results
    post `results' ("`dim'") (`b_high') (`b_low') (`diff_obs') (`p_fisher') (`n_high') (`n_low')
    
    local i = `i' + 1
}

postclose `results'

/*--------------------------------------------------------------
  Step 3: Also run reghdfe subgroup regressions for actual coef/SE
--------------------------------------------------------------*/
di _n "=== Step 3: Subgroup reghdfe regressions ==="

foreach dim in SOE HighInfoEnv HighSA HighInstHold {
    di _n "--- `dim' ---"
    di "High group (`dim'==1):"
    qui reghdfe PriceDelay DU_kw $controls if `dim' == 1, ///
        absorb(Stkcd_num year_num) cluster(IndYear_num)
    di "  DU_kw: b=" %9.4f _b[DU_kw] " se=" %9.4f _se[DU_kw] ///
       " t=" %6.2f _b[DU_kw]/_se[DU_kw] " N=" e(N) " R2=" %6.3f e(r2)
    
    di "Low group (`dim'==0):"
    qui reghdfe PriceDelay DU_kw $controls if `dim' == 0, ///
        absorb(Stkcd_num year_num) cluster(IndYear_num)
    di "  DU_kw: b=" %9.4f _b[DU_kw] " se=" %9.4f _se[DU_kw] ///
       " t=" %6.2f _b[DU_kw]/_se[DU_kw] " N=" e(N) " R2=" %6.3f e(r2)
}

/*--------------------------------------------------------------
  Step 4: Display summary
--------------------------------------------------------------*/
di _n "=========================================="
di "Fisher Permutation Test Results Summary"
di "=========================================="

use "/Users/mac/computerscience/15会计研究/results/v9_tables/fisher_test_results.dta", clear
list, sep(0) noobs abbreviate(20)

di _n "Significant at 10%:"
list if p_fisher < 0.10, sep(0) noobs abbreviate(20)
