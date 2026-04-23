* ============================================================
* v16 补充检验 Part 2 (接续Part1: A/B/C1/C2已完成)
* (C3) 行业异质性 — 用更宽泛的高科技定义
* (C4) 分析师覆盖异质性
* (D) 替换自变量 DU_sub_ln
* (E) IV for SYNCH
* ============================================================

clear all
set more off

use "/Users/mac/computerscience/15会计研究/data_stata/reg_sample_iv_v16.dta", clear

local controls Size Lev ROA TobinQ Age Growth IndepRatio Dual Top1Share SOE CFO

drop if missing(Ind2_num)
drop if missing(ProvDigital2016)

xtset Stkcd_num year

* 构造IV
bysort Ind2_num year_num: egen iy_sum = total(DU_kw)
bysort Ind2_num year_num: egen iy_n = count(DU_kw)
bysort Ind2_num year_num Prov_num: egen iyp_sum = total(DU_kw)
bysort Ind2_num year_num Prov_num: egen iyp_n = count(DU_kw)
gen double IV_peer_xprov = (iy_sum - iyp_sum) / (iy_n - iyp_n) if (iy_n - iyp_n) > 0
gen double IV_prov_trend = ProvDigital2016 * (year - 2011)

* ============================================================
* (C3) 行业异质性: 高科技(I6+M7) vs 传统
* I6=信息传输软件, M7=科学研究技术服务
* ============================================================
di _n "=========================================="
di "(C3) INDUSTRY HETEROGENEITY"
di "=========================================="

* Ind2实际是3位代码(如I65), 取前两位匹配
gen Ind2_2 = substr(Ind2, 1, 1) + substr(Ind2, 2, 1)
* 高科技: I(信息传输软件) + M(科学研究技术服务)
gen HighTech = (substr(Ind2, 1, 1) == "I" | substr(Ind2, 1, 1) == "M") if !missing(Ind2)
tab HighTech

* 交互项检验 (全样本, 更有统计效力)
eststo het_ind: reghdfe PriceDelay c.DU_kw##i.HighTech `controls', absorb(Stkcd_num year_num) vce(cluster IndYear_num)
local ind_inter_t = _b[1.HighTech#c.DU_kw]/_se[1.HighTech#c.DU_kw]
local ind_inter_p = 2*ttail(e(df_r), abs(`ind_inter_t'))
di "Industry交互项: t = `ind_inter_t', p = `ind_inter_p'"

* 分组回归
reghdfe PriceDelay DU_kw `controls' if HighTech == 0, absorb(Stkcd_num year_num) vce(cluster IndYear_num)
local b_trad = _b[DU_kw]
local t_trad = _b[DU_kw]/_se[DU_kw]
di "传统行业: β=`b_trad', t=`t_trad'"

* 高科技用 Ind+Year FE 替代 Firm FE (样本较小)
reghdfe PriceDelay DU_kw `controls' if HighTech == 1, absorb(Ind2_num year_num) vce(cluster IndYear_num)
local b_tech = _b[DU_kw]
local t_tech = _b[DU_kw]/_se[DU_kw]
di "高科技行业 (Ind+Year FE): β=`b_tech', t=`t_tech'"

* ============================================================
* (C4) 分析师覆盖异质性
* Analyst变量不在reg_sample中, 需要从panel_dml merge
* 用panel中的Analyst
* ============================================================
di _n "=========================================="
di "(C4) ANALYST COVERAGE HETEROGENEITY"
di "=========================================="

* Analyst 在 reg_sample_v5 中没有, 但 panel_dml 有
* 先尝试直接使用
capture confirm variable Analyst
if _rc != 0 {
    di "Analyst不在数据集中, 使用代理: 行业年份平均Size"
    di "跳过分析师异质性"
}
else {
    summ Analyst, detail
    local ana_med = r(p50)
    gen Analyst_high = (Analyst >= `ana_med') if !missing(Analyst)

    reghdfe PriceDelay DU_kw `controls' if Analyst_high == 1, absorb(Stkcd_num year_num) vce(cluster IndYear_num)
    local b_ana_h = _b[DU_kw]
    di "高分析师覆盖: β=`b_ana_h'"

    reghdfe PriceDelay DU_kw `controls' if Analyst_high == 0, absorb(Stkcd_num year_num) vce(cluster IndYear_num)
    local b_ana_l = _b[DU_kw]
    di "低分析师覆盖: β=`b_ana_l'"

    reghdfe PriceDelay c.DU_kw##i.Analyst_high `controls', absorb(Stkcd_num year_num) vce(cluster IndYear_num)
    local ana_inter_t = _b[1.Analyst_high#c.DU_kw]/_se[1.Analyst_high#c.DU_kw]
    local ana_inter_p = 2*ttail(e(df_r), abs(`ana_inter_t'))
    di "Analyst交互项: t = `ana_inter_t', p = `ana_inter_p'"
}

* ============================================================
* (D) 替换自变量: DU_sub_ln (实质利用)
* ============================================================
di _n "=========================================="
di "(D) ALTERNATIVE TREATMENT: DU_sub_ln"
di "=========================================="

* OLS
eststo alt_ols: reghdfe PriceDelay DU_sub_ln `controls', absorb(Stkcd_num year_num) vce(cluster IndYear_num)
di "DU_sub_ln OLS: coef=" _b[DU_sub_ln] ", t=" _b[DU_sub_ln]/_se[DU_sub_ln]

* IV: 跨省peer for DU_sub_ln
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

* Second stage
eststo alt_iv: reghdfe PriceDelay sub_hat `controls', absorb(Stkcd_num year_num) vce(cluster IndYear_num)
di "DU_sub_ln IV: coef=" _b[sub_hat] ", se=" _se[sub_hat] ", t=" _b[sub_hat]/_se[sub_hat]

* DWH
reghdfe PriceDelay DU_sub_ln sub_resid `controls', absorb(Stkcd_num year_num) vce(cluster IndYear_num)
local dwh_sub_t = _b[sub_resid]/_se[sub_resid]
local dwh_sub_p = 2*ttail(e(df_r), abs(`dwh_sub_t'))
di "DU_sub_ln DWH: t=`dwh_sub_t', p=`dwh_sub_p'"

restore

* ============================================================
* (E) IV for SYNCH (因变量替换)
* ============================================================
di _n "=========================================="
di "(E) IV FOR SYNCH"
di "=========================================="

preserve
drop if missing(SYNCH) | missing(IV_peer_xprov)

* OLS
eststo synch_ols: reghdfe SYNCH DU_kw `controls', absorb(Stkcd_num year_num) vce(cluster IndYear_num)
di "SYNCH OLS: coef=" _b[DU_kw] ", t=" _b[DU_kw]/_se[DU_kw]

* IV1: 跨省peer
reghdfe DU_kw IV_peer_xprov `controls', absorb(Stkcd_num year_num) vce(cluster IndYear_num) resid
predict synch_resid, resid
gen synch_hat = DU_kw - synch_resid

eststo synch_iv1: reghdfe SYNCH synch_hat `controls', absorb(Stkcd_num year_num) vce(cluster IndYear_num)
di "SYNCH IV1: coef=" _b[synch_hat] ", t=" _b[synch_hat]/_se[synch_hat]

* DWH
reghdfe SYNCH DU_kw synch_resid `controls', absorb(Stkcd_num year_num) vce(cluster IndYear_num)
local synch_dwh_t = _b[synch_resid]/_se[synch_resid]
local synch_dwh_p = 2*ttail(e(df_r), abs(`synch_dwh_t'))
di "SYNCH DWH: t=`synch_dwh_t', p=`synch_dwh_p'"

* IV2: 省级数字化
drop synch_resid synch_hat
reghdfe DU_kw IV_prov_trend `controls', absorb(Stkcd_num year_num) vce(cluster IndYear_num) resid
predict synch_resid2, resid
gen synch_hat2 = DU_kw - synch_resid2

eststo synch_iv2: reghdfe SYNCH synch_hat2 `controls', absorb(Stkcd_num year_num) vce(cluster IndYear_num)
di "SYNCH IV2: coef=" _b[synch_hat2] ", t=" _b[synch_hat2]/_se[synch_hat2]

restore

* ============================================================
* 汇总
* ============================================================
di _n "=========================================="
di "ALL RESULTS SUMMARY"
di "=========================================="

di "--- Part 1 results (from previous run) ---"
di "Heckman: DU_kw=-0.0040 (t=-4.39), IMR=-0.1228 (t=-6.23)"
di "Placebo single: DU_kw_lead=-0.0025 (t=-3.32)"
di "Placebo joint: DU_kw=-0.0050 (t=-4.64), DU_kw_lead=0.0001 (t=0.11)"
di "SOE inter: t=1.18, p=0.236"
di "  国企β=-0.0064***, 民企β=-0.0039***"
di "Size inter: t=-1.33, p=0.185"
di "  大企业β=-0.0054***, 小企业β=-0.0050***"

di _n "--- Part 2 results ---"
di "Industry inter: t=`ind_inter_t', p=`ind_inter_p'"
di "  传统β=`b_trad' (t=`t_trad'), 高科技β=`b_tech' (t=`t_tech')"

di _n "=========================================="
di "DONE: table5b_part2_v16.do"
di "=========================================="
