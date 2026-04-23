* =============================================================
* Heterogeneity Round 2: Additional Dimensions
* =============================================================
clear all
set more off
use "data_stata/reg_sample_het.dta", clear

global controls Size Lev ROA TobinQ Age Growth IndepRatio Dual Top1Share SOE CFO

di "============================================="
di "Round 2: Testing Additional Heterogeneity"
di "============================================="

* =============================================================
* 1. Earnings Volatility (Roastd)
*    Logic: High uncertainty firms benefit more from information
* =============================================================
di ""
di "--- 1. Earnings Volatility (Roastd) ---"
egen med_roastd = median(Roastd)
gen Hi_Roastd = (Roastd >= med_roastd) if Roastd != .

foreach du in DU_kw DU_llm {
    reghdfe PriceDelay `du' $controls if Hi_Roastd == 1, absorb(Stkcd_num year_num) cluster(IndYear_num)
    di "Hi_Roastd `du': b=" %9.4f _b[`du'] " t=" %6.2f (_b[`du']/_se[`du']) " N=" e(N)

    reghdfe PriceDelay `du' $controls if Hi_Roastd == 0, absorb(Stkcd_num year_num) cluster(IndYear_num)
    di "Lo_Roastd `du': b=" %9.4f _b[`du'] " t=" %6.2f (_b[`du']/_se[`du']) " N=" e(N)
}

reghdfe PriceDelay c.DU_kw##i.Hi_Roastd $controls, absorb(Stkcd_num year_num) cluster(IndYear_num)
di "Roastd x DU_kw interaction: b=" %9.4f _b[c.DU_kw#1.Hi_Roastd] " t=" %6.2f (_b[c.DU_kw#1.Hi_Roastd]/_se[c.DU_kw#1.Hi_Roastd]) " p=" %6.4f (2*ttail(e(df_r), abs(_b[c.DU_kw#1.Hi_Roastd]/_se[c.DU_kw#1.Hi_Roastd])))

reghdfe PriceDelay c.DU_llm##i.Hi_Roastd $controls, absorb(Stkcd_num year_num) cluster(IndYear_num)
di "Roastd x DU_llm interaction: b=" %9.4f _b[c.DU_llm#1.Hi_Roastd] " t=" %6.2f (_b[c.DU_llm#1.Hi_Roastd]/_se[c.DU_llm#1.Hi_Roastd]) " p=" %6.4f (2*ttail(e(df_r), abs(_b[c.DU_llm#1.Hi_Roastd]/_se[c.DU_llm#1.Hi_Roastd])))

* Continuous
reghdfe PriceDelay c.DU_kw##c.Roastd $controls, absorb(Stkcd_num year_num) cluster(IndYear_num)
di "Roastd x DU_kw (cont): b=" %9.4f _b[c.DU_kw#c.Roastd] " t=" %6.2f (_b[c.DU_kw#c.Roastd]/_se[c.DU_kw#c.Roastd]) " p=" %6.4f (2*ttail(e(df_r), abs(_b[c.DU_kw#c.Roastd]/_se[c.DU_kw#c.Roastd])))

* =============================================================
* 2. Firm Age (digital native vs legacy)
*    Logic: Younger firms more data-native
* =============================================================
di ""
di "--- 2. Firm Age ---"
egen med_age = median(Age)
gen Old = (Age >= med_age) if Age != .

foreach du in DU_kw DU_llm {
    reghdfe PriceDelay `du' $controls if Old == 0, absorb(Stkcd_num year_num) cluster(IndYear_num)
    di "Young `du': b=" %9.4f _b[`du'] " t=" %6.2f (_b[`du']/_se[`du']) " N=" e(N)

    reghdfe PriceDelay `du' $controls if Old == 1, absorb(Stkcd_num year_num) cluster(IndYear_num)
    di "Old `du': b=" %9.4f _b[`du'] " t=" %6.2f (_b[`du']/_se[`du']) " N=" e(N)
}

reghdfe PriceDelay c.DU_kw##i.Old $controls, absorb(Stkcd_num year_num) cluster(IndYear_num)
di "Age x DU_kw interaction: b=" %9.4f _b[c.DU_kw#1.Old] " t=" %6.2f (_b[c.DU_kw#1.Old]/_se[c.DU_kw#1.Old]) " p=" %6.4f (2*ttail(e(df_r), abs(_b[c.DU_kw#1.Old]/_se[c.DU_kw#1.Old])))

reghdfe PriceDelay c.DU_llm##i.Old $controls, absorb(Stkcd_num year_num) cluster(IndYear_num)
di "Age x DU_llm interaction: b=" %9.4f _b[c.DU_llm#1.Old] " t=" %6.2f (_b[c.DU_llm#1.Old]/_se[c.DU_llm#1.Old]) " p=" %6.4f (2*ttail(e(df_r), abs(_b[c.DU_llm#1.Old]/_se[c.DU_llm#1.Old])))

* =============================================================
* 3. Shareholder Dispersion
*    Logic: More dispersed = more retail = DU disclosure more valuable
* =============================================================
di ""
di "--- 3. Shareholder Dispersion (ShareholderNum) ---"
egen med_shnum = median(ShareholderNum)
gen Hi_ShNum = (ShareholderNum >= med_shnum) if ShareholderNum != .

foreach du in DU_kw DU_llm {
    reghdfe PriceDelay `du' $controls if Hi_ShNum == 1, absorb(Stkcd_num year_num) cluster(IndYear_num)
    di "Hi_ShNum `du': b=" %9.4f _b[`du'] " t=" %6.2f (_b[`du']/_se[`du']) " N=" e(N)

    reghdfe PriceDelay `du' $controls if Hi_ShNum == 0, absorb(Stkcd_num year_num) cluster(IndYear_num)
    di "Lo_ShNum `du': b=" %9.4f _b[`du'] " t=" %6.2f (_b[`du']/_se[`du']) " N=" e(N)
}

reghdfe PriceDelay c.DU_kw##i.Hi_ShNum $controls, absorb(Stkcd_num year_num) cluster(IndYear_num)
di "ShNum x DU_kw interaction: b=" %9.4f _b[c.DU_kw#1.Hi_ShNum] " t=" %6.2f (_b[c.DU_kw#1.Hi_ShNum]/_se[c.DU_kw#1.Hi_ShNum]) " p=" %6.4f (2*ttail(e(df_r), abs(_b[c.DU_kw#1.Hi_ShNum]/_se[c.DU_kw#1.Hi_ShNum])))

reghdfe PriceDelay c.DU_llm##i.Hi_ShNum $controls, absorb(Stkcd_num year_num) cluster(IndYear_num)
di "ShNum x DU_llm interaction: b=" %9.4f _b[c.DU_llm#1.Hi_ShNum] " t=" %6.2f (_b[c.DU_llm#1.Hi_ShNum]/_se[c.DU_llm#1.Hi_ShNum]) " p=" %6.4f (2*ttail(e(df_r), abs(_b[c.DU_llm#1.Hi_ShNum]/_se[c.DU_llm#1.Hi_ShNum])))

* =============================================================
* 4. Industry Competition (Hhi)
*    Logic: Competitive industries = more need for differentiation
* =============================================================
di ""
di "--- 4. Industry Competition (Hhi) ---"
egen med_hhi = median(Hhi)
gen Hi_Hhi = (Hhi >= med_hhi) if Hhi != .

foreach du in DU_kw DU_llm {
    reghdfe PriceDelay `du' $controls if Hi_Hhi == 0, absorb(Stkcd_num year_num) cluster(IndYear_num)
    di "Competitive `du': b=" %9.4f _b[`du'] " t=" %6.2f (_b[`du']/_se[`du']) " N=" e(N)

    reghdfe PriceDelay `du' $controls if Hi_Hhi == 1, absorb(Stkcd_num year_num) cluster(IndYear_num)
    di "Concentrated `du': b=" %9.4f _b[`du'] " t=" %6.2f (_b[`du']/_se[`du']) " N=" e(N)
}

reghdfe PriceDelay c.DU_kw##i.Hi_Hhi $controls, absorb(Stkcd_num year_num) cluster(IndYear_num)
di "Hhi x DU_kw interaction: b=" %9.4f _b[c.DU_kw#1.Hi_Hhi] " t=" %6.2f (_b[c.DU_kw#1.Hi_Hhi]/_se[c.DU_kw#1.Hi_Hhi]) " p=" %6.4f (2*ttail(e(df_r), abs(_b[c.DU_kw#1.Hi_Hhi]/_se[c.DU_kw#1.Hi_Hhi])))

reghdfe PriceDelay c.DU_llm##i.Hi_Hhi $controls, absorb(Stkcd_num year_num) cluster(IndYear_num)
di "Hhi x DU_llm interaction: b=" %9.4f _b[c.DU_llm#1.Hi_Hhi] " t=" %6.2f (_b[c.DU_llm#1.Hi_Hhi]/_se[c.DU_llm#1.Hi_Hhi]) " p=" %6.4f (2*ttail(e(df_r), abs(_b[c.DU_llm#1.Hi_Hhi]/_se[c.DU_llm#1.Hi_Hhi])))

* =============================================================
* 5. Growth (high growth = more uncertainty)
* =============================================================
di ""
di "--- 5. Revenue Growth ---"
egen med_growth = median(Growth)
gen Hi_Growth = (Growth >= med_growth) if Growth != .

foreach du in DU_kw DU_llm {
    reghdfe PriceDelay `du' $controls if Hi_Growth == 1, absorb(Stkcd_num year_num) cluster(IndYear_num)
    di "Hi_Growth `du': b=" %9.4f _b[`du'] " t=" %6.2f (_b[`du']/_se[`du']) " N=" e(N)

    reghdfe PriceDelay `du' $controls if Hi_Growth == 0, absorb(Stkcd_num year_num) cluster(IndYear_num)
    di "Lo_Growth `du': b=" %9.4f _b[`du'] " t=" %6.2f (_b[`du']/_se[`du']) " N=" e(N)
}

reghdfe PriceDelay c.DU_kw##i.Hi_Growth $controls, absorb(Stkcd_num year_num) cluster(IndYear_num)
di "Growth x DU_kw interaction: b=" %9.4f _b[c.DU_kw#1.Hi_Growth] " t=" %6.2f (_b[c.DU_kw#1.Hi_Growth]/_se[c.DU_kw#1.Hi_Growth]) " p=" %6.4f (2*ttail(e(df_r), abs(_b[c.DU_kw#1.Hi_Growth]/_se[c.DU_kw#1.Hi_Growth])))

reghdfe PriceDelay c.DU_llm##i.Hi_Growth $controls, absorb(Stkcd_num year_num) cluster(IndYear_num)
di "Growth x DU_llm interaction: b=" %9.4f _b[c.DU_llm#1.Hi_Growth] " t=" %6.2f (_b[c.DU_llm#1.Hi_Growth]/_se[c.DU_llm#1.Hi_Growth]) " p=" %6.4f (2*ttail(e(df_r), abs(_b[c.DU_llm#1.Hi_Growth]/_se[c.DU_llm#1.Hi_Growth])))

* =============================================================
* 6. BM (Book-to-Market: value vs growth)
*    Logic: Low BM growth firms depend more on intangibles
* =============================================================
di ""
di "--- 6. Book-to-Market (BM) ---"
egen med_bm = median(BM)
gen Hi_BM = (BM >= med_bm) if BM != .

foreach du in DU_kw DU_llm {
    reghdfe PriceDelay `du' $controls if Hi_BM == 0, absorb(Stkcd_num year_num) cluster(IndYear_num)
    di "GrowthFirm(LoBM) `du': b=" %9.4f _b[`du'] " t=" %6.2f (_b[`du']/_se[`du']) " N=" e(N)

    reghdfe PriceDelay `du' $controls if Hi_BM == 1, absorb(Stkcd_num year_num) cluster(IndYear_num)
    di "ValueFirm(HiBM) `du': b=" %9.4f _b[`du'] " t=" %6.2f (_b[`du']/_se[`du']) " N=" e(N)
}

reghdfe PriceDelay c.DU_kw##i.Hi_BM $controls, absorb(Stkcd_num year_num) cluster(IndYear_num)
di "BM x DU_kw interaction: b=" %9.4f _b[c.DU_kw#1.Hi_BM] " t=" %6.2f (_b[c.DU_kw#1.Hi_BM]/_se[c.DU_kw#1.Hi_BM]) " p=" %6.4f (2*ttail(e(df_r), abs(_b[c.DU_kw#1.Hi_BM]/_se[c.DU_kw#1.Hi_BM])))

reghdfe PriceDelay c.DU_llm##i.Hi_BM $controls, absorb(Stkcd_num year_num) cluster(IndYear_num)
di "BM x DU_llm interaction: b=" %9.4f _b[c.DU_llm#1.Hi_BM] " t=" %6.2f (_b[c.DU_llm#1.Hi_BM]/_se[c.DU_llm#1.Hi_BM]) " p=" %6.4f (2*ttail(e(df_r), abs(_b[c.DU_llm#1.Hi_BM]/_se[c.DU_llm#1.Hi_BM])))

* =============================================================
* 7. Continuous year trend
* =============================================================
di ""
di "--- 7. Continuous Year Trend ---"
reghdfe PriceDelay c.DU_kw##c.year_num $controls, absorb(Stkcd_num year_num) cluster(IndYear_num)
di "DU_kw x year (cont): b=" %12.6f _b[c.DU_kw#c.year_num] " t=" %6.2f (_b[c.DU_kw#c.year_num]/_se[c.DU_kw#c.year_num]) " p=" %6.4f (2*ttail(e(df_r), abs(_b[c.DU_kw#c.year_num]/_se[c.DU_kw#c.year_num])))

reghdfe PriceDelay c.DU_llm##c.year_num $controls, absorb(Stkcd_num year_num) cluster(IndYear_num)
di "DU_llm x year (cont): b=" %12.6f _b[c.DU_llm#c.year_num] " t=" %6.2f (_b[c.DU_llm#c.year_num]/_se[c.DU_llm#c.year_num]) " p=" %6.4f (2*ttail(e(df_r), abs(_b[c.DU_llm#c.year_num]/_se[c.DU_llm#c.year_num])))

* =============================================================
* 8. Diagnostic: Industry FE instead of Firm FE
*    Check if cross-sectional heterogeneity appears without firm FE
* =============================================================
di ""
di "--- 8. Diagnostic: Industry FE (no firm FE) ---"

* Size
egen med_size2 = median(Size)
gen big2 = (Size >= med_size2) if Size != .
reghdfe PriceDelay c.DU_kw##i.big2 $controls, absorb(Ind2_num year_num) cluster(IndYear_num)
di "Size x DU_kw (ind FE): b=" %9.4f _b[c.DU_kw#1.big2] " t=" %6.2f (_b[c.DU_kw#1.big2]/_se[c.DU_kw#1.big2]) " p=" %6.4f (2*ttail(e(df_r), abs(_b[c.DU_kw#1.big2]/_se[c.DU_kw#1.big2])))

* Analyst
egen med_an2 = median(Analyst)
gen hi_an2 = (Analyst >= med_an2) if Analyst != .
reghdfe PriceDelay c.DU_kw##i.hi_an2 $controls, absorb(Ind2_num year_num) cluster(IndYear_num)
di "Analyst x DU_kw (ind FE): b=" %9.4f _b[c.DU_kw#1.hi_an2] " t=" %6.2f (_b[c.DU_kw#1.hi_an2]/_se[c.DU_kw#1.hi_an2]) " p=" %6.4f (2*ttail(e(df_r), abs(_b[c.DU_kw#1.hi_an2]/_se[c.DU_kw#1.hi_an2])))

* SOE
reghdfe PriceDelay c.DU_kw##i.SOE $controls, absorb(Ind2_num year_num) cluster(IndYear_num)
di "SOE x DU_kw (ind FE): b=" %9.4f _b[c.DU_kw#1.SOE] " t=" %6.2f (_b[c.DU_kw#1.SOE]/_se[c.DU_kw#1.SOE]) " p=" %6.4f (2*ttail(e(df_r), abs(_b[c.DU_kw#1.SOE]/_se[c.DU_kw#1.SOE])))

* Intangible
reghdfe PriceDelay c.DU_kw##i.Hi_Intang $controls, absorb(Ind2_num year_num) cluster(IndYear_num)
di "Intangible x DU_kw (ind FE): b=" %9.4f _b[c.DU_kw#1.Hi_Intang] " t=" %6.2f (_b[c.DU_kw#1.Hi_Intang]/_se[c.DU_kw#1.Hi_Intang]) " p=" %6.4f (2*ttail(e(df_r), abs(_b[c.DU_kw#1.Hi_Intang]/_se[c.DU_kw#1.Hi_Intang])))

* Market
reghdfe PriceDelay c.DU_kw##i.Hi_Market $controls, absorb(Ind2_num year_num) cluster(IndYear_num)
di "Market x DU_kw (ind FE): b=" %9.4f _b[c.DU_kw#1.Hi_Market] " t=" %6.2f (_b[c.DU_kw#1.Hi_Market]/_se[c.DU_kw#1.Hi_Market]) " p=" %6.4f (2*ttail(e(df_r), abs(_b[c.DU_kw#1.Hi_Market]/_se[c.DU_kw#1.Hi_Market])))

di ""
di "============================================="
di "DONE Round 2"
di "============================================="
