* =============================================================
* Table 7: Heterogeneity Analysis (New Dimensions)
* Data: reg_sample_het.dta (= reg_sample_iv_v16 + Intangible/Market/Big4)
* =============================================================
clear all
set more off
use "data_stata/reg_sample_het.dta", clear

global controls Size Lev ROA TobinQ Age Growth IndepRatio Dual Top1Share SOE CFO

* ---- Construct grouping variables ----
gen Post2020 = (year_num >= 2020)

* Intangible median split
egen med_intang = median(Intangible)
gen Hi_Intang = (Intangible >= med_intang) if Intangible != .

* Market median split
egen med_mkt = median(Market)
gen Hi_Market = (Market >= med_mkt) if Market != .

di "============================================="
di "Sample overview"
di "============================================="
tab Post2020
tab Hi_Intang
tab Hi_Market
tab Big4

* =============================================================
* Part A: Time dimension (Pre/Post 2020)
* =============================================================
di ""
di "============================================="
di "Part A: Time Dimension (Pre/Post 2020)"
di "============================================="

reghdfe PriceDelay DU_kw $controls if Post2020 == 0, absorb(Stkcd_num year_num) cluster(IndYear_num)
di "Pre-2020: b=" %9.4f _b[DU_kw] " se=" %9.4f _se[DU_kw] " t=" %6.2f (_b[DU_kw]/_se[DU_kw]) " N=" e(N)

reghdfe PriceDelay DU_kw $controls if Post2020 == 1, absorb(Stkcd_num year_num) cluster(IndYear_num)
di "Post-2020: b=" %9.4f _b[DU_kw] " se=" %9.4f _se[DU_kw] " t=" %6.2f (_b[DU_kw]/_se[DU_kw]) " N=" e(N)

reghdfe PriceDelay c.DU_kw##i.Post2020 $controls, absorb(Stkcd_num year_num) cluster(IndYear_num)
di "Time interaction: b=" %9.4f _b[c.DU_kw#1.Post2020] " se=" %9.4f _se[c.DU_kw#1.Post2020] " t=" %6.2f (_b[c.DU_kw#1.Post2020]/_se[c.DU_kw#1.Post2020]) " p=" %6.4f (2*ttail(e(df_r), abs(_b[c.DU_kw#1.Post2020]/_se[c.DU_kw#1.Post2020])))

* DU_llm
reghdfe PriceDelay DU_llm $controls if Post2020 == 0, absorb(Stkcd_num year_num) cluster(IndYear_num)
di "Pre-2020 (llm): b=" %9.4f _b[DU_llm] " se=" %9.4f _se[DU_llm] " t=" %6.2f (_b[DU_llm]/_se[DU_llm]) " N=" e(N)

reghdfe PriceDelay DU_llm $controls if Post2020 == 1, absorb(Stkcd_num year_num) cluster(IndYear_num)
di "Post-2020 (llm): b=" %9.4f _b[DU_llm] " se=" %9.4f _se[DU_llm] " t=" %6.2f (_b[DU_llm]/_se[DU_llm]) " N=" e(N)

reghdfe PriceDelay c.DU_llm##i.Post2020 $controls, absorb(Stkcd_num year_num) cluster(IndYear_num)
di "Time interaction (llm): b=" %9.4f _b[c.DU_llm#1.Post2020] " se=" %9.4f _se[c.DU_llm#1.Post2020] " t=" %6.2f (_b[c.DU_llm#1.Post2020]/_se[c.DU_llm#1.Post2020]) " p=" %6.4f (2*ttail(e(df_r), abs(_b[c.DU_llm#1.Post2020]/_se[c.DU_llm#1.Post2020])))

* =============================================================
* Part B: Intangible Assets Intensity
* =============================================================
di ""
di "============================================="
di "Part B: Intangible Assets Intensity"
di "============================================="

reghdfe PriceDelay DU_kw $controls if Hi_Intang == 1, absorb(Stkcd_num year_num) cluster(IndYear_num)
di "Hi Intangible: b=" %9.4f _b[DU_kw] " se=" %9.4f _se[DU_kw] " t=" %6.2f (_b[DU_kw]/_se[DU_kw]) " N=" e(N)

reghdfe PriceDelay DU_kw $controls if Hi_Intang == 0, absorb(Stkcd_num year_num) cluster(IndYear_num)
di "Lo Intangible: b=" %9.4f _b[DU_kw] " se=" %9.4f _se[DU_kw] " t=" %6.2f (_b[DU_kw]/_se[DU_kw]) " N=" e(N)

reghdfe PriceDelay c.DU_kw##i.Hi_Intang $controls, absorb(Stkcd_num year_num) cluster(IndYear_num)
di "Intangible interaction: b=" %9.4f _b[c.DU_kw#1.Hi_Intang] " se=" %9.4f _se[c.DU_kw#1.Hi_Intang] " t=" %6.2f (_b[c.DU_kw#1.Hi_Intang]/_se[c.DU_kw#1.Hi_Intang]) " p=" %6.4f (2*ttail(e(df_r), abs(_b[c.DU_kw#1.Hi_Intang]/_se[c.DU_kw#1.Hi_Intang])))

* DU_llm
reghdfe PriceDelay DU_llm $controls if Hi_Intang == 1, absorb(Stkcd_num year_num) cluster(IndYear_num)
di "Hi Intangible (llm): b=" %9.4f _b[DU_llm] " se=" %9.4f _se[DU_llm] " t=" %6.2f (_b[DU_llm]/_se[DU_llm]) " N=" e(N)

reghdfe PriceDelay DU_llm $controls if Hi_Intang == 0, absorb(Stkcd_num year_num) cluster(IndYear_num)
di "Lo Intangible (llm): b=" %9.4f _b[DU_llm] " se=" %9.4f _se[DU_llm] " t=" %6.2f (_b[DU_llm]/_se[DU_llm]) " N=" e(N)

reghdfe PriceDelay c.DU_llm##i.Hi_Intang $controls, absorb(Stkcd_num year_num) cluster(IndYear_num)
di "Intangible interaction (llm): b=" %9.4f _b[c.DU_llm#1.Hi_Intang] " se=" %9.4f _se[c.DU_llm#1.Hi_Intang] " t=" %6.2f (_b[c.DU_llm#1.Hi_Intang]/_se[c.DU_llm#1.Hi_Intang]) " p=" %6.4f (2*ttail(e(df_r), abs(_b[c.DU_llm#1.Hi_Intang]/_se[c.DU_llm#1.Hi_Intang])))

* =============================================================
* Part C: Marketization Index
* =============================================================
di ""
di "============================================="
di "Part C: Marketization Index"
di "============================================="

reghdfe PriceDelay DU_kw $controls if Hi_Market == 1, absorb(Stkcd_num year_num) cluster(IndYear_num)
di "Hi Market: b=" %9.4f _b[DU_kw] " se=" %9.4f _se[DU_kw] " t=" %6.2f (_b[DU_kw]/_se[DU_kw]) " N=" e(N)

reghdfe PriceDelay DU_kw $controls if Hi_Market == 0, absorb(Stkcd_num year_num) cluster(IndYear_num)
di "Lo Market: b=" %9.4f _b[DU_kw] " se=" %9.4f _se[DU_kw] " t=" %6.2f (_b[DU_kw]/_se[DU_kw]) " N=" e(N)

reghdfe PriceDelay c.DU_kw##i.Hi_Market $controls, absorb(Stkcd_num year_num) cluster(IndYear_num)
di "Market interaction: b=" %9.4f _b[c.DU_kw#1.Hi_Market] " se=" %9.4f _se[c.DU_kw#1.Hi_Market] " t=" %6.2f (_b[c.DU_kw#1.Hi_Market]/_se[c.DU_kw#1.Hi_Market]) " p=" %6.4f (2*ttail(e(df_r), abs(_b[c.DU_kw#1.Hi_Market]/_se[c.DU_kw#1.Hi_Market])))

* DU_llm
reghdfe PriceDelay DU_llm $controls if Hi_Market == 1, absorb(Stkcd_num year_num) cluster(IndYear_num)
di "Hi Market (llm): b=" %9.4f _b[DU_llm] " se=" %9.4f _se[DU_llm] " t=" %6.2f (_b[DU_llm]/_se[DU_llm]) " N=" e(N)

reghdfe PriceDelay DU_llm $controls if Hi_Market == 0, absorb(Stkcd_num year_num) cluster(IndYear_num)
di "Lo Market (llm): b=" %9.4f _b[DU_llm] " se=" %9.4f _se[DU_llm] " t=" %6.2f (_b[DU_llm]/_se[DU_llm]) " N=" e(N)

reghdfe PriceDelay c.DU_llm##i.Hi_Market $controls, absorb(Stkcd_num year_num) cluster(IndYear_num)
di "Market interaction (llm): b=" %9.4f _b[c.DU_llm#1.Hi_Market] " se=" %9.4f _se[c.DU_llm#1.Hi_Market] " t=" %6.2f (_b[c.DU_llm#1.Hi_Market]/_se[c.DU_llm#1.Hi_Market]) " p=" %6.4f (2*ttail(e(df_r), abs(_b[c.DU_llm#1.Hi_Market]/_se[c.DU_llm#1.Hi_Market])))

* =============================================================
* Part D: Big4 Audit (supplementary)
* =============================================================
di ""
di "============================================="
di "Part D: Big4 Audit"
di "============================================="

* Big4=1 subsample: use industry FE (too few firms for firm FE)
reghdfe PriceDelay DU_kw $controls if Big4 == 1, absorb(Ind2_num year_num) cluster(IndYear_num)
di "Big4=1 (ind FE): b=" %9.4f _b[DU_kw] " se=" %9.4f _se[DU_kw] " t=" %6.2f (_b[DU_kw]/_se[DU_kw]) " N=" e(N)

reghdfe PriceDelay DU_kw $controls if Big4 == 0, absorb(Stkcd_num year_num) cluster(IndYear_num)
di "Big4=0: b=" %9.4f _b[DU_kw] " se=" %9.4f _se[DU_kw] " t=" %6.2f (_b[DU_kw]/_se[DU_kw]) " N=" e(N)

* Full sample interaction
reghdfe PriceDelay c.DU_kw##i.Big4 $controls, absorb(Stkcd_num year_num) cluster(IndYear_num)
di "Big4 interaction: b=" %9.4f _b[c.DU_kw#1.Big4] " se=" %9.4f _se[c.DU_kw#1.Big4] " t=" %6.2f (_b[c.DU_kw#1.Big4]/_se[c.DU_kw#1.Big4]) " p=" %6.4f (2*ttail(e(df_r), abs(_b[c.DU_kw#1.Big4]/_se[c.DU_kw#1.Big4])))

* =============================================================
* Part E: Continuous Interactions
* =============================================================
di ""
di "============================================="
di "Part E: Continuous Interactions"
di "============================================="

reghdfe PriceDelay c.DU_kw##c.Intangible $controls, absorb(Stkcd_num year_num) cluster(IndYear_num)
di "DU_kw x Intangible (cont): b=" %9.4f _b[c.DU_kw#c.Intangible] " se=" %9.4f _se[c.DU_kw#c.Intangible] " t=" %6.2f (_b[c.DU_kw#c.Intangible]/_se[c.DU_kw#c.Intangible]) " p=" %6.4f (2*ttail(e(df_r), abs(_b[c.DU_kw#c.Intangible]/_se[c.DU_kw#c.Intangible])))

reghdfe PriceDelay c.DU_kw##c.Market $controls, absorb(Stkcd_num year_num) cluster(IndYear_num)
di "DU_kw x Market (cont): b=" %9.4f _b[c.DU_kw#c.Market] " se=" %9.4f _se[c.DU_kw#c.Market] " t=" %6.2f (_b[c.DU_kw#c.Market]/_se[c.DU_kw#c.Market]) " p=" %6.4f (2*ttail(e(df_r), abs(_b[c.DU_kw#c.Market]/_se[c.DU_kw#c.Market])))

reghdfe PriceDelay c.DU_kw##c.Analyst $controls, absorb(Stkcd_num year_num) cluster(IndYear_num)
di "DU_kw x Analyst (cont): b=" %9.4f _b[c.DU_kw#c.Analyst] " se=" %9.4f _se[c.DU_kw#c.Analyst] " t=" %6.2f (_b[c.DU_kw#c.Analyst]/_se[c.DU_kw#c.Analyst]) " p=" %6.4f (2*ttail(e(df_r), abs(_b[c.DU_kw#c.Analyst]/_se[c.DU_kw#c.Analyst])))

reghdfe PriceDelay c.DU_kw##c.Size $controls, absorb(Stkcd_num year_num) cluster(IndYear_num)
di "DU_kw x Size (cont): b=" %9.4f _b[c.DU_kw#c.Size] " se=" %9.4f _se[c.DU_kw#c.Size] " t=" %6.2f (_b[c.DU_kw#c.Size]/_se[c.DU_kw#c.Size]) " p=" %6.4f (2*ttail(e(df_r), abs(_b[c.DU_kw#c.Size]/_se[c.DU_kw#c.Size])))

* =============================================================
* Part F: Legacy dimensions (for transparency)
* =============================================================
di ""
di "============================================="
di "Part F: Legacy Dimensions (SOE/Size/Analyst/Industry)"
di "============================================="

* SOE
reghdfe PriceDelay c.DU_kw##i.SOE $controls, absorb(Stkcd_num year_num) cluster(IndYear_num)
di "SOE interaction: b=" %9.4f _b[c.DU_kw#1.SOE] " se=" %9.4f _se[c.DU_kw#1.SOE] " t=" %6.2f (_b[c.DU_kw#1.SOE]/_se[c.DU_kw#1.SOE]) " p=" %6.4f (2*ttail(e(df_r), abs(_b[c.DU_kw#1.SOE]/_se[c.DU_kw#1.SOE])))

* Size
egen med_size = median(Size)
gen big = (Size >= med_size) if Size != .
reghdfe PriceDelay c.DU_kw##i.big $controls, absorb(Stkcd_num year_num) cluster(IndYear_num)
di "Size interaction: b=" %9.4f _b[c.DU_kw#1.big] " se=" %9.4f _se[c.DU_kw#1.big] " t=" %6.2f (_b[c.DU_kw#1.big]/_se[c.DU_kw#1.big]) " p=" %6.4f (2*ttail(e(df_r), abs(_b[c.DU_kw#1.big]/_se[c.DU_kw#1.big])))

* Analyst
egen med_analyst = median(Analyst)
gen hi_analyst = (Analyst >= med_analyst) if Analyst != .
reghdfe PriceDelay c.DU_kw##i.hi_analyst $controls, absorb(Stkcd_num year_num) cluster(IndYear_num)
di "Analyst interaction: b=" %9.4f _b[c.DU_kw#1.hi_analyst] " se=" %9.4f _se[c.DU_kw#1.hi_analyst] " t=" %6.2f (_b[c.DU_kw#1.hi_analyst]/_se[c.DU_kw#1.hi_analyst]) " p=" %6.4f (2*ttail(e(df_r), abs(_b[c.DU_kw#1.hi_analyst]/_se[c.DU_kw#1.hi_analyst])))

* Industry
gen hitech = inlist(Ind2_num, 60, 61, 62, 70, 71, 72)
reghdfe PriceDelay c.DU_kw##i.hitech $controls, absorb(Stkcd_num year_num) cluster(IndYear_num)
di "Industry interaction: b=" %9.4f _b[c.DU_kw#1.hitech] " se=" %9.4f _se[c.DU_kw#1.hitech] " t=" %6.2f (_b[c.DU_kw#1.hitech]/_se[c.DU_kw#1.hitech]) " p=" %6.4f (2*ttail(e(df_r), abs(_b[c.DU_kw#1.hitech]/_se[c.DU_kw#1.hitech])))

di ""
di "============================================="
di "DONE"
di "============================================="
