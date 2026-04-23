clear all
use "data_stata/reg_sample_iv_v16.dta", clear
describe, short
describe Analyst Amihud RetVol Turnover InstHold
describe DU_kw DU_llm DU_sub_ln llm_binary
describe IndYear_num Stkcd_num year_num Ind2_num
* Check industry codes for high-tech filter
tab Ind2_num if Ind2_num != ., nolabel summarize(DU_kw)
* Check year range
tab year_num, nolabel
