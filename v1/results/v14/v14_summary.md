# v14 Results Summary

## Construct Boundary

- `DU_kw_lag + ln_total_chars_lag`: β=-0.0038, t=-3.80
- `DU_kw_lag + ln_mda_chars_lag`: β=-0.0039, t=-3.89
- Joint lag model:
  - `DU_kw_lag`: β=-0.0027, t=-1.29
  - `GenericNarr_lag`: β=-0.0017, t=-0.75
  - FE-adjusted VIF: DU=7.651, GenericNarr=7.611
- `DU_kw_resid_lag`: β=-0.0027, t=-1.29
- `DU_kw_strict_lag`: β=-0.0077, t=-2.61

## WashGap

- `WashGap_lag`: β=+0.0035, t=2.09
- `BroadShallow_lag`: β=+0.0046, t=1.28
- Joint depth model:
  - `DU_kw_lag`: β=-0.0016, t=-1.26
  - `DU_llm_lenstd_lag`: β=-0.0146, t=-3.76

## Heterogeneity

- `Post2021`: interaction β=+0.0009, p=0.4483
- `Post2022`: interaction β=+0.0026, p=0.0307
- `Post2021_ex2020`: interaction β=-0.0001, p=0.9402
- `DigEconCore`: interaction β=-0.0012, p=0.5069
