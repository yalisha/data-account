# v15 Measurement Upgrade Summary

## Baseline

- `DU_kw_lag`: -0.0039*** (t=-3.87, p=0.0001)

## Measurement Upgrade

- `DUevent_lag`: -0.0023 (t=-1.51, p=0.1318)
- `DUevent_ratio_lag`: -0.0016 (t=-0.79, p=0.4280)
- `DUchain_count_lag`: -0.0018** (t=-2.13, p=0.0338)
- `DUclosedloop_lag`: -0.0083*** (t=-2.74, p=0.0062)
- `DUcore_lag`: -0.0095*** (t=-3.46, p=0.0006)
- `DUkw_mda_lag`: -0.0006*** (t=-2.95, p=0.0032)

### Joint Models

- `DUevent_lag + DU_kw_lag`
  - `DUevent_lag`: +0.0078*** (t=2.98, p=0.0030)
  - `DU_kw_lag`: -0.0085*** (t=-5.09, p=0.0000)
- `DUevent_lag + GenericNarr_lag`
  - `DUevent_lag`: +0.0045** (t=2.00, p=0.0463)
  - `GenericNarr_lag`: -0.0083*** (t=-4.76, p=0.0000)
- `DUchain_count_lag + DU_kw_lag`
  - `DUchain_count_lag`: -0.0006 (t=-0.67, p=0.5061)
  - `DU_kw_lag`: -0.0037*** (t=-3.61, p=0.0003)
- `DUclosedloop_lag + DU_kw_lag`
  - `DUclosedloop_lag`: -0.0043 (t=-1.51, p=0.1317)
  - `DU_kw_lag`: -0.0035*** (t=-3.58, p=0.0004)
- `DUclosedloop_lag + GenericNarr_lag`
  - `DUclosedloop_lag`: -0.0051* (t=-1.77, p=0.0771)
  - `GenericNarr_lag`: -0.0046*** (t=-3.89, p=0.0001)
- `DUcore_lag + DU_kw_lag`
  - `DUcore_lag`: -0.0013 (t=-0.46, p=0.6445)
  - `DU_kw_lag`: -0.0036*** (t=-3.15, p=0.0017)
- `DUcore_lag + GenericNarr_lag`
  - `DUcore_lag`: -0.0029 (t=-1.11, p=0.2678)
  - `GenericNarr_lag`: -0.0044*** (t=-3.52, p=0.0004)
- `DUkw_mda_lag + DU_kw_lag`
  - `DUkw_mda_lag`: +0.0002 (t=0.48, p=0.6300)
  - `DU_kw_lag`: -0.0046** (t=-2.53, p=0.0114)
- `DUkw_mda_lag + GenericNarr_lag`
  - `DUkw_mda_lag`: -0.0000 (t=-0.01, p=0.9932)
  - `GenericNarr_lag`: -0.0050** (t=-2.53, p=0.0117)

## Dimension Decomposition

- `DU_stock_lag`: -0.0124*** (t=-4.09, p=0.0000)
- `DU_dev_lag`: -0.0048*** (t=-2.97, p=0.0030)
- `DU_app_lag`: -0.0116*** (t=-4.46, p=0.0000)
- `DU_value_lag`: -0.0617 (t=-1.51, p=0.1325)
- `DU_gov_lag`: -0.0427* (t=-1.66, p=0.0969)

## Takeaways

- `DUclosedloop` and `DUchain_count` provide the cleanest supportive evidence that a more complete data-use chain is associated with lower stock price delay.
- `DUkw_mda` is significantly negative in the preferred lagged specification, suggesting that operating-section disclosure carries pricing content, but it does not survive joint models with `DU_kw` or `GenericNarr`.
- `DUevent` and `DUevent_ratio` do not provide stable standalone support in the current archived feature pipeline and should remain supplementary.
- `DUcore` is negative on a standalone basis but does not displace `DU_kw` once entered jointly, so it should be treated as a narrower supportive proxy rather than a replacement baseline.
