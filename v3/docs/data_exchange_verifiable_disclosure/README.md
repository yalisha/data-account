# 数据交易所与可核验披露：当前主线文档入口

更新日期：2026-05-07

这个目录是当前主线版本。`docs/research_memos/` 继续作为试跑记录、证据档案和历史分支，不再作为合作者或网页版 Pro 的首读材料。

## 阅读顺序

1. `00_research_plan.md`  
   总体研究设计、贡献边界、识别框架、当前结果层级和下一步计划。

2. `01_x_pre_data_washing_exposure.md`  
   解释 X：事前 data-washing 暴露如何由“数据要素叙事 - 可验证硬能力”构造。

3. `02_y_verifiable_disclosure.md`  
   解释 Y：剔除机构名词后的资产化、交易化、可核验数据要素披露如何构造，以及为什么它比旧 `DU_kw` 更适合主文。

4. `03_mechanisms_and_validation.md`  
   解释机制、识别检验、组件检验、真实数据资产入表验证和保守边界。

## 当前一句话版本

本文研究城市数据交易所这一数据要素市场基础设施，是否促使事前存在 data-washing 暴露的上市公司，从泛化数据叙事转向更具资产化、交易化和可核验特征的数据要素披露。

## 当前主变量层级

```text
X: DID_city_ct x PreWashAny1617 / PreStrictWashAny1617

主 Y:
  asset_trade_noinst_kw
  strict_noinst_kw

补充 Y:
  verif_noinst_kw
  verif_noinst_share
  product_tx_noinst_kw
  acct_kw
  pricing_kw

验证层:
  BookDataResource_bs
  lnBookDataResource_bs
  BookEntry / DataAsset_ln

支持证据:
  NarrHardGap_direct_v1
  NarrHardWashing_direct_v1
```

## 不再主推的方向

- 不把 `DU_kw` 当主 Y；它太宽，更适合作 L0 泛数据叙事对照。
- 不把 `NarrHardGap_direct_v1` / `NarrHardWashing_direct_v1` 当唯一主 Y；静态显著但动态 pretrend 不干净。
- 不把真实 `DataAsset` 入表当长期主 Y；正值基本集中在 2024 年，适合验证，不适合主 DID。
- 不把“数据交易所/数据交易中心”机构名词放进主 Y；新版 Y 已先屏蔽机构名词。

## 关键证据文件

```text
src/python/build_verifiable_disclosure_y_noinst_v1.py
src/python/extract_balance_sheet_data_resource_for_stata.py
src/stata/did_data_exchange_verif_noinst_y_v1.do
src/stata/did_data_exchange_actual_data_asset_validation_v1.do
src/stata/did_data_exchange_bs_data_resource_validation_v1.do
results/stata/did_data_exchange_verif_noinst_y_v1_static.csv
results/stata/did_data_exchange_verif_noinst_y_v1_pretrend_joint.csv
results/stata/did_data_exchange_verif_noinst_y_v1_dataasset_validation.csv
results/stata/did_data_exchange_bs_data_resource_validation_v1_measurement.csv
docs/research_memos/verif_noinst_y_trial_20260506.md
docs/research_memos/actual_data_asset_validation_20260506.md
docs/research_memos/bs_data_resource_validation_20260507.md
```
