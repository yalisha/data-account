# v3 docs index

更新日期：2026-05-06

## 当前主线

当前主线文档已整理到：

```text
docs/data_exchange_verifiable_disclosure/
```

建议阅读顺序：

```text
docs/data_exchange_verifiable_disclosure/README.md
docs/data_exchange_verifiable_disclosure/00_research_plan.md
docs/data_exchange_verifiable_disclosure/01_x_pre_data_washing_exposure.md
docs/data_exchange_verifiable_disclosure/02_y_verifiable_disclosure.md
docs/data_exchange_verifiable_disclosure/03_mechanisms_and_validation.md
```

## 旧 memo 的定位

`docs/research_memos/` 是历史试跑、分支判断和证据记录。它们不再作为当前主线的首读材料。

当前最重要的证据 memo：

```text
docs/research_memos/research_design_data_exchange_verifiable_disclosure_20260506.md
docs/research_memos/data_exchange_verifiable_disclosure_pro_checks_20260506.md
docs/research_memos/actual_data_asset_validation_20260506.md
docs/research_memos/verif_noinst_y_trial_20260506.md
```

## 当前结论

主线从旧的“年报数据要素披露作为 X”转为：

```text
城市数据交易所设立 x 事前 data-washing 暴露
-> 企业可核验数据要素披露
```

主 Y 已从旧 `DU_kw / asset_trade_kw` 更新为：

```text
asset_trade_noinst_kw
strict_noinst_kw
```

`DU_kw` 只作为泛数据叙事对照；`BookDataResource_bs / lnBookDataResource_bs` 和旧 `BookEntry / DataAsset_ln` 只作为 2024 年真实入表验证层。
