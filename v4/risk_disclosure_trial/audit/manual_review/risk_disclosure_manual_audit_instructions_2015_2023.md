# 风险披露抽取人工核验说明

样本文件：`/Users/mac/computerscience/0做完了/15会计研究/v4/risk_disclosure_trial/risk_disclosure_manual_audit_sheet_2015_2023.csv`

样本量：209 条。

## 核验列填写规则

- `manual_valid_risk_section`：1=确实是年报风险披露段，0=误抓。
- `manual_false_positive_type`：若误抓，填 `financial_instrument`、`footnote`、`governance`、`bond_guarantee`、`litigation_note`、`other`。
- `manual_verifiable_true`：1=确实包含可外部核验的具体风险信息，0=主要是模板化/空泛表述。
- `manual_np_liability_true`：1=确实涉及自然人责任主体、个人担保、控制人质押、追偿/代偿/控制权风险，0=不涉及。
- `manual_notes`：记录明显问题，如风险段太短、标题误抓、金融工具附注、只出现政策名词等。

## 抽样层

- `top_verifiable`：可核验披露指数最高的样本。
- `top_np_liability`：自然人责任可核验指数最高的样本。
- `suspect_header`：标题可能误抓的样本，用来估计误抓率。
- `random_has_risk`：随机有风险段样本。
- `random_no_risk`：随机无风险段样本，用来检查漏抓。
