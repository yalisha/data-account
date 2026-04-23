# Codex 独立复核说明

## 适用文件

- 原始样本：`results/v14/llm_validation_sample.csv`
- 独立评分：`results/v15_llm_review/codex_independent_rerun_scores.csv`
- 手工复核指南：`results/v15_llm_review/llm_manual_review_guide.md`

## 判分标准

- `0`：口号、背景、定义、名单、政策复述，未见企业层面的具体应用。
- `1`：出现建设方向、能力提升、推进转型等表述，但缺少清晰落地场景。
- `2`：出现明确的数据应用场景、治理动作或决策用途。
- `3`：在 `2` 的基础上，进一步出现系统嵌入、流程闭环、量化结果或价值实现。

## 本次判分原则

- 只依据样本中的 `keyword` 和 `context`。
- 不使用企业外部信息，不参考样本归档分值。
- 若介于两档之间，取较低分。
- `codex_notes` 仅对歧义样本写简短原因。

