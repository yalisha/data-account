# LLM 人工复核结果汇总

- 总样本数: 120
- 已填写人工评分: 120
- 未填写人工评分: 0
- 标记需要二次裁决: 29

## 人工评分分布
- 0 分: 56
- 1 分: 24
- 2 分: 31
- 3 分: 9

## 与归档分值比较
- exact agreement: 0.325
- weighted kappa: 0.157

## 与独立 Codex 分值比较
- exact agreement: 0.650
- weighted kappa: 0.724

## 人工置信度分布
- high: 59
- medium: 59
- low: 2

## 说明
- `archived_parent_score` 是原归档企业-年份众数分值。
- `codex_independent_score` 是独立 Codex worker 的样本级复核分值。
- 混淆矩阵见 `llm_manual_review_confusion.csv`。
- 分歧样本清单见 `llm_manual_review_disagreements.csv`。
