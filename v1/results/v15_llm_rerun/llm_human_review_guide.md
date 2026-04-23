# v15 LLM rerun review guide

This pack is for manual review of the 120 stratified contexts.

## Scoring scale

- `0`: boilerplate, location, tax, subsidy, subsidiary name, policy quotation, or other generic mention without a substantive data-use action.
- `1`: data-related wording or digital buzzwords appear, but the context is still generic, organizational, or descriptive rather than a concrete business/governance use.
- `2`: a concrete data-use action is visible, but the context is still partial, descriptive, or not fully embedded in an operating/gov scenario.
- `3`: explicit substantive use. The same context shows a data object plus a concrete action plus a business, operational, or governance scenario.

## Review rule

- Score the context itself. Ignore the parent LLM score when you judge the rerun.
- If the passage only names a project, subsidiary, policy, or product, keep it at `0` or `1`.
- If one element is present but the use case is vague, keep it at `1`.
- If the passage makes a clear operational or governance use case, use `2`.
- If the passage clearly shows data being used in business operations, governance, or decision-making, use `3`.

## File layout

- `llm_human_review_pack.csv`: all 120 contexts with blank `human_score` and `human_note` columns.
- `llm_rerun_prompt_template.txt`: exact prompt used by the independent Codex rerun.
- `batches/`: batch payloads for each Codex call.
- `codex_outputs/`: raw batch outputs from Codex.

## Suggested manual review workflow

1. Open `llm_human_review_pack.csv`.
2. Fill in `human_score` and `human_note`.
3. Use the summary file to compare your scores with the rerun outputs.
