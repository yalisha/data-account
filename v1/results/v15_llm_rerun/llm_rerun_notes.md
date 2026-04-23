# v15 LLM rerun notes

- The independent `codex exec` batches completed with nonzero exit codes because the CLI emits a Linear MCP authentication warning in `stderr`.
- The warning did not prevent `--output-last-message` from being written.
- All 15 batch outputs were parsed successfully into `llm_rerun_context_scores.csv` and `llm_rerun_context_scores.jsonl`.
- The batch-specific `*.status.txt` files record the nonzero exit code for traceability.
