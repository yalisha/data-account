# v15 LLM rerun summary

- contexts: 120
- exact agreement vs parent score: 0.258

## Codex score distribution
codex_score
0    53
1    27
2    16
3    24

## Parent score distribution
parent_llm_score
0    13
1    47
2    55
3     5

## Review strata counts
generic_bucket  score_bucket  density_bucket
generic         high          dense             15
                              sparse            15
                low           dense             15
                              sparse            15
nongeneric      high          dense             15
                              sparse            15
                low           dense             15
                              sparse            15

## Notes
- `parent_llm_score` is only for comparison; the rerun agent does not see it.
- `human_score` and `human_note` are left blank for manual review.