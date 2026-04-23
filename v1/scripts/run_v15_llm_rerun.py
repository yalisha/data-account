#!/usr/bin/env python3
"""
Build and optionally rerun the v15 LLM validation sample with independent Codex agents.

Outputs are written under results/v15_llm_rerun/.

Artifacts:
  - llm_human_review_pack.csv: 120 contexts with blank human review columns
  - llm_human_review_guide.md: scoring rubric and review instructions
  - llm_rerun_prompt_template.txt: exact prompt template used for Codex rerun
  - llm_rerun_schema.json: JSON schema for Codex batch outputs
  - batches/batch_XXX.json: batch payloads
  - codex_outputs/batch_XXX.json: Codex last-message outputs
  - llm_rerun_context_scores.csv/jsonl: combined rerun scores
  - llm_rerun_summary.md: summary and agreement statistics
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

BASE = Path(__file__).resolve().parents[1]
SAMPLE_PATH = BASE / "results" / "v14" / "llm_validation_sample.csv"
OUT_DIR = BASE / "results" / "v15_llm_rerun"
BATCH_DIR = OUT_DIR / "batches"
OUTPUT_DIR = OUT_DIR / "codex_outputs"
LOG_DIR = OUT_DIR / "logs"

DEFAULT_BATCH_SIZE = 8
DEFAULT_MODEL = "gpt-5.4"

RUBRIC_MD = """# v15 LLM rerun review guide

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
"""

PROMPT_TEMPLATE = """You are an independent reviewer scoring Chinese annual report contexts for a 0-3 semantic depth rubric.

Goal: score each context only by its own text. Ignore the parent LLM score and any previous labels.

Rubric:
- 0 = boilerplate, location, tax, subsidy, subsidiary name, policy quotation, or other generic mention without a substantive data-use action.
- 1 = data-related wording or digital buzzwords appear, but the context is still generic, organizational, or descriptive rather than a concrete business/governance use.
- 2 = a concrete data-use action is visible, but the context is still partial, descriptive, or not fully embedded in an operating/governance scenario.
- 3 = explicit substantive use. The same context shows a data object plus a concrete action plus a business, operational, or governance scenario.

Return ONLY valid JSON matching the schema. Do not wrap in markdown.

Batch id: {batch_id}
Items:
{items_json}
"""

SCHEMA = {
    "type": "object",
    "required": ["batch_id", "results"],
    "additionalProperties": False,
    "properties": {
        "batch_id": {"type": "string"},
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["sample_id", "score", "reason", "confidence"],
                "additionalProperties": False,
                "properties": {
                    "sample_id": {"type": "string"},
                    "score": {"type": "integer", "minimum": 0, "maximum": 3},
                    "reason": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
            },
        },
    },
}


@dataclass(frozen=True)
class Batch:
    batch_id: str
    rows: list[dict]


def ensure_dirs() -> None:
    for path in [OUT_DIR, BATCH_DIR, OUTPUT_DIR, LOG_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def load_sample() -> pd.DataFrame:
    if not SAMPLE_PATH.exists():
        raise FileNotFoundError(f"Missing sample file: {SAMPLE_PATH}")
    df = pd.read_csv(SAMPLE_PATH)
    expected = {
        "sample_id",
        "Stkcd",
        "year",
        "parent_llm_score",
        "keyword",
        "generic_bucket",
        "score_bucket",
        "density_bucket",
        "window_hits",
        "context",
    }
    missing = expected.difference(df.columns)
    if missing:
        raise ValueError(f"Sample file missing columns: {sorted(missing)}")
    return df


def make_review_pack(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["human_score"] = pd.NA
    out["human_note"] = pd.NA
    out["codex_score"] = pd.NA
    out["codex_reason"] = pd.NA
    out["codex_confidence"] = pd.NA
    cols = [
        "sample_id",
        "Stkcd",
        "year",
        "parent_llm_score",
        "keyword",
        "generic_bucket",
        "score_bucket",
        "density_bucket",
        "window_hits",
        "context",
        "human_score",
        "human_note",
        "codex_score",
        "codex_reason",
        "codex_confidence",
    ]
    return out[cols]


def chunk_rows(rows: list[dict], batch_size: int) -> list[Batch]:
    batches = []
    for idx in range(0, len(rows), batch_size):
        batch_rows = rows[idx : idx + batch_size]
        batches.append(Batch(batch_id=f"batch_{len(batches)+1:03d}", rows=batch_rows))
    return batches


def write_support_files(review_pack: pd.DataFrame, batch_size: int) -> list[Batch]:
    review_pack.to_csv(OUT_DIR / "llm_human_review_pack.csv", index=False)
    try:
        rubric_df = pd.DataFrame(
            [
                ("0", "boilerplate, location, tax, subsidy, subsidiary name, policy quotation, or other generic mention without a substantive data-use action."),
                ("1", "data-related wording or digital buzzwords appear, but the context is still generic, organizational, or descriptive rather than a concrete business/governance use."),
                ("2", "a concrete data-use action is visible, but the context is still partial, descriptive, or not fully embedded in an operating/governance scenario."),
                ("3", "explicit substantive use. The same context shows a data object plus a concrete action plus a business, operational, or governance scenario."),
            ],
            columns=["score", "definition"],
        )
        with pd.ExcelWriter(OUT_DIR / "llm_human_review_pack.xlsx", engine="openpyxl") as writer:
            review_pack.to_excel(writer, index=False, sheet_name="review")
            rubric_df.to_excel(writer, index=False, sheet_name="rubric")
    except Exception as exc:
        (LOG_DIR / "xlsx_warning.txt").write_text(f"Could not write Excel review pack: {exc}", encoding="utf-8")
    (OUT_DIR / "llm_human_review_guide.md").write_text(RUBRIC_MD, encoding="utf-8")
    (OUT_DIR / "llm_rerun_prompt_template.txt").write_text(PROMPT_TEMPLATE, encoding="utf-8")
    (OUT_DIR / "llm_rerun_schema.json").write_text(json.dumps(SCHEMA, ensure_ascii=False, indent=2), encoding="utf-8")

    rows = review_pack[["sample_id", "keyword", "context"]].to_dict(orient="records")
    batches = chunk_rows(rows, batch_size=batch_size)
    for batch in batches:
        batch_path = BATCH_DIR / f"{batch.batch_id}.json"
        batch_path.write_text(json.dumps({"batch_id": batch.batch_id, "items": batch.rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    return batches


def build_prompt(batch: Batch) -> str:
    payload = {"batch_id": batch.batch_id, "items": batch.rows}
    items_json = json.dumps(payload["items"], ensure_ascii=False, indent=2)
    return PROMPT_TEMPLATE.format(batch_id=batch.batch_id, items_json=items_json)


def run_codex_batch(batch: Batch, model: str, effort: str) -> dict:
    prompt = build_prompt(batch)
    output_path = OUTPUT_DIR / f"{batch.batch_id}.json"
    log_path = LOG_DIR / f"{batch.batch_id}.log"
    cmd = [
        "codex",
        "exec",
        "--full-auto",
        "--sandbox",
        "workspace-write",
        "--skip-git-repo-check",
        "-C",
        str(BASE),
        "-c",
        f"model_reasoning_effort={effort}",
        "-m",
        model,
        "--output-schema",
        str(OUT_DIR / "llm_rerun_schema.json"),
        "--output-last-message",
        str(output_path),
        "-",
    ]
    proc = subprocess.run(
        cmd,
        input=prompt,
        text=True,
        capture_output=True,
        check=False,
    )
    log_path.write_text(
        "STDOUT\n" + proc.stdout + "\n\nSTDERR\n" + proc.stderr,
        encoding="utf-8",
    )
    if not output_path.exists():
        raise RuntimeError(f"Codex batch {batch.batch_id} did not create {output_path}")
    raw = output_path.read_text(encoding="utf-8").strip()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Codex output for {batch.batch_id} is not valid JSON: {exc}\n{raw[:1000]}")
    if proc.returncode != 0:
        (LOG_DIR / f"{batch.batch_id}.status.txt").write_text(
            f"nonzero_exit_code={proc.returncode}\nOutput was still parsed successfully.\nSee {log_path}",
            encoding="utf-8",
        )
    return payload


def flatten_results(batch_payloads: Iterable[dict], parent_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    parent_lookup = parent_df.set_index("sample_id")["parent_llm_score"].to_dict()
    for payload in batch_payloads:
        batch_id = payload["batch_id"]
        for item in payload["results"]:
            sample_id = item["sample_id"]
            rows.append(
                {
                    "batch_id": batch_id,
                    "sample_id": sample_id,
                    "parent_llm_score": parent_lookup.get(sample_id),
                    "codex_score": int(item["score"]),
                    "codex_reason": item["reason"],
                    "codex_confidence": float(item["confidence"]),
                }
            )
    out = pd.DataFrame(rows).sort_values("sample_id").reset_index(drop=True)
    return out


def summarize(df: pd.DataFrame, review_pack: pd.DataFrame) -> str:
    parent = review_pack[["sample_id", "parent_llm_score"]].copy()
    merged = df.merge(parent, on="sample_id", how="left", suffixes=("", "_orig"))
    merged["agree"] = merged["codex_score"] == merged["parent_llm_score"]
    acc = merged["agree"].mean()
    score_counts = merged["codex_score"].value_counts().sort_index()
    parent_counts = merged["parent_llm_score"].value_counts().sort_index()
    stratum_counts = review_pack.groupby(["generic_bucket", "score_bucket", "density_bucket"]).size()
    lines = [
        "# v15 LLM rerun summary",
        "",
        f"- contexts: {len(merged)}",
        f"- exact agreement vs parent score: {acc:.3f}",
        "",
        "## Codex score distribution",
        score_counts.to_string(),
        "",
        "## Parent score distribution",
        parent_counts.to_string(),
        "",
        "## Review strata counts",
        stratum_counts.to_string(),
        "",
        "## Notes",
        "- `parent_llm_score` is only for comparison; the rerun agent does not see it.",
        "- `human_score` and `human_note` are left blank for manual review.",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true", help="Only generate review pack and batch files.")
    parser.add_argument("--run-codex", action="store_true", help="Run codex exec on each batch.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-batches", type=int, default=None, help="Only run the first N batches.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--effort", type=str, default="high", choices=["minimal", "low", "medium", "high"])
    args = parser.parse_args()

    ensure_dirs()
    sample = load_sample()
    review_pack = make_review_pack(sample)
    batches = write_support_files(review_pack, args.batch_size)

    if args.prepare_only and args.run_codex:
        raise SystemExit("--prepare-only and --run-codex are mutually exclusive")

    if args.prepare_only:
        print(f"Prepared review pack at {OUT_DIR}")
        return

    if not args.run_codex:
        print(f"Prepared review pack at {OUT_DIR}. Use --run-codex to execute the independent rerun.")
        return

    payloads = []
    if args.max_batches is not None:
        batches = batches[: max(args.max_batches, 0)]

    for i, batch in enumerate(batches, start=1):
        print(f"[{i}/{len(batches)}] running {batch.batch_id} with {len(batch.rows)} contexts")
        payloads.append(run_codex_batch(batch, model=args.model, effort=args.effort))

    combined = flatten_results(payloads, review_pack)
    combined.to_csv(OUT_DIR / "llm_rerun_context_scores.csv", index=False)
    combined.to_json(OUT_DIR / "llm_rerun_context_scores.jsonl", orient="records", force_ascii=False, lines=True)
    (OUT_DIR / "llm_rerun_summary.md").write_text(summarize(combined, review_pack), encoding="utf-8")
    print(f"Saved rerun outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
