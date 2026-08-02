# MedRouteBench

## Fixed two-stage PubMedQA revision experiment

`staged_eval` implements Idea #1 as a commit-then-revise evaluation:

1. Stage 1 shows the PubMedQA question and preliminary labelled context, then
   requires `{"action": "ANSWER", "answer": "yes|no|maybe"}`.
2. Stage 2 shows the question, the valid Stage 1 output, and the full labelled
   context, then requires `KEEP_ANSWER`, `REVISE_ANSWER`, or `ABSTAIN` with
   action-consistent answer semantics.

The evidence boundary is the first normalized section label containing
`RESULT`. Earlier contexts are preliminary evidence; that section and all later
contexts are added at Stage 2. If no usable results boundary exists, contexts
are split at `len(CONTEXTS) // 2`. Cases that cannot form two non-empty groups
are excluded before model calls.

Only `QUESTION`, `CONTEXTS`, `LABELS`, and the model's own Stage 1 output enter
the prompts. `LONG_ANSWER`, `final_decision`, `test_ground_truth`, and gold
labels are never included. Official labels from
`data/pubmedqa/test_ground_truth.json` are used only for scoring and traces.

Run an evaluation with the built-in Azure adapter:

```bash
python -m staged_eval.pipeline --n 50 --model gpt-4o-mini --inspect 0
```

The selected model is passed to the backend that performs the request and is
recorded with the backend identity in both `manifest.json` and `report.json`.
Shell, scheduler, and CI environment variables take precedence over local
`.env` defaults.

The benchmark pipeline is backend-neutral in Python. Supply any callable with
the signature `(system: str, user: str) -> str` and identify the actual model:

```python
report, traces = run_pipeline(
    n=50,
    model="provider/model-name",
    call_fn=my_json_backend,
)
```

Production data must exist; the runner no longer silently substitutes the
bundled fixtures. Use `--use-fixtures` only for an explicit offline fixture
run, or pass `--cases` and `--ground-truth` paths.

Artifacts are written atomically to a unique
`staged_eval/runs/<UTC timestamp>_<id>/` directory. Each run contains a
provenance manifest, one trace per selected PMID, and a report. `--resume`
rejects changes to the model, backend, data hashes, evaluation code, or
generation settings. Legacy run directories without a manifest remain readable,
but cannot be resumed safely.

Stage 1 and final answer accuracy use the same selected-case denominator.
Invalid outputs and final abstentions count as incorrect instead of disappearing
from the final-accuracy cohort.

Run offline verification:

```bash
python -m pytest staged_eval/tests medcta_eval/tests judge_eval/tests -q
```

## Reusable LLM-as-a-judge evaluation

`judge_eval` is a separate offline scoring layer for open-ended outputs. It
uses versioned rubrics, repeated structured judgments, model-family safeguards,
complete raw artifacts, and a blinded two-rater human-validation workflow.
Judge calls never occur inside candidate-model inference.

For a completed MedCTA run:

```bash
python -m medcta_eval.judge score medcta_eval/runs/<run_id> \
  --judge-model <deployment> \
  --judge-family <provider-family> \
  --repeats 3
```

Complete the generated `human_validation_sample.csv` independently by two
raters, adjudicate disagreements, then calculate agreement:

```bash
python -m medcta_eval.judge validate-human \
  medcta_eval/runs/<run_id>/judge_runs/<judge_run_id> \
  medcta_eval/runs/<run_id>/judge_runs/<judge_run_id>/human_validation_sample.csv
```

See [the LLM judge protocol](docs/llm_judge_protocol.md) before treating judged
scores as paper results.
