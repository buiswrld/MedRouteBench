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

Run an evaluation:

```powershell
python -m staged_eval.pipeline --n 50 --inspect 0
```

Artifacts are written to `staged_eval/runs/<UTC timestamp>/`, with one trace per
selected PMID and a `report.json` containing the eight revision metrics.

Run offline verification:

```powershell
python -m pytest staged_eval/tests -q
```
