# PubMedQA staged revision

Two model calls are made per case. Stage 1 receives the question and pre-results text. Stage 2 receives the valid prior output, added evidence, and the full abstract in its original order. `--reversed` swaps which block comes first, not the full-abstract order.

The first normalized section label containing `RESULT` defines the boundary. Both blocks must contain text. The supplied 1,000-case file is filtered by the 500 official test-label IDs, yielding 482 eligible cases.

- Stage 1: `ANSWER` with `yes`, `no`, or `maybe`.
- Stage 2: `KEEP_ANSWER` with the same label, `REVISE_ANSWER` with another valid label, or `ABSTAIN` with a null answer.
- Confidence and a short reason are required. Invalid responses receive one repair attempt.
- `maybe` is a valid label, not a separately identified deferral. Both stages use the full-abstract gold label; a reasonable preliminary answer may be scored wrong.

The retained prompts are in [`prompts.py`](prompts.py). The prompt builder does not insert gold labels, `LONG_ANSWER`, or hidden final decisions into candidate prompts.

## New runs

Install and configure credentials using the [root README](../README.md). These commands make model calls:

```bash
python -m staged_eval.pipeline --n 482 --workers 1 --model "$OPENROUTER_MODEL" \
  --cases data/pubmedqa/ori_pqal.json \
  --ground-truth data/pubmedqa/test_ground_truth.json

python -m staged_eval.pipeline --n 482 --workers 1 --model "$OPENROUTER_MODEL" --reversed \
  --cases data/pubmedqa/ori_pqal.json \
  --ground-truth data/pubmedqa/test_ground_truth.json
```

Default `--n` is 50. Smaller samples are proportionally stratified unless `--no-stratify` is set. `--limit` caps raw cases before filtering; leave it unset for the full cohort. `--use-fixtures` selects the tiny fixture dataset but still calls the model unless a Python caller injects a mock backend.

Python callers can pass `call_fn(system, user) -> str` to `run_pipeline` with an explicit model label. The normal adapter defaults to Responses; `LLM_API=chat` selects chat completions. `SEED` is sent only by the chat adapter. Normal calls do not explicitly set temperature or reasoning effort.

## Metrics

The retained implementation uses valid Stage-1 cases for Stage-1 accuracy and completed cases for final accuracy. Invalid/incomplete cases are excluded from final accuracy; completed abstentions count as wrong. Missed revision excludes abstentions from its denominator.

Those rules differ from the paper's fixed-cohort description and earlier README. Cleanup preserves the formulas and tests and documents the disagreement in [metric definitions](../docs/metric_definitions.md). Until reconciled with historical execution, current runs are not exact regeneration of the paper's metrics. An accuracy difference is not a matched-cohort change when denominators differ.

## Artifacts and tests

Runs write manifests, traces, reports, and usage records under `staged_eval/runs/<run_id>/`. `--resume PATH` verifies the implemented provenance signature; older manifests are not automatically compatible. See the CLI `--help` for details.

```bash
python -m pytest staged_eval/tests -q
```

Old development runs are excluded from this cleaned snapshot and [inventoried separately](../docs/legacy_artifacts.md); the original archive preserves them unchanged.
