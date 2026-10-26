# staged_eval — Two-Stage PubMedQA Commit-Then-Revise

Evaluates how well a model updates its beliefs when given additional evidence.
The design is a fixed two-stage protocol on the PubMedQA yes/no/maybe
biomedical question-answering dataset.

---

## Concept

PubMedQA cases contain a question, multiple labelled context sections
(e.g. BACKGROUND, METHODS, RESULTS), and an official gold label. The
evaluation splits those contexts at the **first section labelled RESULTS** so
Stage 1 sees only the preliminary evidence and Stage 2 sees everything. Cases
without a usable RESULTS boundary are excluded.

**Stage 1 — Preliminary Commitment**

The model receives the question and the preliminary contexts. It must commit
to one of `yes`, `no`, or `maybe` via `{"action": "ANSWER", "answer": "..."}`.
No hedging or deferral is allowed.

**Stage 2 — Final Revision Decision**

The model receives the question again, its own Stage 1 JSON output verbatim,
and the full context. It must choose exactly one of:

| Action | Constraint | Meaning |
|---|---|---|
| `KEEP_ANSWER` | answer must equal Stage 1 answer | Confidence maintained |
| `REVISE_ANSWER` | answer must differ from Stage 1 | Belief updated |
| `ABSTAIN` | answer must be null | Deliberate non-answer |

---

## Running an Evaluation

```bash
# Run 50 stratified cases (default)
python -m staged_eval.pipeline

# Run a specific number
python -m staged_eval.pipeline --n 20

# Disable stratified sampling
python -m staged_eval.pipeline --n 20 --no-stratify

# Flip which evidence half is shown at Stage 1 vs. Stage 2
python -m staged_eval.pipeline --n 20 --reversed

# Specify a model override
python -m staged_eval.pipeline --n 50 --model openai/gpt-5-mini

# Resume an interrupted run
python -m staged_eval.pipeline --n 50 --resume staged_eval/runs/<run_id>
```

# Backend-neutral usage
```python
from staged_eval.pipeline import run_pipeline

report, traces = run_pipeline(
    n=50,
    model="provider/model-name",
    call_fn=lambda system, user: my_backend(system, user),
)
```

Artifacts are written atomically to `staged_eval/runs/<UTC-timestamp>_<id>/`.
Each run contains:
- `manifest.json` — model, backend, data and code SHA-256 hashes, provenance
- `trace_<pmid>.json` — per-case trace (stage outputs, label, gold)
- `usage.jsonl` — append-only token and provider-reported cost record per successful API call
- `usage_summary.json` — usage totals by provider and model
- `partial_report.json` — updated after every case during the run
- `report.json` — final aggregated metrics (replaces `partial_report.json`)

---

## Pipeline Mechanics

### Data and sampling

The runner loads cases from `data/pubmedqa/test_set.json` (or `ori_pqal.json`
as a fallback) and gold labels from `data/pubmedqa/test_ground_truth.json`.

Only **eligible** cases enter the evaluation: cases must have an official gold
label, a usable RESULTS boundary, and non-empty evidence on both sides of that
boundary. The current PubMedQA test set contains 482 eligible cases.

By default the selected sample is **stratified** — proportionally distributed
across `yes`, `no`, and `maybe` gold labels using the largest-remainder
(Hamilton) method so results aren't distorted by label imbalance.

### Evidence splitting (`data.py`)

For each case, the evidence split boundary is determined as follows:
1. Find the first normalized section label containing `RESULT`.
2. If that index creates two non-empty groups, use it (`strategy: first_results_section`).
3. Otherwise, exclude the case.

Everything **before** the split index goes to `stage1_evidence`; the split
index and everything after it goes to `stage2_added_evidence`.

### Evidence direction (`--reversed`)

By default Stage 1 sees the pre-RESULTS evidence and Stage 2 reveals
RESULTS-onward. `--reversed` (internally `reversed_order=True`) tests
whether the *order* evidence is revealed in — not just its content — drives
revision behavior, by flipping which half is shown first:

| | Normal | `--reversed` |
|---|---|---|
| Stage 1 sees | pre-RESULTS evidence | RESULTS-onward evidence |
| Stage 2 "ADDED EVIDENCE" shows | RESULTS-onward evidence | pre-RESULTS evidence |

What stays fixed either way:
- **`full_context`** — always the complete case text in original document
  order, since it's the "everything, for reference" recap shown at Stage 2,
  not something direction-dependent.
- **The RESULTS-boundary detection itself** (`split_index`, `strategy`) —
  only which side becomes `stage1_evidence` vs. `stage2_added_evidence`
  changes, not where the case is split.
- **Every metric formula** — `stage1_answer_accuracy`, `final_answer_accuracy`,
  the revision-rate metrics, `final_abstention_rate`, and `maintenance_rate`
  only look at parsed stage1/stage2 answers vs. gold, never which evidence
  half was shown at which stage.

Every trace and report self-documents its `reversed` value. `--resume`
refuses to mix a `--reversed` run with a normal one — resuming a saved run
with a different `reversed_order` raises the same provenance-mismatch error
as resuming with a different model.

### Per-case execution (`runner.py`)

For each case, the runner:

1. Builds the Stage 1 prompt and calls the model.
2. If the response is invalid (bad JSON or constraint violation), attempts
   **one repair call** with the error and original prompt included.
3. If Stage 1 remains invalid after repair, the case is labelled `invalid_stage1`.
4. Builds the Stage 2 prompt — which includes the case question, the model's
   exact Stage 1 JSON output, and the full labelled context.
5. Same repair logic applies for Stage 2.
6. Classifies the outcome (see table below) and records the full trace.

### Outcome classification

Given `stage1_answer` and the final (`stage2`) answer:

| Condition | Label |
|---|---|
| Stage 2 action is `ABSTAIN` | `abstention` |
| Stage 1 wrong, final answer correct | `successful_revision` |
| Stage 1 wrong, final answer still wrong | `missed_revision` |
| Stage 1 correct, final answer still correct | `kept_correct` |
| Stage 1 correct, final answer wrong (changed to wrong) | `overreaction` |

---

## Metrics

All accuracy metrics use **scorable traces** as their base cohort: traces
that have an official gold label and did not get skipped. Invalid Stage 2
and abstentions count as **incorrect** in final accuracy (they do not
disappear from the denominator).

### Accuracy

| Metric | Description |
|---|---|
| `stage1_answer_accuracy` | Fraction of scorable cases where Stage 1 answer matches gold |
| `final_answer_accuracy` | Fraction of scorable cases where the final answer (Stage 2 outcome) matches gold; invalid and abstained cases count as wrong |

### Revision behaviour

| Metric | Denominator | Numerator |
|---|---|---|
| `successful_revision_rate` | Cases with a valid but wrong Stage 1 answer | Those that `REVISE_ANSWER` to the correct answer |
| `missed_revision_rate` | Cases with wrong Stage 1, non-abstaining Stage 2 | Those where the final answer is still wrong |
| `overreaction_rate` | Cases with a correct Stage 1 answer | Those that `REVISE_ANSWER` to a wrong answer |
| `kept_correct_rate` | Cases with a correct Stage 1 answer | Those that explicitly `KEEP_ANSWER` with the correct answer |
| `final_abstention_rate` | All scorable cases | Those that `ABSTAIN` at Stage 2 |
| `maintenance_rate` | Valid Stage 1 cases that do not abstain | Those where the final answer equals the Stage 1 answer (regardless of correctness) |

### Relationship between metrics

The stage1 and final accuracy numbers measure the **same cohort** so the delta
is directly interpretable:

- `final > stage1`: the model successfully revises wrong Stage 1 answers more
  often than it corrupts correct ones.
- `final < stage1`: the model's revisions are net harmful (overreaction exceeds
  successful revision).
- `final ≈ stage1` with high `maintenance_rate`: the model is predominantly
  keeping its answers rather than revising.

---

## Action contract

Both stages use the same JSON schema:

```json
{"action": "...", "answer": "yes|no|maybe|null"}
```

The validator enforces cross-field consistency:
- `KEEP_ANSWER` requires `answer == prior_answer`
- `REVISE_ANSWER` requires `answer != prior_answer` and `answer` non-null
- `ABSTAIN` requires `answer == null`
- Stage 1 only accepts `ANSWER` with a non-null `yes/no/maybe`

---

## Offline Tests

```bash
python -m pytest staged_eval/tests/ -q
```
