# medcta_golden_eval — MedCTA Forced Golden-Path Evaluator

Sibling of [`medcta_eval`](../medcta_eval/README.md) that answers a different
question: *if the model were kept on the human expert's reference tool
trajectory no matter what it tried to do, how does its answer quality evolve,
and how often would it have tried to bail early?*

---

## Why a separate package, not a flag

`medcta_eval` terminates a case the moment the model deviates from the
reference trajectory (e.g. an early `FINAL_ANSWER` triggers
`premature_finalization` and ends the case there) — the right design for
measuring routing agreement. This package does the opposite: it **never**
abandons the reference trajectory because of the model's declared action.
The control flow is fundamentally different (never terminate early on
deviation vs. always terminate on the first deviation), so it's a distinct
runner rather than a flag on `medcta_eval`.

Everything else is unchanged from `medcta_eval`: same adapted dataset, same
JSON action contract (`action`/`tool_name`/`answer`/`reasoning`), same
prompts, same LLM-judge functions/thresholds, same `stage_transitions`
labelling logic. Results are directly comparable to `medcta_eval` on that
shared machinery.

---

## Concept

At every reference step the model is still asked to decide `CALL_TOOL` or
`FINAL_ANSWER` and still asked for its current-best answer — but that
declared action never controls the loop:

- **Non-terminal reference step** (reference expects `CALL_TOOL`): the golden
  reference observation is revealed and the loop advances **regardless** of
  what the model declared. If the model declared `FINAL_ANSWER` here, that's
  recorded as `attempted_early_exit = True` on the step, but the case is
  forced to continue anyway.
- **Terminal reference step** (reference expects `FINAL_ANSWER`): there is no
  further golden step to force into, so this is always where the case ends.
  The model's own answer here becomes the case's final answer, regardless of
  which action it actually declared (a stray `CALL_TOOL` here is recorded as
  `attempted_extra_tool_call`, a diagnostic field, not part of the metric
  set below).

Because deviation never ends a case early, there are no
`premature_finalization` / `missed_finalization` **terminal statuses** here
— every evaluable case reaches `status == "completed"` (or ends early only
on genuine inference failure / persistent response invalidity, where there's
no usable model output to force through).

### Transcript consistency: the model always sees the golden path

The transcript shown back to the model on later turns (`PRIOR MODEL
ACTIONS` in the prompt) always records the **forced/golden** action —
`{"action": "CALL_TOOL", "tool_name": <reference tool>}` — never the model's
real overridden attempt, even when the model tried to exit early or call a
different tool. This keeps the model's own context consistent with "you are
on the golden path": it never sees itself having attempted something the
loop then ignored. The model's real parsed output is still preserved,
unshown, in `step["model_output"]` for scoring and diagnostics.

---

## Running an Evaluation

```bash
# Run the default 11-case starter subset
python -m medcta_golden_eval.pipeline

# Run a larger set
python -m medcta_golden_eval.pipeline --n 100

# Specify the OpenRouter model to use
python -m medcta_golden_eval.pipeline --n 50 --model gpt-5-mini

# Resume an interrupted run
python -m medcta_golden_eval.pipeline --n 100 --resume medcta_golden_eval/runs/<run_id>

# Recompute a report from saved traces (no inference)
python -m medcta_golden_eval.pipeline --report-existing medcta_golden_eval/runs/<run_id>
```

Artifacts are written atomically to `medcta_golden_eval/runs/<UTC-timestamp>_<id>/`,
with the same `manifest.json` / `trace_<case_id>.json` / `partial_report.json`
/ `report.json` layout as `medcta_eval`.

---

## Trace fields specific to this package

| Field | Level | Meaning |
|---|---|---|
| `steps[i].attempted_early_exit` | step | Model declared `FINAL_ANSWER` at a non-terminal reference step (overridden; the loop forced `CALL_TOOL` anyway) |
| `steps[i].attempted_extra_tool_call` | step | Model declared `CALL_TOOL` at the terminal reference step (overridden; diagnostic only) |
| `attempted_premature_finalization` | case | `True` if any step attempted an early exit (case-level, mirrors `medcta_eval`'s per-case `premature_finalization` definition) |
| `premature_finalization_progress` | case | Fraction of the reference tool sequence already forced-completed before the *first* attempted early exit; `None` if never attempted. Captured during the loop — `model_tool_sequence` alone can't reconstruct this here, since it always equals the *full* reference sequence once a case completes |

---

## Metrics

`build_report` here is deliberately smaller than `medcta_eval`'s: routing
metrics like `next_tool_accuracy`, `tool_precision`, `trajectory_exact_match_rate`,
and `missed_finalization_rate` don't mean anything once routing is forced
(a completed case's `model_tool_sequence` is definitionally the reference
sequence). What's left is final-answer quality, revision behavior, and
attempted-exit timing.

All metrics operate on **evaluable traces** — cases that did not fail due to
an inference/infrastructure error.

### Final answer metrics

| Metric | Thresholded? | Description |
|---|---|---|
| `final_answer_accuracy` | Yes — rate | Fraction of evaluable cases where `final_answer_match` is True (LLM judge ≥ 0.8). Added so free-routing (`medcta_eval`) vs. oracle-routing (here) accuracy is directly comparable, not just the raw mean score |
| `final_answer_mean_score` | No — raw mean | Mean LLM judge score (0–1, un-thresholded) across all evaluable cases |
| `first_step_answer_score` | No — raw mean | Mean LLM judge score of each case's step-0 answer, before any evidence. Pairs with `final_answer_mean_score` as the start/end of the score trajectory |
| `mean_score_improvement` | No — raw mean | Mean of `(final_answer_score - first_step_answer_score) / tools_called`, one delta per case, weighted equally. Cases missing either score, or with zero tool calls, are excluded rather than counted as zero. **Unlike `medcta_eval`'s version of this metric, `tools_called` here is in practice always `len(reference_tool_sequence(case))`** — the path is forced to completion, so this denominator is fixed per case rather than model-dependent. Don't read the two frameworks' `mean_score_improvement` numbers as measuring identical things |

`first_step_answer_score` and `mean_score_improvement` are deliberately not
bucketed by `step_index`, for the same reason `medcta_eval` avoids it:
reference-trajectory length varies per case, so a fixed high `step_index`
would average over a shrinking, non-random subset of (likely longer/harder)
cases.

### Revision behaviour

Same `stage_transitions` labelling as `medcta_eval` (`successful_revision` /
`missed_revision` / `overreaction` / `kept_correct`), computed identically
over consecutive scored steps.

| Metric | Denominator | Numerator |
|---|---|---|
| `successful_revision_rate` | Transitions where the prior step's answer was wrong | Those that become correct at the next step |
| `missed_revision_rate` | Same denominator as above | Those still wrong at the next step |
| `overreaction_rate` | Transitions where the prior step's answer was correct | Those that become wrong at the next step |
| `kept_correct_rate` | Same denominator as above | Those that remain correct |

`missed_revision_rate` and `kept_correct_rate` are the numerical complements
of `successful_revision_rate` and `overreaction_rate` respectively (same
denominators, mutually exclusive/exhaustive labels) — but each is still its
own explicit function counting its own label, exactly as `medcta_eval.metrics`
does it, not derived as `1 - x`.

### Attempted-exit metrics

| Metric | Shape | Description |
|---|---|---|
| `premature_finalization_rate` | Rate | Fraction of evaluable cases where the model attempted at least one early exit before the reference trajectory allowed it — case-level, regardless of the fact the attempt was overridden |
| `premature_finalization_progress` | Mean rate (`{"mean_rate", "n_cases"}`) | Mean fraction of the reference tool sequence completed before the model's *first* attempted early exit, over cases that attempted one. `0.0` = attempted immediately; near `1.0` = attempted just short of the reference's last tool |
| `mean_score_before_premature_finalization` | No — raw mean | Mean `current_answer_score` at each case's *first* attempted early exit, over cases that attempted one |

`mean_score_before_premature_finalization` is the metric directly comparable
to `medcta_eval`'s `premature_finalization_mean_score`: both restrict to the
cohort of cases that attempted/executed a premature exit (not all evaluable
cases, unlike `final_answer_mean_score`), and per case the two numbers
should match — same judge, same answer text, identical prior context up to
the point of the first deviation. In aggregate they're comparable because
both frameworks now restrict to the same cohort definition.

### Operational bookkeeping

`run_id`, `model`, `backend`, `provenance`, `n_selected_cases`,
`n_evaluable_cases`, `trace_status_counts`, `inference_failure_count`,
`inference_failure_case_count`, `repair_count` — same meaning as in
`medcta_eval`.

---

## Important Caveats

Because deviation never terminates a case, `trace_status_counts` should show
only `completed` (plus, occasionally, `inference_failure` / `invalid_action`
/ `invalid_reference_trajectory` on genuine API or response problems) — never
`premature_finalization` or `missed_finalization`. Seeing either of those
statuses in a `medcta_golden_eval` run indicates a bug in the forced-continuation
loop, not a real model outcome.

As with `medcta_eval`, comparing `report.json` files across the two
frameworks (or across runs) is a manual/notebook step, out of scope here.

---

## Offline Tests

```bash
python -m pytest medcta_golden_eval/tests/ -q
```
