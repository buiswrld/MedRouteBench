# medcta_eval — MedCTA Reference-Trajectory Evaluator

Evaluates a model's ability to navigate a multi-step clinical image-analysis
workflow by comparing its tool-call sequence against a human expert's reference
trajectory from the [IVUL-KAUST/MedCTA](https://huggingface.co/datasets/IVUL-KAUST/MedCTA)
dataset.

---

## Concept

Each MedCTA case is a clinical question paired with a medical image. A human
expert solved the case by calling a sequence of diagnostic tools
(e.g. `OCR`, `ImageDescription`, `RegionAttributeDescription`) before issuing
a `FINAL_ANSWER`. The evaluator replays this reference trajectory step-by-step:

1. The model sees the question, the image reference URL, and the list of
   available tools.
2. At each step it must decide: call a tool (`CALL_TOOL`) or give a final
   answer (`FINAL_ANSWER`).
3. After each `CALL_TOOL`, the model receives the **reference observation**
   (the tool output recorded in the dataset) — not a live tool result.
4. The loop ends when the model issues `FINAL_ANSWER` or deviates from the
   reference trajectory's expected step count.

This controlled replay design lets us measure routing quality independently of
whether a live tool would succeed.

---

## Running an Evaluation

```bash
# Run the default 11-case starter subset
python -m medcta_eval.pipeline

# Run a larger set
python -m medcta_eval.pipeline --n 100

# Specify the OpenRouter model to use
python -m medcta_eval.pipeline --n 50 --model openai/gpt-5-mini

# Resume an interrupted run
python -m medcta_eval.pipeline --n 100 --resume medcta_eval/runs/<run_id>

# Recompute a report from saved traces (no inference)
python -m medcta_eval.pipeline --report-existing medcta_eval/runs/<run_id>

# Use a custom call function (backend-neutral)
from medcta_eval.pipeline import run_pipeline

report, traces = run_pipeline(
    n=50,
    model="provider/model-name",
    call_fn=lambda system, user, image_url: my_backend(system, user, image_url),
)
```

Artifacts are written atomically to `medcta_eval/runs/<UTC-timestamp>_<id>/`.
Each run contains:
- `manifest.json` — model, backend, code SHA-256 hashes, provenance
- `trace_<case_id>.json` — per-case trace (steps, model outputs, scores)
- `usage.jsonl` — append-only token and provider-reported cost record per successful API call
- `usage_summary.json` — usage totals by provider, model, and candidate/judge role
- `partial_report.json` — updated after every case during the run
- `report.json` — final aggregated metrics (replaces `partial_report.json`)

OpenRouter reports the charged request cost directly. Azure judge calls retain
their token counts but remain marked unpriced because the response has no
dollar-cost field.

To create a seeded, model-blinded sample for human review without rerunning a
model or changing saved metrics:

```bash
python -m shared.judge_audit medcta_eval/runs/<run_id> \
  medcta_golden_eval/runs/<run_id> --out-dir judge_human_audit
```

---

## Pipeline Mechanics

### Input: the adapted subset

`data/medcta/subset_v1.json` (or override with `--cases`) is the pre-built
evaluation file. It contains validated cases in the format expected by the
runner — clinical questions, image references, available tools, and reference
step sequences including per-step observations and the ground-truth answer.

### Per-case execution (`runner.py`)

For each case, the runner:

1. Builds a step prompt showing the question, image URL, available tools, and
   all prior model actions (`action`/`tool_name` only) and reference
   observations. The model's own prior `answer`/`reasoning` are tracked for
   scoring but deliberately **not** shown back to it, so its current guess
   isn't anchored on an earlier pre-evidence guess.
2. Calls the model, expecting a JSON object with exactly `action`,
   `tool_name`, `answer`, and `reasoning`. `answer` is required and nonempty
   at **every** step, not just `FINAL_ANSWER` — it doubles as the model's
   current-best clinical hypothesis (before evidence is gathered) and, at
   `FINAL_ANSWER`, its final answer. `reasoning` is a brief justification,
   also required at every step.
3. If the response is invalid (bad JSON or constraint violation), attempts
   **one repair call** with the error and original prompt included.
4. Checks whether the model's action matches the reference step's expected
   action and tool.
5. Scores that step's `answer` against the ground truth via the LLM judge
   (see below) — uniformly, whether the step was `CALL_TOOL` or
   `FINAL_ANSWER` — and records `current_answer`/`current_answer_score`/
   `current_answer_correct` on the step trace.
6. If `CALL_TOOL`: appends the reference observation and continues.
7. If `FINAL_ANSWER` at the correct step: reuses that step's already-computed
   score/correctness as the case's final answer and marks `completed`.
8. Terminates early on inference failure, persistent invalidity, or premature
   finalization.
9. After the loop, walks consecutive scored steps to build
   `stage_transitions` — one entry per pair, labelling how the current
   answer evolved as new tool evidence arrived (see "Revision behaviour"
   below).

### Answer scoring

Every step's `answer` — `CALL_TOOL` or `FINAL_ANSWER` alike — is scored via
one **LLM-as-judge** function, `judge_answer` (`ANSWER_ACCURACY_SYSTEM_PROMPT`).
There is deliberately no separate prompt for intermediate vs. final answers:
an earlier two-prompt design (one framed as "the predicted FINAL answer,"
one framed as "current-best hypothesis") let the *same literal answer text*
score differently purely because of which prompt graded it — which could
flip a stage-transition label (e.g. `successful_revision`) across the
`FINAL_ANSWER` boundary even when the model's answer never actually
changed. Using one prompt for every step means identical answer text always
receives an identical score, so transition labels can only change because
the answer itself changed.

`judge_answer` is also given every entry in `accepted_answers`, not just the
first — a case's gold answer may have more than one valid phrasing, and
matching any single one of them is sufficient for a 1.0.

`judge_answer` uses the fixed judge configured for the run (currently an Azure
GPT-5.4 deployment), the same 0–1 semantic correctness scale and scoring rubric,
and the same threshold
(`FINAL_ACCURACY_CONFIDENCE_THRESHOLD`, 0.8) at every step. A score ≥ the
threshold sets `current_answer_correct = True` (and, at the terminal step,
`final_answer_match = True`). The mean final-step score across all cases is
reported (unthresholded) as `final_answer_mean_score`.

The per-case correctness cache (`answer_score_cache` in `runner.py`) keys on
normalized answer text alone: since one judge function/prompt is used
throughout, identical text always shares one cache entry and one judge
call, regardless of whether it recurs across `CALL_TOOL` steps, at the
`FINAL_ANSWER` step, or both.

Consecutive steps' answers are also compared to each other, to detect
whether the model's answer actually changed between steps — used by the
revision metrics below. This uses a **separate** judge call,
`judge_equivalence(previous, current)` with `ANSWER_EQUIVALENCE_SYSTEM_PROMPT`,
not `judge_answer`/`ANSWER_ACCURACY_SYSTEM_PROMPT`: the correctness prompt is
asymmetric (predicted-vs-gold, with a rule that scores 1.0 whenever the
predicted answer contains the gold answer), which would bias a same/changed
check toward calling refinements "unchanged" while calling generalizations
"changed". The equivalence prompt instead asks symmetrically whether two of
the model's own answers express the same conclusion, and is thresholded by
its own separate constant, `ANSWER_EQUIVALENCE_CONFIDENCE_THRESHOLD` (also
0.8 by default, but tunable independently of the correctness threshold).
Identical text (after whitespace/case normalization) skips this judge call
entirely, since it's trivially unchanged.

---

## Metrics

All routing metrics operate on **evaluable traces** — cases that did not fail
due to an inference/infrastructure error. Infrastructure failures are counted
separately and excluded to avoid penalising model quality for provider issues.

### Routing metrics

| Metric | Denominator | Numerator |
|---|---|---|
| `next_tool_accuracy` | Steps where reference expected `CALL_TOOL` | Model chose the same tool |
| `trajectory_step_accuracy` | All reference steps (tool calls + final answer) | Steps where model action matched reference |
| `trajectory_exact_match_rate` | All evaluable cases | Cases where the full tool sequence matched AND model finalized at the right step |
| `tool_precision` | All model `CALL_TOOL` decisions | Those that matched the reference-expected tool |
| `unnecessary_tool_rate` | All model `CALL_TOOL` decisions | Those made when the reference expected `FINAL_ANSWER` (reference-position-based — see `unnecessary_tool_calls` below for the correctness-based counterpart) |
| `premature_finalization_rate` | All evaluable cases | Cases where model gave `FINAL_ANSWER` before the reference expected it |
| `missed_finalization_rate` | Steps where reference expected `FINAL_ANSWER` and model gave a valid response | Model chose `CALL_TOOL` instead |

### Final answer metrics

| Metric | Thresholded? | Description |
|---|---|---|
| `final_answer_accuracy` | Yes — rate | Fraction of evaluable cases where `final_answer_match` is True (LLM judge ≥ 0.8) |
| `final_answer_mean_score` | No — raw mean | Mean LLM judge score (0–1, un-thresholded) across all evaluable cases that received a final answer |
| `first_step_answer_score` | No — raw mean | Mean LLM judge score of each case's step-0 answer (before any evidence). Pairs with `final_answer_mean_score` as the start/end of the score trajectory |
| `mean_score_improvement` | No — raw mean | Mean of `(final_answer_score - first_step_answer_score) / tools_called`, one delta per case, weighted equally regardless of how many tools that case used. Cases missing either score, or with zero tool calls (division by zero), are excluded from the mean rather than counted as zero |

`first_step_answer_score` and `mean_score_improvement` are deliberately *not*
bucketed by `step_index`: reference-trajectory length varies per case (3–6
steps in the current 11-case starter subset), so accuracy/score at a fixed
high `step_index` would reflect a shrinking, non-random subset of (likely
longer/harder) cases rather than a real trend. `stage_answer_accuracy_by_step`
below still buckets by `step_index` as a "bonus breakdown" for inspection,
but the same caveat applies to it — don't read a trend into its tail.

### Revision behaviour

These track the model's *evolving* answer across steps, not just its final
one — do new tool observations fix wrong hypotheses, corrupt right ones, or
get ignored? Denominators are counted over `stage_transitions` (one entry
per consecutive pair of scored steps) unless noted otherwise.

| Metric | Denominator | Numerator |
|---|---|---|
| `stage_answer_accuracy` | All scored steps (`answer` present) across evaluable traces | Steps where `current_answer_correct` is True |
| `successful_revision_rate` | Transitions where the prior step's answer was wrong | Those that become correct at the next step |
| `missed_revision_rate` | Same denominator as above | Those still wrong at the next step |
| `overreaction_rate` | Transitions where the prior step's answer was correct | Those that become wrong at the next step |
| `kept_correct_rate` | Same denominator as above | Those that remain correct |
| `answer_change_rate` | All transitions | Those where `answer_changed` is True (LLM-judged non-equivalence) |
| `maintenance_rate` | All transitions | Those where the answer did not change — complement of `answer_change_rate`, reported separately for symmetry |
| `maintained_wrong_rate` | Transitions where the prior step's answer was wrong | Those that stayed wrong *and* the answer didn't change |
| `answer_stability` | Evaluable traces with ≥2 scored steps | Traces where the answer never changed across any transition |

### Stopping / confidence metrics

A secondary, distinct set from the revision metrics above — these study
*when* the model decides it has gathered enough evidence to stop, rather
than whether its answer improves as evidence arrives. Use the two sets
together: e.g. a model with high `stage_answer_accuracy` but also high
`unnecessary_tool_calls` is accurate but over-cautious about stopping.

| Metric | Denominator | Numerator |
|---|---|---|
| `premature_finalization_wrong` | Cases with status `premature_finalization` | Those where `final_answer_match` is False |
| `early_correct_finalization` | Same denominator | Those where `final_answer_match` is True |
| `unnecessary_tool_calls` | Scored steps where `current_answer_correct` is True *and* `evidence_shown` is non-empty | Those where the model's actual action was `CALL_TOOL` anyway |

Each trace's step 0 is always excluded from `unnecessary_tool_calls`'
denominator: it precedes any revealed reference observation, so a correct
answer there is a lucky guess rather than a sign the model had gathered
enough evidence to stop. Without this, a model that dutifully calls every
reference-expected tool — the ideal routing behavior — would still be
flagged as "unnecessary" purely for guessing right before evidence existed.

Two more, over the same `status == "premature_finalization"` cohort, but not
thresholded to pass/fail:

| Metric | Shape | Description |
|---|---|---|
| `premature_finalization_mean_score` | No — raw mean | Mean RAW `final_answer_score` restricted to `premature_finalization` cases — the raw-score sibling of `premature_finalization_wrong`/`early_correct_finalization` above, which only report that cohort's pass/fail split |
| `premature_finalization_progress` | No — mean rate | Mean fraction of the reference tool sequence completed before the model bailed (`len(model_tool_sequence) / len(reference_tool_sequence)` at the point of termination), over the same cohort. `0.0` = bailed immediately at step 0; near `1.0` = bailed just short of the reference's last tool. A rate, not a raw `step_index`, so cases with different reference-trajectory lengths stay comparable |

`premature_finalization_mean_score` is the metric directly comparable to
`medcta_golden_eval`'s `mean_score_before_premature_finalization`: both
restrict to the cohort of cases that attempted/executed a premature exit
(not all evaluable cases, unlike `final_answer_mean_score`), and per case the
two numbers should match — same judge, same answer text, identical prior
context up to the point of the first deviation.

### Operational metrics

| Metric | Description |
|---|---|
| `invalid_action_rate` | Fraction of model responses that were initially invalid (before repair) |
| `repair_count` | Total number of repair calls made across all steps |
| `inference_failure_count` | Total steps that failed with an unrecoverable API error |
| `inference_failure_case_count` | Number of cases that terminated due to inference failure |

---

## Important Caveats

The report includes this note in every output:

> *"Routing metrics measure agreement with one MedCTA reference trajectory,
> not absolute clinical correctness or uniqueness of the tool path."*

A model may use a different but equally valid tool order and score poorly on
routing while still arriving at the correct clinical answer. `final_answer_mean_score`
(the raw, un-thresholded judge score — see "Final answer metrics" above) is
the most clinically meaningful single number.

---

## Offline Tests

```bash
python -m pytest medcta_eval/tests/ -q
```
