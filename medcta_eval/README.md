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

# Run 100 cases from the explicitly selected full data file
python -m medcta_eval.pipeline --cases data/medcta/fullset_v1.json --n 100

# Specify the Azure deployment to use
python -m medcta_eval.pipeline --n 50 --model gpt-5-mini

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
- `partial_report.json` — updated after every case during the run
- `report.json` — final aggregated metrics (replaces `partial_report.json`)

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
   all prior model actions and reference observations.
2. Calls the model, expecting a JSON object with exactly `action`,
   `tool_name`, and `answer`.
3. If the response is invalid (bad JSON or constraint violation), attempts
   **one repair call** with the error and original prompt included.
4. Checks whether the model's action matches the reference step's expected
   action and tool.
5. If `CALL_TOOL`: appends the reference observation and continues.
6. If `FINAL_ANSWER` at the correct step: records the answer, computes a strict
   normalized-whitelist diagnostic, and marks `completed`.
7. Terminates early on inference failure, persistent invalidity, or premature
   finalization.

### Final answer scoring

Inference and semantic evaluation are intentionally separate:

- The trajectory run records `strict_final_answer_match_rate`, a normalized
  equality check against every accepted answer. This remains a transparent
  diagnostic, not a semantic clinical-accuracy claim.
- After candidate outputs are saved, `python -m medcta_eval.judge score ...`
  runs the versioned LLM judge. This permits rescoring without paying to rerun
  the candidate, and keeps judge failures out of inference metrics.
- The judge defaults to three repeated calls, preserves raw responses, rotates
  accepted-reference order, and refuses a detectable same-family judge unless
  the override is explicit in provenance.

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
| `unnecessary_tool_rate` | All model `CALL_TOOL` decisions | Those made when the reference expected `FINAL_ANSWER` |
| `premature_finalization_rate` | All evaluable cases | Cases where model gave `FINAL_ANSWER` before the reference expected it |
| `missed_finalization_rate` | Steps where reference expected `FINAL_ANSWER` and model gave a valid response | Model chose `CALL_TOOL` instead |

### Final answer metrics

| Metric | Description |
|---|---|
| `strict_final_answer_match_rate` | Fraction of evaluable cases exactly equal to an accepted answer after conservative normalization |

Semantic judge reports are stored under the source run's `judge_runs/`
directory. They report `correct`, `partially_correct`, `incorrect`,
`not_scorable`, and `inconclusive` separately, together with judge failure and
stability counts.

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
routing while still arriving at a correct clinical answer. Conversely, a model
may answer correctly while stopping before obtaining the reference evidence.
Report routing and answer correctness as separate axes; neither is a sufficient
single-number summary.

LLM-judged correctness is provisional until the generated blinded sample has
two independent human labels, disagreements are adjudicated, and judge-human
agreement is reported. See `docs/llm_judge_protocol.md`.

---

## Offline semantic judging

Configure a separate judge deployment in `.env`, preferably from a different
model family:

```dotenv
JUDGE_AZURE_ENDPOINT=...
JUDGE_AZURE_API_KEY=...
JUDGE_AZURE_DEPLOYMENT=...
JUDGE_MODEL_FAMILY=anthropic
JUDGE_TEMPERATURE=0
```

Then score a completed run and create a blinded human sample:

```bash
python -m medcta_eval.judge score medcta_eval/runs/<run_id> \
  --judge-model <deployment> \
  --judge-family <provider-family> \
  --repeats 3 \
  --human-sample-size 30
```

After two raters complete the CSV and adjudicate disagreements:

```bash
python -m medcta_eval.judge validate-human \
  medcta_eval/runs/<run_id>/judge_runs/<judge_run_id> \
  medcta_eval/runs/<run_id>/judge_runs/<judge_run_id>/human_validation_sample.csv
```

---

## Offline Tests

```bash
python -m pytest medcta_eval/tests/ -q
```
