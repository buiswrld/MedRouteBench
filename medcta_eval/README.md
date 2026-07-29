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
6. If `FINAL_ANSWER` at the correct step: scores the answer and marks
   `completed`.
7. Terminates early on inference failure, persistent invalidity, or premature
   finalization.

### Final answer scoring

The final answer goes through **two independent scoring passes**:

- **Heuristic match** — included for diagnostics but not the primary metric.
- **LLM-as-judge** — the same Azure deployment evaluates the predicted answer
  against the gold answer on a 0–1 semantic correctness scale using
  `FINAL_ACCURACY_SYSTEM_PROMPT`. A score ≥ 0.8 sets `final_answer_match = True`.
  The mean score across all cases is reported as `llm_final_answer_accuracy`.

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
| `final_answer_accuracy` | Fraction of evaluable cases where `final_answer_match` is True (LLM judge ≥ 0.8) |
| `llm_final_answer_accuracy` | Mean LLM judge score (0–1) across all evaluable cases that received a final answer |

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
routing while still arriving at the correct clinical answer. `llm_final_answer_accuracy`
is the most clinically meaningful single number.

---

## Offline Tests

```bash
python -m pytest medcta_eval/tests/ -q
```
