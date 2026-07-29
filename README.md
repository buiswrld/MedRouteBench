# MedRouteBench

## Idea #3: MedCTA reference-trajectory routing

`medcta_eval` evaluates whether a model selects the same next tool as a MedCTA
reference trajectory and finalizes at the same point. This is reference-path
agreement, not absolute clinical correctness: other tool routes may also be
reasonable.

The pinned adapter is available as an 11-case starter subset
(`data/medcta/subset_v1.json`) and the complete 107-case release
(`data/medcta/full_v1.json`). Together they cover `OCR`, `ImageDescription`,
`RegionAttributeDescription`, `GoogleSearch`, and `Calculator`. Each case
includes its question, pinned image URL, available tools, reference calls and
offline observations, and answer whitelist. The model never sees the whitelist,
future actions, future observations, or MedCTA's reference thoughts.

Regenerate or verify the pinned adapter:

```powershell
python -m medcta_eval.adapter
python -m medcta_eval.adapter --check
python -m medcta_eval.adapter --ids 0-106 --output data/medcta/full_v1.json
python -m medcta_eval.adapter --ids 0-106 --output data/medcta/full_v1.json --check
```

Azure OpenAI is the only built-in provider. Add the resource values to `.env`
(never commit the key):

```dotenv
AZURE_OPENAI_ENDPOINT=https://YOUR-RESOURCE.services.ai.azure.com/openai/v1/
AZURE_OPENAI_API_KEY=
AZURE_OPENAI_DEPLOYMENT=gpt-5-mini
MEDCTA_AZURE_REASONING_EFFORT=low
MEDCTA_AZURE_MAX_COMPLETION_TOKENS=1024
MEDCTA_AZURE_IMAGE_DETAIL=auto
MEDCTA_AZURE_SEED=42
```

The endpoint may also be the resource root; the adapter normalizes it to
`/openai/v1/`. The model argument is the Azure deployment name. Run a two-case
image smoke test before the complete pass:

```powershell
python -m medcta_eval.pipeline --model gpt-5-mini --cases data/medcta/full_v1.json --n 2 --inspect 0
python -m medcta_eval.pipeline --model gpt-5-mini --cases data/medcta/full_v1.json --n 107
```

Azure accepts JPEG, PNG, GIF, and WebP image inputs. The Azure adapter preserves
supported MedCTA images as pinned URLs and losslessly transcodes source TIFF/BMP
files to in-memory PNG data URLs. The pinned source reference remains in the
trace and provenance; converted image bytes are never written to run artifacts.

The Azure adapter resolves each pinned Hugging Face URL to its current image
CDN target in memory, then sends that image at every decision. The pinned URL
remains in prompts, traces, and provenance. Resolved signed URLs use a 45-minute
TTL by default; adjust it with `MEDCTA_IMAGE_URL_CACHE_TTL_SECONDS` if the
upstream lifetime changes.
The model must return one of these exact JSON shapes:

```json
{"action":"CALL_TOOL","tool_name":"OCR","answer":null}
```

```json
{"action":"FINAL_ANSWER","tool_name":null,"answer":"..."}
```

No real MedCTA tool is executed. After a valid tool call, the runner compares
the selected tool with the next reference tool and reveals that reference
tool's saved observation. A wrong but available tool therefore records a
reference mismatch and continues under controlled replay. Premature
finalization, missed finalization, unrepaired invalid output, and exhausted
inference failures stop only that case.

Before a built-in run, the pipeline validates the Azure key, endpoint, and
deployment configuration without spending inference tokens. Azure responses
retain prompt, cached-input, completion, and reasoning-token counts in the run
artifacts so cost can be calculated from the applicable Azure meter rates.

Python callers can inject another multimodal backend with the signature
`(system: str, user: str, image_url: str | None) -> str`:

```python
from medcta_eval import run_pipeline

report, traces = run_pipeline(
    n=11,
    model="provider/model-name",
    call_fn=my_json_vision_backend,
)
```

Each `medcta_eval/runs/<run-id>/` directory contains a provenance manifest, one
trace per attempted case, and `report.json` after completion. While a run is in
progress, `partial_report.json` is atomically refreshed after every saved trace.
An interrupted run can be reported again without any model call:

```powershell
python -m medcta_eval.pipeline --report-existing medcta_eval/runs/<run-id>
```

Reports include `next_tool_accuracy`,
`trajectory_step_accuracy`, `premature_finalization_rate`,
`missed_finalization_rate`, `tool_precision`, `unnecessary_tool_rate`,
`trajectory_exact_match_rate`, `final_answer_accuracy`,
`invalid_action_rate`, and `inference_failure_count`. Every rate records its
numerator and denominator. Rate cohorts exclude traces whose case terminated
with `inference_failure`; `n_evaluable_cases`, `inference_failure_count`, and
`inference_failure_case_count` expose that separation. A run with no evaluable
cases therefore reports null rates with zero denominators, not misleading 0%
accuracy. Model-caused early termination, including premature finalization,
remains evaluable and unreached reference steps remain misses. Final-answer
accuracy is normalized exact matching against MedCTA's whitelist; it is not an
LLM-based clinical-equivalence score.

The evaluator intentionally requests one tool call per model turn. MedCTA case
`19` contains two parallel calls in one reference assistant turn; the adapter
sequentializes those calls in source order, records that provenance on both
steps, and reveals the two saved observations one at a time.

Run all offline verification for Ideas #1 and #3:

```powershell
python -m pytest staged_eval/tests medcta_eval/tests -q
python -m compileall -q staged_eval medcta_eval
```

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

The PubMedQA experiment uses the same Azure resource and deployment. Optional
stage-specific settings are `STAGED_AZURE_REASONING_EFFORT`,
`STAGED_AZURE_MAX_COMPLETION_TOKENS`, `STAGED_AZURE_SEED`, and
`STAGED_MAX_RETRIES`.

Run an evaluation with the built-in Azure adapter:

```powershell
python -m staged_eval.pipeline --n 50 --model gpt-5-mini --inspect 0
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

```powershell
python -m pytest staged_eval/tests -q
```
