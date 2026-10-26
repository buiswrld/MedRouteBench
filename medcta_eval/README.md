# MedCTA stopping-allowed reference replay

The model sees a clinical question, image, available tools, and accumulated reference observations. It provides a current-best answer at every step and chooses `CALL_TOOL` or `FINAL_ANSWER`.

At a nonterminal position, a valid tool call reveals the next recorded reference observation **even when the selected tool differs**. A tool-name mismatch lowers agreement scores but does not terminate replay. An early final answer ends replay. Calling another tool at the reference endpoint produces `missed_finalization`. Inference failure or persistent invalid output can also terminate a case.

This is neither live tool execution nor autonomous evidence acquisition. Agreement with one route is not clinical validity; an early answer may already be adequate.

## Running

Configure OpenRouter candidate access and the separate judge using the [root README](../README.md):

```bash
python -m medcta_eval.pipeline --n 107 --workers 1 \
  --model "$OPENROUTER_MODEL" --cases data/medcta/fullset_v1.json
```

The default input has 11 cases. Increasing `--n` alone does not select the full file. Images remain remote URLs. The full adaptation contains a [known duplicated observation in case 19](../docs/reproducibility.md); the data have not been silently corrected.

To resume, repeat the same configuration with `--resume PATH_TO_RUN`. To rebuild a report without new inference:

```bash
python -m medcta_eval.pipeline --report-existing PATH_TO_RUN
```

Live calls require a valid candidate identifier and configured judge. Hosted endpoint compatibility was not verified in this cleanup.

## Prompts, scoring, and outputs

[`prompts.py`](prompts.py) contains candidate, repair, correctness, and symmetric equivalence prompts. [`schema.py`](schema.py) validates actions; [`runner.py`](runner.py) implements replay and one repair attempt for invalid output.

Correctness and equivalence use separate judge rubrics, both thresholded at `>= 0.8`. The retained default judge route is Azure, independent of OpenRouter candidate access. An OpenRouter judge is configurable but defines a different evaluation configuration.

Missing correctness scores fail the threshold; missing equivalence scores count as changed. Identical normalized answers short-circuit equivalence scoring. Correctness caching reuses scores within a case, but separate stochastic runs do not guarantee identical answers or scores.

See [`metrics.py`](metrics.py) and [metric definitions](../docs/metric_definitions.md) for denominators. Inference-failure cases are excluded from evaluable cohorts, and raw means use available numeric scores. These are protocol measurements, not independently validated clinical-performance scores.

Artifacts go to `medcta_eval/runs/<run_id>/`: manifests, per-case traces, partial/final reports, and usage records. Traces can contain prompts, model responses, gold answers used for scoring, and judge outputs. Generated artifacts are ignored by Git.

```bash
python -m pytest medcta_eval/tests -q
```

`python -m shared.judge_audit --help` describes human-review sampling when suitable traces exist. This utility does not establish that the paper's judge was independently validated.
