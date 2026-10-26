# MedCTA forced-continuation replay

This package uses the same adapted MedCTA cases, candidate contract, prompt structure, and judge rules as [`medcta_eval`](../medcta_eval/README.md), while continuing after attempted early exits.

At nonterminal steps, declared actions remain in the trace, but the reference action and observation are supplied to the next call. At the terminal position, the current answer is used even if the model proposes another tool. A usable output is still required: inference failure, persistent invalidity, or malformed reference data can prevent completion.

Later model prompts show the forced reference actions, not overridden attempts or prior answers. The actual attempts remain in saved outputs. This changes evidence exposure and action history and does not isolate a causal effect of stopping.

Stopping-allowed replay also supplies reference observations after alternative valid tool choices. Its difference is that an early final answer can terminate replay; it does **not** stop on every tool-name mismatch.

## Running

Configure candidate and judge access using the [root README](../README.md):

```bash
python -m medcta_golden_eval.pipeline --n 107 --workers 1 \
  --model "$OPENROUTER_MODEL" --cases data/medcta/fullset_v1.json
```

Default input and `--n` select 11 starter cases. `--n 107` alone is insufficient. The full adaptation's [case 19 duplicated observation](../docs/reproducibility.md) is unchanged.

Resume by repeating the same configuration with `--resume PATH_TO_RUN`. Rebuild an existing report without new inference:

```bash
python -m medcta_golden_eval.pipeline --report-existing PATH_TO_RUN
```

## Metrics and outputs

- Final accuracy uses evaluable traces, excluding inference failures. Raw means use available numeric scores.
- Successful/missed revision condition on prior incorrectness; overreaction/kept-correct condition on prior correctness.
- Attempted early finalization is a case-level event. Progress and first-attempt scores condition on cases with that event and available values as implemented.
- A completed forced route is not evidence of successful autonomous routing. Normal routing-accuracy metrics are not reported here.
- Separate stochastic replays do not guarantee matched early-stop cohorts, answers, or scores.

Exact rules are in [`prompts.py`](prompts.py), [`runner.py`](runner.py), [`schema.py`](schema.py), and [`metrics.py`](metrics.py). Both judge thresholds are 0.8 by default. See [metric definitions](../docs/metric_definitions.md) for failure policies and interpretive limits.

Artifacts go to `medcta_golden_eval/runs/<run_id>/`. Completed paths have status `completed`; attempted exits are diagnostics, not premature-finalization terminal statuses.

```bash
python -m pytest medcta_golden_eval/tests -q
```
