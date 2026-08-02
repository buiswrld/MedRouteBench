# LLM judge protocol

Use an LLM judge only for outputs that cannot be scored deterministically. Tool
names, action validity, and exact route agreement should remain rule-based.

## Required procedure

1. Freeze the rubric, prompt, pass labels, and denominator before scoring the
   held-out run.
2. Use a separate judge deployment. Prefer a different model family from the
   candidate model.
3. Hide candidate-model identity from the judge. Treat candidate text as
   untrusted and do not reward style, confidence, length, or citations.
4. Set temperature to zero and judge each answer three times. Keep raw responses
   and report label consistency.
5. Supply every accepted reference answer. Rotate reference order across
   repetitions.
6. Record judge failures, invalid output, `not_scorable`, candidate inference
   failures, and missing final answers separately.
7. Keep a deterministic exact-match result as a diagnostic. Do not replace it
   after seeing the semantic score.
8. Blindly label a frozen sample of at least 30 answers with two independent
   human raters. Adjudicate disagreements and report human-human and judge-human
   agreement before using the judge result as a headline metric.

## Repository implementation

`judge_eval/` provides the reusable scoring layer. A judge run saves:

- the versioned rubric and prompt hashes;
- judge and candidate model-family metadata;
- source-run and code hashes;
- all prompts, raw responses, repairs, parsed labels, and errors;
- per-answer repetition aggregates and run-level denominators;
- a seeded, blinded CSV for human validation.

The default clinical rubric is
`judge_eval/rubrics/clinical_correctness_v1.json`. It distinguishes `correct`,
`partially_correct`, `incorrect`, and `not_scorable`. Only `correct` is a pass.

## Idea 3

Candidate inference and answer judging are separate. First complete a MedCTA
run, then score its saved traces:

```bash
python -m medcta_eval.judge score medcta_eval/runs/<run_id> \
  --judge-model <deployment> \
  --judge-family <family> \
  --repeats 3 \
  --human-sample-size 30
```

After both raters complete the generated CSV:

```bash
python -m medcta_eval.judge validate-human \
  medcta_eval/runs/<run_id>/judge_runs/<judge_run_id> \
  <completed_labels.csv>
```

Idea 3 reports route agreement and final-answer correctness separately. A model
can give a correct answer while following a different or prematurely terminated
trajectory. Neither metric substitutes for the other.

## Limits

- Repeated agreement does not prove correctness.
- A different-family judge can still share training data and biases.
- `not_scorable` and inconclusive items require review; they must not silently
  disappear from the denominator.
- Pairwise comparisons need both A/B and B/A orders. Idea 3 uses pointwise
  reference-based correctness, so pairwise ordering is outside this adapter.
