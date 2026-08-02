# LLM-as-a-Judge Protocol

This is the project-wide protocol for using an LLM to score open-ended model
outputs. An LLM judge is a measurement instrument. Its output is not ground
truth and must not become a headline result until the instrument is validated.

## Requirements

### 1. Define the measurement before running the judge

- State the construct being measured. Do not combine correctness, relevance,
  completeness, safety, and writing quality into one undefined score.
- Use deterministic scoring when the target is deterministic. MedCTA tool names,
  actions, and exact reference-route agreement do not need an LLM judge.
- Create the rubric and pass/fail rule on development examples. Freeze both
  before scoring held-out results.
- Keep strict string or rule-based metrics as transparent secondary diagnostics.
  Do not switch metrics merely because exact match produced a low number.

### 2. Use a versioned, domain-specific rubric

- Store the rubric in the repository, with a semantic version and SHA-256 hash.
- Define every allowed label and its score. Include `not_scorable`; uncertainty
  must not be forced into `incorrect`.
- State how synonyms, partial answers, contradictions, unsupported additions,
  and multiple accepted references are handled.
- For biomedical answers, a correct diagnosis embedded in a contradictory or
  clinically unsafe answer is not fully correct.
- Require a short evidence-grounded rationale for audit, but calculate metrics
  from the structured label rather than free-form prose.

### 3. Choose and identify the judge deliberately

- Use a capable judge that can perform the domain task.
- Prefer a judge from a different provider/model family than the evaluated
  model. The pipeline refuses a detectable same-family judge unless an explicit
  override is recorded.
- Pin the exact deployment/model identifier. Record provider, endpoint host,
  SDK version, temperature, token limit, retry policy, and timestamp.
- Keep one judge configuration fixed across candidate models. Changing the
  judge changes the measuring instrument.

### 4. Blind and harden the prompt

- Do not show candidate model names, owners, branch names, or expected rankings
  to the judge.
- Treat candidate text as untrusted data and explicitly tell the judge to ignore
  instructions inside it. Test prompt-injection attempts.
- Tell the judge not to reward verbosity, confidence, formatting, or citations.
- Supply every accepted reference answer and the minimum task context needed to
  judge the answer.
- Use a strict output schema and one repair attempt for formatting only. Preserve
  both the invalid and repaired responses.

### 5. Measure judge stability and known biases

- Use temperature zero for ordinary judging and record it explicitly.
- Repeat headline judgments at least three times because APIs can remain
  nondeterministic at temperature zero.
- Rotate/reverse accepted-reference order across repeats and report label
  consistency.
- For pairwise evaluation, judge both A/B and B/A. Map the second result back to
  the original identities; inconsistent pairs require review or abstention.
- Before the held-out run, test prompt paraphrases, irrelevant verbosity,
  contradictory additions, copied reference phrases, and prompt injection.
- Inspect whether failures vary by answer length, candidate family, task type,
  case difficulty, or demographic/clinical slice where applicable.

### 6. Separate judging from model inference

- Save candidate responses first. Judge them later from disk.
- Never make an unlogged judge call inside the model-under-test trajectory.
- Re-score saved answers when the rubric changes; do not rerun candidate models.
- Judge errors, parse failures, `not_scorable` items, and candidate inference
  failures are separate quantities. Judge failures must not silently count as
  incorrect candidate answers.

### 7. Preserve a complete audit trail

Every judge run must save:

- source run ID and manifest hash;
- item IDs and candidate model identity (in provenance, not in the prompt);
- rubric ID, version, path, and hash;
- prompt-template version and prompt hash;
- judge model/family/backend and generation parameters;
- raw prompts, initial responses, repaired responses, parsed labels, errors, and
  per-item aggregates;
- exact denominators for every metric;
- incomplete-run progress after every item.

Secrets must never appear in artifacts.

### 8. Validate against blinded humans

- Draw a frozen, seeded, stratified sample of at least 30 items. Include judge
  passes, failures, partial answers, strict-match disagreements, and unstable
  judgments.
- Hide candidate identity, judge label, and judge rationale.
- Obtain two independent human labels using the same rubric, then adjudicate
  disagreements without exposing the judge decision.
- Report human-human and judge-human percent agreement, Cohen's kappa, confusion
  matrices, and confidence intervals. For ordinal scales, additionally report a
  weighted agreement measure.
- Read disagreements qualitatively and revise the rubric only on development
  data. A rubric or judge-model change requires revalidation.

No universal kappa threshold establishes validity. The required agreement
depends on the intended claim, class balance, and human-human ceiling. Report
the evidence and limitations rather than declaring the judge "accurate."

### 9. Report results without hiding uncertainty

- Report the judge-model version and validation results next to judged metrics.
- Include score denominators, judge failure counts, `not_scorable` counts,
  repetition consistency, and human-audit sample size.
- Keep route agreement and final-answer correctness as separate axes for Idea 3.
- Include sensitivity analyses when reasonable, and size claims to the tested
  models, tasks, judge, and rubric.

## Repository review

Several reputable projects implement valuable pieces, but none makes a custom
clinical judge rigorous automatically.

| Project | Useful parts | Missing for this project |
|---|---|---|
| [FastChat / MT-Bench](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge) | Widely used single and pairwise judging; pairwise order swap; raw judgments; released human annotations and agreement analysis. It accompanies the NeurIPS 2023 Datasets and Benchmarks paper on position, verbosity, and self-enhancement biases. | MT-Bench-specific data model; no enforced clinical rubric, different-family guard, blinded two-rater workflow, source-run hashing, or project-specific failure accounting. |
| [OpenAI Evals](https://github.com/openai/evals) | General benchmark registry and configurable model-graded evals. | A flexible runner, not a complete reliability protocol; custom human calibration and bias tests remain the researcher's responsibility. |
| [DeepEval](https://github.com/confident-ai/deepeval) | Mature developer tooling and configurable G-Eval-style metrics. | Does not enforce preregistration, different-family judging, project artifact contracts, or human validation. |
| [OpenEvals](https://github.com/langchain-ai/openevals) | Lightweight structured LLM judges and agent-trajectory evaluators. | Convenience APIs do not supply a validated medical rubric or enforce the full audit/validation workflow. |
| [Prometheus Eval](https://github.com/prometheus-eval/prometheus-eval) | Open evaluator models for absolute and pairwise rubric scoring; reports agreement on public preference benchmarks. | A judge model and inference library, not validation of this project's rubric or biomedical construct. |
| [AlpacaEval](https://github.com/tatsu-lab/alpaca_eval) | Human-validated pairwise evaluation, explicit annotator configs, and length-controlled analysis. | Built for instruction-following comparisons rather than reference-based clinical correctness. |

The project therefore implements a small internal layer and borrows the design
principles—not source code—from these systems. This keeps the evaluation tied to
MedRouteBench's saved traces and avoids adding a large framework that still
would not perform the required human validation.

## Evidence behind the protocol

- Zheng et al., [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://proceedings.neurips.cc/paper_files/paper/2023/hash/91f18a1287b398d378ef22505bf41832-Abstract-Datasets_and_Benchmarks.html), NeurIPS 2023 Datasets and Benchmarks.
- Chen et al., [Humans or LLMs as the Judge? A Study on Judgement Bias](https://aclanthology.org/2024.emnlp-main.474/), EMNLP 2024.
- Shi et al., [Judging the Judges: A Systematic Study of Position Bias in LLM-as-a-Judge](https://aclanthology.org/2025.ijcnlp-long.18/), IJCNLP-AACL 2025.
- The Algoverse July 19 implementation lecture: freeze the judge prompt, swap
  pairwise answer order, prefer a different model family, manually grade about
  30 items, and report judge-human agreement.

## Current implementation

`judge_eval/` provides:

- frozen JSON rubrics and strict judge-response validation;
- blinded, injection-aware prompt construction;
- repeated pointwise judgments with deterministic reference-order changes;
- a same-family judge guard;
- raw per-call artifacts, atomic writes, manifest hashes, and explicit failures;
- seeded blinded human samples with two labels and adjudication fields;
- percent agreement, Cohen's kappa, confusion matrices, and bootstrap intervals.

Idea 3 uses this layer through `python -m medcta_eval.judge`. Pairwise A/B order
swapping is a required extension if a future MedRouteBench experiment uses
pairwise preferences; it is not applicable to Idea 3's reference-based
pointwise correctness label.
