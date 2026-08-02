# MedRouteBench Worklog

This file records repository changes, verification evidence, and decisions that
future contributors should know. Dates use the project timezone
(`America/New_York`).

## 2026-08-01 — Offline, validated LLM-as-a-judge layer

### What was done

- Removed the same-model, in-trajectory MedCTA judge call. Candidate inference
  now saves answers first and computes only a transparent normalized exact-match
  diagnostic.
- Added `judge_eval`, a reusable offline judge package with a frozen JSON rubric,
  blinded prompts, strict response validation, one formatting repair, repeated
  calls, accepted-reference order variation, and atomic raw artifacts.
- Added a default safeguard against using the same model family as both
  candidate and judge. An override is possible but is explicit in provenance.
- Added `medcta_eval.judge` to rescore saved Idea 3 traces without rerunning the
  candidate model. Judge failures, `not_scorable` items, inconsistent repeats,
  and candidate inference failures remain separate.
- Added a seeded blinded human-audit sample requiring two independent labels
  and adjudication, plus percent agreement, unweighted and quadratic-weighted
  Cohen's kappa, confusion matrices, and bootstrap intervals.
- Added judge/model/generation metadata, rubric and prompt hashes, source-run
  manifest hash, code hashes, raw prompts/responses, Wilson intervals, and exact
  denominators to judge artifacts.
- Documented the full protocol and compared FastChat/MT-Bench, OpenAI Evals,
  DeepEval, OpenEvals, Prometheus Eval, and AlpacaEval. None replaces the need to
  validate this biomedical rubric against blinded humans.

### Decisions

- Keep deterministic routing metrics deterministic. Use LLM judging only for
  open-ended clinical answer correctness.
- Preserve exact matching as a diagnostic and publish semantic judged results
  only after human validation.
- Treat routing agreement and answer correctness as separate Idea 3 outcomes.
- Prefer a small internal layer tied to MedRouteBench artifacts over adopting a
  large general framework that does not enforce the research protocol.

### Verification

- `.venv/bin/python -m pytest -q`: **104 passed**.
- The judge and human-validation suites use injected offline backends; tests
  make no paid API calls.

## 2026-07-12 — Staged evaluation reliability hardening

### What was done

- Reviewed the fixed two-stage PubMedQA evaluation path and identified
  correctness and reproducibility risks around model identity, metric cohorts,
  fixture fallback, artifact writes, resume behavior, and environment
  precedence.
- Made the pipeline backend-neutral for Python callers. A custom backend only
  needs to implement `(system: str, user: str) -> str`; the Groq adapter is
  loaded lazily only when the built-in backend is selected.
- Plumbed the selected model into the actual Groq request so the model recorded
  in reports and manifests cannot differ from the requested model.
- Changed Stage 1 and final answer accuracy to use the same selected-case
  denominator. Invalid outputs and final abstentions now count as incorrect
  rather than disappearing from the final cohort.
- Extended the same anti-survivor-bias treatment to conditional revision,
  abstention, and maintenance metrics where their Stage 1 cohort is known.
- Removed silent production-to-fixture fallback. Fixture runs now require
  `use_fixtures=True` or `--use-fixtures`; explicit case and ground-truth
  paths are supported.
- Added unique microsecond-plus-random run IDs, atomic JSON writes, and a
  `manifest.json` containing model/backend identity, generation settings,
  data hashes, code hashes, action schema, and fixture status.
- Hardened `--resume`: runs without a manifest are rejected, provenance
  mismatches are rejected, unreadable partial traces are recomputed, and saved
  traces must still match the current PMID and gold label.
- Changed `.env` loading so shell, scheduler, and CI environment values take
  precedence over local defaults.
- Lazy-loaded package and runner entry points to keep custom backends independent
  of the Groq SDK and to remove the module-execution warning from the CLI.
- Updated `README.md` with backend usage, explicit fixtures, artifact layout,
  metric semantics, resume constraints, and legacy-run behavior.
- Expanded the offline suite from 44 to 50 tests with adversarial coverage for
  model plumbing, invalid-output denominators, explicit fixture selection,
  unique run directories, resume provenance, and corrupted trace recovery.

### Decisions

- Keep the command-line built-in backend focused on Groq while making the core
  Python pipeline provider-agnostic through injected callables.
- Treat invalid model output as benchmark failure for accuracy metrics instead
  of conditioning headline results on successful parsing.
- Prefer failing fast on missing production data over silently producing a
  valid-looking fixture report.
- Require verifiable manifests for resume rather than guessing whether a legacy
  run is compatible.
- Preserve old artifacts unchanged; recompute analyses from their traces when
  compatible instead of rewriting historical reports in place.

### Verification

- `python -m pytest staged_eval/tests -q`: **50 passed**.
- `python -m compileall -q staged_eval`: passed.
- `python -m pip check`: no broken requirements.
- `git diff --check`: passed.
- Zero-case CLI smoke test created a unique manifest/report without making an
  API call and recorded the requested model correctly.
- Custom backend selection succeeded with the `groq` module deliberately
  blocked, confirming that provider injection no longer imports Groq.
- A shell-level `GROQ_MODEL` sentinel remained authoritative over `.env`.

### Legacy result audit

- `staged_eval/runs/20260710T161217Z` remains usable for analysis: all 500
  traces are valid and completed, and all PMIDs, gold labels, and evidence
  splits match the current dataset.
- Recomputing the corrected metrics from those 500 traces produced the same
  values as the historical report because that run had zero invalid outputs.
- `staged_eval/runs/20260710T161150Z` is also schema-compatible but contains
  only four cases.
- The July 7 runs use the older many-stage schema and must not be merged with
  or directly compared to the July 10 fixed two-stage experiment.
- Legacy runs have no provenance manifest. They remain readable but cannot be
  resumed safely, and their artifacts alone cannot independently prove the
  backend configuration used at execution time.

### Remaining follow-up

- Add `pytest` to a development/test dependency set so the documented test
  command works in a fresh environment.
- Remove the tracked `staged_eval/tests/__pycache__/*.pyc` artifacts.
- For publication-grade reproduction, rerun the 500-case experiment with the
  hardened manifest-producing pipeline.
