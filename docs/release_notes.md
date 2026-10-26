# Repository cleanup and release checklist

## Changes made

- Replaced the PubMedQA-only root README with documentation of all three protocols, exact input files, installation, API setup, commands, scoring, outputs, citation, and reproducibility limits.
- Rewrote the three package READMEs to match retained control flow and denominators. Removed misleading statements about stopping on every tool mismatch, universally matched accuracy cohorts, clinically validated scoring, and guaranteed equality across separate stochastic runs.
- Added `requirements-dev.txt` with pytest, data provenance/checksums, third-party notices, citation metadata, known issues, and an inventory of surviving development artifacts.
- Excluded generated `runs/` directories, Python bytecode/cache files, the old Git object database, macOS archive metadata, the stale worklog, and unused standalone reference scripts from the cleaned snapshot. The original uploaded archive and its history remain unchanged.
- Cleared notebook execution metadata. Its input cells remain.
- Disabled the preconfigured three-replicate paid batch in `script.py`. Experiments must now be selected explicitly; use `--dry-run` to inspect the plan. The default disabled state produces an explanatory error, not model calls.
- Corrected a stale API-mode docstring in `staged_eval/llm.py` and expanded ignore patterns for new local artifacts.

Evaluation prompts, schemas, runner logic, metric formulas, configuration defaults, and the four production data JSON files are otherwise unchanged. No benchmark result was recomputed, and no model or judge request was made. A docstring edit changes its containing file's hash; historical resume checks may therefore reject that file even though its request behavior is unchanged.

## What still needs an author/code-owner decision

1. **Reconcile the metric contract.** PubMedQA code/tests use valid/completed cohorts in places where the paper describes the fixed selected cohort. Missed-revision denominators also treat abstentions differently. Documentation now describes the actual implementation and flags the disagreement. Decide the intended public protocol and identify the historical evaluator version; version any formula correction rather than silently overwriting records. The supplied data do not establish how this affected the final paper numbers.
2. **Choose a license for the team's code.** No top-level license exists in the supplied snapshot. Dataset licenses do not provide a license for the team's implementation.
3. **Run the full offline suite in an installed environment.** Dependency installation was blocked here. Syntax and dependency-free checks passed; the full pytest suite and hosted API compatibility were not verified.
4. **Record the MedCTA case 19 erratum.** The duplicate is confirmed, but the intended replacement and effect on historical numbers are unverified. Keep the warning with unchanged data, or issue an explicitly versioned correction after checking its source. Do not silently edit or drop the case while claiming identical benchmark conditions.
5. **Publish a checked repository and add its real link to the paper.** No remote repository was changed or made public by this cleanup. A clean ZIP does not erase commits from an existing remote repository; commit the intended files or publish a reviewed source snapshot deliberately.

## Validation performed

- Parsed all 48 retained Python files with Python 3.10 syntax compatibility selected.
- Exercised JSON/schema helpers from all three packages without network calls.
- Executed the retained pure splitting/eligibility functions on the supplied PubMedQA files: 1,000 raw records, 500 official gold IDs, and 482 eligible cases with the expected 264/165/53 labels.
- Verified production data copies byte-for-byte against the source archive and recorded SHA-256 checksums.
- Confirmed the case 19 duplicated observation directly.
- Demonstrated the PubMedQA denominator discrepancy with the actual retained metric functions on a synthetic completed/invalid pair: Stage-1 denominator 2, final denominator 1. This is a code check, not a new model-performance result.
- Scanned current files and 2,085 reachable historical Git objects for several high-confidence credential patterns; no matches were found. This was a limited pattern scan, not a guarantee that every possible secret, private path, or private record is absent. The release copy excludes the historical object database and generated responses.
- Inspected reachable historical reports: models named `gpt-5-mini` and `llama-3.1-8b-instant`, with no MedCTA report blobs found among reachable report paths. This does not exhaustively audit unreachable Git objects or external backups.
- No pytest pass count is claimed. No dependency lockfile or historical model settings were fabricated.

Structured check summary: [validation.json](validation.json).
