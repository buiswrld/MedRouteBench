# Reproducibility and known issues

This release supports inspection and new evaluations of a retained implementation. It is not an exact historical rerun package for the ten-model paper experiment.

## Available

- The three evaluator packages, exact retained prompts and judge rubrics, schemas, adapters, metrics, and tests.
- PubMedQA inputs and test labels; the two MedCTA adapted JSON files with pinned source metadata and remote image references.
- New-run manifest generation, implemented resume checks, and usage/report utilities.
- An inventory of older artifacts found during cleanup. The unchanged original uploaded archive retains those traces and Git history.

## Unavailable or unverified

- The full original trace archive for the paper's ten models and three replicates per condition.
- Complete historical decoding settings, exact hosted snapshots, provider routing, and all final-run failure counts.
- A dependency lockfile verified against the historical experiment.
- Independent clinical validation of the judge, sufficient-evidence annotations, and a first-attempted-stop-to-endpoint analysis of the unavailable paper traces.
- Live API compatibility and full offline-test execution in the cleanup environment.

Current source defaults and new-run manifests do not retroactively establish missing historical settings. Some retained development manifests identify Azure and `gpt-5-mini`, whereas current candidate calls default to OpenRouter. Do not relabel those historical records as OpenRouter runs.

## PubMedQA denominator disagreement

See [metric definitions](metric_definitions.md). Code, tests, and paper prose use different denominator policies in some places. No evaluation formula or historical result has been silently modified in this cleanup.

## Known adapted-data issue: MedCTA case 19

Both adapter modules already contain `KNOWN_ERRATA` for case `19`. In the supplied `data/medcta/fullset_v1.json`, `reference_steps[3].reference_observation` duplicates step 2's observation. The existing adapter comment identifies a left-lung observation in place of the intended right-lung observation. This cleanup directly confirms the duplicate; it does not independently recover the intended replacement text or establish whether the upstream source has the same defect.

The data are preserved unchanged. No replacement was guessed, and no case was silently dropped. Results involving this case require that qualification. A corrected data version should have a new identifier/hash and a documented erratum; its results would constitute a different evaluation configuration. The impact on the historical paper metrics has not been measured.

## Paper artifacts versus legacy development artifacts

The archived working tree has seven older PubMedQA run directories with 970 trace files in total. Two contain 482 traces each, labeled `gpt-5-mini` with Azure in their manifests. No MedCTA run archive was found in the current tree or reachable historical report paths. The legacy inventory does not demonstrate recovery of the final ten-model paper records.

Do not combine these runs with new models or evidence orders. Some old manifests lack the current `reversed` field and have different prompt/code hashes, output-token limits, or seeds. Resume is not guaranteed to succeed, and relaxing manifest checks merely to force a resume is not a valid provenance fix.
