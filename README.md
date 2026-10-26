# MedRouteBench

Code for **MedRouteBench: Evaluating Answer Revision and Stopping under Staged Medical Evidence**, accepted as a poster at the non-archival GenAI4Health workshop at NeurIPS 2026.

MedRouteBench measures how answers change and when models stop under fixed biomedical evidence sequences. It contains a two-stage PubMedQA task and two MedCTA reference-evidence replay conditions.

**Release status:** this is a retained implementation snapshot, not a verified archive that exactly regenerates every paper result. Historical traces for the paper's ten-model experiment and complete final-run settings are unavailable following storage corruption. Older development traces survive in the original author archive; they must not be presented as the paper experiment. See [reproducibility and known issues](docs/reproducibility.md).

The retained PubMedQA metric implementation has denominator rules that differ from the paper's stated fixed-cohort rules. Those rules are documented below rather than silently changed during repository cleanup. The full MedCTA adaptation also has a known duplicated observation in case `19`. These are substantive limitations, not just installation details.

## Protocols

| Package | Task | Evidence and stopping behavior |
|---|---|---|
| [`staged_eval`](staged_eval/README.md) | PubMedQA answer revision | Stage 1 receives the question and pre-results text; Stage 2 receives the prior answer, added evidence, and complete context. `--reversed` swaps which block comes first. |
| [`medcta_eval`](medcta_eval/README.md) | MedCTA stopping-allowed replay | Each valid tool call reveals the next **recorded reference observation**, including when the selected tool differs. An early final answer ends replay. |
| [`medcta_golden_eval`](medcta_golden_eval/README.md) | MedCTA forced continuation | Early final answers and alternative tool choices are recorded and overridden; remaining reference observations are supplied. |

These are controlled replay evaluations, not live execution of clinical tools. A tool mismatch is disagreement with one reference route, not proof that an alternative route is invalid. An early answer may already be adequate. A correctness transition does not establish evidence sufficiency, clinical reasoning validity, or a causal effect of the newly supplied evidence.

## Installation

Use Python 3.10 or newer and run commands from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
# Add the offline test dependency:
python -m pip install -r requirements-dev.txt
```

On Windows, activate with `.venv\Scripts\activate` instead. Notebook use additionally requires your preferred Jupyter installation.

`requirements.txt` specifies minimum package versions, not the historical environment used for the paper. There is no verified historical lockfile. An installation with current packages is not proof of compatibility with every hosted model or provider endpoint.

## API configuration

Copy `.env.example` to `.env` and fill in your own values. Environment variables already set in the shell take precedence over `.env`.

```bash
cp .env.example .env
```

| Setting | Purpose |
|---|---|
| `OPENROUTER_API_KEY` | Candidate-model access through OpenRouter; `OPENROUTER_KEY` is a legacy alias. |
| `OPENROUTER_MODEL` | Actual OpenRouter model identifier. Choose an available identifier; paper display names are not API identifiers. |
| `MEDCTA_JUDGE_PROVIDER` | `azure` by default; `openrouter` is also supported. |
| `MEDCTA_JUDGE_DEPLOYMENT` or `MEDCTA_JUDGE_MODEL` | Judge deployment/model identifier. The paper describes a fixed GPT-5.4 judge. |
| `JUDGE_AZURE_ENDPOINT`, `JUDGE_AZURE_API_KEY` | Separate Azure judge endpoint and credential when using the default judge route. |
| `MAX_TOKENS` | PubMedQA output-token limit; retained default: 2048. |
| `MEDCTA_MAX_TOKENS` | MedCTA candidate output-token limit; retained default: 512. |
| `LLM_API` | PubMedQA API mode: `responses` by default, or `chat`. |
| `SEED` | PubMedQA chat-request seed, default 42. It is not sent by the Responses path and is not a determinism guarantee. |
| `WORKERS` | Shared concurrency default, 4; MedCTA CLI runs sequentially unless `--workers` is set. |

The code does not explicitly set temperature or reasoning effort in the normal API calls. Provider defaults apply. Current code defaults must not be labeled as verified final-run settings from the paper. Changing the judge provider/model defines a different evaluation configuration.

API commands below make paid model calls, and MedCTA additionally makes judge calls and retrieves remote images. `--use-fixtures` changes the dataset; it does **not** make inference offline. Keep credentials in `.env` or environment variables, not in committed scripts or command examples.

## Data

| Local file | Contents |
|---|---|
| `data/pubmedqa/ori_pqal.json` | 1,000 expert-labeled PQA-L records. |
| `data/pubmedqa/test_ground_truth.json` | Labels for the predefined 500-case test partition. The loader filters cases by these IDs. |
| `data/medcta/subset_v1.json` | 11-case starter adaptation; the MedCTA default input. |
| `data/medcta/fullset_v1.json` | 107-case adaptation used for full-cohort replay. Images remain remote URLs. |
| `staged_eval/fixtures/pubmedqa/` | Small explicit fixtures for development. |

For PubMedQA, the first normalized section label containing `RESULT` defines the boundary. Cases need a gold label, a usable boundary, and non-empty text on both sides. Applied to the supplied files, this yields 482 test cases: 264 `yes`, 165 `no`, and 53 `maybe`.

The MedCTA adaptation records source revision `0be777092121a18ba2a85a0c4145a9d4fe8ed7db` from [IVUL-KAUST/MedCTA](https://huggingface.co/datasets/IVUL-KAUST/MedCTA). Source information and checksums are in [data provenance](docs/data_provenance.md). **Setting `--n 107` alone does not select the full MedCTA file:** pass `--cases data/medcta/fullset_v1.json` as well.

## Running new evaluations

These commands run the retained implementation. They are not a claim that current APIs reproduce the published numbers. Configure `OPENROUTER_MODEL` before running.

### PubMedQA: normal and reversed evidence

```bash
python -m staged_eval.pipeline \
  --n 482 --workers 1 --model "$OPENROUTER_MODEL" \
  --cases data/pubmedqa/ori_pqal.json \
  --ground-truth data/pubmedqa/test_ground_truth.json

python -m staged_eval.pipeline \
  --n 482 --workers 1 --model "$OPENROUTER_MODEL" --reversed \
  --cases data/pubmedqa/ori_pqal.json \
  --ground-truth data/pubmedqa/test_ground_truth.json
```

Use a smaller `--n` for a pilot. Leave `--limit` unset for full-cohort evaluation: it caps raw cases **before** eligibility filtering. Repeat the command three times for three new runs; this does not reconstruct lost historical replicates.

### MedCTA: stopping allowed and forced continuation

```bash
python -m medcta_eval.pipeline \
  --n 107 --workers 1 --model "$OPENROUTER_MODEL" \
  --cases data/medcta/fullset_v1.json

python -m medcta_golden_eval.pipeline \
  --n 107 --workers 1 --model "$OPENROUTER_MODEL" \
  --cases data/medcta/fullset_v1.json
```

`script.py` supports explicit batches, but no experiment is enabled in the cleaned snapshot. Edit `EXPERIMENTS`, review `python script.py --dry-run`, then run it deliberately. See each package's `--help` for options. Resume with the same cohort, model, API configuration, and `--resume PATH`; manifests must pass the implemented provenance checks.

## Prompts, scoring, and outputs

The actual retained prompts and judge rubrics are in:

- [`staged_eval/prompts.py`](staged_eval/prompts.py): Stage 1 and Stage 2 instructions, evidence formatting, and repair instructions.
- [`medcta_eval/prompts.py`](medcta_eval/prompts.py) and [`medcta_golden_eval/prompts.py`](medcta_golden_eval/prompts.py): candidate prompts, correctness rubric, equivalence rubric, and repair instructions.
- Each package's `schema.py`, `runner.py`, and `metrics.py`: validation, execution/failure handling, and exact aggregation rules.

PubMedQA accepts the answer labels `yes`, `no`, and `maybe`. `maybe` is not a separately annotated deferral; `ABSTAIN` is a distinct Stage-2 action. Both stages are scored against the full-abstract label.

MedCTA correctness uses judge score `>= 0.8`; equivalence uses a separate `>= 0.8` threshold. A missing correctness score fails the threshold; a missing equivalence score is treated as a change. Raw-score means include only available scores. Read [metric definitions](docs/metric_definitions.md) before interpreting denominators or comparing configurations.

New runs create a unique `<package>/runs/<timestamp>_<id>/` directory containing a manifest, per-case traces, partial/final reports, and usage records as implemented by the pipeline. Traces can include supplied prompts, gold labels used only for scoring, model outputs, and judge results. Usage ledgers are not a complete billing audit: Azure dollar costs may remain unpriced and failed attempts may lack a usage record.

For existing **MedCTA** traces, reporting can run without new inference:

```bash
python -m medcta_eval.pipeline --report-existing PATH_TO_SAVED_RUN
python -m medcta_golden_eval.pipeline --report-existing PATH_TO_SAVED_RUN
```

Historical manifests are not automatically compatible with the current code. PubMedQA cross-run analysis requires explicit compatible run selections; do not mix models, orders, or old schemas just because they share PMIDs.

## Tests and validation status

After installing development dependencies:

```bash
python -m pytest staged_eval/tests medcta_eval/tests medcta_golden_eval/tests shared/tests -q
```

These are offline tests with mocked calls. For this cleanup, source compilation and dependency-free structural checks were performed. The full pytest suite and live API compatibility could not be verified in the cleanup environment because required packages were unavailable and dependency installation was blocked. No live model or judge calls were made. See [release notes](docs/release_notes.md).

## Reproducibility, licensing, and citation

- Historical final-run settings, the final ten-model trace archive, and independent expert validation of the judge are not supplied here.
- Seven older `gpt-5-mini` Azure run directories, including two complete 482-case runs, survive in the original uploaded archive. They are inventoried in [legacy artifacts](docs/legacy_artifacts.md), not included as the paper's results.
- **A license for the team's own code has not been selected in the supplied repository.** The cleanup does not invent one. The code owner should add the chosen license before describing this as an open-source release.
- Dataset and third-party materials retain their own terms; see [third-party notices](THIRD_PARTY_NOTICES.md). The team's eventual code license must not be presented as licensing the source datasets, model services, or upstream reference scripts.

```bibtex
@misc{man2026medroutebench,
  title = {MedRouteBench: Evaluating Answer Revision and Stopping under Staged Medical Evidence},
  author = {Man, David and Xu, Chuanhai and Tran, Quang and Liu, Joshua and Bui, Ryan and Liu, Benjamin and Zhu, Kevin},
  year = {2026},
  howpublished = {NeurIPS 2026 GenAI4Health Workshop, non-archival poster}
}
```

Please also cite the original [PubMedQA paper](https://aclanthology.org/D19-1259/) and [MedCTA paper](https://arxiv.org/abs/2606.11702) when using their data.
