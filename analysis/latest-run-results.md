# Analysis of the latest complete staged-evaluation run

Date prepared: 2026-07-13

Run: [`staged_eval/runs/20260710T161217Z`](../staged_eval/runs/20260710T161217Z/report.json)

Reported model: `llama-3.1-8b-instant`

Cases: 500 PubMedQA examples with official gold labels

## Executive summary

The second stage improved accuracy from **36.2% to 41.6%**, a net gain of **27 correct answers** or **5.4 percentage points**. The paired change is unlikely to be sampling noise within this fixed test set: there were 44 gains versus 17 losses (exact two-sided McNemar `p = 0.00073`; approximate paired 95% confidence interval for the accuracy change: **+2.4 to +8.4 points**).

That headline improvement is real, but it does not show broad evidence integration:

- **All 44 successful revisions were gold-`no` cases.**
- Accuracy on gold `no` rose from 2.4% to 28.4%, while accuracy fell on gold `yes` from 55.4% to 50.0% and on gold `maybe` from 43.6% to 40.0%.
- The final 41.6% accuracy remains below the **55.2% always-`yes` majority baseline** on this label-imbalanced sample.
- Stage 2 behaves primarily like a negative-evidence detector: `no` predictions increase from 5 to 76, generating useful corrections but also 17 losses.
- The dominant error remains inaction. Of 316 initially wrong, non-abstained cases that could be assessed for correction, 272 (86.1%) were still wrong after Stage 2.

The appropriate conclusion is therefore: **the revision mechanism adds statistically detectable value over this weak Stage 1 result, but the current configuration is not yet a competitive QA system and the gain is concentrated in one answer class.**

## Run integrity

The latest complete full run in the repository is `20260710T161217Z`. Direct re-reading of its artifacts found:

- 500 trace files, all with `status: completed`;
- zero invalid or repaired model outputs;
- exact agreement between recomputed headline metrics and [`report.json`](../staged_eval/runs/20260710T161217Z/report.json);
- exact agreement of PMID, gold label, and evidence split with the current PubMedQA data files.

The run predates the newer manifest/resume hardening. Its results are usable for analysis, but the artifact has no run manifest with immutable configuration, dataset fingerprint, or independently verifiable model provenance. It should not be resumed in place; a replication should create a new run.

## Headline metrics

| Metric | Count | Rate |
|---|---:|---:|
| Stage 1 correct | 181 / 500 | 36.2% |
| Final correct | 208 / 500 | 41.6% |
| Net change | +27 / 500 | +5.4 pp |
| Successful revisions | 44 / 319 initially wrong | 13.8% |
| Missed revisions | 272 / 316 assessable wrong cases | 86.1% |
| Overreactions | 17 / 181 initially correct | 9.4% |
| Correct answers retained | 164 / 181 initially correct | 90.6% |
| Abstentions | 3 / 500 | 0.6% |
| Answered-case final accuracy | 208 / 497 | 41.9% |

“Assessable wrong cases” excludes the three initially wrong cases that ended in abstention. Under the run's scoring, abstentions receive no correct answer and are included as incorrect in overall final accuracy.

## Paired outcome analysis

Each case has both a Stage 1 and final correctness value, so the most informative comparison is paired:

| Paired outcome | Cases | Share |
|---|---:|---:|
| Wrong -> correct | 44 | 8.8% |
| Correct -> wrong | 17 | 3.4% |
| No correctness change | 439 | 87.8% |

The 44-to-17 imbalance yields the net gain of 27 cases. An exact two-sided McNemar test gives `p = 0.0007299`. A normal approximation applied to the item-level paired differences gives a 95% interval of `[0.0238, 0.0842]` for the accuracy change.

This supports a narrow causal statement about the pipeline on this fixed run: exposing Stage 2 to the added evidence and prior answer improved its final decisions relative to its own Stage 1 decisions. It does **not** isolate whether the gain comes from the new evidence, seeing more total text, the explicit revision instruction, or anchoring on the preliminary answer.

## Performance by gold class

| Gold label | N | Stage 1 correct | Stage 1 accuracy | Final correct | Final accuracy | Gains | Losses | Net |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| yes | 276 | 153 | 55.4% | 138 | 50.0% | 0 | 15 | -15 |
| no | 169 | 4 | 2.4% | 48 | 28.4% | 44 | 0 | +44 |
| maybe | 55 | 24 | 43.6% | 22 | 40.0% | 0 | 2 | -2 |

The overall improvement is the sum of a **+44 correction effect on `no`** and **-17 combined losses on `yes`/`maybe`**. There are no successful revisions into `yes` or `maybe`:

- 25 successful `yes -> no` revisions;
- 19 successful `maybe -> no` revisions;
- 0 successful revisions to either `yes` or `maybe`.

This asymmetry is the most important qualification on the headline result.

## Prediction and action shifts

### Answer distribution

| Answer | Stage 1 | Final |
|---|---:|---:|
| yes | 251 (50.2%) | 199 (39.8%) |
| no | 5 (1.0%) | 76 (15.2%) |
| maybe | 244 (48.8%) | 222 (44.4%) |
| abstain / no answer | 0 | 3 (0.6%) |

Stage 1 almost never emits `no`, despite `no` making up 33.8% of the sample. Stage 2 partially repairs this underproduction, but still predicts `no` less than half as often as its gold prevalence.

### Stage 2 actions

| Action | Count | Share | Correctness consequence |
|---|---:|---:|---|
| Keep | 423 | 84.6% | 164 stayed correct; 259 stayed wrong |
| Revise | 74 | 14.8% | 44 became correct; 17 became wrong; 13 stayed wrong |
| Abstain | 3 | 0.6% | All three were scored wrong and had gold `no` |

Among revisions, 59.5% improved correctness, 23.0% harmed correctness, and 17.6% changed the label without correcting the case. The 44 beneficial changes outweigh the 17 harmful ones, but the large `KEEP` volume contains most of the remaining error.

Revision was also label-dependent:

| Gold label | Keep | Revise | Abstain | Revision rate |
|---|---:|---:|---:|---:|
| yes | 258 | 18 | 0 | 6.5% |
| no | 120 | 46 | 3 | 27.2% |
| maybe | 45 | 10 | 0 | 18.2% |

## Confusion matrices

Rows are gold labels; columns are predictions.

### Stage 1

| Gold \\ Predicted | yes | no | maybe |
|---|---:|---:|---:|
| yes | 153 | 0 | 123 |
| no | 68 | 4 | 97 |
| maybe | 30 | 1 | 24 |

### Final

| Gold \\ Predicted | yes | no | maybe | abstain |
|---|---:|---:|---:|---:|
| yes | 138 | 17 | 121 | 0 |
| no | 39 | 48 | 79 | 3 |
| maybe | 22 | 11 | 22 | 0 |

The final matrix makes the tradeoff visible: many gold-`no` cases move out of `yes`/`maybe`, but there are also 28 final false-`no` predictions on gold-`yes` or gold-`maybe` cases. Of the 17 paired losses caused by Stage 2, 16 were revisions into `no` and one was a `yes -> maybe` revision.

## Effect of evidence-split strategy

| Split strategy | N | Stage 1 accuracy | Final accuracy | Gains | Losses | Net |
|---|---:|---:|---:|---:|---:|---:|
| Before first `RESULTS` section | 482 | 35.7% | 41.5% | 43 | 15 | +28 |
| Fallback half split | 18 | 50.0% | 44.4% | 1 | 2 | -1 |

Nearly all cases use the intended first-`RESULTS` split, and that group drives the aggregate gain. The fallback group is too small for a stable comparison, but it does not provide evidence of benefit in this run. Future reports should continue separating these strategies because “new results section” and “second half of an abstract” are different interventions.

## What the run supports

The run supports the following claims:

1. **Revision adds value relative to the model's preliminary answer.** The paired gain is positive and statistically detectable on this set.
2. **The value is concentrated in detecting negative conclusions.** All gains occur on gold-`no` cases, usually through `yes -> no` or `maybe -> no` transitions.
3. **The revision stage is conservative in action frequency but not fully calibrated.** It keeps 84.6% of answers, misses many decisive updates, and occasionally reverses answers that the added evidence supports.
4. **Better revision does not compensate for a weak initial classifier.** Both Stage 1 and final accuracy are below the simple majority baseline.

Concrete trace-level examples for each behavior are in [`revision-behavior-examples.md`](revision-behavior-examples.md).

## What the run does not establish

- It does not show that staged prompting beats a single-pass model given the same full evidence.
- It does not establish model-agnostic generality; only one reported model and one run are represented.
- It does not separate the value of added evidence from the value of a second inference call.
- It does not demonstrate calibrated abstention; there are only three abstentions.
- It does not support resuming this legacy run after configuration or code changes because no manifest was written.
- It is a single 500-case fixed evaluation, not a multi-seed estimate of stochastic model variance.

## Recommended next analyses

1. **Run a blind full-context control.** Give the same model the full abstract once, without the Stage 1 answer. This is the highest-priority control for measuring whether revision itself helps beyond access to full evidence.
2. **Replicate under the hardened runner.** Use a new run directory so the manifest records exact configuration, prompt version, dataset fingerprint, and model identifier.
3. **Report macro metrics and balanced accuracy.** Overall accuracy hides the extreme class-specific behavior; per-class recall and macro recall should be first-class outputs.
4. **Repeat across models and stochastic replicates.** The pipeline is now model-configurable, but the behavioral claim needs evidence beyond one model/run.
5. **Audit `KEEP` errors and false-`no` revisions separately.** The first group tests evidence sensitivity; the second tests polarity, comparison direction, and handling of indirect evidence.
6. **Evaluate abstention selectively.** Measure risk versus coverage and check whether abstention improves answered-case accuracy; in this run it barely changes it (41.6% overall versus 41.9% among answered cases).

## Calculation notes

Counts were recomputed directly from all 500 trace JSON files rather than copied only from the aggregate report. Percentages are rounded to one decimal place in tables. The confidence interval is a normal approximation for the mean paired correctness difference; the McNemar value is an exact two-sided test using the 44 discordant gains and 17 discordant losses.

Primary sources:

- [`report.json`](../staged_eval/runs/20260710T161217Z/report.json)
- [`trace directory`](../staged_eval/runs/20260710T161217Z/)
- [`PubMedQA source records`](../data/pubmedqa/ori_pqal.json)
- [`official test labels`](../data/pubmedqa/test_ground_truth.json)
