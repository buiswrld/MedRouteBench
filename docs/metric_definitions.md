# Metric definitions in the retained code

This document describes the checked-in implementation. It does not establish the settings or evaluator version used for every historical paper run. Metric functions return numerators and denominators; inspect them alongside rates.

## PubMedQA

| Metric | Implemented denominator | Implemented numerator |
|---|---|---|
| Stage-1 accuracy | Gold-labeled, non-skipped traces with a valid Stage-1 answer | Answer equals full-abstract gold label |
| Final accuracy | Traces with `status == "completed"` | Final answer equals full-abstract gold label; completed abstentions are incorrect |
| Successful revision | Completed traces with initially wrong answers | `REVISE_ANSWER` ends on gold label |
| Missed revision | Completed traces with initially wrong answers **and non-abstaining Stage 2** | Final answer remains wrong |
| Overreaction | Completed traces with initially correct answers | `REVISE_ANSWER` ends on a wrong answer |
| Kept correct | Completed traces with initially correct answers | `KEEP_ANSWER` retains the gold label |
| Final abstention | Completed traces | `ABSTAIN` at Stage 2 |
| Maintenance | Completed, non-abstaining traces | Final answer equals Stage-1 answer |

### Known disagreement with the paper's description

The paper describes both accuracies over the same fixed selected cohort, counting invalid outputs as incorrect. The retained code instead omits invalid Stage-1 answers from Stage-1 accuracy and invalid/incomplete cases from final accuracy. Its tests explicitly encode this completed-case policy. Consequently, the two accuracies need not have the same denominator.

The paper also describes missed revision with all initially wrong answers as denominator and abstentions excluded from its numerator. The retained implementation removes abstentions from that denominator as well. The distinction can matter even in runs without invalid outputs.

The earlier README and worklog claimed fixed-cohort handling; that claim did not match `staged_eval/metrics.py`. Cleanup corrects the documentation, not the formulas. Do not infer that a new code change explains or validates the historical results. Resolve the discrepancy against the intended protocol and the historical evaluator version before claiming exact paper reproduction. If formulas are changed later, version that change explicitly and preserve existing artifacts.

Matching the full-abstract gold label is not a stage-specific evidence-sufficiency judgment. `maybe` can be an appropriate preliminary answer without matching that label.

## MedCTA

- `status == "inference_failure"` traces are excluded from evaluable-case cohorts. Other statuses, including invalid actions, remain evaluable as implemented.
- Final accuracy counts evaluable cases passing the correctness threshold. Missing correctness scores do not pass. In stopping-allowed replay, calling another tool at the terminal reference position leaves no final answer in the retained implementation; the case does not count as correct.
- Raw-score means average available numeric scores, so `n_scored` may differ from the accuracy denominator.
- Consecutive-step transition rates condition on the previous step's correctness, at threshold 0.8; a case may contribute multiple transitions.
- Equivalence uses a separate judge and threshold 0.8. Identical normalized text short-circuits to equivalence 1.0. Missing equivalence scores are treated as changed, not as verified stability.
- Next-tool agreement counts unreached reference tool steps as misses. It compares with one reference route and does not establish clinical validity.
- Early-finalization progress and answer scores condition on cases with an early declaration. The stopping-allowed and forced-continuation conditions have different observed histories and are separate stochastic runs: matching those conditional definitions does not guarantee equal cohorts, answers, or scores.
- Forced continuation changes observations received, action history, and the number of calls after an attempted early exit. Its endpoint advantage does not isolate a causal cost of stopping.

The definitive implementation is in the three `metrics.py` files and the corresponding `runner.py` files. The LLM judge has not been independently calibrated against expert clinical ratings in this release.
