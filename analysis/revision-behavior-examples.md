# Concrete examples of revision behavior

Date prepared: 2026-07-13

Run: [`20260710T161217Z`](../staged_eval/runs/20260710T161217Z/report.json)

Reported model: `llama-3.1-8b-instant`

## Purpose and reading guide

These examples ground the aggregate analysis in individual PubMedQA traces. In each case, Stage 1 answered from the evidence available before the first `RESULTS` section (or before the fallback half split). Stage 2 then saw the added evidence and returned one of `KEEP_ANSWER`, `REVISE_ANSWER`, or `ABSTAIN`.

The question text comes from [`data/pubmedqa/ori_pqal.json`](../data/pubmedqa/ori_pqal.json); the trace files record the evidence and outputs. They do **not** record a model rationale. The explanations below are therefore analyst interpretations of the visible evidence, not claims about the model's hidden reasoning.

## Example index

| Behavior | PMID | Gold | Stage 1 | Stage 2 action | Final | Why it matters |
|---|---:|---:|---:|---|---:|---|
| Successful correction | 11138995 | no | yes | revise | no | Added results directly reject the broad risk-factor claim. |
| Successful correction from uncertainty | 15919266 | no | maybe | revise | no | Comparative outcomes show no benefit from adjuvant radiation. |
| Correct retention | 10158597 | yes | yes | keep | yes | The main benefit survives despite important secondary caveats. |
| Missed correction | 10201555 | yes | maybe | keep | maybe | Added adjusted effect estimates support a `yes` answer. |
| Harmful overreaction | 10548670 | yes | yes | revise | no | Added quantitative evidence supports, rather than overturns, Stage 1. |
| Wrong-direction revision | 19302863 | yes | maybe | revise | no | Stage 2 changes the answer but still moves away from the gold label. |
| Ambiguity collapsed to `no` | 11867487 | maybe | maybe | revise | no | Added evidence concerns beliefs and use, not efficacy. |
| Abstention despite directional evidence | 17054994 | no | yes | abstain | -- | Added results clearly say the procedure never changed management. |

## Successful revisions

### PMID 11138995: a direct negative result corrects an affirmative prior answer

**Question:** Is alexithymia a risk factor for unexplained physical symptoms in general medical outpatients?

**Observed path:** `yes` -> `REVISE_ANSWER` -> `no` (gold: `no`).

**What changed:** Stage 1 saw the study framing and methods and answered `yes`. The added results reported that patients with medically unexplained symptoms were not more alexithymic overall, and that alexithymia was not associated with subjective health or service use. A narrower subgroup association remained, but it did not support the broad question.

**Interpretation:** This is the intended revision behavior. Stage 2 uses outcome evidence to replace a hypothesis-consistent answer with the study's broader null result. It also shows why the question's scope matters: a subgroup signal should not override the overall finding.

Source: [`trace_11138995.json`](../staged_eval/runs/20260710T161217Z/trace_11138995.json)

### PMID 15919266: uncertainty resolves to `no` after a comparative null result

**Question:** Adjuvant radiation of stage III thymoma: is it necessary?

**Observed path:** `maybe` -> `REVISE_ANSWER` -> `no` (gold: `no`).

**What changed:** The added evidence reported no alteration in local or distant recurrence and similar 10-year disease-specific survival with and without radiation (79% versus 75%, `p = 0.21`).

**Interpretation:** Stage 2 appropriately converts uncertainty into a negative answer when the comparative results do not demonstrate benefit. This is a stronger example of evidence-based revision than merely switching because a `RESULTS` section contains negative language.

Source: [`trace_15919266.json`](../staged_eval/runs/20260710T161217Z/trace_15919266.json)

## Appropriate maintenance

### PMID 10158597: keeping the main answer while tolerating caveats

**Question:** Does a dedicated discharge coordinator improve the quality of hospital discharge?

**Observed path:** `yes` -> `KEEP_ANSWER` -> `yes` (gold: `yes`).

**What changed:** The added results showed improved discharge planning and fewer post-discharge problems, while finding no improvement in the timing of community services or bed-use efficiency.

**Interpretation:** Keeping `yes` is reasonable because the primary claim is supported even though not every secondary outcome improved. This trace demonstrates that a good revision policy must integrate mixed evidence rather than treating any caveat as a reason to reverse.

Source: [`trace_10158597.json`](../staged_eval/runs/20260710T161217Z/trace_10158597.json)

## Missed and harmful revisions

### PMID 10201555: strong new evidence is ignored

**Question:** Is low serum chloride level a risk factor for cardiovascular mortality?

**Observed path:** `maybe` -> `KEEP_ANSWER` -> `maybe` (gold: `yes`).

**What changed:** The added results described serum chloride as an independent predictor after adjustment, with elevated cardiovascular-death risk for low chloride in both men and women and a dose-response relationship.

**Interpretation:** This is a clear missed opportunity. The added evidence supplies the adjusted association and effect estimates needed to move from `maybe` to `yes`, yet Stage 2 leaves the preliminary answer unchanged.

Source: [`trace_10201555.json`](../staged_eval/runs/20260710T161217Z/trace_10201555.json)

### PMID 10548670: added evidence triggers the opposite of the supported answer

**Question:** Does the National Institutes of Health Stroke Scale favor left hemisphere strokes?

**Observed path:** `yes` -> `REVISE_ANSWER` -> `no` (gold: `yes`).

**What changed:** At comparable NIH Stroke Scale scores below 20, right-hemisphere infarcts were approximately twice as large as left-hemisphere infarcts. That pattern supports the proposition that the scale gives relatively more weight to deficits associated with left-hemisphere strokes.

**Interpretation:** The revision is harmful and difficult to justify from the visible result. A plausible failure mode is losing the direction of a comparative relationship: “right lesions are larger at the same score” is evidence of left-hemisphere score bias, not evidence against it.

Source: [`trace_10548670.json`](../staged_eval/runs/20260710T161217Z/trace_10548670.json)

### PMID 19302863: revising is not the same as correcting

**Question:** Is the use of cyanoacrylate in intestinal anastomosis a good and reliable alternative?

**Observed path:** `maybe` -> `REVISE_ANSWER` -> `no` (gold: `yes`).

**What changed:** The added evidence found no significant difference in bursting pressure and reported better hydroxyproline levels and anastomosis time for the cyanoacrylate subgroups.

**Interpretation:** Stage 2 reacts to the new evidence but chooses the wrong direction. This trace is counted as a missed revision, not an overreaction, because Stage 1 was already wrong. It illustrates why action counts alone are inadequate: `REVISE_ANSWER` can leave accuracy unchanged or make the semantic interpretation worse.

Source: [`trace_19302863.json`](../staged_eval/runs/20260710T161217Z/trace_19302863.json)

## Ambiguity and abstention

### PMID 11867487: non-efficacy evidence is treated as evidence of no efficacy

**Question:** Does rugby headgear prevent concussion?

**Observed path:** `maybe` -> `REVISE_ANSWER` -> `no` (gold: `maybe`).

**What changed:** The added evidence measured player and coach beliefs, headgear use, and reasons for non-use. It did not test whether headgear actually prevents concussion.

**Interpretation:** `maybe` remains the defensible answer because the added section does not resolve the causal question. The revision appears to conflate weak or indirect evidence with negative evidence.

Source: [`trace_11867487.json`](../staged_eval/runs/20260710T161217Z/trace_11867487.json)

### PMID 17054994: abstention is used where the result is unusually clear

**Question:** Does frozen section alter surgical management of multinodular thyroid disease?

**Observed path:** `yes` -> `ABSTAIN` -> no final answer (gold: `no`).

**What changed:** The added results reported 25% sensitivity and, most directly, that frozen-section analysis altered intraoperative management in none of 135 patients.

**Interpretation:** Abstention avoids retaining the wrong `yes`, but it also fails to convert highly directional evidence into the correct `no`. Because all three abstentions in the run had gold label `no`, the small abstention set is more consistent with unresolved negative evidence than with a calibrated, label-neutral reject option.

Source: [`trace_17054994.json`](../staged_eval/runs/20260710T161217Z/trace_17054994.json)

## Cross-example pattern

Taken together, the examples support three conclusions from the aggregate results:

1. Stage 2 can correctly exploit explicit null or negative findings, especially when revising `yes` or `maybe` to `no`.
2. It sometimes treats caveats, indirect evidence, or added complexity as a generic signal to move toward `no`, even when the evidence supports `yes` or remains inconclusive.
3. Many correctable Stage 1 errors remain untouched. The main limitation is not only harmful revision; it is failure to revise when added evidence is decisive.

These examples were selected to illustrate distinct behaviors, not to estimate their prevalence. Prevalence and class-level effects are reported in [`latest-run-results.md`](latest-run-results.md).
