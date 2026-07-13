"""
Offline test suite — no network, no LLM calls.

Run with:  cd MedRouteBench && python -m pytest staged_eval/tests -q
"""
from idea1_two_stage_eval.schema import (
    ACTIONS, STAGE1_ALLOWED, STAGE2_ALLOWED, AgentOutput, safe_json_loads, validate,
)
from idea1_two_stage_eval.data import (
    load_cases, load_ground_truth,
    labeled_contexts, normalized_labels, split_revision_evidence, stratified_sample,
)
from idea1_two_stage_eval.prompts import (
    REPAIR_TEMPLATE,
    STAGE1_SYSTEM_PROMPT,
    STAGE2_SYSTEM_PROMPT,
    build_stage1_user_prompt,
    build_stage2_user_prompt,
)
from idea1_two_stage_eval.metrics import (
    build_report,
    final_abstention_rate,
    final_answer_accuracy,
    kept_correct_rate,
    maintenance_rate,
    missed_revision_rate,
    overreaction_rate,
    stage1_answer_accuracy,
    successful_revision_rate,
)
from idea1_two_stage_eval.runner import run_case


# ── helpers ───────────────────────────────────────────────────────────────────

_SMOKE_N = 5


def _cases_and_gt():
    return load_cases(limit=_SMOKE_N), load_ground_truth()


# ── schema ────────────────────────────────────────────────────────────────────

def test_validate_good():
    out, err = validate({
        "action": "ANSWER", "answer": "yes", "confidence": 0.8,
        "reason_for_action": "clear evidence",
    }, stage=1)
    assert isinstance(out, AgentOutput) and err is None
    assert out.action == "ANSWER" and out.answer == "yes"


def test_validate_missing_key():
    _, err = validate({}, stage=1)
    assert err is not None and "missing key" in err


def test_validate_bad_action():
    _, err = validate({
        "action": "CALL_TOOL", "answer": None, "confidence": 0.5,
        "reason_for_action": "x",
    }, stage=1)
    assert err is not None and "invalid action" in err


def test_validate_unknown_answer():
    _, err = validate({
        "action": "ANSWER", "answer": "dunno", "confidence": 0.5,
        "reason_for_action": "x",
    }, stage=1)
    assert err is not None


def test_validate_normalises_case():
    out, err = validate({
        "action": "answer", "answer": "YES", "confidence": 0.9,
        "reason_for_action": "check",
    }, stage=1)
    assert err is None
    assert out.action == "ANSWER" and out.answer == "yes"


def test_validate_confidence_clamp():
    out, err = validate({
        "action": "ABSTAIN", "answer": None, "confidence": 1.5,
        "reason_for_action": "overshoot",
    }, stage=2, prior_answer="yes")
    assert err is None and out.confidence == 1.0


def test_validate_stage2_keep_requires_same_answer():
    _, err = validate({
        "action": "KEEP_ANSWER", "answer": "no", "confidence": 0.5,
        "reason_for_action": "staying put",
    }, stage=2, prior_answer="yes")
    assert err is not None and "KEEP_ANSWER" in err


def test_validate_stage2_revise_requires_change():
    _, err = validate({
        "action": "REVISE_ANSWER", "answer": "yes", "confidence": 0.5,
        "reason_for_action": "changing",
    }, stage=2, prior_answer="yes")
    assert err is not None and "REVISE_ANSWER" in err


def test_safe_json_loads_good():
    obj, err = safe_json_loads('{"ok": true}')
    assert obj == {"ok": True} and err is None


def test_safe_json_loads_bad():
    obj, err = safe_json_loads("not json{{{")
    assert obj is None and err is not None


# ── data ──────────────────────────────────────────────────────────────────────

def test_load_cases():
    cases, _ = _cases_and_gt()
    assert len(cases) == _SMOKE_N
    assert all("pmid" in c and "CONTEXTS" in c and "QUESTION" in c for c in cases)


def test_normalized_labels_align_to_contexts():
    cases, _ = _cases_and_gt()
    labels = normalized_labels(cases[0])
    assert len(labels) == len(cases[0]["CONTEXTS"])
    assert all(isinstance(label, str) and label for label in labels)


def test_labeled_contexts_zip_labels_and_contexts():
    cases, _ = _cases_and_gt()
    pairs = labeled_contexts(cases[0])
    assert len(pairs) == len(cases[0]["CONTEXTS"])
    assert all(isinstance(label, str) and isinstance(context, str) for label, context in pairs)


def test_split_revision_evidence_prefers_results_boundary():
    case = {
        "pmid": "X",
        "QUESTION": "Q?",
        "CONTEXTS": ["Intro", "Methods", "Results"],
        "LABELS": ["BACKGROUND", "METHODS", "RESULTS"],
    }
    prepared = split_revision_evidence(case)
    assert prepared is not None
    assert prepared["split_strategy"] == "results_boundary"
    assert [label for label, _ in prepared["stage1_evidence"]] == ["Background", "Methods"]
    assert [label for label, _ in prepared["stage2_added_evidence"]] == ["Results"]


def test_split_revision_evidence_skips_non_results_cases():
    case = {
        "pmid": "Y",
        "QUESTION": "Q?",
        "CONTEXTS": ["A", "B", "C", "D"],
        "LABELS": ["BACKGROUND", "METHODS", "DISCUSSION", "CONCLUSION"],
    }
    assert split_revision_evidence(case) is None


def test_split_revision_evidence_skips_results_first_cases():
    case = {
        "pmid": "Y2",
        "QUESTION": "Q?",
        "CONTEXTS": ["Results first", "Later"],
        "LABELS": ["RESULTS", "DISCUSSION"],
    }
    assert split_revision_evidence(case) is None


def test_split_revision_evidence_skips_one_chunk_cases():
    case = {
        "pmid": "Z",
        "QUESTION": "Q?",
        "CONTEXTS": ["Only one"],
        "LABELS": ["RESULTS"],
    }
    assert split_revision_evidence(case) is None


def test_test_set_has_482_results_boundary_cases():
    cases = load_cases()
    prepared = [split_revision_evidence(case) for case in cases]
    prepared = [case for case in prepared if case is not None]
    assert len(prepared) == 482


def test_stratified_sample_size_and_diversity():
    cases = load_cases()
    gt = load_ground_truth()
    sample = stratified_sample(cases, gt, 15)
    assert len(sample) == 15
    labels = {gt.get(c["pmid"]) for c in sample}
    assert len(labels) > 1, "stratified sample should cover multiple labels"


# ── prompts ───────────────────────────────────────────────────────────────────

def test_stage1_prompt_mentions_preliminary_evidence():
    cases, _ = _cases_and_gt()
    prepared = split_revision_evidence(cases[0])
    p = build_stage1_user_prompt(prepared)
    assert "PRELIMINARY EVIDENCE" in p
    assert prepared["QUESTION"] in p


def test_stage2_prompt_mentions_stage1_output_and_full_context():
    cases, _ = _cases_and_gt()
    prepared = split_revision_evidence(cases[0])
    p = build_stage2_user_prompt(prepared, {"action": "ANSWER", "answer": "yes", "confidence": 0.6, "reason_for_action": "initial"})
    assert "STAGE 1 OUTPUT" in p
    assert "FULL CONTEXT" in p
    assert "KEEP_ANSWER" in p


# ── metrics ───────────────────────────────────────────────────────────────────

_TOY = [
    {
        "pmid": "A",
        "gold_label": "yes",
        "completed": True,
        "stage1_valid": True,
        "stage1_model_output": {"action": "ANSWER", "answer": "no", "confidence": 0.4, "reason_for_action": "initial"},
        "stage2_model_output": {"action": "REVISE_ANSWER", "answer": "yes", "confidence": 0.9, "reason_for_action": "results"},
        "final_answer": "yes",
        "label": "successful_revision",
    },
    {
        "pmid": "B",
        "gold_label": "no",
        "completed": True,
        "stage1_valid": True,
        "stage1_model_output": {"action": "ANSWER", "answer": "no", "confidence": 0.8, "reason_for_action": "initial"},
        "stage2_model_output": {"action": "KEEP_ANSWER", "answer": "no", "confidence": 0.7, "reason_for_action": "stable"},
        "final_answer": "no",
        "label": "kept_correct",
    },
    {
        "pmid": "C",
        "gold_label": "maybe",
        "completed": True,
        "stage1_valid": True,
        "stage1_model_output": {"action": "ANSWER", "answer": "maybe", "confidence": 0.7, "reason_for_action": "initial"},
        "stage2_model_output": {"action": "REVISE_ANSWER", "answer": "yes", "confidence": 0.6, "reason_for_action": "overreacted"},
        "final_answer": "yes",
        "label": "overreaction",
    },
    {
        "pmid": "D",
        "gold_label": "yes",
        "completed": True,
        "stage1_valid": True,
        "stage1_model_output": {"action": "ANSWER", "answer": "no", "confidence": 0.4, "reason_for_action": "initial"},
        "stage2_model_output": {"action": "KEEP_ANSWER", "answer": "no", "confidence": 0.5, "reason_for_action": "stuck"},
        "final_answer": "no",
        "label": "missed_revision",
    },
    {
        "pmid": "E",
        "gold_label": "yes",
        "completed": True,
        "stage1_valid": True,
        "stage1_model_output": {"action": "ANSWER", "answer": "no", "confidence": 0.3, "reason_for_action": "initial"},
        "stage2_model_output": {"action": "ABSTAIN", "answer": None, "confidence": 0.2, "reason_for_action": "insufficient"},
        "final_answer": None,
        "label": "abstention",
    },
]


def test_stage1_answer_accuracy():
    r = stage1_answer_accuracy(_TOY)
    assert r["accuracy"] == 0.4 and r["n_valid"] == 5


def test_final_answer_accuracy():
    r = final_answer_accuracy(_TOY)
    assert r["accuracy"] == 0.4 and r["n_completed"] == 5


def test_successful_revision_rate():
    r = successful_revision_rate(_TOY)
    assert r["rate"] == 1 / 3 and r["n_eligible"] == 3


def test_missed_revision_rate():
    # Abstentions excluded from denominator per spec ("excluding final abstentions").
    # stage1-wrong non-abstention cases: A (successful_revision), D (missed_revision) → n=2
    r = missed_revision_rate(_TOY)
    assert r["rate"] == 1 / 2 and r["n_eligible"] == 2


def test_overreaction_rate():
    r = overreaction_rate(_TOY)
    assert r["rate"] == 0.5 and r["n_eligible"] == 2


def test_kept_correct_rate():
    r = kept_correct_rate(_TOY)
    assert r["rate"] == 0.5 and r["n_eligible"] == 2


def test_final_abstention_rate():
    r = final_abstention_rate(_TOY)
    assert r["rate"] == 0.2 and r["n_completed"] == 5


def test_maintenance_rate():
    r = maintenance_rate(_TOY)
    assert r["rate"] == 0.5 and r["n_eligible"] == 4


def test_build_report_keys():
    report = build_report(_TOY, model="test")
    expected = {
        "model", "n_cases", "n_completed", "n_skipped_ineligible",
        "sample_label_counts", "split_strategy_counts",
        "stage1_answer_accuracy", "final_answer_accuracy",
        "successful_revision_rate", "missed_revision_rate",
        "overreaction_rate", "kept_correct_rate",
        "final_abstention_rate", "maintenance_rate",
    }
    assert set(report.keys()) == expected


# ── runner (mocked LLM) ───────────────────────────────────────────────────────

def _stub_responses():
    """Cyclic stub: returns valid JSON strings for each stage call."""
    _bank = [
        '{"action":"ANSWER","answer":"no","confidence":0.7,"reason_for_action":"partial evidence"}',
        '{"action":"REVISE_ANSWER","answer":"yes","confidence":0.85,"reason_for_action":"results change conclusion"}',
    ]
    idx = [0]

    def stub(system, user):
        r = _bank[idx[0] % len(_bank)]
        idx[0] += 1
        return r

    return stub


def test_run_case_mocked_structure():
    cases, gt = _cases_and_gt()
    case = split_revision_evidence(cases[0])
    trace = run_case(case, gt.get(case["pmid"]), call_fn=_stub_responses())
    assert trace["pmid"] == case["pmid"]
    assert trace["stage1_model_output"]["action"] == "ANSWER"
    assert trace["stage2_model_output"]["action"] == "REVISE_ANSWER"
    assert trace["completed"] is True


def test_run_case_mocked_no_parse_errors():
    cases, gt = _cases_and_gt()
    prepared = split_revision_evidence(cases[0])
    trace = run_case(prepared, gt.get(prepared["pmid"]), call_fn=_stub_responses())
    assert trace["stage1_valid"] is True
    assert trace["stage2_valid"] is True
